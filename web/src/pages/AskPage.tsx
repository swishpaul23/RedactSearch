import { useMutation, useQuery } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";
import { Link } from "react-router";
import { api, call, type AgentAnswerOut, type AnswerOut, type AskIn, type Mode } from "../api/client";
import { ChevronIcon, ExternalIcon } from "../components/Icons";
import { PageHeading } from "../components/PageHeading";
import { RedactedText } from "../components/RedactedText";
import { Empty, ErrorBanner, SkeletonLines } from "../components/States";
import { Transcript } from "../components/Transcript";
import { clock, seconds, SOURCE_LABEL, SPEAKER_LABEL, SPLIT_LABEL } from "../lib/format";
import { useDocumentTitle, useElapsed, usePersistentState, useSlashFocus } from "../lib/hooks";

type Method = "search" | "agent";
type Settings = { mode: Mode; k: number; split: "" | "train" | "dev" | "test" | "upload"; method?: Method };
const DEFAULT_SETTINGS: Settings = { mode: "bm25", k: 5, split: "", method: "search" };

const MODE_LABEL: Record<Mode, string> = {
  bm25: "Keyword match (recommended)",
  dense: "Meaning match",
  hybrid: "Keyword and meaning",
};

const EXAMPLES: Record<Method, string[]> = {
  search: [
    "Why did the customer want a refund?",
    "Which train arrives in Norwich at 16:55 on Saturday?",
    "What is the postcode of the Cambridge Arts Theatre?",
    "What is the wifi password at the University Arms Hotel?",
  ],
  agent: [
    "In the test split, do agents or customers mention 'parking' in more turns, and what was said?",
    "How many test-split calls mention 'museum'? Name one of them.",
    "What name recall does Regex + DistilBERT get, and how many [PERSON] placeholders are in test-split calls?",
    "Which test-split call mentions a sauna, and what did the agent offer?",
  ],
};

const TOOL_LABEL: Record<string, string> = {
  search_calls: "Searched turns",
  get_transcript: "Read a transcript",
  compute_stats: "Counted calls and turns",
  redaction_report: "Read redaction results",
};

/** One answer shape for both methods, so the result view is shared. */
type Source = { key: string; call_id: string; turn_id: number | null; label: string; snippet: string | null };
type View = {
  question: string;
  answer: string;
  found: boolean;
  rejected: string | null;
  sources: Source[];
  meta: [string, string][];
  steps: AgentAnswerOut["steps"] | null;
  citedResults: number[];
  stamp: number;
};

function fromAnswer(a: AnswerOut): View {
  return {
    question: a.question, answer: a.answer, found: a.found, rejected: a.rejected ?? null,
    sources: a.citations.map((c) => ({ key: c.chunk_id, call_id: c.call_id, turn_id: c.turn_id, label: `${SPEAKER_LABEL[c.speaker]} at ${clock(c.start_sec)}`, snippet: c.snippet })),
    meta: [["Model", a.provenance.model], ["Search", MODE_LABEL[a.provenance.retrieval].replace(" (recommended)", "")], ["Excerpts", String(a.provenance.k)], ["Time", seconds(a.elapsed_ms)]],
    steps: null, citedResults: [], stamp: a.elapsed_ms,
  };
}

function fromAgent(a: AgentAnswerOut): View {
  const turnCalls = new Set(a.citations.map((c) => c.call_id));
  const sources: Source[] = [
    ...a.citations.map((c) => ({ key: c.chunk_id, call_id: c.call_id, turn_id: c.turn_id, label: `${SPEAKER_LABEL[c.speaker]} at ${clock(c.start_sec)}`, snippet: c.snippet })),
    ...a.cited_calls.filter((id) => !turnCalls.has(id)).map((id) => ({ key: id, call_id: id, turn_id: null, label: "Whole call", snippet: null })),
  ];
  return {
    question: a.question, answer: a.answer, found: a.found, rejected: a.rejected ?? null, sources,
    meta: [["Model", a.provenance.model], ["Method", "Agent"], ["Steps", `${a.steps.length} of ${a.provenance.max_steps}`], ["Time", seconds(a.elapsed_ms)]],
    steps: a.steps, citedResults: a.cited_results, stamp: a.elapsed_ms,
  };
}

function stepArgs(args: Record<string, unknown>): string {
  return Object.entries(args)
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `${k} ${typeof v === "string" ? `“${v}”` : JSON.stringify(v)}`)
    .join(", ");
}

function Steps({ steps, cited }: { steps: NonNullable<View["steps"]>; cited: number[] }) {
  const id = useId();
  if (steps.length === 0) return null;
  return (
    <div className="agent-steps">
      <h3 id={id} className="subhead">
        How the agent worked
      </h3>
      <ol aria-labelledby={id}>
        {steps.map((s) => (
          <li key={s.n} className={s.ok ? "agent-step" : "agent-step is-off"}>
            <span className="agent-step-tool">
              {s.tool ? (TOOL_LABEL[s.tool] ?? s.tool) : "Unreadable reply"}
              {cited.includes(s.n) && <span className="agent-step-cited">cited</span>}
            </span>
            {s.tool && stepArgs(s.args) && <span className="agent-step-args">{stepArgs(s.args)}</span>}
            <span className="agent-step-summary">{s.summary}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Turns the server's discard reason into a sentence a reviewer can act on. */
export function explainRejection(reason: string): string {
  const term = /'([^']+)'/.exec(reason)?.[1];
  if (reason.includes("appears nowhere in the index")) return `No call mentions “${term}”, so there is nothing to answer from.`;
  if (reason.includes("is not in the cited excerpts"))
    return `The cited excerpt does not mention “${term}”, so the answer was withheld rather than guessed.`;
  if (reason.includes("no citations")) return "The model answered without citing a call, so the answer was withheld.";
  if (reason.includes("not retrieved")) return "The model cited an excerpt it was not given, so the answer was withheld.";
  if (reason.includes("unparseable")) return "The model's reply could not be read. Ask again.";
  if (reason.includes("nothing retrieved")) return "No transcript turns matched the question.";
  if (reason.includes("not in any tool result")) return "The agent cited a source it never looked at, so the answer was withheld.";
  if (reason.includes("step cap")) return "The agent used all of its tool calls without reaching an answer. Try a narrower question.";
  return reason;
}

function AnswerStatus({ a }: { a: View }) {
  const n = a.sources.length + a.citedResults.length;
  if (a.found) return <span className="status status-ok">Answered from {n === 1 ? "1 source" : `${n} sources`}</span>;
  const nothingToAnswer = !a.rejected || /nothing retrieved|appears nowhere/.test(a.rejected);
  if (!nothingToAnswer) return <span className="status status-warn">Answer withheld</span>;
  return <span className="status">Not in the transcripts</span>;
}

function Preview({ chunkId, callId, turnId }: { chunkId: string; callId: string; turnId: number | null }) {
  const scrollRoot = useRef<HTMLDivElement>(null);
  const detail = useQuery({
    queryKey: ["call", callId],
    queryFn: () => call(() => api.GET("/api/calls/{call_id}", { params: { path: { call_id: callId } } })),
  });
  const headingId = useId();
  return (
    <section className="panel preview" aria-labelledby={headingId} id="preview" tabIndex={-1}>
      <div className="panel-head">
        <div>
          <h2 id={headingId}>{callId}</h2>
          {detail.data && (
            <p className="hint">
              {detail.data.split === "upload" ? "Uploaded call" : `${SOURCE_LABEL[detail.data.source]}, ${SPLIT_LABEL[detail.data.split]} split`}
              {turnId === null ? ", cited as a whole call" : `, cited turn at ${clock(detail.data.turns.find((t) => t.turn_id === turnId)?.start_sec ?? 0)}`}
            </p>
          )}
        </div>
        <Link className="btn btn-sm" to={turnId === null ? `/calls/${callId}` : `/calls/${callId}?turn=${turnId}`} aria-label={`Open full call ${callId}`}>
          Open full call
          <ExternalIcon />
        </Link>
      </div>
      <div className="preview-body" ref={scrollRoot} key={chunkId} tabIndex={0} aria-label={`Transcript of ${callId}, scrollable`}>
        {detail.isPending && <SkeletonLines lines={6} label="Loading transcript" />}
        {detail.isError && <ErrorBanner error={detail.error} onRetry={() => detail.refetch()} title="The transcript could not be loaded" />}
        {detail.data && <Transcript call={detail.data} citedTurn={turnId ?? undefined} scrollRoot={scrollRoot} />}
      </div>
    </section>
  );
}

function Result({ a }: { a: View }) {
  const [selected, setSelected] = useState(0);
  const listId = useId();
  const cits = a.sources;
  const current = cits[selected];

  function onKeyDown(e: React.KeyboardEvent<HTMLUListElement>) {
    const last = cits.length - 1;
    const next = { ArrowDown: Math.min(selected + 1, last), ArrowUp: Math.max(selected - 1, 0), Home: 0, End: last }[
      e.key as "ArrowDown" | "ArrowUp" | "Home" | "End"
    ];
    if (next !== undefined) {
      e.preventDefault();
      setSelected(next);
    } else if (e.key === "Enter") {
      e.preventDefault();
      document.getElementById("preview")?.focus();
    }
  }

  return (
    <div className="split">
      <section className="panel" aria-labelledby={`${listId}-answer`}>
        <div className="answer">
          <div className="composer-row">
            <h2 id={`${listId}-answer`}>Answer</h2>
            <AnswerStatus a={a} />
          </div>
          {a.found ? (
            <p className="answer-text">
              <RedactedText text={a.answer} />
            </p>
          ) : (
            <div className="answer-reason">
              <p className="answer-text">The transcripts do not answer this question.</p>
              {a.rejected && <p>{explainRejection(a.rejected)}</p>}
              <p>Questions that name a specific detail, such as a train ID, venue, date or amount, work best.</p>
            </div>
          )}
          <dl className="meta" aria-label="How this answer was made">
            {a.meta.map(([k, v]) => (
              <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
            ))}
          </dl>
          {a.steps && <Steps steps={a.steps} cited={a.citedResults} />}
        </div>
        {cits.length > 0 && (
          <>
            <h3 className="sr-only" id={`${listId}-label`}>Sources</h3>
            <ul
              className="sources"
              role="listbox"
              tabIndex={0}
              aria-labelledby={`${listId}-label`}
              aria-activedescendant={`${listId}-opt-${selected}`}
              onKeyDown={onKeyDown}
            >
              {cits.map((c, i) => (
                <li
                  key={c.key}
                  id={`${listId}-opt-${i}`}
                  role="option"
                  aria-selected={i === selected}
                  className="source"
                  onClick={() => setSelected(i)}
                >
                  <span className="source-head">
                    <span className="source-call">{c.call_id}</span>
                    <span className="source-meta">{c.label}</span>
                  </span>
                  {c.snippet && (
                    <span className="source-snippet">
                      <RedactedText text={c.snippet} />
                    </span>
                  )}
                </li>
              ))}
            </ul>
            <p className="hint" style={{ padding: "8px 16px 12px" }}>
              Use <kbd className="kbd">↑</kbd> <kbd className="kbd">↓</kbd> to move between sources and{" "}
              <kbd className="kbd">Enter</kbd> to read the transcript.
            </p>
          </>
        )}
      </section>
      {current ? (
        <Preview chunkId={current.key} callId={current.call_id} turnId={current.turn_id} />
      ) : (
        <section className="panel" aria-label="Transcript preview">
          <Empty title="No call to show">
            <p>When an answer cites a call, its transcript opens here with the cited turn highlighted.</p>
          </Empty>
        </section>
      )}
    </div>
  );
}

export function AskPage() {
  useDocumentTitle("Ask");
  const [question, setQuestion] = useState("");
  const [settings, setSettings] = usePersistentState<Settings>("redactsearch.ask-settings", DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  useSlashFocus(input);
  const ids = { q: useId(), hint: useId(), settings: useId() };

  const method: Method = settings.method ?? "search";
  const ask = useMutation({
    mutationFn: async (body: AskIn & { method: Method }): Promise<View> => {
      if (body.method === "agent") return fromAgent(await call(() => api.POST("/api/agent", { body: { question: body.question } })));
      const { method: _m, ...rest } = body;
      return fromAnswer(await call(() => api.POST("/api/ask", { body: rest })));
    },
  });
  const elapsed = useElapsed(ask.isPending);
  const tooShort = question.trim().length < 3;

  function submit(q = question) {
    const text = q.trim();
    if (text.length < 3 || ask.isPending) return;
    ask.mutate({ question: text, k: settings.k, mode: settings.mode, split: settings.split || null, method });
  }

  return (
    <div className="page">
      <div className="page-head">
        <PageHeading>Ask the call archive</PageHeading>
        <p className="page-desc">
          Answers come only from redacted transcripts and cite the calls they came from. When the transcripts do not
          contain an answer, you get “not found” instead of a guess. The agent can also count, compare and read whole
          calls for questions that need several steps.
        </p>
      </div>

      <form
        className="panel composer"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <div className="field">
          <label className="label" htmlFor={ids.q}>
            Question
          </label>
          <textarea
            id={ids.q}
            ref={input}
            className="textarea"
            rows={2}
            maxLength={500}
            value={question}
            placeholder="For example: which train arrives in Norwich at 16:55 on Saturday?"
            aria-describedby={ids.hint}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <p className="hint" id={ids.hint}>
            <kbd className="kbd">Enter</kbd> to ask, <kbd className="kbd">Shift</kbd> + <kbd className="kbd">Enter</kbd> for
            a new line, <kbd className="kbd">/</kbd> to come back here.
          </p>
        </div>

        <fieldset className="method">
          <legend className="label">Answer with</legend>
          <div className="segmented">
            {([
              ["search", "Search and answer", "One search, one cited answer"],
              ["agent", "Agent", "Several tools: search, count, read calls"],
            ] as const).map(([value, label, hint]) => (
              <label key={value} className="segment">
                <input
                  type="radio"
                  name={`${ids.q}-method`}
                  value={value}
                  checked={method === value}
                  onChange={() => setSettings({ ...settings, method: value })}
                />
                <span className="segment-label">{label}</span>
                <span className="segment-hint">{hint}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <div className="composer-row">
          <button
            type="button"
            hidden={method === "agent"}
            className="btn btn-sm btn-quiet disclosure-btn"
            aria-expanded={showSettings}
            aria-controls={ids.settings}
            onClick={() => setShowSettings(!showSettings)}
          >
            <ChevronIcon />
            Search settings
          </button>
          <button type="submit" className="btn btn-primary" disabled={tooShort || ask.isPending}>
            {ask.isPending && <span className="spinner" aria-hidden="true" />}
            {ask.isPending ? "Asking…" : "Ask"}
          </button>
        </div>

        <div id={ids.settings} className="settings" hidden={!showSettings || method === "agent"}>
          <div className="field">
            <label className="label" htmlFor={`${ids.settings}-mode`}>
              Match transcripts by
            </label>
            <select
              id={`${ids.settings}-mode`}
              className="select"
              value={settings.mode}
              onChange={(e) => setSettings({ ...settings, mode: e.target.value as Mode })}
            >
              {(Object.keys(MODE_LABEL) as Mode[]).map((m) => (
                <option key={m} value={m}>
                  {MODE_LABEL[m]}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label className="label" htmlFor={`${ids.settings}-k`}>
              Excerpts to read
            </label>
            <select
              id={`${ids.settings}-k`}
              className="select"
              value={settings.k}
              onChange={(e) => setSettings({ ...settings, k: Number(e.target.value) })}
            >
              {[3, 5, 8, 10].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label className="label" htmlFor={`${ids.settings}-split`}>
              Calls to search
            </label>
            <select
              id={`${ids.settings}-split`}
              className="select"
              value={settings.split}
              onChange={(e) => setSettings({ ...settings, split: e.target.value as Settings["split"] })}
            >
              <option value="">All calls</option>
              <option value="train">Train split</option>
              <option value="dev">Dev split</option>
              <option value="test">Test split</option>
              <option value="upload">Uploaded calls</option>
            </select>
          </div>
        </div>
      </form>

      <p className="sr-only" aria-live="polite">
        {ask.isPending
          ? "Drafting an answer."
          : ask.isError
            ? "The question could not be answered."
            : ask.data
              ? ask.data.found
                ? `Answer ready, citing ${ask.data.sources.length + ask.data.citedResults.length} ${ask.data.sources.length + ask.data.citedResults.length === 1 ? "source" : "sources"}.`
                : "The transcripts do not answer this question."
              : ""}
      </p>
      <div aria-busy={ask.isPending}>
        {ask.isPending && (
          <section className="panel">
            <div className="answer">
              <p className="status-line">
                <span className="spinner" aria-hidden="true" />
                {ask.variables?.method === "agent"
                  ? "The agent is searching, counting and reading calls. On a local model this can take a minute"
                  : "Reading the closest transcript turns and drafting a cited answer"}
                {elapsed >= 2 ? `, ${elapsed} s` : ""}
              </p>
              <SkeletonLines lines={3} label="Drafting answer" />
            </div>
          </section>
        )}
        {ask.isError && !ask.isPending && (
          <ErrorBanner error={ask.error} onRetry={() => submit()} title="The question could not be answered" />
        )}
        {ask.data && !ask.isPending && <Result key={ask.data.question + ask.data.stamp} a={ask.data} />}
        {ask.isIdle && (
          <section className="panel" aria-labelledby={`${ids.q}-ex`}>
            <div className="answer">
              <h2 id={`${ids.q}-ex`}>Try a question</h2>
              <div className="examples">
                {EXAMPLES[method].map((ex) => (
                  <button
                    key={ex}
                    type="button"
                    className="btn btn-sm"
                    onClick={() => {
                      setQuestion(ex);
                      submit(ex);
                    }}
                  >
                    {ex}
                  </button>
                ))}
              </div>
              <p className="hint">The last example has no answer in the calls, to show how a refusal looks.</p>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
