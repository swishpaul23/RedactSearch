import { useMutation, useQuery } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";
import { Link } from "react-router";
import { api, call, type AnswerOut, type AskIn, type Mode } from "../api/client";
import { ChevronIcon, ExternalIcon } from "../components/Icons";
import { PageHeading } from "../components/PageHeading";
import { RedactedText } from "../components/RedactedText";
import { Empty, ErrorBanner, SkeletonLines } from "../components/States";
import { Transcript } from "../components/Transcript";
import { clock, seconds, SOURCE_LABEL, SPEAKER_LABEL, SPLIT_LABEL } from "../lib/format";
import { useDocumentTitle, useElapsed, usePersistentState, useSlashFocus } from "../lib/hooks";

type Settings = { mode: Mode; k: number; split: "" | "train" | "dev" | "test" | "upload" };
const DEFAULT_SETTINGS: Settings = { mode: "bm25", k: 5, split: "" };

const MODE_LABEL: Record<Mode, string> = {
  bm25: "Keyword match (recommended)",
  dense: "Meaning match",
  hybrid: "Keyword and meaning",
};

const EXAMPLES = [
  "Why did the customer want a refund?",
  "Which train arrives in Norwich at 16:55 on Saturday?",
  "What is the postcode of the Cambridge Arts Theatre?",
  "What is the wifi password at the University Arms Hotel?",
];

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
  return reason;
}

function AnswerStatus({ a }: { a: AnswerOut }) {
  if (a.found) return <span className="status status-ok">Answered from {a.citations.length === 1 ? "1 call turn" : `${a.citations.length} call turns`}</span>;
  const nothingToAnswer = !a.rejected || /nothing retrieved|appears nowhere/.test(a.rejected);
  if (!nothingToAnswer) return <span className="status status-warn">Answer withheld</span>;
  return <span className="status">Not in the transcripts</span>;
}

function Preview({ chunkId, callId, turnId }: { chunkId: string; callId: string; turnId: number }) {
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
              {detail.data.split === "upload" ? "Uploaded call" : `${SOURCE_LABEL[detail.data.source]}, ${SPLIT_LABEL[detail.data.split]} split`}, cited turn at{" "}
              {clock(detail.data.turns.find((t) => t.turn_id === turnId)?.start_sec ?? 0)}
            </p>
          )}
        </div>
        <Link className="btn btn-sm" to={`/calls/${callId}?turn=${turnId}`} aria-label={`Open full call ${callId}`}>
          Open full call
          <ExternalIcon />
        </Link>
      </div>
      <div className="preview-body" ref={scrollRoot} key={chunkId} tabIndex={0} aria-label={`Transcript of ${callId}, scrollable`}>
        {detail.isPending && <SkeletonLines lines={6} label="Loading transcript" />}
        {detail.isError && <ErrorBanner error={detail.error} onRetry={() => detail.refetch()} title="The transcript could not be loaded" />}
        {detail.data && <Transcript call={detail.data} citedTurn={turnId} scrollRoot={scrollRoot} />}
      </div>
    </section>
  );
}

function Result({ a }: { a: AnswerOut }) {
  const [selected, setSelected] = useState(0);
  const listId = useId();
  const cits = a.citations;
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
            <div><dt>Model</dt><dd>{a.provenance.model}</dd></div>
            <div><dt>Search</dt><dd>{MODE_LABEL[a.provenance.retrieval].replace(" (recommended)", "")}</dd></div>
            <div><dt>Excerpts</dt><dd>{a.provenance.k}</dd></div>
            <div><dt>Time</dt><dd>{seconds(a.elapsed_ms)}</dd></div>
          </dl>
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
                  key={c.chunk_id}
                  id={`${listId}-opt-${i}`}
                  role="option"
                  aria-selected={i === selected}
                  className="source"
                  onClick={() => setSelected(i)}
                >
                  <span className="source-head">
                    <span className="source-call">{c.call_id}</span>
                    <span className="source-meta">
                      {SPEAKER_LABEL[c.speaker]} at {clock(c.start_sec)}
                    </span>
                  </span>
                  <span className="source-snippet">
                    <RedactedText text={c.snippet} />
                  </span>
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
        <Preview chunkId={current.chunk_id} callId={current.call_id} turnId={current.turn_id} />
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

  const ask = useMutation({
    mutationFn: (body: AskIn) => call(() => api.POST("/api/ask", { body })),
  });
  const elapsed = useElapsed(ask.isPending);
  const tooShort = question.trim().length < 3;

  function submit(q = question) {
    const text = q.trim();
    if (text.length < 3 || ask.isPending) return;
    ask.mutate({ question: text, k: settings.k, mode: settings.mode, split: settings.split || null });
  }

  return (
    <div className="page">
      <div className="page-head">
        <PageHeading>Ask the call archive</PageHeading>
        <p className="page-desc">
          Answers come only from redacted transcripts and cite the calls they came from. When the transcripts do not
          contain an answer, you get “not found” instead of a guess.
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

        <div className="composer-row">
          <button
            type="button"
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

        <div id={ids.settings} className="settings" hidden={!showSettings}>
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
                ? `Answer ready, citing ${ask.data.citations.length} call ${ask.data.citations.length === 1 ? "turn" : "turns"}.`
                : "The transcripts do not answer this question."
              : ""}
      </p>
      <div aria-busy={ask.isPending}>
        {ask.isPending && (
          <section className="panel">
            <div className="answer">
              <p className="status-line">
                <span className="spinner" aria-hidden="true" />
                Reading the closest transcript turns and drafting a cited answer
                {elapsed >= 2 ? `, ${elapsed} s` : ""}
              </p>
              <SkeletonLines lines={3} label="Drafting answer" />
            </div>
          </section>
        )}
        {ask.isError && !ask.isPending && (
          <ErrorBanner error={ask.error} onRetry={() => submit()} title="The question could not be answered" />
        )}
        {ask.data && !ask.isPending && <Result key={ask.data.question + ask.data.elapsed_ms} a={ask.data} />}
        {ask.isIdle && (
          <section className="panel" aria-labelledby={`${ids.q}-ex`}>
            <div className="answer">
              <h2 id={`${ids.q}-ex`}>Try a question</h2>
              <div className="examples">
                {EXAMPLES.map((ex) => (
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
