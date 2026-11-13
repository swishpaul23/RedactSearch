/* Upload — page pattern 3, Form (page): one narrow column, labels above fields,
   a submit that is always clickable, and a warning before leaving with files chosen. */
import { useQuery } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { Link, useBlocker } from "react-router";
import { api, call, type UploadJob } from "../api/client";
import { AlertIcon, CloseIcon, DoneIcon, FileIcon, SpinnerIcon, UploadIcon } from "../components/Icons";
import { PageHeading } from "../components/PageHeading";
import { REDACTION_LABEL } from "../components/RedactedText";
import { ErrorBanner } from "../components/States";
import { useDocumentTitle } from "../lib/hooks";

const ACCEPT = [".jsonl", ".json", ".txt"];
const MAX_FILES = 20;
const MAX_BYTES = 2_000_000;

type Picked = { key: string; file: File; problem: string | null };

function sizeLabel(bytes: number) {
  if (bytes < 1000) return `${bytes} B`;
  if (bytes < 1_000_000) return `${Math.round(bytes / 1000)} KB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

function plural(n: number, one: string, many = `${one}s`) {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}

function check(file: File, index: number): string | null {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ACCEPT.includes(ext)) return "Not a .jsonl, .json or .txt file";
  if (file.size > MAX_BYTES) return "Larger than 2 MB";
  if (file.size === 0) return "File is empty";
  if (index >= MAX_FILES) return `Over the ${MAX_FILES}-file limit`;
  return null;
}

function withChecks(files: File[]): Picked[] {
  return files.map((file, i) => ({ key: `${file.name}:${file.size}:${file.lastModified}:${i}`, file, problem: check(file, i) }));
}

async function sendFiles(files: File[]): Promise<UploadJob> {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  return call(() =>
    api.POST("/api/uploads", {
      body: { files: [] },
      bodySerializer: () => form,
    }),
  );
}

const STEPS = [
  { key: "read", label: "Read files" },
  { key: "redacting", label: "Redact personal data" },
  { key: "indexing", label: "Add to search" },
  { key: "done", label: "Done" },
] as const;

function stepIndex(job: UploadJob) {
  if (job.status === "queued" || job.status === "redacting") return 1;
  if (job.status === "indexing") return 2;
  if (job.status === "done") return 3;
  return job.call_ids.length ? 2 : 1;
}

function Progress({ job }: { job: UploadJob }) {
  const current = stepIndex(job);
  const failed = job.status === "failed";
  return (
    <ol className="steps" aria-label="Upload progress">
      {STEPS.map((s, i) => {
        const state = i < current || job.status === "done" ? "done" : i === current ? (failed ? "failed" : "current") : "todo";
        return (
          <li key={s.key} className={`step is-${state}`} aria-current={state === "current" ? "step" : undefined}>
            <span className="step-mark" aria-hidden="true">
              {state === "done" && <DoneIcon />}
              {state === "current" && <SpinnerIcon className="icon spin" />}
              {state === "failed" && <AlertIcon />}
            </span>
            <span>
              {s.label}
              <span className="sr-only">
                {state === "done" ? ", complete" : state === "current" ? ", in progress" : state === "failed" ? ", stopped" : ""}
              </span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function redactionSummary(counts: Record<string, number>) {
  const parts = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([type, n]) => plural(n, REDACTION_LABEL[type] ?? type.toLowerCase()));
  return parts.length ? parts.join(", ") : "nothing needed redacting";
}

function FileReports({ job }: { job: UploadJob }) {
  const withIssues = job.files.filter((f) => f.issues.length > 0);
  return (
    <>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Files read">
        <table className="table">
          <caption className="sr-only">What was read from each file</caption>
          <thead>
            <tr>
              <th scope="col">File</th>
              <th scope="col" className="num">Calls accepted</th>
              <th scope="col" className="num">Turns</th>
              <th scope="col" className="num">Issues</th>
            </tr>
          </thead>
          <tbody>
            {job.files.map((f, i) => (
              <tr key={`${f.name}-${i}`}>
                <th scope="row" className="file-cell">
                  {f.name}
                </th>
                <td className="num">{f.calls}</td>
                <td className="num">{f.turns}</td>
                <td className={f.issues.length ? "num text-bad" : "num"}>{f.issues.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {withIssues.map((f, i) => (
        <section key={`${f.name}-${i}`} className="issues" aria-label={`Issues in ${f.name}`}>
          <h3>{f.name}</h3>
          <ul>
            {f.issues.map((issue, j) => (
              <li key={j}>
                {issue.where && <span className="issue-where">{issue.where}</span>}
                {issue.message}
              </li>
            ))}
          </ul>
        </section>
      ))}
      {withIssues.length > 0 && (
        <p className="hint">Calls with an issue were skipped. Fix those lines and upload the file again.</p>
      )}
    </>
  );
}

function Result({ job, onReset }: { job: UploadJob; onReset: () => void }) {
  const finished = job.status === "done" || job.status === "failed";
  return (
    <section className="panel" aria-labelledby="upload-result">
      <div className="panel-head">
        <h2 id="upload-result">{finished ? (job.status === "done" ? "Upload complete" : "Upload stopped") : "Uploading"}</h2>
        {finished && (
          <button type="button" className="btn btn-sm" onClick={onReset}>
            Upload more files
          </button>
        )}
      </div>
      <div className="panel-body stack">
        <Progress job={job} />
        <div role="status" aria-live="polite" className="stack">
          {job.status === "done" && (
            <p className="result-line">
              <strong>Imported {plural(job.call_ids.length, "call")}.</strong> Redacted {redactionSummary(job.redactions)}.
            </p>
          )}
          {job.status === "failed" && (
            <div className="banner banner-error">
              <div className="banner-text">
                <p className="banner-title">Nothing was imported</p>
                <p>{job.message}</p>
              </div>
            </div>
          )}
          {job.status === "done" && job.message && (
            <div className="banner banner-warn">
              <div className="banner-text">
                <p>{job.message}</p>
              </div>
            </div>
          )}
          {!finished && <p className="hint">{plural(job.n_calls, "call")} in this upload. You can leave this page; the import continues.</p>}
        </div>
        {job.call_ids.length > 0 && (
          <div className="stack-sm">
            <h3 className="subhead">Imported calls</h3>
            <ul className="call-links">
              {job.call_ids.map((id) => (
                <li key={id}>
                  <Link to={`/calls/${id}`}>{id}</Link>
                </li>
              ))}
            </ul>
            <p>
              <Link to="/calls?source=upload">View all uploaded calls</Link>
            </p>
          </div>
        )}
        <FileReports job={job} />
      </div>
    </section>
  );
}

function FormatHelp() {
  return (
    <details className="format-help">
      <summary>File formats</summary>
      <div className="stack-sm">
        <p>
          <strong>.txt</strong>: one call per file, one turn per line. A time like <code>[01:23]</code> at the start is
          optional; lines without a speaker continue the turn above.
        </p>
        <pre>{`[00:04] Agent: Thanks for calling, how can I help?
[00:09] Customer: I was charged twice this month.`}</pre>
        <p>
          <strong>.jsonl</strong>: one call per line. <strong>.json</strong>: one call, or a list of calls.
        </p>
        <pre>{`{"turns": [
  {"speaker": "agent", "text": "Thanks for calling.", "start_sec": 0},
  {"speaker": "customer", "text": "I was charged twice."}
]}`}</pre>
        <p>
          Speakers can be agent, rep or advisor, and customer, caller or client. Missing start times are estimated from
          the words spoken before.
        </p>
      </div>
    </details>
  );
}

export function UploadPage() {
  useDocumentTitle("Upload");
  const inputId = useId();
  const labelId = useId();
  const hintId = useId();
  const errorId = useId();
  const input = useRef<HTMLInputElement>(null);
  const leaveDialog = useRef<HTMLDialogElement>(null);
  const [picked, setPicked] = useState<Picked[]>([]);
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<unknown>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const job = useQuery({
    queryKey: ["upload", jobId],
    queryFn: () => call(() => api.GET("/api/uploads/{job_id}", { params: { path: { job_id: jobId! } } })),
    enabled: jobId !== null,
    staleTime: 0,
    refetchInterval: (q) => (q.state.data && ["done", "failed"].includes(q.state.data.status) ? false : 800),
  });

  const dirty = picked.length > 0 && jobId === null;
  const blocker = useBlocker(({ currentLocation, nextLocation }) => dirty && currentLocation.pathname !== nextLocation.pathname);

  useEffect(() => {
    if (blocker.state === "blocked") leaveDialog.current?.showModal?.();
  }, [blocker.state]);

  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  function add(files: FileList | File[] | null) {
    if (!files || files.length === 0) return;
    const added = Array.from(files); // copy now: the input's FileList empties when its value is reset
    setPicked((prev) => withChecks([...prev.map((p) => p.file), ...added]));
    setFieldError(null);
  }

  function remove(key: string) {
    setPicked((prev) => withChecks(prev.filter((p) => p.key !== key).map((p) => p.file)));
  }

  function reset() {
    setPicked([]);
    setJobId(null);
    setSendError(null);
    setFieldError(null);
    if (input.current) input.current.value = "";
    requestAnimationFrame(() => input.current?.focus());
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const ready = picked.filter((p) => !p.problem).map((p) => p.file);
    if (ready.length === 0) {
      setFieldError(picked.length ? "None of the chosen files can be uploaded. Remove the marked files and choose others." : "Choose at least one file.");
      input.current?.focus();
      return;
    }
    setSending(true);
    setSendError(null);
    try {
      const created = await sendFiles(ready);
      setJobId(created.id);
    } catch (err) {
      setSendError(err);
    } finally {
      setSending(false);
    }
  }

  const problems = picked.filter((p) => p.problem).length;
  const current = job.data;

  return (
    <div className="page page-narrow">
      <div className="page-head">
        <PageHeading>Upload transcripts</PageHeading>
        <p className="page-desc">
          Add calls from your own files. Names, card numbers, phone numbers, emails, SINs and account numbers are
          redacted before anything is saved, and the original text is never stored.
        </p>
      </div>

      {jobId === null ? (
        <form className="panel upload-form" onSubmit={submit} noValidate>
          <div className="panel-body stack">
            <div className="field">
              <label className="label" htmlFor={inputId} id={labelId}>
                Transcript files
              </label>
              <p className="hint" id={hintId}>
                .jsonl, .json or .txt. Up to {MAX_FILES} files, 2 MB each.
              </p>
              <div
                className={dragging ? "dropzone is-over" : fieldError ? "dropzone is-invalid" : "dropzone"}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  add(e.dataTransfer.files);
                }}
              >
                <UploadIcon className="dropzone-icon" size={24} />
                <p className="dropzone-text">Drop files here, or</p>
                <input
                  ref={input}
                  id={inputId}
                  className="file-input"
                  type="file"
                  multiple
                  accept={ACCEPT.join(",")}
                  aria-labelledby={labelId}
                  aria-describedby={fieldError ? `${hintId} ${errorId}` : hintId}
                  aria-invalid={fieldError ? true : undefined}
                  onChange={(e) => {
                    add(e.target.files);
                    e.target.value = "";
                  }}
                />
                <label htmlFor={inputId} className="btn file-button" aria-hidden="true">
                  Choose files
                </label>
              </div>
              {fieldError && (
                <p className="field-error" id={errorId}>
                  <AlertIcon />
                  {fieldError}
                </p>
              )}
            </div>

            {picked.length > 0 && (
              <div className="stack-sm">
                <h2 className="subhead" id="chosen-files">
                  {plural(picked.length, "file")} chosen
                  {problems > 0 && <span className="text-bad">, {problems} can’t be uploaded</span>}
                </h2>
                <ul className="file-list" aria-labelledby="chosen-files">
                  {picked.map((p) => (
                    <li key={p.key} className={p.problem ? "file-row is-bad" : "file-row"}>
                      <FileIcon />
                      <span className="file-name">{p.file.name}</span>
                      <span className="file-meta">{p.problem ?? sizeLabel(p.file.size)}</span>
                      <button
                        type="button"
                        className="btn btn-icon btn-quiet"
                        onClick={() => remove(p.key)}
                        aria-label={`Remove ${p.file.name}`}
                        title="Remove"
                      >
                        <CloseIcon />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <FormatHelp />

            {sendError !== null && <ErrorBanner error={sendError} title="The files were not uploaded" />}
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" aria-busy={sending || undefined}>
              {sending ? <SpinnerIcon className="icon spin" /> : <UploadIcon />}
              {sending ? "Uploading…" : "Upload and redact"}
            </button>
            {picked.length > 0 && (
              <button type="button" className="btn btn-quiet" onClick={reset}>
                Clear files
              </button>
            )}
          </div>
        </form>
      ) : current ? (
        <Result job={current} onReset={reset} />
      ) : job.isError ? (
        <ErrorBanner error={job.error} onRetry={() => job.refetch()} title="Upload status could not be loaded" />
      ) : (
        <p className="hint" role="status">
          Starting upload…
        </p>
      )}

      <dialog
        ref={leaveDialog}
        className="dialog"
        aria-labelledby="leave-title"
        aria-describedby="leave-desc"
        onClose={() => blocker.state === "blocked" && blocker.reset()}
      >
        <h2 id="leave-title">Leave without uploading?</h2>
        <p id="leave-desc">The {plural(picked.length, "file")} you chose will be cleared.</p>
        <div className="dialog-actions">
          <button
            type="button"
            className="btn"
            autoFocus
            onClick={() => {
              leaveDialog.current?.close();
            }}
          >
            Stay on page
          </button>
          <button
            type="button"
            className="btn btn-danger"
            onClick={() => {
              if (blocker.state === "blocked") blocker.proceed();
              leaveDialog.current?.close();
            }}
          >
            Leave and clear files
          </button>
        </div>
      </dialog>
    </div>
  );
}
