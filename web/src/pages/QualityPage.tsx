import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { api, call, type MetricsOut } from "../api/client";
import { PageHeading } from "../components/PageHeading";
import { Empty, ErrorBanner, SkeletonLines } from "../components/States";
import { pct, score } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";

function Kpis({ m }: { m: MetricsOut }) {
  const hf = m.redaction.find((d) => d.detector === "pipeline_hf");
  const tiles = [
    hf && { label: "Name detection F1", value: score(hf.person_f1), note: "DistilBERT on 266 held-out names" },
    hf && { label: "Personal data left in text", value: pct(hf.leak_rate), note: "Share of all planted values, test split" },
    m.rag && { label: "Right call in top 5", value: pct(m.rag.recall_at_5, 0), note: `${m.rag.n_answerable} test questions` },
    m.rag && {
      label: "Unanswerable questions refused",
      value: `${Math.round(m.rag.abstention_rate * m.rag.n_unanswerable)} of ${m.rag.n_unanswerable}`,
      note: "Said not found instead of guessing",
    },
    m.guard && {
      label: "Card, SIN or contact data in the index",
      value: String(m.guard.structured_leaks),
      note: `Checked across ${m.guard.n_chunks.toLocaleString()} indexed turns`,
    },
  ].filter(Boolean) as { label: string; value: string; note: string }[];
  if (tiles.length === 0) return null;
  return (
    <dl className="kpis">
      {tiles.map((t) => (
        <div className="kpi" key={t.label}>
          <dt>{t.label}</dt>
          <dd>
            <span className="kpi-value">{t.value}</span>
            <span className="kpi-note">{t.note}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

function LeakChart({ m }: { m: MetricsOut }) {
  const rows = m.redaction;
  if (rows.length === 0) {
    return (
      <Empty title="No redaction results yet">
        <p>Run the redaction eval on the test split to fill this chart.</p>
      </Empty>
    );
  }
  const max = Math.max(...rows.map((r) => r.leak_rate), 0.0001);
  const best = Math.min(...rows.map((r) => r.leak_rate));
  return (
    <>
      <ul className="bars" aria-label="Personal data left in text, by redaction method (lower is better)">
        {rows.map((r) => (
          <li className="bar-row" key={r.detector}>
            <span>{r.label}</span>
            <span className="bar-track" aria-hidden="true">
              <span
                className={r.leak_rate === best ? "bar-fill is-best" : "bar-fill"}
                style={{ width: `${Math.max(1.5, (r.leak_rate / max) * 100)}%`, display: "block" }}
              />
            </span>
            <span className="bar-value">{pct(r.leak_rate)}</span>
          </li>
        ))}
      </ul>
      <div className="table-wrap">
        <table className="table">
          <caption className="sr-only">Redaction results on the test split</caption>
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col" className="num">Name precision</th>
              <th scope="col" className="num">Name recall</th>
              <th scope="col" className="num">Name F1</th>
              <th scope="col" className="num">Other data found</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.detector}>
                <th scope="row">{r.label}</th>
                <td className="num">{score(r.person_precision)}</td>
                <td className="num">{score(r.person_recall)}</td>
                <td className="num">{score(r.person_f1)}</td>
                <td className="num">{pct(r.structured_recall, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function QualityPage() {
  useDocumentTitle("Quality");
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: () => call(() => api.GET("/api/metrics")) });
  const m = metrics.data;

  return (
    <div className="page">
      <div className="page-head">
        <PageHeading>Quality</PageHeading>
        <p className="page-desc">
          How well redaction and search hold up on the held-out test split: 100 calls the models never trained on, and
          54 questions written after every setting was chosen.
        </p>
      </div>

      {metrics.isPending && <SkeletonLines lines={6} label="Loading results" />}
      {metrics.isError && <ErrorBanner error={metrics.error} onRetry={() => metrics.refetch()} title="Results could not be loaded" />}

      {m && (
        <>
          <Kpis m={m} />
          <div className="dashboard-grid">
            <section className="panel" aria-labelledby="q-redaction">
              <div className="panel-head">
                <h2 id="q-redaction">Personal data left in text</h2>
                <span className="hint">Lower is better</span>
              </div>
              <LeakChart m={m} />
            </section>

            <div style={{ display: "grid", gap: 16 }}>
              {m.rag && (
                <section className="panel" aria-labelledby="q-search">
                  <div className="panel-head">
                    <h2 id="q-search">Search and answers</h2>
                  </div>
                  <dl className="facts">
                    <div><dt>Right call in top 5 results</dt><dd>{pct(m.rag.recall_at_5, 0)}</dd></div>
                    <div><dt>Answers citing the right call</dt><dd>{pct(m.rag.citation_accuracy, 0)}</dd></div>
                    <div><dt>Given answers that cite the right call</dt><dd>{pct(m.rag.answered_correct_call, 0)}</dd></div>
                    <div><dt>Answers containing the expected fact</dt><dd>{pct(m.rag.answer_term_match, 0)}</dd></div>
                    <div><dt>Unanswerable questions refused</dt><dd>{pct(m.rag.abstention_rate, 0)}</dd></div>
                  </dl>
                  <p className="footnote" style={{ padding: "0 16px 16px" }}>
                    Keyword search with one neighbouring turn, answered by {m.rag.model ?? "the configured model"}. The
                    expected-fact score is a strict text match, so it undercounts answers phrased differently.
                  </p>
                </section>
              )}
              {m.guard && (
                <section className="panel" aria-labelledby="q-guard">
                  <div className="panel-head">
                    <h2 id="q-guard">Index check</h2>
                    <span className={m.guard.passed ? "badge badge-ok" : "badge badge-warn"}>
                      {m.guard.passed ? "Passed" : "Failed"}
                    </span>
                  </div>
                  <dl className="facts">
                    <div><dt>Cards, SINs, accounts, emails, phones</dt><dd>{m.guard.structured_leaks} found</dd></div>
                    <div>
                      <dt>Names missed by redaction</dt>
                      <dd>
                        {m.guard.person_leaks} of {m.guard.person_support.toLocaleString()}
                      </dd>
                    </div>
                  </dl>
                </section>
              )}
              {m.models.length > 0 && (
                <section className="panel" aria-labelledby="q-models">
                  <div className="panel-head">
                    <h2 id="q-models">Name models (fine-tuned)</h2>
                  </div>
                  <div className="table-wrap">
                    <table className="table">
                      <caption className="sr-only">Name model size and training time</caption>
                      <thead>
                        <tr>
                          <th scope="col">Model</th>
                          <th scope="col" className="num">Dev F1</th>
                          <th scope="col" className="num">Size</th>
                          <th scope="col" className="num">Training</th>
                        </tr>
                      </thead>
                      <tbody>
                        {m.models.map((md) => (
                          <tr key={md.name}>
                            <th scope="row">{md.name}</th>
                            <td className="num">{score(md.dev_f1)}</td>
                            <td className="num">{md.size_mb.toFixed(0)} MB</td>
                            <td className="num">{Math.round(md.train_seconds / 60)} min</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
            </div>
          </div>
          <p className="footnote">
            Numbers come from the eval scripts' saved results. Try the search yourself on the <Link to="/">Ask</Link> page.
          </p>
        </>
      )}
    </div>
  );
}
