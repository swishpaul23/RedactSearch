import { useQuery } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { ApiError, api, call } from "../api/client";
import { PageHeading } from "../components/PageHeading";
import { REDACTION_LABEL } from "../components/RedactedText";
import { Empty, ErrorBanner, SkeletonLines } from "../components/States";
import { Transcript } from "../components/Transcript";
import { clock, SOURCE_LABEL, SPLIT_LABEL } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";

const TABS = ["Transcript", "Redactions"] as const;
type Tab = (typeof TABS)[number];

const REDACTION_HELP: Record<string, string> = {
  PERSON: "People's names, found by the DistilBERT name model",
  CARD: "Card numbers that pass the Luhn checksum",
  PHONE: "Phone numbers",
  EMAIL: "Email addresses",
  SIN: "Social Insurance Numbers that pass the SIN check digit",
  ACCOUNT: "Account numbers",
};

export function CallPage() {
  const { callId = "" } = useParams();
  const [params] = useSearchParams();
  const turnParam = params.get("turn");
  const citedTurn = turnParam === null ? undefined : Number(turnParam);
  const [tab, setTab] = useState<Tab>("Transcript");
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const base = useId();
  useDocumentTitle(callId);

  const detail = useQuery({
    queryKey: ["call", callId],
    queryFn: () => call(() => api.GET("/api/calls/{call_id}", { params: { path: { call_id: callId } } })),
    retry: (n, e) => !(e instanceof ApiError && e.status === 404) && n < 2,
  });

  function onTabKey(e: React.KeyboardEvent, i: number) {
    const next = { ArrowRight: (i + 1) % TABS.length, ArrowLeft: (i - 1 + TABS.length) % TABS.length, Home: 0, End: TABS.length - 1 }[
      e.key as "ArrowRight" | "ArrowLeft" | "Home" | "End"
    ];
    if (next === undefined) return;
    e.preventDefault();
    setTab(TABS[next]!);
    tabRefs.current[next]?.focus();
  }

  const d = detail.data;
  const totalRedactions = d ? Object.values(d.redactions).reduce((a, b) => a + b, 0) : 0;
  const notFound = detail.error instanceof ApiError && detail.error.status === 404;

  return (
    <div className="page">
      <nav className="breadcrumb" aria-label="Breadcrumb">
        <ol>
          <li>
            <Link to="/calls">Calls</Link>
          </li>
          <li aria-current="page">{callId}</li>
        </ol>
      </nav>

      <div className="page-head">
        <PageHeading>{callId}</PageHeading>
        {d && (
          <dl className="meta" aria-label="Call details">
            <div><dt>Split</dt><dd>{SPLIT_LABEL[d.split]}</dd></div>
            <div><dt>Source</dt><dd>{SOURCE_LABEL[d.source]}</dd></div>
            <div><dt>Turns</dt><dd>{d.turns.length}</dd></div>
            <div><dt>Length</dt><dd>{clock(d.duration_sec)}</dd></div>
            <div><dt>Redactions</dt><dd>{totalRedactions}</dd></div>
          </dl>
        )}
      </div>

      {notFound ? (
        <section className="panel">
          <Empty title={`There is no call ${callId}`}>
            <p>The link may be mistyped, or the call is not in the redacted archive.</p>
            <Link className="btn btn-sm" to="/calls">
              Browse all calls
            </Link>
          </Empty>
        </section>
      ) : detail.isError ? (
        <ErrorBanner error={detail.error} onRetry={() => detail.refetch()} title="This call could not be loaded" />
      ) : (
        <section className="panel">
          <div className="tabs" role="tablist" aria-label="Call views" style={{ padding: "0 16px" }}>
            {TABS.map((t, i) => (
              <button
                key={t}
                ref={(el) => {
                  tabRefs.current[i] = el;
                }}
                type="button"
                role="tab"
                id={`${base}-tab-${i}`}
                aria-selected={tab === t}
                aria-controls={`${base}-panel-${i}`}
                tabIndex={tab === t ? 0 : -1}
                className="tab"
                onClick={() => setTab(t)}
                onKeyDown={(e) => onTabKey(e, i)}
              >
                {t === "Redactions" && d ? `Redactions (${totalRedactions})` : t}
              </button>
            ))}
          </div>
          <div
            role="tabpanel"
            id={`${base}-panel-${TABS.indexOf(tab)}`}
            aria-labelledby={`${base}-tab-${TABS.indexOf(tab)}`}
            tabIndex={0}
          >
            {detail.isPending && <SkeletonLines lines={10} label="Loading transcript" />}
            {d && tab === "Transcript" && (
              <>
                {citedTurn !== undefined && (
                  <p className="hint" style={{ padding: "12px 16px 0" }}>
                    The highlighted turn is the one cited in your answer.
                  </p>
                )}
                <Transcript call={d} citedTurn={citedTurn} />
              </>
            )}
            {d && tab === "Redactions" &&
              (totalRedactions === 0 ? (
                <Empty title="Nothing was redacted in this call">
                  <p>No names, card numbers or other personal details were found.</p>
                </Empty>
              ) : (
                <div className="table-wrap">
                  <table className="table">
                    <caption className="sr-only">Redactions in {callId} by type</caption>
                    <thead>
                      <tr>
                        <th scope="col">Type</th>
                        <th scope="col" className="num">Count</th>
                        <th scope="col">What it covers</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(d.redactions).map(([type, n]) => (
                        <tr key={type}>
                          <th scope="row">
                            <span className="redaction">{REDACTION_LABEL[type] ?? type.toLowerCase()}</span>
                          </th>
                          <td className="num">{n}</td>
                          <td>{REDACTION_HELP[type] ?? "Personal data"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
          </div>
        </section>
      )}
    </div>
  );
}
