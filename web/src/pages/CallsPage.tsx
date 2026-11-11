import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { api, call } from "../api/client";
import { CloseIcon, FilterIcon, SearchIcon, SortIcon } from "../components/Icons";
import { PageHeading } from "../components/PageHeading";
import { RedactedText } from "../components/RedactedText";
import { Empty, ErrorBanner, SkeletonLines } from "../components/States";
import { clock, SOURCE_LABEL, SPLIT_LABEL } from "../lib/format";
import { useDocumentTitle, useSlashFocus } from "../lib/hooks";

type SortKey = "call_id" | "n_turns" | "duration_sec" | "redactions";
const SORTABLE: { key: SortKey; label: string; num?: boolean }[] = [
  { key: "call_id", label: "Call" },
  { key: "n_turns", label: "Turns", num: true },
  { key: "duration_sec", label: "Length", num: true },
  { key: "redactions", label: "Redactions", num: true },
];
const SORT_NAME: Record<SortKey, string> = { call_id: "call ID", n_turns: "turns", duration_sec: "length", redactions: "redactions" };
const PAGE_SIZES = [25, 50, 100];
const SPLITS = ["train", "dev", "test"] as const;
const SOURCES = ["template", "multiwoz"] as const;

function Filters({ split, source, onChange, onClose }: {
  split: string;
  source: string;
  onChange: (key: "split" | "source", value: string) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.querySelector<HTMLInputElement>("input:checked")?.focus();
  }, []);
  return (
    <div
      ref={ref}
      className="filter-panel"
      id="call-filters"
      role="group"
      aria-label="Filters"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
      }}
    >
      <fieldset>
        <legend>Split</legend>
        {[["", "All splits"], ...SPLITS.map((s) => [s, SPLIT_LABEL[s]!])].map(([v, label]) => (
          <label className="radio" key={v}>
            <input type="radio" name="split" value={v} checked={split === v} onChange={() => onChange("split", v!)} />
            {label}
          </label>
        ))}
      </fieldset>
      <fieldset>
        <legend>Source</legend>
        {[["", "All sources"], ...SOURCES.map((s) => [s, SOURCE_LABEL[s]!])].map(([v, label]) => (
          <label className="radio" key={v}>
            <input type="radio" name="source" value={v} checked={source === v} onChange={() => onChange("source", v!)} />
            {label}
          </label>
        ))}
      </fieldset>
      <div style={{ alignSelf: "end" }}>
        <button type="button" className="btn btn-sm" onClick={onClose}>
          Done
        </button>
      </div>
    </div>
  );
}

export function CallsPage() {
  useDocumentTitle("Calls");
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const split = params.get("split") ?? "";
  const source = params.get("source") ?? "";
  const sort = (params.get("sort") as SortKey) || "call_id";
  const order = params.get("order") === "desc" ? "desc" : "asc";
  const page = Math.max(1, Number(params.get("page")) || 1);
  const size = PAGE_SIZES.includes(Number(params.get("size"))) ? Number(params.get("size")) : 25;

  const [draft, setDraft] = useState(q);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const filtersBtn = useRef<HTMLButtonElement>(null);
  useSlashFocus(searchRef);
  const ids = { search: useId(), caption: useId() };

  function update(changes: Record<string, string>, resetPage = true) {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(changes)) {
      if (v === "") next.delete(k);
      else next.set(k, v);
    }
    if (resetPage) next.delete("page");
    setParams(next, { replace: true });
  }

  // Search as you type, settled after 250 ms.
  useEffect(() => {
    if (draft === q) return;
    const id = window.setTimeout(() => update({ q: draft.trim() }), 250);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  const list = useQuery({
    queryKey: ["calls", { q, split, source, sort, order, page, size }],
    queryFn: () =>
      call(() =>
        api.GET("/api/calls", {
          params: {
            query: {
              q: q || undefined,
              split: (split || undefined) as "train" | "dev" | "test" | undefined,
              source: (source || undefined) as "template" | "multiwoz" | undefined,
              sort,
              order,
              page,
              page_size: size,
            },
          },
        }),
      ),
    placeholderData: keepPreviousData,
  });

  const total = list.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / size));
  const from = total === 0 ? 0 : (page - 1) * size + 1;
  const to = Math.min(page * size, total);
  const chips = [
    split && { key: "split", label: `Split: ${SPLIT_LABEL[split]}` },
    source && { key: "source", label: `Source: ${SOURCE_LABEL[source]}` },
  ].filter(Boolean) as { key: string; label: string }[];

  function toggleSort(key: SortKey) {
    const nextOrder = sort === key && order === "asc" ? "desc" : "asc";
    update({ sort: key === "call_id" ? "" : key, order: nextOrder === "asc" ? "" : "desc" });
  }

  return (
    <div className="page">
      <div className="page-head">
        <PageHeading>Calls</PageHeading>
        <p className="page-desc">
          Every transcript in the archive, after redaction. Names, card numbers, phone numbers, emails, SINs and account
          numbers appear as solid marks.
        </p>
      </div>

      <div className="toolbar">
        <div className="field search-field">
          <label className="label" htmlFor={ids.search}>
            Search calls
          </label>
          <div className="search-box">
            <SearchIcon />
            <input
              id={ids.search}
              ref={searchRef}
              type="search"
              className="input"
              value={draft}
              placeholder="Call ID or words in the transcript  (/)"
              onChange={(e) => setDraft(e.target.value)}
            />
          </div>
        </div>
        <button
          ref={filtersBtn}
          type="button"
          className="btn"
          aria-expanded={filtersOpen}
          aria-controls="call-filters"
          onClick={() => setFiltersOpen(!filtersOpen)}
        >
          <FilterIcon />
          Filters{chips.length > 0 ? ` (${chips.length})` : ""}
        </button>
      </div>

      {filtersOpen && (
        <Filters
          split={split}
          source={source}
          onChange={(k, v) => update({ [k]: v })}
          onClose={() => {
            setFiltersOpen(false);
            filtersBtn.current?.focus();
          }}
        />
      )}

      {chips.length > 0 && (
        <div className="chips" aria-label="Active filters" role="group">
          {chips.map((c) => (
            <span className="chip" key={c.key}>
              {c.label}
              <button type="button" className="chip-remove" aria-label={`Remove filter ${c.label}`} onClick={() => update({ [c.key]: "" })}>
                <CloseIcon />
              </button>
            </span>
          ))}
          <button type="button" className="btn btn-sm btn-quiet" onClick={() => update({ split: "", source: "" })}>
            Clear all
          </button>
        </div>
      )}

      <section className="panel" aria-labelledby={ids.caption}>
        {list.isError && (
          <div className="panel-body">
            <ErrorBanner error={list.error} onRetry={() => list.refetch()} title="Calls could not be loaded" />
          </div>
        )}
        <div className="table-wrap">
          <table className="table" aria-busy={list.isFetching}>
            <caption id={ids.caption} className="sr-only">
              Calls, sorted by {SORT_NAME[sort]} {order === "asc" ? "ascending" : "descending"}
            </caption>
            <thead>
              <tr>
                {SORTABLE.slice(0, 1).map((c) => (
                  <SortHeader key={c.key} col={c} sort={sort} order={order} onSort={toggleSort} />
                ))}
                <th scope="col">Source</th>
                <th scope="col">Split</th>
                {SORTABLE.slice(1).map((c) => (
                  <SortHeader key={c.key} col={c} sort={sort} order={order} onSort={toggleSort} />
                ))}
                <th scope="col">Opening line</th>
              </tr>
            </thead>
            <tbody>
              {list.data?.items.map((c) => (
                <tr key={c.call_id}>
                  <th scope="row">
                    <Link className="row-link" to={`/calls/${c.call_id}`}>
                      {c.call_id}
                    </Link>
                  </th>
                  <td>{SOURCE_LABEL[c.source]}</td>
                  <td>
                    <span className="badge">{SPLIT_LABEL[c.split]}</span>
                  </td>
                  <td className="num">{c.n_turns}</td>
                  <td className="num">{clock(c.duration_sec)}</td>
                  <td className="num">{c.redactions}</td>
                  <td className="cell-preview">
                    <RedactedText text={c.preview} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {list.isPending && <SkeletonLines lines={8} label="Loading calls" />}
        {list.data && total === 0 && (
          <Empty title={q ? `No calls match “${q}”` : "No calls match these filters"}>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                setDraft("");
                update({ q: "", split: "", source: "" });
              }}
            >
              Clear search and filters
            </button>
          </Empty>
        )}
        <nav className="pagination" aria-label="Pagination">
          <p aria-live="polite">
            {list.data ? (total === 0 ? "No calls" : `Showing ${from}–${to} of ${total} calls`) : "Loading calls…"}
          </p>
          <div className="pagination-controls">
            <label className="hint" htmlFor="page-size">
              Rows per page
            </label>
            <select
              id="page-size"
              className="select"
              value={size}
              onChange={(e) => update({ size: e.target.value === "25" ? "" : e.target.value })}
            >
              {PAGE_SIZES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="btn btn-sm"
              disabled={page <= 1}
              onClick={() => update({ page: page - 1 === 1 ? "" : String(page - 1) }, false)}
            >
              Previous
            </button>
            <span>
              Page {Math.min(page, pages)} of {pages}
            </span>
            <button
              type="button"
              className="btn btn-sm"
              disabled={page >= pages}
              onClick={() => update({ page: String(page + 1) }, false)}
            >
              Next
            </button>
          </div>
        </nav>
      </section>
    </div>
  );
}

function SortHeader({ col, sort, order, onSort }: {
  col: { key: SortKey; label: string; num?: boolean };
  sort: SortKey;
  order: "asc" | "desc";
  onSort: (k: SortKey) => void;
}) {
  const active = sort === col.key;
  const dir = active ? (order === "asc" ? "ascending" : "descending") : "none";
  return (
    <th scope="col" aria-sort={dir} className={col.num ? "num" : undefined}>
      <button type="button" className="sort-btn" onClick={() => onSort(col.key)}>
        {col.label}
        <SortIcon dir={dir} />
        <span className="sr-only">, sort {active && order === "asc" ? "descending" : "ascending"}</span>
      </button>
    </th>
  );
}
