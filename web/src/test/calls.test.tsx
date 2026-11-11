import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axeViolations, CALL_0171, HEALTH, mockApi, renderApp } from "./utils";

const ROW = (i: number, extra = {}) => ({
  call_id: `call_${String(i).padStart(4, "0")}`, split: "test", source: "multiwoz", n_turns: 10 + i,
  duration_sec: 60 + i, redactions: i % 4, preview: "Hi, this is [PERSON] calling about a refund.", ...extra,
});

function setup(total = 60) {
  const seen: URLSearchParams[] = [];
  const fetch = mockApi({
    "GET /api/health": { body: HEALTH },
    "GET /api/calls": (req) => {
      const p = new URL(req.url).searchParams;
      seen.push(p);
      const q = p.get("q");
      const n = q === "zzz" ? 0 : total;
      const size = Number(p.get("page_size"));
      const page = Number(p.get("page"));
      const items = Array.from({ length: Math.max(0, Math.min(size, n - (page - 1) * size)) }, (_, i) => ROW((page - 1) * size + i + 1));
      return { body: { items, total: n, page, page_size: size } };
    },
    "GET /api/calls/call_0171": { body: CALL_0171 },
  });
  return { fetch, seen, last: () => seen[seen.length - 1]! };
}

test("lists calls in a labelled table with redaction marks and pagination", async () => {
  setup();
  renderApp("/calls");
  const table = await screen.findByRole("table", { name: "Calls, sorted by call ID ascending" });
  expect(await within(table).findByRole("link", { name: "call_0001" })).toHaveAttribute("href", "/calls/call_0001");
  expect(within(table).getAllByText("redacted", { exact: false }).length).toBeGreaterThan(0);
  expect(screen.getByText("Showing 1–25 of 60 calls")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
  expect(screen.getByText("Page 1 of 3")).toBeInTheDocument();
});

test("sorting toggles aria-sort and the request", async () => {
  const { last } = setup();
  const user = userEvent.setup();
  renderApp("/calls");
  await screen.findByRole("link", { name: "call_0001" });
  await user.click(screen.getByRole("button", { name: /^Redactions/ }));
  await waitFor(() => expect(last().get("sort")).toBe("redactions"));
  expect(screen.getByRole("columnheader", { name: /Redactions/ })).toHaveAttribute("aria-sort", "ascending");
  await user.click(screen.getByRole("button", { name: /^Redactions/ }));
  await waitFor(() => expect(last().get("order")).toBe("desc"));
  expect(screen.getByRole("columnheader", { name: /Redactions/ })).toHaveAttribute("aria-sort", "descending");
});

test("one Filters control; active filters show as removable chips", async () => {
  const { last } = setup();
  const user = userEvent.setup();
  renderApp("/calls");
  await screen.findByRole("link", { name: "call_0001" });
  const filters = screen.getByRole("button", { name: "Filters" });
  await user.click(filters);
  expect(filters).toHaveAttribute("aria-expanded", "true");
  await user.click(screen.getByRole("radio", { name: "Dev" }));
  await waitFor(() => expect(last().get("split")).toBe("dev"));
  const chips = screen.getByRole("group", { name: "Active filters" });
  expect(within(chips).getByText("Split: Dev")).toBeInTheDocument();
  await user.keyboard("{Escape}");
  expect(filters).toHaveFocus();
  await user.click(within(chips).getByRole("button", { name: "Remove filter Split: Dev" }));
  await waitFor(() => expect(last().get("split")).toBeNull());
});

test("search is debounced, and an empty result stays inside the table frame", async () => {
  const { last } = setup();
  const user = userEvent.setup();
  renderApp("/calls");
  await screen.findByRole("link", { name: "call_0001" });
  document.body.focus();
  await user.keyboard("/");
  const search = screen.getByRole("searchbox", { name: "Search calls" });
  expect(search).toHaveFocus();
  await user.type(search, "zzz");
  await waitFor(() => expect(last().get("q")).toBe("zzz"));
  expect(await screen.findByText("No calls match “zzz”")).toBeInTheDocument();
  expect(screen.getByRole("table")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Clear search and filters" }));
  await waitFor(() => expect(last().get("q")).toBeNull());
});

test("next page and page size", async () => {
  const { last } = setup();
  const user = userEvent.setup();
  renderApp("/calls");
  await screen.findByRole("link", { name: "call_0001" });
  await user.click(screen.getByRole("button", { name: "Next" }));
  await waitFor(() => expect(last().get("page")).toBe("2"));
  expect(await screen.findByText("Showing 26–50 of 60 calls")).toBeInTheDocument();
  await user.selectOptions(screen.getByRole("combobox", { name: "Rows per page" }), "50");
  await waitFor(() => expect(last().get("page_size")).toBe("50"));
  expect(last().get("page")).toBe("1");
});

test("no accessibility violations", async () => {
  setup();
  const { container } = renderApp("/calls");
  await screen.findByRole("link", { name: "call_0001" });
  expect(await axeViolations(container)).toEqual([]);
});
