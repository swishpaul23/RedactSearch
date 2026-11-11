import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { explainRejection } from "../pages/AskPage";
import { axeViolations, CALL_0171, HEALTH, mockApi, renderApp } from "./utils";

const AGENT = {
  question: "Do agents or customers mention 'theatre' more?",
  answer: "Customers mention it in 3 turns, agents in 1; one caller asked for a theatre in town.",
  found: true,
  citations: [{ chunk_id: "call_0171:1", call_id: "call_0171", turn_id: 1, speaker: "customer", start_sec: 4, snippet: "I would like to find a theatre in town." }],
  cited_calls: ["call_0171", "call_0003"],
  cited_results: [1],
  steps: [
    { n: 1, tool: "compute_stats", args: { query: "theatre", split: "test", source: null }, summary: "100 calls (test split), 3 matching the query", ok: true, ms: 40 },
    { n: 2, tool: "search_calls", args: { query: "theatre", k: 5 }, summary: "5 turns from 3 calls", ok: true, ms: 12 },
    { n: 3, tool: "search_calls", args: { query: "theatre", k: 5 }, summary: "repeat of r2", ok: false, ms: 0 },
  ],
  rejected: null,
  provenance: { provider: "ollama", model: "qwen2:7b", prompt_version: "v1", max_steps: 6, retrieval: "bm25" },
  llm_calls: 4,
  elapsed_ms: 18400,
};

function setup(reply: object = AGENT) {
  const bodies: unknown[] = [];
  mockApi({
    "GET /api/health": { body: HEALTH },
    "POST /api/agent": (req) => {
      void req.clone().json().then((b) => bodies.push(b));
      return { body: reply };
    },
    "POST /api/ask": { status: 500, body: { error: { code: "x", message: "wrong route" } } },
    "GET /api/calls/call_0171": { body: CALL_0171 },
    "GET /api/calls/call_0003": { body: { ...CALL_0171, call_id: "call_0003" } },
  });
  return bodies;
}

beforeEach(() => localStorage.clear());

test("agent mode posts to the agent, shows its steps and every cited source", async () => {
  const bodies = setup();
  const user = userEvent.setup();
  const { container } = renderApp("/");
  await user.click(await screen.findByRole("radio", { name: /Agent/ }));
  expect(screen.queryByRole("button", { name: "Search settings" })).not.toBeInTheDocument();
  await user.type(screen.getByRole("textbox", { name: "Question" }), "Do agents or customers mention 'theatre' more?{Enter}");

  expect(await screen.findByText(/Customers mention it in 3 turns/)).toBeInTheDocument();
  expect(bodies[0]).toEqual({ question: "Do agents or customers mention 'theatre' more?" });
  expect(screen.getByText("Answered from 3 sources")).toBeInTheDocument();
  const steps = screen.getByRole("list", { name: "How the agent worked" });
  const items = within(steps).getAllByRole("listitem");
  expect(items).toHaveLength(3);
  expect(within(items[0]!).getByText("Counted calls and turns")).toBeInTheDocument();
  expect(within(items[0]!).getByText("cited")).toBeInTheDocument();
  expect(within(items[0]!).getByText("query “theatre”, split “test”")).toBeInTheDocument();
  expect(within(items[2]!).getByText("repeat of r2")).toBeInTheDocument();
  expect(screen.getByText("3 of 6")).toBeInTheDocument();

  const sources = screen.getByRole("listbox", { name: "Sources" });
  expect(within(sources).getAllByRole("option")).toHaveLength(2);
  expect(within(sources).getByText("Whole call")).toBeInTheDocument();
  await user.click(within(sources).getByText("call_0003"));
  expect(await screen.findByText(/cited as a whole call/)).toBeInTheDocument();
  expect(await axeViolations(container)).toEqual([]);
});

test("the method choice is remembered and agent examples are offered", async () => {
  setup();
  const user = userEvent.setup();
  const first = renderApp("/");
  await user.click(await screen.findByRole("radio", { name: /Agent/ }));
  expect(screen.getByRole("button", { name: /How many test-split calls mention 'museum'/ })).toBeInTheDocument();
  first.unmount();
  renderApp("/");
  expect(await screen.findByRole("radio", { name: /Agent/ })).toBeChecked();
});

test("agent refusals are explained", async () => {
  setup({ ...AGENT, found: false, answer: "NOT_FOUND", citations: [], cited_calls: [], cited_results: [], rejected: "step cap of 6 tool calls reached" });
  const user = userEvent.setup();
  renderApp("/");
  await user.click(await screen.findByRole("radio", { name: /Agent/ }));
  await user.type(screen.getByRole("textbox", { name: "Question" }), "Something hard?{Enter}");
  expect(await screen.findByText(/used all of its tool calls/)).toBeInTheDocument();
  expect(screen.getByText("Answer withheld")).toBeInTheDocument();
  expect(explainRejection("cited 'call_9:1', which was not in any tool result")).toMatch(/never looked at/);
});
