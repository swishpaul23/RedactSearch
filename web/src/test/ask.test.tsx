import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { explainRejection } from "../pages/AskPage";
import { axeViolations, CALL_0171, HEALTH, mockApi, PROVENANCE, renderApp } from "./utils";

const ANSWER = {
  question: "What is the postcode of the Cambridge Arts Theatre?",
  answer: "Its postcode is cb23pj.",
  found: true, rejected: null, retrieved: ["call_0171:2", "call_0003:1"], provenance: PROVENANCE, elapsed_ms: 2400,
  citations: [
    { chunk_id: "call_0171:2", call_id: "call_0171", turn_id: 2, speaker: "agent", start_sec: 21, snippet: "The Cambridge Arts Theatre, postcode cb23pj." },
    { chunk_id: "call_0003:1", call_id: "call_0003", turn_id: 1, speaker: "customer", start_sec: 4, snippet: "Is it near [PERSON]'s place?" },
  ],
};
const CALL_0003 = { ...CALL_0171, call_id: "call_0003", source: "template" };

function routes(ask: unknown = { body: ANSWER }) {
  return {
    "GET /api/health": { body: HEALTH },
    "POST /api/ask": ask as never,
    "GET /api/calls/call_0171": { body: CALL_0171 },
    "GET /api/calls/call_0003": { body: CALL_0003 },
  };
}

test("asks with Enter, shows the cited answer and the transcript with the cited turn", async () => {
  const fetch = mockApi(routes());
  const user = userEvent.setup();
  renderApp("/");
  const box = screen.getByRole("textbox", { name: "Question" });
  await user.type(box, "What is the postcode of the Cambridge Arts Theatre?{Enter}");

  expect(await screen.findByText("Its postcode is cb23pj.")).toBeInTheDocument();
  expect(screen.getByText("Answered from 2 call turns")).toBeInTheDocument();
  const body = await (fetch.mock.calls.find(([r]) => (r as Request).method === "POST")![0] as Request).json();
  expect(body).toEqual({ question: "What is the postcode of the Cambridge Arts Theatre?", k: 5, mode: "bm25", split: null });

  const transcript = await screen.findByRole("list", { name: "Transcript of call_0171" });
  const cited = within(transcript).getAllByRole("listitem").find((li) => li.getAttribute("aria-current") === "true");
  expect(cited).toHaveTextContent("Cambridge Arts Theatre");
  expect(screen.getByRole("link", { name: "Open full call call_0171" })).toHaveAttribute("href", "/calls/call_0171?turn=2");
  expect(screen.getByText("Answer ready, citing 2 call turns.")).toBeInTheDocument(); // live region
});

test("arrow keys move between sources and Enter moves focus to the transcript", async () => {
  mockApi(routes());
  const user = userEvent.setup();
  renderApp("/");
  await user.type(screen.getByRole("textbox", { name: "Question" }), "postcode of the theatre{Enter}");
  const list = await screen.findByRole("listbox", { name: "Sources" });
  const options = within(list).getAllByRole("option");
  expect(options[0]).toHaveAttribute("aria-selected", "true");

  list.focus();
  await user.keyboard("{ArrowDown}");
  expect(options[1]).toHaveAttribute("aria-selected", "true");
  expect(list).toHaveAttribute("aria-activedescendant", options[1]!.id);
  expect(await screen.findByRole("heading", { name: "call_0003" })).toBeInTheDocument();
  await user.keyboard("{Home}");
  expect(options[0]).toHaveAttribute("aria-selected", "true");
  await user.keyboard("{Enter}");
  expect(document.activeElement).toHaveAttribute("id", "preview");
});

test("explains a withheld answer in plain words", async () => {
  mockApi(routes({ body: { ...ANSWER, answer: "NOT_FOUND", found: false, citations: [],
    rejected: "question term 'sauna' is not in the cited excerpts" } }));
  const user = userEvent.setup();
  renderApp("/");
  await user.type(screen.getByRole("textbox", { name: "Question" }), "Which guesthouse has a sauna?{Enter}");
  expect(await screen.findByText("Answer withheld")).toBeInTheDocument();
  expect(screen.getByText(/does not mention “sauna”/)).toBeInTheDocument();
  expect(screen.getByText("No call to show")).toBeInTheDocument();
});

test("shows the server's error message with a retry", async () => {
  let calls = 0;
  mockApi(routes(() => (++calls === 1
    ? { status: 503, body: { error: { code: "answer_model_unavailable", message: "The answer model could not be reached." } } }
    : { body: ANSWER })));
  const user = userEvent.setup();
  renderApp("/");
  await user.type(screen.getByRole("textbox", { name: "Question" }), "postcode?{Enter}");
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("The answer model could not be reached.");
  await user.click(within(alert).getByRole("button", { name: "Try again" }));
  expect(await screen.findByText("Its postcode is cb23pj.")).toBeInTheDocument();
});

test("Ask is disabled until the question is long enough; Shift+Enter adds a line", async () => {
  mockApi(routes());
  const user = userEvent.setup();
  renderApp("/");
  const ask = screen.getByRole("button", { name: "Ask" });
  const box = screen.getByRole("textbox", { name: "Question" });
  expect(ask).toBeDisabled();
  await user.type(box, "hi{Shift>}{Enter}{/Shift}there");
  expect(box).toHaveValue("hi\nthere");
  expect(ask).toBeEnabled();
});

test("slash focuses the question box; settings are a labelled disclosure", async () => {
  mockApi(routes());
  const user = userEvent.setup();
  renderApp("/");
  document.body.focus();
  await user.keyboard("/");
  expect(screen.getByRole("textbox", { name: "Question" })).toHaveFocus();
  const toggle = screen.getByRole("button", { name: "Search settings" });
  expect(toggle).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByRole("combobox", { name: "Match transcripts by" })).not.toBeInTheDocument();
  await user.click(toggle);
  await user.selectOptions(screen.getByRole("combobox", { name: "Match transcripts by" }), "hybrid");
  expect(JSON.parse(window.localStorage.getItem("redactsearch.ask-settings")!)).toMatchObject({ mode: "hybrid" });
});

test("example questions run a search", async () => {
  mockApi(routes());
  const user = userEvent.setup();
  renderApp("/");
  await user.click(screen.getByRole("button", { name: "Why did the customer want a refund?" }));
  expect(await screen.findByText("Its postcode is cb23pj.")).toBeInTheDocument();
});

test("no accessibility violations, idle and with an answer", async () => {
  mockApi(routes());
  const user = userEvent.setup();
  const { container } = renderApp("/");
  expect(await axeViolations(container)).toEqual([]);
  await user.type(screen.getByRole("textbox", { name: "Question" }), "postcode?{Enter}");
  await screen.findByRole("list", { name: "Transcript of call_0171" });
  await waitFor(async () => expect(await axeViolations(container)).toEqual([]));
});

test("explainRejection covers each server reason", () => {
  expect(explainRejection("question term 'ferry' appears nowhere in the index")).toMatch(/No call mentions “ferry”/);
  expect(explainRejection("answer had no citations")).toMatch(/without citing/);
  expect(explainRejection("cited 9, which was not retrieved")).toMatch(/not given/);
  expect(explainRejection("unparseable reply")).toMatch(/could not be read/);
  expect(explainRejection("something new")).toBe("something new");
});
