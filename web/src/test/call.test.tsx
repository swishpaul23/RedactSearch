import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axeViolations, CALL_0171, HEALTH, mockApi, renderApp } from "./utils";

function setup() {
  mockApi({
    "GET /api/health": { body: HEALTH },
    "GET /api/calls/call_0171": { body: CALL_0171 },
    "GET /api/calls/call_9999": { status: 404, body: { error: { code: "not_found", message: "No call with id call_9999." } } },
  });
}

test("shows the transcript, reads redactions aloud as words, and highlights the cited turn", async () => {
  setup();
  renderApp("/calls/call_0171?turn=2");
  expect(await screen.findByRole("heading", { level: 1, name: "call_0171" })).toBeInTheDocument();
  const transcript = await screen.findByRole("list", { name: "Transcript of call_0171" });
  const turns = within(transcript).getAllByRole("listitem");
  expect(turns[0]).toHaveTextContent("my name's redacted name, how may I help?");
  expect(turns[2]).toHaveAttribute("aria-current", "true");
  expect(turns[2]).toHaveTextContent("cited turn");
  expect(screen.getByRole("navigation", { name: "Breadcrumb" })).toHaveTextContent("Callscall_0171");
  expect(screen.getByLabelText("Call details")).toHaveTextContent("Length1:14");
});

test("tabs follow the arrow-key pattern", async () => {
  setup();
  const user = userEvent.setup();
  renderApp("/calls/call_0171");
  await screen.findByRole("list", { name: "Transcript of call_0171" });
  const [transcript, redactions] = screen.getAllByRole("tab");
  expect(transcript).toHaveAttribute("aria-selected", "true");
  expect(redactions).toHaveAttribute("tabindex", "-1");
  transcript!.focus();
  await user.keyboard("{ArrowRight}");
  expect(redactions).toHaveFocus();
  expect(redactions).toHaveAttribute("aria-selected", "true");
  const table = screen.getByRole("table", { name: "Redactions in call_0171 by type" });
  expect(within(table).getByText("email")).toBeInTheDocument();
});

test("an unknown call explains itself and links back", async () => {
  setup();
  renderApp("/calls/call_9999");
  expect(await screen.findByText("There is no call call_9999")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Browse all calls" })).toHaveAttribute("href", "/calls");
});

test("no accessibility violations", async () => {
  setup();
  const { container } = renderApp("/calls/call_0171?turn=2");
  await screen.findByRole("list", { name: "Transcript of call_0171" });
  expect(await axeViolations(container)).toEqual([]);
});
