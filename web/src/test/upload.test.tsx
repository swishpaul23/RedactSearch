import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axeViolations, HEALTH, mockApi, renderApp } from "./utils";

const file = (name: string, text = "Agent: hello", type = "text/plain") => new File([text], name, { type });

const JOB = {
  id: "abc123def456", created_at: "2026-11-11T10:00:00+00:00", n_calls: 2, call_ids: [], redactions: {},
  indexed: false, message: null,
  files: [
    { name: "calls.jsonl", calls: 2, turns: 9, issues: [{ where: "Line 3", message: "Not valid JSON." }] },
  ],
};

function setup(final: Record<string, unknown>) {
  let polls = 0;
  const sent: string[] = [];
  mockApi({
    "GET /api/health": { body: HEALTH },
    "POST /api/uploads": (req) => {
      sent.push(req.headers.get("content-type") ?? "");
      return { status: 202, body: { ...JOB, status: "queued" } };
    },
    "GET /api/uploads/abc123def456": () => {
      polls += 1;
      return { body: polls < 2 ? { ...JOB, status: "redacting" } : { ...JOB, ...final } };
    },
  });
  return { sent };
}

test("submit with no files explains what to do and keeps the button enabled", async () => {
  setup({ status: "done" });
  const user = userEvent.setup();
  const { container } = renderApp("/upload");
  expect(await screen.findByRole("heading", { level: 1, name: "Upload transcripts" })).toBeInTheDocument();
  const submit = screen.getByRole("button", { name: "Upload and redact" });
  await user.click(submit);
  expect(screen.getByText("Choose at least one file.")).toBeInTheDocument();
  expect(screen.getByLabelText("Transcript files")).toHaveAttribute("aria-invalid", "true");
  expect(submit).toBeEnabled();
  expect(await axeViolations(container)).toEqual([]);
});

test("files are checked on choose, can be removed, and the error clears", async () => {
  setup({ status: "done" });
  const user = userEvent.setup({ applyAccept: false });
  renderApp("/upload");
  const input = await screen.findByLabelText("Transcript files");
  await user.click(screen.getByRole("button", { name: "Upload and redact" }));
  await user.upload(input, [file("a.txt"), file("notes.docx")]);
  expect(screen.queryByText("Choose at least one file.")).not.toBeInTheDocument();
  const list = screen.getByRole("list", { name: /2 files chosen/ });
  expect(within(list).getByText("Not a .jsonl, .json or .txt file")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Remove notes.docx" }));
  expect(screen.getByRole("list", { name: "1 file chosen" })).toBeInTheDocument();
});

test("uploads, shows progress, then links the imported calls and the file issues", async () => {
  const { sent } = setup({ status: "done", call_ids: ["up-abc123def456-001", "up-abc123def456-002"],
    redactions: { PERSON: 3, CARD: 1 }, indexed: true });
  const user = userEvent.setup();
  const { container } = renderApp("/upload");
  await user.upload(await screen.findByLabelText("Transcript files"), [file("calls.jsonl", '{"turns": []}')]);
  await user.click(screen.getByRole("button", { name: "Upload and redact" }));
  expect(sent[0]).toMatch(/^multipart\/form-data; boundary=/);
  expect(await screen.findByText("Redact personal data", {}, { timeout: 3000 })).toBeInTheDocument();
  expect(await screen.findByText("Imported 2 calls.", {}, { timeout: 3000 })).toBeInTheDocument();
  expect(screen.getByText(/Redacted 3 names, 1 card number/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "up-abc123def456-001" })).toHaveAttribute("href", "/calls/up-abc123def456-001");
  const issues = screen.getByRole("region", { name: "Issues in calls.jsonl" });
  expect(within(issues).getByText("Line 3")).toBeInTheDocument();
  expect(await axeViolations(container)).toEqual([]);
  await user.click(screen.getByRole("button", { name: "Upload more files" }));
  await waitFor(() => expect(screen.getByLabelText("Transcript files")).toHaveFocus());
});

test("a failed upload says nothing was imported and why", async () => {
  setup({ status: "failed", message: "No calls could be imported. Fix the issues listed and upload again." });
  const user = userEvent.setup();
  renderApp("/upload");
  await user.upload(await screen.findByLabelText("Transcript files"), [file("calls.jsonl")]);
  await user.click(screen.getByRole("button", { name: "Upload and redact" }));
  expect(await screen.findByText("Nothing was imported", {}, { timeout: 3000 })).toBeInTheDocument();
  expect(screen.getByText(/Fix the issues listed/)).toBeInTheDocument();
});

test("leaving with chosen files asks first", async () => {
  setup({ status: "done" });
  const user = userEvent.setup();
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.setAttribute("open", "");
  };
  HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
    this.removeAttribute("open");
    this.dispatchEvent(new Event("close"));
  };
  renderApp("/upload");
  await user.upload(await screen.findByLabelText("Transcript files"), [file("a.txt")]);
  await user.click(screen.getByRole("link", { name: "Calls" }));
  const dialog = await screen.findByRole("dialog", { name: "Leave without uploading?" });
  await user.click(within(dialog).getByRole("button", { name: "Stay on page" }));
  expect(screen.getByRole("heading", { level: 1, name: "Upload transcripts" })).toBeInTheDocument();
});
