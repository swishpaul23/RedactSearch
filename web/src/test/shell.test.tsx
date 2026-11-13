import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { clock, parseChunkId, pct } from "../lib/format";
import { axeViolations, HEALTH, mockApi, renderApp } from "./utils";

test("skip link, labelled navigation with the current page, and system status", async () => {
  mockApi({ "GET /api/health": { body: HEALTH }, "GET /api/metrics": { body: { split: "test", redaction: [], rag: null, guard: null, models: [] } } });
  renderApp("/quality");
  expect(screen.getByRole("link", { name: "Skip to main content" })).toHaveAttribute("href", "#main");
  expect(screen.getByRole("link", { name: "Quality" })).toHaveAttribute("aria-current", "page");
  expect(await screen.findByText("Index ready, 5,186 turns")).toBeInTheDocument();
  expect(screen.getByText("Answers by qwen2:7b")).toBeInTheDocument();
});

test("API down shows in the sidebar", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
  renderApp("/calls");
  expect(await screen.findByText("API unreachable")).toBeInTheDocument();
  expect(await screen.findByText("The server could not be reached. Check that the API is running.")).toBeInTheDocument();
});

test("theme button cycles and remembers the choice", async () => {
  mockApi({ "GET /api/health": { body: HEALTH } });
  const user = userEvent.setup();
  renderApp("/nope");
  await user.click(screen.getByRole("button", { name: /^Theme: Match system/ }));
  expect(document.documentElement).toHaveAttribute("data-theme", "light");
  await user.click(screen.getByRole("button", { name: /^Theme: Light/ }));
  expect(document.documentElement).toHaveAttribute("data-theme", "dark");
  expect(JSON.parse(window.localStorage.getItem("redactsearch.theme")!)).toBe("dark");
});

test("unknown page offers ways back, accessibly", async () => {
  mockApi({ "GET /api/health": { body: HEALTH } });
  const { container } = renderApp("/nope");
  expect(screen.getByRole("heading", { level: 1, name: "Page not found" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Browse calls" })).toHaveAttribute("href", "/calls");
  expect(await axeViolations(container)).toEqual([]);
});

test("formatters", () => {
  expect(clock(75.4)).toBe("1:15");
  expect(clock(0)).toBe("0:00");
  expect(pct(0.0227)).toBe("2.3%");
  expect(parseChunkId("call_0079:3")).toEqual({ callId: "call_0079", turnId: 3 });
});
