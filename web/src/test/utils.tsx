import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import axe from "axe-core";
import { MemoryRouter } from "react-router";
import { vi } from "vitest";
import { App } from "../App";

type Reply = { status?: number; body: unknown } | ((req: Request) => { status?: number; body: unknown });

/** Mocks fetch from a table like { "GET /api/calls": {...}, "POST /api/ask": (req) => ... }. Returns the spy. */
export function mockApi(routes: Record<string, Reply>) {
  const spy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const req = input instanceof Request ? input : new Request(input, init);
    const url = new URL(req.url);
    const key = `${req.method} ${url.pathname}`;
    const route = routes[key];
    if (!route) return new Response(JSON.stringify({ error: { code: "not_found", message: `No mock for ${key}` } }), { status: 404 });
    const { status = 200, body } = typeof route === "function" ? route(req) : route;
    return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

export function renderApp(path = "/") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

/** axe-core in jsdom. Colour contrast needs real layout, so it is checked in the browser run instead. */
export async function axeViolations(container: Element) {
  const r = await axe.run(container, { rules: { "color-contrast": { enabled: false }, region: { enabled: false } } });
  return r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
}

export const HEALTH = {
  status: "ok", version: "0.1.0", calls: 400,
  index: { available: true, embed_model: "bge-small", detector: "pipeline_hf", n_chunks: 5186 },
  llm: { provider: "ollama", model: "qwen2:7b" },
};

export const CALL_0171 = {
  call_id: "call_0171", split: "test", source: "multiwoz", duration_sec: 74,
  redactions: { EMAIL: 1, PERSON: 1 },
  turns: [
    { turn_id: 0, speaker: "agent", start_sec: 0, text: "Reservations line, my name's [PERSON], how may I help?" },
    { turn_id: 1, speaker: "customer", start_sec: 4, text: "I would like to find a theatre in town." },
    { turn_id: 2, speaker: "agent", start_sec: 21, text: "The Cambridge Arts Theatre, postcode cb23pj." },
    { turn_id: 3, speaker: "customer", start_sec: 53, text: "Drop me a line at [EMAIL]." },
  ],
};

export const PROVENANCE = { provider: "ollama", model: "qwen2:7b", prompt_version: "v1", k: 5, retrieval: "bm25", neighbors: 1, grounding: true };
