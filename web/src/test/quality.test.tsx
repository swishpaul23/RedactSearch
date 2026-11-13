import { screen, within } from "@testing-library/react";
import { axeViolations, HEALTH, mockApi, renderApp } from "./utils";

const METRICS = {
  split: "test",
  redaction: [
    { detector: "pipeline_hf", label: "Regex + DistilBERT", person_precision: 0.981, person_recall: 0.959, person_f1: 0.9696, person_leak_rate: 0.041, leak_rate: 0.0227, structured_recall: 1 },
    { detector: "pipeline", label: "Regex + spaCy", person_precision: 0.924, person_recall: 0.865, person_f1: 0.893, person_leak_rate: 0.143, leak_rate: 0.079, structured_recall: 1 },
  ],
  rag: { retrieval: "bm25", neighbors: 1, model: "qwen2:7b", n_answerable: 44, n_unanswerable: 10, recall_at_5: 0.818, citation_accuracy: 0.636, answer_term_match: 0.477, abstention_rate: 1, answered_correct_call: 0.82 },
  guard: { passed: true, n_chunks: 5186, structured_leaks: 0, person_leaks: 17, person_support: 1044 },
  models: [{ name: "DistilBERT", dev_f1: 0.945, size_mb: 261.46, train_seconds: 1860, parameters: 65193219 }],
};

test("shows at most five KPIs, the comparison and the index check", async () => {
  mockApi({ "GET /api/health": { body: HEALTH }, "GET /api/metrics": { body: METRICS } });
  const { container } = renderApp("/quality");
  await screen.findByText("Name detection F1");
  const kpis = container.querySelector(".kpis")!;
  expect(within(kpis as HTMLElement).getByText("0.970")).toBeInTheDocument();
  expect(kpis.querySelectorAll(".kpi").length).toBeLessThanOrEqual(5);
  expect(within(kpis as HTMLElement).getByText("2.3%")).toBeInTheDocument();
  expect(within(kpis as HTMLElement).getByText("10 of 10")).toBeInTheDocument();
  expect(screen.getByRole("list", { name: /lower is better/ })).toHaveTextContent("Regex + spaCy7.9%");
  expect(screen.getByText("Passed")).toBeInTheDocument();
  expect(screen.getByText("17 of 1,044")).toBeInTheDocument();
  expect(await axeViolations(container)).toEqual([]);
});

test("missing results show guidance instead of empty charts", async () => {
  mockApi({ "GET /api/health": { body: HEALTH }, "GET /api/metrics": { body: { split: "test", redaction: [], rag: null, guard: null, models: [] } } });
  renderApp("/quality");
  expect(await screen.findByText("No redaction results yet")).toBeInTheDocument();
});
