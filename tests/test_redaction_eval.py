"""Tests for redaction_eval metrics on a tiny hand-made fixture, and the results file format."""

import json
import re
import sys
from pathlib import Path

import pytest

from callsearch.data.generator import write_jsonl
from callsearch.eval import redaction_eval
from callsearch.eval.redaction_eval import evaluate, save_results
from callsearch.redact.base import Span

TEXT = "Card 4539 1488 0343 6467, call 604-555-0182, order 1111 2222 3333 4444, I'm Ann Lee."


def span(sub, typ):
    i = TEXT.index(sub)
    return {"turn_id": 0, "start": i, "end": i + len(sub), "type": typ, "value": sub}


CALLS = [{
    "call_id": "c1", "split": "test",
    "turns": [{"turn_id": 0, "speaker": "customer", "start_sec": 0.0, "text": TEXT}],
    "labels": [span("4539 1488 0343 6467", "CARD"), span("604-555-0182", "PHONE"), span("Ann Lee", "PERSON")],
    "distractors": [span("1111 2222 3333 4444", "ORDER_NUMBER")],
}]


class FakeDetector:
    """CARD exact, PHONE partial (overlap but not exact), one FP on the order number, PERSON missed."""
    name = "fake"

    def detect(self, text):
        c, p, o = (TEXT.index(s) for s in ("4539", "604", "1111"))
        return [Span(c, c + 19, "CARD", "fake"), Span(p, p + 7, "PHONE", "fake"), Span(o, o + 19, "CARD", "fake")]


def test_metrics_on_fixture():
    res = evaluate(CALLS, FakeDetector())
    card, phone, person = (res["per_type"][t] for t in ("CARD", "PHONE", "PERSON"))

    assert (card["tp"], card["fp"], card["fn"]) == (1, 1, 0)
    assert card["precision"] == 0.5 and card["recall"] == 1.0 and card["exact_recall"] == 1.0
    assert (phone["tp"], phone["fp"], phone["fn"]) == (1, 0, 0)
    assert phone["recall"] == 1.0 and phone["exact_recall"] == 0.0  # overlap counts, exact does not
    assert person["recall"] == 0.0 and person["support"] == 1

    m = res["micro"]
    assert (m["tp"], m["fp"], m["fn"], m["support"]) == (2, 1, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)


def test_leaks_and_distractors():
    res = evaluate(CALLS, FakeDetector())
    assert res["per_type"]["CARD"]["leaked"] == 0
    assert res["per_type"]["PHONE"]["leaked"] == 0  # partial redaction still breaks the verbatim value
    assert res["per_type"]["PERSON"]["leaked"] == 1
    assert res["distractor_fp"] == {"ORDER_NUMBER": {"flagged": 1, "total": 1}}


def test_one_to_one_matching():
    calls = [{**CALLS[0], "labels": [span("4539 1488 0343 6467", "CARD")], "distractors": []}]

    class Twice:
        name = "twice"

        def detect(self, text):
            c = TEXT.index("4539")
            return [Span(c, c + 9, "CARD", "t"), Span(c + 10, c + 19, "CARD", "t")]

    card = evaluate(calls, Twice())["per_type"]["CARD"]
    assert (card["tp"], card["fp"], card["fn"]) == (1, 1, 0)


# --- results files: metrics only, never transcript text ----------------------

METRIC_KEYS = {"precision", "recall", "f1", "exact_recall", "leak_rate", "tp", "fp", "fn", "support", "leaked"}
TOP_KEYS = {"detector", "split", "n_calls", "per_type", "micro", "micro_excl_person", "distractor_fp", "llm"}
TYPE_NAME = re.compile(r"^[A-Z_]{2,20}$")
NAME = re.compile(r"^[a-z_]{1,20}$")


def assert_metrics_only(res: dict) -> None:
    assert set(res) <= TOP_KEYS and {"detector", "per_type", "micro"} <= set(res)
    assert NAME.match(res["detector"])
    if "split" in res:
        assert res["split"] in {"train", "dev", "test", "all"}
    assert type(res["n_calls"]) is int
    if "llm" in res:  # provenance for LLM detectors
        assert set(res["llm"]) == {"provider", "model", "prompt_version"}
        assert all(re.match(r"^[\w.:\-]{1,40}$", v) for v in res["llm"].values())

    def metrics(m: dict) -> None:
        assert set(m) == METRIC_KEYS
        assert all(type(v) in (int, float) for v in m.values())

    for group in ("micro", "micro_excl_person"):
        metrics(res[group])
    for t, m in res["per_type"].items():
        assert TYPE_NAME.match(t)
        metrics(m)
    for t, d in res.get("distractor_fp", {}).items():
        assert TYPE_NAME.match(t)
        assert set(d) == {"flagged", "total"} and all(type(v) is int for v in d.values())


def test_saved_results_are_metrics_only(tmp_path):
    res = evaluate(CALLS, FakeDetector())
    res["split"] = "test"
    path = tmp_path / "eval.json"
    save_results(res, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert_metrics_only(loaded)
    assert TEXT not in path.read_text(encoding="utf-8")
    for label in CALLS[0]["labels"]:
        assert label["value"] not in path.read_text(encoding="utf-8")


RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.parametrize("path", sorted(RESULTS.rglob("eval_*.json")), ids=lambda p: p.name)
def test_committed_results_are_metrics_only(path):
    assert_metrics_only(json.loads(path.read_text(encoding="utf-8")))


TRAIN_KEYS = {"base", "n_train_examples", "n_dev_examples", "n_train_calls", "n_dev_calls", "n_person_labels",
              "failed_alignments", "epochs_run", "best_epoch", "best_dev_f1", "train_seconds", "model_size_mb",
              "history"}
EPOCH_KEYS = {"epoch", "loss", "dev_precision", "dev_recall", "dev_f1"}


@pytest.mark.parametrize("path", sorted(RESULTS.rglob("train_*.json")), ids=lambda p: p.name)
def test_training_summaries_are_metrics_only(path):
    res = json.loads(path.read_text(encoding="utf-8"))
    assert set(res) == TRAIN_KEYS
    assert res["base"] in {"sm", "blank"}
    assert all(type(v) in (int, float) for k, v in res.items() if k not in {"base", "history"})
    for row in res["history"]:
        assert set(row) == EPOCH_KEYS and all(type(v) in (int, float) for v in row.values())


def test_results_dir_has_only_known_files():
    """results/ (including archived subfolders like v1/) holds only metric files."""
    files = [p for p in RESULTS.rglob("*") if p.is_file()]
    ok = lambda p: (p.suffix == ".json" and p.name.startswith(("eval_", "train_", "presidio_extras_",
                                                               "llm_additions_"))) or (
        p.suffix == ".md" and p.name.startswith("comparison_"))
    assert all(ok(p) for p in files), files


@pytest.mark.parametrize("path", sorted(RESULTS.rglob("presidio_extras_*.json")), ids=lambda p: p.name)
def test_presidio_extras_are_metrics_only(path):
    res = json.loads(path.read_text(encoding="utf-8"))
    assert set(res) == {"split", "detector", "model", "distractors_flagged", "distractor_hits_by_entity",
                        "labels_touched_only_by_unmapped"}
    assert res["detector"] == "presidio" and re.match(r"^en_core_web_(sm|md|lg|trf)-[\d.]+$", res["model"])
    for t, d in res["distractors_flagged"].items():
        assert TYPE_NAME.match(t) and all(type(v) is int for v in d.values())
    for key in ("distractor_hits_by_entity", "labels_touched_only_by_unmapped"):
        for k, v in res[key].items():
            assert re.match(r"^[A-Z_]+(:[A-Z_]+)?$", k) and type(v) is int


@pytest.mark.parametrize("path", sorted(RESULTS.rglob("llm_additions_*.json")), ids=lambda p: p.name)
def test_llm_additions_are_metrics_only(path):
    res = json.loads(path.read_text(encoding="utf-8"))
    kinds = {"correct", "overlaps_label", "distractor", "false"}
    assert set(res) == {"split", "detector", "added", "by_type", "provider", "model", "prompt_version", "requests",
                        "cache_hits", "parse_failures"} | kinds
    assert res["provider"] in {"anthropic", "ollama"} and re.match(r"^[\w.:\-]{1,40}$", res["model"])
    assert all(type(res[k]) is int for k in kinds | {"added", "requests", "cache_hits", "parse_failures"})
    for t, d in res["by_type"].items():
        assert TYPE_NAME.match(t) and set(d) == kinds and all(type(v) is int for v in d.values())


@pytest.mark.parametrize("path", sorted(RESULTS.rglob("comparison_*.md")), ids=lambda p: p.name)
def test_comparison_tables_are_metrics_only(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith(("| detector", "| type", "|---")):
            cells = [c.strip() for c in line.strip("|").split("|")]
            assert re.match(r"^[a-z_]+( \[[\w.:\-]+ prompt v\d+\]| \[model unknown\])?$|^[A-Z]+$|^overall$",
                            cells[0]), line
            assert all(re.match(r"^[A-Z]+$|^overall$|^[\d.]+%?$", c) for c in cells[1:]), line


def _run_cli(monkeypatch, tmp_path, *extra):
    data = tmp_path / "calls.jsonl"
    write_jsonl(CALLS, data)
    monkeypatch.setattr(redaction_eval, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["redaction_eval", "--split", "test", "--data", str(data), *extra])
    redaction_eval.main()


def test_cli_saves_to_default_path(monkeypatch, tmp_path):
    _run_cli(monkeypatch, tmp_path)
    out = tmp_path / "results" / "eval_regex_test.json"
    assert_metrics_only(json.loads(out.read_text(encoding="utf-8")))


def test_cli_json_override_and_no_save(monkeypatch, tmp_path):
    _run_cli(monkeypatch, tmp_path, "--json", str(tmp_path / "custom.json"))
    assert (tmp_path / "custom.json").exists() and not (tmp_path / "results").exists()
    (tmp_path / "custom.json").unlink()
    _run_cli(monkeypatch, tmp_path, "--no-save")
    assert not (tmp_path / "custom.json").exists() and not (tmp_path / "results").exists()
