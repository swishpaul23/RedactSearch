"""Tests for the S3 -> Lambda redaction path, offline: a fake S3 client and the regex detector."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from callsearch.aws import handler as h
from callsearch.data.generator import generate_calls
from callsearch.redact.export import ManifestError, load_redacted
from callsearch.redact.pipeline import PipelineDetector
from callsearch.redact.regex import RegexDetector

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


split_calls = load_script("split_calls")
pull_redacted = load_script("pull_redacted")


class FakeS3:
    def __init__(self):
        self.objects = {}  # (bucket, key) -> bytes

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        body = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + 2]  # tiny pages, to exercise pagination
        more = start + 2 < len(keys)
        return {"Contents": [{"Key": k} for k in page], "IsTruncated": more,
                **({"NextContinuationToken": str(start + 2)} if more else {})}


def event(*keys, bucket="raw"):
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": k}}} for k in keys]}


@pytest.fixture
def raw_calls():
    return generate_calls(n=6, seed=11, multiwoz_frac=0, test_frac=0, dev_frac=0)


@pytest.fixture
def s3_with_batches(tmp_path, raw_calls, monkeypatch):
    monkeypatch.setenv("REDACTED_BUCKET", "red")
    paths = split_calls.split(raw_calls, tmp_path / "up", per_file=4)
    s3 = FakeS3()
    for p in paths:
        s3.put_object("raw", f"incoming/{p.name}", p.read_bytes())
    return s3, paths


def test_split_strips_labels_and_batches(tmp_path, raw_calls):
    paths = split_calls.split(raw_calls, tmp_path, per_file=4)
    assert [p.name for p in paths] == ["batch_000.jsonl", "batch_001.jsonl"]
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    assert len(rows) == 6 and all(set(r) == {"call_id", "split", "source", "turns"} for r in rows)
    assert all(set(t) == {"turn_id", "speaker", "start_sec", "text"} for r in rows for t in r["turns"])
    assert split_calls.split(raw_calls, tmp_path, per_file=10)[0].name == "batch_000.jsonl"
    assert len(list(tmp_path.glob("batch_*.jsonl"))) == 1  # old batches removed
    with pytest.raises(ValueError):
        split_calls.split(raw_calls, tmp_path, per_file=0)


def test_handler_redacts_and_writes_manifest(s3_with_batches, raw_calls):
    s3, _ = s3_with_batches
    det = PipelineDetector([RegexDetector()])
    out = h.handler(event("incoming/batch_000.jsonl", "incoming/notes.txt", "other/batch_001.jsonl"), s3=s3, detector=det)
    assert [r["out_key"] for r in out["processed"]] == ["redacted/batch_000.jsonl"]
    assert out["skipped"] == ["incoming/notes.txt", "other/batch_001.jsonl"]
    body = s3.objects[("red", "redacted/batch_000.jsonl")]
    manifest = json.loads(s3.objects[("red", "redacted/batch_000.manifest.json")])
    assert manifest["sha256"] == hashlib.sha256(body).hexdigest() and manifest["n_calls"] == 4
    assert manifest["created_by"] == "callsearch.aws.handler" and manifest["source_key"] == "incoming/batch_000.jsonl"
    text = body.decode()
    structured = [l["value"] for c in raw_calls[:4] for l in c["labels"] if l["type"] != "PERSON"]
    assert structured and not any(v in text for v in structured)  # regex layer masked every structured value


def test_handler_decodes_url_encoded_keys(s3_with_batches):
    s3, paths = s3_with_batches
    s3.put_object("raw", "incoming/my batch.jsonl", paths[0].read_bytes())
    out = h.handler(event("incoming/my+batch.jsonl"), s3=s3, detector=PipelineDetector([RegexDetector()]))
    assert out["processed"][0]["out_key"] == "redacted/my batch.jsonl"


def test_handler_logs_counts_never_text(s3_with_batches, capsys, raw_calls):
    s3, _ = s3_with_batches
    h.handler(event("incoming/batch_000.jsonl"), s3=s3, detector=PipelineDetector([RegexDetector()]))
    logged = capsys.readouterr().out
    assert json.loads(logged.splitlines()[-1])["n_calls"] == 4
    assert not any(t["text"][:20] in logged for c in raw_calls for t in c["turns"] if len(t["text"]) > 20)


def test_parse_rejects_bad_batches():
    with pytest.raises(ValueError, match="missing fields"):
        h.parse_calls(b'{"call_id": "c1", "turns": []}\n', "k")
    with pytest.raises(ValueError, match="no calls"):
        h.parse_calls(b"\n\n", "k")


def test_pull_assembles_a_valid_local_export(s3_with_batches, tmp_path):
    s3, _ = s3_with_batches
    h.handler(event("incoming/batch_000.jsonl", "incoming/batch_001.jsonl"), s3=s3,
              detector=PipelineDetector([RegexDetector()], name="pipeline"))
    calls, detectors = pull_redacted.fetch(s3, "red")
    assert len(calls) == 6 and detectors == {"pipeline"}
    assert [c["call_id"] for c in calls] == sorted(c["call_id"] for c in calls)
    from callsearch.redact.export import write_redacted

    write_redacted(calls, tmp_path / "out", "pipeline")
    loaded, manifest = load_redacted(tmp_path / "out")  # what build_index.py reads
    assert loaded == calls and manifest["detector"] == "pipeline"


def test_pull_refuses_tampered_or_unvouched_batches(s3_with_batches):
    s3, _ = s3_with_batches
    det = PipelineDetector([RegexDetector()])
    h.handler(event("incoming/batch_000.jsonl", "incoming/batch_001.jsonl"), s3=s3, detector=det)
    good = dict(s3.objects)

    s3.objects[("red", "redacted/batch_001.jsonl")] += b"\n"
    with pytest.raises(ManifestError, match="sha256"):
        pull_redacted.fetch(s3, "red")

    s3.objects = dict(good)
    del s3.objects[("red", "redacted/batch_001.manifest.json")]
    with pytest.raises(ManifestError, match="no manifest"):
        pull_redacted.fetch(s3, "red")

    s3.objects = dict(good)
    s3.objects[("red", "redacted/copy.jsonl")] = good[("red", "redacted/batch_000.jsonl")]
    s3.objects[("red", "redacted/copy.manifest.json")] = good[("red", "redacted/batch_000.manifest.json")]
    with pytest.raises(ManifestError, match="more than one batch"):
        pull_redacted.fetch(s3, "red")

    with pytest.raises(ManifestError, match="no redacted batches"):
        pull_redacted.fetch(FakeS3(), "red")


def test_dockerignore_keeps_data_and_secrets_out():
    rules = (ROOT / ".dockerignore").read_text().split()
    assert rules[rules.index("*")] == "*"  # deny everything, then allow-list
    allowed = {r[1:] for r in rules if r.startswith("!")}
    assert allowed == {"pyproject.toml", "src/", "models/ner_distilbert/"}
