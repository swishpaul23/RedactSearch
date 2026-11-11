"""Read uploaded transcript files into raw calls, before redaction.

Three formats, one call or many per file:

.jsonl  one call per line:  {"turns": [{"speaker": "agent", "text": "...", "start_sec": 0}, ...]}
.json   one call object, a list of them, or {"calls": [...]}
.txt    one call, one turn per line:  "Agent: ..." / "Customer: ..."  (optional "[01:23]" first)

Speakers may be agent/rep/representative/advisor/operator or customer/caller/client/member.
start_sec is optional; missing times are estimated from the words spoken before.

Problems are reported by location only ("Line 4", "Call 2, turn 5"), never by quoting the
text: the input is unredacted, so nothing from it may appear in a response or a log.
A call with any problem is rejected whole, so a transcript is never imported with holes in it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import PurePath

MAX_FILE_BYTES = 2_000_000
MAX_FILES = 20
MAX_CALLS = 200          # per upload, across all files
MAX_TURNS = 400          # per call
MAX_TURN_CHARS = 4000
WORDS_PER_SECOND = 2.5   # for estimating missing start times
EXTENSIONS = (".jsonl", ".json", ".txt")

SPEAKERS = {
    **dict.fromkeys(("agent", "rep", "representative", "advisor", "operator", "support"), "agent"),
    **dict.fromkeys(("customer", "caller", "client", "member", "user"), "customer"),
}
TXT_TURN = re.compile(r"^\s*(?:\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s+)?(?P<who>[A-Za-z]{2,20})\s*:\s?(?P<text>.*)$")


@dataclass
class Issue:
    where: str | None
    message: str


@dataclass
class ParsedFile:
    name: str
    calls: list[list[dict]] = field(default_factory=list)  # each call: [{speaker, text, start_sec}]
    issues: list[Issue] = field(default_factory=list)

    @property
    def n_turns(self) -> int:
        return sum(len(c) for c in self.calls)


def safe_name(name: str | None) -> str:
    base = PurePath((name or "upload").replace("\\", "/")).name or "upload"
    return base[:120]


def _clock(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return float(sec)


def _fill_times(turns: list[dict]) -> list[dict]:
    t = 0.0
    for turn in turns:
        if turn.get("start_sec") is None:
            turn["start_sec"] = round(t, 1)
        t = max(t, float(turn["start_sec"])) + len(turn["text"].split()) / WORDS_PER_SECOND
    return turns


def _check_turns(turns: list[dict], where: str) -> list[Issue]:
    if not turns:
        return [Issue(where, "No turns found.")]
    if len(turns) > MAX_TURNS:
        return [Issue(where, f"More than {MAX_TURNS} turns. Split the call into smaller files.")]
    return [Issue(where, f"Turn {i} is longer than {MAX_TURN_CHARS} characters.")
            for i, t in enumerate(turns, 1) if len(t["text"]) > MAX_TURN_CHARS][:5]


def _turns_from_objects(raw, where: str) -> tuple[list[dict], list[Issue]]:
    if isinstance(raw, dict):
        raw = raw.get("turns")
    if not isinstance(raw, list):
        return [], [Issue(where, 'Expected an object with a "turns" list.')]
    turns, issues = [], []
    for i, t in enumerate(raw, 1):
        at = f"{where}, turn {i}"
        if not isinstance(t, dict):
            issues.append(Issue(at, "Each turn must be an object with speaker and text."))
            continue
        who = SPEAKERS.get(str(t.get("speaker", t.get("role", ""))).strip().lower())
        text = t.get("text")
        start = t.get("start_sec")
        if who is None:
            issues.append(Issue(at, "Speaker must be agent or customer."))
        if not isinstance(text, str) or not text.strip():
            issues.append(Issue(at, "Text is missing or empty."))
        if start is not None and (isinstance(start, bool) or not isinstance(start, (int, float)) or start < 0):
            issues.append(Issue(at, "start_sec must be a number of seconds, 0 or more."))
            start = None
        if who and isinstance(text, str) and text.strip():
            turns.append({"speaker": who, "text": text.strip(), "start_sec": start})
    return turns, issues[:10]


def _parse_jsonl(text: str, out: ParsedFile) -> None:
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        where = f"Line {n}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            out.issues.append(Issue(where, "Not valid JSON."))
            continue
        _add_call(obj, where, out)


def _parse_json(text: str, out: ParsedFile) -> None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        out.issues.append(Issue(f"Line {e.lineno}", "Not valid JSON."))
        return
    if isinstance(obj, dict) and isinstance(obj.get("calls"), list):
        obj = obj["calls"]
    if isinstance(obj, list) and obj and all(isinstance(c, dict) and "turns" in c for c in obj):
        for i, c in enumerate(obj, 1):
            _add_call(c, f"Call {i}", out)
    else:
        _add_call(obj, "Call 1", out)


def _parse_txt(text: str, out: ParsedFile) -> None:
    turns, issues = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = TXT_TURN.match(line)
        if m:
            who = SPEAKERS.get(m["who"].lower())
            if who is None:
                issues.append(Issue(f"Line {n}", "Start the line with Agent: or Customer:."))
                continue
            turns.append({"speaker": who, "text": m["text"].strip(),
                          "start_sec": _clock(m["ts"]) if m["ts"] else None})
        elif turns:
            turns[-1]["text"] = f"{turns[-1]['text']} {line.strip()}".strip()
        else:
            issues.append(Issue(f"Line {n}", "Start the line with Agent: or Customer:."))
    for i, t in enumerate(turns, 1):
        if not t["text"]:
            issues.append(Issue(f"Turn {i}", "Text is missing or empty."))
    issues += [] if issues else _check_turns(turns, "File")
    if issues:
        out.issues += issues[:10]
    else:
        out.calls.append(_fill_times(turns))


def _add_call(obj, where: str, out: ParsedFile) -> None:
    turns, issues = _turns_from_objects(obj, where)
    issues = issues or _check_turns(turns, where)
    if issues:
        out.issues += issues
    else:
        out.calls.append(_fill_times(turns))


def parse_file(name: str | None, data: bytes) -> ParsedFile:
    out = ParsedFile(safe_name(name))
    ext = PurePath(out.name).suffix.lower()
    if ext not in EXTENSIONS:
        out.issues.append(Issue(None, "Use a .jsonl, .json or .txt file."))
        return out
    if len(data) > MAX_FILE_BYTES:
        out.issues.append(Issue(None, f"File is larger than {MAX_FILE_BYTES // 1_000_000} MB."))
        return out
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        out.issues.append(Issue(None, "File is not UTF-8 text."))
        return out
    if not text.strip():
        out.issues.append(Issue(None, "File is empty."))
        return out
    {".jsonl": _parse_jsonl, ".json": _parse_json, ".txt": _parse_txt}[ext](text, out)
    if not out.calls and not out.issues:
        out.issues.append(Issue(None, "No calls found."))
    return out
