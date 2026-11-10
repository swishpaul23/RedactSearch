"""Phase 3: Split redacted transcripts into chunks with call/turn metadata.

One chunk per speaker turn. The text that gets embedded can include the previous turns
(context_turns), so a short answer like "yes, the second one" still carries what it answers.
The text shown to the user is always just the turn itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str           # "<call_id>:<turn_id>"
    call_id: str
    turn_id: int
    speaker: str
    start_sec: float
    split: str
    source: str
    text: str         # the redacted turn, shown in results and citations
    embed_text: str   # what is embedded: previous turns + this turn, each prefixed with the speaker

    def metadata(self) -> dict:
        return {"call_id": self.call_id, "turn_id": self.turn_id, "speaker": self.speaker,
                "start_sec": self.start_sec, "split": self.split, "source": self.source}


def chunk_calls(calls: list[dict], context_turns: int = 1) -> list[Chunk]:
    if context_turns < 0:
        raise ValueError("context_turns must be >= 0")
    out = []
    for c in calls:
        turns = c["turns"]
        for i, t in enumerate(turns):
            if not t["text"].strip():
                continue
            window = turns[max(0, i - context_turns):i + 1]
            embed_text = "\n".join(f"{w['speaker']}: {w['text']}" for w in window)
            out.append(Chunk(f"{c['call_id']}:{t['turn_id']}", c["call_id"], t["turn_id"], t["speaker"],
                             float(t["start_sec"]), c["split"], c["source"], t["text"], embed_text))
    return out
