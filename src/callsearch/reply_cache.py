"""Model replies cached on disk, shared by the redactor, the answerer and the agent.

A reply lives at <directory>/<key[:2]>/<key>.json as {"response": ...}, where the key is the
sha256 of its parts joined by NUL. Callers put the provider, model and prompt version in the
key, so changing any of them misses the cache instead of reusing a stale reply.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def reply_key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


class ReplyCache:
    """A directory of cached replies; with no directory, every lookup misses and nothing is saved."""

    def __init__(self, directory: Path | None):
        self.directory = directory

    def _path(self, key: str) -> Path | None:
        return None if self.directory is None else self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if path is not None and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        return None

    def put(self, key: str, response: str) -> None:
        path = self._path(key)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"response": response}), encoding="utf-8")
