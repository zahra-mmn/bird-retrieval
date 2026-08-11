"""On-disk cache for LLM calls, keyed by a hash of the prompt — the scope doc explicitly flags
"don't recall the API on every UI iteration" as a cost risk; this makes that the default
behavior rather than something to remember to do.
"""

import hashlib
import json
from pathlib import Path


class DiskCache:
    def __init__(self, cache_dir="./.cache/llm"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def set(self, key: str, value) -> None:
        self._path(key).write_text(json.dumps(value), encoding="utf-8")
