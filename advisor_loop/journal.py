import json, os
from datetime import datetime, timezone
from pathlib import Path

class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: set[str] = set()
        self._events: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._events.append(ev)
                if ev.get("task_key"):
                    self._keys.add(ev["task_key"])

    def append(self, event: dict) -> None:
        event = {**event, "ts": datetime.now(timezone.utc).isoformat()}
        with open(self.path, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._events.append(event)
        if event.get("task_key"):
            self._keys.add(event["task_key"])

    def has(self, task_key: str) -> bool:
        return task_key in self._keys

    def events(self, kind: str | None = None) -> list[dict]:
        return [e for e in self._events if kind is None or e.get("kind") == kind]
