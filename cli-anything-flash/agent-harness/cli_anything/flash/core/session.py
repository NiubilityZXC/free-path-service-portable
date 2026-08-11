"""Small JSON session with undo/redo for agent workflows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_session_path() -> Path:
    """Return the per-user session path."""

    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "cli-anything-flash" / "session.json"


@dataclass
class Session:
    """Persist current FLASH root/project and command history."""

    path: Path = field(default_factory=default_session_path)
    state: dict[str, Any] = field(default_factory=lambda: {"history": [], "undo": [], "redo": []})

    def load(self) -> "Session":
        """Load session JSON if present."""

        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        return self

    def save(self) -> None:
        """Save session JSON atomically."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def snapshot(self, label: str, data: dict[str, Any]) -> None:
        """Push a snapshot into the undo stack."""

        self.state.setdefault("undo", []).append({"label": label, "data": data})
        self.state["redo"] = []
        self.save()

    def record(self, command: str, result: dict[str, Any]) -> None:
        """Record one command result."""

        self.state.setdefault("history", []).append({"command": command, "result": result})
        self.state["history"] = self.state["history"][-100:]
        self.save()

    def undo(self) -> dict[str, Any]:
        """Move the last snapshot to redo and return it."""

        undo_stack = self.state.setdefault("undo", [])
        if not undo_stack:
            raise RuntimeError("Nothing to undo")
        item = undo_stack.pop()
        self.state.setdefault("redo", []).append(item)
        self.save()
        return item

    def redo(self) -> dict[str, Any]:
        """Move the last redo snapshot back to undo and return it."""

        redo_stack = self.state.setdefault("redo", [])
        if not redo_stack:
            raise RuntimeError("Nothing to redo")
        item = redo_stack.pop()
        self.state.setdefault("undo", []).append(item)
        self.save()
        return item
