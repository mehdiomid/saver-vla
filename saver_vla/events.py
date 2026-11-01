"""Append-only JSONL event log: one file per episode.

Every record has a `kind` (FAULT, VERIFIER_EVENT, FAILURE, DECISION, SKILL_OUTCOME,
POLICY_TRIAL_OUTCOME, ...). The same file feeds metrics, reports and the replay agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class EventLog:
    def __init__(self, path: str | Path | None = None, echo: bool = False):
        self.path = Path(path) if path else None
        self.echo = echo
        self.records: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        record = {"kind": kind, **fields}
        # Round-trip so stored records are plain JSON types.
        record = json.loads(json.dumps(record, default=_default))
        self.records.append(record)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if self.echo:
            summary = record.get("executed_skill") or record.get("event_type") or record.get("phase") or ""
            print(f"[SAVER] {kind} {summary}".rstrip(), flush=True)
        return record

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["kind"] == kind]


def read_events(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
