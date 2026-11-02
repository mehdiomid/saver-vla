"""Recovery agents. Every agent answers one question: which skill next?

An agent receives a JSON-serializable observation and returns a decision
{"skill", "variant", "reason", "expected_outcome"}. It never commands motors.
The coordinator validates every decision before anything moves.

Backends
  ScriptedAgent  fixed pregrasp -> approach -> resume sequence (no API calls)
  LLMAgent       OpenAI structured-output agent (the notebook's GPT-5.x agent)
  ReplayAgent    replays decisions from a previous event log (exact reproduction)
  CallableAgent  wraps any Python function (tests, custom planners)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..skills import ACTION_MENU, VARIANTS, valid_variant
from .prompts import build_instructions


class Agent:
    """Base class. Subclasses implement `_decide`."""

    name = "agent"

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.calls = 0
        self.latency_s = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self) -> None:
        self.history = []

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        decision = self._decide(observation)
        self.latency_s += time.perf_counter() - start
        self.calls += 1
        decision.setdefault("variant", "CENTER")
        decision.setdefault("reason", "")
        decision.setdefault("expected_outcome", "")
        self.history.append({"observation": observation, "decision": decision})
        return decision

    def record_result(self, result: dict[str, Any]) -> None:
        if self.history:
            self.history[-1]["execution_result"] = result

    def stats(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "calls": self.calls,
            "latency_s": round(self.latency_s, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    def _decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class ScriptedAgent(Agent):
    """The Phase 5 fixed-sequence baseline: pregrasp, approach, resume; else abort."""

    name = "scripted"

    def _decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        previous = observation.get("last_skill")
        result = observation.get("last_skill_result")
        if previous is None:
            skill = "MOVE_TO_PREGRASP"
        elif previous == "MOVE_TO_PREGRASP":
            skill = "APPROACH_OBJECT" if result == "REACHED" else "ABORT_RECOVERY"
        elif previous == "APPROACH_OBJECT" and result == "REACHED":
            skill = "RESUME_POLICY"
        else:
            skill = "ABORT_RECOVERY"
        return {
            "skill": skill,
            "variant": "CENTER",
            "reason": "Fixed scripted recovery sequence",
            "expected_outcome": "Execute next predefined stage",
        }


class CallableAgent(Agent):
    name = "callable"

    def __init__(self, fn: Callable[[dict[str, Any]], dict[str, Any]], name: str = "callable"):
        super().__init__()
        self.fn = fn
        self.name = name

    def _decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        return dict(self.fn(observation))


class ReplayAgent(Agent):
    """Returns the agent's decisions from a previous run, in order.

    Use it to reproduce an episode exactly (no LLM randomness) or to check that
    a code change does not alter how a recorded decision sequence is executed.
    """

    name = "replay"

    def __init__(self, decisions: list[dict[str, Any]]):
        super().__init__()
        self._queue = list(decisions)

    @classmethod
    def from_log(cls, path: str | Path) -> "ReplayAgent":
        from ..events import read_events

        records = read_events(path)
        decisions = [r["decision"] for r in records if r.get("kind") == "AGENT_CALL"]
        if not decisions:
            # Notebook-era `agent_events.jsonl` (Phases 4-6.5.1): one DECISION per agent call
            # when no request was rejected. Rejected requests are listed in SKILL_DISPATCH.
            for r in records:
                if r.get("kind") == "DECISION" and isinstance(r.get("decision"), dict):
                    initial = r.get("initial_requested_decision")
                    if initial and initial != r["decision"]:
                        decisions.append(dict(initial))
                    decisions.append(dict(r["decision"]))
        return cls(decisions)

    def _decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self._queue:
            return {"skill": "ABORT_RECOVERY", "variant": "CENTER", "reason": "Replay log exhausted"}
        return dict(self._queue.pop(0))


class LLMAgent(Agent):
    """GPT-based recovery agent with a strict JSON schema (OpenAI Responses API)."""

    name = "llm"

    def __init__(
        self,
        model: str = "gpt-5.5",
        skills: list[str] | None = None,
        blind: bool = True,
        capability_aware: bool = True,
        history_length: int = 7,
        timeout_s: float = 90.0,
        max_retries: int = 4,
        client: Any = None,
    ):
        super().__init__()
        menu_skills = list(dict.fromkeys(name for name, _ in ACTION_MENU))
        self.skills = [s for s in menu_skills if skills is None or s in skills]
        self.model = model
        self.history_length = history_length
        self.instructions = build_instructions(blind, capability_aware, self.skills)
        self.schema = {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "enum": self.skills},
                "variant": {"type": "string", "enum": list(VARIANTS)},
                "reason": {"type": "string"},
                "expected_outcome": {"type": "string"},
            },
            "required": ["skill", "variant", "reason", "expected_outcome"],
            "additionalProperties": False,
        }
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError("LLMAgent needs `pip install saver-vla[llm]` (openai).") from exc
            client = OpenAI(timeout=timeout_s, max_retries=max_retries)
        self.client = client
        self.name = f"llm:{model}"

    def _decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "observation": observation,
            "recent_history": self.history[-self.history_length :],
            "available_skills": observation.get("available_actions", self.skills),
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=json.dumps(payload),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "saver_recovery_decision",
                    "strict": True,
                    "schema": self.schema,
                }
            },
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        decision = json.loads(response.output_text)
        if decision["skill"] not in self.skills or decision["variant"] not in VARIANTS:
            raise ValueError("Invalid agent action")
        if not valid_variant(decision["skill"], decision["variant"]):
            raise ValueError("Variant not permitted for selected skill")
        return decision


def make_agent(recovery_cfg: Any) -> Agent | None:
    kind = recovery_cfg.agent
    if not recovery_cfg.enabled or kind in ("none", "queue_reset"):
        return None
    if kind == "scripted":
        return ScriptedAgent()
    if kind == "llm":
        return LLMAgent(
            model=recovery_cfg.model,
            skills=recovery_cfg.skills,
            blind=recovery_cfg.blind,
            capability_aware=recovery_cfg.capability_aware,
            history_length=recovery_cfg.history_length,
            timeout_s=recovery_cfg.llm_timeout_s,
            max_retries=recovery_cfg.llm_max_retries,
        )
    if kind == "replay":
        if not recovery_cfg.replay_log:
            raise ValueError("recovery.agent=replay needs recovery.replay_log")
        return ReplayAgent.from_log(recovery_cfg.replay_log)
    raise ValueError(f"Unknown agent: {kind}")


__all__ = ["Agent", "CallableAgent", "LLMAgent", "ReplayAgent", "ScriptedAgent", "make_agent"]
