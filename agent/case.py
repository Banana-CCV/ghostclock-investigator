"""Cases, evidence and the run's case board.

A case manifest (data/cases/<case_id>.json) looks like:
{
  "case_id": "venue_hero",
  "title": "...",
  "brief": "what the investigator needs to know (plain language)",
  "claim": {"description": "...", "claimed_start_local": "2026-11-08 13:42", "timezone": "Europe/London"},
  "evidence": [ {"id": "E1", "label": "CAM 1 ...", "path": "...", "kind": "video",
                 "stated_start_utc": "2026-11-08T13:42:37Z", "source": "claimant upload",
                 "cache": {"video": "path/to/trace.csv", "audio": "..."}} ],
  "on_request": [ { ...evidence item..., "matches": ["other camera", "same recorder"] } ]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "cases"


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q)


@dataclass
class Case:
    case_id: str
    title: str
    brief: str
    claim: dict
    evidence: list
    on_request: list = field(default_factory=list)

    @classmethod
    def load(cls, case_id: str) -> "Case":
        d = json.loads((CASES / f"{case_id}.json").read_text(encoding="utf-8"))
        return cls(d["case_id"], d["title"], d["brief"], d.get("claim", {}), d["evidence"],
                   d.get("on_request", []))

    def item(self, eid: str) -> dict:
        for e in self.evidence:
            if e["id"] == eid:
                return e
        raise KeyError(f"no evidence item {eid!r}; available: {[e['id'] for e in self.evidence]}")

    def public_view(self) -> dict:
        """What the agent is told up front (no file paths, no ground truth)."""
        return {
            "case_id": self.case_id, "title": self.title, "brief": self.brief, "claim": self.claim,
            "evidence": [{k: v for k, v in e.items() if k not in ("path", "cache", "truth", "matches")}
                         for e in self.evidence],
        }


@dataclass
class Board:
    """The case board the agent maintains (rendered live in the UI)."""
    hypotheses: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    plan_version: int = 0

    def update(self, hypotheses=None, open_questions=None, plan=None):
        if hypotheses is not None:
            self.hypotheses = hypotheses
        if open_questions is not None:
            self.open_questions = open_questions
        if plan is not None:
            if plan != self.plan:
                self.plan_version += 1
            self.plan = plan
        return self.snapshot()

    def snapshot(self):
        return {"hypotheses": self.hypotheses, "open_questions": self.open_questions,
                "plan": self.plan, "plan_version": self.plan_version}
