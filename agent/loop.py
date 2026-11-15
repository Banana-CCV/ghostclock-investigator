"""The investigation loop: Claude decides, tools act, the case board and events update live.

Run states: running -> (awaiting_evidence <-> running) -> reported | stopped.
Everything that happens is appended to runs/<run_id>/events.jsonl (the UI and video replay it).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import anthropic

from .case import ROOT, Board, Case
from .tools import Toolbox

MODEL = "claude-opus-5-5"
RUNS = ROOT / "runs"


def _load_dotenv():
    """Minimal .env loader (repo root): KEY=value lines; never overrides the real environment."""
    import os
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# ---------------- runaway safeguards: per-run turn limit + per-run cost ceiling (no overall budget by default)
PRICE_IN, PRICE_OUT = 4.0 / 1e6, 20.0 / 1e6        # Opus 5.5 $/token (cache reads counted at full price)
LEDGER = RUNS / "_spend.json"


def spend_cap_total() -> float:
    import os
    return float(os.environ.get("GHOSTCLOCK_SPEND_CAP_USD", "1000000"))


def spend_cap_run() -> float:
    import os
    return float(os.environ.get("GHOSTCLOCK_RUN_CAP_USD", "5.0"))


def spent_total() -> float:
    try:
        return float(json.loads(LEDGER.read_text())["usd"])
    except Exception:
        return 0.0


def _add_spend(usd: float):
    RUNS.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps({"usd": round(spent_total() + usd, 4), "updated": time.time()}))


class BudgetExceeded(RuntimeError):
    pass

SYSTEM = """You are GhostClock, an autonomous evidence investigator. Investigators bring you recordings \
(CCTV, phone video, audio) whose timing matters to a decision: an insurance claim, an alibi, a dispute. \
Your job is to establish, independently of any device clock or metadata, WHEN a recording was really made, \
and to tell the investigator what that means for their case.

Your instrument: mains electricity. Britain's grid runs near 50 Hz but wanders by tens of millihertz, and \
the wander never repeats. Recordings made near mains power capture it - as hum in the audio (100/150/200 Hz \
harmonics) or as flicker of the lights, which a rolling-shutter camera records row by row. Your tools extract \
that wander from the evidence and search the national grid record for where it fits.

How to work:
- Keep a case board (update_case_board): the competing explanations (e.g. "device clock is right", \
"clock is off by minutes/hours", "footage is from another day", "no usable grid signal", "edited/spliced"), \
open questions, and your current plan. Revise the plan when evidence changes it - say why.
- Choose each next action because it best separates the live hypotheses, not because it is "the next step". \
Look before you measure (view_frames shows you the scene: light sources, windows, on-screen clocks). \
Pick the stronger signal channel and say why the other is weaker.
- Search sensibly: start around the stated time, widen if nothing locks. A high score alone is not proof: \
challenge your best candidate (rival days, independent halves, sharpness) before you rely on it.
- If the evidence cannot settle the question (too short, signal too weak, rival matches too close), do not \
guess. Use request_evidence to ask the investigator for exactly what would settle it (a longer export, \
another camera on the same recorder, the original file), and explain why.
- Every tool call takes a one-sentence `reason`: what you saw that makes this the right move now.
- Finish with submit_report: plain language for a non-technical investigator, the recovered time, how it \
compares with the claim, how confident you are and why, and what the human should do next. You inform a \
decision; you never make an accusation.
Times are UTC unless stated. In November, UK local time equals UTC."""

R = {"type": "string", "description": "One sentence: what you observed that makes this the right move now."}
TOOLS = [
    {"name": "view_frames", "description": "Look at keyframes from a video evidence item (scene, light sources, "
     "windows, on-screen timestamps, camera type).",
     "input_schema": {"type": "object", "properties": {"evidence_id": {"type": "string"},
                      "count": {"type": "integer", "minimum": 1, "maximum": 4}, "reason": R},
                      "required": ["evidence_id", "reason"]}},
    {"name": "probe_signals", "description": "Measure how much grid signal each channel of an evidence item "
     "carries: audio mains hum and light flicker. Returns strength, usable fraction, whether it behaves like the grid.",
     "input_schema": {"type": "object", "properties": {"evidence_id": {"type": "string"}, "reason": R},
                      "required": ["evidence_id", "reason"]}},
    {"name": "extract_trace", "description": "Extract the grid-frequency trace (1 value per second) from one "
     "channel. channel = 'video' (light flicker) or 'audio' (mains hum). Optionally restrict audio harmonics "
     "(multiples of 50 Hz, e.g. [2,3,4,6]).",
     "input_schema": {"type": "object", "properties": {"evidence_id": {"type": "string"},
                      "channel": {"type": "string", "enum": ["video", "audio"]},
                      "harmonics": {"type": "array", "items": {"type": "integer"}}, "reason": R},
                      "required": ["evidence_id", "channel", "reason"]}},
    {"name": "grid_coverage", "description": "Which national grid frequency records are available (NESO 1 s "
     "archive months; Elexon 15 s, any date on demand).",
     "input_schema": {"type": "object", "properties": {"reason": R}, "required": ["reason"]}},
    {"name": "search_grid", "description": "Slide an extracted trace along the GB grid record between two UTC "
     "times (max 14 days) and return the best-matching clip start times with match scores (1.0 = perfect).",
     "input_schema": {"type": "object", "properties": {"trace_id": {"type": "string"},
                      "start_utc": {"type": "string"}, "end_utc": {"type": "string"},
                      "source": {"type": "string", "enum": ["auto", "NESO", "ELEXON"]}, "reason": R},
                      "required": ["trace_id", "start_utc", "end_utc", "reason"]}},
    {"name": "challenge", "description": "Cross-examine a candidate start time: rival_days (best match on "
     "neighbouring days), split_halves (do both halves independently agree?), sharpness (does the score fall "
     "off a few seconds away?).",
     "input_schema": {"type": "object", "properties": {"trace_id": {"type": "string"},
                      "candidate_start_utc": {"type": "string"},
                      "tests": {"type": "array", "items": {"type": "string",
                                "enum": ["rival_days", "split_halves", "sharpness"]}}, "reason": R},
                      "required": ["trace_id", "candidate_start_utc", "reason"]}},
    {"name": "research_context", "description": "Search the web for context that could change the "
     "interpretation (camera/recorder clock behaviour, local power cuts, events on that date). Returns cited findings.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "reason": R},
                      "required": ["query", "reason"]}},
    {"name": "update_case_board", "description": "Update the case board: hypotheses (each with id, statement, "
     "status: open|supported|weakened|ruled_out, and a short note), open questions, and your current plan.",
     "input_schema": {"type": "object", "properties": {
         "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
             "id": {"type": "string"}, "statement": {"type": "string"},
             "status": {"type": "string", "enum": ["open", "supported", "weakened", "ruled_out"]},
             "note": {"type": "string"}}, "required": ["id", "statement", "status"]}},
         "open_questions": {"type": "array", "items": {"type": "string"}},
         "plan": {"type": "array", "items": {"type": "string"}}, "reason": R}, "required": ["reason"]}},
    {"name": "request_evidence", "description": "Pause and ask the investigator for more evidence that would "
     "settle the question (e.g. a longer export, another camera on the same recorder, the original file).",
     "input_schema": {"type": "object", "properties": {"what": {"type": "string"}, "why": {"type": "string"}},
                      "required": ["what", "why"]}},
    {"name": "submit_report", "description": "Finish the investigation with a report for the investigator.",
     "input_schema": {"type": "object", "properties": {
         "verdict": {"type": "string", "enum": ["CORROBORATED", "DISCREPANCY", "INCONCLUSIVE"]},
         "recovered_start_utc": {"type": "string", "description": "empty if inconclusive"},
         "claimed_start_utc": {"type": "string"},
         "difference": {"type": "string", "description": "plain-language difference, e.g. 'clock 7 min fast'"},
         "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
         "bottom_line": {"type": "string", "description": "the answer to the investigator's actual question in at most 8 words, e.g. 'Car was in its bay at 20:36'"},
         "supports_claimant": {"type": "string", "enum": ["yes", "no", "unclear"], "description": "does the timing evidence support the account of the person who supplied the recording?"},
         "headline": {"type": "string", "description": "one sentence for the investigator"},
         "findings": {"type": "array", "items": {"type": "string"}},
         "what_it_means": {"type": "string"},
         "next_steps": {"type": "array", "items": {"type": "string"}}},
         "required": ["verdict", "confidence", "bottom_line", "supports_claimant", "headline", "findings", "next_steps"]}},
]


def _dump(blocks):
    return [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b for b in blocks]


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class OfflineClient:
    """Zero-cost stand-in for the API: replays the assistant turns recorded in a past run's
    state.json (tools still execute for real). For testing the loop, server and UI."""

    def __init__(self, source_run_id: str):
        st = json.loads((RUNS / source_run_id / "state.json").read_text(encoding="utf-8"))
        self.turns = [m["content"] for m in st["messages"] if m["role"] == "assistant"]
        self.i = 0
        self.beta = _Obj(messages=_Obj(create=self.create))

    def create(self, **_):
        if self.i >= len(self.turns):
            content = [{"type": "text", "text": "(offline replay exhausted)"}]
        else:
            content = self.turns[self.i]
        self.i += 1
        time.sleep(1.0)  # pace like a real call, so the UI animates
        stop = "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn"
        return _Obj(content=content, stop_reason=stop,
                    usage=_Obj(input_tokens=0, output_tokens=0, cache_read_input_tokens=0,
                               cache_creation_input_tokens=0))


class Run:
    def __init__(self, case_id: str, run_id: str | None = None, model: str = MODEL, effort: str = "medium",
                 on_event=None, offline_from: str | None = None):
        self.case = Case.load(case_id)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.dir = RUNS / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model, self.effort = model, effort
        self.on_event = on_event
        self.board = Board()
        self.tools = Toolbox(self.case, self.dir)
        self.messages: list = []
        self.status = "new"
        self.pending_request = None
        self.report = None
        self.seq = 0
        self.usage = {"input": 0, "output": 0, "calls": 0}
        self.offline = bool(offline_from)
        self.client = OfflineClient(offline_from) if offline_from else anthropic.Anthropic()

    # ---------------- events
    def emit(self, type_, **kw):
        self.seq += 1
        ev = {"seq": self.seq, "t": round(time.time(), 3), "type": type_, **kw}
        with open(self.dir / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, default=str) + "\n")
        if self.on_event:
            self.on_event(ev)
        return ev

    def save(self):
        (self.dir / "state.json").write_text(json.dumps({
            "case_id": self.case.case_id, "status": self.status, "messages": self.messages,
            "pending_request": self.pending_request, "board": self.board.snapshot(), "report": self.report,
            "usage": self.usage, "active_evidence": [e["id"] for e in self.case.evidence]}, default=str),
            encoding="utf-8")

    # ---------------- tool dispatch
    def _call_tool(self, name, inp):
        if name == "update_case_board":
            snap = self.board.update(inp.get("hypotheses"), inp.get("open_questions"), inp.get("plan"))
            self.emit("board", board=snap)
            return {"ok": True, "plan_version": snap["plan_version"]}
        fn = getattr(self.tools, name)
        args = {k: v for k, v in inp.items() if k != "reason"}
        return fn(**args)

    def _result_block(self, tu, result):
        if isinstance(result, list):  # rich content (images)
            return {"type": "tool_result", "tool_use_id": tu["id"], "content": result}
        return {"type": "tool_result", "tool_use_id": tu["id"], "content": json.dumps(result, default=str)}

    # ---------------- the loop
    def start(self, auto_supply: bool = False, max_turns: int = 25):
        self.status = "running"
        self.emit("start", case=self.case.public_view(),
                  model="offline-replay (no API)" if self.offline else self.model)
        self.messages = [{"role": "user", "content": "New case.\n" + json.dumps(self.case.public_view(), indent=1)}]
        return self._loop(auto_supply, max_turns)

    def supply(self, evidence_id: str | None, note: str = "", auto_supply=False, max_turns=25):
        """Resume after request_evidence: add an on-request item (or decline) and continue."""
        if evidence_id:
            return self._supply_with(evidence_id, auto_supply, max_turns)
        req = self.pending_request
        other = req.pop("other_results", [])
        content = {"supplied": False, "note": note or "Nothing further is available."}
        self.emit("evidence_declined", note=content["note"])
        self.messages.append({"role": "user", "content": other + [
            {"type": "tool_result", "tool_use_id": req["tool_use_id"], "content": json.dumps(content)}]})
        self.pending_request = None
        self.status = "running"
        return self._loop(auto_supply, max_turns)

    def _cost(self):
        return self.usage["input"] * PRICE_IN + self.usage["output"] * PRICE_OUT

    def _check_budget(self):
        if self.offline:
            return
        if spent_total() >= spend_cap_total():
            raise BudgetExceeded(f"total spend cap ${spend_cap_total():.2f} reached (ledger {LEDGER})")
        if self._cost() >= spend_cap_run():
            raise BudgetExceeded(f"per-run cap ${spend_cap_run():.2f} reached")

    def _loop(self, auto_supply, max_turns):
        nudges = 0
        for _ in range(max_turns):
            try:
                self._check_budget()
            except BudgetExceeded as e:
                self.status = "stopped"
                self.emit("end", status="budget", note=str(e), usage=self.usage)
                self.save()
                return self.status
            try:
                resp = self.client.beta.messages.create(
                    model=self.model, max_tokens=16000,
                    system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    tools=TOOLS, messages=self.messages,
                    thinking={"type": "adaptive", "display": "summarized"},
                    output_config={"effort": self.effort},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            except anthropic.RateLimitError:
                time.sleep(20)
                continue
            except anthropic.APIError as e:  # never retry blindly on other API errors
                self.status = "stopped"
                self.emit("end", status="api_error", note=str(e)[:300], usage=self.usage)
                self.save()
                return self.status
            before = self._cost()
            self.usage["calls"] += 1
            self.usage["input"] += (resp.usage.input_tokens + (resp.usage.cache_read_input_tokens or 0)
                                    + (resp.usage.cache_creation_input_tokens or 0))
            self.usage["output"] += resp.usage.output_tokens
            _add_spend(self._cost() - before)
            content = _dump(resp.content)
            self.messages.append({"role": "assistant", "content": content})
            for b in content:
                if b.get("type") == "thinking" and b.get("thinking"):
                    self.emit("thinking", text=b["thinking"])
                elif b.get("type") == "text" and b.get("text", "").strip():
                    self.emit("say", text=b["text"])
            if resp.stop_reason == "refusal":
                self.status = "stopped"
                self.emit("end", status="refused")
                break
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            if not tool_uses:
                nudges += 1
                if nudges > 1:  # the model stopped without reporting twice: end, don't loop
                    self.status = "stopped"
                    self.emit("end", status="no_report", usage=self.usage)
                    break
                self.messages.append({"role": "user", "content": "Continue the investigation, or submit_report if you are done."})
                continue
            results, paused = [], None
            for tu in tool_uses:
                name, inp = tu["name"], tu["input"]
                self.emit("tool_call", id=tu["id"], name=name, input=inp)
                if name == "request_evidence":
                    paused = tu
                    continue
                if name == "submit_report":
                    self.report = inp
                    self.emit("report", report=inp)
                    results.append(self._result_block(tu, {"ok": True}))
                    continue
                t0 = time.time()
                try:
                    res = self._call_tool(name, inp)
                    err = False
                except Exception as e:  # tool errors go back to the agent
                    res, err = {"error": f"{type(e).__name__}: {e}"}, True
                shown = res if not isinstance(res, list) else {"frames": sum(1 for x in res if x.get("type") == "image"),
                                                               "info": json.loads(res[0]["text"])}
                self.emit("tool_result", id=tu["id"], name=name, result=shown, error=err,
                          seconds=round(time.time() - t0, 1))
                blk = self._result_block(tu, res)
                if err:
                    blk["is_error"] = True
                results.append(blk)
            if self.report:
                self.messages.append({"role": "user", "content": results})
                self.status = "reported"
                self.emit("end", status="reported", usage=self.usage)
                break
            if paused:
                self.pending_request = {"tool_use_id": paused["id"], **paused["input"]}
                if results:  # other tool results must be returned with the pending one later
                    self.pending_request["other_results"] = results
                self.status = "awaiting_evidence"
                self.emit("request", request=paused["input"],
                          available=[{"id": e["id"], "label": e["label"]} for e in self.case.on_request])
                self.save()
                if auto_supply and self.case.on_request:
                    nxt = [e for e in self.case.on_request if e["id"] not in [x["id"] for x in self.case.evidence]]
                    if nxt:
                        return self._supply_with(nxt[0]["id"], auto_supply, max_turns)
                return self.status
            self.messages.append({"role": "user", "content": results})
            self.save()
        else:  # turn limit reached without a report
            self.status = "stopped"
            self.emit("end", status="turn_limit", usage=self.usage)
        self.save()
        return self.status

    def _supply_with(self, eid, auto_supply, max_turns):
        other = self.pending_request.pop("other_results", [])
        item = next(e for e in self.case.on_request if e["id"] == eid)
        self.case.evidence.append(item)
        self.emit("evidence_supplied", evidence={k: v for k, v in item.items() if k not in ("path", "cache")})
        content = {"supplied": True, "new_evidence": {k: v for k, v in item.items() if k not in ("path", "cache", "matches")}}
        self.messages.append({"role": "user", "content": other + [
            {"type": "tool_result", "tool_use_id": self.pending_request["tool_use_id"], "content": json.dumps(content)}]})
        self.pending_request = None
        self.status = "running"
        return self._loop(auto_supply, max_turns)
