"""GhostClock web server: start investigations, stream their events live, serve run artefacts.

    python -m uvicorn server.app:app --port 8765
GET  /api/cases                      list case manifests
POST /api/runs {case_id, auto_supply} start a run (background thread) -> {run_id}
GET  /api/runs                       list past runs
GET  /api/runs/{id}/events?after=N   Server-Sent Events: replays then tails events.jsonl
POST /api/runs/{id}/supply {evidence_id|null, note}  answer a request_evidence pause
GET  /runs/{id}/{file}               run artefacts (frames, traces, searches, overlays)
GET  /                               the investigator UI (web/index.html)
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.case import CASES, ROOT
from agent.loop import RUNS, Run

app = FastAPI(title="GhostClock")
WEB = ROOT / "web"
LIVE: dict[str, Run] = {}


class StartReq(BaseModel):
    case_id: str
    auto_supply: bool = False
    effort: str = "medium"
    offline_from: str | None = None   # zero-cost test mode


class SupplyReq(BaseModel):
    evidence_id: str | None = None
    note: str = ""


@app.get("/api/cases")
def cases():
    out = []
    for p in sorted(CASES.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({"case_id": d["case_id"], "title": d["title"], "brief": d["brief"],
                    "evidence": [e["label"] for e in d["evidence"]]})
    return out


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(RUNS.glob("*"), reverse=True):
        s = d / "state.json"
        if s.exists():
            j = json.loads(s.read_text(encoding="utf-8"))
            out.append({"run_id": d.name, "case_id": j["case_id"], "status": j["status"],
                        "verdict": (j.get("report") or {}).get("verdict")})
    return out


@app.get("/api/budget")
def budget():
    from agent.loop import spend_cap_run, spend_cap_total, spent_total
    return {"spent_usd": spent_total(), "cap_total_usd": spend_cap_total(), "cap_per_run_usd": spend_cap_run()}


@app.post("/api/runs")
def start(req: StartReq):
    from agent.loop import spend_cap_total, spent_total
    if any(r.status == "running" for r in LIVE.values()):
        raise HTTPException(409, "a run is already in progress")
    if not req.offline_from and spent_total() >= spend_cap_total():
        raise HTTPException(402, "spend cap reached; raise GHOSTCLOCK_SPEND_CAP_USD to continue")
    run = Run(req.case_id, effort=req.effort, offline_from=req.offline_from)
    LIVE[run.run_id] = run
    threading.Thread(target=run.start, kwargs={"auto_supply": req.auto_supply}, daemon=True).start()
    return {"run_id": run.run_id}


@app.post("/api/runs/{run_id}/supply")
def supply(run_id: str, req: SupplyReq):
    run = LIVE.get(run_id)
    if not run or run.status != "awaiting_evidence":
        raise HTTPException(409, "run is not waiting for evidence")
    threading.Thread(target=run.supply, args=(req.evidence_id, req.note), daemon=True).start()
    return {"ok": True}


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str, after: int = 0, speed: float = 0.0):
    """speed=0: live tail. speed>0: replay a finished run at that time multiple (for the video)."""
    path = RUNS / run_id / "events.jsonl"

    async def gen():
        sent, last_t = after, None
        idle = 0
        while True:
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            new = [json.loads(x) for x in lines if x.strip()][sent:]
            for ev in new:
                if speed > 0 and last_t is not None:
                    await asyncio.sleep(min(4.0, max(0.0, (ev["t"] - last_t) / speed)))
                last_t = ev["t"]
                sent += 1
                yield f"data: {json.dumps(ev)}\n\n"
                if ev["type"] == "request" and speed > 0:
                    await asyncio.sleep(3.0)       # let a replay viewer read the agent's request
                if ev["type"] == "end":
                    return
            idle = 0 if new else idle + 1
            live = run_id in LIVE and LIVE[run_id].status == "running"
            if idle > (1800 if live else 6):  # finished/paused runs: stop once replayed
                yield "event: done\ndata: {}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/runs/{run_id}/{name}")
def artefact(run_id: str, name: str):
    p = (RUNS / run_id / name).resolve()
    if RUNS.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


MEDIA = ROOT / "data" / "media"


@app.get("/media/{name}")
def media(name: str):
    p = (MEDIA / name).resolve()
    if MEDIA.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
