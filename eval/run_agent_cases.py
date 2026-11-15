"""Run the live agent on a list of cases and score it against each case's hidden truth.

    python -m eval.run_agent_cases eval_8min_a eval_8min_b ...     (uses the Claude API)
Writes eval/results/agent_cases.json. 'correct' = reported start within 5 s of the truth.
A run that declines (INCONCLUSIVE or asks for evidence) counts as declined, never as wrong.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from agent.loop import Run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/results/agent_cases.json"


def main(ids):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for cid in ids:
        truth = json.loads((ROOT / f"data/cases/{cid}.json").read_text(encoding="utf-8")).get("truth", {})
        run = Run(cid)
        status = run.start(auto_supply=False)
        rep = run.report or {}
        rec = rep.get("recovered_start_utc") or ""
        err = None
        if rec and truth.get("true_start_utc"):
            try:
                err = float((pd.Timestamp(rec) - pd.Timestamp(truth["true_start_utc"])).total_seconds())
            except Exception:
                err = None
        outcome = ("declined" if status != "reported" or rep.get("verdict") == "INCONCLUSIVE" or not rec
                   else "correct" if err is not None and abs(err) <= 5 else "wrong")
        res[cid] = {"run_id": run.run_id, "status": status, "verdict": rep.get("verdict"), "confidence": rep.get("confidence"),
                    "recovered_start_utc": rec, "true_start_utc": truth.get("true_start_utc"), "error_s": err,
                    "outcome": outcome, "calls": run.usage["calls"], "bottom_line": rep.get("bottom_line")}
        OUT.write_text(json.dumps(res, indent=1))
        print(cid, res[cid]["outcome"], res[cid]["verdict"], rec, "err", err, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
