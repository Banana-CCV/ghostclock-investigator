"""CLI: python -m agent investigate <case_id> [--auto-supply] [--effort medium]

Prints each step as a compact decision card; writes runs/<run_id>/events.jsonl and state.json.
"""
import argparse
import json
import sys
import textwrap


def card(ev):
    t = ev["type"]
    if t == "start":
        print(f"\n=== CASE {ev['case']['case_id']}: {ev['case']['title']}  (model {ev['model']})")
    elif t == "thinking":
        print("  ~ " + textwrap.shorten(ev["text"].replace("\n", " "), 220))
    elif t == "say":
        print("  > " + textwrap.shorten(ev["text"].replace("\n", " "), 300))
    elif t == "tool_call":
        inp = {k: v for k, v in ev["input"].items() if k != "reason"}
        print(f"\n[{ev['name']}] {json.dumps(inp)[:160]}")
        if "reason" in ev["input"]:
            print(f"   because: {ev['input']['reason']}")
    elif t == "tool_result":
        flag = " ERROR" if ev.get("error") else ""
        print(f"   -> ({ev['seconds']} s){flag} {json.dumps(ev['result'], default=str)[:400]}")
    elif t == "board":
        b = ev["board"]
        print(f"   [board v{b['plan_version']}] " + "; ".join(f"{h['id']}:{h['status']}" for h in b["hypotheses"]))
    elif t == "request":
        print(f"\n!! REQUEST: {ev['request']['what']}\n   why: {ev['request']['why']}")
    elif t == "evidence_supplied":
        print(f"   << investigator supplied {ev['evidence']['id']}: {ev['evidence'].get('label')}")
    elif t == "report":
        r = ev["report"]
        print(f"\n### REPORT {r['verdict']} ({r['confidence']}): {r['headline']}")
        for f in r.get("findings", []):
            print("   - " + f)
        for s in r.get("next_steps", []):
            print("   > " + s)
    elif t == "end":
        print(f"\n=== {ev['status']}  usage={ev.get('usage')}")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(prog="ghostclock")
    sub = ap.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("investigate")
    inv.add_argument("case_id")
    inv.add_argument("--auto-supply", action="store_true", help="answer evidence requests automatically from the case file")
    inv.add_argument("--effort", default="medium")
    inv.add_argument("--model", default=None)
    inv.add_argument("--offline-from", default=None,
                     help="zero-cost test: replay the model turns of a past run id (tools run for real)")
    a = ap.parse_args()
    from .loop import MODEL, Run, spend_cap_run, spend_cap_total, spent_total
    if not a.offline_from:
        print(f"LIVE run: per-run cap ${spend_cap_run():.2f}; spent ${spent_total():.2f} of ${spend_cap_total():.2f}")
    run = Run(a.case_id, model=a.model or MODEL, effort=a.effort, on_event=card, offline_from=a.offline_from)
    print("run:", run.run_id)
    status = run.start(auto_supply=a.auto_supply)
    print("final status:", status, "| run dir:", run.dir)


if __name__ == "__main__":
    main()
