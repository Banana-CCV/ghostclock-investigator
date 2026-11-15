"""Zero-cost reliability benchmark: the deterministic GhostClock pipeline on excerpts of the real
venue recording (18.4 min, ceiling-light flicker), searched blind across 7 days of GB grid history.

Per excerpt: extract (cached real trace) -> blind search Nov 5-11 2026 (Elexon 15 s, 1 s interp)
-> challenges (margin vs anywhere else, split halves) -> ANSWER or ABSTAIN.
Ground truth: the camera's network-synced clock, 2026-11-08T11:42:37Z + excerpt offset.
Correct = answered start within +-5 s of truth.

usage: python -m eval.bench_venue_excerpts            (writes eval/results/venue_excerpts.{json,md})
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.tools import _smooth, epoch, iso, load_grid, ncc_scan

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data/cases/venue/C004_video.csv"
TRUTH0 = epoch("2026-11-08T11:42:37Z")
SEARCH = (epoch("2026-11-05T00:00:00Z"), epoch("2026-11-12T00:00:00Z"))
LENGTHS_MIN = [2, 3, 5, 8, 12]
STEP_S = 90
QMIN = 15.0
MARGIN_MIN, HALF_TOL_S, SCORE_MIN, CORRECT_TOL_S = 0.05, 10, 0.80, 5


def prep(t, f, q):
    tt = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1)
    ff, qq = np.interp(tt, t, f), np.interp(tt, t, q)
    good = (qq >= QMIN) & (np.abs(ff - np.median(ff)) < 0.15)
    if good.sum() < 30:
        good = qq >= np.percentile(qq, 50)
    return tt, np.where(good, ff, np.median(ff[good])), good.astype(float) + 1e-6, good


def main():
    arr = np.loadtxt(TRACE, delimiter=",", skiprows=1)
    T, F, Q = arr.T
    rt, rf, src, cad = load_grid(SEARCH[0], SEARCH[1])
    rs = _smooth(rf)
    rows = []
    for L in LENGTHS_MIN:
        for a in range(0, int(T[-1] - L * 60), STEP_S):
            b = a + L * 60
            m = (T >= a + 8) & (T <= b - 8)
            if m.sum() < 30:
                continue
            tt, ff, w, good = prep(T[m] - a, F[m], Q[m])
            sc = ncc_scan(rs, ff, w)
            starts = rt[: len(sc)] - tt[0]
            i = int(np.argmax(sc))
            far = np.abs(starts - starts[i]) > 600
            second = float(np.max(sc[far]))
            margin = float(sc[i]) - second
            # split halves, each searched +-2 h around the best
            h = len(tt) // 2
            loc = (starts > starts[i] - 7200) & (starts < starts[i] + 7200)
            offs = []
            for sl in (slice(0, h), slice(h, None)):
                s2 = ncc_scan(rs, ff[sl], w[sl])
                st2 = rt[: len(s2)] - tt[sl][0]
                l2 = (st2 > starts[i] - 7200) & (st2 < starts[i] + 7200)
                j = np.flatnonzero(l2)[int(np.argmax(s2[l2]))]
                offs.append(int(st2[j] - starts[i]))
            halves_ok = all(abs(o) <= HALF_TOL_S for o in offs)
            answer = sc[i] >= SCORE_MIN and margin >= MARGIN_MIN and halves_ok
            truth = TRUTH0 + a
            err = float(starts[i] - truth)
            rows.append(dict(length_min=L, offset_s=a, best=iso(starts[i]), truth=iso(truth), error_s=err,
                             score=round(float(sc[i]), 4), runner_up=round(second, 4), margin=round(margin, 4),
                             halves_offsets_s=offs, answered=bool(answer),
                             correct=bool(answer and abs(err) <= CORRECT_TOL_S),
                             wrong=bool(answer and abs(err) > CORRECT_TOL_S),
                             top1_correct=bool(abs(err) <= CORRECT_TOL_S)))
        print(f"{L} min done ({sum(r['length_min'] == L for r in rows)} excerpts)")
    df = pd.DataFrame(rows)
    summ = df.groupby("length_min").agg(excerpts=("answered", "size"), answered=("answered", "sum"),
                                        correct=("correct", "sum"), wrong=("wrong", "sum"),
                                        top1_correct=("top1_correct", "sum"),
                                        median_abs_error_s=("error_s", lambda e: float(np.median(np.abs(e)))))
    out = ROOT / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "venue_excerpts.json", orient="records", indent=1)
    md = ["# Venue excerpts benchmark (real recording, blind 7-day search)", "",
          f"Reference: {src} ({cad} s cadence), {iso(SEARCH[0])} to {iso(SEARCH[1])}. "
          f"Rule: answer only if score >= {SCORE_MIN}, margin over anywhere else >= {MARGIN_MIN}, "
          f"halves agree within {HALF_TOL_S} s. Correct = within +-{CORRECT_TOL_S} s of the camera clock.", "",
          "```", summ.to_string(), "```", "",
          f"Totals: {int(df.answered.sum())}/{len(df)} answered, {int(df.correct.sum())} correct, "
          f"{int(df.wrong.sum())} wrong. Excerpts overlap (step {STEP_S} s) and all come from ONE recording."]
    (out / "venue_excerpts.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
