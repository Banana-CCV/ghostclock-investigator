"""The investigator's toolbox: thin, cached wrappers around the ghostclock engine.

Every tool returns a JSON-serialisable dict (or, for view_frames, text + images). Heavy results
(traces, match landscapes) are kept in the Toolbox and written to the run folder for the UI.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ghostclock import audio_enf, gb_reference, rs_enf
from ghostclock._paths import find_tool
from .case import ROOT, Case, resolve

CACHE = ROOT / "data" / "cache"
GB = ROOT / "data" / "gb"
SMOOTH_S = 16


def iso(ts: float) -> str:
    return pd.to_datetime(int(round(ts)), unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch(s: str) -> int:
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.timestamp())


# ---------------------------------------------------------------- reference data
def reference_coverage() -> dict:
    neso, elexon = [], []
    for p in sorted(GB.glob("neso_*.parquet")):
        neso.append(p.stem.replace("neso_", "").replace("_", "-"))
    live = GB / "elexon_live.json"
    if live.exists():
        j = json.loads(live.read_text())
        elexon.append(j.get("coverage_iso"))
    return {"NESO_1s_months_local": neso, "ELEXON_15s_local": elexon,
            "ELEXON_15s_remote": "any date via the Elexon API (fetched on demand, ~15 s cadence, noisier)"}


def _neso_months_available(a: int, b: int) -> bool:
    m = pd.Timestamp(a, unit="s").to_period("M")
    end = pd.Timestamp(b, unit="s").to_period("M")
    while m <= end:
        if not (GB / f"neso_{m.year:04d}_{m.month:02d}.parquet").exists():
            return False
        m += 1
    return True


def _elexon_days(a: int, b: int) -> pd.DataFrame:
    """Elexon 15 s data for [a, b]: local live archive + per-day cache, fetching missing days."""
    import datetime as dt
    from ghostclock import elexon_archiver as ea
    frames = []
    live = GB / "elexon_live.parquet"
    live_df = pd.read_parquet(live)[["ts_utc", "f_hz"]] if live.exists() else None
    day = pd.Timestamp(a, unit="s", tz="UTC").normalize()
    while day.timestamp() <= b:
        d0, d1 = int(day.timestamp()), int(day.timestamp()) + 86400
        cp = CACHE / f"elexon_day_{day:%Y-%m-%d}.parquet"
        covered = live_df is not None and live_df[(live_df.ts_utc >= d0) & (live_df.ts_utc < d1)].ts_utc.nunique() > 5700
        if covered:
            frames.append(live_df[(live_df.ts_utc >= d0) & (live_df.ts_utc < d1)])
        elif cp.exists():
            frames.append(pd.read_parquet(cp))
        else:
            got = ea.fetch(dt.datetime.fromtimestamp(d0, dt.timezone.utc),
                           dt.datetime.fromtimestamp(min(d1, int(pd.Timestamp.now(tz="UTC").timestamp())),
                                                     dt.timezone.utc))[["ts_utc", "f_hz"]]
            if day + pd.Timedelta(days=1) < pd.Timestamp.now(tz="UTC"):
                got.to_parquet(cp)
            frames.append(got)
        day += pd.Timedelta(days=1)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts_utc").sort_values("ts_utc")
    return df[(df.ts_utc >= a) & (df.ts_utc <= b)]


def load_grid(a: int, b: int, source: str = "auto"):
    """1 s grid frequency over [a, b) as (t, f, source_used, cadence). NaN-free (gaps interpolated)."""
    src = source.upper()
    if src == "AUTO":
        src = "NESO" if _neso_months_available(a, b) else "ELEXON"
    if src == "NESO":
        df = gb_reference.load_reference(a, b, source="NESO", fetch=False)
    else:
        df = _elexon_days(a, b)
    df = df[(df.ts_utc >= a) & (df.ts_utc < b)].dropna()
    df = df[(df.f_hz > 49.0) & (df.f_hz < 51.0)]
    if len(df) < 60:
        raise ValueError(f"no usable {src} reference between {iso(a)} and {iso(b)}")
    t1 = np.arange(int(df.ts_utc.min()), int(df.ts_utc.max()) + 1)
    f1 = np.interp(t1, df.ts_utc.to_numpy(float), df.f_hz.to_numpy(float))
    cadence = 15 if src == "ELEXON" else 1
    return t1, f1, src, cadence


def _smooth(x, w=SMOOTH_S):
    k = np.ones(int(w)) / int(w)
    return np.convolve(x, k, mode="same")


def ncc_scan(ref, q, w):
    n = len(q)
    L = len(ref) - n + 1
    if L <= 0:
        return np.array([])
    w = w / w.sum()
    qc = q - np.sum(w * q)
    qs = np.sqrt(np.sum(w * qc ** 2))
    win = np.lib.stride_tricks.sliding_window_view(ref, n)
    out = np.empty(L)
    step = max(1000, int(2e7 // n))
    for i in range(0, L, step):
        R = win[i:i + step]
        mu = R @ w
        Rc = R - mu[:, None]
        out[i:i + step] = (Rc * qc) @ w / (np.sqrt((Rc ** 2) @ w) * qs + 1e-15)
    return out


def _peaks(sc, starts, k=5, sep=600):
    order = np.argsort(sc)[::-1]
    chosen = []
    for i in order:
        if not np.isfinite(sc[i]):
            continue
        if all(abs(i - j) > sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= k:
            break
    return [{"start_utc": iso(starts[i]), "score": round(float(sc[i]), 4)} for i in chosen]


# ---------------------------------------------------------------- the toolbox
class Toolbox:
    def __init__(self, case: Case, run_dir: Path, emit=None):
        self.case = case
        self.run_dir = run_dir
        self.emit = emit or (lambda *a, **k: None)
        self.traces: dict[str, dict] = {}
        self.searches: dict[str, dict] = {}
        CACHE.mkdir(parents=True, exist_ok=True)

    # ---- helpers
    def _path(self, eid):
        return resolve(self.case.item(eid)["path"])

    def _probe_media(self, eid):
        out = subprocess.run([find_tool("ffprobe"), "-v", "error", "-show_streams", "-show_format",
                              "-of", "json", str(self._path(eid))], capture_output=True, text=True).stdout
        j = json.loads(out)
        v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
        a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
        info = {"duration_s": round(float(j["format"].get("duration", 0)), 1),
                "has_audio": a is not None, "has_video": v is not None}
        if v:
            n, d = v.get("avg_frame_rate", "0/1").split("/")
            info.update(resolution=f"{v.get('width')}x{v.get('height')}", codec=v.get("codec_name"),
                        fps=round(float(n) / float(d), 3) if float(d) else None)
        if a:
            info["audio"] = f"{a.get('codec_name')} {a.get('sample_rate')} Hz"
        ex = self._excerpt(eid)
        if ex:
            info["duration_s"] = round(ex[1] - ex[0], 1)
        return info

    def _trace_cache_path(self, eid, channel, tag=""):
        return CACHE / f"{self.case.case_id}_{eid}_{channel}{tag}.csv"

    def _excerpt(self, eid):
        ex = self.case.item(eid).get("excerpt_s")
        return (float(ex[0]), float(ex[1])) if ex else None

    def _get_trace(self, eid, channel, harmonics=None, max_s=None):
        """Trace for the evidence item; for an excerpt item, the slice of the source file's trace,
        re-timed so t=0 is the excerpt's first frame."""
        t, f, q, how = self._get_trace_full(eid, channel, harmonics, max_s)
        ex = self._excerpt(eid)
        if ex:
            a, b = ex
            m = (t >= a + 8) & (t <= b - 8)   # whole analysis windows only
            t, f, q = t[m] - a, f[m], q[m]
        return t, f, q, how

    def _get_trace_full(self, eid, channel, harmonics=None, max_s=None):
        item = self.case.item(eid)
        tag = "" if not harmonics else "_h" + "".join(map(str, harmonics))
        if max_s:
            tag += f"_first{int(max_s)}"
        cp = self._trace_cache_path(eid, channel, tag)
        pre = (item.get("cache") or {}).get(channel) if not harmonics else None
        if pre and resolve(pre).exists():
            arr = np.loadtxt(resolve(pre), delimiter=",", skiprows=1)
            if max_s:
                arr = arr[arr[:, 0] <= arr[0, 0] + max_s]
            return arr[:, 0], arr[:, 1], arr[:, 2], "cached extraction"
        if cp.exists():
            arr = np.loadtxt(cp, delimiter=",", skiprows=1)
            return arr[:, 0], arr[:, 1], arr[:, 2], "cached extraction"
        path = str(self._path(eid))
        if channel == "audio":
            x, fs = audio_enf.load_audio(path)
            if max_s:
                x = x[: int(max_s * fs)]
            win = 16.0 if len(x) / fs >= 40 else 8.0
            r = audio_enf.extract_enf(x, fs, harmonics=tuple(harmonics or (1, 2, 3, 4, 5, 6, 7)), win_s=win)
            t, f, q = r["t"], r["f"], r["q"]
        else:
            info = self._probe_media(eid)
            mf = int(max_s * (info.get("fps") or 30)) if max_s else None
            dur = min(max_s or 1e9, info["duration_s"])
            v = rs_enf.run_video(path, max_frames=mf, win_s=16.0 if dur >= 40 else 8.0)
            t, f, q = (np.asarray(x) for x in v["rs"])
        np.savetxt(cp, np.c_[t, f, q], delimiter=",", header="t_s,enf_hz,q_db", comments="")
        return t, f, q, "fresh extraction"

    @staticmethod
    def _quality(t, f, q, channel):
        thr = 15.0 if channel == "video" else 8.0
        good = (q >= thr) & (np.abs(f - np.median(f)) < 0.15)
        g = f[good]
        out = {"duration_s": round(float(t[-1] - t[0] + 1), 0), "windows": int(len(t)),
               "median_quality_db": round(float(np.median(q)), 1),
               "usable_fraction": round(float(good.mean()), 3)}
        if good.sum() >= 10:
            out.update(median_hz=round(float(np.median(g)), 4),
                       wander_mhz_p5_p95=round(float((np.percentile(g, 95) - np.percentile(g, 5)) * 1e3), 1),
                       looks_like_grid=bool(49.8 < np.median(g) < 50.2 and
                                            (np.percentile(g, 95) - np.percentile(g, 5)) > 0.003))
        else:
            out["looks_like_grid"] = False
        u = out["usable_fraction"] * out["median_quality_db"]
        out["strength"] = "strong" if out["median_quality_db"] >= 15 and out["usable_fraction"] > 0.5 \
            else "weak" if out["usable_fraction"] > 0.2 else "unusable"
        return out

    # ---- tools
    def view_frames(self, evidence_id: str, count: int = 3):
        info = self._probe_media(evidence_id)
        dur = info["duration_s"]
        blocks = [{"type": "text", "text": json.dumps({"evidence_id": evidence_id, **info})}]
        off = (self._excerpt(evidence_id) or (0.0, 0.0))[0]
        for i in range(max(1, min(count, 4))):
            ts = off + dur * (i + 1) / (count + 1)
            jpg = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-ss", f"{ts:.2f}", "-i",
                                  str(self._path(evidence_id)), "-frames:v", "1", "-vf", "scale=768:-2",
                                  "-f", "image2", "-vcodec", "mjpeg", "-q:v", "4", "-"],
                                 capture_output=True).stdout
            if jpg:
                (self.run_dir / f"frame_{evidence_id}_{i}.jpg").write_bytes(jpg)
                blocks.append({"type": "text", "text": f"frame at {ts:.1f} s"})
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                           "data": base64.b64encode(jpg).decode()}})
        return blocks

    def probe_signals(self, evidence_id: str):
        info = self._probe_media(evidence_id)
        res = {"evidence_id": evidence_id, "media": info, "channels": {}}
        if info["has_audio"]:
            t, f, q, how = self._get_trace(evidence_id, "audio", max_s=None)
            res["channels"]["audio_hum"] = self._quality(t, f, q, "audio")
        if info["has_video"]:
            t, f, q, how = self._get_trace(evidence_id, "video", max_s=None)
            res["channels"]["light_flicker"] = self._quality(t, f, q, "video")
        return res

    def extract_trace(self, evidence_id: str, channel: str, harmonics: list | None = None):
        channel = "video" if channel in ("video", "light", "light_flicker", "flicker") else "audio"
        t, f, q, how = self._get_trace(evidence_id, channel, harmonics=harmonics)
        tid = f"{evidence_id}:{channel}" + (":" + "".join(map(str, harmonics)) if harmonics else "")
        self.traces[tid] = {"t": t, "f": f, "q": q, "channel": channel, "evidence_id": evidence_id}
        qual = self._quality(t, f, q, channel)
        step = max(1, len(t) // 600)
        (self.run_dir / f"trace_{tid.replace(':', '_')}.json").write_text(json.dumps(
            {"t": t[::step].round(1).tolist(), "f": f[::step].round(5).tolist(), "q": q[::step].round(1).tolist()}))
        return {"trace_id": tid, "channel": channel, "how": how, **qual}

    def _prep(self, tid, qmin=None):
        tr = self.traces[tid]
        t, f, q = tr["t"], tr["f"], tr["q"]
        qmin = qmin if qmin is not None else (15.0 if tr["channel"] == "video" else 8.0)
        tt = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1)
        ff, qq = np.interp(tt, t, f), np.interp(tt, t, q)
        good = (qq >= qmin) & (np.abs(ff - np.median(ff)) < 0.15)
        if good.sum() < 30:
            good = qq >= np.percentile(qq, 50)
        w = good.astype(float) + 1e-6
        ff = np.where(good, ff, np.median(ff[good]))
        return tt, ff, w, good

    def _scan(self, tid, a, b, source="auto"):
        tt, ff, w, good = self._prep(tid)
        rt, rf, src, cad = load_grid(a, b + int(tt[-1] - tt[0]) + 60, source)
        sc = ncc_scan(_smooth(rf), ff, w)
        starts = rt[: len(sc)] - tt[0]
        return sc, starts, src, cad, good

    def search_grid(self, trace_id: str, start_utc: str, end_utc: str, source: str = "auto"):
        a, b = epoch(start_utc), epoch(end_utc)
        if b - a > 14 * 86400:
            return {"error": "search window too large (max 14 days per call)"}
        if trace_id not in self.traces:
            return {"error": f"unknown trace {trace_id}; call extract_trace first"}
        sc, starts, src, cad, good = self._scan(trace_id, a, b, source)
        m = (starts >= a) & (starts <= b)
        sc, starts = sc[m], starts[m]
        if not len(sc):
            return {"error": "reference does not cover that window"}
        peaks = _peaks(sc, starts)
        days = {}
        for d in pd.to_datetime(starts, unit="s", utc=True).strftime("%Y-%m-%d").unique():
            dm = pd.to_datetime(starts, unit="s", utc=True).strftime("%Y-%m-%d") == d
            days[d] = round(float(np.nanmax(sc[dm])), 4)
        sid = f"S{len(self.searches) + 1}"
        self.searches[sid] = {"trace_id": trace_id, "peaks": peaks}
        step = max(1, len(sc) // 1500)
        (self.run_dir / f"search_{sid}.json").write_text(json.dumps({
            "trace_id": trace_id, "starts": starts[::step].astype(int).tolist(),
            "score": np.nan_to_num(sc[::step]).round(4).tolist(), "peaks": peaks}))
        best, second = peaks[0], (peaks[1] if len(peaks) > 1 else None)
        return {"search_id": sid, "reference": f"{src} ({cad} s cadence)",
                "window": [iso(a), iso(b)], "hours_searched": round(len(sc) / 3600, 1),
                "usable_fraction_of_trace": round(float(good.mean()), 3),
                "top_candidates": peaks, "best_per_day": days,
                "margin_best_vs_next": round(best["score"] - second["score"], 4) if second else None}

    def challenge(self, trace_id: str, candidate_start_utc: str, tests: list | None = None,
                  rival_days: int = 3):
        tests = tests or ["rival_days", "split_halves", "sharpness"]
        c = epoch(candidate_start_utc)
        tt, ff, w, good = self._prep(trace_id)
        dur = int(tt[-1] - tt[0])
        rt, rf, src, cad = load_grid(c - 7200, c + dur + 7200)
        rs = _smooth(rf)
        sc = ncc_scan(rs, ff, w)
        starts = rt[: len(sc)] - tt[0]
        i0 = int(np.argmin(np.abs(starts - c)))
        lo, hi = max(0, i0 - 30), min(len(sc), i0 + 31)
        i0 = lo + int(np.argmax(sc[lo:hi]))
        cand_score = float(sc[i0])
        out = {"candidate_start_utc": iso(starts[i0]), "candidate_score": round(cand_score, 4),
               "reference": f"{src} ({cad} s)", "tests": {}}
        if "sharpness" in tests:
            prof = {f"{o:+d}s": round(float(sc[i0 + o]), 4) for o in (-60, -10, -5, -2, 2, 5, 10, 60)
                    if 0 <= i0 + o < len(sc)}
            drop10 = cand_score - max(sc[max(0, i0 - 10)], sc[min(len(sc) - 1, i0 + 10)])
            out["tests"]["sharpness"] = {"scores_at_offsets": prof, "drop_at_10s": round(float(drop10), 4),
                                         "pass": bool(drop10 > 0.02)}
        if "split_halves" in tests:
            half = len(tt) // 2
            res = []
            for name, sl in (("first_half", slice(0, half)), ("second_half", slice(half, None))):
                q_, w_ = ff[sl], w[sl]
                s2 = ncc_scan(rs, q_, w_)
                st2 = rt[: len(s2)] - tt[sl][0]
                j = int(np.argmax(s2))
                res.append({"part": name, "implied_clip_start_utc": iso(st2[j]), "score": round(float(s2[j]), 4),
                            "offset_from_candidate_s": int(st2[j] - starts[i0])})
            agree = all(abs(r["offset_from_candidate_s"]) <= 10 for r in res)
            out["tests"]["split_halves"] = {"halves": res, "pass": bool(agree)}
        if "rival_days" in tests:
            rivals = []
            for d in [x for k in range(1, rival_days + 1) for x in (-k, k)]:
                a = c + d * 86400 - 6 * 3600
                try:
                    rt2, rf2, src2, _ = load_grid(a, a + 12 * 3600 + dur + 60)
                except Exception as e:  # reference not available for that day
                    rivals.append({"day_offset": d, "skipped": str(e)[:80]})
                    continue
                s3 = ncc_scan(_smooth(rf2), ff, w)
                j = int(np.argmax(s3))
                rivals.append({"day_offset": d, "best_start_utc": iso(rt2[j] - tt[0]),
                               "best_score": round(float(s3[j]), 4)})
            scored = [r["best_score"] for r in rivals if "best_score" in r]
            best_rival = max(scored) if scored else None
            out["tests"]["rival_days"] = {"window": "same day +-6 h of clock time", "rivals": rivals,
                                          "best_rival_score": best_rival,
                                          "margin": round(cand_score - best_rival, 4) if best_rival else None,
                                          "pass": bool(best_rival is not None and cand_score - best_rival > 0.05)}
        step = 1
        seg = rs[i0: i0 + len(ff)]
        (self.run_dir / f"overlay_{trace_id.replace(':', '_')}.json").write_text(json.dumps({
            "t": (starts[i0] + tt - tt[0]).astype(int)[::step].tolist(),
            "grid": seg[::step].round(5).tolist(),
            "recording": [None if not g_ else round(float(v), 5) for v, g_ in
                          zip(ff - np.mean(ff[good]) + np.mean(seg[good]), good)][::step]}))
        passed = [v.get("pass") for v in out["tests"].values()]
        out["summary"] = f"{sum(passed)}/{len(passed)} challenges passed"
        return out

    def grid_coverage(self):
        return reference_coverage()

    def research_context(self, query: str):
        import os
        if not os.environ.get("LINKUP_API_KEY"):
            return {"status": "unavailable", "note": "context research is not configured in this deployment"}
        from .research import linkup_search
        return linkup_search(query)
