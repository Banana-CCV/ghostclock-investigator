r"""banding.py (v9) - mains-light flicker as a rolling-shutter camera records it.

Physical model (all in scene-linear light, before noise, compression and the recorder's text overlay):
  * Light groups: every fitting belongs to one of three mains phases (120 deg apart). A pixel's
    lamp light is a weighted mix of the groups that reach it, so bands shift from zone to zone and
    partly cancel where zones overlap. A share of the light is steady (not mains-lit).
  * Steady sources (car headlights, anything brighter than the empty-scene plate) are added
    un-modulated: bands wash out inside a headlight beam.
  * Lamp waveform: rectified sine with a floor (phosphor afterglow) and an exponential decay tail
    (asymmetric), integrated over the exposure window. Blue dips more than red in the troughs.
  * Rolling shutter: sensor row r of frame k is read at t = k/fps + r*T_ro/H (+ small frame jitter).
    A wide-angle recorder de-warps the picture afterwards, so the bands bow slightly.
  * The fittings themselves and their halo pulse with their own phase.
  * Sensor noise is added after the light, then H.264; the OSD text is drawn last.
Outputs <out>.mp4 and, with --flicker-out, <out>_flicker.mp4 (the isolated modulation, for the
analysis view), plus <out>_params.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess

import cv2
import numpy as np
import pandas as pd

from inject_flicker import LUT_IN, LUT_OUT, _N_OUT, load_grid, probe, read_frames


def lamp_template(floor=0.45, decay=0.10, n=1024):
    """One flicker period (mean 1): |sin| with a floor, smeared by a one-sided phosphor decay."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    w = floor + (1 - floor) * np.abs(np.sin(th / 2))
    k = np.exp(-np.arange(n) / (decay * n))
    w = np.real(np.fft.ifft(np.fft.fft(w) * np.fft.fft(k / k.sum())))
    return w / w.mean()


def exposed_wave(template, frac):
    """Template averaged over an exposure window covering `frac` of a period."""
    n = len(template)
    F = np.fft.rfft(template)
    h = np.arange(len(F))
    return np.fft.irfft(F * np.sinc(h * frac) * np.exp(-1j * np.pi * h * frac), n)


def find_lamps(plum, thresh=0.75, min_area=60):
    n, lab, stats, cent = cv2.connectedComponentsWithStats((plum > thresh).astype(np.uint8), 8)
    lamps = [dict(cx=float(cent[i][0]), cy=float(cent[i][1]), area=int(stats[i, cv2.CC_STAT_AREA]), label=i)
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return lamps, lab


def group_weights(lamps, lab, H, W, r0_frac, ds=8):
    """Per-pixel share of lamp light from each of the three mains-phase groups (sums to 1)."""
    h, w = H // ds, W // ds
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = np.zeros((3, h, w), np.float32) + 1e-6
    r0 = (r0_frac * W / ds) ** 2
    for L in lamps:
        d2 = (xx - L["cx"] / ds) ** 2 + (yy - L["cy"] / ds) ** 2
        acc[L["group"]] += L.get("power", np.sqrt(L["area"])) / (d2 + r0)
    acc /= acc.sum(axis=0, keepdims=True)
    Wg = np.stack([cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC) for a in acc]).clip(0, 1)
    for L in lamps:                                    # a fitting's own surface is purely its own phase
        if L["label"] <= 0:
            continue
        own = cv2.GaussianBlur((lab == L["label"]).astype(np.float32), (0, 0), 2.0)
        for g in range(3):
            Wg[g] = Wg[g] * (1 - own) + own * (1.0 if g == L["group"] else 0.0)
    return Wg / (Wg.sum(axis=0, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("out")
    ap.add_argument("--grid", required=True); ap.add_argument("--utc-start", required=True)
    ap.add_argument("--t-src", type=float, default=0.0); ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None); ap.add_argument("--src-speed", type=float, default=1.0)
    ap.add_argument("--loop-tail", type=float, default=None)
    ap.add_argument("--k", type=int, default=3, help="flicker = k x 100 Hz (bands per frame = k*100*T_ro)")
    ap.add_argument("--tro-ms", type=float, default=30.0)
    ap.add_argument("--shutter", type=float, default=400.0, help="exposure 1/x s")
    ap.add_argument("--depth", type=float, default=0.5, help="scales the lamp waveform's swing")
    ap.add_argument("--floor", type=float, default=0.45); ap.add_argument("--decay", type=float, default=0.10)
    ap.add_argument("--lit", type=float, default=0.85, help="share of scene light that is mains-lit")
    ap.add_argument("--r0", type=float, default=0.16, help="lamp falloff radius (fraction of width)")
    ap.add_argument("--virtual-lamps", default=None, help="'x,y,power,group;...' fittings outside the frame")
    ap.add_argument("--bow", type=float, default=0.03, help="band curvature from de-warping (0 = straight)")
    ap.add_argument("--jitter-ms", type=float, default=0.25, help="frame timing jitter")
    ap.add_argument("--tint", type=float, default=0.10, help="extra blue dip / less red dip in troughs")
    ap.add_argument("--halo", type=float, default=0.35, help="pulsing glow around fittings")
    ap.add_argument("--lamp-headroom", type=float, default=1.12)
    ap.add_argument("--roll-boost", type=float, default=1.0)
    ap.add_argument("--sat", type=float, default=0.85); ap.add_argument("--noise", type=float, default=1.1)
    ap.add_argument("--crf", type=int, default=19); ap.add_argument("--scale-h", type=int, default=1080)
    ap.add_argument("--cover", default=None); ap.add_argument("--osd-cam", default=None); ap.add_argument("--osd-start", default=None)
    ap.add_argument("--flicker-out", action="store_true"); ap.add_argument("--stills", default=None)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    W0, H0, src_fps, _ = probe(a.clip)
    src, W, H = read_frames(a.clip, W0, H0, a.scale_h)
    n_src = len(src)
    src_dur_s = (n_src - 1) / src_fps
    fps = a.fps or src_fps
    dur = a.dur or (n_src / src_fps - a.t_src) / a.src_speed
    utc0 = pd.Timestamp(a.utc_start).value / 1e9
    tg, fg = load_grid(a.grid, utc0, dur)
    rng = np.random.default_rng(a.seed)
    sc_ = H / 1080

    fs_ph = 4000.0
    tt = np.arange(-1.0, dur + 1.0, 1 / fs_ph)
    dev = np.interp(utc0 + tt, tg, fg) - 50.0
    phi = 2 * np.pi * a.k * (100.0 * tt + 2 * a.roll_boost * np.cumsum(dev) / fs_ph) + rng.uniform(0, 6.28)

    lin_s = LUT_IN[src[:: max(1, n_src // 50)]]
    plate = np.median(lin_s, axis=0).astype(np.float32)
    del lin_s
    cover = None
    if a.cover:
        cm = np.zeros((H, W), np.uint8)
        for spec in a.cover.split(";"):
            x, y, w_, h_ = (int(v) for v in spec.split(","))
            cm[y:y + h_, x:x + w_] = 255
        fill = cv2.inpaint(np.clip(plate * 255, 0, 255).astype(np.uint8), cm, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255
        cover = (cv2.GaussianBlur(cm, (0, 0), 2).astype(np.float32) / 255)[..., None]
        plate = plate * (1 - cover) + fill * cover
    plum = plate.mean(axis=2)
    lamps, lab = find_lamps(plum)
    order = np.argsort([L["cx"] for L in lamps])
    for rank, i in enumerate(order):                   # neighbouring fittings share a circuit / phase
        lamps[i]["group"] = 1 + (rank * 2 // max(1, len(lamps)))        # detected: phases B and C
    for spec in (a.virtual_lamps or "").split(";"):
        if spec.strip():
            x, y, pw, gi = (float(v) for v in spec.split(","))
            lamps.append(dict(cx=x * sc_, cy=y * sc_, area=1, power=pw, group=int(gi), label=-1))
    Wg = group_weights(lamps, lab, H, W, a.r0)                           # (3,H,W)
    lampmask = cv2.GaussianBlur((plum > 0.75).astype(np.float32), (0, 0), 1.5)
    halo = np.stack([cv2.GaussianBlur(((plum > 0.75) & (Wg[g] > 0.5)).astype(np.float32), (0, 0), 38 * sc_) for g in range(3)])
    boost = (1 + (a.lamp_headroom - 1) * lampmask)[..., None]
    plate_b = plate * boost

    tmpl = lamp_template(a.floor, a.decay)
    wave = 1 + a.depth * (exposed_wave(tmpl, min(a.k * 100.0 / a.shutter, 8.0)) - 1)
    NW = len(wave)
    xs = (np.arange(W, dtype=np.float32) - W / 2) / W
    rowmap = np.arange(H, dtype=np.float32)[:, None] + a.bow * H * 4 * xs[None, :] ** 2   # effective sensor row
    T_ro = a.tro_ms / 1e3
    chan = np.array([1 - a.tint, 1.0, 1 + a.tint], np.float32)          # R, G, B dip multipliers
    bank = rng.standard_normal((16, H, W, 1)).astype(np.float32)
    jit = rng.normal(0, a.jitter_ms / 1e3, 100000)

    def render(k):
        t = k / fps
        st = a.t_src + t * a.src_speed
        if a.loop_tail is not None and st > src_dur_s:
            tail = src_dur_s - a.loop_tail
            u = (st - src_dur_s) % (2 * tail)
            st = src_dur_s - u if u < tail else a.loop_tail + (u - tail)
        cur = LUT_IN[src[min(int(round(st * src_fps)), n_src - 1)]]
        if cover is not None:
            cur = cur * (1 - cover) + plate * cover
        d = cv2.GaussianBlur(np.abs(cur - plate).max(axis=2), (0, 0), 4)
        mot = cv2.GaussianBlur((d > 0.03 + 0.15 * plum).astype(np.float32), (0, 0), 5)[..., None]
        scene = (plate * (1 - mot) + cur * mot) * boost
        lamp_part = np.minimum(scene, plate_b)             # light the fittings put on the scene
        steady = scene - lamp_part                         # headlights etc: never modulated
        ph = np.interp(t + jit[k % len(jit)] + rowmap * (T_ro / H), tt, phi)              # (H,W)
        mod = np.zeros((H, W), np.float32)
        glow = np.zeros((H, W), np.float32)
        for g in range(3):
            Lg = wave[((ph + g * 2 * np.pi / 3) % (2 * np.pi) * (NW / (2 * np.pi))).astype(np.int32) % NW].astype(np.float32)
            mod += Wg[g] * Lg
            glow += halo[g] * Lg
        m = 1 + a.lit * (mod - 1)                          # (H,W) around 1
        m3 = 1 + (m[..., None] - 1) * chan                 # colour shift in the troughs
        lin = lamp_part * m3 + steady + (a.halo * 0.05) * glow[..., None]
        out = LUT_OUT[np.clip(lin * (_N_OUT - 1), 0, _N_OUT - 1).astype(np.int32)]
        if a.sat != 1.0:
            g_ = out.mean(axis=2, keepdims=True)
            out = g_ + a.sat * (out - g_)
        out += bank[k % 16] * a.noise
        img = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        # isolated flicker (for the analysis view): modulation actually applied, weighted by lamp light
        share = lamp_part.mean(axis=2) / (scene.mean(axis=2) + 1e-4)
        fl = np.clip(128 + 127 * (m - 1) * share / max(1e-6, a.depth * 0.55), 0, 255).astype(np.uint8)
        if a.osd_start:
            img = np.ascontiguousarray(img)
            ts = (pd.Timestamp(a.osd_start) + pd.Timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")
            for txt, org in ((ts, (int(W - 395 * sc_), int(62 * sc_))), (a.osd_cam or "", (int(40 * sc_), int(H - 42 * sc_)))):
                cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.95 * sc_, (0, 0, 0), max(1, int(5 * sc_)), cv2.LINE_AA)
                cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.95 * sc_, (245, 245, 245), max(1, int(2 * sc_)), cv2.LINE_AA)
        return img, fl

    if a.stills:
        for s in a.stills.split(","):
            img, fl = render(int(float(s) * fps))
            cv2.imwrite(f"{a.out}_{s}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{a.out}_{s}_flicker.png", fl)
        print("stills written; lamps:", len(lamps)); return
    n_out = int(round(dur * fps))

    def enc(path, pix):
        return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{W}x{H}",
                                 "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
                                 "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    e1 = enc(a.out + ".mp4", "rgb24")
    e2 = enc(a.out + "_flicker.mp4", "gray") if a.flicker_out else None
    for k in range(n_out):
        img, fl = render(k)
        e1.stdin.write(img.tobytes())
        if e2:
            e2.stdin.write(fl.tobytes())
    for e in (e1, e2):
        if e:
            e.stdin.close(); e.wait()
    json.dump(dict(vars(a), W=W, H=H, fps=fps, lamps=len(lamps)), open(a.out + "_params.json", "w"), indent=1)
    print("wrote", a.out + ".mp4", n_out, f"frames {W}x{H}@{fps}", "lamps", len(lamps))


if __name__ == "__main__":
    main()
