r"""Physically modelled mains-flicker injection into an existing video clip.

Takes a real (flicker-free) clip and adds rolling-shutter banding whose phase follows
a real grid-frequency trace, so the ENF extractor recovers that trace.

Model (per output frame k, per row r):
  row time      t_kr = t0 + k/fps + r*T_ro/H  (+ T_exp/2, exposure centre)
  grid phase    Phi(t) = 2*pi * integral 2*f_grid(s) ds        (light flickers at 2f)
  lamp output   L(t) = 1 + m1*g1*cos(Phi - lag1) + m2*g2*cos(2*Phi - lag2)
                g_h = sinc(h*2f*T_exp) exposure-integration gain; lag_h = phosphor delay
  pixel (linear light)  I' = I * (1 + M(x,y) * (L(t_kr) - 1))
  M(x,y) = how much of that pixel's light comes from the mains lamps (0..1),
           estimated from a smoothed median frame; saturated pixels get M ~ 0.

Then: back to display gamma, sensor noise, 8-bit, H.264 (yuv420p).
Writes <out>.mp4, <out>_truth.csv (t_rel_s, utc_s, f_hz) and <out>_params.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", path])
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return int(s["width"]), int(s["height"]), num / den, float(s.get("duration", 0))


def read_frames(path, W, H, scale_h=None):
    vf = [] if not scale_h else ["-vf", f"scale=-2:{scale_h}"]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, *vf, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    if scale_h:
        W, H = int(round(W * scale_h / H / 2) * 2), scale_h
    n = W * H * 3
    frames = []
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        frames.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
    p.wait()
    return np.stack(frames), W, H


def srgb_to_lin(x):
    x = np.asarray(x, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(y):
    y = np.clip(y, 0, 1)
    return 255.0 * np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055)


LUT_IN = srgb_to_lin(np.arange(256)).astype(np.float32)          # uint8 -> linear
_N_OUT = 16384
LUT_OUT = lin_to_srgb(np.linspace(0, 1, _N_OUT)).astype(np.float32)  # linear -> sRGB


def load_grid(parquet, utc_start, dur_s):
    df = pd.read_parquet(parquet)
    d = df[(df.ts_utc >= utc_start - 5) & (df.ts_utc <= utc_start + dur_s + 5)].copy()
    d = d[(d.f_hz > 49.0) & (d.f_hz < 51.0)]
    return d.ts_utc.to_numpy(float), d.f_hz.to_numpy(float)


def lamp_map(frames_lin, sigma=25, sat_thresh=0.85, gamma=0.7, tube_gain=2.5):
    """Fraction of each pixel's light that comes from the mains lamps (heuristic), plus a
    'tube' mask: near-saturated lamp pixels, which (with headroom) show the strongest bands."""
    med = np.median(frames_lin[:: max(1, len(frames_lin) // 15)], axis=0).mean(axis=2)
    tube = gaussian_filter((med > sat_thresh).astype(np.float32), 1.5)
    lum = gaussian_filter(med, sigma)
    M = (lum / (np.percentile(lum, 98) + 1e-6)).clip(0, 1) ** gamma
    M = 0.35 + 0.65 * M            # artificial light dominates everywhere indoors
    M = M * (1 - tube) + tube_gain * tube   # the lamp surface itself flickers hardest
    return M.astype(np.float32), tube.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out", help="output prefix (no extension)")
    ap.add_argument("--grid", required=True, help="parquet with ts_utc,f_hz (1 s)")
    ap.add_argument("--utc-start", required=True, help="true start, e.g. 2026-09-17T07:31:00Z")
    ap.add_argument("--dur", type=float, default=None, help="output seconds (loops clip)")
    ap.add_argument("--tro-ms", type=float, default=28.0)
    ap.add_argument("--texp-ms", type=float, default=4.0)
    ap.add_argument("--m1", type=float, default=0.03, help="100 Hz depth at the lamp")
    ap.add_argument("--m2", type=float, default=0.006)
    ap.add_argument("--phosphor-ms", type=float, default=1.5)
    ap.add_argument("--noise", type=float, default=0.6, help="sensor noise, grey levels")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--tube-gain", type=float, default=2.5, help="extra depth on lamp surfaces")
    ap.add_argument("--tube-headroom", type=float, default=0.8, help="scale clipped tubes to this")
    ap.add_argument("--clean-plate", action="store_true", help="median background for static cams")
    ap.add_argument("--motion-thresh", type=float, default=0.03, help="linear diff = 'moving'")
    ap.add_argument("--scale-h", type=int, default=None)
    ap.add_argument("--pingpong", action="store_true", help="loop forward/backward")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-fps", type=float, default=None,
                    help="re-time output (frames picked by time); avoid fps dividing 100")
    a = ap.parse_args()

    W0, H0, src_fps, src_dur = probe(a.clip)
    fps = a.out_fps or src_fps
    src, W, H = read_frames(a.clip, W0, H0, a.scale_h)
    n_src = len(src)
    dur = a.dur or n_src / fps
    n_out = int(round(dur * fps))
    utc0 = pd.Timestamp(a.utc_start).value / 1e9
    tg, fg = load_grid(a.grid, utc0, dur)

    # light phase on a fine grid (relative seconds)
    fs_ph = 4000.0
    tt = np.arange(-1.0, dur + 1.0, 1 / fs_ph)
    f_fine = np.interp(utc0 + tt, tg, fg)
    rng = np.random.default_rng(a.seed)
    phi = 2 * np.pi * np.cumsum(2 * f_fine) / fs_ph + rng.uniform(0, 2 * np.pi)

    T_ro, T_exp, tau = a.tro_ms / 1e3, a.texp_ms / 1e3, a.phosphor_ms / 1e3
    g1, g2 = np.sinc(100 * T_exp), np.sinc(200 * T_exp)
    lag1, lag2 = np.arctan(2 * np.pi * 100 * tau), np.arctan(2 * np.pi * 200 * tau)
    att1 = 1 / np.sqrt(1 + (2 * np.pi * 100 * tau) ** 2)
    att2 = 1 / np.sqrt(1 + (2 * np.pi * 200 * tau) ** 2)

    M, tube = lamp_map(srgb_to_lin(src[:: max(1, n_src // 30)].astype(np.float32)),
                       tube_gain=a.tube_gain)
    M = M[..., None]
    headroom = (1 - (1 - a.tube_headroom) * tube)[..., None]   # pull clipped tubes below 1.0

    # clean plate: static camera -> per-pixel median removes the stock clip's own codec noise;
    # original pixels are kept only where something moves (soft motion mask).
    plate = None
    if a.clean_plate:
        lin_all = LUT_IN[src[:: max(1, n_src // 60)]]
        plate = np.median(lin_all, axis=0).astype(np.float32)
        del lin_all
    rows = np.arange(H)
    order = np.arange(n_src)
    if a.pingpong:
        order = np.r_[order, order[-2:0:-1]]

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                            "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                            a.out + ".mp4"], stdin=subprocess.PIPE)
    noise_bank = rng.standard_normal((32, H, W, 1)).astype(np.float32) * a.noise
    lin_src = None
    for k in range(n_out):
        j = int(k / fps * src_fps)
        lin = LUT_IN[src[order[j % len(order)]]]
        if plate is not None:
            diff = np.abs(lin - plate).max(axis=2)
            mot = gaussian_filter((diff > a.motion_thresh).astype(np.float32), 4).clip(0, 1)[..., None]
            lin = plate * (1 - mot) + lin * mot
        lin = lin * headroom
        t_rows = k / fps + rows * T_ro / H + T_exp / 2
        ph = np.interp(t_rows, tt, phi)
        L = 1 + a.m1 * att1 * g1 * np.cos(ph - lag1) + a.m2 * att2 * g2 * np.cos(2 * ph - lag2)
        lin *= 1 + M * (L[:, None, None].astype(np.float32) - 1)
        idx = np.clip(lin * (_N_OUT - 1), 0, _N_OUT - 1).astype(np.int32)
        out = LUT_OUT[idx]
        out += noise_bank[k % 32]
        out += 0.5
        np.clip(out, 0, 255, out=out)
        enc.stdin.write(out.astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()

    sec = np.arange(0, int(dur) + 1)
    pd.DataFrame({"t_rel_s": sec, "utc_s": utc0 + sec, "f_hz": np.interp(utc0 + sec, tg, fg)}
                 ).to_csv(a.out + "_truth.csv", index=False)
    json.dump(dict(vars(a), fps=fps, W=W, H=H, n_out=n_out, g1=float(g1), att1=float(att1)),
              open(a.out + "_params.json", "w"), indent=1)
    print("wrote", a.out + ".mp4", n_out, "frames", f"{W}x{H}@{fps:.3f}")


if __name__ == "__main__":
    main()
