"""
Epoch tensors for ds007738 (whole-head cocktail party fNIRS), in the format of derived/ds004830/v2.

Why this exists
---------------
ds004830 gives 12 subjects on one 28-channel montage. ds007738 (same lab family, same
cocktail-party paradigm, CC0) gives ~30 more on a whole-head high-density cap, and is the
replication dataset for the paper's two-class (left vs right) results. Its authors decode
within-subject only (per-subject random forest, nested repeated stratified k-fold; their
code, github.com/duwadisudan/wholehead-cocktail-party-fnirs, has no leave-one-subject-out
or pooled model anywhere), so cross-subject decoding is untested there too.

The tensors are written in exactly the format build_dataset_v2.py writes, so decode.py,
bootstrap_ci.py, normalisation_check.py and calibration_curve.py run on them unchanged.

Pipeline: ds004830's, adapted only where the hardware forces it
------------------------------------------------------------------
  same as ds004830  optical density; 0.01-0.5 Hz 3rd-order zero-phase Butterworth;
                    modified Beer-Lambert (DPF 6); mean short-channel regression per
                    chromophore; epoch -2 to +12 s; baseline -2 to 0 s; HbO block then HbR
  adapted           * no 8 mm channels exist: the nearest-neighbour ring is ~16-20 mm in
                      the (template) probe geometry. The dataset authors use a 20 mm
                      split (cfg_GLM 'distance_threshold': 20 mm), so channels < 20 mm
                      are the short regressors and >= 20 mm are signal channels.
                    * TDDR motion correction (Fishburn et al. 2019) on optical density,
                      as the authors do: this is a wearable whole-head system.
                    * native 8.99 Hz is kept (decode.featurise bins by time, not samples).
                    * extinction coefficients for 760/850 nm from Prahl's tabulation
                      (the .nirs files of ds004830 carried their own; SNIRF does not).
  cross-subject     channels are matched by (source, detector); the probe geometry is
                    checked to be identical for every subject first. One shared channel
                    set: long channels whose raw-intensity SNR (mean/sd) is >= 1.5 at both
                    wavelengths in >= 90% of recordings.

Trials: every event whose trial_type contains Left or Right. y: Right = 1, Left = 2 (the
ds004830 codes). `correct` = response_correct == 1 AND include == 1 (include is the
authors' own trial-quality flag); gaze_ok and include are saved separately too.

Resumable: one .npz per subject, skipped if present (--force to rebuild).

Writes <out>/<task>/sub-XX.npz (default derived/ds007738/<task>/), and
results/ds007738/build_ds007738/ds007738_<task>_build_ds007738.log with the matching .png
under plots/ (see layout.py).

Usage:  python build_ds007738.py --task overt [--raw D:/fnirs/ds007738/raw]
                                 [--out ../../derived/ds007738] [--no-tddr]
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
from scipy.signal import butter, filtfilt

from build_dataset import BAND, DPF, EPOCH, BASELINE, regress_short
import layout

HERE = os.path.dirname(os.path.abspath(__file__))
SS_SPLIT_MM = 20.0
SNR_MIN, SNR_FRAC = 1.5, 0.90
LABELS = {"Right": 1, "Left": 2}
# Prahl, molar extinction coefficients (cm^-1 / M, base 10) for (HbO, HbR)
PRAHL = {760: (1486.5865, 3843.707), 850: (2526.391, 1798.643)}


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


# ------------------------------------------------------------------------ reading

def read_snirf(path):
    with h5py.File(path, "r") as f:
        d, p = f["nirs/data1"], f["nirs/probe"]
        I = d["dataTimeSeries"][()]
        t = d["time"][()].ravel()
        n = I.shape[1]
        g = lambda k: np.array([d[f"measurementList{i}"][k][()] for i in range(1, n + 1)]).ravel()
        src, det, wl = g("sourceIndex").astype(int), g("detectorIndex").astype(int), \
            g("wavelengthIndex").astype(int)
        S, D = p["sourcePos3D"][()], p["detectorPos3D"][()]
        wls = p["wavelengths"][()].ravel()
    fs = 1.0 / np.median(np.diff(t))
    dist = np.linalg.norm(S[src - 1] - D[det - 1], axis=1)
    raw_hash = hash(I.tobytes())
    I, n_nan = clean_nans(I)
    return {"I": I, "t": t, "fs": fs, "src": src, "det": det, "wl": wl, "dist": dist,
            "wls": wls, "geom": np.concatenate([S.ravel(), D.ravel()]), "n_nan": n_nan,
            "hash": raw_hash}


def clean_nans(I):
    """Recordings from sub-20 onward carry NaN in the first sample (t = 0) of ~1000
    channels. The first sample is replaced by the second (it is 0.11 s of a 10-min run and
    is filtered away anyway), and any interior NaN is linearly interpolated per channel.
    Onsets are unaffected because sample times are kept."""
    n = int(np.isnan(I).sum())
    if n == 0:
        return I, 0
    I = I.copy()
    for c in np.where(np.isnan(I).any(0))[0]:
        x = I[:, c]
        ok = ~np.isnan(x)
        if ok.sum() < 2:
            I[:, c] = np.nanmean(x) if ok.any() else 1.0
            continue
        I[:, c] = np.interp(np.arange(len(x)), np.where(ok)[0], x[ok])
    return I, n


def read_events(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    out = []
    for r in rows:
        tt = r["trial_type"]
        lab = next((v for k, v in LABELS.items() if k in tt), None)
        if lab is None:
            continue
        num = lambda k: int(float(r[k])) if r.get(k) not in (None, "", "n/a") else 1
        out.append((float(r["onset"]), lab, num("response_correct"), num("gaze_ok"),
                    num("include"), tt))
    return out


# --------------------------------------------------------------------- processing

def tddr(od, fs):
    """Temporal Derivative Distribution Repair (Fishburn et al. 2019), per channel."""
    b, a = butter(3, 0.5 / (fs / 2), "low")
    low = filtfilt(b, a, od, axis=0)
    high = od - low
    deriv = np.diff(low, axis=0)
    w = np.ones_like(deriv)
    mu = np.zeros(deriv.shape[1])
    for _ in range(50):
        mu_new = (w * deriv).sum(0) / np.maximum(w.sum(0), 1e-12)
        dev = np.abs(deriv - mu_new)
        sigma = 1.4826 * np.median(dev, axis=0) + 1e-12
        r = dev / (4.685 * sigma)
        w = ((1 - r ** 2) * (r < 1)) ** 2
        if np.max(np.abs(mu_new - mu)) < 1e-10:
            mu = mu_new
            break
        mu = mu_new
    fixed = np.vstack([np.zeros((1, od.shape[1])), np.cumsum(w * (deriv - mu), axis=0)])
    fixed += low.mean(0) - fixed.mean(0)
    return fixed + high


def pair_channels(r):
    """(source, detector) -> column index per wavelength."""
    idx = {}
    for i, (s, d, w) in enumerate(zip(r["src"], r["det"], r["wl"])):
        idx.setdefault((s, d), {})[w] = i
    keys = sorted(k for k, v in idx.items() if len(v) == 2)
    c1 = np.array([idx[k][1] for k in keys])
    c2 = np.array([idx[k][2] for k in keys])
    dist = r["dist"][c1]
    return keys, c1, c2, dist


def process_run(r, keys_keep, use_tddr):
    keys, c1, c2, dist = pair_channels(r)
    pos = {k: i for i, k in enumerate(keys)}
    I = np.maximum(r["I"], np.abs(r["I"]).mean(0) * 1e-6 + 1e-12)
    od = -np.log(I / I.mean(0))
    if use_tddr:
        od = tddr(od, r["fs"])
    b, a = butter(3, [BAND[0] / (r["fs"] / 2), BAND[1] / (r["fs"] / 2)], btype="band")
    od = filtfilt(b, a, od, axis=0)
    e = np.array([PRAHL[int(round(r["wls"][0]))], PRAHL[int(round(r["wls"][1]))]]) * np.log(10)
    einv = np.linalg.pinv(e)                              # (chromophore, wavelength)
    L = dist / 10.0                                       # mm -> cm
    a1 = od[:, c1] / (L * DPF[0])
    a2 = od[:, c2] / (L * DPF[1])
    hbo = (einv[0, 0] * a1 + einv[0, 1] * a2) * 1e6       # micromolar
    hbr = (einv[1, 0] * a1 + einv[1, 1] * a2) * 1e6
    short = dist < SS_SPLIT_MM
    sel = np.array([pos[k] for k in keys_keep])
    hbo = regress_short(hbo[:, sel], hbo[:, short])
    hbr = regress_short(hbr[:, sel], hbr[:, short])
    return np.concatenate([hbo, hbr], axis=1), dist[sel]


def snr_ok(r):
    keys, c1, c2, _ = pair_channels(r)
    snr = np.nanmean(r["I"], 0) / (np.nanstd(r["I"], 0) + 1e-12)
    return {k: bool(snr[a] >= SNR_MIN and snr[b] >= SNR_MIN) for k, a, b in zip(keys, c1, c2)}


def epoch(data, fs, onsets):
    n_pre = int(round(-EPOCH[0] * fs))
    n = int(round((EPOCH[1] - EPOCH[0]) * fs))
    b1 = int(round((BASELINE[1] - EPOCH[0]) * fs))
    X, used = [], []
    for i, o in enumerate(onsets):
        s0 = int(round(o * fs)) - n_pre
        if s0 < 0 or s0 + n > len(data):
            continue
        seg = data[s0:s0 + n]
        X.append(seg - seg[:b1].mean(0))
        used.append(i)
    return np.asarray(X, np.float32), np.array(used, int)


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="overt")
    ap.add_argument("--raw", default=layout.DS007738_RAW)
    ap.add_argument("--out", default=str(layout.DERIVED / "ds007738"))
    ap.add_argument("--no-tddr", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--channels-from", default=None,
                    help="reuse another task's channel set, e.g. overt (for transfer tests)")
    args = ap.parse_args()
    tag = args.task + ("-notddr" if args.no_tddr else "")
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)
    log = Log(layout.result(out, "build_ds007738", ".log"))
    log(f"build_ds007738.py  task={args.task}  tddr={not args.no_tddr}  "
        f"{time.strftime('%Y-%m-%d %H:%M')}\n  raw={args.raw}\n  out={out}")

    runs = {}
    for f in sorted(glob.glob(os.path.join(args.raw, "sub-*", "nirs",
                                           f"*_task-{args.task}_run-*_nirs.snirf"))):
        sub = os.path.basename(f).split("_")[0]
        ev = f.replace("_nirs.snirf", "_events.tsv")
        with h5py.File(f, "r") as h:
            intact = "probe" in h["nirs"] and "data1" in h["nirs"]
        if not intact:
            # sub-24's visualorient file ships without a /nirs/probe group
            log(f"  ! skipped {os.path.basename(f)}: malformed SNIRF (no probe or data group)")
            continue
        if os.path.exists(ev):
            runs.setdefault(sub, []).append((f, ev))
    log(f"  {len(runs)} subjects, {sum(len(v) for v in runs.values())} runs with events\n")

    # pass 1: probe identity and the shared channel set (read once, cached to json)
    cache = os.path.join(out, "_channels.json")
    if os.path.exists(cache) and not args.force:
        meta = json.load(open(cache))
    else:
        geoms, oks, fsets, dists = [], [], [], None
        hashes, nans, dup = {}, {}, []
        for sub, rr in runs.items():
            for f, _ in rr:
                r = read_snirf(f)
                hashes.setdefault(sub, []).append(r["hash"])
                nans[os.path.basename(f)] = r["n_nan"]
                geoms.append(r["geom"])
                fsets.append(r["fs"])
                if dists is None:
                    keys, _, _, dist = pair_channels(r)
                    dists = dict(zip(keys, dist))
            # a subject whose runs carry byte-identical signal arrays has a duplicated
            # recording (sub-11: run-02 == run-01 but with different events); which events
            # belong to the data cannot be known, so the subject is excluded outright
            if len(set(hashes[sub])) < len(hashes[sub]):
                dup.append(sub)
            else:
                for f, _ in rr:
                    oks.append(snr_ok(read_snirf(f)))
        same = all(np.allclose(g, geoms[0]) for g in geoms)
        longk = [k for k, d in dists.items() if d >= SS_SPLIT_MM]
        frac = {k: np.mean([o.get(k, False) for o in oks]) for k in longk}
        keep = sorted(k for k in longk if frac[k] >= SNR_FRAC)
        meta = {"probe_identical": bool(same), "n_long": len(longk),
                "n_short": int(sum(d < SS_SPLIT_MM for d in dists.values())),
                "keep": [list(map(int, k)) for k in keep], "fs": float(np.median(fsets)),
                "fs_range": [float(min(fsets)), float(max(fsets))],
                "duplicated_subjects": dup,
                "recordings_with_nan": {k: v for k, v in nans.items() if v}}
        json.dump(meta, open(cache, "w"))
    keep = [tuple(k) for k in meta["keep"]]
    if args.channels_from:
        # control tasks must use the main task's channel set, so models transfer across
        # tasks feature for feature (control_eye_movement.py trains on one, tests on other)
        ref = json.load(open(os.path.join(args.out, args.channels_from, "_channels.json")))
        keep = [tuple(k) for k in ref["keep"]]
        log(f"  channel set taken from '{args.channels_from}': {len(keep)} channels "
            f"(this task's own SNR rule would keep {len(meta['keep'])})")
    for sub in meta["duplicated_subjects"]:
        log(f"  ! {sub} EXCLUDED: its runs contain byte-identical signal arrays (a duplicated\n"
            f"    recording) with different event files, so trial timing cannot be trusted")
        runs.pop(sub, None)
    nn = meta["recordings_with_nan"]
    if nn:
        log(f"  {len(nn)} recordings contain NaN samples (all in the first sample; replaced "
            f"and interpolated): {min(nn.values())}-{max(nn.values())} values each")
    log(f"  probe geometry identical across all recordings: {meta['probe_identical']}")
    log(f"  channels: {meta['n_long']} long (>= {SS_SPLIT_MM:g} mm), {meta['n_short']} short; "
        f"{len(keep)} long channels pass SNR >= {SNR_MIN} in >= {SNR_FRAC:.0%} of recordings")
    log(f"  sampling rate {meta['fs']:.4f} Hz (range {meta['fs_range'][0]:.4f}-"
        f"{meta['fs_range'][1]:.4f})\n")
    if not meta["probe_identical"]:
        log("  ! probe geometry differs between recordings: channel-index alignment across\n"
            "    subjects is NOT valid; stopping.")
        sys.exit(1)

    log(f"  {'subject':<8}{'runs':>5}{'trials':>8}{'L':>5}{'R':>5}{'correct':>9}{'include':>9}"
        f"{'gaze_ok':>9}")
    summary = {}
    for sub, rr in runs.items():
        dst = os.path.join(out, sub + ".npz")
        if os.path.exists(dst) and not args.force:
            z = np.load(dst, allow_pickle=True)
            summary[sub] = z
            log(f"  {sub:<8}{len(rr):>5}{len(z['y']):>8}  (cached)")
            continue
        Xs, ys, cor, gaz, inc, ons, runi = [], [], [], [], [], [], []
        dist_keep = None
        for j, (f, evf) in enumerate(rr):
            r = read_snirf(f)
            data, dist_keep = process_run(r, keep, not args.no_tddr)
            ev = read_events(evf)
            X, used = epoch(data, r["fs"], [e[0] for e in ev])
            Xs.append(X)
            ys += [ev[i][1] for i in used]
            cor += [ev[i][2] == 1 and ev[i][4] == 1 for i in used]
            gaz += [ev[i][3] == 1 for i in used]
            inc += [ev[i][4] == 1 for i in used]
            ons += [ev[i][0] for i in used]
            runi += [j + 1] * len(used)
        X = np.concatenate(Xs)
        y = np.array(ys, np.int8)
        np.savez_compressed(
            dst, X=X, y=y, subject=sub, correct=np.array(cor), gaze_ok=np.array(gaz),
            include=np.array(inc), onsets=np.array(ons), run=np.array(runi),
            trial_index=np.arange(len(y)), fs=meta["fs"], tmin=EPOCH[0],
            channels=np.array(keep), dist_mm=dist_keep, task=args.task,
            tddr=not args.no_tddr)
        summary[sub] = np.load(dst, allow_pickle=True)
        log(f"  {sub:<8}{len(rr):>5}{len(y):>8}{int((y == 2).sum()):>5}{int((y == 1).sum()):>5}"
            f"{int(np.sum(cor)):>9}{int(np.sum(inc)):>9}{int(np.sum(gaz)):>9}")

    # figure: grand-average HbO, Left vs Right, over subjects
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    curves = {1: [], 2: []}
    ntr = []
    for sub, z in summary.items():
        X, y, c = z["X"], z["y"], z["correct"].astype(bool)
        n = X.shape[2] // 2
        for lab in (1, 2):
            m = (y == lab) & c
            if m.any():
                curves[lab].append(X[m][:, :, :n].mean(axis=(0, 2)))
        ntr.append(int(c.sum()))
    t = EPOCH[0] + np.arange(len(curves[1][0])) / meta["fs"]
    for lab, col, name in ((2, "#1f5fa8", "Left"), (1, "#c0502a", "Right")):
        ax = axes[0]
        m = np.mean(curves[lab], axis=0)
        ax.plot(t, m, color=col, label=f"{name} (n = {len(curves[lab])} subjects)")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("time from event onset (s)")
    axes[0].set_ylabel("HbO, mean over long channels")
    axes[0].set_title(f"ds007738 {args.task}: grand average (correct & include trials)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(range(len(ntr)), ntr, color="#8fb3de")
    axes[1].set_xticks(range(len(ntr)))
    axes[1].set_xticklabels([s[4:] for s in summary], fontsize=6, rotation=90)
    axes[1].set_ylabel("usable trials (correct & include)")
    axes[1].set_title("trials per subject")
    fig.tight_layout()
    png = layout.result(out, "build_ds007738", ".png")
    fig.savefig(png, dpi=150)
    log(f"\n  total usable trials: {sum(ntr)} over {len(ntr)} subjects")
    log(f"wrote {layout.rel(out)}, {layout.rel(layout.result(out, 'build_ds007738', '.log'))}, "
        f"{layout.rel(png)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
