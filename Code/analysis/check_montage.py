"""
What does the probe montage actually give us across subjects?

Why this exists
---------------
Two decisions in this project rest on the montage, and both were checked by hand on
2026-09-18 before this script recorded them:

  1. ROI features. The plan was to "register" channels onto regions (FEF, IPS, STG)
     using the optode coordinates. That is only meaningful if the coordinates differ
     between subjects, i.e. were digitised per head.
  2. The shared channel set. build_dataset.common_long_channels() intersects
     (source, detector) INDEX pairs across subjects and returns 28 channels. If a
     subject numbers its optodes differently, a channel at the same position can be
     dropped silently.

What it checks
--------------
  * The BIDS *_optodes.tsv positions of every subject, compared on the channels.
  * The Homer .nirs SD geometry of every subject, every long channel located by its
    source-detector midpoint, so channels can be matched by POSITION.
  * Which positions are shared by all twelve subjects, and which of those the
    index intersection misses.
  * Which region each channel falls in (by cluster, with the paper's naming).

Writes results/ds004830/check_montage/ds004830_check_montage.log and the matching .png
under plots/ (probe layout, channels coloured by region, the index-dropped channels
marked; see layout.py).

Usage:  python check_montage.py [--root PATH]
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import scipy.io as sio

import build_dataset as bd
import layout
import paths

HERE = os.path.dirname(os.path.abspath(__file__))
RES = lambda ext: layout.result("ds004830", "check_montage", ext)

ROI_BY_SOURCE = {1: "L-FEF", 2: "L-FEF", 3: "L-FEF", 4: "R-FEF", 5: "R-FEF",
                 6: "R-FEF", 9: "L-IPS", 11: "L-IPS", 10: "R-IPS", 12: "R-IPS",
                 7: "L-STG", 8: "R-STG", 13: "L-STG(08 only)", 14: "R-STG(08 only)"}
COLOURS = {"FEF": "#1f5fa8", "IPS": "#2a9d5c", "STG": "#c0502a"}


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def geometry(path, exp):
    runs, _, _ = bd.load_manifest(path, exp)
    f = os.path.join(path, runs[0] + ".nirs")
    sd = sio.loadmat(f, variable_names=["SD"])["SD"][0, 0]
    src = np.array(sd["SrcPos"], dtype=float)
    det = np.array(sd["DetPos"], dtype=float)
    _, dist, ident = bd.probe_geometry(f)
    chans = {}
    for (s, d), dd in zip(ident, dist):
        mid = tuple(float(v) for v in np.round((src[s - 1] + det[d - 1]) / 2, 1)[:2])
        chans[(s, d)] = {"mid": mid, "dist": float(dd), "long": dd >= bd.SS_MAX_MM}
    return src, det, chans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.dataset_root())
    args = ap.parse_args()
    log = Log(RES(".log"))
    log("montage check -- what the probe geometry allows across subjects\n")

    # 1. BIDS optode files: digitised per subject, or a template?
    bids = os.path.normpath(os.path.join(args.root, ".."))
    tsvs = sorted(glob.glob(os.path.join(bids, "sub-*", "nirs", "*_optodes.tsv")))
    log(f"1. BIDS *_optodes.tsv, {len(tsvs)} subjects")
    pos = {}
    for f in tsvs:
        rows = {r["name"]: (float(r["x"]), float(r["y"]), float(r["z"]))
                for r in csv.DictReader(open(f), delimiter="\t")}
        pos[os.path.basename(f).split("_")[0]] = rows
    # compare on the optodes the 28 index-shared channels use (S1-S6, S9-S12, D1-D15);
    # sub-08 numbers detectors differently from D16 on, so comparing every common NAME
    # would pair different physical optodes and report a spurious difference
    names = ([f"S{i}" for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12)]
             + [f"D{i}" for i in range(1, 16)])
    ref = next(iter(pos.values()))
    worst = max(np.abs(np.subtract(r[n], ref[n])).max() for r in pos.values() for n in names)
    zs = {v[2] for r in pos.values() for v in r.values()}
    log(f"   optodes compared (those of the shared channels): {len(names)};  max position difference between "
        f"subjects: {worst:.3f} mm;  distinct z values: {sorted(zs)}")
    cs = glob.glob(os.path.join(bids, "sub-*", "nirs", "*_coordsystem.json"))
    if cs:
        log(f"   coordsystem: {open(cs[0]).read().strip()}")
    log("   -> the positions are a flat 2-D TEMPLATE, identical for every subject, with no\n"
        "      anatomical landmarks. There is no per-subject geometry to register onto, so\n"
        "      'ROI space' can only mean pooling channels inside a region.\n")

    # 2. Homer geometry: index matching vs position matching
    log("2. Homer .nirs geometry, long channels (>= "
        f"{bd.SS_MAX_MM:g} mm) per subject")
    per = {}
    for exp, path in bd.experiment_dirs(args.root):
        src, det, chans = geometry(path, exp)
        per[exp] = (src, det, chans)
        n_long = sum(c["long"] for c in chans.values())
        log(f"   {exp}: {len(src)} sources, {len(det)} detectors, {n_long} long channels")
    by_index = set.intersection(*({k for k, c in ch.items() if c["long"]}
                                  for _, _, ch in per.values()))
    by_pos = set.intersection(*({c["mid"] for c in ch.values() if c["long"]}
                                for _, _, ch in per.values()))
    std = min(per, key=lambda e: len(per[e][2]))
    std_mid = {c["mid"]: k for k, c in per[std][2].items() if c["long"]}
    missed = sorted(std_mid[m] for m in by_pos if std_mid[m] not in by_index)
    log(f"\n   shared by index (source, detector): {len(by_index)} channels")
    log(f"   shared by position:                 {len(by_pos)} channels")
    for k in missed:
        m = per[std][2][k]["mid"]
        other = {e: [kk for kk, c in ch.items() if c["mid"] == m][0]
                 for e, (_, _, ch) in per.items()}
        diff = {e: v for e, v in other.items() if v != k}
        log(f"   dropped by index matching: S{k[0]}-D{k[1]} at {m} mm "
            f"({ROI_BY_SOURCE.get(k[0], '?')}); numbered "
            + ", ".join(f"S{v[0]}-D{v[1]} in {e}" for e, v in diff.items()))
    log("   -> index matching silently drops the superior temporal channels: the only\n"
        "      auditory-cortex coverage in an auditory-attention task. build_dataset_v2.py\n"
        "      --by-position recovers them (derived/ds004830/v2_30ch/).\n")

    # 3. region assignment of the 28 shared channels
    log("3. region of each shared channel (clusters named as Ning et al. name the targets)")
    counts = {}
    for k in sorted(by_index):
        counts.setdefault(ROI_BY_SOURCE[k[0]], []).append(f"S{k[0]}-D{k[1]}")
    for r, ch in counts.items():
        log(f"   {r:<6} {len(ch):>2}: {' '.join(ch)}")
    log(f"   hemisphere split: {sum('L-' in r for r in counts for _ in counts[r])} left, "
        f"{sum('R-' in r for r in counts for _ in counts[r])} right, mirrored about x = 0\n")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src, det, chans = per[std]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for (s, d), c in chans.items():
        a, b = src[s - 1][:2], det[d - 1][:2]
        roi = ROI_BY_SOURCE.get(s, "")
        col = next((v for k, v in COLOURS.items() if k in roi), "grey")
        if not c["long"]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="grey", lw=1, alpha=0.6)
            continue
        dropped = (s, d) in missed
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=3 if not dropped else 2.5,
                ls="-" if not dropped else "--")
    ax.scatter(src[:, 0], src[:, 1], marker="s", s=60, c="#d62728", zorder=3, label="source")
    ax.scatter(det[:, 0], det[:, 1], marker="s", s=40, c="#1f77b4", zorder=3,
               label="detector")
    for i, p in enumerate(src):
        ax.annotate(f"S{i + 1}", p[:2], xytext=(3, 3), textcoords="offset points", fontsize=7)
    for r, col in COLOURS.items():
        ax.plot([], [], color=col, lw=3, label=r)
    ax.plot([], [], color="k", lw=2.5, ls="--", label="dropped by index matching")
    ax.plot([], [], color="grey", lw=1, label="short separation")
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm, template; negative = left)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"ds004830 probe ({std}, standard numbering): {len(by_index)} channels "
                 f"shared by index, {len(by_pos)} by position", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(RES(".png"), dpi=150)
    log(f"wrote {layout.rel(RES('.log'))}, {layout.rel(RES('.png'))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
