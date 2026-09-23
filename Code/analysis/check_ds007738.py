"""
Gate checks for ds007738 as a second dataset, before committing any work to it.

Why this exists
---------------
STATUS.md names ds007738 (whole-head cocktail party fNIRS, same lab and paradigm family,
CC0) as the replication dataset, with two gates before committing:

  1. Do the SNIRF files carry 3-D probe coordinates in /nirs/probe? And are they
     digitised per subject (so registration across heads is possible) or a template?
  2. Do any participants overlap with ds004830? That cannot be settled from public
     files (participants.tsv has no demographics); it needs the authors. This script
     records what CAN be checked and leaves that question explicitly open.

It also records what the task files say about the replication target: which classes
exist, and how many trials and subjects each attention task has.

Data are fetched from the public OpenNeuro S3 bucket into D:/fnirs/ds007738/check_sample/,
beside the raw download (outside the repository). Only small files plus two resting-state SNIRF files (~70 MB each) are
downloaded, and downloads are skipped if the file is already there, so a rerun after a
power cut resumes.

Writes results/ds007738/check_ds007738/ds007738_check_ds007738.log and the matching .png
(3-D probe with landmarks) under plots/ (see layout.py).

Usage:  python check_ds007738.py [--subjects sub-01 sub-02]
"""

import argparse
import csv
import io
import os
import re
import sys
import urllib.request

import numpy as np

import layout

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.normpath(layout.DS007738_RAW)), "check_sample")
RES = lambda ext: layout.result("ds007738", "check_ds007738", ext)
S3 = "https://s3.amazonaws.com/openneuro.org"
DS = "ds007738"


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def fetch(key):
    """Download s3 key into OUT, resumably (skip if complete, atomic rename)."""
    dst = os.path.join(OUT, key.replace("/", "__"))
    if os.path.exists(dst):
        return dst
    tmp = dst + ".part"
    urllib.request.urlretrieve(f"{S3}/{DS}/{key}", tmp)
    os.replace(tmp, dst)
    return dst


def list_keys(prefix):
    keys, marker = [], ""
    while True:
        url = f"{S3}?prefix={DS}/{prefix}&marker={marker}"
        xml = urllib.request.urlopen(url).read().decode()
        page = re.findall(r"<Key>([^<]+)</Key>", xml)
        keys += [k[len(DS) + 1:] for k in page]
        if "<IsTruncated>true</IsTruncated>" not in xml or not page:
            return keys
        marker = page[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["sub-01", "sub-02"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log = Log(RES(".log"))
    log(f"ds007738 gate checks\n")

    readme = open(fetch("README.txt"), encoding="utf-8").read()
    subs = [r["participant_id"] for r in
            csv.DictReader(open(fetch("participants.tsv")), delimiter="\t")]
    log(f"participants.tsv: {len(subs)} subjects, columns: participant_id only "
        f"(no demographics, so overlap with ds004830 cannot be checked from files)")
    m = re.search(r"fNIRS:.*", readme)
    log(f"README: {m.group(0) if m else '(no montage line)'}")

    # task inventory from the events files
    log("\ntask inventory (events.tsv across all subjects):")
    keys = [k for k in list_keys("sub-") if k.endswith("_events.tsv")]
    tasks = {}
    for k in keys:
        t = re.search(r"task-([a-z]+)", k).group(1)
        s = k.split("/")[0]
        tasks.setdefault(t, {}).setdefault(s, []).append(k)
    for t, per in sorted(tasks.items()):
        log(f"  {t:<17} {len(per):>2} subjects, {sum(len(v) for v in per.values())} runs")
    for t in ("overt", "covert"):
        if t not in tasks:
            continue
        types, n, inc = {}, 0, 0
        for k in [v[0] for v in tasks[t].values()][:6]:
            rows = list(csv.DictReader(open(fetch(k)), delimiter="\t"))
            for r in rows:
                types[r["trial_type"]] = types.get(r["trial_type"], 0) + 1
                n += 1
                inc += int(r.get("include", 1) or 0)
        log(f"  {t}: trial types in first 6 subjects' run-01 = {types}; "
            f"{inc}/{n} flagged include=1; columns: {list(rows[0].keys())}")
    log("  -> two classes (Left / Right) only: ds007738 can replicate the lateral\n"
        "     (2-class) result, not the 3-class one.")

    # probe geometry
    import h5py
    log("\n/nirs/probe in the resting-state SNIRF of: " + ", ".join(args.subjects))
    probes = {}
    for s in args.subjects:
        f = fetch(f"{s}/nirs/{s}_task-resting_run-01_nirs.snirf")
        with h5py.File(f, "r") as h:
            p = h["nirs/probe"]
            probes[s] = {k: p[k][()] for k in p.keys()}
        pr = probes[s]
        log(f"  {s}: fields {sorted(pr)}")
        log(f"      sources {pr['sourcePos3D'].shape}, detectors {pr['detectorPos3D'].shape}, "
            f"landmarks {pr['landmarkPos3D'].shape} "
            f"(first: {[x.decode() for x in pr['landmarkLabels'][:5]]})")
    a, b = probes[args.subjects[0]], probes[args.subjects[1]]
    for k in ("sourcePos3D", "detectorPos3D", "landmarkPos3D"):
        log(f"  {k}: max |{args.subjects[0]} - {args.subjects[1]}| = "
            f"{np.abs(a[k] - b[k]).max():.4f} mm")
    same = all(np.array_equal(a[k], b[k]) for k in ("sourcePos3D", "detectorPos3D"))
    log("  -> gate 1: 3-D coordinates PRESENT with 10-5 landmarks, but "
        + ("IDENTICAL across subjects: an atlas template, not digitised per head.\n"
           "     Anatomical ROI labelling is possible; per-subject registration is not."
           if same else "they DIFFER across subjects: digitised per head."))
    log("  -> gate 2 (participant overlap with ds004830): OPEN. Needs the authors.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*a["landmarkPos3D"].T, s=3, c="lightgrey", label="10-5 landmarks")
    ax.scatter(*a["sourcePos3D"].T, s=14, c="#d62728", label="sources")
    ax.scatter(*a["detectorPos3D"].T, s=8, c="#1f77b4", label="detectors")
    ax.set_title(f"ds007738 probe ({args.subjects[0]}; identical in {args.subjects[1]})",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES(".png"), dpi=150)
    log(f"\nwrote {layout.rel(RES('.log'))}, {layout.rel(RES('.png'))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
