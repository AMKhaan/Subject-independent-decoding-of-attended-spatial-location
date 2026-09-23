"""
ROI-space features against channel-index features, within-subject and cross-subject.

Why this exists
---------------
The plan (STATUS.md, next-session item 3) was to map the 28 channels onto the regions
Ning et al. target, frontal eye field (FEF), intraparietal sulcus (IPS) and superior
temporal gyrus (STG), and test whether region-level features transfer across subjects
better than channel-index features. The idea came from the misaligned era, when
cross-subject decoding sat at chance and a registration step looked like the fix.

What the montage actually allows
--------------------------------
Checked 2026-09-18 against every subject's *_optodes.tsv:

  * The coordinates are a flat 2-D probe layout (z = 0, no anatomical landmarks), and
    they are IDENTICAL for all twelve subjects on the 28 shared channels. They are a
    template, not digitised positions. There is no per-subject geometry to register,
    and channel j already sits at the same nominal probe position in every subject.
    "ROI space" can therefore only mean spatial pooling within a region.
  * The 28 channels form four clusters, split exactly by hemisphere (x < 0 is left):
        left FEF   sources 1-3,   10 channels   (y -5..35 mm, x -36..-64)
        right FEF  sources 4-6,   10 channels
        left IPS   sources 9, 11,  4 channels   (y -71..-85 mm, x -13..-39)
        right IPS  sources 10, 12, 4 channels
  * STG is NOT in the shared montage. Its channels (S7-D16, S8-D17 in the eleven
    standard subjects) are dropped by common_long_channels() because sub-08's larger
    probe numbers its detectors differently, so the (source, detector) intersection
    misses them. Recovering STG would need channel matching by position rather than by
    index; it is recorded here as an open item, not attempted.

So the question this script can answer honestly is narrower than planned: does
averaging within the four regions (8 region x chromophore signals x 6 time bins = 48
features, against 336) help or hurt, within-subject and under LOSO? Averaging trades
spatial detail for noise reduction and a 7x smaller feature space; which wins is an
empirical question.

Same trials (behaviourally correct), same time bins, same four models and the same
fold logic as decode.py; only the spatial feature map changes.

Writes results/ds004830/roi_features/ds004830_v2_roi_features.log and .json (see layout.py);
the figure is drawn by plot_diagnostics.py.

Usage:  python roi_features.py [--data ../../derived/ds004830/v2] [--perms 200]
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
from scipy import stats

import decode
import layout

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

ROIS = {                                   # by source index of the channel
    "L-FEF": (1, 2, 3),
    "R-FEF": (4, 5, 6),
    "L-IPS": (9, 11),
    "R-IPS": (10, 12),
}


class Log:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def load_roi(data_dir, task):
    """decode.load, with channels averaged inside each ROI before binning."""
    import glob
    subs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        X, ch = z["X"], np.asarray(z["channels"])
        n = len(ch)
        cols = []
        for chrom in (0, 1):                               # HbO block, then HbR block
            for _, srcs in ROIS.items():
                m = np.isin(ch[:, 0], srcs)
                cols.append(X[:, :, chrom * n + np.where(m)[0]].mean(axis=2))
        R = np.stack(cols, axis=2)                         # (trials, time, 8)
        F = decode.featurise(R, float(z["tmin"]), float(z["fs"]))
        y = z["y"].astype(int)
        ok = z["correct"].astype(bool)
        F, y = F[ok], y[ok]
        if task == "lateral":
            k = np.isin(y, (1, 2))
            F, y = F[k], y[k]
        subs.append((str(z["subject"]), F, y))
    return subs


def check_rois(data_dir, log):
    import glob
    z = np.load(sorted(glob.glob(os.path.join(data_dir, "*.npz")))[0], allow_pickle=True)
    ch = np.asarray(z["channels"])
    used = np.zeros(len(ch), bool)
    for name, srcs in ROIS.items():
        m = np.isin(ch[:, 0], srcs)
        used |= m
        log(f"  {name:<6} {int(m.sum()):>2} channels: "
            + " ".join(f"S{s}-D{d}" for s, d in ch[m]))
    assert used.all(), "every shared channel must fall in exactly one ROI"
    log(f"  all {len(ch)} channels assigned; STG absent from the shared montage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=layout.derived("ds004830", "v2"))
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    log = Log(layout.result(args.data, "roi_features", ".log"))
    log(f"\n##### roi_features.py  {time.strftime('%Y-%m-%d %H:%M')}  data={args.data}")
    check_rois(args.data, log)

    summary = {}
    for task in ("3class", "lateral"):
        chance = 1 / 3 if task == "3class" else 1 / 2
        feats = {"channel (336)": decode.load(args.data, task, correct_only=True),
                 "ROI (48)": load_roi(args.data, task)}
        log(f"\n=== {task}, correct trials, chance {chance * 100:.1f}% ===")
        res = {}
        for proto in ("within", "loso"):
            log(f"  -- {proto} --")
            log(f"  {'model':<20}{'channel':>10}{'ROI':>10}{'ROI - ch':>11}   paired Wilcoxon p")
            for name, clf in decode.models().items():
                a = {}
                for fk, subs in feats.items():
                    a[fk] = (decode.within_subject(subs, clf, rng=0) if proto == "within"
                             else decode.loso(subs, clf))
                c, r = a["channel (336)"], a["ROI (48)"]
                p = stats.wilcoxon(r, c).pvalue if np.any(r != c) else 1.0
                log(f"  {name:<20}{c.mean() * 100:9.2f}%{r.mean() * 100:9.2f}%"
                    f"{(r - c).mean() * 100:+10.2f}   {p:.4f}")
                res[(proto, name)] = a
        # permutation test of the best ROI model, so a ROI win cannot be noise
        for proto in ("within", "loso"):
            best = max(decode.models(),
                       key=lambda m: res[(proto, m)]["ROI (48)"].mean())
            obs = res[(proto, best)]["ROI (48)"].mean()
            if args.perms:
                null = decode.permute(feats["ROI (48)"], decode.models()[best],
                                      args.perms, 0, proto)
                p = (1 + (null >= obs).sum()) / (1 + args.perms)
                log(f"  ROI best {proto}: {best} {obs * 100:.2f}%, permutation null "
                    f"{null.mean() * 100:.2f}%, p = {p:.4f}")
        summary[task] = {f"{k[0]}|{k[1]}": {fk: v.tolist() for fk, v in a.items()}
                         for k, a in res.items()}

    import json
    js = layout.result(args.data, "roi_features", ".json")
    with open(js, "w") as fh:
        json.dump(summary, fh, indent=1)
    log(f"wrote {layout.rel(js)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
