"""
When in the trial does the decodable information appear?

The grand-average waveform in this dataset is dominated by slow systemic components,
so it is a poor test of whether the reconstructed onsets are correct. Time-resolved
decoding is a much better one, and it is a test the pipeline can fail:

  - a classifier trained and tested on a window *before* stimulus onset should be at
    chance, because the subject does not yet know where to attend;
  - accuracy should rise on the hemodynamic timescale, a few seconds after onset;
  - it should decay as the response returns to baseline.

Any of those failing points at a problem. Above-chance accuracy in the pre-onset
window in particular would mean information is leaking across trials, or that the
onsets are misaligned by more than a trial.

Because a condition-independent offset is identical for all three classes, it cancels
in the decision function. That is why this analysis is trustworthy where the average
waveform is not.

Usage:  python time_resolved.py [--win 4] [--step 0.5] [--perms 200]
"""

import argparse
import glob
import os
import sys
import warnings
import numpy as np

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

import paths

warnings.filterwarnings("ignore")

DATA = paths.data_dir()
CLF = lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def load(data_dir):
    subs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        subs.append((str(z["subject"]), z["X"], z["y"].astype(int),
                     float(z["tmin"]), float(z["fs"])))
    return subs


def window_accuracy(X, y, s0, s1, repeats=3, seed=0):
    """Within-subject 5-fold accuracy using only samples [s0, s1) of the epoch."""
    F = X[:, s0:s1, :].mean(axis=1)
    accs = []
    for rep in range(repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed + rep)
        for tr, te in cv.split(F, y):
            m = clone(CLF()).fit(F[tr], y[tr])
            accs.append((m.predict(F[te]) == y[te]).mean())
    return float(np.mean(accs))


def curve(subs, win_s, step_s, shuffle_seed=None):
    """Group-mean accuracy at each window position. shuffle_seed permutes labels."""
    name, X0, y0, tmin, fs = subs[0]
    w = int(round(win_s * fs))
    step = int(round(step_s * fs))
    starts = list(range(0, X0.shape[1] - w + 1, step))
    centres = np.array([tmin + (s + w / 2) / fs for s in starts])

    g = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    out = np.empty((len(subs), len(starts)))
    for i, (_, X, y, _, _) in enumerate(subs):
        yy = g.permutation(y) if g is not None else y
        for j, s0 in enumerate(starts):
            out[i, j] = window_accuracy(X, yy, s0, s0 + w)
    return centres, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--fig", default="fig_time_resolved.png")
    args = ap.parse_args()

    subs = load(args.data)
    print(f"{len(subs)} subjects, {sum(len(s[2]) for s in subs)} trials")
    print(f"{args.win:g} s window, {args.step:g} s step, shrinkage LDA, "
          f"within-subject 5-fold x3\n")

    t, acc = curve(subs, args.win, args.step)
    grp = acc.mean(axis=0)
    sem = acc.std(axis=0, ddof=1) / np.sqrt(len(subs))

    print("=== group accuracy by window centre ===")
    print(f"  {'centre (s)':>11}{'accuracy':>11}{'sem':>8}")
    for ti, a, s in zip(t, grp, sem):
        mark = "  <-- pre-onset" if ti < 0 else ""
        print(f"  {ti:>11.2f}{a * 100:>10.2f}%{s * 100:>7.2f}{mark}")

    pre = t < 0
    post = (t >= 2) & (t <= 10)
    print(f"\n  mean over pre-onset windows   {grp[pre].mean() * 100:.2f}%")
    print(f"  mean over 2-10 s windows      {grp[post].mean() * 100:.2f}%")
    print(f"  peak {grp.max() * 100:.2f}% at {t[grp.argmax()]:+.2f} s")

    if args.perms:
        print(f"\n=== permutation envelope, {args.perms} permutations ===")
        null = np.empty((args.perms, len(t)))
        for k in range(args.perms):
            null[k] = curve(subs, args.win, args.step, shuffle_seed=k)[1].mean(axis=0)
            if (k + 1) % max(1, args.perms // 10) == 0:
                print(f"    {k + 1}/{args.perms}", end="\r", flush=True)
        print(" " * 30, end="\r")
        # max-statistic correction across window positions
        thresh = np.percentile(null.max(axis=1), 95)
        sig = t[grp > thresh]
        print(f"  null mean {null.mean() * 100:.2f}%   "
              f"family-wise 95% threshold {thresh * 100:.2f}%")
        if len(sig):
            print(f"  windows above threshold: {sig[0]:+.2f} to {sig[-1]:+.2f} s "
                  f"({len(sig)} of {len(t)})")
        else:
            print("  no window survives correction for multiple comparisons")
        np.savez("time_resolved.npz", t=t, acc=acc, null=null, thresh=thresh)
    else:
        np.savez("time_resolved.npz", t=t, acc=acc)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # plotted as percentages to match every other number in the paper
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        # the windows lying entirely before onset: the negative control, marked as such
        pre = t + args.win / 2 <= 0
        if pre.any():
            ax.axvspan(t[pre].min() - args.win / 2, 0.0, color="#eceff3", zorder=0)
            ax.text(t[pre].min() - args.win / 2 + 0.1, 100 * (grp.max() + sem.max()),
                    "windows entirely\npre-onset", fontsize=7.5, color="0.45",
                    ha="left", va="top")
        ax.axhline(100 / 3, color="0.5", ls="--", lw=1, label="chance 33.3%", zorder=2)
        ax.axvline(0, color="0.3", lw=1, zorder=2)
        if args.perms:
            ax.axhline(100 * thresh, color="C3", ls=":", lw=1.2, zorder=2,
                       label="permutation 95% (max-stat)")
        ax.fill_between(t, 100 * (grp - sem), 100 * (grp + sem),
                        alpha=0.25, color="C0", zorder=3)
        ax.plot(t, 100 * grp, color="C0", lw=2, label="group mean", zorder=4)
        ax.set_xlabel("window centre relative to stimulus onset (s)")
        ax.set_ylabel("3-class accuracy (%)")
        ax.set_xlim(t.min() - args.win / 2, t.max() + 0.3)
        ax.set_title(f"Time-resolved within-subject decoding ({args.win:g} s window)",
                     fontsize=10.5)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=300)
        print(f"\nwrote {args.fig}")
    except Exception as e:
        print(f"\n(no figure: {type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
