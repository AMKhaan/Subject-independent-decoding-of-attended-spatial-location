"""Figure: per-subject within-subject accuracy against leave-one-subject-out accuracy.

The numbers are the per-subject table printed by
    decode.py --task 3class --correct-only
(best model in each protocol: Logistic L2 within, Random Forest LOSO) and are written
here as literals so the figure can be regenerated without re-running the benchmark.
If decode.py is re-run, update ROWS; the numbers are asserted in one place only.

Usage:  python fig_subjects.py [--out fig_subjects.png]
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANCE = 100.0 / 3.0

# (subject, within-subject % [Logistic L2], LOSO % [Random Forest])
ROWS = [
    ("08", 37.75, 33.73),
    ("12", 41.15, 36.05),
    ("13", 42.14, 36.78),
    ("14", 28.61, 40.28),
    ("15", 38.70, 28.79),
    ("16", 46.35, 34.12),
    ("19", 40.63, 32.95),
    ("21", 53.27, 28.57),
    ("22", 33.47, 28.77),
    ("23", 33.45, 40.28),
    ("24", 54.24, 36.78),
    ("25", 39.82, 34.78),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_subjects.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    subs = [r[0] for r in ROWS]
    win = np.array([r[1] for r in ROWS])
    los = np.array([r[2] for r in ROWS])
    r = float(np.corrcoef(win, los)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    # -- left: paired slope plot, one line per participant ------------------------
    ax = axes[0]
    # named in the legend rather than as floating text: twelve crossing lines leave no
    # reliably empty spot in this panel for an inline label
    ax.axhline(CHANCE, ls="--", lw=1.1, color="0.25", zorder=1, label="chance 33.3%")
    for s, w, l in zip(subs, win, los):
        ax.plot([0, 1], [w, l], "-o", ms=4, lw=1.1,
                color="#b8443c" if w > l else "#2c6fbb", alpha=0.85, zorder=3)
        ax.annotate(s, xy=(0, w), xytext=(-6, 0), textcoords="offset points",
                    fontsize=7, ha="right", va="center", color="0.35")
    ax.plot([0, 1], [win.mean(), los.mean()], "-", lw=3.0, color="0.15",
            zorder=4, label="group mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"within-subject\n(mean {win.mean():.1f}%)",
                        f"leave-one-subject-out\n(mean {los.mean():.1f}%)"], fontsize=9)
    ax.set_xlim(-0.28, 1.24)
    ax.set_ylabel("3-class accuracy (%)")
    ax.set_title("Every participant, both protocols", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # -- right: the two protocols against each other ------------------------------
    ax = axes[1]
    lo, hi = 26, 57
    ax.plot([lo, hi], [lo, hi], ls=":", lw=1.0, color="0.6", zorder=1)
    ax.axhline(CHANCE, ls="--", lw=1.0, color="0.25", zorder=1)
    ax.axvline(CHANCE, ls="--", lw=1.0, color="0.25", zorder=1)
    ax.scatter(win, los, s=42, color="#2c6fbb", zorder=3, ec="white", lw=0.8)
    for s, w, l in zip(subs, win, los):
        ax.annotate(s, xy=(w, l), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="0.35")
    ax.set_xlabel("within-subject accuracy (%)")
    ax.set_ylabel("LOSO accuracy (%)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f"Who decodes well within does not\ntransfer across   (r = {r:+.2f})",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")
    print(f"  within mean {win.mean():.2f}%  sd {win.std(ddof=1):.2f}  "
          f"range {win.min():.2f}-{win.max():.2f}")
    print(f"  loso   mean {los.mean():.2f}%  sd {los.std(ddof=1):.2f}  "
          f"range {los.min():.2f}-{los.max():.2f}")
    print(f"  within-vs-LOSO correlation across participants r = {r:+.4f}")
    print(f"  above chance: within {int((win > CHANCE).sum())}/12, "
          f"LOSO {int((los > CHANCE).sum())}/12")


if __name__ == "__main__":
    main()
