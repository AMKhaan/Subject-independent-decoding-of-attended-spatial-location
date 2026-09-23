"""Figure: the null control for the trial-onset reconstruction.

Two panels, identical pipeline, differing only in where the epochs were cut:
reconstructed onsets (startT + Trigger3) against onsets drawn at random times in the
same recordings. Reads the built epoch sets directly, so the figure and the numbers in
validate_alignment.py come from the same arrays.

Requires both `data/` and `data_null/` (the latter from build_dataset.py --null).

Usage:  python fig_alignment.py [--out fig_alignment.png]
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths

DATA = paths.data_dir()
NULL = paths.null_dir()


def group_average(d):
    """Trial- and channel-averaged HbO/HbR pooled over every subject in `d`."""
    Xs, tmin, fs = [], None, None
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        Xs.append(z["X"])
        tmin, fs = float(z["tmin"]), float(z["fs"])
    if not Xs:
        return None
    X = np.concatenate(Xs, axis=0)
    n_ch = X.shape[2] // 2
    hbo = X[:, :, :n_ch].mean(axis=(0, 2))
    hbr = X[:, :, n_ch:].mean(axis=(0, 2))
    t = tmin + np.arange(X.shape[1]) / fs
    post = t >= 0
    r = float(np.corrcoef(hbo[post], hbr[post])[0, 1])
    i = int(np.argmax(np.abs(hbo[post])))
    return dict(t=t, hbo=hbo, hbr=hbr, r=r, n=X.shape[0],
                peak_t=float(t[post][i]), peak_a=float(hbo[post][i]))


def panel(ax, g, title, show_ylabel):
    ax.axhline(0, lw=0.8, color="0.75", zorder=1)
    ax.axvspan(4, 8, color="#f3e4b8", alpha=0.55, zorder=0)
    ax.axvline(0, ls="--", lw=1.0, color="0.35", zorder=2)

    ax.plot(g["t"], g["hbo"], lw=2.0, color="#c1392b", label="HbO", zorder=3)
    ax.plot(g["t"], g["hbr"], lw=2.0, color="#2471a3", label="HbR", zorder=3)

    ax.plot([g["peak_t"]], [g["peak_a"]], "o", ms=5, color="#c1392b",
            mec="white", mew=1.0, zorder=4)
    # anchored in axes coordinates with a leader line, so the label cannot drift into
    # the title or off the panel wherever the peak happens to fall
    ax.annotate(f"peak {g['peak_t']:.1f} s, {g['peak_a']:.3f} µM",
                xy=(g["peak_t"], g["peak_a"]), xycoords="data",
                xytext=(0.97, 0.90), textcoords="axes fraction",
                fontsize=8, ha="right", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="0.45",
                                shrinkA=2, shrinkB=4))

    ax.set_title(f"{title}\nHbO/HbR r = {g['r']:+.3f}   ({g['n']} epochs)", fontsize=9.5)
    ax.set_xlabel("time from onset (s)")
    if show_ylabel:
        ax.set_ylabel("concentration change (µM)")
    ax.set_xlim(g["t"][0], g["t"][-1])
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--null", default=NULL)
    ap.add_argument("--out", default="fig_alignment.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    real = group_average(args.data)
    if real is None:
        sys.exit(f"no epochs in {args.data}; run build_dataset.py first")
    null = group_average(args.null)
    if null is None:
        sys.exit(f"no null set in {args.null}; run: python build_dataset.py --null")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    panel(axes[0], real, "reconstructed onsets  (startT + Trigger3)", True)
    panel(axes[1], null, "null control  (random onsets)", False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    # headroom for the anchored annotations, then name the band once along the floor
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo - 0.06 * (hi - lo), hi + 0.16 * (hi - lo))
    lo, hi = axes[0].get_ylim()
    axes[0].text(6, lo + 0.02 * (hi - lo), "canonical 4–8 s window",
                 ha="center", va="bottom", fontsize=7.5, color="#8a6d1f")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")
    print(f"  true: r={real['r']:+.3f} peak {real['peak_t']:.1f}s "
          f"amp {real['peak_a']:.4f} uM")
    print(f"  null: r={null['r']:+.3f} peak {null['peak_t']:.1f}s "
          f"amp {null['peak_a']:.4f} uM")


if __name__ == "__main__":
    main()
