"""Figure 1: what ds004830 ships, what has to be reconstructed, and what happens next.

This is a schematic, not a data figure -- nothing here is computed. It exists because the
single most useful thing this paper releases is the trial-onset reconstruction (C1), and a
reader deciding whether to use the dataset needs to see in one glance that the timing does
not come from the BIDS layer at all. The counts on the boxes (28 channels, 1079 trials,
140 x 56) are the ones asserted in Methods; if those change, change them here too.

Usage:  python fig_pipeline.py [--out fig_pipeline.png]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# palette: grey = as-distributed, red = unusable as distributed, blue = our contribution,
# dark = ordinary processing, amber = the evaluation protocols that are the paper's subject
C_SHIP, C_DEAD, C_OURS = "#f2f2f2", "#f6dcda", "#dde7f3"
E_SHIP, E_DEAD, E_OURS = "#9a9a9a", "#b8443c", "#2c6fbb"
C_PROC, E_PROC = "#ffffff", "#555555"
C_EVAL, E_EVAL = "#fbeedd", "#c98b28"


def box(ax, x, y, w, h, text, fc, ec, fs=7.6, weight="normal", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, zorder=3, linespacing=1.35)


def arrow(ax, p0, p1, ec="0.3", ls="-", lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 color=ec, lw=lw, ls=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_pipeline.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- row 1: the dataset as distributed -------------------------------------------
    ax.text(0.015, 0.965, "as distributed", fontsize=8.5, style="italic", color="0.4")
    box(ax, 0.03, 0.855, 0.20, 0.085,
        "BIDS layer\n$\\tt{events.tsv}$", C_DEAD, E_DEAD)
    box(ax, 0.27, 0.855, 0.20, 0.085,
        "Homer $\\tt{.nirs}$\nstim matrix $\\tt{s}$", C_DEAD, E_DEAD)
    box(ax, 0.51, 0.855, 0.20, 0.085,
        "Homer $\\tt{.nirs}$\n$\\tt{d}$, $\\tt{t}$, $\\tt{SD}$", C_SHIP, E_SHIP)
    box(ax, 0.75, 0.855, 0.22, 0.085,
        "PsychToolbox logs\n$\\tt{cfg}$, $\\tt{indexMoviesTest}$", C_SHIP, E_SHIP)

    ax.text(0.13, 0.828, "empty", fontsize=7.4, color=E_DEAD, ha="center", va="top",
            fontweight="bold")
    ax.text(0.37, 0.828, "identically zero", fontsize=7.4, color=E_DEAD, ha="center",
            va="top", fontweight="bold")
    ax.text(0.61, 0.828, "usable", fontsize=7.4, color="0.45", ha="center", va="top")
    ax.text(0.86, 0.828, "usable", fontsize=7.4, color="0.45", ha="center", va="top")

    # the two dead ends
    for x in (0.13, 0.37):
        ax.plot([x - 0.018, x + 0.018], [0.795, 0.762], color=E_DEAD, lw=1.6, zorder=3)
        ax.plot([x - 0.018, x + 0.018], [0.762, 0.795], color=E_DEAD, lw=1.6, zorder=3)
    ax.text(0.185, 0.735, "no timing survives\nin either layer", fontsize=7.8,
            color=E_DEAD, ha="center", va="top", style="italic")

    # ---- row 2: the reconstruction (our contribution) ---------------------------------
    # The right-hand column from here down is a reserved corridor (x >= 0.62 above the
    # epoch box, x >= 0.82 alongside the processing chain). Nothing else may enter it, so
    # the onset path stays visually separate from the ordinary signal path on the left.
    arrow(ax, (0.86, 0.852), (0.86, 0.760))
    box(ax, 0.62, 0.620, 0.35, 0.135,
        "trial-onset reconstruction  (C1, §3.2)\n"
        "$t_{\\rm onset}(i)=\\tt{startT}+\\tt{Trigger3}(i)$\n"
        "labels $\\leftarrow$ $\\tt{indexMoviesTest}$ col 2\n"
        "competing-talker trials $\\leftarrow$ col 5",
        C_OURS, E_OURS, fs=7.8)

    # validation callout
    box(ax, 0.62, 0.525, 0.35, 0.070,
        "validated: null-control re-epoching (Fig. 2)\n"
        "and the rediscovered sub-15 trial drop",
        "#ffffff", E_OURS, fs=7.4, lw=1.0)
    arrow(ax, (0.795, 0.617), (0.795, 0.597), ec=E_OURS, ls=":")

    # ---- row 3: signal processing chain, one straight run so no wrap arrow is needed ---
    # placed right of the descending arrow rather than left of it, where they collide
    ax.text(0.28, 0.545, "processing (§3.3)", fontsize=8.5, style="italic", color="0.4")

    steps = [
        "28 long\nchannels",
        "optical\ndensity",
        "bandpass\n0.01–0.5 Hz",
        "MBLL $\\rightarrow$\nHbO/HbR",
        "short-sep.\nregression",
        "decimate\n50 $\\rightarrow$ 10 Hz",
    ]
    # widths chosen so the chain stops at x = 0.802, leaving the onset corridor clear
    w, gap, y = 0.107, 0.026, 0.415
    for i, s in enumerate(steps):
        x = 0.03 + i * (w + gap)
        box(ax, x, y, w, 0.082, s, C_PROC, E_PROC, fs=6.6)
        if i:
            arrow(ax, (x - gap + 0.004, y + 0.041), (x - 0.004, y + 0.041))
    # bowed up-left so the final approach is steep: a sagging curve would run along the
    # box tops, which are drawn over it, and swallow the arrowhead
    arrow(ax, (0.61, 0.852), (0.0835, 0.505), rad=0.22)

    # epoching pulls in the reconstructed onsets
    box(ax, 0.03, 0.290, 0.94, 0.080,
        "epoch −2 to +12 s on the reconstructed onsets, baseline-corrected on −2–0 s\n"
        "$\\Rightarrow$  1079 trials  ×  140 time points  ×  56 (28 channels × 2 chromophores)",
        C_OURS, E_OURS, fs=8.2)
    arrow(ax, (0.0835, 0.413), (0.0835, 0.372))
    arrow(ax, (0.90, 0.523), (0.90, 0.372), ec=E_OURS)

    # ---- row 4: the fan-out that is the paper's subject -------------------------------
    ax.text(0.015, 0.252, "evaluation (§3.5)", fontsize=8.5, style="italic", color="0.4")

    evals = [
        ("A / W\nhonest\nLOSO · within", "32.7% / 40.8%"),
        ("B\npooled random\ntrial split", "37.9%"),
        ("C\npooled random\nwindow split", "43.2%"),
        ("D\nchannel-as-sample,\nlabel by channel", "79.3%"),
    ]
    w2, gap2 = 0.222, 0.019
    for i, (label, acc) in enumerate(evals):
        x = 0.03 + i * (w2 + gap2)
        box(ax, x, 0.098, w2, 0.108, label, C_EVAL, E_EVAL, fs=7.4)
        ax.text(x + w2 / 2, 0.070, acc, fontsize=9.0, fontweight="bold",
                ha="center", va="center", color="0.15")
        # the epoch box spans the same width as this row, so each protocol drops straight
        # out of it: four parallel arrows, no crossings, and the shared source is obvious
        arrow(ax, (x + w2 / 2, 0.288), (x + w2 / 2, 0.208))

    ax.text(0.5, 0.028,
            "identical trials, identical features, identical classifier — only the "
            "train/test rule differs",
            fontsize=8.2, ha="center", color="0.3", style="italic")

    fig.tight_layout(pad=0.4)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
