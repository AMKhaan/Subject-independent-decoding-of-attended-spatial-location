"""Figure: the protocol staircase (Table 3 rendered).

Self-contained -- the four accuracies are the published output of leakage_demo.py and
are written here as literals so the figure can be regenerated without re-running the
ablation. If leakage_demo.py is re-run, update ROWS to match; the numbers are asserted
in one place only.

Usage:  python fig_ablation.py [--out fig_ablation.png]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANCE = 100.0 / 3.0

# (label, accuracy %, delta vs A in pp, description).
# The deltas are quoted from leakage_demo.py's own output rather than recomputed from
# the rounded accuracies above -- 37.87 - 32.72 rounds to +5.1, but the unrounded
# difference is +5.2, which is what the manuscript's Table 3 reports.
ROWS = [
    ("A", 32.72, None, "leave-one-subject-out\n(no leakage)"),
    ("B", 37.87, 5.2, "pooled random\ntrial split"),
    ("C", 43.25, 10.5, "pooled random\nwindow split"),
    ("D", 79.28, 46.6, "channel-as-sample,\nlabel by channel"),
]

# The comparison band is restricted to the *three-class* rows of Table 1 (~45% for Ning et
# al., >90% for the classical-ML survey). The two-class rows reach 96.8%, but shading them
# on a three-class accuracy axis would compare across different chance levels.
LIT_LO, LIT_HI = 45.0, 90.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_ablation.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    xs = range(len(ROWS))
    accs = [r[1] for r in ROWS]

    # the span of published accuracies, for scale
    ax.axhspan(LIT_LO, LIT_HI, color="0.90", zorder=0)
    ax.text(1.55, LIT_HI - 2.5, "accuracies reported for 3-class fNIRS (Table 1)",
            ha="center", va="top", fontsize=8, color="0.45")

    ax.bar(xs, accs, width=0.62, color=["#2c6fbb"] + ["#b8443c"] * 3, zorder=3)

    # chance line, labelled clear of bar A so the two labels cannot collide
    ax.axhline(CHANCE, ls="--", lw=1.1, color="0.25", zorder=4)
    # below the line and to the right of the last bar -- the only region of the panel that
    # no bar, value label or arrow occupies
    ax.text(4.15, CHANCE - 3.0, "chance 33.3%", fontsize=8, color="0.25", ha="right")

    for x, (lab, acc, delta, _) in zip(xs, ROWS):
        ax.text(x, acc + 1.6, f"{acc:.1f}%", ha="center", fontsize=10,
                fontweight="bold", zorder=5)
        if delta is not None:
            ax.text(x, acc / 2, f"+{delta:.1f} pp", ha="center",
                    fontsize=9, color="white", fontweight="bold", zorder=5)

    # the total swing, annotated
    ax.annotate("", xy=(3.42, accs[0]), xytext=(3.42, accs[-1]),
                arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.2))
    ax.text(3.52, (accs[0] + accs[-1]) / 2, f"{ROWS[-1][2]:.1f} pp\nswing",
            fontsize=9, va="center", fontweight="bold")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{r[0]}\n{r[3]}" for r in ROWS], fontsize=8.5)
    ax.set_ylabel("3-class accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.60, 4.20)
    ax.set_title("Identical data and features; only the train/test split rule changes",
                 fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
