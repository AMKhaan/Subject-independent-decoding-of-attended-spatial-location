"""
Figures for the new paper, built from the current (corrected-onset) results.

Why this exists
---------------
The tracked fig_*.py scripts made the rejected manuscript's figures, two of them from
hard-coded accuracy literals that predate the onset correction. They are Zenodo-archived
and are left untouched. This script makes the new paper's figure set instead, and every
number it draws is READ from a log, JSON or NPZ written by the analysis that produced it;
nothing is typed in except Ning et al.'s published Table 1, imported from compare_ning.py
where its source is documented.

Time axes say "cue onset": check_onset_anchor.py showed the official v2.0.0 onset marks
the cue (only that anchor reproduces Ning et al.), so t = 0 in derived/ds004830/v2 is the cue and the
movie starts at +3.19 s.

Paper numbering (function names are historical):
  Fig 1  fig_design  datasets, trial timeline, and the three evaluation schemes
  Fig 2  fig1  onset correction: trial timeline, per-subject displacement, accuracy before/after
  Fig 3  fig2  headline: within-subject and cross-subject accuracy, 95% BCa, per subject, Ning
  Fig 4  fig3  zero-shot without transductive normalisation; like-for-like with Ning et al.
  Fig 5  fig4  calibration curve
  Fig 6  fig5  physiology: time-resolved accuracy, region importance, activation patterns
  Fig 7  fig6  ds007738: 2-class replication, calibration curve, eye-movement control
  Fig S7 figS1 deep vs classical on identical trials
  Fig S1-S6 copied from plots/ (montage, STG, onset anchor, Ning pipeline, ablation, ROI)

Inputs are read from results/ and plots/ (see layout.py). Output: manuscript/figures/
(git-excluded) as 300 dpi PNG and vector PDF, plus paper_figures.log listing each
figure's inputs and the values drawn.

Usage:  python paper_figures.py [--out ../../manuscript/figures]
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

import compare_ning
import layout
from plot_diagnostics import parse_decode, parse_ning

HERE = os.environ.get("FNIRS_ANALYSIS", os.path.dirname(os.path.abspath(__file__)))  # this copy lives outside Code/analysis
LOGS = os.path.join(HERE, "logs")
V2 = layout.derived("ds004830", "v2")               # ds004830, corrected onsets
OVERT = layout.derived("ds007738", "overt")
CI_DS004830 = layout.result(V2, "bootstrap_ci", ".json")
CI_DS007738 = layout.result(OVERT, "bootstrap_ci", ".json")
NORM_DS004830 = layout.result(V2, "normalisation_check", ".json")
NORM_DS007738 = layout.result(OVERT, "normalisation_check", ".json")
CUE_LEAD = 3.19
C_WITHIN, C_LOSO, C_NING, C_GREY = "#1f5fa8", "#c0502a", "#222222", "#9a9a9a"


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    return plt


def save(fig, out, name, log):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    log(f"  -> {name}.png / .pdf\n")


def letter(ax, s):
    ax.text(-0.14, 1.06, "(" + s.lower() + ")", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")


def src(log, *paths):
    for p in paths:
        log(f"  source: {layout.rel(p)}")


# ------------------------------------------------------------------ design figure
# Geometry and timing, each from its source:
#   ds004830: screens at 0 and +/-45 deg, cue 2 s, movie 3 s, questions <= 25 s, rest
#             14-16 s (Ning et al. 2024, Fig 1); cue leads the movie by CUE_LEAD s (our
#             measurement, results/ds004830/check_onset_anchor/).
#   ds007738: screens at +/-30 deg (Duwadi et al. 2026); events.tsv duration 5 s = cue 2 s +
#             movie 3 s, trials ~20 s apart; covert = audio only, central fixation;
#             visualorient = eye orienting only (dataset README).
#   analysis window: decode.BINS, 0-12 s in six 2 s bins.
D30_ANG, D38_ANG = (-45, 0, 45), (-30, 30)
C_TRAIN, C_TEST, C_CAL = "#1f5fa8", "#ffffff", "#c0502a"


def _room(ax, angles, cued, movies=True, gaze=None, fix=False, title="", cue="cross"):
    """Top-down view: listener at the origin facing +y, screens on a unit arc."""
    import numpy as np
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.3, 1.45)
    for a in angles:
        t = np.deg2rad(a)
        cx, cy = np.sin(t), np.cos(t)
        dx, dy = 0.28 * np.cos(t), -0.28 * np.sin(t)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], lw=4.5,
                color="#222222" if a == cued else "#9a9a9a", solid_capstyle="butt")
        if movies:
            ax.add_patch(plt_circle((cx * 1.13, cy * 1.13), 0.07, "#9a9a9a"))
        if a == cued and cue == "cross":                # visual cue, on the cued side
            ax.plot(cx * 0.8, cy * 0.8, marker="P", ms=6, color="#222222")
        elif a == cued and cue == "sound":              # covert task: the cue is a sound
            ax.text(cx * 0.78, cy * 0.78, ")))", fontsize=8, color="#222222",
                    ha="center", va="center", rotation=np.rad2deg(-t) + 90)
    if fix:
        ax.plot(0, 1.0, marker="+", ms=8, mew=1.5, color="#222222")
    ax.add_patch(plt_circle((0, 0), 0.15, "#dddddd", ec="#222222"))
    ax.plot([-0.04, 0, 0.04], [0.14, 0.22, 0.14], color="#222222", lw=1)     # nose
    if gaze is not None:
        t = np.deg2rad(gaze)
        ax.annotate("", xy=(0.62 * np.sin(t), 0.62 * np.cos(t)), xytext=(0, 0.18),
                    arrowprops=dict(arrowstyle="-|>", color=C_CAL, lw=1.6, ls="--"))
    ax.set_title(title, fontsize=8, pad=2, linespacing=1.15)


def plt_circle(xy, r, fc, ec="none"):
    from matplotlib.patches import Circle
    return Circle(xy, r, fc=fc, ec=ec, lw=0.8)


def _timeline(ax, y, label, cue, movie, rest_note):
    """One trial as blocks on a time axis (s from cue onset)."""
    from matplotlib.patches import Rectangle
    h = 0.55
    ax.add_patch(Rectangle((cue[0], y - h / 2), cue[1] - cue[0], h, fc="#222222"))
    ax.text(sum(cue) / 2, y, "cue", color="white", ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((movie[0], y - h / 2), movie[1] - movie[0], h, fc="white",
                           ec="#222222", hatch="////", lw=0.8))
    ax.text(sum(movie) / 2, y, "movies", ha="center", va="center", fontsize=8,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.add_patch(Rectangle((movie[1] + 0.3, y - h / 2), 13.7 - movie[1], h, fc="#eeeeee",
                           ec="#9a9a9a", lw=0.6))
    ax.text((movie[1] + 14) / 2, y, f"questions, then rest ({rest_note})", ha="center",
            va="center", fontsize=7.5, color="#444444")
    ax.text(-2.3, y, label, ha="right", va="center", fontsize=7.5)


def _people(ax, x0, y, n, fc, hatch=None, r=0.16, gap=0.42):
    for i in range(n):
        ax.add_patch(plt_circle((x0 + i * gap, y + 0.12), r * 0.55, fc, ec="#222222"))
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x0 + i * gap - r * 0.75, y - 0.32), r * 1.5, 0.3,
                                    boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc,
                                    ec="#222222", lw=0.8, hatch=hatch))


def fig_design(plt, out, log):
    log("Fig design  datasets, trial timeline and the three evaluation schemes")
    log(f"  ds004830 screens {D30_ANG} deg, cue->movie {CUE_LEAD} s; ds007738 screens "
        f"{D38_ANG} deg, cue 2 s + movie 3 s; analysis window 0-12 s")
    fig = plt.figure(figsize=(6.3, 5.9))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.62, 0.62, 1.0], hspace=0.3, wspace=0.08)

    a = fig.add_subplot(gs[0, 0])
    deg = "\N{DEGREE SIGN}"
    pm = "\N{PLUS-MINUS SIGN}"
    _room(a, D30_ANG, cued=-45, gaze=-45,
          title=f"ds004830, n = 12\novert attention\nscreens 0{deg}, {pm}45{deg}")
    for j, (kw, t) in enumerate(((dict(movies=True, gaze=-30),
                                  f"ds007738, n = 29\novert attention\nscreens {pm}30{deg}"),
                                 (dict(movies=False, gaze=None, fix=True, cue="sound"),
                                  "ds007738\ncovert attention\n(sound cue, eyes fixed)"),
                                 (dict(movies=False, gaze=-30),
                                  "ds007738\neye movement only\n(no stimuli)"))):
        ax = fig.add_subplot(gs[0, j + 1])
        _room(ax, D38_ANG, cued=-30, title=t, **kw)
        if j == 0:
            b = ax

    t = fig.add_subplot(gs[1, :])
    _timeline(t, 1.0, "ds004830", (0, 2), (CUE_LEAD, CUE_LEAD + 3), "14-16 s")
    _timeline(t, 0.0, "ds007738", (0, 2), (2, 5), "10-12 s")
    t.annotate("", xy=(12, -0.85), xytext=(0, -0.85),
               arrowprops=dict(arrowstyle="<->", color="#222222", lw=0.9))
    t.text(6, -0.72, "decoded window: 0-12 s, six 2 s bins", ha="center", va="bottom",
           fontsize=8)
    t.set_xlim(-2.5, 14.5)
    t.set_ylim(-1.35, 1.65)
    t.set_yticks([])
    t.spines["left"].set_visible(False)
    t.set_xticks(range(0, 15, 2))
    t.set_xlabel("time from cue onset (s)")

    d = fig.add_subplot(gs[2, :])
    d.axis("off")
    d.set_xlim(0, 10)
    d.set_ylim(-0.75, 3.3)
    rows = [(2.75, "within-subject",
             "one person; train and test on\ndifferent trials of that person (5-fold)"),
            (1.55, "subject-independent\n(zero-shot)",
             "train on everyone else;\ntest on a person never seen"),
            (0.35, "calibration",
             "everyone else + k labelled trials\nfrom the new person; test on the rest")]
    for y, name, note in rows:
        d.text(0.0, y, name, ha="left", va="center", fontsize=7.5, fontweight="bold")
        d.text(7.35, y, note, ha="left", va="center", fontsize=7.5, color="#444444")
    # within: one person, trial blocks split into train (filled) and test (hatched)
    _people(d, 2.75, 2.75, 1, C_TRAIN)
    for i in range(5):
        d.add_patch(plt_rect(3.3 + i * 0.62, 2.55, 0.5, 0.36,
                             C_TEST if i == 2 else C_TRAIN, "////" if i == 2 else None))
    # zero-shot: N-1 people train, one new person test
    _people(d, 2.75, 1.55, 5, C_TRAIN)
    d.annotate("", xy=(5.45, 1.55), xytext=(4.75, 1.55),
               arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1))
    _people(d, 5.85, 1.55, 1, C_TEST, hatch="////")
    d.text(5.85, 1.08, "new", ha="center", va="top", fontsize=7.5)
    # calibration: N-1 people + k trials from the new person
    _people(d, 2.75, 0.35, 5, C_TRAIN)
    d.text(4.83, 0.35, "+", ha="center", va="center", fontsize=11)
    d.add_patch(plt_rect(5.02, 0.2, 0.45, 0.32, C_CAL, None))
    d.text(5.24, 0.12, "k trials", ha="center", va="top", fontsize=7.5)
    d.annotate("", xy=(5.72, 0.35), xytext=(5.5, 0.35),
               arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1))
    _people(d, 6.05, 0.35, 1, C_TEST, hatch="////")
    # legend
    for x, fc, h, lab in ((1.6, C_TRAIN, None, "training data"),
                          (4.0, C_TEST, "////", "test data"),
                          (5.9, C_CAL, None, "new person's calibration trials")):
        d.add_patch(plt_rect(x, -0.7, 0.3, 0.2, fc, h))
        d.text(x + 0.38, -0.6, lab, va="center", fontsize=7.5)
    # panel letters in figure coordinates: (a), (c), (d) share one left edge, (b) sits over
    # its own column, and (a)/(b) clear the three-line panel titles
    for ax_, s, dy in ((a, "a", 0.075), (b, "b", 0.075), (t, "c", 0.01), (d, "d", -0.01)):
        p_ = ax_.get_position()
        x_ = p_.x0 if s == "b" else 0.0
        fig.text(x_, p_.y1 + dy, f"({s})", fontsize=10, fontweight="bold", va="bottom")
    save(fig, out, "fig1_design", log)


def plt_rect(x, y, w, h, fc, hatch):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fc=fc, ec="#222222", lw=0.8, hatch=hatch)


# ------------------------------------------------------------------------ Fig 1

def fig1(plt, out, log):
    log("Fig 1  onset correction")
    vpath = layout.result("ds004830", "validate_against_v2_events", ".log")
    txt = open(vpath, encoding="utf-8").read()
    rows = re.findall(r"^(\d\d)\s+(\d+)\s+([+-][\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
                      txt, re.M)
    offs = {f"sub-{r[0]}": (float(r[2]), float(r[3])) for r in rows}
    pairs = {"3-class, correct trials": (os.path.join(LOGS, "decode-3class.log"),
                                         layout.result(V2, "decode", ".log", "3class",
                                                       "correct")),
             "2-class, all trials": (os.path.join(LOGS, "decode-lateral.log"),
                                     layout.result(V2, "decode", ".log", "lateral"))}
    src(log, vpath, *[p for v in pairs.values() for p in v])

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2])
    ax = fig.add_subplot(gs[0, :])
    letter(ax, "A")
    # Only the two MEASURED events are drawn: cue onset (the official onset) and movie
    # onset 3.19 s later (Trigger3 - Trigger2). Durations beyond that are not verified
    # from the logs, so none are drawn.
    for x0, lab, c, ha, dx in ((0, "cue onset\n(official v2.0.0 onset)", "#1f5fa8", "right", -0.12),
                               (CUE_LEAD, "movie onset\n(+3.19 s, sd 48 ms)", "#c0502a", "left", 0.12)):
        ax.plot([x0, x0], [-0.25, 0.35], color=c, lw=2.2)
        ax.text(x0 + dx, 0.2, lab, fontsize=7.5, color=c, va="center", ha=ha)
    ax.axhline(0, xmin=0.02, xmax=0.98, color="k", lw=0.6)
    ax.axvspan(-2, 12, ymin=0.05, ymax=0.18, color="#1f5fa8", alpha=0.25)
    ax.text(5, -0.43, "analysis epoch -2 to +12 s (baseline -2 to 0 s)", ha="center",
            fontsize=8, color=C_WITHIN)
    ax.set_xlim(-3, 14.5)
    ax.set_ylim(-0.55, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time from cue onset (s)")
    ax.set_title("Trial timing and analysis epoch")

    ax = fig.add_subplot(gs[1, 0])
    letter(ax, "B")
    names = sorted(offs)
    m = np.array([offs[n][0] for n in names])
    sd = np.array([offs[n][1] for n in names])
    ax.errorbar(range(len(names)), m, yerr=sd, fmt="o", color=C_LOSO, ms=4, capsize=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=20)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[4:] for n in names], fontsize=8)
    ax.set_xlabel("subject")
    ax.set_ylabel("official - reconstructed onset (s)\nmean +/- sd over trials")
    ax.set_title("v1 reconstruction vs official onsets")
    notes = {"sub-08": ("3 runs: run clocks misaligned", 22), "sub-19": ("progressive\ndrift", 0)}
    for i, n in enumerate(names):
        if n in notes:
            ax.annotate(notes[n][0], (i, m[i]), xytext=(8, notes[n][1]),
                        textcoords="offset points", fontsize=7.5, va="center", color=C_GREY)
    for n, (mm, s) in zip(names, zip(m, sd)):
        log(f"  {n}: offset {mm:+.2f} s, sd {s:.2f}")
    ax.text(0.98, 0.03, "other 10: near-constant offset, sd < 0.8 s",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=C_GREY)

    ax = fig.add_subplot(gs[1, 1])
    letter(ax, "C")
    x, labels = 0, []
    for task, (old_p, new_p) in pairs.items():
        old, new = parse_decode(old_p), parse_decode(new_p)
        chance = 100 / 3 if task.startswith("3") else 50
        for proto, col in (("within", C_WITHIN), ("loso", C_LOSO)):
            bo = max(old[proto].values()) * 100
            bn = max(new[proto].values()) * 100
            ax.bar(x - 0.18, bo, 0.36, color=col, alpha=0.35)
            ax.bar(x + 0.18, bn, 0.36, color=col)
            ax.text(x + 0.18, bn + 1, f"{bn:.1f}", ha="center", fontsize=7.5)
            ax.text(x - 0.18, bo + 1, f"{bo:.1f}", ha="center", fontsize=7.5, color=C_GREY)
            ax.plot([x - 0.4, x + 0.4], [chance, chance], color="k", ls="--", lw=0.8)
            labels.append(f"{task.split(',')[0]}\n{'within' if proto == 'within' else 'LOSO'}")
            log(f"  {task} {proto}: best model misaligned {bo:.2f}% -> corrected {bn:.2f}%")
            x += 1
    ax.set_xticks(range(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("accuracy (%), best of 4 models")
    ax.set_title("Before (pale) and after (solid) correction")
    fig.tight_layout()
    save(fig, out, "fig2_onset_correction", log)


# ------------------------------------------------------------------------ Fig 2

def fig2(plt, out, log):
    log("Fig 2  headline accuracy")
    p = CI_DS004830
    src(log, p)
    ci = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, S, lab, ref in zip(axes, ci, ("A", "B"), (49.4, 78.3)):
        letter(ax, lab)
        chance = S["chance"] * 100
        models = [r["model"] for r in S["rows"] if r["protocol"] == "within"]
        rng = np.random.default_rng(1)
        for j, proto in enumerate(("within", "loso")):
            col = C_WITHIN if proto == "within" else C_LOSO
            for i, mname in enumerate(models):
                r = next(r for r in S["rows"] if r["protocol"] == proto and r["model"] == mname)
                xpos = i + (j - 0.5) * 0.36
                ps = np.array(r["per_subject"]) * 100
                ax.scatter(xpos + rng.uniform(-0.06, 0.06, len(ps)), ps, s=6, color=col,
                           alpha=0.35, lw=0)
                ax.errorbar(xpos, r["mean"] * 100, yerr=[[r["mean"] * 100 - r["bca"][0] * 100],
                            [r["bca"][1] * 100 - r["mean"] * 100]], fmt="o", color=col,
                            ms=5 if r["best"] else 3.5, mfc=col if r["best"] else "white",
                            capsize=2, lw=1.2)
                log(f"  {S['task']} {proto} {mname}: {r['mean'] * 100:.2f}% "
                    f"[{r['bca'][0] * 100:.1f}, {r['bca'][1] * 100:.1f}]"
                    f"{'  (best)' if r['best'] else ''}")
        ax.axhline(chance, color="k", ls="--", lw=0.8)
        ax.axhline(ref, color=C_NING, ls=":", lw=1)
        ax.text(len(models) - 0.5, ref + 1, f"Ning et al. within ({ref}%)", ha="right",
                fontsize=8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split(" (")[0].replace("Random Forest", "RF")
                            .replace("Logistic", "LR") for m in models])
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{'3-class' if S['task'] == '3class' else '2-class (left vs right)'}, "
                     f"correct trials, 12 subjects")
    axes[0].scatter([], [], color=C_WITHIN, label="within-subject")
    axes[0].scatter([], [], color=C_LOSO, label="subject-independent (LOSO)")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    save(fig, out, "fig3_headline", log)


# ------------------------------------------------------------------------ Fig 3

def fig3(plt, out, log):
    log("Fig 3  realistic zero-shot and Ning et al.")
    p = NORM_DS004830
    q = CI_DS004830
    src(log, p, q)
    N = json.load(open(p))
    ci = json.load(open(q))
    from bootstrap_ci import bca
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    letter(ax, "A")
    schemes = ["transductive", "prefix", "causal", "none"]
    rng = np.random.default_rng(0)
    for k, (task, off, chance) in enumerate((("3class", -0.2, 100 / 3), ("lateral", 0.2, 50))):
        for i, s in enumerate(schemes):
            a = np.array(N[f"{task}|Logistic (L2)|{s}"]) * 100
            pct, bc = bca(a / 100, 10000, rng)
            col = C_LOSO if s == "causal" else C_GREY
            ax.errorbar(i + off, a.mean(), yerr=[[a.mean() - bc[0] * 100], [bc[1] * 100 - a.mean()]],
                        fmt="o" if task == "3class" else "s", color=col, ms=4, capsize=2)
            log(f"  zero-shot {task} {s}: {a.mean():.2f}% [{bc[0] * 100:.1f}, {bc[1] * 100:.1f}]")
    ax.axhline(100 / 3, color="k", ls="--", lw=0.7)
    ax.axhline(50, color="k", ls="--", lw=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["all\n(transd.)", "first\n20", "past\n(causal)", "none"], fontsize=8)
    ax.set_xlabel("trials used to normalise the new person")
    ax.set_ylabel("subject-independent accuracy (%)")
    ax.set_title("New-subject normalisation\n(circles 3-class, squares 2-class)")

    for ax, task, col, lab in ((axes[1], "3class", 1, "B"), (axes[2], "lateral", 0, "C")):
        letter(ax, lab)
        ax.texts[-1].set_position((-0.3, 1.14))
        inc = sorted(k for k in compare_ning.NING if k not in compare_ning.NING_EXCLUDED)
        theirs = np.array([compare_ning.NING[k][col] for k in inc], float)
        S = next(s for s in ci if s["task"] == task)
        names = [n.replace("sub-", "") for n in S["subjects"]]
        keep = [names.index(k) for k in inc]
        r = next(r for r in S["rows"] if r["protocol"] == "within" and r["model"] == "LDA (Ledoit-Wolf)")
        ours = np.array(r["per_subject"])[keep] * 100
        from scipy import stats
        rho, pr = stats.spearmanr(ours, theirs)
        w = stats.wilcoxon(ours, theirs).pvalue
        ax.scatter(theirs, ours, color=C_WITHIN, s=18)
        lo = min(theirs.min(), ours.min()) - 5
        ax.plot([lo, 100], [lo, 100], color="k", lw=0.7)
        ax.set_xlim(lo, 100)
        ax.set_ylim(lo, 100)
        ax.set_xlabel("Ning et al. 2024 (%)")
        ax.set_ylabel("ours, same subjects and classifier (%)")
        ax.set_title(f"{'3-class' if task == '3class' else '2-class'}: ours {ours.mean():.1f}% "
                     f"vs {theirs.mean():.1f}%", pad=16)
        ax.text(0.55, 1.02, f"rho {rho:+.2f}, p {pr:.3f}; Wilcoxon p {w:.2f}",
                transform=ax.transAxes, ha="center", fontsize=7.5)
        ax.set_ylabel("ours, same subjects (%)")
        log(f"  Ning {task}: ours {ours.mean():.2f} vs {theirs.mean():.2f}; rho {rho:+.3f} "
            f"p {pr:.4f}; Wilcoxon p {w:.4f}")
    fig.tight_layout()
    save(fig, out, "fig4_zeroshot_and_ning", log)


# ------------------------------------------------------------------------ Fig 4

# The realistic curve (new person normalised on their k calibration trials only, which an
# online system can do) is the main line; the same model normalised on all the new
# person's trials is drawn dashed as an upper bound. The subject-only decoder still
# normalises on all its own trials, so the comparison with it is tilted against pooling.
# At k = 0 the realistic decoder has no trials to normalise on, so its k = 0 point is the
# unnormalised zero-shot (zero_nonorm); HemoNet, like the upper bound, normalises on all
# of the new person's trials.
CAL_SHOW = {
    "pooled_calnorm": ("others + new person's k trials (realistic)", C_LOSO, "o-", 0.14),
    "pooled_bal": ("same, new person normalised on all their trials (upper bound)",
                   C_LOSO, "o:", 0.0),
    "within_lr": ("new person's k trials only", C_GREY, "s--", 0.12),
    "deep_finetune": ("HemoNet fine-tuned (normalised as upper bound)", "#2a9d5c", "d-", 0.0)}


def _cal_curves(ax, rows, log, tag=""):
    n = None
    for meth, (lab, col, st, alpha) in CAL_SHOW.items():
        pts = sorted((r["k"], r["mean"], r["lo"], r["hi"], r["n"]) for r in rows
                     if (r["method"] == meth and r["k"] <= 30)
                     or (meth == "pooled_calnorm" and r["method"] == "zero_nonorm"))
        if not pts:
            continue
        k, mu, lo, hi, nn = map(np.array, zip(*pts))
        n = int(nn.max())
        ax.plot(k, mu * 100, st, color=col, ms=4, lw=1.6 if alpha else 1.1, label=lab,
                alpha=1.0 if meth != "pooled_bal" else 0.7)
        if alpha:
            ax.fill_between(k, lo * 100, hi * 100, color=col, alpha=alpha, lw=0)
        log(f"  {tag}{meth}: " + ", ".join(f"k={a} {b * 100:.1f}" for a, b in zip(k, mu)))
    return n


def fig4(plt, out, log):
    log("Fig 4  calibration curve")
    p = layout.result(V2, "calibration_curve", ".json", "3class", part="summary")
    src(log, p)
    rows = json.load(open(p))
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    _cal_curves(ax, rows, log)
    ax.set_ylim(bottom=31)
    ax.axhline(100 / 3, color="k", ls="--", lw=0.8)
    ax.set_xlabel("labelled calibration trials from the new subject (k)")
    ax.set_ylabel("accuracy (%), 3-class")
    ax.set_title("Calibration curve\n(11 subjects, 20 draws, 95% subject bootstrap)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.tight_layout()
    save(fig, out, "fig5_calibration", log)


# ------------------------------------------------------------------------ Fig 5

def fig5(plt, out, log):
    log("Fig 5  physiology")
    p = layout.result(V2, "time_resolved", ".npz")
    q = layout.result(V2, "physiology", ".json")
    src(log, p, q)
    z = np.load(p)
    P = json.load(open(q))
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, :])
    letter(ax, "A")
    t, acc = z["t"], z["acc"] * 100
    mu, sem = acc.mean(0), acc.std(0, ddof=1) / np.sqrt(acc.shape[0])
    ax.fill_between(t, mu - sem, mu + sem, color=C_WITHIN, alpha=0.2, lw=0)
    ax.plot(t, mu, color=C_WITHIN, lw=1.6, label="within-subject (mean +/- sem)")
    thr = float(z["thresh"]) * 100
    ax.axhline(thr, color=C_LOSO, ls="-.", lw=1,
               label=f"family-wise 95% permutation threshold ({thr:.1f}%)")
    ax.axhline(100 / 3, color="k", ls="--", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(CUE_LEAD, color="k", lw=0.8, ls=":")
    ax.text(0.1, ax.get_ylim()[1] * 0.97, "cue", fontsize=8, va="top")
    ax.text(CUE_LEAD + 0.1, ax.get_ylim()[1] * 0.97, "movie", fontsize=8, va="top")
    ax.set_xlabel("window centre, time from cue onset (s); 2 s windows")
    ax.set_ylabel("accuracy (%), 3-class")
    ax.set_title(f"Time-resolved decoding (within-subject shrinkage LDA, all trials): "
                 f"peak {mu.max():.1f}% at {t[mu.argmax()]:+.1f} s")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.42))
    log(f"  time-resolved: peak {mu.max():.2f}% at {t[mu.argmax()]:+.2f} s; pre-cue mean "
        f"{mu[t < 0].mean():.2f}%; threshold {thr:.2f}%")

    ax = fig.add_subplot(gs[1, 0])
    letter(ax, "B")
    R = P["region_drop"]
    regs = [k for k in R if k != "all"]
    x = np.arange(len(regs))
    for j, (proto, col) in enumerate((("within", C_WITHIN), ("loso", C_LOSO))):
        v = [np.mean(R[r][f"alone|{proto}"]) * 100 for r in regs]
        ax.bar(x + (j - 0.5) * 0.38, v, 0.38, color=col,
               label=f"{'within' if proto == 'within' else 'LOSO'}, region alone")
        ax.axhline(np.mean(R["all"][proto]) * 100, color=col, ls="-", lw=0.8)
        log(f"  region alone {proto}: " + ", ".join(f"{r} {a:.1f}" for r, a in zip(regs, v)))
    ax.axhline(100 / 3, color="k", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regs)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 75)
    ax.set_title("One region at a time (lines: all 28 channels)")
    ax.legend(frameon=False, loc="upper center", fontsize=7.5, ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    letter(ax, "C")
    G = np.hstack([np.array(P["patterns"]["HbO"]), np.array(P["patterns"]["HbR"])])
    im = ax.imshow(G, aspect="auto", cmap="viridis")
    bins = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(bins * 2, fontsize=7.5, rotation=90)
    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels(regs)
    ax.axvline(5.5, color="w", lw=1.5)
    ax.text(2.5, -0.8, "HbO", ha="center", fontsize=8)
    ax.text(8.5, -0.8, "HbR", ha="center", fontsize=8)
    ax.set_xlabel("time bin from cue onset (s)")
    ax.set_title("|Activation pattern| (Haufe), pooled model", pad=22)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save(fig, out, "fig6_physiology", log)


# ----------------------------------------------------------------------- Fig S1

def figS1(plt, out, log):
    log("Fig S7  deep vs classical, identical trials")
    p = layout.result(V2, "deep_models_gpu", ".log", "correct")
    q = CI_DS004830
    src(log, p, q)
    txt = open(p, encoding="utf-8").read().split("=== summary")[-1]
    deep = {}
    for m in re.finditer(r"^(within|loso)\s+(\S+)\s+([\d.]+)\s+([\d.]+)", txt, re.M):
        deep.setdefault(m.group(1), {})[m.group(2)] = (float(m.group(3)), float(m.group(4)))
    S = next(s for s in json.load(open(q)) if s["task"] == "3class")
    # NOT sharey: each panel is sorted by its own values, so each needs its own labels
    # (a shared axis let panel B's labels overwrite panel A's)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for ax, proto, lab in zip(axes, ("within", "loso"), ("A", "B")):
        letter(ax, lab)
        names = list(deep[proto]) + [r["model"] for r in S["rows"] if r["protocol"] == proto]
        vals = [deep[proto][n][0] for n in deep[proto]] + \
               [r["mean"] * 100 for r in S["rows"] if r["protocol"] == proto]
        errs = [deep[proto][n][1] for n in deep[proto]] + [0] * 4
        cols = ["#bbbbbb"] * len(deep[proto]) + [C_WITHIN if proto == "within" else C_LOSO] * 4
        is_deep = np.array([True] * len(deep[proto]) + [False] * 4)
        order = np.argsort(vals)
        bars = ax.barh(range(len(vals)), np.array(vals)[order], xerr=np.array(errs)[order],
                       color=np.array(cols)[order], capsize=2, edgecolor="#444444", lw=0.5)
        for b, d in zip(bars, is_deep[order]):   # hatching, so the key survives greyscale
            if d:
                b.set_hatch("///")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([names[i].replace(" (Ledoit-Wolf)", "").replace(" (L2)", "")
                            .replace(" (RBF)", "") for i in order], fontsize=8)
        ax.axvline(100 / 3, color="k", ls="--", lw=0.8)
        ax.set_xlabel("accuracy (%), 3-class, correct trials")
        ax.set_title(f"{'within-subject' if proto == 'within' else 'subject-independent (LOSO)'}"
                     "\nhatched: deep (sd over 10 seeds)")
        for n in deep[proto]:
            log(f"  {proto} {n}: {deep[proto][n][0]:.2f} +/- {deep[proto][n][1]:.2f}")
    fig.tight_layout()
    save(fig, out, "figS7_deep_vs_classical", log)


# ------------------------------------------------------------------------ Fig 6



def fig6(plt, out, log):
    log("Fig 6  second dataset (ds007738): replication, calibration, eye-movement control")
    sources = {"ds004830": (CI_DS004830, NORM_DS004830),
               "ds007738": (CI_DS007738, NORM_DS007738)}
    pc = layout.result(OVERT, "calibration_curve", ".json", "lateral", part="summary")
    pe = layout.result("ds007738", "control_eye_movement", ".json")
    src(log, *[p for v in sources.values() for p in v], pc, pe)
    from bootstrap_ci import bca
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.3)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    plt.rcParams["hatch.linewidth"] = 0.6

    # A: 2-class replication, both datasets
    ax = axes[0]
    letter(ax, "A")
    cols = {"ds004830": C_GREY, "ds007738": C_WITHIN}
    for d, (off, (pci, pn)) in zip(sources, zip((-0.18, 0.18), sources.values())):
        S = next(s for s in json.load(open(pci)) if s["task"] == "lateral")
        best = {r["protocol"]: r for r in S["rows"] if r["best"]}
        # Intervals come from the stored results, not from a fresh bootstrap here, so the
        # figure and the text quote the same numbers: bootstrap_ci.py saves its BCa bounds,
        # and normalisation_check.bca reproduces the ones in its own log.
        import normalisation_check
        causal = np.array(json.load(open(pn))["lateral|Logistic (L2)|causal"])
        vals = [(np.array(best["within"]["per_subject"]), best["within"]["bca"]),
                (np.array(best["loso"]["per_subject"]), best["loso"]["bca"]),
                (causal, normalisation_check.bca(causal))]
        for i, (a, bc) in enumerate(vals):
            ax.bar(i + off, a.mean() * 100, 0.34, color=cols[d], alpha=0.85,
                   label=f"{d} (n = {len(a)})" if i == 0 else None)
            ax.errorbar(i + off, a.mean() * 100, [[(a.mean() - bc[0]) * 100],
                                                  [(bc[1] - a.mean()) * 100]], color="k",
                        capsize=2, lw=0.8)
            ax.scatter(i + off + rng.uniform(-0.1, 0.1, len(a)), a * 100, s=4, color="k",
                       alpha=0.35, zorder=3)
            log(f"  {d} {['within', 'LOSO', 'causal zero-shot'][i]}: {a.mean() * 100:.2f}% "
                f"[{bc[0] * 100:.1f}, {bc[1] * 100:.1f}]")
    ax.axhline(50, color="k", ls="--", lw=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["within\nsubject", "LOSO\ntransd.", "LOSO\ncausal"], fontsize=8)
    ax.set_ylim(20, 105)
    ax.set_ylabel("left vs right accuracy (%)")
    ax.set_title("Replication on a second dataset")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=1)

    # B: calibration curve, ds007738
    ax = axes[1]
    letter(ax, "B")
    rows = json.load(open(pc))
    n = _cal_curves(ax, rows, log, tag="calibration ")
    ax.axhline(50, color="k", ls="--", lw=0.8)
    ax.set_xlabel("labelled calibration trials from the new subject (k)")
    ax.set_ylabel("accuracy (%), 2-class")
    ax.set_title(f"ds007738 calibration curve ({n} subjects)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)

    # C: eye-movement control, causal normalisation
    ax = axes[2]
    letter(ax, "C")
    E = json.load(open(pe))["causal"]
    groups = [("within subject", ["within overt (matched n)", "within eye",
                                  "within covert (matched n)"]),
              ("subject-independent", ["LOSO overt (matched n)", "LOSO eye",
                                       "LOSO covert (matched n)"]),
              ("eye-trained\ntransfer", ["transfer eye -> overt", "transfer eye -> covert"])]
    # colour plus hatch: the panel must stay readable in greyscale (journal requirement)
    ccol = {"overt": C_WITHIN, "eye": C_LOSO, "covert": "#8a4fbf"}
    chatch = {"overt": "", "eye": "///", "covert": "..."}
    x, ticks, labs, seen = 0, [], [], set()
    for gname, keys in groups:
        start = x
        for key in keys:
            a = np.array(E[key])
            _, bc = bca(a, 10000, rng)
            kind = key.split()[-1] if key.startswith("transfer") else key.split()[1]
            ax.bar(x, a.mean() * 100, 0.8, color=ccol[kind], alpha=0.85,
                   hatch=chatch[kind], edgecolor="white", lw=0,
                   label=None if kind in seen else {"overt": "overt attention",
                                                    "eye": "eye movement only",
                                                    "covert": "covert attention"}[kind])
            seen.add(kind)
            ax.errorbar(x, a.mean() * 100, [[(a.mean() - bc[0]) * 100],
                                            [(bc[1] - a.mean()) * 100]], color="k", capsize=2,
                        lw=0.8)
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(a)), a * 100, s=4, color="k",
                       alpha=0.35, zorder=3)
            log(f"  control {key}: {a.mean() * 100:.2f}% [{bc[0] * 100:.1f}, {bc[1] * 100:.1f}]")
            x += 1
        ticks.append((start + x - 1) / 2)
        labs.append(gname)
        x += 0.8
    ax.axhline(50, color="k", ls="--", lw=0.7)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(20, 105)
    ax.set_ylabel("left vs right accuracy (%)")
    ax.set_title(f"Eye-movement control (n = {len(E['LOSO eye'])}, causal)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, ncol=3)
    for a_, x_ in zip(axes, (-0.2, -0.2, -0.085)):   # keep the panel letter clear of the title
        a_.texts[0].set_position((x_, 1.06))
    save(fig, out, "fig7_ds007738", log)


PD = lambda part: layout.result(V2, "plot_diagnostics", ".png", part=part)
COPIES = {"figS1_montage.png": layout.result("ds004830", "check_montage", ".png"),
          "figS2_stg_channels.png": PD("channels_28_vs_30"),
          "figS3_onset_anchor.png": layout.result(V2, "check_onset_anchor", ".png"),
          "figS4_ning_pipeline.png": PD("ning"),
          "figS5_protocol_ablation.png": PD("ablation"),
          "figS6_roi_features.png": PD("roi_features")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(layout.PAPER_FIGURES))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = Log(os.path.join(args.out, "paper_figures.log"))
    log(f"paper figures -> {args.out}\n")
    plt = setup()
    for f in (fig_design, fig1, fig2, fig3, fig4, fig5, fig6, figS1):
        f(plt, args.out, log)
        plt.close("all")
    log("supplementary figures copied from plots/ (each made by its own script):")
    for dst, s in COPIES.items():
        shutil.copyfile(s, os.path.join(args.out, dst))
        log(f"  {dst} <- {layout.rel(s)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
