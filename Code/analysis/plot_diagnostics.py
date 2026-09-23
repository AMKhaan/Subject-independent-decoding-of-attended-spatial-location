"""
Figures for the 2026-09-18 analyses that produced logs but no graph.

Why this exists
---------------
Three analyses ended in a log only: the protocol ablation re-run on corrected onsets,
ROI-pooled against channel features, and the 28- against 30-channel montage (STG
recovered). Every number drawn here is PARSED from those logs or their JSON, never
typed in, so re-running an analysis and then this script keeps figure and log in
agreement. (fig_ablation.py and fig_subjects.py carry hard-coded literals and have
already drifted from the tables they illustrate; that is the failure this avoids.)

Inputs (under results/ds004830/ unless noted):
  ablation   Code/analysis/logs/leakage+decode-3class+decode-lateral.log (misaligned onsets)
             leakage_demo/ds004830_v2_leakage_demo.log (corrected onsets)
  roi        roi_features/ds004830_v2_roi_features.json
  channels   decode/ds004830_v2_3class_correct_decode.log (28 ch) and
             decode/ds004830_v2_30ch_3class_correct_decode.log (30 ch)
  ning       ning_decode/ds004830_ning_v2_<anchor>_<task>_ning_decode.log

Outputs: plots/ds004830/plot_diagnostics/ds004830_v2_plot_diagnostics_<part>.png for
part = ablation, roi_features, channels_28_vs_30, ning, and
results/ds004830/plot_diagnostics/ds004830_v2_plot_diagnostics.log listing every value
used, with its source file (see layout.py).

Usage:  python plot_diagnostics.py [--only ablation roi channels]
"""

import argparse
import json
import os
import re
import sys

import numpy as np

import layout

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
V2 = layout.derived("ds004830", "v2")
RES = lambda ext, part="": layout.result(V2, "plot_diagnostics", ext, part=part)
DECODE = lambda variant, task: layout.result(layout.derived("ds004830", variant), "decode",
                                             ".log", task, "correct")


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def parse_ablation(path):
    """Last complete 'inflation' table in a leakage_demo log -> {tag: accuracy}."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    block = txt.split("=== inflation over the honest protocol ===")[-1]
    out = {}
    for m in re.finditer(r"^\s+([ABCD])\s+(.+?)\s+([\d.]+)%", block, re.M):
        out[m.group(1)] = (m.group(2).strip(), float(m.group(3)) / 100)
    return out


def parse_decode(path):
    """decode.py log -> means per (protocol, model), per-subject best, perm p-values."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    res = {"within": {}, "loso": {}, "subjects": [], "p": {}}
    parts = re.split(r"=== (within-subject|leave-one-subject-out)", txt)
    for i in range(1, len(parts), 2):
        proto = "within" if parts[i].startswith("within") else "loso"
        for m in re.finditer(r"^\s+(.+?)\s+mean ([\d.]+)%", parts[i + 1].split("===")[1]
                             if parts[i + 1].count("===") else parts[i + 1], re.M):
            res[proto][m.group(1).strip()] = float(m.group(2)) / 100
    # header is "=== per-subject accuracy, best model ... ===", so the table is the
    # text after the header's closing "===" and before the next section
    tab = txt.split("=== per-subject accuracy")[1].split("===")[1]
    for m in re.finditer(r"^\s+(sub-\d+)\s+([\d.]+)%\s+([\d.]+)%", tab, re.M):
        res["subjects"].append((m.group(1), float(m.group(2)) / 100, float(m.group(3)) / 100))
    for m in re.finditer(r"(within|loso) / (.+?): observed.*?p = ([\d.]+)", txt, re.S):
        res["p"][m.group(1)] = (m.group(2), float(m.group(3)))
    return res


def ablation(log):
    old_p = os.path.join(LOGS, "leakage+decode-3class+decode-lateral.log")
    new_p = layout.result(V2, "leakage_demo", ".log")
    old, new = parse_ablation(old_p), parse_ablation(new_p)
    log(f"ablation  old <- {old_p}\n          new <- {new_p}")
    for t in "ABCD":
        log(f"  {t}  {old[t][0]:<44} misaligned {old[t][1] * 100:6.2f}%   "
            f"corrected {new[t][1] * 100:6.2f}%")
    plt = plt_()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(4)
    w = 0.38
    for off, data, lab, col in ((-w / 2, old, "misaligned onsets (v1 reconstruction)", "#bbbbbb"),
                                (w / 2, new, "corrected onsets (v2.0.0 events)", "#1f5fa8")):
        v = [data[t][1] * 100 for t in "ABCD"]
        bars = ax.bar(x + off, v, w, color=col, label=lab)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.8, f"{val:.1f}", ha="center",
                    fontsize=8)
    ax.axhline(100 / 3, color="k", ls="--", lw=1)
    ax.text(3.45, 100 / 3 + 0.8, "chance", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["A  leave-one-\nsubject-out", "B  random split\nover trials",
                        "C  random split\nover windows", "D  channel-as-sample\n(label bug)"],
                       fontsize=8)
    ax.set_ylabel("3-class accuracy (%), all trials")
    ax.set_ylim(0, 90)
    ax.set_title("Protocol ablation before and after onset correction\n"
                 f"inflation of D over A: {(old['D'][1] - old['A'][1]) * 100:+.1f} pp -> "
                 f"{(new['D'][1] - new['A'][1]) * 100:+.1f} pp; D is onset-independent",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES(".png", "ablation"), dpi=150)
    log("  -> ablation-v2.png\n")


def roi(log):
    p = layout.result(V2, "roi_features", ".json")
    S = json.load(open(p))
    log(f"roi <- {p}")
    plt = plt_()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, task in zip(axes, ("3class", "lateral")):
        chance = 1 / 3 if task == "3class" else 1 / 2
        keys = list(S[task])
        labels, ch, ro = [], [], []
        for k in keys:
            d = S[task][k]
            c, r = np.array(d["channel (336)"]), np.array(d["ROI (48)"])
            labels.append(k.replace("|", "\n").replace(" (Ledoit-Wolf)", "").replace(
                " (L2)", "").replace(" (RBF)", "").replace("Random Forest", "RF").replace("Logistic", "LR"))
            ch.append(c)
            ro.append(r)
            log(f"  {task:<8}{k:<30} channel {c.mean() * 100:6.2f}%  ROI "
                f"{r.mean() * 100:6.2f}%  diff {(r - c).mean() * 100:+6.2f}")
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [c.mean() * 100 for c in ch], 0.4, color="#1f5fa8",
               label="28 channels (336 features)")
        ax.bar(x + 0.2, [r.mean() * 100 for r in ro], 0.4, color="#2a9d5c",
               label="4 ROIs (48 features)")
        for i, (c, r) in enumerate(zip(ch, ro)):
            for a, b in zip(c, r):
                ax.plot([i - 0.2, i + 0.2], [a * 100, b * 100], color="k", lw=0.4, alpha=0.25)
        ax.axhline(chance * 100, color="k", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("accuracy (%), correct trials")
        ax.set_title(f"{task}: ROI pooling vs channel features (lines = subjects)",
                     fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RES(".png", "roi_features"), dpi=150)
    log("  -> roi-features.png\n")


def channels(log):
    p28 = DECODE("v2", "3class")
    p30 = DECODE("v2_30ch", "3class")
    a, b = parse_decode(p28), parse_decode(p30)
    log(f"channels  28 <- {p28}\n          30 <- {p30}")
    for proto in ("within", "loso"):
        for m in a[proto]:
            log(f"  {proto:<7}{m:<20} 28ch {a[proto][m] * 100:6.2f}%   "
                f"30ch {b[proto][m] * 100:6.2f}%   diff {(b[proto][m] - a[proto][m]) * 100:+.2f}")
    log(f"  permutation (best model): 28ch {a['p']}  30ch {b['p']}")
    if not a["subjects"] or not b["subjects"]:
        raise ValueError("per-subject table not parsed; check the decode log format")
    for (s, w28, l28), (_, w30, l30) in zip(a["subjects"], b["subjects"]):
        log(f"  {s}  within {w28 * 100:6.2f} -> {w30 * 100:6.2f}   "
            f"LOSO {l28 * 100:6.2f} -> {l30 * 100:6.2f}")
    plt = plt_()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    models = list(a["within"])
    x = np.arange(len(models))
    for j, proto in enumerate(("within", "loso")):
        ax.bar(x + (j * 2 - 1.5) * 0.2, [a[proto][m] * 100 for m in models], 0.2,
               color=["#1f5fa8", "#c0502a"][j], alpha=0.5, label=f"{proto}, 28 ch")
        ax.bar(x + (j * 2 - 0.5) * 0.2, [b[proto][m] * 100 for m in models], 0.2,
               color=["#1f5fa8", "#c0502a"][j], label=f"{proto}, 30 ch (+STG)")
    ax.axhline(100 / 3, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(" (")[0] for m in models], fontsize=8)
    ax.set_ylabel("3-class accuracy (%), correct trials")
    ax.set_title("Group mean by model", fontsize=9)
    ax.set_ylim(0, 75)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center")
    ax = axes[1]
    s28 = {s: (w, l) for s, w, l in a["subjects"]}
    s30 = {s: (w, l) for s, w, l in b["subjects"]}
    for s in s28:
        ax.plot([0, 1], [s28[s][0] * 100, s30[s][0] * 100], "o-", color="#1f5fa8",
                alpha=0.5, ms=3)
        ax.plot([2, 3], [s28[s][1] * 100, s30[s][1] * 100], "o-", color="#c0502a",
                alpha=0.5, ms=3)
    ax.axhline(100 / 3, color="k", ls="--", lw=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["within\n28", "within\n30", "LOSO\n28", "LOSO\n30"], fontsize=8)
    short = lambda n: n.split(" (")[0]
    ax.set_title("Per subject, best model of each run (models differ between runs)\n"
                 f"within: 28 = {short(a['p']['within'][0])}, 30 = {short(b['p']['within'][0])};"
                 f"  LOSO: 28 = {short(a['p']['loso'][0])}, 30 = {short(b['p']['loso'][0])}",
                 fontsize=8)
    fig.suptitle("Recovering the two STG channels (position-matched montage)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES(".png", "channels_28_vs_30"), dpi=150)
    log("  -> channels-28-vs-30.png\n")


def parse_ning(path):
    """ning_decode.py log -> windows, per-subject rows, 'included' mean row."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    head = re.search(r"subject\s+trials\s+(.+?)\s+note", txt)
    wins = [float(w.rstrip("s")) for w in head.group(1).split()]
    subs = {}
    for m in re.finditer(r"^\s+(sub-\d+)\s+(\d+)\s+([\d.\s]+?)\s*(EXCLUDED.*|behaviour.*|pruned.*)?$",
                         txt, re.M):
        vals = [float(v) for v in m.group(3).split()][:len(wins)]
        if len(vals) == len(wins):
            subs[m.group(1)] = (vals, bool(m.group(4)))
    inc = re.search(r"^\s+included\s+(\d+)\s+([\d.\s]+)$", txt, re.M)
    mean = [float(v) for v in inc.group(2).split()] if inc else None
    return wins, subs, mean


def ning(log):
    """Ning et al.'s own feature path on corrected timing, both onset anchors."""
    import compare_ning
    plt = plt_()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, task, col, ref in ((axes[0], "3class", 1, 49.4), (axes[1], "2class", 0, 78.3)):
        chance = 100 / 3 if task == "3class" else 50
        inc = sorted(k for k in compare_ning.NING if k not in compare_ning.NING_EXCLUDED)
        theirs = np.array([compare_ning.NING[k][col] for k in inc], float)
        for anchor, c in (("movie", "#1f5fa8"), ("cue", "#c0502a")):
            p = layout.result(layout.derived("ds004830", f"ning_v2_{anchor}"), "ning_decode",
                              ".log", task)
            wins, subs, mean = parse_ning(p)
            if mean is None:
                raise FileNotFoundError(p)
            lab = ("window from official onset - 3.19 s (onset = movie)" if anchor == "movie"
                   else "window from official onset (onset = cue)")
            ax.plot(wins, mean, "o-", color=c, ms=4, label=lab)
            ours5 = np.array([subs["sub-" + k][0][wins.index(5.0)] for k in inc])
            from scipy import stats
            rho, prho = stats.spearmanr(ours5, theirs)
            w = stats.wilcoxon(ours5, theirs).pvalue
            log(f"ning {task} anchor={anchor} <- {p}")
            log("   included mean by window: " + ", ".join(f"{a:g}s {b:.1f}" for a, b in
                                                          zip(wins, mean)))
            log(f"   5 s window, their 11 subjects: ours {ours5.mean():.2f}% vs theirs "
                f"{theirs.mean():.2f}%  (diff {ours5.mean() - theirs.mean():+.2f}, Wilcoxon "
                f"p = {w:.4f}; per-subject Spearman rho {rho:+.3f}, p = {prho:.4f})")
        ax.plot([5.0], [ref], "*", color="k", ms=13, label=f"Ning et al. published ({ref}%)")
        ax.axhline(chance, color="k", ls="--", lw=1)
        ax.set_xlabel("cumulative window length from window start (s)")
        ax.set_ylabel("accuracy (%), their 11 subjects, correct trials")
        ax.set_title(f"{task}: Ning et al.'s pipeline (HbT GLM-AUC, in-fold SS, "
                     f"shrinkage LDA)\non corrected v2.0.0 timing", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES(".png", "ning"), dpi=150)
    log("  -> ning-v2.png\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["ablation", "roi", "channels", "ning"])
    args = ap.parse_args()
    log = Log(RES(".log"))
    for name in args.only:
        try:
            {"ablation": ablation, "roi": roi, "channels": channels, "ning": ning}[name](log)
        except FileNotFoundError as e:
            log(f"{name}: skipped, input missing ({e.filename})\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
