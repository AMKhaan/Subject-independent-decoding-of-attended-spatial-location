"""
Our within-subject accuracy against Ning et al. (2024), subject by subject, on their subjects.

Why this exists
---------------
The paper compares our within-subject accuracy with the dataset paper's. Until now that
comparison was group means on different subject sets (our 12 against their 11), with our
best-of-four model against their shrinkage LDA. This script makes it like for like:

  * the same subjects: Ning et al. include 11 subjects (Table 1: excluded 10 and 25 for
    behavioural correctness below 70%, 17, 18 and 20 for >= 20 channels pruned). Our 12
    are exactly their 11 plus sub-25, so dropping sub-25 gives the identical set;
  * the same classifier: shrinkage (Ledoit-Wolf) LDA is the primary comparison, because it
    is theirs; our best model (RF) is reported beside it, labelled as selected;
  * per subject, paired: Wilcoxon signed-rank on the 11 pairs, and the Spearman
    correlation between our and their per-subject accuracies. A positive correlation
    means the two pipelines find the same subjects decodable, which is evidence that
    both are reading the same physiological signal.

What still differs, and is stated rather than hidden: their features are HbT GLM-HRF AUC
over a cumulative 0-5 s window from cue onset with SS regression refit per fold and 10 x
5-fold nested CV; ours are HbO + HbR binned means over 0-12 s from stimulus onset, 5 x
5-fold CV. So this compares pipelines end to end, on identical subjects and trials
(behaviourally correct only, as they do).

Reference data
--------------
NING below is transcribed from Ning et al. (2024), Front. Hum. Neurosci., Table 1,
columns "CV accuracy in percentage (2 class)" and "(3 class)". It is external published
data, not a result of this repository, which is why it is typed in; every number of OURS
is read from results/ds004830/bootstrap_ci/ds004830_v2_bootstrap_ci.json (bootstrap_ci.py). Check: their included means
are 49.36% and 78.27%, matching the 49.4% / 78.3% in STATUS.md.

Writes results/ds004830/compare_ning/ds004830_v2_compare_ning.log and the matching .png
under plots/ (see layout.py).

Usage:  python compare_ning.py
"""

import json
import os
import sys

import numpy as np
from scipy import stats

import layout

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = layout.derived("ds004830", "v2")
RES = lambda ext: layout.result(V2, "compare_ning", ext)

# Ning et al. 2024, Table 1: subject -> (2-class %, 3-class %), all 16 subjects
NING = {"08": (95, 67), "10": (55, 34), "12": (92, 84), "13": (60, 33), "14": (66, 46),
        "15": (55, 35), "16": (92, 62), "17": (68, 33), "18": (65, 33), "19": (83, 42),
        "20": (66, 49), "21": (86, 55), "22": (75, 40), "23": (65, 46), "24": (92, 33),
        "25": (50, 30)}
NING_EXCLUDED = {"10", "17", "18", "20", "25"}


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def main():
    log = Log(RES(".log"))
    ci = json.load(open(layout.result(V2, "bootstrap_ci", ".json")))
    inc = sorted(k for k in NING if k not in NING_EXCLUDED)
    for task, col in (("lateral", 0), ("3class", 1)):
        m = np.mean([NING[k][col] for k in inc])
        log(f"check: Ning et al. {task} mean over their {len(inc)} included = {m:.2f}%")
    log(f"their included subjects: {', '.join('sub-' + k for k in inc)}\n")

    results = {}
    for task, col in (("3class", 1), ("lateral", 0)):
        S = next(s for s in ci if s["task"] == task)
        names = [n.replace("sub-", "") for n in S["subjects"]]
        keep = [i for i, n in enumerate(names) if n in inc]
        assert [names[i] for i in keep] == inc, "subject sets differ"
        theirs = np.array([NING[names[i]][col] for i in keep], float)
        log(f"=== {task}, within-subject, their 11 subjects, correct trials ===")
        log(f"  Ning et al. (shrinkage LDA, HbT GLM-AUC 0-5 s): mean {theirs.mean():.2f}%")
        results[task] = {"subjects": inc, "ning": theirs.tolist()}
        for model, tag in (("LDA (Ledoit-Wolf)", "same classifier (primary)"),
                           ("Random Forest", "our best, selected on these data")):
            row = next(r for r in S["rows"] if r["protocol"] == "within" and r["model"] == model)
            ours = np.array(row["per_subject"])[keep] * 100
            d = ours - theirs
            w = stats.wilcoxon(ours, theirs)
            rho, prho = stats.spearmanr(ours, theirs)
            rng = np.random.default_rng(0)
            boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            log(f"  ours, {model} ({tag}): mean {ours.mean():.2f}%")
            log(f"    difference ours - theirs: {d.mean():+.2f} pp, 95% CI [{lo:+.1f}, {hi:+.1f}],"
                f" Wilcoxon p = {w.pvalue:.4f}, better in {int((d > 0).sum())}/11")
            log(f"    per-subject agreement: Spearman rho = {rho:+.3f}, p = {prho:.4f}")
            results[task][model] = ours.tolist()
        log(f"  {'subject':<8}{'Ning':>7}{'ours LDA':>10}{'ours RF':>9}")
        for j, n in enumerate(inc):
            log(f"  sub-{n:<4}{theirs[j]:7.1f}{results[task]['LDA (Ledoit-Wolf)'][j]:10.1f}"
                f"{results[task]['Random Forest'][j]:9.1f}")
        log("")
    log("reading: a positive, significant Spearman rho means both pipelines rank the same\n"
        "subjects as decodable; a difference CI spanning 0 means we match, not beat, them.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, task in zip(axes, ("3class", "lateral")):
        R = results[task]
        chance = 100 / 3 if task == "3class" else 50
        th = np.array(R["ning"])
        for model, mk, col in (("LDA (Ledoit-Wolf)", "o", "#1f5fa8"),
                               ("Random Forest", "s", "#c0502a")):
            ours = np.array(R[model])
            rho = stats.spearmanr(ours, th)[0]
            ax.scatter(th, ours, marker=mk, color=col, s=36, alpha=0.8,
                       label=f"ours, {model.split(' (')[0]} (rho {rho:+.2f})")
        for j, n in enumerate(R["subjects"]):
            ax.annotate(n, (th[j], R["LDA (Ledoit-Wolf)"][j]), xytext=(3, 2),
                        textcoords="offset points", fontsize=6, color="grey")
        lim = [min(chance - 10, th.min() - 5), 100]
        ax.plot(lim, lim, color="k", lw=0.8)
        ax.axhline(chance, color="k", ls="--", lw=0.8)
        ax.axvline(chance, color="k", ls="--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Ning et al. 2024, Table 1 (%)")
        ax.set_ylabel("ours, within-subject (%)")
        ax.set_title(f"{task}: their 11 subjects, paired (diagonal = equal)", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES(".png"), dpi=150)
    log(f"wrote {layout.rel(RES('.log'))}, {layout.rel(RES('.png'))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
