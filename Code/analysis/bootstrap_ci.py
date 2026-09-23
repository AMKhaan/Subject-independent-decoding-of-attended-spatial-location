"""
Confidence intervals, per-subject tests and a selection-aware null for the headline numbers.

Why this exists
---------------
decode.py reports a group mean and a permutation p-value, but no interval. A reviewer
will ask how precise 58.55% and 44.20% are with twelve subjects, and "10/12 above
chance" says only that ten accuracies exceed 33.3% by eye, not that ten are
individually distinguishable from chance. This script answers three questions on
exactly the data, features, folds and models decode.py uses:

  1. Interval on the group mean. The subject is the unit of replication, so the
     primary interval resamples subjects (percentile and BCa, 10,000 resamples). A
     two-level bootstrap that also resamples trials inside each subject is reported
     beside it; it is wider because it adds within-subject sampling noise, and the
     honest statement is the wider of the two.

  2. Which subjects are individually above chance. Every trial is predicted exactly
     once in LOSO, and once per repeat within-subject, so an exact one-sided binomial
     test on the trial count is valid (within-subject uses the first repeat so each
     trial is counted once). Holm correction across the twelve subjects.

  3. Is the best model's p-value biased by choosing it? decode.py picks the best of
     four models on the observed labels and then permutes only that one. The
     selection-aware null takes, for every permutation, the maximum group mean over
     all four models, which is the distribution the reported maximum should be
     compared against.

Point estimates are computed as decode.py computes them (mean of fold accuracies,
5 x 5-fold within-subject; one fold per subject in LOSO), so they match its log.

Every model/protocol cell and every permutation chunk is checkpointed atomically, so a
power cut costs at most one chunk.

Usage:  python bootstrap_ci.py [--data ../../derived/ds004830/v2] [--task 3class|lateral|both]
                               [--perms 200] [--boot 10000] [--jobs 20]
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone

import decode
import layout

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
NING = {"3class": 0.494, "lateral": 0.783}   # their 11 included subjects, see STATUS.md


# --------------------------------------------------------------------------- logging

class Log:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def save_ckpt(path, obj):
    """Atomic write: a power cut during the write cannot corrupt the checkpoint."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


# ------------------------------------------------------------------ out-of-fold preds

def models_1job():
    """decode.models(), but RF single-threaded so permutations parallelise outside it."""
    m = decode.models()
    m["Random Forest"].set_params(n_jobs=1)
    return m


def within_cell(subs, clf, repeats=5):
    """Same folds as decode.within_subject. Returns per-subject fold-mean accuracy and
    per-trial correctness (repeats x trials) for every subject."""
    from sklearn.model_selection import StratifiedKFold
    acc, per_trial = [], []
    for _, F, y in subs:
        folds, hit = [], np.zeros((repeats, len(y)), bool)
        for rep in range(repeats):
            cv = StratifiedKFold(5, shuffle=True, random_state=rep)
            for tr, te in cv.split(F, y):
                p = clone(clf).fit(F[tr], y[tr]).predict(F[te])
                hit[rep, te] = p == y[te]
                folds.append((p == y[te]).mean())
        acc.append(float(np.mean(folds)))
        per_trial.append(hit.tolist())
    return acc, per_trial


def loso_cell(subs, clf):
    """Same as decode.loso, keeping per-trial correctness."""
    Z = [((F - F.mean(0)) / (F.std(0) + 1e-9), y) for _, F, y in subs]
    acc, per_trial = [], []
    for i in range(len(Z)):
        Xtr = np.concatenate([Z[j][0] for j in range(len(Z)) if j != i])
        ytr = np.concatenate([Z[j][1] for j in range(len(Z)) if j != i])
        hit = clone(clf).fit(Xtr, ytr).predict(Z[i][0]) == Z[i][1]
        acc.append(float(hit.mean()))
        per_trial.append([hit.tolist()])
    return acc, per_trial


# ------------------------------------------------------------------------ bootstrap

def bca(x, B, rng):
    """Percentile and BCa 95% intervals for the mean of x, resampling its elements."""
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    boot = x[idx].mean(axis=1)
    pct = np.percentile(boot, [2.5, 97.5])
    theta = x.mean()
    z0 = stats.norm.ppf(np.clip((boot < theta).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6 * ((d ** 2).sum() ** 1.5) + 1e-300)
    zs = stats.norm.ppf([0.025, 0.975])
    adj = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    return pct, np.percentile(boot, adj * 100)


def two_level(per_trial, B, rng):
    """Resample subjects, then trials inside each resampled subject."""
    rates = [np.asarray(t, float).mean(axis=0) for t in per_trial]   # per-trial hit rate
    n = len(rates)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, n, size=n)
        out[b] = np.mean([rates[i][rng.integers(0, len(rates[i]), len(rates[i]))].mean()
                          for i in pick])
    return np.percentile(out, [2.5, 97.5])


# ---------------------------------------------------------------------- permutations

def one_perm(subs, k, seed):
    """Group mean of every model under one shared within-subject label shuffle."""
    g = np.random.default_rng([seed, k])
    perm = [(n, F, g.permutation(y)) for n, F, y in subs]
    w, l = {}, {}
    for name, clf in models_1job().items():
        w[name] = float(decode.within_subject(perm, clf, rng=0, repeats=1).mean())
        l[name] = float(decode.loso(perm, clf).mean())
    return k, w, l


# ----------------------------------------------------------------------------- main

def run_task(task, args, log):
    chance = 1 / 3 if task == "3class" else 1 / 2
    subs = decode.load(args.data, task, correct_only=True)
    names = [s[0] for s in subs]
    ck_path = layout.result(args.data, "bootstrap_ci", ".json", task, part="checkpoint")
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) and not args.fresh else {}
    ck.setdefault("cells", {})
    ck.setdefault("perms", {})

    log(f"\n{'=' * 78}\ntask {task}   chance {chance * 100:.1f}%   {len(subs)} subjects, "
        f"{sum(len(s[2]) for s in subs)} correct trials   data {args.data}\n{'=' * 78}")

    # 1. observed accuracies with per-trial correctness
    for proto in ("within", "loso"):
        for name, clf in decode.models().items():
            key = f"{proto}|{name}"
            if key in ck["cells"]:
                continue
            t0 = time.time()
            acc, pt = (within_cell if proto == "within" else loso_cell)(subs, clf)
            ck["cells"][key] = {"acc": acc, "per_trial": pt}
            save_ckpt(ck_path, ck)
            log(f"  computed {key:<32} {np.mean(acc) * 100:6.2f}%  ({time.time() - t0:.0f} s)")

    rng = np.random.default_rng(12345)
    summary = {"task": task, "chance": chance, "subjects": names, "rows": []}
    for proto in ("within", "loso"):
        log(f"\n--- {proto}: group mean with 95% intervals (unit = subject, B = {args.boot}) ---")
        log(f"  {'model':<20}{'mean':>8}{'percentile':>18}{'BCa':>18}{'2-level':>18}"
            f"{'t-interval':>18}   Wilcoxon p")
        best = max(decode.models(), key=lambda m: np.mean(ck["cells"][f"{proto}|{m}"]["acc"]))
        for name in decode.models():
            c = ck["cells"][f"{proto}|{name}"]
            a = np.array(c["acc"])
            pct, bc = bca(a, args.boot, rng)
            tl = two_level(c["per_trial"], min(args.boot, 4000), rng)
            tci = stats.t.interval(0.95, len(a) - 1, loc=a.mean(), scale=stats.sem(a))
            wil = stats.wilcoxon(a - chance, alternative="greater").pvalue
            f = lambda lo_hi: f"{lo_hi[0] * 100:5.1f}-{lo_hi[1] * 100:5.1f}"
            log(f"  {name + (' *' if name == best else ''):<20}{a.mean() * 100:7.2f}%"
                f"{f(pct):>18}{f(bc):>18}{f(tl):>18}{f(tci):>18}   {wil:.4f}")
            summary["rows"].append({"protocol": proto, "model": name, "best": name == best,
                                    "mean": a.mean(), "pct": list(pct), "bca": list(bc),
                                    "two_level": list(tl), "t": list(tci),
                                    "wilcoxon_p": wil, "per_subject": a.tolist()})
        log("  * = best model on the observed data, the one decode.py reports")

        # 2. per-subject exact binomial tests for the best model
        c = ck["cells"][f"{proto}|{best}"]
        ps, ks, ns = [], [], []
        for hit in c["per_trial"]:
            h = np.asarray(hit[0], bool)            # one prediction per trial
            ks.append(int(h.sum()))
            ns.append(len(h))
            ps.append(stats.binomtest(int(h.sum()), len(h), chance, "greater").pvalue)
        order = np.argsort(ps)
        holm = np.empty(len(ps))
        run = 0.0
        for r, i in enumerate(order):
            run = max(run, min(1.0, (len(ps) - r) * ps[i]))
            holm[i] = run
        log(f"\n  per-subject exact binomial, {best}"
            f"{' (first CV repeat)' if proto == 'within' else ''}:")
        log(f"  {'subject':<9}{'correct':>10}{'acc':>9}{'p':>10}{'Holm p':>10}")
        for i, n in enumerate(names):
            log(f"  {n:<9}{ks[i]:>5}/{ns[i]:<4}{ks[i] / ns[i] * 100:8.1f}%{ps[i]:10.4f}"
                f"{holm[i]:10.4f}{'  *' if holm[i] < 0.05 else ''}")
        n_raw = int((np.array(ps) < 0.05).sum())
        n_holm = int((holm < 0.05).sum())
        log(f"  individually above chance: {n_raw}/{len(ps)} at p < .05 uncorrected, "
            f"{n_holm}/{len(ps)} after Holm   (by eye, > chance: "
            f"{int((np.array(c['acc']) > chance).sum())}/{len(ps)})")
        summary[f"{proto}_binomial"] = {"best": best, "p": ps, "holm": holm.tolist(),
                                        "n_sig_raw": n_raw, "n_sig_holm": n_holm}

    # 3. selection-aware permutation null
    if args.perms:
        todo = [k for k in range(args.perms) if str(k) not in ck["perms"]]
        log(f"\n--- selection-aware permutation null, {args.perms} permutations "
            f"({args.perms - len(todo)} already checkpointed) ---")
        chunk = max(args.jobs, 1)
        for s in range(0, len(todo), chunk):
            t0 = time.time()
            res = Parallel(n_jobs=args.jobs)(delayed(one_perm)(subs, k, 777)
                                             for k in todo[s:s + chunk])
            for k, w, l in res:
                ck["perms"][str(k)] = {"within": w, "loso": l}
            save_ckpt(ck_path, ck)
            log(f"    {len(ck['perms'])}/{args.perms} permutations "
                f"({time.time() - t0:.0f} s for {len(res)})")
        P = [ck["perms"][str(k)] for k in range(args.perms)]
        for proto in ("within", "loso"):
            obs = {m: np.mean(ck["cells"][f"{proto}|{m}"]["acc"]) for m in decode.models()}
            best = max(obs, key=obs.get)
            null_best = np.array([p[proto][best] for p in P])
            null_max = np.array([max(p[proto].values()) for p in P])
            p_naive = (1 + (null_best >= obs[best]).sum()) / (1 + len(P))
            p_sel = (1 + (null_max >= obs[best]).sum()) / (1 + len(P))
            log(f"  {proto:<7} best = {best}, observed {obs[best] * 100:.2f}%")
            log(f"    null of that model alone : mean {null_best.mean() * 100:.2f}%  "
                f"95th {np.percentile(null_best, 95) * 100:.2f}%  p = {p_naive:.4f}")
            log(f"    null of max over 4 models: mean {null_max.mean() * 100:.2f}%  "
                f"95th {np.percentile(null_max, 95) * 100:.2f}%  p = {p_sel:.4f}"
                + ("  (floor)" if p_sel <= 1 / (1 + len(P)) else ""))
            summary[f"{proto}_perm"] = {"best": best, "obs": obs[best],
                                        "null_best_mean": null_best.mean(),
                                        "null_max_mean": null_max.mean(),
                                        "null_max_95": np.percentile(null_max, 95),
                                        "p_naive": p_naive, "p_selection_aware": p_sel}
    return summary


def figure(summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(summaries), figsize=(6.2 * len(summaries), 4.6),
                             squeeze=False)
    for ax, S in zip(axes[0], summaries):
        rows = S["rows"]
        ys = np.arange(len(rows))[::-1]
        for y, r in zip(ys, rows):
            col = "#1f5fa8" if r["protocol"] == "within" else "#c0502a"
            ax.plot(r["two_level"], [y, y], color=col, lw=1, alpha=0.35)
            ax.plot(r["bca"], [y, y], color=col, lw=3)
            ax.plot(r["mean"], y, "o", color=col, ms=7 if r["best"] else 5,
                    mfc=col if r["best"] else "white")
            ax.scatter(r["per_subject"], np.full(len(r["per_subject"]), y - 0.25),
                       s=6, color=col, alpha=0.4)
        ax.axvline(S["chance"], color="k", ls="--", lw=1)
        ax.axvline(NING[S["task"]], color="grey", ls=":", lw=1)
        ax.text(NING[S["task"]], len(rows) - 0.4, " Ning et al.\n (within)", fontsize=7,
                color="grey", va="top")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r['protocol']}: {r['model']}" for r in rows], fontsize=8)
        ax.set_xlabel("accuracy (group mean, 95% BCa; faint = two-level)")
        ax.set_title(f"{S['task']}, correct trials, {len(S['subjects'])} subjects",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=layout.derived("ds004830", "v2"))
    ap.add_argument("--task", default="both", choices=("3class", "lateral", "both"))
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    args = ap.parse_args()

    log = Log(layout.result(args.data, "bootstrap_ci", ".log"))
    log(f"\n##### bootstrap_ci.py  {time.strftime('%Y-%m-%d %H:%M')}  data={args.data}")
    tasks = ("3class", "lateral") if args.task == "both" else (args.task,)
    out = [run_task(t, args, log) for t in tasks]
    js = layout.result(args.data, "bootstrap_ci", ".json")
    png = layout.result(args.data, "bootstrap_ci", ".png")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    figure(out, png)
    log(f"\nwrote {layout.rel(js)}, {layout.rel(png)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
