"""
Is "attended location" decoding in the overt task really eye-movement decoding?

Why this exists
---------------
In the overt cocktail-party task (ds004830's task, and ds007738's `overt`) subjects move
their eyes to the cued side. Frontal eye field sits under our most informative channels
(physiology.py), and saccades also bring ocular and systemic artefacts. So a left/right
decoder could be reading the eye movement rather than the allocation of attention. ds004830
cannot separate the two. ds007738 can: its `visualorient` task is an eye-movement-only
baseline ("Subjects make ~5 s eye orienting and fixation movements with no audiovisual
stimuli", README) with the same Left/Right structure.

Three tests, on the subjects that have both tasks, with the same features, trial rules and
classifier (L2 logistic C = 0.1, decode.py's LOSO model) as the replication:

  1. Matched decodability. visualorient has ~20 unbalanced trials per subject; overt ~60.
     Overt is subsampled to visualorient's exact per-class counts per subject (20 random
     draws), so both tasks are decoded from the same amount of data. Within-subject
     (stratified CV) and LOSO. If eye movements alone decode as well as overt attention,
     overt decoding is not evidence of attention.
  2. Cross-task transfer, LOSO. Train on the other subjects' visualorient (eye movements
     only), test on the held-out subject's overt trials, and the reverse. A model built on
     pure eye movement that decodes overt attention trials means the shared component
     is eye movement.
  3. Everything with the per-subject normalisation schemes of normalisation_check.py
     (transductive and causal), so a positive result cannot hinge on transductivity.

Writes results/ds007738/control_eye_movement/ds007738_control_eye_movement.log and .json,
and the matching .png under plots/ (see layout.py).

Usage:  python control_eye_movement.py [--overt ../../derived/ds007738/overt]
                                        [--orient ../../derived/ds007738/visualorient]
                                        [--covert ../../derived/ds007738/covert] [--report]
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import decode
import layout
from normalisation_check import normalise, bca

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = lambda ext: layout.result("ds007738", "control_eye_movement", ext)
LR = lambda: LogisticRegression(max_iter=2000, C=0.1)
DRAWS = 20


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "sub-*.npz"))):
        z = np.load(f, allow_pickle=True)
        order = np.argsort(z["onsets"] + 1e5 * z["run"])       # chronological across runs
        F = decode.featurise(z["X"], float(z["tmin"]), float(z["fs"]))[order]
        out[str(z["subject"])] = {"F": F, "y": z["y"].astype(int)[order],
                                  "use": z["correct"].astype(bool)[order]}
    return out


def zs(s, scheme):
    return normalise(s["F"], s["use"], scheme, W=min(10, len(s["F"]) // 2))


def within(F, y, rng_seed):
    k = min(5, np.bincount(y)[np.bincount(y) > 0].min())
    if k < 2:
        return np.nan
    a = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=rng_seed).split(F, y):
        a.append((LR().fit(F[tr], y[tr]).predict(F[te]) == y[te]).mean())
    return float(np.mean(a))


def subsample(D, s, cnt, rng):
    """Boolean mask: D[s]'s usable trials subsampled to cnt[c] per class."""
    idx = []
    for c in (1, 2):
        pool = np.where(D[s]["use"] & (D[s]["y"] == c))[0]
        idx.append(rng.choice(pool, size=min(int(cnt[c]), len(pool)), replace=False))
    m = np.zeros(len(D[s]["y"]), bool)
    m[np.concatenate(idx)] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overt", default=layout.derived("ds007738", "overt"))
    ap.add_argument("--orient", default=layout.derived("ds007738", "visualorient"))
    ap.add_argument("--covert", default=layout.derived("ds007738", "covert"),
                    help="covert attention (eyes fixed); used when the directory exists")
    ap.add_argument("--report", action="store_true", help="only redraw the figure from the JSON")
    args = ap.parse_args()
    if args.report:
        return plot(json.load(open(OUT(".json"))))
    log = Log(OUT(".log"))
    T = {"overt": load(args.overt), "eye": load(args.orient)}
    if glob.glob(os.path.join(args.covert, "sub-*.npz")):
        T["covert"] = load(args.covert)
    subs = sorted(set.intersection(*(set(v) for v in T.values())))
    log(f"eye-movement control: {len(subs)} subjects with all of {', '.join(T)} "
        f"(eye = visualorient, eye movements only)\n")
    rng = np.random.default_rng(0)
    res = {}

    def loso(Z, D, pick, Zt=None, Dt=None, pickt=None):
        Zt, Dt, pickt = Zt or Z, Dt or D, pickt or pick
        acc = []
        for s in subs:
            tr = [t for t in subs if t != s]
            X = np.concatenate([Z[t][pick[t]] for t in tr])
            y = np.concatenate([D[t]["y"][pick[t]] for t in tr])
            acc.append((LR().fit(X, y).predict(Zt[s][pickt[s]]) == Dt[s]["y"][pickt[s]]).mean())
        return np.array(acc)

    for scheme in ("transductive", "causal"):
        Z = {k: {s: zs(D[s], scheme) for s in subs} for k, D in T.items()}
        use = {k: {s: D[s]["use"] for s in subs} for k, D in T.items()}
        # counts to match, per subject, from visualorient's usable trials
        cnt = {s: np.bincount(T["eye"][s]["y"][use["eye"][s]], minlength=3) for s in subs}
        rows = {}
        for k, D in T.items():
            if k == "eye":
                rows["within eye"] = np.array([within(Z[k][s][use[k][s]], D[s]["y"][use[k][s]], 0)
                                               for s in subs])
                rows["LOSO eye"] = loso(Z[k], D, use[k])
                continue
            w = []
            for s in subs:
                w.append(np.nanmean([within(Z[k][s][m], D[s]["y"][m], d) for d in range(DRAWS)
                                     for m in [subsample(D, s, cnt[s], rng)]]))
            rows[f"within {k} (matched n)"] = np.array(w)
            rows[f"LOSO {k} (matched n)"] = np.mean(
                [loso(Z[k], D, {s: subsample(D, s, cnt[s], rng) for s in subs})
                 for _ in range(DRAWS)], axis=0)
            rows[f"LOSO {k} (all trials)"] = loso(Z[k], D, use[k])
        # cross-task transfer, LOSO: train on the other subjects' task a, test on held-out task b
        for a, b in (("eye", "overt"), ("overt", "eye"), ("eye", "covert"), ("overt", "covert"),
                     ("covert", "overt"), ("covert", "eye")):
            if a in T and b in T:
                rows[f"transfer {a} -> {b}"] = loso(Z[a], T[a], use[a], Z[b], T[b], use[b])

        log(f"=== normalisation: {scheme} ===")
        out = {}
        for name, a in rows.items():
            a = a[~np.isnan(a)]
            lo, hi = bca(a)
            p = stats.wilcoxon(a - 0.5, alternative="greater").pvalue if len(a) > 5 else np.nan
            log(f"  {name:<30} {a.mean() * 100:6.2f}%  [{lo * 100:4.1f}, {hi * 100:5.1f}]  "
                f"n={len(a):>2}  vs 50%: Wilcoxon p = {p:.4f}")
            out[name] = a.tolist()
        for k in [k for k in T if k != "eye"]:
            for kind in ("within", "LOSO"):
                a, b = rows[f"{kind} {k} (matched n)"], rows[f"{kind} eye"]
                ok = ~np.isnan(a) & ~np.isnan(b)
                p = stats.wilcoxon(a[ok], b[ok]).pvalue
                log(f"  {k} - eye movement ({kind}, matched n): "
                    f"{(a[ok] - b[ok]).mean() * 100:+.2f} pp, paired Wilcoxon p = {p:.4f}")
        log("")
        res[scheme] = out

    log("reading: if eye-movement-only decoding matches overt decoding, and a model trained\n"
        "on eye movements transfers to overt trials, the overt left/right signal is at\n"
        "least partly oculomotor. Covert attention (eyes fixed) is the test of attention\n"
        "proper: above-chance covert decoding, and no transfer from eye movements to covert\n"
        "trials, would mean a left/right attention signal that is not the eye movement.")
    json.dump(res, open(OUT(".json"), "w"), indent=1)
    plot(res)
    log(f"wrote {layout.rel(OUT('.log'))}, {layout.rel(OUT('.json'))}, {layout.rel(OUT('.png'))}")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(res["causal"])
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.75 * len(names) + 2), 8), sharex=True)
    for ax, scheme in zip(axes, ("transductive", "causal")):
        for i, n in enumerate(names):
            v = np.array(res[scheme][n]) * 100
            lo, hi = bca(v / 100)
            col = ("#2a9d5c" if n.startswith("transfer") else "#c0502a" if "eye" in n
                   else "#8a4fbf" if "covert" in n else "#1f5fa8")
            ax.bar(i, v.mean(), color=col, alpha=0.85)
            ax.errorbar(i, v.mean(), [[v.mean() - lo * 100], [hi * 100 - v.mean()]],
                        color="k", capsize=3)
            ax.scatter(i + np.random.default_rng(i).uniform(-0.2, 0.2, len(v)), v, s=5,
                       color="k", alpha=0.3)
            ax.text(i, 3, f"{v.mean():.1f}", ha="center", fontsize=7, color="w")
        ax.axhline(50, color="k", ls="--", lw=1)
        ax.set_ylim(0, 105)
        ax.set_ylabel("left vs right accuracy (%)")
        ax.set_title(f"normalisation: {scheme}", fontsize=9)
    axes[-1].set_xticks(range(len(names)))
    axes[-1].set_xticklabels([n.replace(" (", "\n(").replace(" -> ", "\n-> ") for n in names],
                             fontsize=7, rotation=45, ha="right")
    fig.suptitle("ds007738 eye-movement control (blue overt, purple covert, red eye movement "
                 "only, green cross-task transfer; BCa 95% CI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT(".png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
