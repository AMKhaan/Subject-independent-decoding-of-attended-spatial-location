"""
Decode the cued spatial location, evaluated honestly.

This is the honest benchmark of the paper -- the number the protocol ablation in
leakage_demo.py should be compared against. Three properties matter:

  labels    Every epoch carries its own trial's cued location, taken from column 2
            of indexMoviesTest, indexed by trial. Indexing that array by channel
            number instead yields labels that describe which channel a window came
            from rather than what the subject attended to; that is protocol D.

  splitting Two evaluations are reported. Within-subject cross-validation is the
            protocol Ning et al. used and is the only fair comparison to their
            ~45% on three classes. Leave-one-subject-out is what a claim of
            decoding attention implies, and is the one that tests generalisation.
            A random 80/20 split over windows, by contrast, puts windows from the
            same subject and the same trial on both sides.

  chance    Chance is 33.3% for the three-class problem and 50% for the lateral
            pair. Significance comes from a label permutation that keeps the
            cross-validation structure intact, not from comparing to chance by eye.

Usage:  python decode.py [--perms 500] [--task 3class|lateral]
"""

import argparse
import glob
import os
import sys
import warnings
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import paths

warnings.filterwarnings("ignore")

DATA = paths.data_dir()
BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]   # s after stimulus onset


def models():
    """Ledoit-Wolf LDA matches the classifier Ning et al. report; the other three are
    the classical baselines this literature reports alongside it."""
    return {
        "LDA (Ledoit-Wolf)": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "Logistic (L2)": make_pipeline(StandardScaler(),
                                       LogisticRegression(max_iter=2000, C=0.1)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=0,
                                                n_jobs=-1),
    }


def featurise(X, tmin, fs):
    """Mean concentration in consecutive time bins, per channel.

    fNIRS responses are slow and smooth, so binned means keep essentially all the
    information at a fraction of the dimensionality of the raw 140-sample epoch.
    """
    t = tmin + np.arange(X.shape[1]) / fs
    cols = [X[:, (t >= a) & (t < b), :].mean(axis=1) for a, b in BINS]
    return np.concatenate(cols, axis=1)


def load(data_dir, task, correct_only=False):
    """correct_only reproduces Ning et al.'s trial selection: keepCorrectTrials.m
    keeps a trial only if the subject's auditory and visual reports both matched
    the cued target, on the reasoning that a trial the subject got wrong is not
    evidence about where they were attending."""
    subs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        F = featurise(z["X"], float(z["tmin"]), float(z["fs"]))
        y = z["y"].astype(int)
        if correct_only:
            ok = z["correct"].astype(bool)
            F, y = F[ok], y[ok]
        if task == "lateral":
            m = np.isin(y, (1, 2))
            F, y = F[m], y[m]
        subs.append((str(z["subject"]), F, y))
    return subs


def within_subject(subs, clf, rng, repeats=5):
    """Stratified k-fold inside each subject. Comparable to Ning et al."""
    out = []
    for _, F, y in subs:
        accs = []
        for rep in range(repeats):
            cv = StratifiedKFold(5, shuffle=True, random_state=rng + rep)
            for tr, te in cv.split(F, y):
                from sklearn.base import clone
                m = clone(clf).fit(F[tr], y[tr])
                accs.append((m.predict(F[te]) == y[te]).mean())
        out.append(float(np.mean(accs)))
    return np.array(out)


def loso(subs, clf):
    """Leave-one-subject-out. Each subject is z-scored on its own trials first,
    which uses no labels and is what makes cross-subject transfer even plausible."""
    from sklearn.base import clone
    Z = [( (F - F.mean(0)) / (F.std(0) + 1e-9), y) for _, F, y in subs]
    out = []
    for i in range(len(Z)):
        Xtr = np.concatenate([Z[j][0] for j in range(len(Z)) if j != i])
        ytr = np.concatenate([Z[j][1] for j in range(len(Z)) if j != i])
        m = clone(clf).fit(Xtr, ytr)
        out.append(float((m.predict(Z[i][0]) == Z[i][1]).mean()))
    return np.array(out)


def permute(subs, clf, n_perms, rng, mode):
    """Null distribution of the group-mean accuracy under shuffled labels.

    Labels are shuffled within subject, so class balance and every subject-level
    property of the features are preserved; only the trial-to-label mapping breaks.
    """
    g = np.random.default_rng(rng)
    null = np.empty(n_perms)
    for k in range(n_perms):
        perm = [(n, F, g.permutation(y)) for n, F, y in subs]
        acc = within_subject(perm, clf, rng=0, repeats=1) if mode == "within" \
            else loso(perm, clf)
        null[k] = acc.mean()
        if (k + 1) % max(1, n_perms // 10) == 0:
            print(f"    permutation {k + 1}/{n_perms}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return null


def report(name, accs, names, chance):
    print(f"  {name:<20} mean {accs.mean() * 100:5.2f}%   "
          f"sd {accs.std() * 100:4.2f}   "
          f"range {accs.min() * 100:5.2f}-{accs.max() * 100:5.2f}   "
          f"above chance: {int((accs > chance).sum())}/{len(accs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--task", default="3class", choices=("3class", "lateral"))
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--correct-only", action="store_true",
                    help="keep only behaviourally correct trials, as Ning et al. do")
    args = ap.parse_args()

    chance = 1 / 3 if args.task == "3class" else 1 / 2
    subs = load(args.data, args.task, args.correct_only)
    names = [s[0] for s in subs]
    ntr = sum(len(s[2]) for s in subs)
    print(f"task: {args.task}   chance = {chance * 100:.1f}%"
          f"{'   correct trials only' if args.correct_only else ''}")
    print(f"{len(subs)} subjects, {ntr} trials, {subs[0][1].shape[1]} features "
          f"({len(BINS)} time bins x {subs[0][1].shape[1] // len(BINS)} channels)\n")

    print("=== within-subject 5-fold CV (5 repeats) -- comparable to Ning et al. ===")
    w = {}
    for name, clf in models().items():
        w[name] = within_subject(subs, clf, rng=0)
        report(name, w[name], names, chance)

    print("\n=== leave-one-subject-out -- tests generalisation to a new person ===")
    l = {}
    for name, clf in models().items():
        l[name] = loso(subs, clf)
        report(name, l[name], names, chance)

    best_w = max(w, key=lambda k: w[k].mean())
    best_l = max(l, key=lambda k: l[k].mean())

    print(f"\n=== per-subject accuracy, best model in each protocol ===")
    print(f"  {'subject':<9} {'within (' + best_w + ')':<26} {'LOSO (' + best_l + ')'}")
    for i, n in enumerate(names):
        print(f"  {n:<9} {w[best_w][i] * 100:21.2f}%      {l[best_l][i] * 100:8.2f}%")

    if args.perms:
        print(f"\n=== label permutation, {args.perms} permutations ===")
        for mode, res, best in (("within", w, best_w), ("loso", l, best_l)):
            obs = res[best].mean()
            print(f"  {mode} / {best}: observed {obs * 100:.2f}%")
            null = permute(subs, models()[best], args.perms, 0, mode)
            # Add-one (Phipson & Smyth 2010): the observed labelling is itself one of the
            # permutations, so the estimator without it can return p = 0, which is not a
            # possible p-value and cannot be defended in a paper. With n permutations the
            # smallest reportable value is 1/(n+1).
            p = (1 + int((null >= obs).sum())) / (1 + args.perms)
            floor = 1.0 / (1 + args.perms)
            print(f"    null mean {null.mean() * 100:.2f}%  sd {null.std() * 100:.2f}  "
                  f"95th pct {np.percentile(null, 95) * 100:.2f}%  ->  p = {p:.4f}"
                  + (f" (floor, p < {floor:.4f})" if p <= floor else "")
                  + f"{'  significant' if p < 0.05 else '  NOT significant'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
