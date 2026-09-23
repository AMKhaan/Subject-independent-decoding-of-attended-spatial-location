"""Measure, rather than assert, the mechanism behind protocol D.

Protocol D in leakage_demo.py reproduces the channel-as-sample construction with the
label taken from the channel index. Its accuracy is only interesting if we can show
what the classifier is actually keying on. Two measurements:

  1. channel-group recovery -- a 24-way classification of (subject, channel) identity
     from the same raw-intensity windows. Because the label is a deterministic function
     of the group, this accuracy is the ceiling that channel recognition alone can reach
     on the label task. It is measured here, not assumed.

  2. classifier invariance -- protocol D re-run with the L2 logistic regression used in
     protocols A-C, so the ablation's "classifier held fixed" claim can be checked
     against the row where it matters most.

Usage:  python oracle_check.py [--jobs 2]
"""

import argparse
import json
import os

import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_dataset as bd
import paths

ROOT = paths.dataset_root()
WIN_S, STEP_S, MAX_CH, N_SPLITS = 4.0, 2.0, 2, 5


def build_windows(root):
    """Exactly the feature construction protocol_d uses, plus the group identity."""
    Fs, ys, gs = [], [], []
    for exp, path in bd.experiment_dirs(root):
        runs, _, _ = bd.load_manifest(path, exp)
        labels, _ = bd.load_labels(path)
        d = np.asarray(sio.loadmat(os.path.join(path, runs[0] + ".nirs"),
                                   variable_names=["d"])["d"], dtype=float)
        w = int(round(WIN_S * bd.FS_RAW))
        step = int(round(STEP_S * bd.FS_RAW))
        for ch in range(MAX_CH):
            lab = labels[ch] if ch < len(labels) else 0
            for s0 in range(0, d.shape[0] - w + 1, step):
                Fs.append(d[s0:s0 + w, ch])
                ys.append(lab)
                gs.append(f"{exp}:ch{ch}")
    groups = sorted(set(gs))
    gidx = {g: i for i, g in enumerate(groups)}
    return (np.asarray(Fs), np.asarray(ys),
            np.asarray([gidx[g] for g in gs]), groups)


def cv_accuracy(F, target, make_model, seed=0):
    accs = []
    for s in range(N_SPLITS):
        tr, te = train_test_split(np.arange(len(target)), test_size=0.2,
                                  random_state=seed + s, stratify=target)
        m = make_model(s)
        m.fit(F[tr], target[tr])
        accs.append(float((m.predict(F[te]) == target[te]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="oracle_check.json")
    args = ap.parse_args()

    F, y, g, groups = build_windows(args.root)
    print(f"{len(y)} windows, {len(groups)} channel groups, "
          f"{len(np.unique(y))} label values", flush=True)

    rf = lambda s: RandomForestClassifier(n_estimators=200, random_state=s,
                                          n_jobs=args.jobs)
    lr = lambda s: make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=0.1))

    res = {}

    # 1. how recoverable is channel-group identity itself?
    acc, sd = cv_accuracy(F, g, rf)
    res["group_recovery_rf"] = acc
    print(f"  channel-group recovery (RF, {len(groups)}-way): {acc * 100:.2f}% "
          f"(sd {sd * 100:.2f}, chance {100 / len(groups):.2f}%)", flush=True)

    # 2. the label task, both classifiers, same windows
    acc, sd = cv_accuracy(F, y, rf)
    res["label_rf"] = acc
    print(f"  protocol D label accuracy (RF):       {acc * 100:.2f}% (sd {sd * 100:.2f})",
          flush=True)

    acc, sd = cv_accuracy(F, y, lr)
    res["label_logistic"] = acc
    print(f"  protocol D label accuracy (logistic): {acc * 100:.2f}% (sd {sd * 100:.2f})",
          flush=True)

    # 3. the ceiling channel identity implies: map each group to its assigned label
    #    and score that rule on the true labels. Deterministic by construction, but
    #    stated explicitly so the chain group -> label is auditable.
    grp_lab = {}
    consistent = True
    for gi in np.unique(g):
        vals = np.unique(y[g == gi])
        grp_lab[int(gi)] = int(vals[0])
        if len(vals) > 1:
            consistent = False
    implied = np.array([grp_lab[int(gi)] for gi in g])
    res["group_to_label_deterministic"] = bool(consistent)
    res["oracle_ceiling"] = float((implied == y).mean())
    uniq, cnt = np.unique(y, return_counts=True)
    res["majority_class"] = float(cnt.max() / cnt.sum())
    res["n_windows"] = int(len(y))
    res["n_groups"] = int(len(groups))

    print(f"  group->label mapping single-valued:   {consistent}")
    print(f"  oracle ceiling given perfect group ID: {res['oracle_ceiling'] * 100:.2f}%")
    print(f"  majority class:                        {res['majority_class'] * 100:.2f}%")

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
