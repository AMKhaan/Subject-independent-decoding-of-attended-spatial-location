"""
How much accuracy does each evaluation flaw manufacture?

Three-class accuracies in this literature range from about 45% -- what this
dataset's own authors report -- to figures above 80%, on a problem where chance is
33.3%. Most of that gap is not a modelling result. This script decomposes it by
running the same classifier on the same correctly preprocessed data under
progressively more permissive protocols, so each step's contribution can be read
off directly.

  A  leave-one-subject-out            the honest protocol
  B  random split over trials         subjects pooled; subject identity leaks
  C  random split over sliding windows  windows from one trial land on both sides
  D  channel-as-sample with the label  channels treated as independent samples, and
     bug reproduced                    a trial-indexed label array read by channel

Only the protocol changes between rows. Preprocessing, features and classifier are
held fixed, so every difference is attributable to the evaluation design.

Usage:  python leakage_demo.py
"""

import argparse
import glob
import os
import sys
import warnings
import numpy as np

import scipy.io as sio
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_dataset as bd
import paths
from lag_sweep import continuous

warnings.filterwarnings("ignore")

DATA = paths.data_dir()
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
WIN_S, STEP_S = 4.0, 0.5      # sliding-window geometry for protocol C


def featurise(X, tmin, fs, bins=BINS):
    t = tmin + np.arange(X.shape[1]) / fs
    return np.concatenate([X[:, (t >= a) & (t < b), :].mean(axis=1) for a, b in bins],
                          axis=1)


def load_trials(data_dir):
    subs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        subs.append((str(z["subject"]), z["X"], z["y"].astype(int),
                     float(z["tmin"]), float(z["fs"])))
    return subs


def protocol_a(subs):
    """Leave-one-subject-out."""
    Z = []
    for n, X, y, tmin, fs in subs:
        F = featurise(X, tmin, fs)
        Z.append(((F - F.mean(0)) / (F.std(0) + 1e-9), y))
    accs = []
    for i in range(len(Z)):
        Xtr = np.concatenate([Z[j][0] for j in range(len(Z)) if j != i])
        ytr = np.concatenate([Z[j][1] for j in range(len(Z)) if j != i])
        m = CLF().fit(Xtr, ytr)
        accs.append((m.predict(Z[i][0]) == Z[i][1]).mean())
    return float(np.mean(accs)), f"{len(Z)} folds, one per held-out subject"


def protocol_b(subs, seed=0):
    """Random 80/20 over trials with all subjects pooled."""
    F = np.concatenate([featurise(X, tmin, fs) for _, X, y, tmin, fs in subs])
    y = np.concatenate([y for _, _, y, _, _ in subs])
    accs = []
    for s in range(5):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed + s,
                                  stratify=y)
        m = CLF().fit(F[tr], y[tr])
        accs.append((m.predict(F[te]) == y[te]).mean())
    return float(np.mean(accs)), f"{len(y)} trials pooled, 5 random 80/20 splits"


def protocol_c(subs, seed=0):
    """Sliding windows inside each trial, then a random split over windows."""
    Fs, ys, tid = [], [], []
    k = 0
    for _, X, y, tmin, fs in subs:
        w = int(round(WIN_S * fs))
        step = int(round(STEP_S * fs))
        t0 = int(round((0 - tmin) * fs))                 # start at stimulus onset
        starts = range(t0, X.shape[1] - w + 1, step)
        for s0 in starts:
            seg = X[:, s0:s0 + w, :]
            Fs.append(seg.mean(axis=1))
            ys.append(y)
            tid.append(np.arange(k, k + len(y)))
        k += len(y)
    F = np.concatenate(Fs)
    y = np.concatenate(ys)
    accs = []
    for s in range(5):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed + s,
                                  stratify=y)
        m = CLF().fit(F[tr], y[tr])
        accs.append((m.predict(F[te]) == y[te]).mean())
    n_win = len(y) // sum(len(s[2]) for s in subs)
    return float(np.mean(accs)), (f"{len(y)} windows ({WIN_S:g} s, {STEP_S:g} s step, "
                                  f"~{n_win} per trial), 5 random 80/20 splits")


def protocol_d(root, seed=0, max_channels=2, win_s=4.0, step_s=2.0):
    """The channel-as-sample protocol, reproduced on the inputs it is used with.

    The construction is `label = label_list[ch]`, so every window cut from channel 0
    is called trial 1's class and every window from channel 1 is called trial 2's.
    Nothing about attention is being predicted; the target is channel identity.

    It also feeds raw 690 nm intensity with no optical-density conversion, no
    Beer-Lambert and no filtering, so channels differ by a large constant offset
    that depends on how much light that detector happened to receive. Predicting
    the label is then the same problem as reading off that offset.

    The claim that the target is channel identity is *measured*, not asserted, in
    oracle_check.py: it recovers the (subject, channel) group from the same windows
    and reports how well the group determines the label. Do not restate that number
    here -- run oracle_check.py.
    """
    Fs, ys, chan = [], [], []
    for exp, path in bd.experiment_dirs(root):
        runs, _, _ = bd.load_manifest(path, exp)
        labels, _ = bd.load_labels(path)     # the bug used the unfiltered array
        f = os.path.join(path, runs[0] + ".nirs")
        d = np.asarray(sio.loadmat(f, variable_names=["d"])["d"], dtype=float)
        w = int(round(win_s * bd.FS_RAW))
        step = int(round(step_s * bd.FS_RAW))
        for ch in range(max_channels):
            lab = labels[ch] if ch < len(labels) else 0      # the bug, verbatim
            for s0 in range(0, d.shape[0] - w + 1, step):
                Fs.append(d[s0:s0 + w, ch])
                ys.append(lab)
                chan.append((exp, ch))
    F = np.asarray(Fs)
    y = np.asarray(ys)
    accs = []
    for s in range(5):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed + s,
                                  stratify=y)
        m = RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1)
        m.fit(F[tr], y[tr])
        accs.append((m.predict(F[te]) == y[te]).mean())

    uniq, cnt = np.unique(y, return_counts=True)
    major = cnt.max() / cnt.sum()
    groups = sorted(set(chan))
    return float(np.mean(accs)), (
        f"{len(y)} raw-intensity windows, {max_channels} channels x "
        f"{len(groups) // max_channels} subjects = {len(groups)} channel groups; "
        f"majority class {major * 100:.1f}% (see oracle_check.py for the mechanism)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()

    subs = load_trials(args.data)
    z0 = np.load(sorted(glob.glob(os.path.join(args.data, "*.npz")))[0], allow_pickle=True)
    channels = [tuple(c) for c in np.array(z0["channels"])]

    rows = []
    print("running protocols (same data, same classifier, same features)...\n")
    for tag, desc, fn in (
        ("A", "leave-one-subject-out", lambda: protocol_a(subs)),
        ("B", "random split over trials, subjects pooled", lambda: protocol_b(subs)),
        ("C", "random split over sliding windows", lambda: protocol_c(subs)),
        ("D", "channel-as-sample with the label bug", lambda: protocol_d(args.root)),
    ):
        acc, note = fn()
        rows.append((tag, desc, acc, note))
        print(f"  {tag}  {acc * 100:6.2f}%   {desc}")
        print(f"      {note}")

    base = rows[0][2]
    print(f"\n=== inflation over the honest protocol ===")
    print(f"  {'':<4}{'protocol':<44}{'accuracy':>10}{'vs A':>10}")
    for tag, desc, acc, _ in rows:
        d = "" if tag == "A" else f"{(acc - base) * 100:+.1f} pp"
        print(f"  {tag:<4}{desc:<44}{acc * 100:9.2f}%{d:>10}")
    print(f"\n  chance = 33.3%   dataset authors (Ning et al.) report ~45% within-subject")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
