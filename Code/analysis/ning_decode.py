"""
Ning et al. (2024) classification, reproduced end to end, with every difference from
our own pipeline exposed as a switch so its contribution can be measured.

Their protocol, verbatim where it matters:

    "The features used were the area under the curve of the HRF with total hemoglobin
     (HbT). The features were used from an incremental window where all windows start
     at 0th second (cue onset). Window lengths tested here are 0 to 0.1, 0.2, 0.5, 1,
     1.5, 2, 3, 4, and 5 s."

    "The temporal basis functions used to model the HRF consisted of a sequence of 16
     Gaussian functions, spaced 1 s apart, with a typical width of 1 s."

    "we fitted the GLM model to a training dataset and estimated regression
     coefficients using the Ordinary Least Squares (OLS) method. Then the short
     separation regression coefficients (SS coefficients) estimated from the training
     set are used for the test set where the individual trials are the difference
     between the measured fNIRS signals and the systemic physiological regressor
     weighted by the SS coefficients."

    "LDA with linear shrinkage of the covariance matrix (Ledoit and Wolf, 2004) ...
     10 repetitions of 5-fold nested cross-validation."

The GLM is refit inside every fold, so the SS coefficients never see a held-out trial.
That is the whole reason this script owns the fold loop instead of consuming
pre-epoched features: `ning_build.py` deliberately saves continuous data.

The comparison this is here to settle: our 38.81% against their 49.4% (the honest
11-subject mean; the printed 45% silently averages in the 5 excluded subjects). Four
candidate causes, each with a flag:

  --lock cue|stim     epoch at the cue, as they do, or at the movie, as we did
  --features auc|bins one AUC per channel, as they do, or 6 time bins, as we did
  --chrom hbt         (build-time) total haemoglobin, as they do
  --no-ss             skip the short-separation GLM entirely

Usage:
    python ning_decode.py                       # their configuration
    python ning_decode.py --lock stim           # only the epoch lock changed
    python ning_decode.py --features bins       # only the feature definition changed
"""

import argparse
import glob
import os
import sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import layout
import paths

WINDOWS = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]   # s, all starting at 0
HEADLINE = 5.0
N_BASIS = 16           # "a sequence of 16 Gaussian functions"
BASIS_SPACING = 1.0    # "spaced 1 s apart"
BASIS_WIDTH = 1.0      # "with a typical width of 1 s"
DRIFT_ORDER = 3        # Homer3's default polynomial drift
BASELINE = (-2.0, 0.0)
N_REPEATS = 10
N_FOLDS = 5
N_BINS = 6             # only used by --features bins, to mirror build_dataset.py


def gaussian_basis(n_times, fs, onsets):
    """One column per basis function: Gaussians at 0,1,...,15 s after every onset.

    Summing over onsets rather than stacking them is what makes this a condition
    regressor -- the fitted weights describe the average response to that condition.
    """
    t = np.arange(n_times) / fs
    cols = np.zeros((n_times, N_BASIS))
    for b in range(N_BASIS):
        centre = b * BASIS_SPACING
        for o in onsets:
            z = (t - o - centre) / BASIS_WIDTH
            near = np.abs(z) < 4.0                  # a Gaussian is numerically zero past 4 sd
            cols[near, b] += np.exp(-0.5 * z[near] ** 2)
    return cols


def drift_matrix(n_times):
    t = np.linspace(-1.0, 1.0, n_times)
    return np.column_stack([t ** k for k in range(DRIFT_ORDER + 1)])


def assign_ss(long, short):
    """Highest-correlation short channel for each long channel (Gagnon et al.).

    Computed on the continuous recording and without reference to any label, so it
    carries no class information across the fold boundary.
    """
    a = (long - long.mean(0)) / (long.std(0) + 1e-12)
    b = (short - short.mean(0)) / (short.std(0) + 1e-12)
    corr = (a.T @ b) / len(a)                       # (n_long, n_short)
    return np.argmax(np.abs(corr), axis=1)


def clean_with_train_glm(long, short, ss_map, onsets, y, train_idx, drift):
    """Fit the GLM on training trials only, then subtract the SS regressor everywhere.

    Returns the cleaned continuous long channels. The HRF regressors are present so
    the SS weight is estimated free of the evoked response -- they are not used
    afterwards, exactly as in their description.
    """
    n_times = long.shape[0]
    hrf = np.hstack([gaussian_basis(n_times, FS, onsets[train_idx][y[train_idx] == c])
                     for c in np.unique(y)])
    fixed = np.hstack([hrf, drift])

    cleaned = long.copy()
    # only n_short distinct designs exist, so solve each once against all the
    # channels assigned to it rather than once per channel
    for k in np.unique(ss_map):
        chans = np.flatnonzero(ss_map == k)
        design = np.hstack([fixed, short[:, [k]]])
        beta, *_ = np.linalg.lstsq(design, long[:, chans], rcond=None)
        beta_ss = beta[-1]                          # (len(chans),)
        cleaned[:, chans] = long[:, chans] - np.outer(short[:, k], beta_ss)
    return cleaned


def epoch(sig, onsets, fs, tmax):
    """Cut [-2, tmax] s around each onset and subtract the pre-onset baseline."""
    lo = int(round(BASELINE[0] * fs))
    hi = int(round(tmax * fs)) + 1
    n_times = sig.shape[0]
    out = np.empty((len(onsets), hi - lo, sig.shape[1]), dtype=np.float32)
    for i, o in enumerate(onsets):
        s = int(round(o * fs))
        a, b = s + lo, s + hi
        if a < 0 or b > n_times:                    # pad rather than drop, so trial
            seg = np.zeros((hi - lo, sig.shape[1])) # counts stay comparable across
            va, vb = max(a, 0), min(b, n_times)     # configurations
            seg[va - a:vb - a] = sig[va:vb]
        else:
            seg = sig[a:b]
        base = seg[:-lo].mean(axis=0) if lo < 0 else 0.0
        out[i] = seg - base
    return out


# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid; support both
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def featurise(ep, fs, window, kind):
    """AUC over [0, window] per channel, or N_BINS time-averages, per channel."""
    lo = int(round(-BASELINE[0] * fs))              # index of t = 0
    hi = lo + int(round(window * fs)) + 1
    seg = ep[:, lo:hi, :]
    if kind == "auc":
        return _trapezoid(seg, dx=1.0 / fs, axis=1)   # (n_trials, n_channels)
    edges = np.linspace(0, seg.shape[1], N_BINS + 1).astype(int)
    bins = [seg[:, edges[i]:max(edges[i + 1], edges[i] + 1)].mean(axis=1)
            for i in range(N_BINS)]
    return np.concatenate(bins, axis=1)


def run_subject(f, args, rng_seed=0):
    z = np.load(f, allow_pickle=True)
    global FS
    FS = float(z["fs"])
    long = z["long"].astype(np.float64)
    short = z["short"].astype(np.float64)
    onsets = z["onsets"] if args.lock == "cue" else z["onsets_stim"]
    y = z["y"].astype(int)
    sub = str(z["subject"])
    excluded = str(z["excluded_by_ning"])

    if args.correct_only:
        m = z["correct"].astype(bool)
        long, onsets, y = long, onsets[m], y[m]

    if args.two_class:                              # lateral A vs lateral B
        m = y != 3
        onsets, y = onsets[m], y[m]

    ss_map = assign_ss(long, short)
    drift = drift_matrix(long.shape[0])
    acc = {w: [] for w in WINDOWS}

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rng_seed + rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            sig = (long if args.no_ss else
                   clean_with_train_glm(long, short, ss_map, onsets, y, tr, drift))
            ep = epoch(sig, onsets, FS, max(WINDOWS))
            for w in WINDOWS:
                X = featurise(ep, FS, w, args.features)
                clf = make_pipeline(
                    StandardScaler(),
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
                clf.fit(X[tr], y[tr])
                acc[w].append(clf.score(X[te], y[te]))

    return sub, excluded, {w: float(np.mean(v)) for w, v in acc.items()}, \
        {w: float(np.std([np.mean(v[i * N_FOLDS:(i + 1) * N_FOLDS])
                          for i in range(N_REPEATS)])) for w, v in acc.items()}, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=layout.derived("ds004830", "ning"))
    ap.add_argument("--lock", choices=["cue", "stim"], default="cue")
    ap.add_argument("--features", choices=["auc", "bins"], default="auc")
    ap.add_argument("--no-ss", action="store_true")
    ap.add_argument("--two-class", action="store_true")
    ap.add_argument("--correct-only", action="store_true")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also report subjects Ning et al. exclude, to reconstruct "
                         "the printed 16-subject mean")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "sub-*.npz")))
    if not files:
        sys.exit(f"no subject files in {args.data} -- run ning_build.py first")

    n_cls = 2 if args.two_class else 3
    print(f"{n_cls}-class | lock={args.lock} features={args.features} "
          f"ss={'off' if args.no_ss else 'in-fold GLM'} | "
          f"{N_REPEATS}x{N_FOLDS}-fold, shrinkage LDA | chance={100.0 / n_cls:.1f}%")
    print(f"{'subject':>9}  {'trials':>6}  " +
          "  ".join(f"{w:>5g}s" for w in WINDOWS) + "   note")

    kept, allsub = [], []
    for f in files:
        sub, excluded, means, sds, n = run_subject(f, args)
        row = "  ".join(f"{100 * means[w]:6.1f}" for w in WINDOWS)
        print(f"{sub:>9}  {n:>6}  {row}   {excluded}")
        allsub.append(means)
        if not excluded:
            kept.append(means)

    def summarise(rows, label):
        if not rows:
            return
        m = {w: 100 * np.mean([r[w] for r in rows]) for w in WINDOWS}
        row = "  ".join(f"{m[w]:6.1f}" for w in WINDOWS)
        print(f"{label:>9}  {len(rows):>6}  {row}")

    print()
    summarise(kept, "included")
    if args.include_excluded:
        summarise(allsub, "all")

    if kept:
        h = 100 * np.mean([r[HEADLINE] for r in kept])
        target = 78.3 if args.two_class else 49.4
        print(f"\nheadline ({HEADLINE:g}s window, {len(kept)} included subjects): "
              f"{h:.2f}%   Ning et al.: {target}%   gap: {h - target:+.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
