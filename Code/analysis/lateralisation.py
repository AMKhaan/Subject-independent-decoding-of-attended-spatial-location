"""
Does the cued location produce a contralateral response, and which label is left?

Two questions answered by one analysis.

1. Are the labels real? Spatial attention to a lateral location is expected to
   modulate the hemisphere opposite that location. If the reconstructed onsets and
   the second column of indexMoviesTest are both correct, the difference between
   the two lateral conditions should have opposite sign in the left and right
   hemispheres. If either is wrong, the difference is noise.

2. Which lateral label is which? The dataset documentation has a typo -- "1 =
   right, 2 = right, 3 = center" -- so the assignment between labels 1 and 2 is
   not established from documentation. The sign of the lateralisation index fixes
   it: attention to the left should raise HbO more in the right hemisphere.

The statistic is a difference of differences, so it is immune to the baseline
question raised by trial_profile.py: any condition-independent offset introduced
by the baseline window cancels.

  LI = (A - B) over right-hemisphere channels  -  (A - B) over left-hemisphere ones

Significance comes from a within-subject label permutation, which preserves every
other property of the data.

Usage:  python lateralisation.py [--window 2 10] [--perms 5000]
"""

import argparse
import glob
import os
import sys
import numpy as np
import scipy.io as sio

import build_dataset as bd
import paths

DATA = paths.data_dir()


def channel_x(root, channels):
    """Midpoint x of each channel; negative is the left hemisphere."""
    for _, path in bd.experiment_dirs(root):
        nirs = sorted(x for x in os.listdir(path) if x.endswith(".nirs"))
        if not nirs:
            continue
        sd = sio.loadmat(os.path.join(path, nirs[0]), variable_names=["SD"])["SD"][0, 0]
        src = np.array(sd["SrcPos"], dtype=float)
        det = np.array(sd["DetPos"], dtype=float)
        return np.array([(src[s - 1, 0] + det[d - 1, 0]) / 2 for s, d in channels])
    raise RuntimeError("no probe found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--window", nargs=2, type=float, default=[2.0, 10.0],
                    metavar=("T0", "T1"), help="s after stimulus onset to average")
    ap.add_argument("--perms", type=int, default=5000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if not files:
        sys.exit(f"no data in {args.data}; run build_dataset.py first")

    z0 = np.load(files[0], allow_pickle=True)
    channels = [tuple(c) for c in np.array(z0["channels"])]
    xs = channel_x(args.root, channels)
    left, right = xs < 0, xs > 0
    n_ch = len(channels)
    tmin, fs = float(z0["tmin"]), float(z0["fs"])
    t = tmin + np.arange(z0["X"].shape[1]) / fs
    win = (t >= args.window[0]) & (t <= args.window[1])

    print(f"{n_ch} channels: {left.sum()} left hemisphere, {right.sum()} right, "
          f"{(xs == 0).sum()} midline")
    print(f"averaging HbO over {args.window[0]:g}-{args.window[1]:g} s after "
          f"{bd.TRIGGER}, {win.sum()} samples\n")

    rng = np.random.default_rng(0)
    lis, perm_lis, names = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        X, y = z["X"], z["y"]
        hbo = X[:, :, :n_ch][:, win, :].mean(axis=1)      # (trials, channels)

        def li(labels):
            a = hbo[labels == 1].mean(axis=0)
            b = hbo[labels == 2].mean(axis=0)
            d = a - b
            return d[right].mean() - d[left].mean()

        lis.append(li(y))
        names.append(str(z["subject"]))

        lat = np.isin(y, (1, 2))
        null = np.empty(args.perms)
        yp = y.copy()
        for i in range(args.perms):
            yp[lat] = rng.permutation(y[lat])
            null[i] = li(yp)
        perm_lis.append(null)
        p = (np.abs(null) >= abs(lis[-1])).mean()
        print(f"  {names[-1]:<8} LI = {lis[-1]:+.4f} uM   "
              f"permutation p = {p:.3f}   {'*' if p < 0.05 else ''}")

    lis = np.array(lis)
    null_group = np.mean(np.array(perm_lis), axis=0)
    obs = lis.mean()
    p = (np.abs(null_group) >= abs(obs)).mean()

    print(f"\n=== group ===")
    print(f"  mean LI = {obs:+.4f} uM   permutation p = {p:.4f}   "
          f"({args.perms} within-subject label permutations)")
    print(f"  null distribution: mean {null_group.mean():+.4f}, "
          f"sd {null_group.std():.4f}")
    print(f"  subjects with LI of the group sign: "
          f"{int((np.sign(lis) == np.sign(obs)).sum())}/{len(lis)}")

    print(f"\n=== interpretation ===")
    if p >= 0.05:
        print("  No reliable lateralisation. Either the effect is too small for a")
        print("  channel-average contrast, or it is not present in these labels.")
        # The label question this used to leave open is now closed: the official
        # ds004830 v2.0.0 events give an explicit trial_type, and it maps one-to-one
        # onto our codes for all twelve subjects as 1 = Right, 2 = Left, 3 = Center.
        # So a null result here is a statement about the univariate contrast, not about
        # the labels -- and it sits alongside multivariate decoding that works well,
        # which is the expected pattern when the informative structure is spatial rather
        # than a whole-hemisphere mean shift.
        print("  Labels are not the explanation: v2.0.0 events fix 1 = Right, 2 = Left,")
        print("  3 = Center, confirmed on all twelve subjects. This is a negative result")
        print("  about the hemisphere-average contrast itself.")
    elif obs > 0:
        print("  Condition 1 raises HbO more in the RIGHT hemisphere than condition 2.")
        print("  Under contralateral spatial attention that makes 1 = LEFT, 2 = right.")
    else:
        print("  Condition 1 raises HbO more in the LEFT hemisphere than condition 2.")
        print("  Under contralateral spatial attention that makes 1 = RIGHT, 2 = left.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
