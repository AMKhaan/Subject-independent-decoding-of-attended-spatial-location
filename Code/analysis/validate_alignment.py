"""
Evidence that the reconstructed trial onsets are correct.

The alignment (onset = startT + Trigger3) is inferred from the response files
rather than documented, so it has to be demonstrated from the data. Three tests:

  1. Shape.  The trial-averaged HbO should be flat before onset, rise, and peak
     4-8 s later -- the canonical haemodynamic response.
  2. Anti-correlation.  HbO and HbR must move in opposite directions. This is the
     signature of a real haemodynamic response; filtering artefacts and epochs cut
     at arbitrary times do not produce it.
  3. Null control.  Epochs cut at random times in the same recordings, through an
     otherwise identical pipeline (build_dataset.py --null), should show neither.

If tests 1 and 2 pass at the reconstructed onsets and fail in the null, the
alignment is right.

Usage:  python validate_alignment.py
"""

import glob
import os
import sys
import numpy as np

import paths

# this script has no command line, so the environment variables paths.py reads are the
# only way run_all.py can point it somewhere other than the in-repo default
DATA = paths.data_dir()
NULL = paths.null_dir()


def load_all(d):
    subs = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        subs.append((str(z["subject"]), z["X"], float(z["tmin"]), float(z["fs"])))
    return subs


def stats(X, tmin, fs):
    """Trial- and channel-averaged HbO/HbR, their post-onset correlation, and peak."""
    n_ch = X.shape[2] // 2
    hbo = X[:, :, :n_ch].mean(axis=(0, 2))
    hbr = X[:, :, n_ch:].mean(axis=(0, 2))
    t = tmin + np.arange(X.shape[1]) / fs
    post = t >= 0
    r = np.corrcoef(hbo[post], hbr[post])[0, 1]
    i = np.argmax(np.abs(hbo[post]))
    return t, hbo, hbr, r, t[post][i], hbo[post][i]


def report(subs, title):
    print(f"\n=== {title} ===")
    rs, peaks = [], []
    for name, X, tmin, fs in subs:
        _, _, _, r, pk, amp = stats(X, tmin, fs)
        print(f"  {name:<8} HbO/HbR r = {r:+.3f}   peak {pk:5.1f} s   amp {amp:+.4f} uM")
        rs.append(r)
        peaks.append(pk)

    Xg = np.concatenate([X for _, X, _, _ in subs], axis=0)
    tmin, fs = subs[0][2], subs[0][3]
    t, hbo, hbr, r, pk, amp = stats(Xg, tmin, fs)
    print(f"  {'GROUP':<8} HbO/HbR r = {r:+.3f}   peak {pk:5.1f} s   amp {amp:+.4f} uM"
          f"   ({Xg.shape[0]} trials)")
    print(f"  subjects with negative HbO/HbR correlation: "
          f"{sum(1 for x in rs if x < 0)}/{len(rs)}   median r = {np.median(rs):+.3f}")
    print(f"  peak latency in the canonical 4-8 s window: "
          f"{sum(1 for p in peaks if 4 <= p <= 8)}/{len(peaks)} subjects")
    return t, hbo, hbr, r, np.median(rs)


def plot(t, a, b, labels):
    """Two overlaid time courses as text, so the check needs no display."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    step = max(1, len(t) // 30)
    for i in range(0, len(t), step):
        cells = [" "] * 46
        for series, ch in zip((a, b), labels):
            j = int(round(45 * (series[i] - lo) / (hi - lo + 1e-12)))
            cells[j] = ch if cells[j] == " " else "*"
        zero = int(round(45 * (0 - lo) / (hi - lo + 1e-12)))
        if cells[zero] == " ":
            cells[zero] = "."
        mark = "onset >" if abs(t[i]) < 0.06 else "       "
        print(f"  {mark}{t[i]:6.1f}s  {''.join(cells)}  {a[i]:+.4f}")


def main():
    real = load_all(DATA)
    if not real:
        sys.exit(f"no data in {DATA}; run build_dataset.py first")
    t, hbo, hbr, r_true, med_true = report(real, "reconstructed onsets (startT + Trigger3)")

    print("\n  group HbO (O) and HbR (R) vs time, "
          f"range {min(hbo.min(), hbr.min()):+.3f} to {max(hbo.max(), hbr.max()):+.3f} uM")
    plot(t, hbo, hbr, "OR")

    null = load_all(NULL)
    if not null:
        print(f"\n  ! no null set in {NULL}; run: python build_dataset.py --null")
        return
    tn, hbon, hbrn, r_null, med_null = report(null, "null control (random onsets)")

    print("\n  null-set HbO (O) and HbR (R) vs time, "
          f"range {min(hbon.min(), hbrn.min()):+.3f} to {max(hbon.max(), hbrn.max()):+.3f} uM")
    plot(tn, hbon, hbrn, "OR")

    print("\n=== verdict ===")
    print(f"  group HbO/HbR correlation   true {r_true:+.3f}   null {r_null:+.3f}")
    print(f"  median subject correlation  true {med_true:+.3f}   null {med_null:+.3f}")
    print(f"  peak HbO amplitude          true {np.abs(hbo).max():.4f} uM   "
          f"null {np.abs(hbon).max():.4f} uM   "
          f"ratio {np.abs(hbo).max() / max(np.abs(hbon).max(), 1e-12):.1f}x")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
