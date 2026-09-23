"""
Localise the trial onset by sweeping a global time shift.

validate_alignment.py compares the reconstructed onsets against epochs cut at
random times. That control is weak here: trials repeat about every 31 s and the
response window is 14 s, so a random onset lands inside a real trial a large
fraction of the time, and the null inherits some of the evoked response.

This is the stronger test. Apply a single global lag to every onset in every
subject and re-epoch. If onset = startT + Trigger3 is correct, the evoked
response is sharpest at lag 0 and degrades as the lag moves away in either
direction. A wrong constant offset would put the optimum somewhere else; no
stimulus locking at all would make the curve flat.

Two statistics per lag, both computed on the group trial-average:
  amp   peak |HbO| in the 0-12 s window (evoked amplitude)
  r     HbO/HbR correlation over 0-12 s (physiological plausibility; should be
        most negative at the true onset)

Usage:  python lag_sweep.py [--lags -20 20 0.5]
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio

import build_dataset as bd
import paths


def continuous(exp, path, channels):
    """Preprocess a subject once and return (data, onsets) in the concatenated timeline."""
    runs, start_t, _ = bd.load_manifest(path, exp)
    trig, _ = bd.load_triggers(path)
    labels, cond = bd.load_labels(path)
    onsets = start_t + trig[bd.TRIGGER]
    n = min(len(labels), len(onsets))
    onsets = onsets[:n][cond[:n] == 1]      # competing-talker trials only, as in build_dataset
    n = len(onsets)

    first = os.path.join(path, runs[0] + ".nirs")
    pairs, dist_ch, ident = bd.probe_geometry(first)
    ext = np.array(sio.loadmat(first, variable_names=["SD"])["SD"][0, 0]["extCoef"],
                   dtype=float)
    pos = {k: i for i, k in enumerate(ident)}
    sel = np.array([pos[k] for k in channels])

    blocks = []
    for r in runs:
        f = os.path.join(path, r + ".nirs")
        if os.path.exists(f):
            blk, _ = bd.process_run(f, pairs, dist_ch, ext, sel)
            blocks.append(blk)
    return np.concatenate(blocks, axis=0), onsets[:n]


def epoch_mean(data, onsets, lag):
    """Trial-average at the given lag. Returns (mean epoch, n trials used)."""
    pre = int(round(bd.EPOCH[0] * bd.FS_OUT))
    post = int(round(bd.EPOCH[1] * bd.FS_OUT))
    n_times = post - pre
    b0 = int(round((bd.BASELINE[0] - bd.EPOCH[0]) * bd.FS_OUT))
    b1 = int(round((bd.BASELINE[1] - bd.EPOCH[0]) * bd.FS_OUT))

    acc, k = np.zeros((n_times, data.shape[1])), 0
    for on in onsets:
        s0 = int(round((on + lag) * bd.FS_OUT)) + pre
        s1 = s0 + n_times
        if s0 < 0 or s1 > len(data):
            continue
        seg = data[s0:s1]
        acc += seg - seg[b0:b1].mean(axis=0)
        k += 1
    return (acc / max(k, 1)), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--lags", nargs=3, type=float, default=[-20.0, 20.0, 1.0],
                    metavar=("LO", "HI", "STEP"))
    args = ap.parse_args()

    channels = bd.common_long_channels(args.root)
    n_ch = len(channels)

    print("preprocessing subjects once...")
    subs = []
    for exp, path in bd.experiment_dirs(args.root):
        try:
            subs.append(continuous(exp, path, channels))
            print(f"  {exp.replace('Experiment', 'sub-')}: "
                  f"{subs[-1][0].shape[0] / bd.FS_OUT:.0f} s, {len(subs[-1][1])} trials")
        except Exception as e:
            print(f"  ! {exp} FAILED: {type(e).__name__}: {e}")

    lags = np.arange(args.lags[0], args.lags[1] + 1e-9, args.lags[2])
    t = bd.EPOCH[0] + np.arange(int(round((bd.EPOCH[1] - bd.EPOCH[0]) * bd.FS_OUT))) / bd.FS_OUT
    post = t >= 0

    rows = []
    for lag in lags:
        tot, ntr = 0.0, 0
        for data, onsets in subs:
            m, k = epoch_mean(data, onsets, lag)
            tot = tot + m * k
            ntr += k
        g = tot / ntr
        hbo, hbr = g[:, :n_ch].mean(axis=1), g[:, n_ch:].mean(axis=1)
        amp = np.abs(hbo[post]).max()
        r = np.corrcoef(hbo[post], hbr[post])[0, 1]
        rows.append((lag, amp, r, ntr))

    amps = np.array([x[1] for x in rows])
    rs = np.array([x[2] for x in rows])
    print(f"\n=== global lag sweep, {rows[0][3]} trials per lag ===")
    print(f"{'lag':>7}  {'peak |HbO| uM':<26} {'HbO/HbR r':<26}")
    for lag, amp, r, _ in rows:
        bar = "#" * int(round(24 * amp / amps.max()))
        # r runs from about -1 to +1; centre the bar on zero
        j = int(round(12 * (1 + max(min(r, 1.0), -1.0))))
        rbar = [" "] * 25
        rbar[12] = "|"
        rbar[j] = "*"
        mark = "  <-- reconstructed onset" if abs(lag) < 1e-9 else ""
        print(f"{lag:>6.1f}s  {bar:<24} {amp:.3f}  {''.join(rbar)} {r:+.3f}{mark}")

    i_amp = int(np.argmax(amps))
    i_r = int(np.argmin(rs))
    print(f"\n  peak amplitude at lag {lags[i_amp]:+.1f} s  ({amps[i_amp]:.3f} uM)")
    print(f"  most negative HbO/HbR r at lag {lags[i_r]:+.1f} s  ({rs[i_r]:+.3f})")
    z = int(np.argmin(np.abs(lags)))
    print(f"  at lag 0: amp {amps[z]:.3f} uM ({100 * amps[z] / amps.max():.0f}% of max), "
          f"r {rs[z]:+.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
