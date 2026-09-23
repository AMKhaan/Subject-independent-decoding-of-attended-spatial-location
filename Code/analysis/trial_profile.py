"""
Where in the trial does the haemodynamic response actually live?

lag_sweep.py found the largest evoked amplitude 8 s *before* Trigger3, not at it.
Two readings: either the onset reconstruction is wrong, or the response begins
earlier in the trial than stimulus onset -- at the cue, or at trial start -- so
that a [-2, 0] s baseline taken relative to Trigger3 already sits on the rising
edge and subtracts away part of the response.

This settles it by epoching the entire trial rather than a 14 s window. Lock to
Trigger1 (trial start), take a genuinely pre-trial baseline, and read off where
the rise begins and where it peaks relative to the four triggers.

  Trigger1  trial start
  Trigger2  cue          (~T1 + 2.5 s)
  Trigger3  stimulus on  (~T1 + 5.7 s)
  Trigger4  stimulus off (variable, 2-28 s after T3)

Usage:  python trial_profile.py
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio

import build_dataset as bd
import paths
from lag_sweep import continuous

WIN = (-8.0, 30.0)     # s relative to Trigger1
BASE = (-8.0, -5.0)    # s, before the trial starts


def subject_triggers(path, exp):
    runs, start_t, _ = bd.load_manifest(path, exp)
    trig, _ = bd.load_triggers(path)
    labels, cond = bd.load_labels(path)
    n = min(len(labels), len(trig["Trigger1"]))
    m = cond[:n] == 1                       # competing-talker trials only
    return {k: (start_t + v[:n])[m] for k, v in trig.items()}, labels[:n][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.dataset_root())
    args = ap.parse_args()

    channels = bd.common_long_channels(args.root)
    n_ch = len(channels)

    pre = int(round(WIN[0] * bd.FS_OUT))
    post = int(round(WIN[1] * bd.FS_OUT))
    n_times = post - pre
    b0 = int(round((BASE[0] - WIN[0]) * bd.FS_OUT))
    b1 = int(round((BASE[1] - WIN[0]) * bd.FS_OUT))
    t = WIN[0] + np.arange(n_times) / bd.FS_OUT

    acc, ntr = np.zeros((n_times, 2 * n_ch)), 0
    lags = {k: [] for k in ("Trigger2", "Trigger3", "Trigger4")}
    iti = []

    print("preprocessing subjects once...")
    for exp, path in bd.experiment_dirs(args.root):
        data, _ = continuous(exp, path, channels)
        trig, _ = subject_triggers(path, exp)
        t1 = trig["Trigger1"]
        for k in lags:
            lags[k].append(trig[k] - t1)
        iti.append(np.diff(t1))

        k = 0
        for on in t1:
            s0 = int(round(on * bd.FS_OUT)) + pre
            s1 = s0 + n_times
            if s0 < 0 or s1 > len(data):
                continue
            seg = data[s0:s1]
            acc += seg - seg[b0:b1].mean(axis=0)
            k += 1
        ntr += k
        print(f"  {exp.replace('Experiment', 'sub-')}: {k}/{len(t1)} trials usable")

    g = acc / ntr
    hbo, hbr = g[:, :n_ch].mean(axis=1), g[:, n_ch:].mean(axis=1)

    print(f"\n=== trial timing across all {sum(len(x) for x in lags['Trigger2'])} trials ===")
    for k in ("Trigger2", "Trigger3", "Trigger4"):
        v = np.concatenate(lags[k])
        print(f"  {k} - Trigger1:  median {np.median(v):6.2f} s   "
              f"range {v.min():6.2f} to {v.max():6.2f} s")
    v = np.concatenate(iti)
    print(f"  Trigger1 spacing:  median {np.median(v):6.2f} s   "
          f"range {v.min():6.2f} to {v.max():6.2f} s")

    m2 = np.median(np.concatenate(lags["Trigger2"]))
    m3 = np.median(np.concatenate(lags["Trigger3"]))
    m4 = np.median(np.concatenate(lags["Trigger4"]))

    print(f"\n=== group trial-average, locked to Trigger1, {ntr} trials, "
          f"baseline {BASE[0]:+g}..{BASE[1]:+g} s ===")
    lo = min(hbo.min(), hbr.min())
    hi = max(hbo.max(), hbr.max())
    marks = {round(0.0, 1): "T1 trial start", round(m2, 1): "T2 cue",
             round(m3, 1): "T3 stimulus on", round(m4, 1): "T4 median stim off"}
    step = max(1, n_times // 48)
    for i in range(0, n_times, step):
        cells = [" "] * 50
        z = int(round(49 * (0 - lo) / (hi - lo + 1e-12)))
        cells[z] = "."
        for series, ch in ((hbo, "O"), (hbr, "R")):
            j = int(round(49 * (series[i] - lo) / (hi - lo + 1e-12)))
            cells[j] = ch if cells[j] in " ." else "*"
        tag = ""
        for mt, name in marks.items():
            if abs(t[i] - mt) < (step / bd.FS_OUT) / 2:
                tag = "  <-- " + name
        print(f"  {t[i]:6.1f}s  {''.join(cells)}  {hbo[i]:+.3f} {hbr[i]:+.3f}{tag}")

    postm = t >= 0
    r = np.corrcoef(hbo[postm], hbr[postm])[0, 1]
    ip = np.argmax(np.abs(hbo))
    # first sample after t=0 where HbO exceeds 4 SD of its own pre-trial baseline
    sd = hbo[b0:b1].std() if hbo[b0:b1].std() > 0 else 1e-9
    thr = 4 * hbo[:b1].std()
    above = np.where((t > 0) & (np.abs(hbo) > thr))[0]
    print(f"\n  HbO/HbR correlation over the trial: r = {r:+.3f}")
    print(f"  HbO peak: {hbo[ip]:+.3f} uM at {t[ip]:+.1f} s after Trigger1 "
          f"({t[ip] - m3:+.1f} s relative to Trigger3)")
    if len(above):
        print(f"  HbO first exceeds 4x pre-trial SD at {t[above[0]]:+.1f} s after Trigger1 "
              f"({t[above[0]] - m2:+.1f} s relative to the cue)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
