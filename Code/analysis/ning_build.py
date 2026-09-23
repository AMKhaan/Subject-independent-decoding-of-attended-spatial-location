"""
Rebuild the data layer the way Ning et al. (2024) built it, so their published
per-subject accuracies become reproducible rather than merely quoted.

Why this exists
---------------
`build_dataset.py` epochs at ``Trigger3``, the movie onset. Ning et al. epoch at the
*cue*, and every decoding window in their paper starts there:

    "The features were used from an incremental window where all windows start at
     0th second (cue onset). Window lengths tested here are 0 to 0.1, 0.2, 0.5, 1,
     1.5, 2, 3, 4, and 5 s."                                     -- Classification, p.6

The cue (Trigger2) leads the movie (Trigger3) by **3.19 s**, fixed: sd 48 ms over all
1170 trials, with 1110 of them inside a single 0.1 s bin. It is frame-timing noise,
not design jitter. So their 0-5 s cue-locked window runs from 3.19 s *before* movie
onset to 1.81 s *after* it: mostly pre-stimulus, but it does overrun movie onset.

(An earlier version of this docstring said the cue leads the movie by 5.5-7.0 s and
concluded that their window "closes before the movie starts". That was wrong. The
5.5-7.0 s figure is Trigger3 - Trigger1, trial start to movie, and Trigger1 is the
fixation onset, not the cue. The same error appears in build_dataset.py:49-50 and
manuscript/paper.md:455-456 and should be corrected there too. Note also that Ning
et al. describe the cue as 2 s with the movie starting at 2 s, which does not match
the measured 3.19 s -- an unexplained discrepancy worth reporting.)

Because the window is mostly pre-stimulus, it weights cue-driven orienting heavily.
Our own time-resolved analysis finds no decodable signal that early -- the one fully
pre-onset window sits at 31.94%, below chance -- so either their feature path
extracts something our binned means do not, or their nested CV is optimistic. That
tension is exactly what this script exists to resolve, and it is a sharper question
than the one the wrong premise posed.

Three further differences from `build_dataset.py`, all reproduced here and all
switchable so their individual contributions can be measured:

  * DPF is 1, not 6 (their stated value; concentration is then in Molar-mm). This is
    a constant rescaling and cannot move a classification score, but it makes the
    saved units theirs.
  * Channels are pruned per subject at SNR 1.5, SNR being mean/std of the raw
    intensity. `build_dataset.py` instead keeps a fixed 28-channel intersection.
  * Motion artifacts are corrected with targeted PCA (Yucel et al. 2014) before
    filtering. `build_dataset.py` does no motion correction at all.

What is deliberately *not* done here
------------------------------------
Short-separation regression. Ning et al. estimate the SS coefficients inside the
cross-validation fold and apply them to the held-out trials:

    "we fitted the GLM model to a training dataset and estimated regression
     coefficients using the Ordinary Least Squares (OLS) method. Then the short
     separation regression coefficients (SS coefficients) estimated from the
     training set are used for the test set..."                            -- p.5

Doing it here, once, over all trials, would leak test-set information into the
regressor weights -- the exact failure von Luhmann et al. (2020) describe and that
they cite. So this script saves the *continuous* long and short channels plus the
cue onsets, and `ning_decode.py` does the GLM inside each fold.

Output: one .npz per subject holding continuous HbT for the surviving long channels,
continuous HbT for the short channels, the cue onsets, labels, and the behavioural
correctness mask.

Usage:  python ning_build.py [--root PATH] [--out PATH] [--no-motion] [--no-prune]

Default output derived/ds004830/ning/ (see layout.py). The console output is the build
log, saved as results/ds004830/ning_build/ds004830_<variant>_ning_build.log.
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, decimate

import layout
import paths
from build_dataset_v2 import read_events, LABELS, BIDS_RUNS, DEFAULT_RUNS
from build_dataset import (
    experiment_dirs, load_manifest, load_triggers, load_labels,
    probe_geometry, FS_RAW,
)

FS_OUT = 10.0          # Hz
BAND = (0.01, 0.5)     # Hz, 3rd-order zero-phase Butterworth -- their stated filter
DPF = (1.0, 1.0)       # "The differential path length factor was held fixed at 1"
SS_MAX_MM = 15.0       # 8 mm short channels vs 30 mm long ones; 15 separates them
CUE_TRIGGER = "Trigger2"
CUE_LEAD = 3.19        # s, Trigger3 - Trigger2, measured over all trials (sd 48 ms)
SNR_CUTOFF = 1.5       # "a cutoff of SNR = 1.5, where SNR was estimated as the mean
                       # divided by the standard deviation of raw intensity"
PRUNE_EXCLUDE = 20     # subjects with >= 20 channels pruned are excluded by them

# Behavioural exclusion: "correctness - if they scored below 70% in both face and
# transcript identification tasks". sub-25 sits at 48.89% and is the only subject in
# our derivatives tree that this removes, taking 12 subjects down to their 11.
BEHAVIOUR_CUTOFF = 0.70

# Targeted PCA parameters follow Homer3's hmrR_MotionCorrectPCArecurse defaults,
# which is the implementation Yucel et al. (2014) describe and Ning et al. cite.
MOTION_TMOTION = 0.5   # s, window for the moving std/amplitude test
MOTION_TMASK = 1.0     # s, padding applied around a detected artifact
MOTION_STDEV = 50.0    # threshold on the ratio of local to global std
MOTION_AMP = 0.5       # threshold on absolute OD change within tMotion
MOTION_NSV = 0.97      # variance fraction removed from the artifact subspace
MOTION_MAXITER = 5


def channel_snr(d):
    """Mean over standard deviation of the raw intensity, per wavelength channel."""
    sd = d.std(axis=0)
    sd[sd == 0] = np.inf
    return d.mean(axis=0) / sd


def find_motion(od, fs):
    """Boolean mask of samples belonging to a motion artifact, per Homer3's test.

    A sample is flagged when either the local standard deviation over ``tMotion``
    exceeds ``MOTION_STDEV`` times the channel's own global std, or the peak-to-peak
    OD change over the same span exceeds ``MOTION_AMP``. Flags are then dilated by
    ``tMask`` because the correction has to cover the recovery, not just the spike.
    """
    n, nch = od.shape
    w = max(2, int(round(MOTION_TMOTION * fs)))
    pad = int(round(MOTION_TMASK * fs))
    mask = np.zeros(n, dtype=bool)

    # cumulative sums give the windowed mean and mean-square in one pass, which
    # matters because this runs per channel over a full recording
    for c in range(nch):
        x = od[:, c]
        cs = np.concatenate([[0.0], np.cumsum(x)])
        cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
        m = (cs[w:] - cs[:-w]) / w
        m2 = (cs2[w:] - cs2[:-w]) / w
        local_std = np.sqrt(np.maximum(m2 - m * m, 0.0))
        gstd = x.std()
        hit_std = local_std > MOTION_STDEV * gstd if gstd > 0 else np.zeros_like(local_std, bool)

        # peak-to-peak inside each window, via a strided view
        sv = np.lib.stride_tricks.sliding_window_view(x, w)
        hit_amp = (sv.max(axis=1) - sv.min(axis=1)) > MOTION_AMP

        hit = hit_std | hit_amp
        idx = np.flatnonzero(hit)
        for i in idx:
            mask[max(0, i - pad):min(n, i + w + pad)] = True
    return mask


def targeted_pca(od, fs):
    """Yucel et al. (2014): run PCA on the motion segments only, remove the top
    components, and repeat while artifacts remain.

    Restricting the PCA to flagged segments is the whole point of the method -- a
    PCA over the entire recording would happily delete the evoked response, which is
    also a component shared across channels.
    """
    out = od.copy()
    for _ in range(MOTION_MAXITER):
        mask = find_motion(out, fs)
        if not mask.any():
            break
        seg = out[mask]
        seg = seg - seg.mean(axis=0)
        # economy SVD of the artifact subspace
        u, s, vt = np.linalg.svd(seg, full_matrices=False)
        var = np.cumsum(s ** 2) / max(np.sum(s ** 2), 1e-30)
        k = int(np.searchsorted(var, MOTION_NSV) + 1)
        k = min(k, vt.shape[0])
        proj = vt[:k]                                  # (k, channels)
        # remove the artifact subspace from the flagged samples only
        block = out[mask]
        mu = block.mean(axis=0)
        centred = block - mu
        out[mask] = centred - (centred @ proj.T) @ proj + mu
    return out


def process_run(nirs_file, pairs, dist_ch, ext, keep_wl, motion=True):
    """Raw intensity -> OD -> targeted PCA -> band-pass -> HbT, at FS_OUT.

    Returns HbT for every wavelength-paired channel plus the run duration; channel
    selection happens in the caller so the SNR mask can be computed once per subject.
    """
    m = sio.loadmat(nirs_file, variable_names=["d", "t"])
    d = np.asarray(m["d"], dtype=float)
    t = np.asarray(m["t"], dtype=float).flatten()

    floor = np.maximum(np.abs(d).mean(axis=0) * 1e-6, 1e-12)
    d = np.maximum(d, floor)
    od = -np.log(d / d.mean(axis=0))                   # mean signal as the reference

    if motion:
        od = targeted_pca(od, FS_RAW)

    b, a = butter(3, [BAND[0] / (FS_RAW / 2), BAND[1] / (FS_RAW / 2)], btype="band")
    od = filtfilt(b, a, od, axis=0)

    od1 = od[:, pairs[:, 0]]
    od2 = od[:, pairs[:, 1]]
    einv = np.linalg.pinv(ext)
    L = dist_ch / 10.0                                 # mm -> cm
    hbo = np.empty_like(od1)
    hbr = np.empty_like(od1)
    for c in range(od1.shape[1]):
        a1 = od1[:, c] / (L[c] * DPF[0])
        a2 = od2[:, c] / (L[c] * DPF[1])
        hbo[:, c] = einv[0, 0] * a1 + einv[0, 1] * a2
        hbr[:, c] = einv[1, 0] * a1 + einv[1, 1] * a2
    hbt = (hbo + hbr) * 1e6                            # HbT, micromolar

    q = int(round(FS_RAW / FS_OUT))
    hbt = decimate(hbt, q, axis=0, ftype="fir", zero_phase=True)
    return hbt, t[-1], channel_snr(d)


def build_subject(exp, path, out_dir, motion=True, prune=True, verbose=True,
                  events=None, anchor="movie"):
    runs, start_t, _ = load_manifest(path, exp)
    if events is None:
        trig, correct = load_triggers(path)
        labels, cond = load_labels(path)
        onsets = start_t + trig[CUE_TRIGGER]            # cue, not movie onset
        onsets_stim = start_t + trig["Trigger3"]        # movie onset, for the ablation
        n = min(len(labels), len(onsets))
        labels, cond, correct = labels[:n], cond[:n], correct[:n]
        onsets, onsets_stim = onsets[:n], onsets_stim[:n]

    first = os.path.join(path, runs[0] + ".nirs")
    pairs, dist_ch, ident = probe_geometry(first)
    ext = np.array(sio.loadmat(first, variable_names=["SD"])["SD"][0, 0]["extCoef"],
                   dtype=float)

    blocks, snrs, offset, durations = [], [], 0.0, []
    for r in runs:
        f = os.path.join(path, r + ".nirs")
        if not os.path.exists(f):
            print(f"  ! {exp}: missing run {r}, skipped")
            continue
        hbt, dur, snr = process_run(f, pairs, dist_ch, ext, None, motion=motion)
        blocks.append(hbt)
        snrs.append(snr)
        durations.append(dur)
        offset += dur
    data = np.concatenate(blocks, axis=0)

    if events is not None:
        # Official v2.0.0 onsets, placed on the concatenated timeline exactly as
        # build_dataset_v2.py does (BIDS restarts each run's clock at zero). Labels and
        # correctness come from the official file too; it lists competing-talker
        # trials only, so every trial is condition 1.
        num = exp.replace("Experiment", "")
        ev = read_events(num, events)
        starts = dict(zip([r for r in runs if os.path.exists(
            os.path.join(path, r + ".nirs"))], np.cumsum([0.0] + durations[:-1])))
        run_off = {b: starts.get(r, 0.0)
                   for b, r in zip(BIDS_RUNS.get(num, DEFAULT_RUNS), runs)}
        official = np.array([o + run_off.get(rn, 0.0) for rn, o, _t, _c in ev])
        labels = np.array([LABELS[t] for _r, _o, t, _c in ev], dtype=int)
        correct = np.array([bool(c) for _r, _o, _t, c in ev])
        cond = np.ones(len(ev), dtype=int)
        # which event the official onset marks is not documented; check_onset_anchor.py
        # favours the movie. Both are built so the conclusion does not rest on the guess.
        onsets = official - CUE_LEAD if anchor == "movie" else official
        onsets_stim = onsets + CUE_LEAD

    # SNR is defined on the raw wavelength channels; a paired channel survives only
    # if both of its wavelengths do, which is what pruning a "channel" has to mean
    snr = np.mean(snrs, axis=0)
    ok_wl = snr >= SNR_CUTOFF
    ok_ch = ok_wl[pairs[:, 0]] & ok_wl[pairs[:, 1]]

    is_short = dist_ch < SS_MAX_MM
    long_all = ~is_short
    n_pruned = int((long_all & ~ok_ch).sum())

    keep_long = long_all & ok_ch if prune else long_all
    keep_short = is_short & ok_ch if prune else is_short
    if keep_short.sum() == 0:                           # no usable SS channel left
        keep_short = is_short

    sub = exp.replace("Experiment", "sub-")
    acc = float(correct.mean())

    # Their two exclusion criteria, applied and recorded rather than silently obeyed,
    # because the paper's headline mean turns out to include the excluded subjects.
    excl = []
    if n_pruned >= PRUNE_EXCLUDE:
        excl.append(f"pruned={n_pruned}")
    if acc < BEHAVIOUR_CUTOFF:
        excl.append(f"behaviour={acc:.2%}")

    keep_trial = cond == 1                              # competing-talker trials only
    np.savez_compressed(
        os.path.join(out_dir, sub + ".npz"),
        long=data[:, keep_long].astype(np.float32),
        short=data[:, keep_short].astype(np.float32),
        onsets=onsets[keep_trial], onsets_stim=onsets_stim[keep_trial],
        y=labels[keep_trial].astype(np.int8),
        correct=correct[keep_trial],
        subject=sub, fs=FS_OUT,
        dist_long=dist_ch[keep_long], dist_short=dist_ch[keep_short],
        channels_long=np.array([ident[i] for i in np.flatnonzero(keep_long)]),
        n_pruned=n_pruned, behaviour=acc,
        excluded_by_ning="; ".join(excl),
        run_durations=np.array(durations),
    )
    if verbose:
        tag = ("  EXCLUDED (" + ", ".join(excl) + ")") if excl else ""
        lag = float(np.median(onsets_stim[keep_trial] - onsets[keep_trial]))
        print(f"  {sub}: long={int(keep_long.sum())} short={int(keep_short.sum())} "
              f"trials={int(keep_trial.sum())} pruned={n_pruned} acc={acc:.2%} "
              f"cue->movie={lag:.1f}s{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--out", default=layout.derived("ds004830", "ning"))
    ap.add_argument("--no-motion", action="store_true",
                    help="skip targeted PCA, to measure what motion correction is worth")
    ap.add_argument("--no-prune", action="store_true",
                    help="skip SNR pruning, to measure what channel pruning is worth")
    ap.add_argument("--events", default=None,
                    help="official v2.0.0 events directory; without it the v1 "
                         "PsychToolbox reconstruction is used (misaligned, see STATUS.md)")
    ap.add_argument("--anchor", choices=("movie", "cue"), default="movie",
                    help="which trial event the official onset marks (with --events)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.events:
        print(f"ONSETS: official v2.0.0 events ({args.events}), official onset taken as "
              f"the {args.anchor}; cue = onset{' - ' + str(CUE_LEAD) + ' s' if args.anchor == 'movie' else ''}")

    print(f"cue-locked ({CUE_TRIGGER}) HbT at {FS_OUT:g} Hz, DPF={DPF[0]:g}, "
          f"motion={'off' if args.no_motion else 'targeted PCA'}, "
          f"prune={'off' if args.no_prune else f'SNR>={SNR_CUTOFF}'}")
    for exp, path in experiment_dirs(args.root):
        try:
            build_subject(exp, path, args.out,
                          motion=not args.no_motion, prune=not args.no_prune,
                          events=args.events, anchor=args.anchor)
        except Exception as e:
            print(f"  ! {exp} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
