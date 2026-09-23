"""
Build the data layer for the ds004830 spatial-attention decoding study.

The epoching here is trial-locked by construction: every epoch is cut at a real
trial onset and carries that trial's own cued-attention label. That sounds like a
statement of the obvious, and it is only worth stating because the failure mode it
rules out -- indexing a trial-indexed label array by channel number -- is easy to
write, silent at runtime, and worth 46 percentage points (see leakage_demo.py).

Pipeline per subject
--------------------
  raw intensity (.nirs, 50 Hz, 42 ch x 2 wavelengths)
    -> optical density
    -> bandpass 0.01-0.5 Hz (Butterworth, zero-phase)
    -> modified Beer-Lambert -> HbO / HbR
    -> short-separation regression (8 mm channels regressed out of the long ones)
    -> restrict to the 28 long channels shared by all 12 subjects
    -> decimate to 10 Hz
    -> epoch at startT + Trigger3 (stimulus onset), window [-2, +12] s
    -> baseline-correct on [-2, 0] s

Output: one .npz per subject with X (n_trials, 140, 56), y (n_trials,), plus
timing/QC metadata. Subject identity is preserved so splits can be leave-one-
subject-out rather than random.

--null repeats the whole pipeline with epochs cut at random times, which is the
control figure for the claim that the reconstructed onsets are correct.

Usage:  python build_dataset.py [--root PATH] [--out PATH] [--null]
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, decimate

import paths

FS_RAW = 50.0          # Hz, native sampling rate of the Techen system
FS_OUT = 10.0          # Hz, matches the rate used by Ning et al.
BAND = (0.01, 0.5)     # Hz
DPF = (6.0, 6.0)       # differential pathlength factor for (690, 830) nm
SS_MAX_MM = 15.0       # channels shorter than this are short-separation
EPOCH = (-2.0, 12.0)   # s relative to stimulus onset; the group HRF has returned
                       # toward baseline by 12 s, and the shortest inter-onset
                       # interval in the dataset is 22.9 s (median 27.6 s), so with
                       # the cue leading its own onset by 5.5-7.0 s the next trial
                       # cannot intrude before ~16 s -- ~4 s of margin at worst
BASELINE = (-2.0, 0.0) # s
TRIGGER = "Trigger3"   # T1 trial start, T2 cue, T3 stimulus onset, T4 stimulus end

# Second column of indexMoviesTest is the cued location. The dataset docx has a
# typo ("1 = right, 2 = right, 3 = center"), so the left/right assignment between
# 1 and 2 is not established from documentation. 3 = center is unambiguous.
# The 3-class accuracy is invariant to swapping 1 and 2; only the interpretation
# of per-class results depends on it. See check_lateralisation.py.
LABEL_NAMES = {1: "lateral_A", 2: "lateral_B", 3: "center"}


def experiment_dirs(root):
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name.startswith("Experiment"):
            out.append((name, path))
    return out


def load_manifest(path, exp):
    """The `s` helper struct: run filenames, label file, and experiment start."""
    num = exp.replace("Experiment", "")
    f = os.path.join(path, num + ".mat")
    if not os.path.exists(f):
        cands = [x for x in os.listdir(path)
                 if x.endswith(".mat") and x[:-4].isdigit()]
        if not cands:
            raise FileNotFoundError("no helper struct in " + path)
        f = os.path.join(path, cands[0])
    s = sio.loadmat(f)["s"]
    runs = [str(np.array(x).flatten()[0]) for x in np.array(s["fName"][0, 0]).flatten()]
    start_t = float(np.array(s["startT"][0, 0]).flatten()[0])
    end_t = float(np.array(s["endT"][0, 0]).flatten()[0])
    return runs, start_t, end_t


def load_triggers(path):
    f = [x for x in os.listdir(path) if x.startswith("response_3M")]
    if not f:
        f = [x for x in os.listdir(path) if x.startswith("responses")]
    r = sio.loadmat(os.path.join(path, f[0]))
    cfg = r["cfg"]
    trig = {k: np.array(cfg[k][0, 0]).flatten().astype(float)
            for k in ("Trigger1", "Trigger2", "Trigger3", "Trigger4")}
    # A trial counts as correct only if the participant identified BOTH the target
    # speaker's face (V) and their transcript (A) -- Ning et al.'s stated criterion.
    # Scoring on the transcript alone is more lenient and inflates the subset by 15
    # trials. The conjunction is what reproduces their published per-subject
    # behavioural accuracies exactly, for all twelve subjects; see check_behaviour.py.
    ok = ((np.array(r["responsesA"]).flatten() == np.array(r["correctRespA"]).flatten())
          & (np.array(r["responsesV"]).flatten() == np.array(r["correctRespV"]).flatten()))
    return trig, ok


def load_labels(path):
    """Cued location and task condition for every trial.

    indexMoviesTest columns: 1 target movie, 2 cued location, 3 audio-visual
    congruence, 4 masker configuration, 5 condition (0 single talker, 1 competing
    talkers). Ning et al. keep only condition 1 -- `multipleIndex = movieIdx(:,5)==1`
    in keepCorrectTrials.m -- because the single-talker trials are a different task
    with no spatial competition to resolve. Eleven subjects have only condition 1;
    Experiment08 additionally ran 90 single-talker trials.
    """
    f = [x for x in os.listdir(path) if x.startswith("iniList")][0]
    idx = np.array(sio.loadmat(os.path.join(path, f))["indexMoviesTest"])
    return idx[:, 1].astype(int), idx[:, 4].astype(int)


def probe_geometry(nirs_file):
    """Return wavelength-paired channels, their source-detector distances, and
    their (source, detector) identity so channels can be matched across subjects."""
    sd = sio.loadmat(nirs_file, variable_names=["SD"])["SD"][0, 0]
    src = np.array(sd["SrcPos"], dtype=float)
    det = np.array(sd["DetPos"], dtype=float)
    ml = np.array(sd["MeasList"], dtype=int)
    dist = np.linalg.norm(src[ml[:, 0] - 1] - det[ml[:, 1] - 1], axis=1)
    wl1 = np.where(ml[:, 3] == 1)[0]
    wl2 = np.where(ml[:, 3] == 2)[0]
    # pair the two wavelengths by (source, detector)
    key1 = {(ml[i, 0], ml[i, 1]): i for i in wl1}
    pairs, dist_ch, ident = [], [], []
    for j in wl2:
        k = (ml[j, 0], ml[j, 1])
        pairs.append((key1[k], j))
        dist_ch.append(dist[key1[k]])
        ident.append(k)
    return np.array(pairs), np.array(dist_ch), ident


def common_long_channels(root):
    """(source, detector) pairs present as long-separation channels in every subject.

    Subject 08 was recorded with a larger probe (14 sources / 29 detectors) than
    the other eleven (12 / 23), so a cross-subject model has to be restricted to
    the shared montage. That intersection is 28 channels.
    """
    sets = []
    for _, path in experiment_dirs(root):
        nirs = sorted(x for x in os.listdir(path) if x.endswith(".nirs"))
        if not nirs:
            continue
        _, dist_ch, ident = probe_geometry(os.path.join(path, nirs[0]))
        sets.append({k for k, d in zip(ident, dist_ch) if d >= SS_MAX_MM})
    return sorted(set.intersection(*sets))


def od_to_conc(od1, od2, dist_mm, ext):
    """Modified Beer-Lambert. od* are (n_times, n_channels) for each wavelength."""
    einv = np.linalg.pinv(ext)                       # (chromophore, wavelength)
    L = (dist_mm / 10.0)                             # mm -> cm
    hbo = np.empty_like(od1)
    hbr = np.empty_like(od1)
    for c in range(od1.shape[1]):
        a = od1[:, c] / (L[c] * DPF[0])
        b = od2[:, c] / (L[c] * DPF[1])
        hbo[:, c] = einv[0, 0] * a + einv[0, 1] * b
        hbr[:, c] = einv[1, 0] * a + einv[1, 1] * b
    return hbo * 1e6, hbr * 1e6                      # molar -> micromolar


def regress_short(long_sig, short_sig):
    """Least-squares removal of the mean short-separation signal from each long channel."""
    ref = short_sig.mean(axis=1)
    A = np.column_stack([ref, np.ones_like(ref)])
    beta, *_ = np.linalg.lstsq(A, long_sig, rcond=None)
    return long_sig - A @ beta


def process_run(nirs_file, pairs, dist_ch, ext, keep):
    m = sio.loadmat(nirs_file, variable_names=["d", "t"])
    d = np.asarray(m["d"], dtype=float)
    t = np.asarray(m["t"], dtype=float).flatten()

    # optical density, guarding against non-positive intensities
    floor = np.maximum(np.abs(d).mean(axis=0) * 1e-6, 1e-12)
    d = np.maximum(d, floor)
    od = -np.log(d / d.mean(axis=0))

    b, a = butter(3, [BAND[0] / (FS_RAW / 2), BAND[1] / (FS_RAW / 2)], btype="band")
    od = filtfilt(b, a, od, axis=0)

    hbo, hbr = od_to_conc(od[:, pairs[:, 0]], od[:, pairs[:, 1]], dist_ch, ext)

    is_short = dist_ch < SS_MAX_MM
    hbo = regress_short(hbo[:, keep], hbo[:, is_short])
    hbr = regress_short(hbr[:, keep], hbr[:, is_short])

    q = int(round(FS_RAW / FS_OUT))
    hbo = decimate(hbo, q, axis=0, ftype="fir", zero_phase=True)
    hbr = decimate(hbr, q, axis=0, ftype="fir", zero_phase=True)
    # Duration from the sample count, NOT t[-1]. Most runs start their clock at 0, so
    # the two agree; sub-08's 04.nirs continues 03.nirs's clock and runs 1799.98 ->
    # 3599.98, so t[-1] reports a 1800 s run as 3600 s and displaces everything after
    # it by half an hour. The concatenation itself is by sample count and was always
    # right; it is the run-start offsets derived from this value that were wrong.
    return np.concatenate([hbo, hbr], axis=1), len(t) / FS_RAW


def build_subject(exp, path, out_dir, channels, null=False, verbose=True):
    runs, start_t, end_t = load_manifest(path, exp)
    trig, correct = load_triggers(path)
    labels, cond = load_labels(path)
    onsets = start_t + trig[TRIGGER]

    n = min(len(labels), len(onsets))
    if len(labels) != len(onsets):
        print(f"  ! {exp}: {len(labels)} labels vs {len(onsets)} triggers, using {n}")
    labels, cond, onsets = labels[:n], cond[:n], onsets[:n]
    correct = correct[:n]

    first = os.path.join(path, runs[0] + ".nirs")
    pairs, dist_ch, ident = probe_geometry(first)
    ext = np.array(sio.loadmat(first, variable_names=["SD"])["SD"][0, 0]["extCoef"], dtype=float)

    # index the shared montage in a fixed order so column j means the same
    # source-detector pair in every subject
    pos = {k: i for i, k in enumerate(ident)}
    sel = np.array([pos[k] for k in channels])

    blocks, bounds, offset = [], [], 0.0
    for r in runs:
        f = os.path.join(path, r + ".nirs")
        if not os.path.exists(f):
            print(f"  ! {exp}: missing run {r}, skipped")
            continue
        blk, dur = process_run(f, pairs, dist_ch, ext, sel)
        blocks.append(blk)
        bounds.append(offset)
        offset += dur
    data = np.concatenate(blocks, axis=0)
    bounds = np.array(bounds[1:])           # interior run boundaries, in seconds

    pre = int(round(EPOCH[0] * FS_OUT))
    post = int(round(EPOCH[1] * FS_OUT))
    n_times = post - pre
    b0 = int(round((BASELINE[0] - EPOCH[0]) * FS_OUT))
    b1 = int(round((BASELINE[1] - EPOCH[0]) * FS_OUT))

    if null:
        # Null control: cut the same number of epochs at random times in the same
        # recording. Everything downstream of the onset -- filtering, Beer-Lambert,
        # short-separation regression, baseline correction -- is identical, so any
        # evoked response that survives here is an artefact of the pipeline rather
        # than evidence of stimulus locking.
        rng = np.random.default_rng(abs(hash(exp)) % (2 ** 32))
        lo = -EPOCH[0]
        hi = len(data) / FS_OUT - EPOCH[1]
        onsets = np.sort(rng.uniform(lo, hi, size=len(onsets)))

    X, y, used, spans = [], [], [], []
    for i, (on, lab) in enumerate(zip(onsets, labels)):
        if cond[i] != 1:                    # single-talker trials are a different task
            continue
        s0 = int(round(on * FS_OUT)) + pre
        s1 = s0 + n_times
        if s0 < 0 or s1 > len(data):
            continue
        seg = data[s0:s1]
        seg = seg - seg[b0:b1].mean(axis=0)
        X.append(seg)
        y.append(lab)
        used.append(i)
        spans.append(bool(np.any((bounds > on + EPOCH[0]) & (bounds < on + EPOCH[1]))))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    used = np.asarray(used)
    spans = np.asarray(spans)

    sub = exp.replace("Experiment", "sub-")
    np.savez_compressed(
        os.path.join(out_dir, sub + ".npz"),
        X=X, y=y, subject=sub, trial_index=used,
        crosses_run_boundary=spans,
        correct=correct[used],
        onsets=onsets[used], start_t=start_t, end_t=end_t,
        fs=FS_OUT, tmin=EPOCH[0],
        dist_mm=dist_ch[sel], channels=np.array(channels),
    )
    if verbose:
        cnt = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        n_multi = int((cond == 1).sum())
        print(f"  {sub}: X={X.shape}  y={cnt}  single-talker excluded={n - n_multi}  "
              f"dropped={n_multi - len(y)}  boundary={int(spans.sum())}  "
              f"acc={correct[used].mean():.2f}")
    return X.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/, or data_null/ under --null)")
    ap.add_argument("--null", action="store_true",
                    help="cut epochs at random times instead of at trial onsets")
    args = ap.parse_args()
    # an explicit --out still gets the _null suffix appended, so that running the control
    # cannot overwrite a real tensor directory the caller has just built
    if args.out:
        out = args.out + ("_null" if args.null else "")
    else:
        out = paths.null_dir() if args.null else paths.data_dir()
    os.makedirs(out, exist_ok=True)

    channels = common_long_channels(args.root)
    print(f"epoch {EPOCH[0]:+g}..{EPOCH[1]:+g} s at {FS_OUT:g} Hz, locked to "
          f"{'RANDOM ONSETS (null)' if args.null else TRIGGER}")
    print(f"{len(channels)} long channels common to all subjects -> "
          f"{2 * len(channels)} features (HbO + HbR)")
    for exp, path in experiment_dirs(args.root):
        try:
            build_subject(exp, path, out, channels, null=args.null)
        except Exception as e:
            print(f"  ! {exp} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
