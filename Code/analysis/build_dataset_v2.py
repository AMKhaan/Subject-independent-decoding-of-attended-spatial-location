"""
Build the epoch tensors from the official ds004830 v2.0.0 event files.

Why a second builder
--------------------
`build_dataset.py` reconstructs onsets as `startT + Trigger3(i)` because v1.0.0 of the
dataset ships empty `events.tsv` files. In February 2026 the authors published v2.0.0
with those files populated, and comparing the two timelines shows the reconstruction
is displaced by a near-constant per-subject offset of about -18 s to +11 s. Re-epoching
on the official onsets raises within-subject three-class accuracy from 39.9% to 58.9%,
so the reconstruction appears to have been wrong and every downstream number in this
project was computed on misaligned epochs.

This script writes tensors in exactly the format `build_dataset.py` produces, so the
existing analysis scripts -- decode.py, deep_models.py, time_resolved.py -- run against
it unchanged, with their permutation tests and per-subject reporting intact. That
matters: the claim "accuracy nearly doubled" has to be checked with the same machinery
that produced the original numbers, not with new code written to confirm it.

Everything except the onsets is identical to build_dataset.py: the same filtering, the
same Beer-Lambert conversion, the same short-separation regression, the same 28-channel
shared montage, the same epoch and baseline windows.

Labels and correctness come from the official file too, which independently confirmed
the mapping 1 = Right, 2 = Left, 3 = Center and reproduced the inferred
face-and-transcript correctness criterion at 100% on all 1080 trials.

Usage:  python build_dataset_v2.py [--root PATH] [--events DIR] [--out PATH] [--null]

Defaults: events from derived/ds004830/events_v2/, output to derived/ds004830/v2/
(v2_null/ with --null; see layout.py). The console output is the build log, saved as
results/ds004830/build_dataset_v2/ds004830_<variant>_build_dataset_v2.log.
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio

import layout
import paths
from build_dataset import (EPOCH, BASELINE, FS_OUT, experiment_dirs, load_manifest,
                           probe_geometry, process_run, common_long_channels)

# the official trial_type strings, mapped onto the numeric codes this project uses.
# Confirmed one-to-one against indexMoviesTest column 2 for all twelve subjects.
LABELS = {"Right": 1, "Left": 2, "Center": 3}
BIDS_RUNS = {"08": ["run-01", "run-02", "run-03"]}
DEFAULT_RUNS = ["run-01"]


def read_events(num, evdir):
    """Official onsets, labels and correctness for one subject, per run."""
    rows = []
    for run in BIDS_RUNS.get(num, DEFAULT_RUNS):
        f = os.path.join(evdir, f"sub-{num}_task-overt_{run}_events.tsv")
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        with open(f, encoding="utf-8") as fh:
            head = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                if line.strip():
                    r = dict(zip(head, line.rstrip("\n").split("\t")))
                    rows.append((run, float(r["onset"]), r["trial_type"],
                                 int(r.get("correct_both_response", 0))))
    return rows


def common_long_channels_by_position(root, ndigits=1):
    """Long channels present in every subject, matched by probe position, not index.

    common_long_channels() intersects (source, detector) index pairs. sub-08's larger
    probe numbers its detectors differently from D16 onward, so the two STG channels
    (S7-D16 and S8-D17 in the other eleven subjects, S7-D18 and S8-D19 in sub-08) sit
    at identical positions but carry different indices, and the index intersection
    drops them: 28 channels instead of 30, and no superior temporal coverage at all.

    Returns {experiment: [(src, det), ...]} in one canonical column order: the 28
    index-matched channels first, in their usual order, so columns 0-27 mean exactly
    what they mean in derived/ds004830/v2, then any position-matched additions.
    """
    import scipy.io as sio
    per = {}
    for exp, path in experiment_dirs(root):
        runs, _, _ = load_manifest(path, exp)
        f = os.path.join(path, runs[0] + ".nirs")
        sd = sio.loadmat(f, variable_names=["SD"])["SD"][0, 0]
        src = np.array(sd["SrcPos"], dtype=float)
        det = np.array(sd["DetPos"], dtype=float)
        _, dist_ch, ident = probe_geometry(f)
        per[exp] = {tuple(np.round((src[s - 1] + det[d - 1]) / 2, ndigits)): (s, d)
                    for (s, d), dd in zip(ident, dist_ch) if dd >= 15.0}
    shared = set.intersection(*(set(m) for m in per.values()))
    base = common_long_channels(root)
    ref = min(per, key=lambda e: len(per[e]))       # the standard probe numbering
    key_to_pos = {v: k for k, v in per[ref].items()}
    order = [key_to_pos[k] for k in base]
    order += sorted(p for p in shared if p not in order)
    return {exp: [m[p] for p in order] for exp, m in per.items()}, [per[ref][p] for p in order]


def build_subject(exp, path, evdir, out_dir, channels, null=False, verbose=True,
                  save_channels=None):
    num = exp.replace("Experiment", "")
    runs, start_t, end_t = load_manifest(path, exp)
    ev = read_events(num, evdir)

    first = os.path.join(path, runs[0] + ".nirs")
    pairs, dist_ch, ident = probe_geometry(first)
    ext = np.array(sio.loadmat(first, variable_names=["SD"])["SD"][0, 0]["extCoef"],
                   dtype=float)
    pos = {k: i for i, k in enumerate(ident)}
    sel = np.array([pos[k] for k in channels])

    # process each run and remember where it starts in the concatenated timeline,
    # because BIDS restarts every run's clock at zero
    blocks, run_start, offset = [], {}, 0.0
    for r in runs:
        f = os.path.join(path, r + ".nirs")
        if not os.path.exists(f):
            print(f"  ! {exp}: missing run {r}, skipped")
            continue
        blk, dur = process_run(f, pairs, dist_ch, ext, sel)
        run_start[r] = offset
        blocks.append(blk)
        offset += dur
    data = np.concatenate(blocks, axis=0)

    bids = BIDS_RUNS.get(num, DEFAULT_RUNS)
    run_off = {b: run_start.get(r, 0.0) for b, r in zip(bids, runs)}
    onsets = np.array([o + run_off.get(rn, 0.0) for rn, o, _t, _c in ev])
    labels = np.array([LABELS[t] for _rn, _o, t, _c in ev], dtype=int)
    correct = np.array([bool(c) for _rn, _o, _t, c in ev])

    if null:
        rng = np.random.default_rng(abs(hash(exp)) % (2 ** 32))
        lo, hi = -EPOCH[0], len(data) / FS_OUT - EPOCH[1]
        onsets = np.sort(rng.uniform(lo, hi, size=onsets.size))

    pre = int(round(EPOCH[0] * FS_OUT))
    post = int(round(EPOCH[1] * FS_OUT))
    n_times = post - pre
    b0 = int(round((BASELINE[0] - EPOCH[0]) * FS_OUT))
    b1 = int(round((BASELINE[1] - EPOCH[0]) * FS_OUT))

    X, y, used = [], [], []
    for i, (on, lab) in enumerate(zip(onsets, labels)):
        s0 = int(round(on * FS_OUT)) + pre
        s1 = s0 + n_times
        if s0 < 0 or s1 > len(data):
            continue
        seg = data[s0:s1]
        X.append(seg - seg[b0:b1].mean(axis=0))
        y.append(lab)
        used.append(i)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    used = np.asarray(used)

    sub = "sub-" + num
    np.savez_compressed(
        os.path.join(out_dir, sub + ".npz"),
        X=X, y=y, subject=sub, trial_index=used,
        crosses_run_boundary=np.zeros(len(used), bool),
        correct=correct[used],
        onsets=onsets[used], start_t=start_t, end_t=end_t,
        fs=FS_OUT, tmin=EPOCH[0],
        dist_mm=dist_ch[sel],
        channels=np.array(save_channels if save_channels is not None else channels),
    )
    if verbose:
        cnt = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        print(f"  {sub}: X={X.shape}  y={cnt}  dropped={len(ev) - len(y)}  "
              f"acc={correct[used].mean():.2f}")
    return X.shape


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--events", default=layout.derived("ds004830", "events_v2"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--null", action="store_true",
                    help="cut epochs at random times instead (negative control)")
    ap.add_argument("--by-position", action="store_true",
                    help="match channels across subjects by probe position (30 channels, "
                         "recovers STG) instead of by index (28)")
    args = ap.parse_args()

    base = args.out or layout.derived("ds004830", "v2")
    out = base + ("_null" if args.null else "")
    os.makedirs(out, exist_ok=True)
    paths.require(args.root, "ds004830 derivatives tree")
    paths.require(args.events, "downloaded v2.0.0 events directory")

    channels = common_long_channels(args.root)
    per_exp, canonical = None, None
    if args.by_position:
        per_exp, canonical = common_long_channels_by_position(args.root)
        channels = canonical
    print(f"epoch {EPOCH[0]:+g}..{EPOCH[1]:+g} s at {FS_OUT:g} Hz, locked to "
          f"{'RANDOM ONSETS (null)' if args.null else 'OFFICIAL v2.0.0 events'}")
    print(f"{len(channels)} long channels common to all subjects -> "
          f"{2 * len(channels)} features (HbO + HbR)")
    print(f"out: {out}")
    for exp, path in experiment_dirs(args.root):
        try:
            build_subject(exp, path, args.events, out,
                          per_exp[exp] if per_exp else channels, null=args.null,
                          save_channels=canonical)
        except Exception as e:                              # noqa: BLE001
            print(f"  ! {exp} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
