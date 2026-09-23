"""
Check the reconstructed trial onsets against the official ds004830 v2.0.0 events.

Background
----------
ds004830 v1.0.0, the version this project was built on, ships `events.tsv` files
containing a single placeholder row of zeros, and a Homer stimulus matrix `s` that is
identically zero. Every onset in this project is therefore reconstructed from the
PsychToolbox logs as `startT + Trigger3(i)`, and every result rests on that being
right.

On 2026-02-27 the authors released **v2.0.0**, whose changelog reads:

    Updated nirs.json and events.tsv files and changed the filenaming with tasks to
    make it BIDS compliant.

The v2.0.0 events files are populated: onset, duration, an explicit `Left`/`Right`/
`Center` trial_type, and three correctness columns. That is independent ground truth
published by the people who ran the experiment, and it lets us replace an inference
with a measurement.

Three things get decided here
-----------------------------
1. **Timing.** Does `startT + Trigger3` match the official onsets, and to what
   tolerance? A per-subject constant offset would mean every epoch is displaced.
2. **The left/right assignment.** The dataset documentation has a typo ("1 = right,
   2 = right, 3 = center"), so which of labels 1 and 2 is left has never been
   established; §4.5 tried to settle it by lateralisation and failed. The official
   trial_type column states it outright.
3. **The correctness criterion.** v2.0.0 ships audio-only, video-only and conjunction
   correctness separately, so the face-and-transcript conjunction this project
   inferred can be checked directly.

Nothing is modified. Output is
results/ds004830/validate_against_v2_events/ds004830_validate_against_v2_events.log; the
official events are downloaded once into derived/ds004830/events_v2/ (see layout.py).

Usage:  python validate_against_v2_events.py [--root PATH] [--no-download]
"""

import argparse
import os
import sys
import urllib.request
import numpy as np

import layout
import paths
from build_dataset import (experiment_dirs, load_manifest, load_triggers,
                           load_labels)

S3 = "https://s3.amazonaws.com/openneuro.org/ds004830"
# experiment number -> the v2.0.0 run files, in acquisition order
RUNS = {"08": ["run-01", "run-02", "run-03"]}
DEFAULT_RUNS = ["run-01"]


class Tee:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def fetch_events(num, outdir, download=True):
    """Return the official events rows for one subject, concatenated across runs."""
    rows = []
    for run in RUNS.get(num, DEFAULT_RUNS):
        name = f"sub-{num}_task-overt_{run}_events.tsv"
        local = os.path.join(outdir, name)
        if download and not os.path.exists(local):
            url = f"{S3}/sub-{num}/nirs/{name}"
            try:
                urllib.request.urlretrieve(url, local)
            except Exception as e:                       # noqa: BLE001
                return None, f"download failed: {e}"
        if not os.path.exists(local):
            return None, "no local copy and --no-download given"
        with open(local, encoding="utf-8") as fh:
            head = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                if line.strip():
                    rows.append(dict(zip(head, line.rstrip("\n").split("\t"))))
    return rows, None


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    evdir = layout.derived("ds004830", "events_v2")
    os.makedirs(evdir, exist_ok=True)
    log_path = layout.result("ds004830", "validate_against_v2_events", ".log")
    log = Tee(log_path)
    paths.require(args.root, "ds004830 derivatives tree")

    log("Reconstructed onsets vs official ds004830 v2.0.0 events")
    log(f"root : {args.root}")
    log(f"out  : {layout.rel(log_path)}, events in {layout.rel(evdir)}")
    log("")

    summary, label_tab, corr_tab = [], [], []

    for exp, path in experiment_dirs(args.root):
        num = exp.replace("Experiment", "")
        log(f"=== sub-{num} ===")
        ev, err = fetch_events(num, evdir, download=not args.no_download)
        if ev is None:
            log(f"    SKIP: {err}")
            log("")
            continue
        try:
            runs, start_t, end_t = load_manifest(path, exp)
            trig, ok = load_triggers(path)
            loc, cond = load_labels(path)
        except Exception as e:                           # noqa: BLE001
            log(f"    SKIP: {e}")
            log("")
            continue

        off_on = np.array([float(r["onset"]) for r in ev])
        off_ty = np.array([r["trial_type"] for r in ev])
        off_both = np.array([int(r.get("correct_both_response", -1)) for r in ev])

        recon_all = start_t + trig["Trigger3"]
        log(f"    official trials {len(ev)}   psychtoolbox trials {recon_all.size}"
            f"   startT {start_t:.3f}")

        # the official file lists every trial; ours keeps competing-talker trials only
        competing = cond == 1
        log(f"    competing-talker trials in psychtoolbox record: {competing.sum()}")

        n = min(off_on.size, recon_all.size)
        if off_on.size == recon_all.size:
            recon, sel = recon_all, np.ones(recon_all.size, bool)
        elif off_on.size == int(competing.sum()):
            recon, sel = recon_all[competing], competing
            log("    matched official file to the competing-talker subset")
        else:
            recon, sel = recon_all[:n], np.zeros(recon_all.size, bool)
            sel[:n] = True
            log(f"    WARNING: counts differ; comparing the first {n}")

        m = min(off_on.size, recon.size)
        d = off_on[:m] - recon[:m]
        log(f"    onset difference (official - reconstructed), n={m}:")
        log(f"      mean {d.mean():+.4f} s   sd {d.std():.4f} s   "
            f"min {d.min():+.4f}   max {d.max():+.4f}")
        log(f"      median {np.median(d):+.4f} s   "
            f"|d| > 0.1 s: {(np.abs(d) > 0.1).sum()} of {m}")

        # label agreement: does our numeric code map one-to-one onto their words?
        ours = loc[sel][:m]
        theirs = off_ty[:m]
        log("    label cross-tab (rows = our code, cols = official):")
        codes = sorted(set(ours.tolist()))
        words = sorted(set(theirs.tolist()))
        log("      " + "code".ljust(6) + "".join(w.rjust(9) for w in words))
        mapping = {}
        for c in codes:
            counts = [(theirs[ours == c] == w).sum() for w in words]
            log("      " + str(c).ljust(6) + "".join(str(x).rjust(9) for x in counts))
            mapping[c] = words[int(np.argmax(counts))]
        pure = all(
            (theirs[ours == c] == mapping[c]).all() for c in codes)
        log(f"      deterministic mapping: {mapping}  "
            f"{'(one-to-one, no exceptions)' if pure else '(NOT clean)'}")

        # correctness agreement
        ours_ok = ok[sel][:m].astype(int)
        agree = (ours_ok == off_both[:m]).mean() if (off_both >= 0).all() else np.nan
        log(f"    correctness agreement with correct_both_response: {agree * 100:.2f}%")

        summary.append((num, m, d.mean(), d.std(), np.abs(d).max(),
                        int((np.abs(d) > 0.1).sum())))
        label_tab.append((num, mapping, pure))
        corr_tab.append((num, agree))
        log("")

    log("--- onset agreement summary ---")
    log(f"{'sub':<7}{'n':>5}{'mean d (s)':>13}{'sd (s)':>10}{'max|d|':>10}{'>0.1s':>8}")
    for num, m, mu, sd, mx, bad in summary:
        log(f"{num:<7}{m:>5}{mu:>+13.4f}{sd:>10.4f}{mx:>10.4f}{bad:>8}")

    if summary:
        allmu = np.array([s[2] for s in summary])
        allmax = np.array([s[4] for s in summary])
        nbad = sum(s[5] for s in summary)
        ntot = sum(s[1] for s in summary)
        log("")
        log(f"across subjects: mean offset {allmu.mean():+.4f} s "
            f"(range {allmu.min():+.4f} to {allmu.max():+.4f}); "
            f"worst single-trial |d| {allmax.max():.4f} s; "
            f"{nbad} of {ntot} trials off by more than 0.1 s")

    log("")
    log("--- label mapping ---")
    for num, mapping, pure in label_tab:
        log(f"sub-{num}: {mapping}  {'clean' if pure else 'INCONSISTENT'}")
    maps = {tuple(sorted(m.items())) for _, m, _ in label_tab}
    log(f"distinct mappings across subjects: {len(maps)}"
        + ("  -> consistent, the left/right assignment is now established"
           if len(maps) == 1 else "  -> subjects disagree; do not assume a global map"))

    log("")
    log("--- correctness ---")
    for num, a in corr_tab:
        log(f"sub-{num}: {a * 100:.2f}%")

    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
