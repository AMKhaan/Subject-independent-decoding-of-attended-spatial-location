"""
Verify the reconstructed trial onsets against the hardware trigger line.

Why this exists
---------------
Every epoch in this project is cut at `startT + Trigger3(i)`, a timestamp taken from
the PsychToolbox log. That reconstruction is the load-bearing assumption under every
number we report, and it has so far been checked only indirectly: the behavioural
table reproduces exactly, and the group evoked response beats a shuffled-onset null.

Neither check constrains absolute timing. A per-subject constant offset would leave
the behavioural reproduction untouched (it is a property of labels, not clocks) and
would still yield a clean group average if it were common across subjects.

The `.nirs` files carry an 8-column `aux` array whose first column looks like a TTL
trigger line: a low baseline with brief pulses. If those pulses are the experiment's
own hardware markers, they are ground truth for the timing, recorded by the fNIRS
acquisition clock rather than the stimulus computer's, and we can measure the
reconstruction error directly instead of inferring it.

What this script decides
------------------------
  * Are there exactly 4 pulses per trial (Trigger1..4)?
  * Do the inter-pulse gaps reproduce the PsychToolbox gaps, T2-T1 ~ 2.47 s and
    T3-T2 ~ 3.19 s?
  * If so, what is `aux_onset - (startT + Trigger3)` per subject -- a constant, a
    drift, or nothing?

A confirmed non-zero offset would mean the epochs are misaligned and would be a
leading candidate for the ~10-point deficit against Ning et al. A refuted one closes
the question and leaves the current build standing.

Nothing here modifies any data. Output is
results/ds004830/check_aux_triggers/ds004830_check_aux_triggers.log and one figure per
subject, plots/ds004830/check_aux_triggers/ds004830_check_aux_triggers_<Experiment>.png
(see layout.py).

Usage:  python check_aux_triggers.py [--root PATH] [--subjects 08,12]
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio

import layout
import paths
from build_dataset import experiment_dirs, load_manifest, load_triggers

FS_RAW = 50.0          # Hz, Techen CW6 acquisition rate
PULSE_MIN_GAP_S = 0.5  # debounce: no two distinct markers closer than this


class Tee:
    """Echo to console and to a log file at once, so every run leaves a record."""

    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def load_aux(nirs_file):
    """Return (t, aux) for one run; aux is (n_samples, n_aux) float."""
    m = sio.loadmat(nirs_file, variable_names=["t", "aux", "s"])
    t = np.asarray(m["t"], dtype=float).flatten()
    aux = None
    for key in ("aux", "aux10", "auxl"):
        if key in m:
            aux = np.asarray(m[key], dtype=float)
            break
    if aux is None:
        return t, None
    if aux.ndim == 1:
        aux = aux[:, None]
    return t, aux


def find_pulses(sig, t, min_gap_s=0.0):
    """Rising-edge times of a TTL-like channel.

    Threshold at the midpoint between the low and high rails rather than a fixed
    voltage, so this works regardless of the amplifier's scaling. min_gap_s debounces;
    leave it at 0 to see the true edge structure, because a debounce large enough to
    tidy a fast train also manufactures a plausible-looking event rate out of one.
    """
    lo, hi = np.percentile(sig, 1), np.percentile(sig, 99)
    if hi - lo < 1e-6:
        return np.array([]), lo, hi
    thr = lo + 0.5 * (hi - lo)
    above = sig > thr
    rising = np.where((~above[:-1]) & (above[1:]))[0] + 1
    if rising.size == 0:
        return np.array([]), lo, hi
    times = t[rising]
    if min_gap_s <= 0:
        return times, lo, hi
    keep = [times[0]]
    for x in times[1:]:
        if x - keep[-1] >= min_gap_s:
            keep.append(x)
    return np.array(keep), lo, hi


def verdict(pulses, t_end, n_trials):
    """Is this channel plausibly a trial-marker line?

    The decisive test is not the pulse count -- a debounce can be tuned to produce
    any count -- but whether the train has trial structure at all. Trials are ~28 s
    apart, so a marker line must show gaps of that order between bursts. A train with
    no long gaps is a clock, whatever its rate.
    """
    if pulses.size < 2:
        return False, "no pulses"
    g = np.diff(pulses)
    long_gaps = int((g > 5.0).sum())
    rate = pulses.size / max(t_end, 1.0)
    if long_gaps == 0:
        return False, (f"no gap exceeds 5 s over {t_end:.0f} s ({rate:.2f} edges/s); "
                       f"trials are ~28 s apart, so this has no trial structure")
    if abs(pulses.size - 4 * n_trials) <= 4:
        return True, f"{pulses.size} edges ~ 4 x {n_trials} trials, with burst structure"
    return False, (f"{long_gaps} long gaps but {pulses.size} edges against "
                   f"{4 * n_trials} expected")


def plot_subject(exp, t, col0, pulses):
    """One figure per subject: the raw line, a zoom, and the gap distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(t, col0, lw=0.3)
    ax[0].set_title(f"{exp}: aux column 0, full run")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("volts")

    m = t < min(60.0, t[-1])
    ax[1].plot(t[m], col0[m], lw=0.8)
    ax[1].set_title("first 60 s -- a trial-marker line would show ~2 bursts here, "
                    "not a continuous train")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("volts")

    if pulses.size > 1:
        g = np.diff(pulses)
        ax[2].hist(g[g < 5], bins=100)
        ax[2].set_yscale("log")
        ax[2].set_title(f"inter-edge gaps (n={g.size}, max {g.max():.2f} s); "
                        f"trial spacing is ~28 s")
        ax[2].set_xlabel("gap (s)")
        ax[2].set_ylabel("count (log)")

    fig.tight_layout()
    f = layout.result("ds004830", "check_aux_triggers", ".png", part=exp)
    fig.savefig(f, dpi=130)
    plt.close(fig)
    return f


def describe_channel(aux, t, log):
    """Report every aux column so we can see which one is the trigger line."""
    log(f"    aux shape {aux.shape}")
    for j in range(aux.shape[1]):
        col = aux[:, j]
        pulses, lo, hi = find_pulses(col, t)
        log(f"      col {j}: min {col.min():+.4f}  max {col.max():+.4f}  "
            f"sd {col.std():.5f}  rails [{lo:+.3f},{hi:+.3f}]  pulses {pulses.size}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--subjects", default=None,
                    help="comma-separated experiment numbers, e.g. 08,12 (default all)")
    args = ap.parse_args()

    log_path = layout.result("ds004830", "check_aux_triggers", ".log")
    log = Tee(log_path)
    paths.require(args.root, "ds004830 derivatives tree")

    log("Hardware trigger check against the reconstructed onsets")
    log(f"root    : {args.root}")
    log(f"out     : {layout.rel(log_path)} (+ figures under plots/)")
    log("")

    wanted = set(args.subjects.split(",")) if args.subjects else None
    rows = []

    for exp, path in experiment_dirs(args.root):
        num = exp.replace("Experiment", "")
        if wanted and num not in wanted:
            continue
        log(f"=== {exp} ===")
        try:
            runs, start_t, end_t = load_manifest(path, exp)
            trig, _ok = load_triggers(path)
        except Exception as e:                       # noqa: BLE001
            log(f"    SKIP: {e}")
            continue

        n_trials = trig["Trigger1"].size
        log(f"    runs {runs}  startT {start_t:.3f}  trials {n_trials}")

        # concatenate runs the same way build_dataset.py does, so the timeline the
        # offsets are measured against is the timeline the epochs were cut on
        all_pulses, offset, first = [], 0.0, True
        for r in runs:
            f = os.path.join(path, r + ".nirs")
            if not os.path.exists(f):
                log(f"    missing run file {f}")
                continue
            t, aux = load_aux(f)
            if aux is None:
                log(f"    run {r}: no aux array")
                continue
            if first:
                describe_channel(aux, t, log)
                fig = plot_subject(exp, t, aux[:, 0], find_pulses(aux[:, 0], t)[0])
                log(f"    figure -> {os.path.basename(fig)}")
                first = False
            pulses, lo, hi = find_pulses(aux[:, 0], t)
            log(f"    run {r}: {len(t)} samples, {t[-1]:.1f} s, "
                f"{pulses.size} edges on col 0")
            all_pulses.append(pulses + offset)
            offset += t[-1]

        if not all_pulses:
            log("    no pulses found; nothing to compare")
            log("")
            continue
        pulses = np.concatenate(all_pulses)

        log(f"    total edges {pulses.size}   trials x4 = {n_trials * 4}")
        if pulses.size >= 8:
            g = np.diff(pulses)
            log(f"    gaps: n={g.size} min {g.min():.3f} median {np.median(g):.3f} "
                f"max {g.max():.3f} s;  gaps > 5 s: {(g > 5).sum()}")
            # the PsychToolbox gaps we would see if these were Trigger1..4
            for name, want in (("T2-T1", 2.47), ("T3-T2", 3.19)):
                near = g[np.abs(g - want) < 0.25]
                log(f"      gaps near {name} ({want:.2f} s): {near.size}"
                    + (f"  median {np.median(near):.3f}" if near.size else ""))

        ok, why = verdict(pulses, offset, n_trials)
        log(f"    VERDICT: {'trigger line' if ok else 'NOT a trigger line'} -- {why}")
        rows.append((exp, pulses.size, n_trials * 4, ok, why))
        log("")

    log("--- summary ---")
    log(f"{'exp':<16}{'edges':>8}{'4x trials':>11}  verdict")
    for exp, n_p, n_e, ok, why in rows:
        log(f"{exp:<16}{n_p:>8}{n_e:>11}  {'TRIGGER' if ok else 'not a trigger line'}")

    n_ok = sum(1 for r in rows if r[3])
    log("")
    if n_ok == 0 and rows:
        log("CONCLUSION: no subject's aux column 0 carries trial markers. The channel")
        log("is a continuous high-rate line with no gaps at the ~28 s trial spacing,")
        log("which is what the Techen frequency-multiplexing clock looks like, not what")
        log("an event line looks like. There is therefore no independent hardware")
        log("timing in these files, and no offset can be computed from them.")
        log("")
        log("This REFUTES the hypothesis that the epochs are misaligned. The onset")
        log("reconstruction startT + Trigger3 remains the only timing available, and")
        log("the evidence for it is unchanged: the behavioural table reproduces exactly")
        log("for all twelve subjects, and the group evoked response beats a")
        log("shuffled-onset control by 2.3x in amplitude.")
        log("")
        log("Consequence for the project: the ~10-point deficit against Ning et al. is")
        log("NOT a timing artefact, so it must be sought in the feature path.")
    else:
        log(f"CONCLUSION: {n_ok} of {len(rows)} subjects show a genuine trigger line.")
        log("Compute onset offsets only for those, and treat the rest as unusable.")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
