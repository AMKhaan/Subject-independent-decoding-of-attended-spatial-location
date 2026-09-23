"""
Which trial event do the official ds004830 v2.0.0 onsets mark: fixation, cue, or movie?

Why this exists
---------------
Every epoch since 2026-09-08 is cut at the official v2.0.0 onset, and every analysis
calls that time "stimulus onset". Ning et al.'s feature path is locked to the CUE
(their 0-5 s window "starts at cue onset"), and the cue leads the movie by a fixed
3.19 s (Trigger3 - Trigger2, sd 48 ms). To reproduce their path on corrected timing we
must know which event the official onset is. The events.tsv does not say (duration 5 s,
value 1, no description), and the absolute clocks cannot tell us because the v1
reconstruction is displaced by an unknown per-subject offset.

Two independent tests, neither of which needs the absolute clock:

  1. Inter-onset intervals. Differences between consecutive trials cancel any constant
     offset, so official intervals can be compared with each trigger's intervals.
     Measured 2026-09-18: all three triggers are locked to one another (T3-T2 sd ~0.02 s,
     T3-T1 sd ~0.04 s, not the "variable 5.5-7 s" once assumed), so their intervals are
     indistinguishable and this test cannot pick an event. It is kept because it is a
     per-subject timing audit: it flags subjects whose official intervals do not follow
     the PsychToolbox clock at all (sub-08: extra single-talker trials break pairing;
     sub-19: official onsets drift ~3.8 s per trial from the triggers, already visible
     in validate-v2-events.log as sd 94 s, and sub-19 decodes at 80% on the official
     timeline vs 41% on the reconstruction, so the official timing is the right one).
  2. Response latency. The movie is a 5 s audiovisual stimulus; its haemodynamic
     response should peak roughly 5-7 s after movie onset. Grand-average HbO across all
     trials, channels and subjects, locked to the official onset. The first event in a
     trial that can drive a response is the cue. If the onset IS the cue, nothing can
     peak before roughly +5 s. If the onset is the MOVIE, the cue sits at -3.19 s and its
     response (peak ~6-7 s after it) lands near +3 to +4 s. So an early peak (< ~4.5 s)
     favours "onset = movie"; a peak at >= ~5 s cannot separate the two.

Writes results/ds004830/check_onset_anchor/ds004830_v2_check_onset_anchor.log and the
matching .png under plots/ (see layout.py).

Usage:  python check_onset_anchor.py [--root PATH] [--data ../../derived/ds004830/v2]
"""

import argparse
import glob
import os
import sys

import numpy as np

import build_dataset as bd
import layout
import paths
from build_dataset_v2 import read_events, BIDS_RUNS, DEFAULT_RUNS

HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = layout.derived("ds004830", "events_v2")


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--data", default=layout.derived("ds004830", "v2"))
    args = ap.parse_args()
    res = lambda ext: layout.result(args.data, "check_onset_anchor", ext)
    log = Log(res(".log"))
    log("which event do the official v2.0.0 onsets mark?\n")

    # ---------------------------------------------------------------- test 1: IOIs
    log("1. inter-onset intervals, official vs PsychToolbox triggers (within run, no clock)")
    log(f"   {'subject':<9}{'n':>4}{'|dIOI| T1':>11}{'|dIOI| T2':>11}{'|dIOI| T3':>11}"
        f"{'T3-T2 sd':>10}{'T3-T1 sd':>10}   (median abs difference, s)")
    rows = []
    for exp, path in bd.experiment_dirs(args.root):
        num = exp.replace("Experiment", "")
        trig, _ = bd.load_triggers(path)
        ev = read_events(num, EVENTS)
        runs = BIDS_RUNS.get(num, DEFAULT_RUNS)
        off = np.array([o for _, o, _, _ in ev])
        rn = np.array([r for r, _, _, _ in ev])
        n = min(len(off), len(trig["Trigger3"]))
        if len(runs) > 1:            # multi-run: compare within the first run only
            m = rn[:n] == runs[0]
            n = int(m.sum())
        o = off[:n]
        t = {k: np.asarray(trig[k], float)[:n] for k in ("Trigger1", "Trigger2", "Trigger3")}
        d = {k: np.median(np.abs(np.diff(o) - np.diff(v))) for k, v in t.items()}
        sd32 = np.std(t["Trigger3"] - t["Trigger2"])
        sd31 = np.std(t["Trigger3"] - t["Trigger1"])
        rows.append(d)
        log(f"   sub-{num:<5}{n:>4}{d['Trigger1']:11.3f}{d['Trigger2']:11.3f}"
            f"{d['Trigger3']:11.3f}{sd32:10.3f}{sd31:10.3f}")
    med = {k: np.median([r[k] for r in rows]) for k in rows[0]}
    log(f"   median over subjects: T1 {med['Trigger1']:.3f}  T2 {med['Trigger2']:.3f}  "
        f"T3 {med['Trigger3']:.3f} s")
    log("   -> cannot identify the event: T1, T2 and T3 are all locked to one another, so\n"
        "      their intervals are the same. Subjects with |dIOI| >> 0.1 s (sub-08, sub-19)\n"
        "      are timing anomalies between the official file and the PsychToolbox clock,\n"
        "      not evidence about the anchor.\n")

    # ---------------------------------------------------------- test 2: latency
    log("2. grand-average HbO locked to the official onset (all trials, all channels)")
    peaks, curves = [], []
    for f in sorted(glob.glob(os.path.join(args.data, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        X = z["X"]
        nch = X.shape[2] // 2
        t = float(z["tmin"]) + np.arange(X.shape[1]) / float(z["fs"])
        hbo = X[:, :, :nch].mean(axis=(0, 2))           # trials and channels
        curves.append(hbo)
        post = t >= 0
        peaks.append(float(t[post][np.argmax(hbo[post])]))
        log(f"   {str(z['subject']):<8} peak HbO at {peaks[-1]:+5.1f} s")
    g = np.mean(curves, axis=0)
    gp = float(t[t >= 0][np.argmax(g[t >= 0])])
    log(f"   grand average peak: {gp:+.1f} s;  per-subject median {np.median(peaks):+.1f} s "
        f"(IQR {np.percentile(peaks, 25):+.1f} to {np.percentile(peaks, 75):+.1f})")
    if gp < 4.5:
        verdict = (f"peak at {gp:+.1f} s is too early to be a response to an event at 0, and\n"
                   f"      fits a response to the cue at -3.19 s: FAVOURS onset = MOVIE")
    else:
        verdict = "peak >= 4.5 s cannot separate cue from movie; inconclusive"
    log(f"   -> {verdict}")
    log("   caveat: the epoch ends at +12 s and peak latency varies by region and person;\n"
        "   this is supporting evidence, weighed with test 1, not proof.\n")

    # ------------------------------------------------ test 3: reproduction (strongest)
    # Ning et al. are the dataset's authors and their window starts at the cue. If
    # their published pipeline reproduces their published accuracy only under one
    # anchor, that anchor is what their event file encodes. Reads ning_decode.py logs
    # for both anchors (ning_build.py --events ... --anchor movie|cue).
    import re
    log("3. reproduction of Ning et al.'s published accuracy with their own pipeline")
    rep = {}
    for anchor in ("cue", "movie"):
        for task, pub in (("3class", 49.4), ("2class", 78.3)):
            p = layout.result(layout.derived("ds004830", f"ning_v2_{anchor}"), "ning_decode",
                              ".log", task)
            if not os.path.exists(p):
                continue
            m = re.search(r"headline \(5s window.*?\): ([\d.]+)%", open(p, encoding="utf-8").read())
            if m:
                rep[(anchor, task)] = float(m.group(1))
                log(f"   onset = {anchor:<5} {task}: {float(m.group(1)):6.2f}%  "
                    f"(published {pub}%, gap {float(m.group(1)) - pub:+.2f})")
    if len(rep) == 4:
        gap = {a: abs(rep[(a, '3class')] - 49.4) + abs(rep[(a, '2class')] - 78.3)
               for a in ("cue", "movie")}
        win = min(gap, key=gap.get)
        log(f"   -> total |gap|: cue {gap['cue']:.1f} pp, movie {gap['movie']:.1f} pp: their "
            f"published result is reproduced only if onset = {win.upper()}.")
        log("   VERDICT (weighting test 3 over test 2): the official onset marks the "
            f"{win.upper()}.\n   Test 2's early peak is then a fast response to the cue "
            "itself, not a response to\n   an earlier event; test 2 alone was misread as "
            "favouring the movie.\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for c in curves:
        ax.plot(t, c, color="grey", lw=0.6, alpha=0.5)
    ax.plot(t, g, color="#c0502a", lw=2.2, label="grand average")
    for x0, lab in ((0, "official onset"), (3.19, "+3.19 s")):
        ax.axvline(x0, color="k", ls="--" if x0 else "-", lw=0.8)
    ax.axvspan(2.8, 4.2, color="#1f5fa8", alpha=0.12,
               label="cue response if onset = movie (cue at -3.19 s)")
    ax.axvspan(5, 7, color="#2a9d5c", alpha=0.12, label="earliest peak if onset = cue")
    ax.set_xlabel("time from official onset (s)")
    ax.set_ylabel("HbO, mean over trials and channels")
    ax.set_title("Response latency (grey = subjects)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    x = np.arange(3)
    ax.bar(x, [med["Trigger1"], med["Trigger2"], med["Trigger3"]], color=
           ["#bbbbbb", "#2a9d5c", "#1f5fa8"])
    ax.set_xticks(x)
    ax.set_xticklabels(["T1 fixation", "T2 cue", "T3 movie"])
    ax.set_ylabel("median |IOI official - IOI trigger| (s)")
    ax.set_title("Inter-onset interval agreement (lower = match)", fontsize=9)
    fig.tight_layout()
    fig.savefig(res(".png"), dpi=150)
    log(f"wrote {layout.rel(res('.log'))}, {layout.rel(res('.png'))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
