"""
Decide which trial timeline is correct: our reconstruction, or the official v2.0.0 events.

The question
------------
Every epoch in this project is cut at `startT + Trigger3(i)`, reconstructed from the
PsychToolbox logs because ds004830 v1.0.0 ships empty `events.tsv` files. In February
2026 the authors released v2.0.0 with those files populated. Comparing the two
(validate_against_v2_events.py) shows a near-constant per-subject offset ranging from
about -18 s to +11 s, with roughly 0.7 s of within-subject scatter. Not one of 1080
trials agrees to 0.1 s.

Labels and behavioural correctness agree perfectly, which tells us the trial *identity*
mapping is right and says nothing about the clock. So one of the two timelines is
wrong, and the project cannot proceed until we know which.

The test
--------
Epoch the same recordings twice, changing only the onsets, and ask which timeline
produces a better-formed haemodynamic response and higher within-subject decoding
accuracy. Both are properties the correct alignment should win on:

  * peak HbO amplitude and a peak latency in the canonical 4-8 s window
  * HbO/HbR anti-correlation, which a systemic oscillation cannot fake
  * within-subject decoding accuracy against 33.3% chance

If the official onsets win, our epochs have been misaligned and that is a leading
explanation for the ~10-point deficit against Ning et al. If ours win, the v2.0.0
events have a problem of their own and should be reported rather than adopted.

Power-cut safety
----------------
Results are checkpointed to a JSON file after every subject. Re-running resumes from
the checkpoint and skips completed subjects, so an interruption costs one subject's
work at most. Delete the checkpoint to force a full recompute.

Usage:  python compare_onset_timelines.py [--root PATH] [--fresh]

Writes results/ds004830/compare_onset_timelines/ds004830_compare_onset_timelines.log and
_checkpoint.json, and plots/ds004830/compare_onset_timelines/
ds004830_compare_onset_timelines_sub-NN.png and _summary.png (see layout.py). Reads the
official events from derived/ds004830/events_v2/.
"""

import argparse
import json
import os
import sys
import numpy as np
import scipy.io as sio

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import layout
import paths
from build_dataset import (EPOCH, BASELINE, FS_OUT, experiment_dirs, load_manifest,
                           load_triggers, load_labels, probe_geometry, process_run,
                           common_long_channels)

BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]   # s, as in decode.py
CANONICAL = (4.0, 8.0)                                        # s, expected HRF peak


RES = lambda ext, part="": layout.result("ds004830", "compare_onset_timelines", ext,
                                         part=part)


class Tee:
    def __init__(self, path, append=False):
        self.fh = open(path, "a" if append else "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def official_onsets(num, evdir):
    """Official onsets for one subject, in the concatenated-run timeline.

    BIDS restarts each run's clock at zero, so a multi-run subject needs the
    cumulative run durations added. Getting this wrong is what produced the absurd
    -2434 s offset for sub-08 in the first comparison.
    """
    runs = ["run-01", "run-02", "run-03"] if num == "08" else ["run-01"]
    out, types = [], []
    for run in runs:
        f = os.path.join(evdir, f"sub-{num}_task-overt_{run}_events.tsv")
        if not os.path.exists(f):
            return None, None, f"missing {os.path.basename(f)}"
        with open(f, encoding="utf-8") as fh:
            head = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                if not line.strip():
                    continue
                row = dict(zip(head, line.rstrip("\n").split("\t")))
                out.append((run, float(row["onset"])))
                types.append(row["trial_type"])
    return out, types, None


def epoch_at(data, onsets, labels, keep_mask):
    """Cut epochs at the given onsets. Returns X, y and the indices used."""
    pre = int(round(EPOCH[0] * FS_OUT))
    post = int(round(EPOCH[1] * FS_OUT))
    n_times = post - pre
    b0 = int(round((BASELINE[0] - EPOCH[0]) * FS_OUT))
    b1 = int(round((BASELINE[1] - EPOCH[0]) * FS_OUT))

    X, y, used = [], [], []
    for i, (on, lab) in enumerate(zip(onsets, labels)):
        if not keep_mask[i]:
            continue
        s0 = int(round(on * FS_OUT)) + pre
        s1 = s0 + n_times
        if s0 < 0 or s1 > len(data):
            continue
        seg = data[s0:s1]
        X.append(seg - seg[b0:b1].mean(axis=0))
        y.append(lab)
        used.append(i)
    if not X:
        return np.zeros((0, n_times, data.shape[1]), np.float32), np.zeros(0), []
    return np.asarray(X, np.float32), np.asarray(y, np.int8), used


def evoked_stats(X):
    """Peak HbO amplitude, its latency, and the HbO/HbR correlation over 0-12 s."""
    n_ch = X.shape[2] // 2
    t = EPOCH[0] + np.arange(X.shape[1]) / FS_OUT
    hbo = X[:, :, :n_ch].mean(axis=(0, 2))
    hbr = X[:, :, n_ch:].mean(axis=(0, 2))
    post = t >= 0
    k = int(np.argmax(np.abs(hbo[post])))
    amp = float(hbo[post][k])
    lat = float(t[post][k])
    r = float(np.corrcoef(hbo[post], hbr[post])[0, 1])
    return amp, lat, r, t, hbo, hbr


def featurise(X):
    t = EPOCH[0] + np.arange(X.shape[1]) / FS_OUT
    return np.concatenate(
        [X[:, (t >= a) & (t < b), :].mean(axis=1) for a, b in BINS], axis=1)


def decode_within(X, y, seed=0):
    """Within-subject 5-fold accuracy for two cheap, well-understood models."""
    if len(np.unique(y)) < 2 or len(y) < 15:
        return {}
    F = featurise(X)
    models = {
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "Logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, C=0.1)),
    }
    out = {}
    for name, clf in models.items():
        acc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y):
            clf.fit(F[tr], y[tr])
            acc.append(float((clf.predict(F[te]) == y[te]).mean()))
        out[name] = float(np.mean(acc))
    return out


def plot_subject(num, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for k, (name, col) in enumerate((("recon", "tab:blue"), ("official", "tab:red"))):
        w = res.get(name)
        if not w:
            continue
        ax[k].plot(w["t"], w["hbo"], color=col, label="HbO")
        ax[k].plot(w["t"], w["hbr"], color=col, ls="--", alpha=0.6, label="HbR")
        ax[k].axvline(0, color="k", lw=0.8)
        ax[k].axvspan(*CANONICAL, color="grey", alpha=0.15)
        ax[k].set_title(f"sub-{num} {name}: peak {w['amp']:+.3f} uM @ {w['lat']:.1f} s, "
                        f"r={w['r']:+.2f}")
        ax[k].set_xlabel("time from onset (s)")
        ax[k].legend(fontsize=8)
    ax[0].set_ylabel("concentration (uM)")
    fig.tight_layout()
    f = RES(".png", f"sub-{num}")
    fig.savefig(f, dpi=130)
    plt.close(fig)
    return f


def plot_summary(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = [r["sub"] for r in rows]
    x = np.arange(len(subs))
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))

    for key, col, lab in (("recon", "tab:blue", "reconstructed"),
                          ("official", "tab:red", "official v2.0.0")):
        ax[0].bar(x + (0.2 if key == "official" else -0.2),
                  [abs(r[key]["amp"]) if r.get(key) else 0 for r in rows],
                  width=0.4, color=col, label=lab)
        ax[1].bar(x + (0.2 if key == "official" else -0.2),
                  [r[key]["lat"] if r.get(key) else 0 for r in rows],
                  width=0.4, color=col, label=lab)
        ax[2].bar(x + (0.2 if key == "official" else -0.2),
                  [100 * r[key]["acc"].get("LDA", 0) if r.get(key) else 0
                   for r in rows], width=0.4, color=col, label=lab)

    ax[0].set_ylabel("|peak HbO| (uM)")
    ax[0].set_title("Peak evoked amplitude -- higher is better alignment")
    ax[1].axhspan(*CANONICAL, color="grey", alpha=0.2)
    ax[1].set_ylabel("peak latency (s)")
    ax[1].set_title("Peak latency -- the canonical HRF window is shaded")
    ax[2].axhline(33.33, color="k", ls=":", lw=1)
    ax[2].set_ylabel("within-subject acc (%)")
    ax[2].set_title("Within-subject decoding, shrinkage LDA (chance 33.3%)")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(subs, rotation=45)
        a.legend(fontsize=8)
    fig.tight_layout()
    f = RES(".png", "summary")
    fig.savefig(f, dpi=130)
    plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--root", default=paths.dataset_root())
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any checkpoint and recompute from scratch")
    args = ap.parse_args()

    evdir = layout.derived("ds004830", "events_v2")
    ckpt_path = RES(".json", "checkpoint")
    paths.require(args.root, "ds004830 derivatives tree")

    done = {}
    if os.path.exists(ckpt_path) and not args.fresh:
        with open(ckpt_path, encoding="utf-8") as fh:
            done = json.load(fh)
    log = Tee(RES(".log"), append=bool(done))

    log("Which timeline is right: reconstructed vs official ds004830 v2.0.0")
    log(f"root       : {args.root}")
    log(f"checkpoint : {ckpt_path}  ({len(done)} subject(s) already done)")
    log("")

    exps = experiment_dirs(args.root)
    channels = common_long_channels(args.root)
    log(f"shared montage: {len(channels)} long channels\n")

    for exp, path in exps:
        num = exp.replace("Experiment", "")
        if num in done:
            log(f"=== sub-{num} === (from checkpoint)")
            continue
        log(f"=== sub-{num} ===")
        try:
            runs, start_t, _end = load_manifest(path, exp)
            trig, correct = load_triggers(path)
            labels, cond = load_labels(path)
        except Exception as e:                            # noqa: BLE001
            log(f"    SKIP: {e}")
            continue

        off, types, err = official_onsets(num, evdir)
        if off is None:
            log(f"    SKIP: {err}")
            continue

        first = os.path.join(path, runs[0] + ".nirs")
        pairs, dist_ch, ident = probe_geometry(first)
        ext = np.array(sio.loadmat(first, variable_names=["SD"])["SD"][0, 0]["extCoef"],
                       dtype=float)
        pos = {k: i for i, k in enumerate(ident)}
        sel = np.array([pos[k] for k in channels])

        blocks, run_start, offset = [], {}, 0.0
        for r in runs:
            f = os.path.join(path, r + ".nirs")
            if not os.path.exists(f):
                continue
            blk, dur = process_run(f, pairs, dist_ch, ext, sel)
            run_start[r] = offset
            blocks.append(blk)
            offset += dur
        data = np.concatenate(blocks, axis=0)
        log(f"    data {data.shape}, {len(data) / FS_OUT:.0f} s over {len(blocks)} run(s)")

        # official onsets -> concatenated timeline, adding each run's start
        bids_runs = ["run-01", "run-02", "run-03"] if num == "08" else ["run-01"]
        run_off = {b: run_start.get(r, 0.0) for b, r in zip(bids_runs, runs)}
        on_off = np.array([o + run_off.get(rn, 0.0) for rn, o in off])
        lab_off = np.array([{"Right": 1, "Left": 2, "Center": 3}[t] for t in types])

        n = min(len(labels), len(trig["Trigger3"]))
        on_rec = (start_t + trig["Trigger3"])[:n]
        lab_rec = labels[:n]
        keep_rec = (cond[:n] == 1)

        res = {"sub": f"sub-{num}", "n_official": len(on_off),
               "n_recon": int(keep_rec.sum())}

        for name, ons, labs, keep in (
                ("recon", on_rec, lab_rec, keep_rec),
                ("official", on_off, lab_off, np.ones(len(on_off), bool))):
            X, y, used = epoch_at(data, ons, labs, keep)
            if len(y) == 0:
                log(f"    {name}: no usable epochs")
                continue
            amp, lat, r, t, hbo, hbr = evoked_stats(X)
            acc = decode_within(X, y)
            res[name] = {"amp": amp, "lat": lat, "r": r, "n": int(len(y)),
                         "acc": acc, "t": t.tolist(),
                         "hbo": hbo.tolist(), "hbr": hbr.tolist()}
            log(f"    {name:<9} n={len(y):<4} peak {amp:+.4f} uM @ {lat:5.1f} s  "
                f"HbO/HbR r={r:+.3f}  "
                + "  ".join(f"{k} {100 * v:.2f}%" for k, v in acc.items()))

        fig = plot_subject(num, res)
        log(f"    figure -> {os.path.basename(fig)}")
        done[num] = res
        with open(ckpt_path, "w", encoding="utf-8") as fh:
            json.dump(done, fh)
        log("    checkpoint saved")
        log("")

    rows = [done[k] for k in sorted(done)]
    if not rows:
        log("nothing computed")
        log.close()
        return 1

    log("--- per-subject comparison ---")
    log(f"{'sub':<9}{'|amp| rec':>11}{'|amp| off':>11}{'lat rec':>9}{'lat off':>9}"
        f"{'r rec':>8}{'r off':>8}{'LDA rec':>9}{'LDA off':>9}")
    for r in rows:
        a, b = r.get("recon"), r.get("official")
        if not (a and b):
            continue
        log(f"{r['sub']:<9}{abs(a['amp']):>11.4f}{abs(b['amp']):>11.4f}"
            f"{a['lat']:>9.1f}{b['lat']:>9.1f}{a['r']:>8.3f}{b['r']:>8.3f}"
            f"{100 * a['acc'].get('LDA', 0):>9.2f}{100 * b['acc'].get('LDA', 0):>9.2f}")

    def agg(key, fn):
        return fn([r[key] for r in rows if r.get(key)])

    both = [r for r in rows if r.get("recon") and r.get("official")]
    log("")
    log("--- group means ---")
    for key, lab in (("recon", "reconstructed"), ("official", "official v2.0.0")):
        amps = [abs(r[key]["amp"]) for r in both]
        lats = [r[key]["lat"] for r in both]
        rs = [r[key]["r"] for r in both]
        lda = [100 * r[key]["acc"].get("LDA", 0) for r in both]
        lg = [100 * r[key]["acc"].get("Logistic", 0) for r in both]
        in_win = sum(CANONICAL[0] <= x <= CANONICAL[1] for x in lats)
        log(f"{lab:<18} |amp| {np.mean(amps):.4f}  lat {np.mean(lats):5.2f} s "
            f"({in_win}/{len(both)} in 4-8 s)  r {np.mean(rs):+.3f}  "
            f"LDA {np.mean(lda):.2f}%  Logistic {np.mean(lg):.2f}%")

    fig = plot_summary(both)
    log(f"\nsummary figure -> {os.path.basename(fig)}")

    # the verdict, stated in terms of the criteria set out in the docstring
    r_lda = np.mean([100 * r["recon"]["acc"].get("LDA", 0) for r in both])
    o_lda = np.mean([100 * r["official"]["acc"].get("LDA", 0) for r in both])
    r_amp = np.mean([abs(r["recon"]["amp"]) for r in both])
    o_amp = np.mean([abs(r["official"]["amp"]) for r in both])
    r_r = np.mean([r["recon"]["r"] for r in both])
    o_r = np.mean([r["official"]["r"] for r in both])
    log("")
    log("--- verdict ---")
    votes = [("decoding accuracy", r_lda, o_lda, r_lda > o_lda),
             ("evoked amplitude", r_amp, o_amp, r_amp > o_amp),
             ("HbO/HbR anti-correlation", r_r, o_r, r_r < o_r)]
    for name, a, b, rec_wins in votes:
        log(f"  {name:<28} reconstructed {a:+.4f}  official {b:+.4f}  "
            f"-> {'reconstructed' if rec_wins else 'official'}")
    n_rec = sum(1 for _, _, _, w in votes if w)
    log("")
    log(f"  reconstructed wins {n_rec} of 3 criteria.")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
