"""
What separates the subjects who decode from the subjects who do not?

On the timing-corrected data, within-subject three-class accuracy ranges from 27% to 90%
and cross-subject from 30% to 60%. A group mean over a distribution that wide is not a
result until the spread is accounted for, and a reviewer will ask first.

This tests the candidate explanations that are already measurable, without collecting
anything new:

  behaviour        did the participant do the task? (published, reproduced exactly)
  data quality     channels surviving an SNR 1.5 prune, both ours and Ning et al.'s
                   published count
  timing           the size of the onset offset we corrected -- if the correction is
                   what drives accuracy, subjects who needed a bigger correction should
                   have gained more, and their *corrected* accuracy should not depend on
                   the offset at all
  agreement        Ning et al.'s own published per-subject accuracy, which is an
                   independent measurement of the same participants

The timing column is the one that matters most. A correlation between corrected accuracy
and the offset we removed would suggest the correction is doing something other than what
we think.

Correlations on n = 12 are weak evidence and are reported with that caveat, uncorrected,
as descriptive statistics rather than tests.

Usage:  python explain_subject_spread.py

Writes results/ds004830/explain_subject_spread/ds004830_v2_explain_subject_spread.log and
.json, and the matching .png under plots/ (see layout.py).
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy import stats

import layout

# --- measured elsewhere in this project, transcribed with their source ---------------

# decode.py --data derived/ds004830/v2 --task 3class --correct-only
#   (results/ds004830/decode/ds004830_v2_3class_correct_decode.log)
WITHIN_V2 = {"08": 63.00, "12": 90.03, "13": 51.25, "14": 27.14, "15": 48.55,
             "16": 73.18, "19": 87.91, "21": 59.18, "22": 54.04, "23": 40.57,
             "24": 69.73, "25": 29.11}
LOSO_V2 = {"08": 60.32, "12": 47.62, "13": 34.48, "14": 30.00, "15": 45.45,
           "16": 54.12, "19": 47.13, "21": 42.47, "22": 43.66, "23": 39.44,
           "24": 53.49, "25": 34.09}

# decode-3class.log, the misaligned build, best model per protocol
WITHIN_V1 = {"08": 37.75, "12": 40.22, "13": 42.14, "14": 24.86, "15": 38.70,
             "16": 46.35, "19": 40.03, "21": 54.25, "22": 32.40, "23": 35.01,
             "24": 49.54, "25": 37.61}

# check-behaviour.log; matches Ning et al. Table 1 exactly for all twelve
BEHAVIOUR = {"08": 94.44, "12": 93.33, "13": 96.67, "14": 77.78, "15": 74.44,
             "16": 94.44, "19": 96.67, "21": 81.11, "22": 78.89, "23": 78.89,
             "24": 95.56, "25": 48.89}

# Ning et al. Table 1: channels pruned at SNR 1.5, and their 3-class CV accuracy
NING_PRUNED = {"08": 2, "12": 18, "13": 8, "14": 10, "15": 2, "16": 11,
               "19": 7, "21": 19, "22": 5, "23": 18, "24": 6, "25": 12}
NING_3CLASS = {"08": 67, "12": 84, "13": 33, "14": 46, "15": 35, "16": 62,
               "19": 42, "21": 55, "22": 40, "23": 46, "24": 33, "25": 30}

# ning_build.py, our own SNR 1.5 prune -- note it does not reproduce theirs
OUR_PRUNED = {"08": 0, "12": 10, "13": 19, "14": 5, "15": 0, "16": 0,
              "19": 3, "21": 0, "22": 0, "23": 1, "24": 1, "25": 0}

# validate-v2-events.log: mean (official - reconstructed) onset, seconds.
# sub-08 and sub-19 are excluded: their comparison is contaminated by run-offset
# handling and an unexplained drift respectively, so their offsets are not trustworthy.
OFFSET = {"12": -14.99, "13": -12.26, "14": -11.50, "15": -18.38, "16": -14.69,
          "21": 2.20, "22": 10.28, "23": 11.43, "24": 7.29, "25": 1.62}


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


def corr(log, name, xd, yd, note=""):
    keys = sorted(set(xd) & set(yd))
    x = np.array([xd[k] for k in keys], float)
    y = np.array([yd[k] for k in keys], float)
    r, pr = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    log(f"  {name:<44} n={len(keys):<3} r={r:+.3f} (p={pr:.3f})   "
        f"rho={rho:+.3f} (p={prho:.3f}){('   ' + note) if note else ''}")
    return {"name": name, "n": len(keys), "r": r, "p_r": pr,
            "rho": rho, "p_rho": prho}


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    args = ap.parse_args()
    res = lambda ext: layout.result(layout.derived("ds004830", "v2"), "explain_subject_spread", ext)
    log = Tee(res(".log"))

    subs = sorted(WITHIN_V2)
    gain = {k: WITHIN_V2[k] - WITHIN_V1[k] for k in subs}
    abs_off = {k: abs(v) for k, v in OFFSET.items()}

    log("What separates the subjects who decode from those who do not?")
    log("")
    log("--- per-subject table (corrected build) ---")
    log(f"{'sub':<7}{'within':>9}{'LOSO':>8}{'v1':>8}{'gain':>8}{'behav':>8}"
        f"{'NingAcc':>9}{'NingPrn':>9}{'ourPrn':>8}{'offset':>9}")
    for k in subs:
        off = f"{OFFSET[k]:+.1f}" if k in OFFSET else "n/a"
        log(f"{'sub-' + k:<7}{WITHIN_V2[k]:>9.2f}{LOSO_V2[k]:>8.2f}{WITHIN_V1[k]:>8.2f}"
            f"{gain[k]:>+8.2f}{BEHAVIOUR[k]:>8.2f}{NING_3CLASS[k]:>9}"
            f"{NING_PRUNED[k]:>9}{OUR_PRUNED[k]:>8}{off:>9}")

    log("")
    log("--- correlates of corrected within-subject accuracy (n=12 unless noted) ---")
    res = [
        corr(log, "behavioural accuracy", BEHAVIOUR, WITHIN_V2),
        corr(log, "Ning et al. published 3-class accuracy", NING_3CLASS, WITHIN_V2),
        corr(log, "channels pruned (Ning et al. Table 1)", NING_PRUNED, WITHIN_V2),
        corr(log, "channels pruned (our SNR 1.5)", OUR_PRUNED, WITHIN_V2),
        corr(log, "|onset offset| corrected", abs_off, WITHIN_V2, "10 subjects"),
    ]
    log("")
    log("--- correlates of cross-subject (LOSO) accuracy ---")
    res += [
        corr(log, "behavioural accuracy", BEHAVIOUR, LOSO_V2),
        corr(log, "within-subject accuracy", WITHIN_V2, LOSO_V2),
        corr(log, "|onset offset| corrected", abs_off, LOSO_V2, "10 subjects"),
    ]
    log("")
    log("--- what the correction bought, per subject ---")
    res += [
        corr(log, "|onset offset| vs accuracy gain", abs_off, gain, "10 subjects"),
        corr(log, "misaligned accuracy vs gain", WITHIN_V1, gain),
    ]

    log("")
    log("--- reading ---")
    log("The decisive one is '|onset offset| corrected' against within-subject accuracy.")
    log("If the correction did what we think, a subject's *corrected* accuracy should be")
    log("independent of how large a correction they needed -- the offset is a property of")
    log("the clock, not of the participant. A strong correlation there would mean the")
    log("correction is doing something else and the result needs re-examining.")
    log("")
    log("'|onset offset| vs accuracy gain' should be positive: subjects whose onsets were")
    log("further off had more to gain. That is the mechanism claim, stated as a number.")
    log("")
    log("n = 12 (10 where the offset is involved). Nothing here survives multiplicity")
    log("correction and none is offered as a test; these are descriptive.")

    with open(res(".json"), "w", encoding="utf-8") as fh:
        json.dump({"correlations": res,
                   "within_v2": WITHIN_V2, "loso_v2": LOSO_V2,
                   "gain": gain, "offset": OFFSET}, fh, indent=1)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    a = ax[0, 0]
    x = [BEHAVIOUR[k] for k in subs]
    a.scatter(x, [WITHIN_V2[k] for k in subs], label="within", color="tab:blue")
    a.scatter(x, [LOSO_V2[k] for k in subs], label="LOSO", color="tab:red", marker="s")
    for k in subs:
        a.annotate(k, (BEHAVIOUR[k], WITHIN_V2[k]), fontsize=7)
    a.axhline(33.33, color="k", ls=":", lw=1)
    a.set_xlabel("behavioural accuracy (%)")
    a.set_ylabel("decoding accuracy (%)")
    a.set_title("Did the participant do the task?")
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.scatter([NING_3CLASS[k] for k in subs], [WITHIN_V2[k] for k in subs])
    lim = [25, 95]
    a.plot(lim, lim, "k--", lw=0.8)
    for k in subs:
        a.annotate(k, (NING_3CLASS[k], WITHIN_V2[k]), fontsize=7)
    a.set_xlabel("Ning et al. published 3-class (%)")
    a.set_ylabel("ours, corrected (%)")
    a.set_title("Agreement with the published per-subject values")

    a = ax[1, 0]
    ks = sorted(OFFSET)
    a.scatter([abs(OFFSET[k]) for k in ks], [WITHIN_V2[k] for k in ks])
    for k in ks:
        a.annotate(k, (abs(OFFSET[k]), WITHIN_V2[k]), fontsize=7)
    a.set_xlabel("|onset offset| corrected (s)")
    a.set_ylabel("corrected within-subject accuracy (%)")
    a.set_title("Should show NO relationship if the fix is sound")

    a = ax[1, 1]
    a.scatter([abs(OFFSET[k]) for k in ks], [gain[k] for k in ks])
    for k in ks:
        a.annotate(k, (abs(OFFSET[k]), gain[k]), fontsize=7)
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("|onset offset| corrected (s)")
    a.set_ylabel("accuracy gained (pp)")
    a.set_title("Should be POSITIVE: bigger error, bigger fix")

    fig.tight_layout()
    f = res(".png")
    fig.savefig(f, dpi=130)
    plt.close(fig)
    log(f"\nfigure -> {layout.rel(f)}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
