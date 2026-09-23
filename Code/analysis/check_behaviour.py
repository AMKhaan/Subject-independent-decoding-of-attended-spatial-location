"""Validate the trial reconstruction against Ning et al.'s published behavioural table.

The shuffled-onset control in validate_alignment.py shows that our reconstructed onsets
beat a null. This script does something stronger and entirely independent: it checks our
per-trial response alignment against numbers the dataset authors published, which we did
not use to build anything.

Ning et al. (2024), Table 1, column "behavioural accuracy" reports, per subject, the
percentage of trials on which the participant identified *both* the target speaker's face
and their transcript. If our trial ordering, our response extraction and our correctness
criterion are all right, we must reproduce those percentages exactly -- they are counts
out of 90, so there is no rounding slack to hide in.

Two things this pinned down:

  * The criterion is the conjunction. Scoring on the transcript response alone (which the
    first version of build_dataset.py did) reproduces only 4 of 12 subjects and inflates
    the correct-trial subset from 907 to 922.

  * sub-08 ran 180 trials, not 90. Its published 94.44% is over all 180; only 90 of them
    fall inside the three .nirs runs that were distributed, and those 90 score 92.22%.
    Both figures are right for what they measure, and the script reports both.

Run it after build_dataset.py. It exits non-zero if any subject disagrees.
"""

import os
import sys

import numpy as np
import scipy.io as sio

import paths

# Ning et al. (2024), Table 1, behavioural accuracy column (percent of trials with both
# the face and the transcript identified correctly). Transcribed from the published table.
PUBLISHED = {
    "08": 94.44, "12": 93.33, "13": 96.67, "14": 77.78,
    "15": 74.44, "16": 94.44, "19": 96.67, "21": 81.11,
    "22": 78.89, "23": 78.89, "24": 95.56, "25": 48.89,
}

# The published percentages are over each subject's full behavioural record, which is not
# always the set we can epoch: sub-08 ran 180 trials and only 90 fall inside the three
# .nirs runs that were distributed, and sub-15 loses one trial to a truncated recording.
# So we test the full record against the published figure and, separately, that the flag
# we store agrees trial-for-trial with the response file on the trials we did keep.
FULL_RECORD = {"08": 180, "15": 90}


def responses(root, sub):
    """Per-trial correctness from the PsychToolbox response file, over the full record."""
    d = os.path.join(root, "Experiment" + sub)
    f = [x for x in os.listdir(d) if x.startswith("response_3M")]
    if not f:
        f = [x for x in os.listdir(d) if x.startswith("responses")]
    m = sio.loadmat(os.path.join(d, f[0]))
    a = np.array(m["responsesA"]).flatten() == np.array(m["correctRespA"]).flatten()
    v = np.array(m["responsesV"]).flatten() == np.array(m["correctRespV"]).flatten()
    return a & v


def main():
    root = paths.dataset_root()
    data = paths.data_dir()
    paths.require(root, "ds004830 derivatives tree")

    print(f"{'sub':<7}{'published':>10}{'full record':>13}{'epoched':>10}"
          f"{'n':>6}   verdict")
    bad = []
    total_used = total_correct = 0

    for sub in sorted(PUBLISHED):
        ok = responses(root, sub)
        npz = os.path.join(data, f"sub-{sub}.npz")
        if not os.path.exists(npz):
            print(f"sub-{sub:<3}{'':>10}{'':>13}{'':>10}{'':>6}   no tensor -- run build_dataset.py")
            bad.append(sub)
            continue
        d = np.load(npz, allow_pickle=True)
        used = np.asarray(d["trial_index"]).ravel()
        epoched = np.asarray(d["correct"]).ravel().astype(bool)

        full_pct = 100.0 * ok.mean()
        epoched_pct = 100.0 * epoched.mean()
        total_used += len(used)
        total_correct += int(epoched.sum())

        # the stored flag must agree trial-for-trial with the response file
        if not np.array_equal(epoched, ok[used]):
            verdict = "MISMATCH: stored flag != response file"
            bad.append(sub)
        elif abs(full_pct - PUBLISHED[sub]) >= 0.01:
            verdict = f"MISMATCH: expected {PUBLISHED[sub]:.2f}"
            bad.append(sub)
        elif sub in FULL_RECORD:
            verdict = f"ok (published covers all {len(ok)})"
        else:
            verdict = "ok"

        print(f"sub-{sub:<3}{PUBLISHED[sub]:>10.2f}{full_pct:>13.2f}"
              f"{epoched_pct:>10.2f}{len(used):>6}   {verdict}")

    print(f"\ncorrect trials in the epoched set: {total_correct} of {total_used}")

    if bad:
        print(f"\nFAILED for {', '.join('sub-' + s for s in bad)}.")
        print("The reconstruction does not reproduce the published behaviour; do not "
              "trust anything downstream until this agrees.")
        return 1
    print("\nAll twelve subjects reproduce the published behavioural accuracy exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
