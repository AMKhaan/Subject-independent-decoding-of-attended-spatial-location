"""
Does the participant who performed the task better also decode better?

This is the §4.4 analysis. It computes nothing new from the recordings: every
input is a per-participant number already produced and logged by an earlier
stage, and the point of collecting them here is that the correlations quoted in
§4.4 should be recomputable from the released logs rather than taken on trust.

Inputs, all read out of logs/:

  check-behaviour.log            behavioural accuracy, verified against Ning et al.
  decode-3class.log              per-subject accuracy, correct trials only
  decode-3class-alltrials.log    per-subject accuracy, all trials
  decode-lateral.log             per-subject accuracy, left-vs-right

Behavioural accuracy comes from button presses and never reaches the classifier,
so a pipeline artefact has no route by which to manufacture a relationship
between the two. That independence is the whole reason the check is worth
running -- an accuracy figure on its own cannot distinguish signal from a
systematic artefact, and this can.

Both Spearman and Pearson are reported for every pair, deliberately: at n = 12
they disagree often enough that quoting only the favourable one would be a
choice, and §4.4's argument depends on admitting where they disagree.

Usage:  python behaviour_correlation.py [--logs PATH]
"""

import argparse
import os
import re
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))

# The per-subject table every decode.py run prints, e.g.
#   sub-08                    37.75%         39.76%
ROW = re.compile(r"^\s*(sub-\d+)\s+([\d.]+)%\s+([\d.]+)%\s*$")
HEADER = "=== per-subject accuracy, best model in each protocol ==="


def read_decode(path):
    """(within, loso) dicts of subject -> accuracy in percent."""
    within, loso = {}, {}
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == HEADER)
    except StopIteration:
        raise SystemExit("no per-subject block in " + path)
    for line in lines[start + 2:]:
        m = ROW.match(line)
        if not m:
            if line.strip():
                break
            continue
        within[m.group(1)] = float(m.group(2))
        loso[m.group(1)] = float(m.group(3))
    return within, loso


def read_behaviour(path):
    """subject -> behavioural accuracy in percent, from check_behaviour.py's table.

    The `epoched` column is the one taken, not `published`: it is scored over
    exactly the trials that reach the classifier, which is what the correlation
    is about. The two differ for only two participants and by less than a third
    of a point, but they break a tie between two participants who are both at
    94.44% in the published table, and at n = 12 one tie is worth ~0.04 of a
    rank correlation.
    """
    out = {}
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0].startswith("sub-"):
                out[parts[0]] = float(parts[3])
    return out


def align(a, b):
    subs = sorted(set(a) & set(b))
    return subs, np.array([a[s] for s in subs]), np.array([b[s] for s in subs])


def report(label, a, b):
    subs, x, y = align(a, b)
    rho, p_rho = spearmanr(x, y)
    r, p_r = pearsonr(x, y)
    print(f"  {label:<46} n={len(subs):2d}   "
          f"rho {rho:+.2f} p {p_rho:.3f}    r {r:+.2f} p {p_r:.3f}")
    return p_rho, p_r


def drop(d, sub):
    return {k: v for k, v in d.items() if k != sub}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(HERE, "logs"))
    args = ap.parse_args()

    def log(name):
        p = os.path.join(args.logs, name)
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}\nRun the stage that produces it first "
                f"(python run_all.py --list shows which).")
        return p

    beh = read_behaviour(log("check-behaviour.log"))
    c_within, c_loso = read_decode(log("decode-3class.log"))
    a_within, a_loso = read_decode(log("decode-3class-alltrials.log"))
    l_within, l_loso = read_decode(log("decode-lateral.log"))

    print(f"{len(beh)} participants; behavioural accuracy "
          f"{min(beh.values()):.1f}-{max(beh.values()):.1f}%\n")

    print("=== behaviour vs decoding ===")
    ps = []
    ps.append(report("behaviour ~ within-subject 3-class (correct only)", beh, c_within))
    ps.append(report("behaviour ~ within-subject 3-class (all trials)", beh, a_within))
    ps.append(report("behaviour ~ LOSO 3-class (correct only)", beh, c_loso))
    ps.append(report("behaviour ~ within-subject lateral", beh, l_within))
    ps.append(report("behaviour ~ LOSO lateral", beh, l_loso))

    flat = [p for pair in ps for p in pair]
    print(f"\n  {len(flat)} comparisons, none corrected. smallest p = {min(flat):.3f}, "
          f"Bonferroni threshold 0.05/{len(flat)} = {0.05 / len(flat):.4f} -> "
          f"{'none survives' if min(flat) > 0.05 / len(flat) else 'survives'}")

    print("\n=== consistency between the two decoding problems ===")
    print("  (not independent: the lateral problem reuses 720 of the same trials)")
    report("lateral within ~ 3-class within (all trials)", l_within, a_within)
    report("lateral within ~ 3-class within (correct only)", l_within, c_within)

    print("\n=== within-subject vs LOSO, same participants (§4.2) ===")
    print("  (a participant who decodes well within-subject need not transfer)")
    report("within 3-class ~ LOSO 3-class (correct only)", c_within, c_loso)

    print("\n=== the cost of excluding sub-25, as Ning et al. do ===")
    for name, d in (("within 3-class (correct only)", c_within),
                    ("LOSO 3-class (correct only)", c_loso),
                    ("within lateral", l_within)):
        all12 = np.mean(list(d.values()))
        eleven = np.mean(list(drop(d, "sub-25").values()))
        print(f"  {name:<32} 12 subj {all12:6.2f}%   11 subj {eleven:6.2f}%   "
              f"{eleven - all12:+.2f} pp")

    print("\n=== spread across participants (within-subject, correct only) ===")
    subs, x, _ = align(c_within, c_within)
    order = np.argsort(-x)
    print(f"  best  {subs[order[0]]} {x[order[0]]:.2f}%      "
          f"worst {subs[order[-1]]} {x[order[-1]]:.2f}%")
    subs, x, _ = align(l_within, l_within)
    order = np.argsort(-x)
    print(f"  lateral: best {subs[order[0]]} {x[order[0]]:.2f}%   "
          f"worst {subs[order[-1]]} {x[order[-1]]:.2f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
