"""Run the whole analysis, from the OpenNeuro download to the manuscript figures.

    python run_all.py                      # everything, full permutation counts
    python run_all.py --quick              # same path, tiny permutation counts (smoke test)
    python run_all.py --list               # what the stages are and roughly how long
    python run_all.py --stage deep         # just one
    python run_all.py --from decode-3class # that stage and everything after it

Point it at the data with the DS004830_ROOT environment variable, or --root:

    python run_all.py --root /path/to/ds004830/derivatives

Stages run one at a time and in order. That is deliberate: several of them already
saturate every core through sklearn's n_jobs or torch's thread pool, so running two
at once on an 8-core machine makes both slower rather than finishing sooner.

Every stage's console output is tee'd to logs/<stage>.log, and a stage that exits
non-zero stops the run -- the later stages all consume the earlier ones' outputs, so
continuing past a failure would silently mix fresh and stale results.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent.parent / "manuscript"


class Stage:
    def __init__(self, name, argv, minutes, note):
        self.name, self.argv, self.minutes, self.note = name, argv, minutes, note


def build_stages(quick):
    # permutation counts and training epochs are the only things --quick changes; the
    # code path is identical, so a quick run still exercises every stage end to end
    p_decode = 20 if quick else 300
    p_lat = 200 if quick else 5000
    p_time = 20 if quick else 200
    epochs = 3 if quick else 40
    fig = lambda n: ["--out", str(FIGDIR / n)]

    return [
        Stage("build", ["build_dataset.py"], 6,
              "epoch on reconstructed onsets -> data/"),
        Stage("build-null", ["build_dataset.py", "--null"], 6,
              "same, on shuffled onsets -> data_null/ (negative control)"),
        Stage("validate", ["validate_alignment.py"], 1,
              "onset reconstruction vs. the null control"),
        Stage("check-behaviour", ["check_behaviour.py"], 1,
              "reconstruction vs. Ning et al.'s published behavioural table"),
        Stage("trial-profile", ["trial_profile.py"], 2,
              "trial counts, class balance, the sub-15 drop"),
        Stage("lag-sweep", ["lag_sweep.py"], 12,
              "accuracy as a function of an imposed onset shift"),
        Stage("lateralisation", ["lateralisation.py", "--perms", str(p_lat)], 4,
              "the univariate lateralisation index (a clean negative)"),
        Stage("decode-3class", ["decode.py", "--task", "3class", "--correct-only",
                                "--perms", str(p_decode)], 133,
              "Table 4 (classical rows): four models, LOSO and within-subject"),
        Stage("decode-3class-alltrials", ["decode.py", "--task", "3class",
                                          "--perms", "0"], 10,
              "Table 4 without the correct-trial restriction (no permutations)"),
        Stage("decode-lateral", ["decode.py", "--task", "lateral",
                                 "--perms", str(p_decode)], 21,
              "Table 5: the two-class left-vs-right variant"),
        Stage("behaviour-corr", ["behaviour_correlation.py"], 1,
              "§4.4: does the participant who performed better decode better?"),
        Stage("leakage", ["leakage_demo.py"], 25,
              "Table 3: protocols A-D on identical trials and features"),
        Stage("oracle", ["oracle_check.py"], 45,
              "how much of protocol D is channel identity rather than attention"),
        Stage("time-resolved", ["time_resolved.py", "--win", "2", "--step", "0.5",
                                "--perms", str(p_time),
                                "--fig", str(FIGDIR / "fig_time_resolved.png")], 40,
              "Figure 5: accuracy by window position, max-statistic corrected"),
        Stage("time-resolved-4s", ["time_resolved.py", "--win", "4", "--step", "0.5",
                                   "--perms", "0"], 8,
              "§4.3 robustness: the same sweep at protocol C's window length"),
        Stage("deep", ["deep_models.py", "--protocol", "both",
                       "--epochs", str(epochs)], 237,
              "Table 4 (deep rows): seven neural architectures (the long pole)"),
        Stage("figures", None, 1,
              "Figures 1-4 into manuscript/ (Figure 5 comes from time-resolved)"),
    ], [
        ["fig_pipeline.py"] + fig("fig_pipeline.png"),
        ["fig_alignment.py"] + fig("fig_alignment.png"),
        ["fig_ablation.py"] + fig("fig_ablation.png"),
        ["fig_subjects.py"] + fig("fig_subjects.png"),
    ]


def run(argv, log, env):
    """Run one script, echoing to the console and to log at the same time."""
    cmd = [sys.executable, "-u"] + argv
    print(f"  $ {' '.join(cmd[2:])}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write("    " + line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
    return proc.wait()


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--root", help="ds004830 derivatives tree "
                                   "(default: $DS004830_ROOT, then ../../Dataset/derivatives)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny permutation counts and 3 training epochs, for a smoke test")
    ap.add_argument("--stage", help="run only this stage")
    ap.add_argument("--from", dest="from_", help="run this stage and everything after it")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()

    stages, figure_cmds = build_stages(args.quick)

    if args.list:
        total = sum(s.minutes for s in stages)
        # --quick only shrinks stages that take a permutation or epoch count; the rest
        # cost the same either way, so quoting a single "quick runtime" would mislead.
        # "--perms 0" is not shortenable -- there is nothing there to shorten.
        def shrinks(s):
            if not s.argv:
                return False
            if "--epochs" in s.argv:
                return True
            return "--perms" in s.argv and s.argv[s.argv.index("--perms") + 1] != "0"
        fixed = sum(s.minutes for s in stages if not shrinks(s))
        print(f"{len(stages)} stages, roughly {total // 60}h{total % 60:02d}m on 8 cores.\n"
              f"--quick shortens the {sum(1 for s in stages if shrinks(s))} stages marked [q]; "
              f"the other {sum(1 for s in stages if not shrinks(s))} are unaffected and\n"
              f"account for ~{fixed} min on their own, so --quick is a smoke test, not a "
              f"fast reproduction.\n")
        for s in stages:
            print(f"  {'[q]' if shrinks(s) else '   '} {s.name:<24} ~{s.minutes:>4} min   {s.note}")
        return 0

    names = [s.name for s in stages]
    if args.stage:
        if args.stage not in names:
            raise SystemExit(f"unknown stage {args.stage!r}; try --list")
        stages = [s for s in stages if s.name == args.stage]
    elif args.from_:
        if args.from_ not in names:
            raise SystemExit(f"unknown stage {args.from_!r}; try --list")
        stages = stages[names.index(args.from_):]

    root = paths.dataset_root(args.root)
    if not args.dry_run:
        paths.require(root, "ds004830 derivatives tree")

    # children inherit the resolved locations, so scripts without a command line
    # (validate_alignment.py) land on the same data as the ones with flags
    env = dict(os.environ)
    env[paths.ENV_ROOT] = root
    env[paths.ENV_DATA] = paths.data_dir()
    env[paths.ENV_NULL] = paths.null_dir()
    env["PYTHONIOENCODING"] = "utf-8"  # the scripts print µM and arrows

    logdir = HERE / "logs"
    logdir.mkdir(exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print(f"dataset : {root}")
    print(f"tensors : {env[paths.ENV_DATA]}")
    print(f"figures : {FIGDIR}")
    print(f"logs    : {logdir}")
    print(f"mode    : {'quick smoke test' if args.quick else 'full'}\n")

    t_run = time.perf_counter()
    for i, s in enumerate(stages, 1):
        cmds = figure_cmds if s.argv is None else [s.argv]
        print(f"[{i}/{len(stages)}] {s.name}  --  {s.note}")
        if args.dry_run:
            for c in cmds:
                print(f"  $ {' '.join(c)}")
            continue
        t0 = time.perf_counter()
        for c in cmds:
            code = run(c, logdir / f"{s.name}.log", env)
            if code != 0:
                print(f"\n{s.name} failed (exit {code}). See {logdir / (s.name + '.log')}.")
                print("Stopping: the later stages read this one's output.")
                return code
        print(f"  done in {(time.perf_counter() - t0) / 60:.1f} min\n")

    if not args.dry_run:
        print(f"all stages complete in {(time.perf_counter() - t_run) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
