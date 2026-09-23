# Running the analysis (ds004830)

How to run the pipeline. For **what the results mean**, see the
[repository README](../../README.md).

> This page describes the original ds004830 pipeline driven by `run_all.py`. The
> second-round scripts behind the subject-independent results (`build_dataset_v2.py`,
> `build_ds007738.py`, `normalisation_check.py`, `calibration_curve.py`,
> `control_eye_movement.py`, `bootstrap_ci.py`, `paper_figures.py` and the rest) run
> standalone and resolve their own paths through `layout.py`; their outputs are in
> `results/` and `plots/` at the top level.

Everything here runs against the public OpenNeuro dataset **ds004830** (Ning et al.); no data
is redistributed in this repository. The pipeline reproduces the protocol ablation, the
benchmark it should be compared against, and the trial-onset reconstruction that makes the
dataset usable at all.

## The onset reconstruction

ds004830 as distributed carries no usable trial timing: the BIDS `events.tsv` files are
empty and the Homer `.nirs` stimulus matrix `s` is identically zero. Onsets are reconstructed
from the PsychToolbox logs as

```
t_onset(i) = startT + Trigger3(i)
```

with condition labels from column 2 of `indexMoviesTest` and competing-talker trials from
column 5. `build_dataset.py` implements this; `validate_alignment.py` checks it against a
shuffled-onset control. If you want the dataset for your own work, that reconstruction is the
piece to take.

## Requirements

Python 3.11 with NumPy, SciPy, scikit-learn, PyTorch (CPU is fine) and Matplotlib. Versions
the reported numbers were produced on:

| package | version |
|---|---|
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| PyTorch | 2.13.0+cpu |
| Matplotlib | 3.11.1 |

## Running it

Download ds004830, then point the pipeline at its **derivatives** tree (the Homer and
PsychToolbox layer; the raw BIDS layer is not read at all):

```bash
python run_all.py --quick --root /path/to/ds004830/derivatives   # verifies the install
python run_all.py         --root /path/to/ds004830/derivatives   # ~9 h on 8 cores, the real run
```

The per-stage estimates printed by `--list` are the ones we measured, and two of them were
badly wrong on the first full run: `decode-3class` took 133 minutes against a 55-minute
estimate, because the 300-permutation null dominates everything else. `deep`, the stage that
looks like the long pole, came in at 237 minutes against an estimate of 240. The numbers in
the table have been corrected to the measured values.

`--quick` runs the identical code path with reduced permutation counts and 3 training
epochs. It is a smoke test, not a cheap version of the result: its numbers are noisy and
should not be quoted. It is also not especially quick: it shortens only the five stages
that take a permutation or epoch count, and the remaining eight cost about 98 minutes
between them regardless. Run `python run_all.py --list` for the per-stage breakdown, which
marks the stages `--quick` affects.

Instead of `--root` you may set `DS004830_ROOT` once in your environment. Useful flags:

```bash
python run_all.py --list                    # the stages, with rough runtimes
python run_all.py --stage deep              # run one stage
python run_all.py --from decode-3class      # resume from a stage onward
python run_all.py --dry-run                 # print the commands, run nothing
```

Stages run sequentially and stop at the first failure, because every later stage consumes an
earlier one's output. Each stage's console output is tee'd to `logs/<stage>.log`. Figures are
written to `../../manuscript/`.

## What produces what

| paper element | stage | script |
|---|---|---|
| §3.2 onset reconstruction, validation | `build`, `build-null`, `validate` | `build_dataset.py`, `validate_alignment.py` |
| §3.2 behavioural check against Ning et al.'s published table | `check-behaviour` | `check_behaviour.py` |
| §3.2 trial counts, the `sub-15` drop | `trial-profile` | `trial_profile.py` |
| §3.2 lag sweep | `lag-sweep` | `lag_sweep.py` |
| Figure 1 (pipeline schematic) | `figures` | `fig_pipeline.py` |
| Table 2, Figure 2 (onset reconstruction vs. null control) | `validate`, `figures` | `validate_alignment.py`, `fig_alignment.py` |
| Table 3, Figure 3 (protocol ablation) | `leakage` | `leakage_demo.py` |
| §4.1 channel-identity mechanism | `oracle` | `oracle_check.py` |
| Table 4, Figure 4 (honest benchmark, classical + deep) | `decode-3class`, `deep` | `decode.py`, `deep_models.py` |
| Table 5 (lateral binary) | `decode-lateral` | `decode.py` |
| Table 6, Figure 5 (time-resolved) | `time-resolved` | `time_resolved.py` |
| §4.5 lateralisation (a negative result) | `lateralisation` | `lateralisation.py` |

Table 1 has no script: it is a survey of accuracies reported in other papers, compiled by hand
from the sources cited in it.

Two figures are rendered from accuracies written into the plotting script as literals rather
than recomputed at plot time: `fig_ablation.py` (Figure 3) and `fig_subjects.py` (Figure 4).
This is so a figure can be restyled without repeating a benchmark that takes an hour. Each
script names, in its docstring, the stage whose output its literals came from, and that stage
is part of the full run, so the values are checkable, but they are not automatically
re-derived. If you re-run the benchmarks and get different numbers, update those literals.

`fig_pipeline.py` (Figure 1) is a schematic and computes nothing at all.

## Paths

No path is hard-coded. `paths.py` resolves locations in this order: an explicit flag, then
the `DS004830_ROOT` / `DS004830_DATA` / `DS004830_NULL` environment variables, then a
location relative to the repository. `run_all.py` sets those variables for its children,
which is how `validate_alignment.py`, which has no command line, ends up looking at the
same data as everything else.

## A caution about protocol D

`leakage_demo.py` deliberately reproduces a leaky, incorrect analysis (protocol D: channels
treated as independent samples, labels indexed by channel). It is there to be measured, not
reused. Its ~79% accuracy is a measurement of which optode a window came from. Do not copy
that code path into anything.
