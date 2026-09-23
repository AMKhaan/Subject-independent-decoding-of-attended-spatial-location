# Subject-independent decoding of attended spatial location from fNIRS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22911029.svg)](https://doi.org/10.5281/zenodo.22911029)

Analysis code and results for a study of whether the location a listener attends to can be
decoded from functional near-infrared spectroscopy **in a person who was not in the training
set**, how quickly accuracy grows once that person contributes a few labelled trials, and how
much of what transfers is shared with eye movement.

Two public OpenNeuro datasets are analysed with one fixed pipeline:

| dataset | participants | task |
|---|---|---|
| [ds004830](https://openneuro.org/datasets/ds004830) | 12 | attend one of three talkers, screens at 0 and ±45° |
| [ds007738](https://openneuro.org/datasets/ds007738) | 29 | attend one of two, screens at ±30°, plus covert attention and an eye-movement-only task |

Neither dataset is redistributed here.

## What the code shows

Decoders are trained on every participant but one and tested on the one left out, with
selection-aware permutation tests. Features for the new person are normalised either over
their whole recording (**transductive**, what most papers do) or on past trials only
(**causal**, what a live system can actually do).

| | chance | causal accuracy | p |
|---|---|---|---|
| ds004830, three locations | 33.3% | **39.0%** | 0.005 (floor) |
| ds004830, two locations | 50% | **58.6%** | 0.020 |
| ds007738, two locations | 50% | **66.9%** | 0.005 (floor) |

Whole-recording normalisation overstates accuracy by up to 3.4 points. Combining other
people's data with four labelled trials from the new person beats a decoder trained on those
four trials alone by 5.0 points, mostly because the trials set the scale of the new person's
signals. A decoder trained **only on eye movements** classifies the attention trials of unseen
people at 62.0%, so in overt tasks much of the transferable signal is oculomotor. Covert
attention, where the eyes stay fixed, transfers only weakly between people (52.8%) and shows
no transfer to or from eye movement.

## Layout

```
Code/analysis/      every analysis script, and requirements.txt
Code/analysis/logs/ console output of the original ds004830 pipeline (run_all.py)
results/            logs, JSON and console captures, by <dataset>/<script>/
plots/              the graph for each of those results, same tree
figures/            the figures as they appear in the paper (fig1-7, figS1-S7)
```

Every result file is named `<dataset>_<variant>[_<task>][_correct]_<script>[_<part>]`, built by
`Code/analysis/layout.py` rather than by hand, so a file always says which dataset, which
epoch tensor and which script it came from. The epoch tensors themselves (`derived/`, about
900 MB) are rebuilt by the build stages and are not versioned.

## Running it

Python 3.11; dependencies pinned in `Code/analysis/requirements.txt` (NumPy, SciPy,
scikit-learn, PyTorch, Matplotlib). Download the datasets from OpenNeuro and point the
environment at them:

```bash
cd Code/analysis
export DS004830_ROOT=/path/to/ds004830/derivatives   # the Homer/PsychToolbox layer
python run_all.py --list                             # stages, with measured runtimes
python run_all.py --quick                            # smoke test, reduced permutations
python run_all.py                                    # full ds004830 run, about 9 h on 8 cores
```

The second-round scripts (`build_dataset_v2.py`, `build_ds007738.py`, `normalisation_check.py`,
`calibration_curve.py`, `control_eye_movement.py`, `bootstrap_ci.py`, `paper_figures.py` and
the rest) run standalone and resolve their own paths through `layout.py`.

Two cautions carried over from the earlier release:

- **`leakage_demo.py` protocol D is deliberately wrong.** It reproduces channels-as-samples
  with a trial-indexed label array read by channel number, which is worth about 79% on a
  three-class problem. It exists to be measured, not reused.
- **`fig_ablation.py` and `fig_subjects.py` hold hard-coded accuracy literals** so a figure can
  be restyled without repeating an hour-long benchmark. Re-running the benchmark does not
  update them.

Trial onsets in ds004830 are **reconstructed, not read**: the released BIDS `events.tsv` files
are empty and the Homer stimulus matrix is identically zero. `build_dataset_v2.py`
reconstructs them from the PsychToolbox logs and validates the result against the official
v2.0.0 events, a shuffled-onset control and a global lag sweep.

## How to cite

Cite the archived release, not this URL. The concept DOI
[10.5281/zenodo.22911029](https://doi.org/10.5281/zenodo.22911029) always resolves to the
newest version; the version archived for the paper is
[10.5281/zenodo.22911030](https://doi.org/10.5281/zenodo.22911030) (v1.0.0).

> Khan A M, Nasir B and Khan H M 2026 *Analysis code for subject-independent decoding of
> attended spatial location from fNIRS* (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.22911030

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## Licence

MIT (see [`LICENSE`](LICENSE)). The datasets are not covered by this licence; ds004830 and
ds007738 carry their own terms on OpenNeuro.

## Related

The earlier study on how the choice of evaluation protocol moves reported accuracy is at
[Evaluation-protocol-in-fNIRS-spatial-attention-decoding](https://github.com/AMKhaan/Evaluation-protocol-in-fNIRS-spatial-attention-decoding).
