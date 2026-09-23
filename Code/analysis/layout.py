"""Where the second-round analyses read their data and write their results.

paths.py (archived with the paper's first version) owns the original pipeline: the
ds004830 derivatives tree and the data/ and data_null/ tensors that run_all.py builds.
This module owns everything added after that, under one rule so that a file's name
tells you where it is and which script made it:

    derived/<dataset>/<variant>/                      epoch tensors (input data)
    results/<dataset>/<script>/<prefix>_<script>[_<part>].<ext>   logs, JSON, checkpoints
    plots/<dataset>/<script>/<prefix>_<script>[_<part>].png       graphs of those results
    manuscript/figures/                               paper figures (paper_figures.py)

<dataset> is ds004830 or ds007738, <script> is the stem of the script that wrote the
file, and <prefix> is <dataset>_<variant>[_<task>], where <variant> is the name of the
derived/ folder the analysis read. So derived/ds007738/overt feeds
results/ds007738/bootstrap_ci/ds007738_overt_bootstrap_ci.log and
plots/ds007738/bootstrap_ci/ds007738_overt_bootstrap_ci.png.

The raw datasets are not moved: ds004830 stays in Dataset/ (see paths.py) and ds007738
in D:/fnirs/ds007738/raw (DS007738_RAW overrides it).
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

DERIVED = REPO / "derived"
RESULTS = REPO / "results"
PLOTS = REPO / "plots"
PAPER_FIGURES = REPO / "manuscript" / "figures"
NOTES = REPO / "notes"
DATASETS = ("ds004830", "ds007738")

DS007738_RAW = os.environ.get("DS007738_RAW", "D:/fnirs/ds007738/raw")


def derived(dataset, variant):
    """derived/<dataset>/<variant>, the epoch tensors one analysis reads."""
    return str(DERIVED / dataset / variant)


def split(data_dir):
    """(dataset, variant) of a derived/ folder, e.g. derived/ds007738/overt ->
    ("ds007738", "overt"). A folder outside derived/ is treated as a ds004830 variant
    named after the folder, which keeps ad hoc --data paths working."""
    p = Path(os.path.normpath(data_dir))
    if p.parent.name in DATASETS:
        return p.parent.name, p.name
    return "ds004830", p.name


def prefix(data_dir, *extra):
    """<dataset>_<variant>[_<extra>...] for a derived/ folder."""
    ds, var = split(data_dir)
    return "_".join([ds, var] + [e for e in extra if e])


def out(dataset, script, stem, ext):
    """Path of one output file; PNGs go under plots/, everything else under results/.

    stem is the file name without extension and must already carry the
    <prefix>_<script> part (use name() to build it). The folder is created."""
    root = PLOTS if ext == ".png" else RESULTS
    d = root / dataset / script
    d.mkdir(parents=True, exist_ok=True)
    return str(d / (stem + ext))


def name(prefix_, script, part=""):
    """<prefix>_<script>[_<part>]."""
    return f"{prefix_}_{script}" + (f"_{part}" if part else "")


def result(data_dir_or_dataset, script, ext, *extra, part=""):
    """Shorthand: the output file of <script> for one data folder (or, for analyses
    that read no single folder, one dataset name), e.g.
    result(args.data, "bootstrap_ci", ".json") ->
    results/ds007738/bootstrap_ci/ds007738_overt_bootstrap_ci.json"""
    if data_dir_or_dataset in DATASETS:
        ds, pre = data_dir_or_dataset, "_".join([data_dir_or_dataset] + [e for e in extra if e])
    else:
        ds = split(data_dir_or_dataset)[0]
        pre = prefix(data_dir_or_dataset, *extra)
    return out(ds, script, name(pre, script, part), ext)


def rel(path):
    """A path relative to the repository root, for log lines."""
    try:
        return str(Path(path).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)
