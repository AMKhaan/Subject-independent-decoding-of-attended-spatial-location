"""Where the dataset and the derived tensors live.

Every script in this directory used to carry an absolute path to one developer's
F: drive as its argparse default, which meant a reader who cloned the repository had
to hand-edit six files before anything would run. Paths are resolved here instead,
in this order:

    1. an explicit argument (``--root`` / ``--data``), if the caller passed one
    2. the environment variables DS004830_ROOT / DS004830_DATA / DS004830_NULL
    3. a location relative to this file, for the in-repo layout

The environment-variable layer is what lets ``run_all.py`` point the whole pipeline
at one dataset copy without every script having to accept a flag -- which matters
for validate_alignment.py, which has no command line at all.

Nothing here touches the filesystem except ``require()``, so importing this module
is free and safe from any working directory.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# analysis/ -> Code/ -> Paper/ ; the dataset sits beside Code/ in the released layout
_REPO = HERE.parent.parent

ENV_ROOT, ENV_DATA, ENV_NULL = "DS004830_ROOT", "DS004830_DATA", "DS004830_NULL"


def dataset_root(explicit=None):
    """Directory holding the ds004830 *derivatives* tree (sub-08/, sub-12/, ...).

    This is the Homer/PsychToolbox derivatives layer, not the BIDS raw layer: the
    trial onsets are reconstructed from the behavioural logs found here (see §3.2),
    and nothing in the BIDS layer is read at all.
    """
    if explicit:
        return str(Path(explicit).expanduser())
    env = os.environ.get(ENV_ROOT)
    if env:
        return str(Path(env).expanduser())
    return str(_REPO / "Dataset" / "derivatives")


def data_dir(explicit=None):
    """Directory of per-subject .npz epoch tensors written by build_dataset.py."""
    if explicit:
        return str(Path(explicit).expanduser())
    env = os.environ.get(ENV_DATA)
    if env:
        return str(Path(env).expanduser())
    return str(HERE / "data")


def null_dir(explicit=None):
    """Same, but epoched on shuffled onsets -- the negative control behind Figure 2."""
    if explicit:
        return str(Path(explicit).expanduser())
    env = os.environ.get(ENV_NULL)
    if env:
        return str(Path(env).expanduser())
    return str(HERE / "data_null")


def require(path, what):
    """Fail early and legibly rather than deep inside a loader.

    A missing derivatives tree and a not-yet-built tensor directory are the two
    mistakes a new user actually makes, and the numpy error they otherwise produce
    names neither the path nor the command that would fix it.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"{what} not found at:\n    {p}\n\n"
            f"Set {ENV_ROOT} (or pass the matching flag) to the location of the\n"
            f"ds004830 derivatives tree, or run  python run_all.py --stage build\n"
            f"first if this is a derived directory."
        )
    return str(p)
