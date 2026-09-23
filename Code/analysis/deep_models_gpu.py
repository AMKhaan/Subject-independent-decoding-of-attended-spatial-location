"""
The seven deep architectures, re-run on GPU, multi-seed, with power-cut recovery.

Why this exists
---------------
`deep_models.py` produced the project's deep-learning null: all seven architectures at
chance under leave-one-subject-out, none beating shrinkage LDA within-subject. That
result is now void for two independent reasons.

  1. It was computed on the misaligned epochs. Correcting the onsets against the
     official v2.0.0 events moved classical within-subject accuracy from 39.5% to 57.8%
     and cross-subject from 33.7% to 44.4%, so every deep number was measured on a
     degraded representation.
  2. It was single-seed. `deep_models.py:33` sets `torch.manual_seed(0)` and each cell
     is run once, with the fold or subject index as the seed. The spread between
     architectures was 1.5 points under LOSO, which is almost certainly inside
     seed-to-seed noise at 90 trials per subject. The ranking was never resolved.

This runner fixes both: it reads whichever data directory you point it at, and it runs
every cell across several seeds so the dispersion is measured rather than assumed.

Three defects in the original training loop are also corrected, and each changes results
rather than merely tidying:

  * **Validation split.** `deep_models.py:291-292` takes `tr[:n_va]` as the within-subject
    validation set -- the *lowest trial indices*, i.e. the earliest trials of the session.
    Early trials differ systematically in drift and arousal, so early stopping was being
    driven by a biased sample. This uses a stratified random split.
  * **Device.** The original is CPU-only; it sets a thread count and nothing else. The
    full run took 237 minutes.
  * **Class weights** are computed on the training fold, as before, but moved to the
    device with the model.

Power-cut recovery
------------------
Every (protocol, architecture, subject/fold, seed) cell is written to a JSON checkpoint
the moment it finishes, along with the config it was computed under. Re-running resumes
and skips completed cells, so an interruption costs at most one cell -- seconds to a
couple of minutes. The checkpoint is rewritten atomically via a temporary file, so a cut
*during* the write cannot corrupt it. Use --fresh to start over.

Usage:
    python deep_models_gpu.py --data ../../derived/ds004830/v2 --seeds 5
    python deep_models_gpu.py --data ../../derived/ds004830/v2 --seeds 5 --arch HemoNet --protocol within
    python deep_models_gpu.py --list-devices
"""

import argparse
import glob
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

import layout
import paths
from deep_models import ARCHS

N_CLASSES = 3


class Tee:
    def __init__(self, path, append=False):
        self.fh = open(path, "a" if append else "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())      # survive a power cut mid-run

    def close(self):
        self.fh.close()


def pick_device(want):
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if want == "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False. "
                         "Install a CUDA build of torch, or pass --device cpu.")
    return torch.device("cpu")


def load(data_dir, correct_only=False):
    """correct_only keeps behaviourally correct trials after the per-subject z-score,
    so the deep models see exactly the trials decode.py --correct-only reports on."""
    subs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        X = z["X"].transpose(0, 2, 1)                     # (trials, channels, time)
        m, s = X.mean(axis=(0, 2), keepdims=True), X.std(axis=(0, 2), keepdims=True)
        X = (X - m) / (s + 1e-9)                          # per-subject z-score
        y = z["y"].astype(int) - 1
        if correct_only:
            ok = z["correct"].astype(bool)
            X, y = X[ok], y[ok]
        subs.append((str(z["subject"]), X.astype(np.float32), y))
    return subs


def train_eval(Model, Xtr, ytr, Xva, yva, Xte, yte, device, epochs,
               lr=1e-3, batch=64, patience=10, seed=0):
    """Train with early stopping on a validation set, then score the test set."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    net = Model(Xtr.shape[1], N_CLASSES).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    cnt = np.bincount(ytr, minlength=N_CLASSES).astype(float)
    w = torch.tensor(cnt.sum() / (N_CLASSES * np.maximum(cnt, 1)),
                     dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)

    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    Xtr_t, ytr_t = t(Xtr, torch.float32), t(ytr, torch.long)
    Xva_t, yva_t = t(Xva, torch.float32), t(yva, torch.long)
    Xte_t = t(Xte, torch.float32)

    best, best_state, bad, ran = -1.0, None, 0, 0
    for ep in range(epochs):
        ran = ep + 1
        net.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            lossf(net(Xtr_t[idx]), ytr_t[idx]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            va = (net(Xva_t).argmax(1) == yva_t).float().mean().item()
        if va > best:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(Xte_t).argmax(1).cpu().numpy()
    return float((pred == yte).mean()), ran, float(best)


def stratified_val_split(y, frac=0.2, min_n=6, seed=0):
    """Random, class-balanced validation indices.

    The original took the lowest trial indices, which are the earliest trials of the
    session and differ systematically in drift and arousal from the rest.
    """
    rng = np.random.default_rng(seed)
    n_va = max(min_n, int(round(frac * len(y))))
    per = max(1, n_va // len(np.unique(y)))
    va = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        va.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
    va = np.array(sorted(set(va)))
    tr = np.setdiff1d(np.arange(len(y)), va)
    return tr, va


def save_ckpt(path, obj):
    """Atomic write: a power cut during the write cannot corrupt the checkpoint."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--data", default=None, help="epoch tensor directory")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--protocol", default="both", choices=("loso", "within", "both"))
    ap.add_argument("--arch", default=None, help="run a single architecture")
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--correct-only", action="store_true",
                    help="behaviourally correct trials only, as the classical headline")
    args = ap.parse_args()

    if args.list_devices:
        print(f"torch {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"device 0: {p.name}, {p.total_memory / 1e9:.1f} GB, cc {p.major}.{p.minor}")
        return 0

    data = args.data or paths.data_dir()
    sel = "correct" if args.correct_only else ""
    tag = layout.prefix(data, sel)
    res = lambda ext, part="": layout.result(data, "deep_models_gpu", ext, sel, part=part)
    ckpt_path = res(".json", "checkpoint")
    device = pick_device(args.device)

    state = {"config": {}, "cells": {}}
    if os.path.exists(ckpt_path) and not args.fresh:
        with open(ckpt_path, encoding="utf-8") as fh:
            state = json.load(fh)
    cfg = {"data": tag, "seeds": args.seeds, "epochs": args.epochs}
    if state["cells"] and state.get("config") and state["config"] != cfg:
        raise SystemExit(
            f"checkpoint at {ckpt_path} was written under a different config:\n"
            f"  stored  {state['config']}\n  current {cfg}\n"
            f"Pass --fresh to discard it, or match the stored settings.")
    state["config"] = cfg

    log = Tee(res(".log"), append=bool(state["cells"]))
    paths.require(data, "epoch tensor directory")
    subs = load(data, args.correct_only)
    n_trials = sum(len(s[2]) for s in subs)

    log(f"deep models | data={data} | device={device} "
        f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    log(f"{len(subs)} subjects, {n_trials} trials, "
        f"input {subs[0][1].shape[1]} channels x {subs[0][1].shape[2]} samples")
    log(f"seeds={args.seeds}  max epochs={args.epochs}  chance={100 / N_CLASSES:.1f}%")
    log(f"checkpoint {ckpt_path} ({len(state['cells'])} cell(s) already done)")
    log("")

    archs = {args.arch: ARCHS[args.arch]} if args.arch else ARCHS
    if args.arch and args.arch not in ARCHS:
        raise SystemExit(f"unknown arch {args.arch}; choices: {', '.join(ARCHS)}")
    protocols = ["within", "loso"] if args.protocol == "both" else [args.protocol]

    t_start = time.perf_counter()
    for proto in protocols:
        for name, Model in archs.items():
            for seed in range(args.seeds):
                for i, (sname, X, y) in enumerate(subs):
                    key = f"{proto}|{name}|{sname}|{seed}"
                    if key in state["cells"]:
                        continue
                    t0 = time.perf_counter()
                    if proto == "loso":
                        others = [j for j in range(len(subs)) if j != i]
                        rng = np.random.default_rng(1000 * seed + i)
                        vj = int(rng.choice(others))
                        tj = [j for j in others if j != vj]
                        Xtr = np.concatenate([subs[j][1] for j in tj])
                        ytr = np.concatenate([subs[j][2] for j in tj])
                        acc, ran, bva = train_eval(
                            Model, Xtr, ytr, subs[vj][1], subs[vj][2], X, y,
                            device, args.epochs, seed=seed)
                    else:
                        accs = []
                        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
                        for tr, te in skf.split(X, y):
                            tr2i, vai = stratified_val_split(y[tr], seed=seed)
                            a, ran, bva = train_eval(
                                Model, X[tr][tr2i], y[tr][tr2i], X[tr][vai], y[tr][vai],
                                X[te], y[te], device, args.epochs, seed=seed)
                            accs.append(a)
                        acc = float(np.mean(accs))
                    dt = time.perf_counter() - t0
                    state["cells"][key] = {"acc": acc, "seconds": dt}
                    save_ckpt(ckpt_path, state)
                    log(f"  {proto:<7}{name:<18}{sname}  seed {seed}  "
                        f"{acc * 100:6.2f}%   {dt:5.1f}s")

    log("")
    log(f"all cells complete in {(time.perf_counter() - t_start) / 60:.1f} min")

    # ---- aggregate -------------------------------------------------------------
    cells = state["cells"]
    log("")
    log("=== summary: mean over subjects, then mean +/- sd over seeds ===")
    log(f"{'protocol':<9}{'architecture':<19}{'mean':>8}{'sd(seed)':>10}"
        f"{'min':>8}{'max':>8}")
    table = {}
    for proto in protocols:
        for name in archs:
            per_seed = []
            for seed in range(args.seeds):
                vals = [cells[k]["acc"] for k in cells
                        if k.startswith(f"{proto}|{name}|") and k.endswith(f"|{seed}")]
                if vals:
                    per_seed.append(float(np.mean(vals)))
            if not per_seed:
                continue
            a = np.array(per_seed) * 100
            table[(proto, name)] = a
            # sd across seeds is undefined with a single seed; say so rather than nan
            sd = f"{a.std(ddof=1):>10.2f}" if a.size > 1 else f"{'n/a':>10}"
            log(f"{proto:<9}{name:<19}{a.mean():>8.2f}{sd}"
                f"{a.min():>8.2f}{a.max():>8.2f}")
    log("")
    log(f"chance = {100 / N_CLASSES:.2f}%")
    log("sd(seed) is the spread of the subject-mean across seeds. An architecture")
    log("ranking is only meaningful where the gap between architectures exceeds it.")

    # ---- figure ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, len(protocols), figsize=(6 * len(protocols), 5),
                           squeeze=False)
    for k, proto in enumerate(protocols):
        a = ax[0][k]
        names = [n for n in archs if (proto, n) in table]
        means = [table[(proto, n)].mean() for n in names]
        errs = [table[(proto, n)].std(ddof=1) if table[(proto, n)].size > 1 else 0.0
                for n in names]
        a.bar(range(len(names)), means, yerr=errs, capsize=4, color="tab:blue")
        a.axhline(100 / N_CLASSES, color="k", ls=":", lw=1, label="chance")
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        a.set_ylabel("accuracy (%)")
        a.set_title(f"{proto} ({args.seeds} seeds, error bars = sd across seeds)")
        a.legend(fontsize=8)
    fig.tight_layout()
    f = res(".png")
    fig.savefig(f, dpi=130)
    plt.close(fig)
    log(f"\nfigure -> {layout.rel(f)}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
