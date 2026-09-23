"""
The calibration curve: how many labelled trials from a new person does a decoder need?

Why this exists
---------------
Leave-one-subject-out answers "does a decoder trained on other people work on a new
one, with no labelled data from them?" (44.2%, three-class, correct trials).
Within-subject CV answers "how well does a decoder trained only on this person work,
given ~70 of their trials?" (57-59%). A BCI builder needs the curve between those two
points: accuracy as a function of k, the number of labelled calibration trials
collected from the new user, with and without the other subjects' data behind it.
Nobody has published that curve for this dataset.

Design
------
For every held-out subject and every draw (20 by default):

  * a stratified TEST set of 6 trials per class is fixed first and never touched;
  * the remaining trials form a calibration pool, shuffled once per class;
  * calibration sets are NESTED prefixes of that pool, m per class, so k = 3m and
    every k in a draw is scored on the same test trials. Differences between k are
    then differences in calibration data, not in test sets.

A subject contributes at a given k only if every class has m trials left in its pool.
sub-25 (44 correct trials, 9 in its smallest class) drops out early; the log reports
the subject count at every k, and the headline curve is restricted to the subjects
that support every k up to K_COMPLETE so that points on it are comparable.

Methods, all on decode.py's features (6 time bins x 56 channels), L2 logistic C = 0.1
(the best LOSO model) unless stated:

  within_lr        calibration trials only, logistic
  within_lda       calibration trials only, shrinkage LDA (the small-sample standard)
  pooled           other subjects + calibration trials, equal per-trial weight
  pooled_bal       as pooled, but the calibration trials carry the same total weight
                   as all source trials together (pre-specified, not tuned)
  pooled_calnorm   as pooled_bal, but the new subject is z-scored with statistics from
                   its k calibration trials only. The other methods z-score each subject
                   on all its own trials (label-free, as decode.py's LOSO does), which
                   uses the unlabelled test trials; this variant does not, and is what an
                   online BCI could actually do.
  zero_nonorm      k = 0 only: no per-subject normalisation of the new subject at all
                   (source scaler applied), to measure how much of the zero-shot LOSO
                   result the label-free per-subject z-score is carrying.

Deep arm (--deep, GPU): HemoNet pretrained on the other eleven subjects with early
stopping on a stratified source validation split, then every parameter fine-tuned on
the k calibration trials for a fixed, pre-specified schedule (no validation data
exists at small k), BatchNorm statistics frozen at their source values. Same draws,
same test sets.

Checkpoints: one per held-out subject for the classical arm, one per (subject, seed) for
the deep arm, written atomically. A power cut costs at most one subject.

Usage:  python calibration_curve.py [--data ../../derived/ds004830/v2] [--draws 20] [--jobs 12]
        python calibration_curve.py --deep [--seeds 5]
        python calibration_curve.py --report          # rebuild log/figure from checkpoints
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import decode
import layout

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 6                                  # test trials per class, per draw
M_GRID = [0, 2, 4, 7, 10, 13, 17, 20]       # calibration trials per class; k = 3m
K_COMPLETE = 30                             # headline curve: subjects supporting k <= 30
CLASSICAL = ["within_lr", "within_lda", "pooled", "pooled_bal", "pooled_calnorm",
             "zero_nonorm"]
DEEP = ["deep_finetune"]
NC = 3                                      # classes; set from --task in main()
DATASET = "ds004830"                         # set from --data in main()


class Log:
    def __init__(self, path, mode="a"):
        self.fh = open(path, mode, encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def draw_split(y, subj, draw):
    """Fixed test set and a nested calibration order, deterministic in (subject, draw)
    so the classical and deep arms see identical splits."""
    rng = np.random.default_rng([subj, draw, 99])
    test, pool = [], {}
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        test.extend(idx[:N_TEST].tolist())
        pool[c] = idx[N_TEST:]
    calib = {}
    for m in M_GRID:
        if all(len(p) >= m for p in pool.values()):
            calib[m] = np.concatenate([p[:m] for p in pool.values()]).astype(int)
    return np.array(test), calib


def zscore(F, stats_from=None):
    ref = F if stats_from is None else stats_from
    return (F - ref.mean(0)) / (ref.std(0) + 1e-9)


LR = lambda: LogisticRegression(max_iter=2000, C=0.1)


# ------------------------------------------------------------------ classical arm

def classical_subject(subs, i, draws):
    """All draws and all k for held-out subject i."""
    Z = [zscore(F) for _, F, _ in subs]
    ys = [y for _, _, y in subs]
    Xs = np.concatenate([Z[j] for j in range(len(subs)) if j != i])
    ys_src = np.concatenate([ys[j] for j in range(len(subs)) if j != i])
    Xraw_src = np.concatenate([subs[j][1] for j in range(len(subs)) if j != i])
    Ft, yt = subs[i][1], ys[i]
    Zt = Z[i]

    base = LR().fit(Xs, ys_src)                        # k = 0 source model
    sc = StandardScaler().fit(Xraw_src)
    base_raw = LR().fit(sc.transform(Xraw_src), ys_src)

    out = {m: {k: [] for k in CLASSICAL} for m in M_GRID}
    for d in range(draws):
        te, calib = draw_split(yt, i, d)
        for m in M_GRID:
            r = out[m]
            if m not in calib:
                for k in CLASSICAL:
                    r[k].append(None)
                continue
            ca = calib[m]
            acc = lambda model, X: float((model.predict(X[te]) == yt[te]).mean())
            if m == 0:
                a0 = acc(base, Zt)
                r["pooled"].append(a0)
                r["pooled_bal"].append(a0)
                r["pooled_calnorm"].append(None)
                r["within_lr"].append(None)
                r["within_lda"].append(None)
                r["zero_nonorm"].append(float((base_raw.predict(sc.transform(Ft[te]))
                                               == yt[te]).mean()))
                continue
            r["zero_nonorm"].append(None)
            r["within_lr"].append(acc(LR().fit(Zt[ca], yt[ca]), Zt))
            r["within_lda"].append(acc(LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto").fit(Zt[ca], yt[ca]), Zt))
            X = np.concatenate([Xs, Zt[ca]])
            Y = np.concatenate([ys_src, yt[ca]])
            r["pooled"].append(acc(LR().fit(X, Y), Zt))
            w = np.concatenate([np.ones(len(ys_src)),
                                np.full(len(ca), len(ys_src) / len(ca))])
            r["pooled_bal"].append(acc(LR().fit(X, Y, sample_weight=w), Zt))
            Zc = zscore(Ft, stats_from=Ft[ca])
            Xc = np.concatenate([Xs, Zc[ca]])
            r["pooled_calnorm"].append(acc(LR().fit(Xc, Y, sample_weight=w), Zc))
    return {str(m): v for m, v in out.items()}


# ----------------------------------------------------------------------- deep arm

def deep_subject(subs_deep, i, seed, draws, device, epochs_ft=25, lr_ft=3e-4):
    import torch
    import torch.nn as nn
    from deep_models import ARCHS
    from deep_models_gpu import train_eval, stratified_val_split

    Model = dict(ARCHS)["HemoNet"] if not isinstance(ARCHS, dict) else ARCHS["HemoNet"]
    Xs = np.concatenate([subs_deep[j][1] for j in range(len(subs_deep)) if j != i])
    ys = np.concatenate([subs_deep[j][2] for j in range(len(subs_deep)) if j != i])
    Xt, yt = subs_deep[i][1], subs_deep[i][2]
    tr, va = stratified_val_split(ys, seed=seed)

    # pretrain: train_eval returns only scores, so replicate its loop to keep the net
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = Model(Xs.shape[1], NC).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = np.bincount(ys[tr], minlength=NC).astype(float)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (NC * cnt),
                                                    dtype=torch.float32, device=device))
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    Xtr, ytr, Xva, yva = T(Xs[tr]), T(ys[tr], torch.long), T(Xs[va]), T(ys[va], torch.long)
    best, state, bad = -1, None, 0
    for ep in range(100):
        net.train()
        perm = torch.randperm(len(Xtr), device=device)
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            lossf(net(Xtr[idx]), ytr[idx]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = (net(Xva).argmax(1) == yva).float().mean().item()
        if v > best:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    Xt_t = T(Xt)

    def score(model, te):
        model.eval()
        with torch.no_grad():
            return float((model(Xt_t[te]).argmax(1).cpu().numpy() == yt[te]).mean())

    res = {str(m): [] for m in M_GRID}
    for d in range(draws):
        te, calib = draw_split(yt, i, d)
        for m in M_GRID:
            if m not in calib:
                res[str(m)].append(None)
                continue
            net.load_state_dict(state)
            if m > 0:
                ca = calib[m]
                torch.manual_seed(seed * 1000 + d)
                opt = torch.optim.AdamW(net.parameters(), lr=lr_ft, weight_decay=1e-4)
                lf = nn.CrossEntropyLoss()
                xb, yb = Xt_t[ca], T(yt[ca], torch.long)
                for ep in range(epochs_ft):
                    net.train()
                    for mod in net.modules():            # freeze BatchNorm statistics
                        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                            mod.eval()
                    for b in range(0, len(ca), 32):
                        sl = torch.randperm(len(ca), device=device)[:32] if len(ca) > 32 \
                            else torch.arange(len(ca), device=device)
                        opt.zero_grad(set_to_none=True)
                        lf(net(xb[sl]), yb[sl]).backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()
            res[str(m)].append(score(net, te))
    return res, best


def load_deep(data_dir, task="3class"):
    """deep_models_gpu.load, restricted to behaviourally correct trials (and to the two
    lateral classes for task 'lateral') so the deep arm sees exactly the trials the
    classical arm does."""
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        X = z["X"].transpose(0, 2, 1)
        X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True)
                                                       + 1e-9)
        ok = z["correct"].astype(bool)
        if task == "lateral":
            ok &= np.isin(z["y"].astype(int), (1, 2))
        out.append((str(z["subject"]), X[ok].astype(np.float32),
                    z["y"].astype(int)[ok] - 1))
    return out


# ------------------------------------------------------------------------- report

def merge_deep(ck, names, dk_path):
    """Fold the deep checkpoint into ck IN MEMORY only (mean over seeds per draw). The
    two arms keep separate files so they can run concurrently without clobbering."""
    if not os.path.exists(dk_path):
        return
    dk = json.load(open(dk_path))
    for n in names:
        seeds = sorted(int(k.split("|")[1]) for k in dk if k.split("|")[0] == n)
        for m in M_GRID:
            per = [dk[f"{n}|{s}"]["res"][str(m)] for s in seeds]
            if not per:
                continue
            merged = [None if any(p[d] is None for p in per)
                      else float(np.mean([p[d] for p in per])) for d in range(len(per[0]))]
            ck.setdefault(n, {}).setdefault(str(m), {})["deep_finetune"] = merged


def boot_ci(v, B=5000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def report(ck, names, log, fig_path, draws):
    from scipy import stats
    methods = [m for m in CLASSICAL + DEEP if any(m in ck.get(n, {}).get("0", {})
                                                  for n in names)]
    # per subject, per method, per m: mean over draws (None where unsupported)
    tab = {}
    for meth in methods:
        tab[meth] = {}
        for m in M_GRID:
            row = []
            for n in names:
                vals = ck.get(n, {}).get(str(m), {}).get(meth)
                vals = [v for v in (vals or []) if v is not None]
                row.append(float(np.mean(vals)) if vals else np.nan)
            tab[meth][m] = np.array(row)

    complete = np.ones(len(names), bool)
    for m in M_GRID:
        if NC * m <= K_COMPLETE:
            complete &= ~np.isnan(tab["pooled"][m]) if m else complete
            complete &= ~np.isnan(tab["within_lr"][m]) if m else complete
    log(f"\nheadline curve: subjects supporting every k <= {K_COMPLETE}: "
        f"{int(complete.sum())}/{len(names)}  "
        f"(excluded: {', '.join(n for n, c in zip(names, complete) if not c) or 'none'})")

    log(f"\n{'k':>4} {'n':>3}  " + "".join(f"{m:>22}" for m in methods))
    rows = []
    for m in M_GRID:
        k = NC * m
        sel = complete if k <= K_COMPLETE else ~np.isnan(tab["pooled"][m])
        cells = []
        for meth in methods:
            v = tab[meth][m][sel]
            v = v[~np.isnan(v)]
            if len(v) < 3:
                cells.append(f"{'--':>22}")
                continue
            lo, hi = boot_ci(v)
            cells.append(f"{v.mean() * 100:8.1f} [{lo * 100:4.1f},{hi * 100:5.1f}]")
            rows.append({"k": k, "method": meth, "n": int(len(v)), "mean": v.mean(),
                         "lo": lo, "hi": hi})
        log(f"{k:>4} {int(sel.sum()):>3}  " + "".join(cells)
            + ("" if k <= K_COMPLETE else "   <- all subjects still supporting this k"))

    log("\npaired comparisons on the headline subjects (Wilcoxon, two-sided):")
    for m in M_GRID[1:]:
        if NC * m > K_COMPLETE:
            break
        a, b = tab["pooled_bal"][m][complete], tab["within_lr"][m][complete]
        p = stats.wilcoxon(a, b).pvalue
        c = tab["pooled_calnorm"][m][complete]
        p2 = stats.wilcoxon(c, tab["pooled_bal"][m][complete]).pvalue
        # calnorm (online-realisable) against the subject-only decoder; within_lr still
        # z-scores on all its own trials, so this comparison is tilted against pooling
        p3 = stats.wilcoxon(c, b).pvalue
        log(f"  k={NC * m:>3}: pooled_bal - within_lr = {(a - b).mean() * 100:+5.1f} pp "
            f"(p = {p:.4f});  calnorm - pooled_bal = {(c - a).mean() * 100:+5.1f} pp "
            f"(p = {p2:.4f});  calnorm - within_lr = {(c - b).mean() * 100:+5.1f} pp "
            f"(p = {p3:.4f})")
    if "zero_nonorm" in methods:
        a, b = tab["pooled"][0], tab["zero_nonorm"][0]
        log(f"  k=  0: per-subject z-score vs none: {(a - b).mean() * 100:+5.1f} pp "
            f"(p = {stats.wilcoxon(a, b).pvalue:.4f}), zero-shot without it "
            f"{np.nanmean(b) * 100:.1f}%")

    log(f"\nper-subject, pooled_bal (mean over {draws} draws):")
    log(f"  {'subject':<9}" + "".join(f"{'k=' + str(NC * m):>8}" for m in M_GRID))
    for s, n in enumerate(names):
        log(f"  {n:<9}" + "".join(
            f"{tab['pooled_bal'][m][s] * 100:8.1f}" if not np.isnan(tab['pooled_bal'][m][s])
            else f"{'--':>8}" for m in M_GRID))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"within_lr": "#999999", "within_lda": "#555555", "pooled": "#8fb3de",
           "pooled_bal": "#1f5fa8", "pooled_calnorm": "#2a9d5c",
           "deep_finetune": "#c0502a"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for meth in methods:
        if meth == "zero_nonorm":
            continue
        pts = sorted((r["k"], r["mean"], r["lo"], r["hi"]) for r in rows
                     if r["method"] == meth and r["k"] <= K_COMPLETE)
        if not pts:
            continue
        k, mu, lo, hi = map(np.array, zip(*pts))
        ax.plot(k, mu * 100, "o-", color=col[meth], label=meth, ms=4)
        ax.fill_between(k, lo * 100, hi * 100, color=col[meth], alpha=0.12)
    ax.axhline(100 / NC, color="k", ls="--", lw=1)
    ax.text(K_COMPLETE, 100 / NC + 0.5, "chance", ha="right", fontsize=8)
    ax.set_xlabel("k = labelled calibration trials from the new subject")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"{DATASET} calibration curve, {NC}-class, correct trials, "
                 f"{int(complete.sum())} subjects, {draws} draws; 95% subject bootstrap",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=layout.derived("ds004830", "v2"))
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--deep", action="store_true", help="run the GPU fine-tuning arm")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--report", action="store_true", help="only rebuild log and figure")
    ap.add_argument("--task", default="3class", choices=("3class", "lateral"),
                    help="lateral = left vs right only (2 classes), e.g. for ds007738")
    args = ap.parse_args()
    global NC, DATASET
    NC = 3 if args.task == "3class" else 2
    DATASET = layout.split(args.data)[0]

    def res(ext, part=""):
        return layout.result(args.data, "calibration_curve", ext, args.task, part=part)

    ck_path = res(".json", "checkpoint")
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else {}
    cfg = {"draws": args.draws, "N_TEST": N_TEST, "M_GRID": M_GRID}
    if ck.get("_config", cfg) != cfg:
        raise SystemExit(f"checkpoint {ck_path} was written under {ck['_config']}, "
                         f"now {cfg}; delete it or match the settings")
    ck["_config"] = cfg

    subs = decode.load(args.data, args.task, correct_only=True)
    names = [s[0] for s in subs]
    log = Log(res(".log"))
    log(f"\n##### calibration_curve.py  {time.strftime('%Y-%m-%d %H:%M')}  "
        f"data={args.data}  draws={args.draws}  test={N_TEST}/class  "
        f"k grid={[NC * m for m in M_GRID]}")

    if not args.report and not args.deep:
        todo = [i for i, n in enumerate(names) if "pooled" not in ck.get(n, {}).get("0", {})]
        log(f"classical arm: {len(names) - len(todo)} subjects checkpointed, "
            f"{len(todo)} to run on {args.jobs} CPU workers")
        t0 = time.time()
        for i, res in zip(todo, Parallel(n_jobs=args.jobs, return_as="generator")(
                delayed(classical_subject)(subs, i, args.draws) for i in todo)):
            ent = ck.setdefault(names[i], {})
            for m, d in res.items():
                ent.setdefault(m, {}).update(d)
            save_json(ck_path, ck)
            log(f"  {names[i]} done ({time.time() - t0:.0f} s)")

    if args.deep:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"deep arm: HemoNet pretrain on N-1, fine-tune on k, {args.seeds} seeds, "
            f"device {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")
        sd = load_deep(args.data, args.task)
        dk_path = res(".json", "deep_checkpoint")
        dk = json.load(open(dk_path)) if os.path.exists(dk_path) else {}
        t0 = time.time()
        for i, n in enumerate(names):
            for s in range(args.seeds):
                key = f"{n}|{s}"
                if key in dk:
                    continue
                res, va = deep_subject(sd, i, s, args.draws, dev)
                dk[key] = {"res": res, "val": va}
                save_json(dk_path, dk)
                log(f"  {n} seed {s}: source val {va * 100:.1f}%  zero-shot "
                    f"{np.mean(res['0']) * 100:.1f}%  k=30 "
                    f"{np.mean([v for v in res['10'] if v is not None] or [np.nan]) * 100:.1f}%"
                    f"  ({time.time() - t0:.0f} s)")

    merge_deep(ck, names, res(".json", "deep_checkpoint"))
    rows = report(ck, names, log, res(".png"), args.draws)
    save_json(res(".json", "summary"), rows)
    log(f"wrote {layout.rel(res('.log'))}, {layout.rel(res('.png'))}, "
        f"{layout.rel(res('.json', 'summary'))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
