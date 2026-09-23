"""
Does zero-shot cross-subject decoding survive normalisation that never sees the test trials?

Why this exists
---------------
decode.py's leave-one-subject-out z-scores every subject, the held-out one included, on
all of that subject's trials. It uses no labels, but for the held-out subject it uses
the TEST trials' feature statistics, which an online decoder meeting a new user would
not have. calibration_curve.py showed the step is worth +6.8 pp at k = 0 (45.5% with it,
37.8% with no per-subject normalisation at all). A reviewer can therefore call the 44.2%
headline transductive. This script measures what is left when the held-out subject is
normalised only with data that would exist at decision time.

Schemes (the same scheme is applied to training subjects, so train and test match):

  transductive  z-score on all of the subject's trials (decode.py; reference only)
  prefix        z-score on the subject's first W trials of the session, chronologically,
                all trials regardless of correctness (unlabelled warm-up); fixed after
  causal        trial t is z-scored with the mean/sd of all of that subject's trials
                before t (running, W minimum); PRE-SPECIFIED PRIMARY scheme, the one an
                online BCI can run without any warm-up block
  none          no per-subject step; one scaler fitted on the pooled training subjects

Every scheme is scored on the SAME test trials: behaviourally correct trials after the
first W of the session (chronological). The first W are never scored, for any scheme, so
the comparison is not confounded by which trials are evaluated.

Classifiers: L2 logistic C = 0.1 (the headline LOSO model) and shrinkage LDA. The primary
scheme gets a label-permutation test (labels shuffled within subject) and a subject
bootstrap CI. A deep arm (--deep, GPU) runs HemoNet LOSO under transductive and causal
input normalisation, several seeds.

Checkpointed per (task, scheme, model) and per permutation chunk / deep seed, atomically.

Writes results/<dataset>/normalisation_check/<dataset>_<variant>_normalisation_check.log,
.json and _checkpoint.json, and the matching .png under plots/ (see layout.py).

Usage:  python normalisation_check.py [--data ../../derived/ds004830/v2] [--warmup 20] [--perms 200]
                                      [--deep --seeds 5]
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import decode
import layout

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMES = ["transductive", "prefix", "causal", "none"]
PRIMARY = "causal"
DATA = ""                                      # set in main() from --data
MODELS = {"Logistic (L2)": lambda: LogisticRegression(max_iter=2000, C=0.1),
          "LDA (Ledoit-Wolf)": lambda: LinearDiscriminantAnalysis(solver="lsqr",
                                                                   shrinkage="auto")}


class Log:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def res(ext, part=""):
    """Output path for the --data folder in use (see layout.py)."""
    return layout.result(DATA, "normalisation_check", ext, part=part)


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- data

def load(data_dir, task):
    """All trials in chronological order (normalisation may use incorrect trials, which
    are unlabelled data an online system would also have), plus a mask of the trials
    that count for this task (correct, and Left/Right only for 'lateral')."""
    subs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        order = np.argsort(z["onsets"])
        F = decode.featurise(z["X"], float(z["tmin"]), float(z["fs"]))[order]
        y = z["y"].astype(int)[order]
        use = z["correct"].astype(bool)[order]
        if task == "lateral":
            use &= np.isin(y, (1, 2))
        subs.append({"name": str(z["subject"]), "F": F, "y": y, "use": use,
                     "X": z["X"][order]})
    return subs


def normalise(F, use, scheme, W):
    """Per-subject z-score of feature rows F (chronological) under a scheme."""
    if scheme == "transductive":
        ref = F[use]                                   # decode.py: its scored trials
        return (F - ref.mean(0)) / (ref.std(0) + 1e-9)
    if scheme == "prefix":
        ref = F[:W]
        return (F - ref.mean(0)) / (ref.std(0) + 1e-9)
    if scheme == "causal":
        out = np.empty_like(F)
        for t in range(len(F)):
            ref = F[:max(t, W)] if t >= W else F[:W]   # first W use the warm-up block
            out[t] = (F[t] - ref.mean(0)) / (ref.std(0) + 1e-9)
        return out
    return F                                           # none


def loso(subs, scheme, W, make, y_override=None):
    """Per-subject accuracy on the common test set (correct trials after the first W)."""
    Z = [normalise(s["F"], s["use"], scheme, W) for s in subs]
    ys = y_override or [s["y"] for s in subs]
    accs = []
    for i in range(len(subs)):
        tr = [j for j in range(len(subs)) if j != i]
        Xtr = np.concatenate([Z[j][subs[j]["use"]] for j in tr])
        ytr = np.concatenate([ys[j][subs[j]["use"]] for j in tr])
        te = subs[i]["use"].copy()
        te[:W] = False
        Xte = Z[i][te]
        if scheme == "none":
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = make().fit(Xtr, ytr)
        accs.append(float((m.predict(Xte) == ys[i][te]).mean()))
    return accs


def one_perm(subs, W, k, make):
    g = np.random.default_rng([k, 4242])
    yp = []
    for s in subs:
        y = s["y"].copy()
        idx = np.where(s["use"])[0]
        y[idx] = g.permutation(y[idx])                 # shuffle labels of scored trials
        yp.append(y)
    return k, float(np.mean(loso(subs, PRIMARY, W, make, y_override=yp)))


def bca(v, B=10000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    z0 = stats.norm.ppf(np.clip((boot < v.mean()).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([np.delete(v, i).mean() for i in range(len(v))])
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6 * ((d ** 2).sum() ** 1.5) + 1e-300)
    zs = stats.norm.ppf([0.025, 0.975])
    return np.percentile(boot, stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs))) * 100)


# ------------------------------------------------------------------------- deep arm

def deep_loso(subs, scheme, W, seed, device, n_cls):
    import torch
    import torch.nn as nn
    from deep_models import ARCHS
    from deep_models_gpu import stratified_val_split

    def norm_x(X, use):
        # per channel, statistics over (trials, time), with the chosen scheme
        Xc = X.transpose(0, 2, 1).astype(np.float32)          # (trials, ch, time)
        if scheme == "transductive":
            ref = Xc[use]
            return (Xc - ref.mean((0, 2), keepdims=True)) / (ref.std((0, 2), keepdims=True) + 1e-9)
        out = np.empty_like(Xc)
        for t in range(len(Xc)):
            ref = Xc[:max(t, W)]
            out[t] = (Xc[t] - ref.mean((0, 2))[:, None]) / (ref.std((0, 2))[:, None] + 1e-9)
        return out

    lab = sorted(np.unique(np.concatenate([s["y"][s["use"]] for s in subs])))
    remap = {c: i for i, c in enumerate(lab)}
    Xs = [norm_x(s["X"], s["use"]) for s in subs]
    accs = []
    for i in range(len(subs)):
        tr = [j for j in range(len(subs)) if j != i]
        X = np.concatenate([Xs[j][subs[j]["use"]] for j in tr])
        y = np.array([remap[v] for j in tr for v in subs[j]["y"][subs[j]["use"]]])
        te = subs[i]["use"].copy()
        te[:W] = False
        Xte = Xs[i][te]
        yte = np.array([remap[v] for v in subs[i]["y"][te]])
        a, va = stratified_val_split(y, seed=seed + 100 * i)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        net = ARCHS["HemoNet"](X.shape[1], n_cls).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        cnt = np.bincount(y[a], minlength=n_cls).astype(float)
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (n_cls * cnt),
                                                        dtype=torch.float32, device=device))
        T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=device)
        Xa, ya, Xv, yv = T(X[a]), T(y[a], torch.long), T(X[va]), T(y[va], torch.long)
        best, state, bad = -1, None, 0
        for ep in range(60):
            net.train()
            perm = torch.randperm(len(Xa), device=device)
            for b in range(0, len(perm), 64):
                idx = perm[b:b + 64]
                if len(idx) < 2:
                    continue
                opt.zero_grad(set_to_none=True)
                lossf(net(Xa[idx]), ya[idx]).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                v = (net(Xv).argmax(1) == yv).float().mean().item()
            if v > best:
                best, bad = v, 0
                state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            p = net(T(Xte)).argmax(1).cpu().numpy()
        accs.append(float((p == yte).mean()))
    return accs


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=layout.derived("ds004830", "v2"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tasks", nargs="+", default=["3class", "lateral"],
                    choices=("3class", "lateral"))
    args = ap.parse_args()
    W = args.warmup
    # Outputs are named by dataset and variant so a second dataset never reads this
    # one's checkpoint (its keys, e.g. "lateral|...", would otherwise collide).
    global DATA
    DATA = args.data
    log = Log(res(".log"))
    ck_path = res(".json", "checkpoint")
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else {}
    if ck.get("_cfg", {"W": W}) != {"W": W}:
        raise SystemExit(f"checkpoint written with {ck['_cfg']}; delete it or match --warmup")
    ck["_cfg"] = {"W": W}
    log(f"\n##### normalisation_check.py  {time.strftime('%Y-%m-%d %H:%M')}  data={args.data}"
        f"  warm-up W={W} trials  primary={PRIMARY}  deep={args.deep}")

    for task in args.tasks:
        subs = load(args.data, task)
        chance = 1 / 3 if task == "3class" else 1 / 2
        n_te = sum(int(s["use"][W:].sum()) for s in subs)
        log(f"\n=== {task}  chance {chance * 100:.1f}%  test trials (correct, after the "
            f"first {W}): {n_te}, same for every scheme ===")
        for name, make in MODELS.items():
            for sch in SCHEMES:
                key = f"{task}|{name}|{sch}"
                if key not in ck:
                    ck[key] = loso(subs, sch, W, make)
                    save_json(ck_path, ck)
        log(f"  {'model':<20}" + "".join(f"{s:>24}" for s in SCHEMES))
        for name in MODELS:
            cells = []
            for sch in SCHEMES:
                a = np.array(ck[f"{task}|{name}|{sch}"])
                lo, hi = bca(a)
                cells.append(f"{a.mean() * 100:6.2f} [{lo * 100:4.1f},{hi * 100:5.1f}]")
            log(f"  {name:<20}" + "".join(f"{c:>24}" for c in cells))
        for name in MODELS:
            ref = np.array(ck[f"{task}|{name}|transductive"])
            for sch in SCHEMES[1:]:
                a = np.array(ck[f"{task}|{name}|{sch}"])
                p = stats.wilcoxon(a, ref).pvalue if np.any(a != ref) else 1.0
                log(f"  {name:<20} {sch:<8} - transductive = {(a - ref).mean() * 100:+6.2f} pp"
                    f"  (Wilcoxon p = {p:.4f})")

        # permutation test of the primary scheme, logistic
        if args.perms:
            pk = f"{task}|perm"
            done = ck.setdefault(pk, {})
            todo = [k for k in range(args.perms) if str(k) not in done]
            for s0 in range(0, len(todo), args.jobs):
                res = Parallel(n_jobs=args.jobs)(
                    delayed(one_perm)(subs, W, k, MODELS["Logistic (L2)"])
                    for k in todo[s0:s0 + args.jobs])
                for k, v in res:
                    done[str(k)] = v
                save_json(ck_path, ck)
                log(f"    permutations {len(done)}/{args.perms}")
            null = np.array([done[str(k)] for k in range(args.perms)])
            obs = np.mean(ck[f"{task}|Logistic (L2)|{PRIMARY}"])
            p = (1 + (null >= obs).sum()) / (1 + args.perms)
            log(f"  permutation, {PRIMARY} / logistic: observed {obs * 100:.2f}%, null mean "
                f"{null.mean() * 100:.2f}%, 95th {np.percentile(null, 95) * 100:.2f}%, "
                f"p = {p:.4f}" + ("  (floor)" if p <= 1 / (1 + args.perms) else ""))

        a = np.array(ck[f"{task}|Logistic (L2)|{PRIMARY}"])
        ps = []
        for s, acc in zip(subs, a):
            te = s["use"].copy()
            te[:W] = False
            n = int(te.sum())
            ps.append(stats.binomtest(int(round(acc * n)), n, chance, "greater").pvalue)
        order = np.argsort(ps)
        holm, run = np.empty(len(ps)), 0.0
        for r, i in enumerate(order):
            run = max(run, min(1.0, (len(ps) - r) * ps[i]))
            holm[i] = run
        log(f"  {PRIMARY} / logistic, per subject: "
            + ", ".join(f"{s['name']} {x * 100:.1f}%{'*' if h < .05 else ''}"
                        for s, x, h in zip(subs, a, holm))
            + f"   -> {int((holm < .05).sum())}/{len(holm)} significant after Holm")

        if args.deep:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_cls = 3 if task == "3class" else 2
            for sch in ("transductive", "causal"):
                for sd in range(args.seeds):
                    key = f"{task}|HemoNet|{sch}|{sd}"
                    if key in ck:
                        continue
                    t0 = time.time()
                    ck[key] = deep_loso(subs, sch, W, sd, dev, n_cls)
                    save_json(ck_path, ck)
                    log(f"    deep {task} {sch} seed {sd}: {np.mean(ck[key]) * 100:.2f}%"
                        f"  ({time.time() - t0:.0f} s, {dev})")
                m = [np.mean(ck[f"{task}|HemoNet|{sch}|{sd}"]) for sd in range(args.seeds)]
                log(f"  HemoNet {sch:<13} {np.mean(m) * 100:6.2f} +/- {np.std(m) * 100:.2f}"
                    f" (sd over {args.seeds} seeds)")

    # summary JSON and figure
    out = {k: v for k, v in ck.items() if not k.endswith("|perm") and k != "_cfg"}
    save_json(res(".json"), out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.tasks), figsize=(5.25 * len(args.tasks), 4.4), squeeze=False)
    axes = axes[0]
    for ax, task in zip(axes, args.tasks):
        chance = 1 / 3 if task == "3class" else 1 / 2
        a = [np.array(ck[f"{task}|Logistic (L2)|{s}"]) for s in SCHEMES]
        x = np.arange(len(SCHEMES))
        cols = ["#bbbbbb", "#8fb3de", "#1f5fa8", "#c0502a"]
        for i, v in enumerate(a):
            lo, hi = bca(v)
            ax.bar(i, v.mean() * 100, 0.6, color=cols[i])
            ax.errorbar(i, v.mean() * 100, [[v.mean() * 100 - lo * 100], [hi * 100 - v.mean() * 100]],
                        color="k", capsize=4)
        for j in range(len(a[0])):
            ax.plot(x, [v[j] * 100 for v in a], color="k", lw=0.4, alpha=0.25)
        deep = [k for k in ck if k.startswith(f"{task}|HemoNet|")]
        if deep:
            for sch, xi in (("transductive", 0), ("causal", 2)):
                m = [np.mean(ck[k]) * 100 for k in deep if f"|{sch}|" in k]
                if m:
                    ax.plot(xi + 0.38, np.mean(m), "D", color="#2a9d5c", ms=6,
                            label="HemoNet (mean over seeds)" if sch == "causal" else None)
        ax.axhline(chance * 100, color="k", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([s + ("\n(primary)" if s == PRIMARY else "") for s in SCHEMES],
                           fontsize=8)
        ax.set_ylabel("LOSO accuracy (%), logistic")
        ax.set_title(f"{task}: new-subject normalisation, same test trials\n"
                     f"(correct trials after the first {W}; bars 95% BCa)", fontsize=9)
        if deep:
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(res(".png"), dpi=150)
    log(f"wrote {layout.rel(res('.log'))}, {layout.rel(res('.png'))}, {layout.rel(res('.json'))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
