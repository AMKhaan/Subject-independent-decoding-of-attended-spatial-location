"""
Where and when is the decodable signal, and is it physiologically plausible?

Why this exists
---------------
A reviewer's first question about any fNIRS decoding number is whether the classifier
reads haemodynamics or something else (motion, systemic drift, a timing artefact). The
time-resolved analysis already answers WHEN: within-subject accuracy is at chance before
onset and peaks at +6.0 s (results/ds004830/time_resolved/ds004830_v2_time_resolved.log). This script answers WHERE
and WHAT, three ways, on the same features, trials and subjects as the headline
(behaviourally correct trials, derived/ds004830/v2, 28 channels x HbO/HbR x 6 time bins):

  A. Group effect maps. Per subject, the standardised difference (Cohen's d) between
     conditions for every channel and chromophore in the 4-8 s window, then a one-sample
     t-test of d across the 12 subjects, Benjamini-Hochberg FDR over the 56 features.
     Contrasts: Left - Right (lateral attention) and Centre - lateral. Plotted on the
     probe layout. A plausible map is spatially smooth, sits over FEF / IPS, and has
     HbR opposite in sign to HbO.

  B. Region-drop importance. Accuracy (logistic, the headline LOSO model; within-subject
     5 x 5 CV and LOSO) with all 28 channels, with each region REMOVED, and with each
     region ALONE. Paired Wilcoxon against all channels. If the signal were artefactual
     and global, every region alone would do as well as all of them.

  C. Activation patterns. Classifier weights are filters, not maps: a large weight can
     cancel noise rather than carry signal. Haufe et al. (2014) patterns A = Cov(X) W
     are the interpretable counterpart. Fitted on all subjects pooled (z-scored per
     subject, as in LOSO), shown as |pattern| per region x time bin x chromophore.

Writes results/ds004830/physiology/ds004830_v2_physiology.log and .json, and the matching
.png under plots/ (see layout.py).

Usage:  python physiology.py [--data ../../derived/ds004830/v2]
"""

import argparse
import csv
import glob
import json
import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import decode
import layout

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
ROIS = {"L-FEF": (1, 2, 3), "R-FEF": (4, 5, 6), "L-IPS": (9, 11), "R-IPS": (10, 12)}
PEAK = (4.0, 8.0)
LR = lambda: LogisticRegression(max_iter=2000, C=0.1)
NAMES = {1: "Right", 2: "Left", 3: "Centre"}


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def load(data_dir):
    subs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        ok = z["correct"].astype(bool)
        subs.append({"name": str(z["subject"]), "X": z["X"][ok], "y": z["y"].astype(int)[ok],
                     "tmin": float(z["tmin"]), "fs": float(z["fs"]),
                     "ch": np.asarray(z["channels"])})
    return subs


def midpoints(ch):
    """Template channel midpoints from the standard subject's optodes (all identical)."""
    f = sorted(glob.glob(os.path.join(REPO, "Dataset", "sub-12", "nirs", "*_optodes.tsv")))[0]
    pos = {r["name"]: (float(r["x"]), float(r["y"])) for r in
           csv.DictReader(open(f), delimiter="\t")}
    return np.array([[(pos[f"S{s}"][k] + pos[f"D{d}"][k]) / 2 for k in (0, 1)] for s, d in ch])


def fdr(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ------------------------------------------------------------------ A. effect maps

def effect_maps(subs, log):
    t = subs[0]["tmin"] + np.arange(subs[0]["X"].shape[1]) / subs[0]["fs"]
    win = (t >= PEAK[0]) & (t < PEAK[1])
    contrasts = {"Left - Right": ((2,), (1,)), "Centre - lateral": ((3,), (1, 2))}
    res = {}
    for name, (a, b) in contrasts.items():
        D = []
        for s in subs:
            v = s["X"][:, win, :].mean(axis=1)                  # trials x 56
            A, B = v[np.isin(s["y"], a)], v[np.isin(s["y"], b)]
            sd = np.sqrt((A.var(0, ddof=1) + B.var(0, ddof=1)) / 2) + 1e-12
            D.append((A.mean(0) - B.mean(0)) / sd)
        D = np.array(D)                                          # subjects x 56
        tt, p = stats.ttest_1samp(D, 0, axis=0)
        q = fdr(p)
        res[name] = {"d": D.mean(0).tolist(), "t": tt.tolist(), "p": p.tolist(),
                     "q": q.tolist()}
        n = D.shape[1] // 2
        log(f"  {name}: features with FDR q < .05: {int((q < .05).sum())}/56 "
            f"(HbO {int((q[:n] < .05).sum())}, HbR {int((q[n:] < .05).sum())});  "
            f"uncorrected p < .05: {int((p < .05).sum())}/56")
        r = stats.spearmanr(D.mean(0)[:n], D.mean(0)[n:])
        log(f"    HbO vs HbR group d across channels: Spearman rho {r[0]:+.3f} "
            f"(p = {r[1]:.4f}); a negative rho is the expected haemodynamic coupling")
        top = np.argsort(p)[:5]
        ch = subs[0]["ch"]
        for i in top:
            c = ch[i % n]
            roi = next(k for k, v in ROIS.items() if c[0] in v)
            log(f"    S{c[0]}-D{c[1]} {'HbO' if i < n else 'HbR'} ({roi}): mean d "
                f"{D.mean(0)[i]:+.2f}, t {tt[i]:+.2f}, p {p[i]:.4f}, q {q[i]:.3f}")
    return res


# -------------------------------------------------------------- B. region drop

def feats(s, keep):
    """decode.featurise on a channel subset (HbO and HbR of the kept channels)."""
    n = len(s["ch"])
    cols = np.concatenate([np.where(keep)[0], n + np.where(keep)[0]])
    return decode.featurise(s["X"][:, :, cols], s["tmin"], s["fs"])


def within_acc(subs, keep):
    """decode.within_subject's model exactly (StandardScaler + logistic), so the
    all-channel row reproduces the headline 56.93%."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = []
    for s in subs:
        F, y = feats(s, keep), s["y"]
        a = []
        for rep in range(5):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(F, y):
                m = make_pipeline(StandardScaler(), LR()).fit(F[tr], y[tr])
                a.append((m.predict(F[te]) == y[te]).mean())
        out.append(np.mean(a))
    return np.array(out)


def loso_acc(subs, keep):
    Z = [(lambda F: (F - F.mean(0)) / (F.std(0) + 1e-9))(feats(s, keep)) for s in subs]
    out = []
    for i in range(len(subs)):
        Xtr = np.concatenate([Z[j] for j in range(len(subs)) if j != i])
        ytr = np.concatenate([subs[j]["y"] for j in range(len(subs)) if j != i])
        out.append((LR().fit(Xtr, ytr).predict(Z[i]) == subs[i]["y"]).mean())
    return np.array(out)


def region_drop(subs, log):
    ch = subs[0]["ch"]
    full = np.ones(len(ch), bool)
    res = {}
    base = {"within": within_acc(subs, full), "loso": loso_acc(subs, full)}
    res["all"] = {k: v.tolist() for k, v in base.items()}
    log(f"  all 28 channels: within {base['within'].mean() * 100:.2f}%   "
        f"LOSO {base['loso'].mean() * 100:.2f}%")
    log(f"  {'region':<8}{'n':>3}  {'without: within':>17}{'LOSO':>9}   "
        f"{'alone: within':>15}{'LOSO':>9}   (Wilcoxon vs all)")
    for r, srcs in ROIS.items():
        m = np.isin(ch[:, 0], srcs)
        row = {}
        cells = []
        for tag, keep in (("without", ~m), ("alone", m)):
            for proto, fn in (("within", within_acc), ("loso", loso_acc)):
                a = fn(subs, keep)
                p = stats.wilcoxon(a, base[proto]).pvalue if np.any(a != base[proto]) else 1
                row[f"{tag}|{proto}"] = a.tolist()
                cells.append(f"{a.mean() * 100:6.1f} ({(a - base[proto]).mean() * 100:+5.1f}, "
                             f"p {p:.3f})")
        res[r] = row
        log(f"  {r:<8}{int(m.sum()):>3}  " + "  ".join(cells))
    return res


# ------------------------------------------------------------ C. Haufe patterns

def patterns(subs, log):
    Z, Y = [], []
    for s in subs:
        F = decode.featurise(s["X"], s["tmin"], s["fs"])
        Z.append((F - F.mean(0)) / (F.std(0) + 1e-9))
        Y.append(s["y"])
    X, y = np.concatenate(Z), np.concatenate(Y)
    m = LR().fit(X, y)
    W = m.coef_.T                                          # features x classes
    A = np.cov(X, rowvar=False) @ W                        # Haufe activation patterns
    nb = len(decode.BINS)
    nf = X.shape[1] // nb                                  # 56 = 28 x (HbO, HbR)
    P = np.abs(A).mean(axis=1).reshape(nb, nf)             # bins x features
    ch = subs[0]["ch"]
    n = len(ch)
    grid = {}
    for chrom, off in (("HbO", 0), ("HbR", n)):
        grid[chrom] = np.array([[P[b, off + np.where(np.isin(ch[:, 0], srcs))[0]].mean()
                                 for b in range(nb)] for srcs in ROIS.values()])
    log("  mean |Haufe pattern| by region (rows) and time bin (cols, s after onset):")
    log("  " + " " * 12 + "".join(f"{f'{a}-{b}':>8}" for a, b in decode.BINS))
    for chrom in ("HbO", "HbR"):
        for r, row in zip(ROIS, grid[chrom]):
            log(f"  {chrom} {r:<7}" + "".join(f"{v:8.3f}" for v in row))
    tb = P.mean(axis=1)
    log(f"  time profile (all features): peak bin {decode.BINS[int(tb.argmax())]} s; "
        + ", ".join(f"{v:.3f}" for v in tb))
    return {k: v.tolist() for k, v in grid.items()}, tb.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=layout.derived("ds004830", "v2"))
    args = ap.parse_args()
    res = lambda ext: layout.result(args.data, "physiology", ext)
    log = Log(res(".log"))
    subs = load(args.data)
    log(f"physiology -- {len(subs)} subjects, {sum(len(s['y']) for s in subs)} correct "
        f"trials, data {args.data}\n")
    log(f"A. group effect maps, {PEAK[0]:g}-{PEAK[1]:g} s, Cohen's d per subject, t-test "
        f"across subjects, BH-FDR over 56 features")
    A = effect_maps(subs, log)
    log("\nB. region-drop importance (logistic C = 0.1)")
    B = region_drop(subs, log)
    log("\nC. Haufe activation patterns, pooled model")
    C, tb = patterns(subs, log)
    json.dump({"effects": A, "region_drop": B, "patterns": C, "time_profile": tb},
              open(res(".json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    ch = subs[0]["ch"]
    mid = midpoints(ch)
    n = len(ch)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4)
    for col, (name, r) in enumerate(A.items()):
        for row, (chrom, off) in enumerate((("HbO", 0), ("HbR", n))):
            ax = fig.add_subplot(gs[row, col])
            tv = np.array(r["t"][off:off + n])
            q = np.array(r["q"][off:off + n])
            lim = max(3, np.abs(tv).max())
            sc = ax.scatter(mid[:, 0], mid[:, 1], c=tv, cmap="RdBu_r", s=140,
                            norm=TwoSlopeNorm(0, -lim, lim), edgecolors=np.where(
                                q < .05, "k", "none"), linewidths=1.8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name}, {chrom} ({PEAK[0]:g}-{PEAK[1]:g} s)\n"
                         f"group t; black ring = FDR q < .05", fontsize=8)
            fig.colorbar(sc, ax=ax, shrink=0.7)
    ax = fig.add_subplot(gs[0, 2:])
    regs = list(ROIS)
    x = np.arange(len(regs))
    for j, (tag, proto, c) in enumerate((("without", "within", "#8fb3de"),
                                         ("without", "loso", "#e0a58f"),
                                         ("alone", "within", "#1f5fa8"),
                                         ("alone", "loso", "#c0502a"))):
        v = [np.mean(B[r][f"{tag}|{proto}"]) * 100 for r in regs]
        ax.bar(x + (j - 1.5) * 0.2, v, 0.2, color=c, label=f"{tag} region, {proto}")
    for proto, ls in (("within", "-"), ("loso", "--")):
        ax.axhline(np.mean(B["all"][proto]) * 100, color="k", ls=ls, lw=1,
                   label=f"all 28 channels, {proto}")
    ax.axhline(100 / 3, color="grey", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(regs)
    ax.set_ylabel("3-class accuracy (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Region-drop importance", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center")
    for k, chrom in enumerate(("HbO", "HbR")):
        ax = fig.add_subplot(gs[1, 2 + k])
        im = ax.imshow(np.array(C[chrom]), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(regs)))
        ax.set_yticklabels(regs, fontsize=8)
        ax.set_xticks(range(len(decode.BINS)))
        ax.set_xticklabels([f"{a}-{b}" for a, b in decode.BINS], fontsize=7)
        ax.set_xlabel("s after onset")
        ax.set_title(f"|Haufe pattern|, {chrom}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Where and when the decodable signal is (3-class, correct trials, "
                 f"{len(subs)} subjects)", fontsize=11)
    fig.tight_layout()
    fig.savefig(res(".png"), dpi=150)
    log(f"\nwrote {layout.rel(res('.log'))}, {layout.rel(res('.json'))}, {layout.rel(res('.png'))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
