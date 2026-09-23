"""
Seven deep architectures, evaluated under the same honest protocols as decode.py.

HemoNet, CNN+BiLSTM, Transformer, CNN+TemporalAttention, TCN, InceptionTime and
GRU+Attention span the designs this literature applies to fNIRS time series. The
question here is not which of them wins; it is whether any of them beats shrinkage
LDA once the evaluation is sound, at a sample size of twelve participants.

Every epoch is cut at a real stimulus onset and carries its own trial's cued
location, and under loso the test set never shares a subject with the training set.

Two protocols, as in decode.py:
  within  5-fold inside each subject; comparable to Ning et al.'s ~45%
  loso    leave-one-subject-out; tests transfer to a new person

Usage:  python deep_models.py [--protocol loso|within|both] [--epochs 60]
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

import paths

DATA = paths.data_dir()
torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# architectures.  input is (batch, channels, time) with channels = 56, time = 140
# --------------------------------------------------------------------------- #

class HemoNet(nn.Module):
    """Three 1-D convolution blocks with global average pooling."""

    def __init__(self, n_ch, n_cls):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(n_ch, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5))
        self.c = nn.Linear(64, n_cls)

    def forward(self, x):
        return self.c(self.f(x))


class CNNBiLSTM(nn.Module):
    def __init__(self, n_ch, n_cls):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(n_ch, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2))
        self.r = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.o = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, n_cls))

    def forward(self, x):
        h, _ = self.r(self.c(x).transpose(1, 2))
        return self.o(h[:, -1])


class TransformerNet(nn.Module):
    def __init__(self, n_ch, n_cls, d=64, heads=4, layers=2):
        super().__init__()
        self.p = nn.Conv1d(n_ch, d, 1)
        self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 128, 0.2, batch_first=True,
                                         norm_first=True)
        self.t = nn.TransformerEncoder(enc, layers)
        self.o = nn.Sequential(nn.Dropout(0.5), nn.Linear(d, n_cls))

    def forward(self, x):
        h = self.p(x).transpose(1, 2)
        h = h + self.pos[:, :h.shape[1]]
        return self.o(self.t(h).mean(1))


class CNNAttention(nn.Module):
    """Convolutional front end with additive attention pooling over time."""

    def __init__(self, n_ch, n_cls):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(n_ch, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.a = nn.Linear(64, 1)
        self.o = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, n_cls))

    def forward(self, x):
        h = self.c(x).transpose(1, 2)
        w = torch.softmax(self.a(h), dim=1)
        return self.o((h * w).sum(1))


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        pad = (k - 1) * d
        self.pad = pad
        self.c1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=d)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=d)
        self.n1, self.n2 = nn.BatchNorm1d(cout), nn.BatchNorm1d(cout)
        self.r = nn.ReLU()
        self.dp = nn.Dropout(0.2)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.dp(self.r(self.n1(self.c1(x)[:, :, :-self.pad])))
        h = self.dp(self.r(self.n2(self.c2(h)[:, :, :-self.pad])))
        return self.r(h + self.skip(x))


class TCN(nn.Module):
    def __init__(self, n_ch, n_cls):
        super().__init__()
        self.f = nn.Sequential(TCNBlock(n_ch, 64, 3, 1), TCNBlock(64, 64, 3, 2),
                               TCNBlock(64, 64, 3, 4), TCNBlock(64, 64, 3, 8))
        self.o = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.o(self.f(x)[:, :, -1])


class InceptionModule(nn.Module):
    def __init__(self, cin, nf=32):
        super().__init__()
        self.bottle = nn.Conv1d(cin, nf, 1, bias=False) if cin > 1 else nn.Identity()
        cb = nf if cin > 1 else cin
        self.convs = nn.ModuleList(
            [nn.Conv1d(cb, nf, k, padding=k // 2, bias=False) for k in (39, 19, 9)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * nf)
        self.r = nn.ReLU()

    def forward(self, x):
        b = self.bottle(x)
        h = [c(b) for c in self.convs] + [self.pool(x)]
        n = min(t.shape[-1] for t in h)
        return self.r(self.bn(torch.cat([t[:, :, :n] for t in h], 1)))


class InceptionTime(nn.Module):
    """Six inception modules with residual connections every third, then GAP."""

    def __init__(self, n_ch, n_cls, nf=32, depth=6):
        super().__init__()
        self.mods = nn.ModuleList()
        self.res = nn.ModuleList()
        cin = n_ch
        for i in range(depth):
            self.mods.append(InceptionModule(cin, nf))
            cin = 4 * nf
            if i % 3 == 2:
                self.res.append(nn.Sequential(
                    nn.Conv1d(n_ch if i == 2 else 4 * nf, 4 * nf, 1, bias=False),
                    nn.BatchNorm1d(4 * nf)))
        self.o = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                               nn.Dropout(0.5), nn.Linear(4 * nf, n_cls))

    def forward(self, x):
        h, skip, k = x, x, 0
        for i, m in enumerate(self.mods):
            h = m(h)
            if i % 3 == 2:
                s = self.res[k](skip)
                n = min(h.shape[-1], s.shape[-1])
                h = torch.relu(h[:, :, :n] + s[:, :, :n])
                skip, k = h, k + 1
        return self.o(h)


class GRUAttention(nn.Module):
    def __init__(self, n_ch, n_cls):
        super().__init__()
        self.r = nn.GRU(n_ch, 64, batch_first=True, bidirectional=True)
        self.a = nn.Linear(128, 1)
        self.o = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, n_cls))

    def forward(self, x):
        h, _ = self.r(x.transpose(1, 2))
        w = torch.softmax(self.a(h), dim=1)
        return self.o((h * w).sum(1))


ARCHS = {
    "HemoNet": HemoNet,
    "CNN+BiLSTM": CNNBiLSTM,
    "Transformer": TransformerNet,
    "CNN+TemporalAttn": CNNAttention,
    "TCN": TCN,
    "InceptionTime": InceptionTime,
    "GRU+Attention": GRUAttention,
}


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def train_eval(Model, Xtr, ytr, Xva, yva, Xte, yte, n_cls, epochs, lr=1e-3,
               batch=64, patience=10, seed=0):
    """Train with early stopping on a held-out validation set, then score the test set."""
    torch.manual_seed(seed)
    net = Model(Xtr.shape[1], n_cls)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    cnt = np.bincount(ytr, minlength=n_cls).astype(float)
    w = torch.tensor(cnt.sum() / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            lossf(net(Xtr_t[idx]), ytr_t[idx]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            va = (net(Xva_t).argmax(1) == yva_t).float().mean().item()
        if va > best:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(Xte_t).argmax(1).numpy()
    return float((pred == yte).mean())


def load(data_dir):
    subs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        X = z["X"].transpose(0, 2, 1)                    # (trials, channels, time)
        m, s = X.mean(axis=(0, 2), keepdims=True), X.std(axis=(0, 2), keepdims=True)
        X = (X - m) / (s + 1e-9)                          # per-subject z-score
        y = z["y"].astype(int) - 1                        # classes 0,1,2
        subs.append((str(z["subject"]), X.astype(np.float32), y))
    return subs


def run_loso(subs, name, Model, epochs):
    accs = []
    for i in range(len(subs)):
        tr = [j for j in range(len(subs)) if j != i]
        va = tr[i % len(tr)]                              # one held-out training subject
        tr = [j for j in tr if j != va]
        Xtr = np.concatenate([subs[j][1] for j in tr])
        ytr = np.concatenate([subs[j][2] for j in tr])
        acc = train_eval(Model, Xtr, ytr, subs[va][1], subs[va][2],
                         subs[i][1], subs[i][2], 3, epochs, seed=i)
        accs.append(acc)
        print(f"    {name:<18} fold {subs[i][0]}  {acc * 100:5.2f}%", flush=True)
    return np.array(accs)


def run_within(subs, name, Model, epochs):
    from sklearn.model_selection import StratifiedKFold
    out = []
    for si, (sname, X, y) in enumerate(subs):
        accs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            n_va = max(6, int(0.2 * len(tr)))
            va, tr2 = tr[:n_va], tr[n_va:]
            accs.append(train_eval(Model, X[tr2], y[tr2], X[va], y[va],
                                   X[te], y[te], 3, epochs, seed=si))
        out.append(float(np.mean(accs)))
        print(f"    {name:<18} {sname}  {out[-1] * 100:5.2f}%", flush=True)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--protocol", default="both", choices=("loso", "within", "both"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="deep_results.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    subs = load(args.data)
    print(f"{len(subs)} subjects, {sum(len(s[2]) for s in subs)} trials, "
          f"input {subs[0][1].shape[1]} channels x {subs[0][1].shape[2]} samples")
    print(f"chance = 33.3%   max {args.epochs} epochs, early stopping on a "
          f"held-out subject\n")

    res = {}
    protocols = ["loso", "within"] if args.protocol == "both" else [args.protocol]
    for proto in protocols:
        print(f"=== {proto} ===")
        res[proto] = {}
        for name, Model in ARCHS.items():
            t0 = time.time()
            accs = (run_loso if proto == "loso" else run_within)(subs, name, Model,
                                                                args.epochs)
            res[proto][name] = accs.tolist()
            print(f"  {name:<18} mean {accs.mean() * 100:5.2f}%  sd {accs.std() * 100:4.2f}"
                  f"  ({time.time() - t0:.0f}s)\n", flush=True)

    print("\n=== summary ===")
    print(f"  {'architecture':<20}" + "".join(f"{p:>10}" for p in protocols))
    for name in ARCHS:
        row = f"  {name:<20}"
        for p in protocols:
            row += f"{np.mean(res[p][name]) * 100:9.2f}%"
        print(row)
    print(f"  {'chance':<20}{33.33:9.2f}%")

    with open(args.out, "w") as f:
        json.dump({"subjects": [s[0] for s in subs], "results": res}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
