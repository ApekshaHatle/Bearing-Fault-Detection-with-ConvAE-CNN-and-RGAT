"""
Data loading for the CWRU bearing dataset (12 kHz drive-end, 10 classes, loads 0-3 HP).

Files live flat in one folder (default: data/). For class base-number B and load L (0-3 HP)
the file is  <B+L>.mat   e.g. Normal: 97 98 99 100, IR007: 105 106 107 108, ...

Base paper (Yuan et al., FSDM 2023) setup: 1024-sample windows -> 32x32 image, AWGN at a
chosen SNR, 70/10/20 split. The paper does not list its exact files; its figure mentions
load 0, so the default is --loads 0.
"""
import os
import numpy as np
import torch
from scipy.io import loadmat

FS = 12000
WIN = 1024
SIDE = 32  # 32 x 32 = 1024
ALL_LOADS = (0, 1, 2, 3)

# (name, CWRU file number at 0 HP, fault diameter in mils; 0 = healthy)
CLASSES = [
    ("Normal", 97, 0),
    ("IR007", 105, 7), ("B007", 118, 7), ("OR007", 130, 7),
    ("IR014", 169, 14), ("B014", 185, 14), ("OR014", 197, 14),
    ("IR021", 209, 21), ("B021", 222, 21), ("OR021", 234, 21),
]
CLASS_NAMES = [c[0] for c in CLASSES]
CLASS_SIZE = np.array([c[2] for c in CLASSES])


def mat_path(data_dir, base, load):
    return os.path.join(data_dir, f"{base + load}.mat")


def load_cwru(data_dir, loads=(0,)):
    """Returns {load: [signal for each of the 10 classes]}."""
    out = {}
    for L in loads:
        sigs = []
        for name, base, _ in CLASSES:
            path = mat_path(data_dir, base, L)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}  (class {name}, load {L} HP). Run check_data.py")
            m = loadmat(path)
            keys = [k for k in m if k.endswith("_DE_time")]
            if not keys:
                raise KeyError(f"No *_DE_time key in {path}")
            sigs.append(m[keys[0]].ravel().astype(np.float32))
        out[L] = sigs
    return out


def synthetic_signals(loads=(0,), n=60000):
    """FAKE bearing-like signals, only for checking that the code runs."""
    freq = {"IR": 162.0, "OR": 107.0, "B": 141.0}
    out = {}
    for L in loads:
        rng = np.random.default_rng(100 + L)
        t = np.arange(n) / FS
        sigs = []
        for name, _, size in CLASSES:
            x = (0.05 + 0.01 * L) * np.sin(2 * np.pi * 30 * t) + 0.02 * rng.standard_normal(n)
            if size > 0:
                f = freq["IR" if name.startswith("IR") else "OR" if name.startswith("OR") else "B"]
                period = int(FS / f)
                burst = np.exp(-np.arange(200) / 40.0) * np.sin(2 * np.pi * 3000 * np.arange(200) / FS)
                for s in range(0, n - 200, period):
                    x[s:s + 200] += (size / 14.0) * 0.4 * burst
            sigs.append(x.astype(np.float32))
        out[L] = sigs
    return out


def _windows(sig, n, lo=0, hi=None):
    hi = len(sig) if hi is None else hi
    starts = np.linspace(lo, hi - WIN, n).astype(int)
    return sig[starts[:, None] + np.arange(WIN)[None, :]]


def _add_awgn(x, snr_db, rng):
    p_sig = (x ** 2).mean(axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(p_noise)


def _split_one(sig, n, split, rng):
    """Windows from ONE recording, split 70/10/20.
    random: shuffled overlapping windows (base paper's style; optimistic, test windows overlap train).
    block : contiguous 70/10/20 segments of the signal (no shared samples; strict)."""
    if split == "block":
        N = len(sig)
        a, b = int(0.7 * N), int(0.8 * N)
        spec = {"train": (0, a, int(0.7 * n)), "val": (a, b, int(0.1 * n)), "test": (b, N, int(0.2 * n))}
        return {k: _windows(sig, c, lo, hi) for k, (lo, hi, c) in spec.items()}
    w = _windows(sig, n)
    perm = rng.permutation(n)
    a, b = int(0.7 * n), int(0.8 * n)
    return {"train": w[perm[:a]], "val": w[perm[a:b]], "test": w[perm[b:]]}


def build_dataset(signals, snr_db, seed=0, split="random", n_per_class=1000,
                  train_loads=(0,), test_loads=None):
    """
    Same loads for train and test (default): windows from those loads are pooled and split 70/10/20.
    Different loads (cross-load test): train/val come from train_loads, and the test set is built
    from test_loads only, so the model is tested on a motor load it never saw in training.
    """
    train_loads = list(train_loads)
    test_loads = list(train_loads if test_loads is None else test_loads)
    cross = set(train_loads) != set(test_loads)
    if cross and set(train_loads) & set(test_loads):
        raise ValueError("For cross-load runs, train_loads and test_loads must not overlap.")
    rng = np.random.default_rng(seed)
    parts = {k: dict(x=[], y=[], load=[]) for k in ("train", "val", "test")}

    def add(k, w, c, L):
        parts[k]["x"].append(w)
        parts[k]["y"].append(np.full(len(w), c))
        parts[k]["load"].append(np.full(len(w), L))

    for c in range(len(CLASSES)):
        for L in train_loads:
            sp = _split_one(signals[L][c], max(10, n_per_class // len(train_loads)), split, rng)
            for k in (("train", "val") if cross else ("train", "val", "test")):
                add(k, sp[k], c, L)
        if cross:
            cnt = max(1, int(round(0.2 * n_per_class / len(test_loads))))
            for L in test_loads:
                add("test", _windows(signals[L][c], cnt), c, L)

    out, scale = {}, None
    for i, k in enumerate(("train", "val", "test")):
        clean = np.concatenate(parts[k]["x"]).astype(np.float32)
        y = np.concatenate(parts[k]["y"]).astype(np.int64)
        load = np.concatenate(parts[k]["load"]).astype(np.int64)
        if scale is None:
            scale = clean.std()  # global scale from the training set only
        noisy = _add_awgn(clean, snr_db, np.random.default_rng(seed * 1000 + 7 + i))
        sh = (-1, 1, SIDE, SIDE)
        out[k] = {
            "clean": torch.from_numpy((clean / scale).reshape(sh)),
            "noisy": torch.from_numpy((noisy / scale).reshape(sh)),
            "y": torch.from_numpy(y),
            "size": torch.from_numpy(CLASS_SIZE[y]),
            "load": torch.from_numpy(load),
        }
    return out
