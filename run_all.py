"""Runs every (SNR x config x seed) combination and appends to results.csv (resumable)."""
import argparse, csv, os
import numpy as np
import torch
from data import load_cwru, synthetic_signals, build_dataset
from train import run_one, CONFIGS

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data", help="folder holding 97.mat, 98.mat, ...")
ap.add_argument("--synthetic", action="store_true", help="fake data, only to test the code")
ap.add_argument("--train_loads", type=int, nargs="+", default=[0], help="motor loads (HP) used for training, 0-3")
ap.add_argument("--test_loads", type=int, nargs="+", default=None,
                help="loads used for testing. Omit = same as train. Different = cross-load test")
ap.add_argument("--snrs", type=int, nargs="+", default=[-4, 0])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--configs", nargs="+", default=["base", "simam", "rgat", "pinn", "rgat_pinn"])
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--split", choices=["random", "block"], default="random")
ap.add_argument("--n_per_class", type=int, default=1000)
ap.add_argument("--lam_phys", type=float, default=0.1)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--out", default="results/results.csv")
a = ap.parse_args()

train_loads = a.train_loads
test_loads = a.test_loads if a.test_loads is not None else a.train_loads
tl_s, te_s = ",".join(map(str, train_loads)), ",".join(map(str, test_loads))
all_loads = sorted(set(train_loads) | set(test_loads))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device} | train loads {tl_s} HP -> test loads {te_s} HP | split={a.split}")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
signals = synthetic_signals(all_loads) if a.synthetic else load_cwru(a.data_dir, all_loads)

key = lambda r: (r["split"], r["train_loads"], r["test_loads"], str(r["snr"]), r["config"], str(r["seed"]))
done = set()
if os.path.exists(a.out):
    with open(a.out) as f:
        done = {key(r) for r in csv.DictReader(f)}

fields = ["split", "train_loads", "test_loads", "snr", "config", "seed", "val_acc", "test_acc", "best_epoch",
          "acc_normal", "acc_mild7", "acc_mid14", "acc_severe21",
          "acc_load0", "acc_load1", "acc_load2", "acc_load3",
          "train_s", "infer_ms_per_sample", "n_params"]
new_file = not os.path.exists(a.out)
with open(a.out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        w.writeheader()
    for snr in a.snrs:
        for seed in a.seeds:
            # identical data for every config at the same (snr, seed) -> fair comparison
            data = build_dataset(signals, snr, seed, a.split, a.n_per_class, train_loads, test_loads)
            for cfg in a.configs:
                if key(dict(split=a.split, train_loads=tl_s, test_loads=te_s, snr=snr, config=cfg, seed=seed)) in done:
                    continue
                row, ex, _ = run_one(cfg, data, seed, a.epochs, a.lam_phys,
                                     a.batch_size, device, want_example=(seed == a.seeds[0]))
                row.update(split=a.split, snr=snr, train_loads=tl_s, test_loads=te_s)
                w.writerow({k: row.get(k, "") for k in fields})
                f.flush()
                if ex is not None and cfg == "rgat":
                    np.savez(os.path.join(os.path.dirname(a.out) or ".", f"example_snr{snr}.npz"), **ex)
                print(f"snr={snr:>3} seed={seed} {cfg:<13} test={row['test_acc']:.4f} "
                      f"mild7={row['acc_mild7']:.3f} ({row['train_s']:.0f}s)", flush=True)
print("done ->", a.out)
