# Bearing fault diagnosis: base AE-CNN vs. RGAT + physics-informed loss

Base paper: Yuan, Xun, Wang, Su (2023), "Bearing Fault Diagnosis Based on Auto-Encoder Combined with CNN", FSDM 2023.
Same setup as the base paper: CWRU, 10 classes, 1024-sample windows reshaped to 32x32, AWGN noise, 70/10/20 split,
AE + CNN trained jointly with loss = cross-entropy + 0.5*MSE, Adam, lr 1e-3, weight decay 1e-4, batch 16, 25 epochs, lr x0.1 at epochs 15 and 20.

## What is compared (all use identical data for the same SNR + seed)
| config | what it is |
|---|---|
| `base` | the base paper's AE-CNN (re-implemented) |
| `simam` | base + SimAM, a generic attention module (to test "does ANY attention help?") |
| `rgat` | **ours**: the CNN also receives the AE's residual map |noisy - denoised| as a 2nd channel |
| `pinn` | base + physics loss only (ablation) |
| `rgat_pinn` | **full method**: RGAT + physics loss |
| `rgat_shuffle` | control: residual maps shuffled between samples (does the *specific* residual matter, or just an extra channel?) |

Physics loss = relative error of the envelope spectrum at BPFO / BPFI / 2xBSF (+ harmonics 1-3), computed from the SKF 6205 geometry at 1797 rpm.
It is "physics-informed" in the bearing-literature sense (like Lu 2023, Shen 2021); it does not solve a differential equation.

## Folder layout
```
bearing_rgat/
  data/            <- your 40 files: 97.mat, 98.mat, 99.mat, 100.mat, 105.mat ... 237.mat (flat, no subfolders)
  data.py  models.py  physics.py  train.py  run_all.py  report.py  check_data.py
```
File number = class base number + load (0-3 HP). Base numbers: Normal 97, IR007 105, B007 118, OR007@6 130, IR014 169,
B014 185, OR014@6 197, IR021 209, B021 222, OR021@6 234. (12 kHz drive-end data.)

## Steps
1. Install: `pip install -r requirements.txt`  (Colab has everything; choose Runtime > GPU)
2. Check your files: `python check_data.py data`   (it lists any missing file)
3. Smoke test with FAKE data (only proves the code runs):
   `python run_all.py --synthetic --snrs 4 --seeds 0 --epochs 2 --n_per_class 100 --configs base rgat_pinn --out results_test/results.csv`
4. **Experiment 1 - same load (matches the base paper, 0 HP):**
   `python run_all.py --snrs -4 --seeds 0 1 2 --configs base simam rgat pinn rgat_pinn`
5. **Experiment 2 - cross-load (train on 0 HP, test on unseen 1, 2, 3 HP):**
   `python run_all.py --train_loads 0 --test_loads 1 2 3 --split block --snrs -4 --seeds 0 1 2`
6. Control (does the *specific* residual matter?): add `--configs rgat_shuffle` to either experiment.
7. Full run (more seeds / SNRs): `--snrs -4 0 4 --seeds 0 1 2 3 4`. Re-running skips finished runs.
8. Strict, leak-free version of experiment 1: add `--split block`.
9. Report: `python report.py`  -> results/summary.md, accuracy plots, residual-map picture.
   Each experiment (loads, split, SNR) gets its own table, so they never get mixed.

Other options: `--train_loads 0 1 2 3` pools all loads for training. Tune only `--lam_phys` (default 0.1; try 0.03, 0.1, 0.3)
and judge it on the validation column, not the test set. If you ran an older version of this code, delete the old
`results/` folder first (the CSV columns changed).

## Things to be honest about in the report
- `random` split reuses overlapping windows across train/test (the base paper's style), so accuracies are optimistic. Report `block` too.
- The signal is reshaped to a 32x32 grid, so "where" in the residual map is just position in the window, not a time-frequency location.
- The base paper has no healthy-only detector or threshold; this code follows the paper (joint denoise + classify).
- Report mean ± std over seeds and the paired difference vs `base`. If RGAT does not beat `simam` / `rgat_shuffle`, say so; that is still a result.
