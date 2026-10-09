"""Turns results.csv into tables, paired statistics and plots."""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/results.csv")
a = ap.parse_args()
out = os.path.dirname(a.csv) or "."
df = pd.read_csv(a.csv, dtype={"train_loads": str, "test_loads": str})
order = [c for c in ["base", "simam", "rgat", "pinn", "rgat_pinn", "rgat_shuffle"] if c in df.config.unique()]
metrics = ["test_acc", "acc_mild7", "acc_mid14", "acc_severe21"]
GROUP = ["split", "train_loads", "test_loads", "snr"]

lines = []
for (split, trl, tel, snr), g in df.groupby(GROUP):
    kind = "same-load" if trl == tel else "CROSS-LOAD"
    lines.append(f"\n### {kind}: train {trl} HP -> test {tel} HP | split={split} | SNR={snr} dB "
                 f"(seeds per config: {g.groupby('config').seed.nunique().min()})\n")
    lines.append("| config | test acc (%) | mild 7mil (%) | 14mil (%) | severe 21mil (%) | infer ms/sample | params |")
    lines.append("|---|---|---|---|---|---|---|")
    fmt = lambda r, m: (f"{100 * r[m].mean():.2f} ± {100 * r[m].std(ddof=1):.2f}" if len(r) > 1 else f"{100 * r[m].mean():.2f}")
    for c in order:
        r = g[g.config == c]
        if r.empty:
            continue
        lines.append(f"| {c} | {fmt(r, 'test_acc')} | {fmt(r, 'acc_mild7')} | {fmt(r, 'acc_mid14')} | "
                     f"{fmt(r, 'acc_severe21')} | {r.infer_ms_per_sample.mean():.3f} | {int(r.n_params.iloc[0]):,} |")
    if kind == "CROSS-LOAD":
        cols = [f"acc_load{L}" for L in range(4) if g[f"acc_load{L}"].notna().any()]
        lines.append("\nAccuracy on each unseen test load (%):\n")
        lines.append("| config | " + " | ".join(c.replace("acc_load", "load ") + " HP" for c in cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for c in order:
            r = g[g.config == c]
            if not r.empty:
                lines.append(f"| {c} | " + " | ".join(fmt(r, k) for k in cols) + " |")
    base = g[g.config == "base"].set_index("seed")
    lines.append("\nPaired difference vs base (same seed = same data), test accuracy in points:")
    for c in order:
        if c == "base":
            continue
        r = g[g.config == c].set_index("seed")
        common = base.index.intersection(r.index)
        if len(common) < 2:
            continue
        d = 100 * (r.loc[common, "test_acc"] - base.loc[common, "test_acc"])
        p = stats.ttest_rel(r.loc[common, "test_acc"], base.loc[common, "test_acc"]).pvalue
        lines.append(f"- {c}: {d.mean():+.2f} ± {d.std(ddof=1):.2f} pts (n={len(common)}, paired t-test p={p:.3f})")

txt = "\n".join(lines)
open(os.path.join(out, "summary.md"), "w").write(txt)
print(txt)

for (split, trl, tel, snr), g in df.groupby(GROUP):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    m = g.groupby("config").test_acc.agg(["mean", "std"]).reindex([c for c in order if c in g.config.unique()])
    ax[0].bar(m.index, 100 * m["mean"], yerr=100 * m["std"].fillna(0), capsize=4)
    ax[0].set_ylabel("test accuracy (%)")
    ax[0].set_title(f"train {trl} -> test {tel} HP, SNR={snr} dB ({split})", fontsize=9)
    ax[0].set_ylim(max(0, 100 * m["mean"].min() - 3), 100.5); ax[0].tick_params(axis="x", rotation=30)
    cs = [c for c in order if c in g.config.unique()]
    w = 0.8 / len(cs)
    for i, c in enumerate(cs):
        r = g[g.config == c]
        ax[1].bar(np.arange(3) + i * w, [100 * r[k].mean() for k in metrics[1:]], w, label=c)
    ax[1].set_xticks(np.arange(3) + 0.4 - w / 2); ax[1].set_xticklabels(["7 mil (mild)", "14 mil", "21 mil (severe)"])
    ax[1].set_title("Accuracy by fault severity"); ax[1].set_ylabel("%"); ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"accuracy_{split}_train{trl.replace(',', '')}_test{tel.replace(',', '')}_snr{snr}.png"), dpi=140)
    plt.close()

for fn in os.listdir(out):
    if fn.startswith("example_") and fn.endswith(".npz"):
        e = np.load(os.path.join(out, fn))
        k = min(4, len(e["y"]))
        fig, ax = plt.subplots(3, k, figsize=(3 * k, 8))
        for j in range(k):
            for i, (key, t) in enumerate([("noisy", "noisy input"), ("denoised", "AE output"), ("resid", "residual map (RGAT)")]):
                ax[i, j].imshow(e[key][j, 0], cmap="viridis"); ax[i, j].axis("off")
                ax[i, j].set_title(f"{t}\nclass {e['y'][j]}", fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(out, fn.replace(".npz", "_residuals.png")), dpi=140); plt.close()
print("\nSaved summary.md and plots in", out)
