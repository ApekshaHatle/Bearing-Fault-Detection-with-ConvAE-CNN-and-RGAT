import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import JointNet
from physics import band_mask, physics_loss
from data import CLASS_SIZE

CONFIGS = {
    "base":         dict(mode="base",         phys=False),  # base paper (AE-CNN)
    "simam":        dict(mode="simam",        phys=False),  # + generic attention
    "rgat":         dict(mode="rgat",         phys=False),  # + residual-guided attention (ours)
    "pinn":         dict(mode="base",         phys=True),   # + physics loss only (ablation)
    "rgat_pinn":    dict(mode="rgat",         phys=True),   # full method
    "rgat_shuffle": dict(mode="rgat_shuffle", phys=False),  # control
}


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, d, device, bs=256):
    model.eval()
    preds = []
    for i in range(0, len(d["y"]), bs):
        logits, _ = model(d["noisy"][i:i + bs].to(device))
        preds.append(logits.argmax(1).cpu())
    p = torch.cat(preds)
    return (p == d["y"]).float().mean().item(), p


def run_one(config, data, seed, epochs=25, lam_phys=0.1, batch_size=16,
            device="cpu", want_example=False):
    cfg = CONFIGS[config]
    set_seed(seed)
    model = JointNet(cfg["mode"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15, 20], gamma=0.1)
    mask = band_mask()
    tr = data["train"]
    loader = DataLoader(TensorDataset(tr["noisy"], tr["clean"], tr["y"]),
                        batch_size=batch_size, shuffle=True)

    best = dict(val=-1, test=0, preds=None, epoch=-1)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for xn, xc, y in loader:
            xn, xc, y = xn.to(device), xc.to(device), y.to(device)
            logits, x_hat = model(xn)
            # base paper Eq. 2: cross-entropy + 0.5 * MSE (denoised vs clean)
            loss = F.cross_entropy(logits, y) + 0.5 * F.mse_loss(x_hat, xc)
            if cfg["phys"]:
                loss = loss + lam_phys * physics_loss(x_hat, xc, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        va, _ = evaluate(model, data["val"], device)
        if va >= best["val"]:  # pick the epoch by VALIDATION accuracy (no test peeking)
            te, preds = evaluate(model, data["test"], device)
            best = dict(val=va, test=te, preds=preds, epoch=ep)
    train_s = time.perf_counter() - t0

    # inference speed
    te_x = data["test"]["noisy"].to(device)
    model.eval()
    with torch.no_grad():
        model(te_x[:64])
        t1 = time.perf_counter()
        for i in range(0, len(te_x), 64):
            model(te_x[i:i + 64])
        infer_ms = (time.perf_counter() - t1) / len(te_x) * 1000

    y, p = data["test"]["y"], best["preds"]
    size = data["test"]["size"]
    row = dict(config=config, seed=seed, val_acc=best["val"], test_acc=best["test"],
               best_epoch=best["epoch"],
               acc_normal=(p[size == 0] == y[size == 0]).float().mean().item(),
               acc_mild7=(p[size == 7] == y[size == 7]).float().mean().item(),
               acc_mid14=(p[size == 14] == y[size == 14]).float().mean().item(),
               acc_severe21=(p[size == 21] == y[size == 21]).float().mean().item(),
               train_s=train_s, infer_ms_per_sample=infer_ms,
               n_params=sum(q.numel() for q in model.parameters()))
    load = data["test"]["load"]
    for L in range(4):
        m = load == L
        row[f"acc_load{L}"] = (p[m] == y[m]).float().mean().item() if m.any() else float("nan")
    example = None
    if want_example:
        with torch.no_grad():
            idx = torch.arange(0, len(y), max(1, len(y) // 10))[:10]
            xn = data["test"]["noisy"][idx].to(device)
            _, xh = model(xn)
            example = dict(noisy=xn.cpu().numpy(), denoised=xh.cpu().numpy(),
                           resid=(xn - xh).abs().cpu().numpy(), y=y[idx].numpy())
    return row, example, best["preds"].numpy()
