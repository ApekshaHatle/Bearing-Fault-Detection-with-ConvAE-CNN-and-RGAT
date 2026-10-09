"""
Physics-informed term (the "PINN" part).

Honest note: this is "physics-informed" in the sense used in the bearing
literature (Lu et al. 2023, Shen et al. 2021): the loss is built from the
bearing's known fault characteristic frequencies. It does NOT solve a PDE.

Idea: a bearing defect shows up as periodic impacts at BPFO / BPFI / 2*BSF
(and harmonics), visible in the ENVELOPE spectrum. We ask the denoiser to
preserve the envelope-spectrum energy in those bands.
"""
import numpy as np
import torch

FS = 12000

# SKF 6205-2RS bearing used by CWRU
N_BALLS, D_BALL, D_PITCH = 9, 0.3126, 1.537


def fault_frequencies(rpm=1797.0):
    fr = rpm / 60.0
    r = D_BALL / D_PITCH
    return {
        "shaft": fr,
        "BPFO": N_BALLS / 2 * fr * (1 - r),
        "BPFI": N_BALLS / 2 * fr * (1 + r),
        "2BSF": 2 * D_PITCH / (2 * D_BALL) * fr * (1 - r ** 2),
    }


def band_mask(length=1024, fs=FS, rpm=1797.0, harmonics=(1, 2, 3), half_width_hz=12.0):
    freqs = np.fft.rfftfreq(length, 1 / fs)
    mask = np.zeros_like(freqs)
    ff = fault_frequencies(rpm)
    for key in ("BPFO", "BPFI", "2BSF"):
        for h in harmonics:
            mask[np.abs(freqs - h * ff[key]) <= half_width_hz] = 1.0
    return torch.tensor(mask, dtype=torch.float32)


def envelope_spectrum(x):  # x: (B, L)
    L = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    h = torch.zeros(L, device=x.device)
    h[0] = 1
    h[1:L // 2] = 2
    h[L // 2] = 1
    a = torch.fft.ifft(X * h, dim=-1)
    env = torch.sqrt(a.real ** 2 + a.imag ** 2 + 1e-12)
    env = env - env.mean(dim=-1, keepdim=True)
    E = torch.fft.rfft(env, dim=-1)
    return torch.sqrt(E.real ** 2 + E.imag ** 2 + 1e-12) / L


def physics_loss(x_hat, x_clean, mask):
    """Relative error of the fault-band envelope spectrum (scale-free, ~O(1))."""
    e_hat = envelope_spectrum(x_hat.flatten(1))
    e_cln = envelope_spectrum(x_clean.flatten(1))
    m = mask.to(x_hat.device)
    num = (((e_hat - e_cln) ** 2) * m).sum(-1).mean()
    den = ((e_cln ** 2) * m).sum(-1).mean().detach() + 1e-8
    return num / den
