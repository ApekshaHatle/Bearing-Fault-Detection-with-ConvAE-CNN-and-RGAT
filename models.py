"""Models. 'base' re-implements the base paper (Tables 1-3, Eq. 2)."""
import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Parameter-free attention (Yang et al. 2021) - generic-attention baseline."""
    def __init__(self, lam=1e-4):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=(2, 3), keepdim=True)) ** 2
        v = d.sum(dim=(2, 3), keepdim=True) / n
        e_inv = d / (4 * (v + self.lam)) + 0.5
        return x * torch.sigmoid(e_inv)


def _block(cin, cout, simam, pool):
    layers = [
        nn.Conv2d(cin, cout, (3, 1), padding=(1, 0)), nn.BatchNorm2d(cout), nn.ReLU(True),
        nn.Conv2d(cout, cout, (1, 3), padding=(0, 1)), nn.BatchNorm2d(cout), nn.ReLU(True),
    ]
    if simam:
        layers.append(SimAM())
    layers.append(nn.MaxPool2d(2) if pool == "max" else nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)


class CNN(nn.Module):  # Table 1
    def __init__(self, in_ch=1, n_classes=10, simam=False):
        super().__init__()
        self.f = nn.Sequential(_block(in_ch, 16, simam, "max"),
                               _block(16, 32, simam, "max"),
                               _block(32, 64, simam, "avg"))
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc(self.f(x).flatten(1))


def _cbr(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True)]


class AE(nn.Module):  # Tables 2 and 3
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            *_cbr(1, 64), nn.MaxPool2d(2),
            *_cbr(64, 128), nn.MaxPool2d(2),
            *_cbr(128, 128), nn.MaxPool2d(2),
            *_cbr(128, 128), nn.AdaptiveAvgPool2d(1))

        def up(cin, cout):
            return [nn.ConvTranspose2d(cin, cout, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(cout), nn.ReLU(True)]
        self.dec = nn.Sequential(*up(128, 128), *up(128, 64), *up(64, 64),
                                 *up(64, 64), *up(64, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        return self.dec(self.enc(x))


class JointNet(nn.Module):
    """
    mode:
      base          AE -> CNN (base paper, trained jointly)
      simam         base + SimAM generic attention inside the CNN
      rgat          CNN also receives the AE's residual map |noisy - denoised|
      rgat_shuffle  CONTROL: residual maps shuffled across samples
    """
    def __init__(self, mode="base", n_classes=10):
        super().__init__()
        self.mode = mode
        self.ae = AE()
        in_ch = 2 if mode.startswith("rgat") else 1
        self.cnn = CNN(in_ch, n_classes, simam=(mode == "simam"))

    def forward(self, x_noisy):
        x_hat = self.ae(x_noisy)
        if self.mode.startswith("rgat"):
            r = (x_noisy - x_hat).abs()
            r = r / (r.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)  # per-sample [0,1]
            r = r.detach()
            if self.mode == "rgat_shuffle":
                r = r[torch.randperm(r.shape[0], device=r.device)]
            inp = torch.cat([x_hat, r], dim=1)
        else:
            inp = x_hat
        return self.cnn(inp), x_hat
