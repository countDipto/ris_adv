"""Victim and surrogate models for partial-CSI -> full-CSI RIS channel estimation.

DSRnet is re-implemented from the paper's description (Alam et al., "DSRnet: Hybrid Deep
Learning-based Channel Estimation for RIS-Aided Wireless Communication"):
  * SR sub-network: conv 9x9 + LReLU on the 4x4 input -> upsample to 12x12 -> conv 1x1
    -> upsample to 36x36 -> conv 5x5.
  * Denoising sub-network: two spatial attention modules (SAMs). SAM = spatial feature
    extraction block (parallel 3x3 convs, dilation 1-4, concatenated with the input)
    -> two local residual blocks -> channel attention block (GAP -> conv down -> sigmoid
    gate). SAM1 output is added to the SR output before SAM2. A tail conv block is added
    back to the input and refined by a final conv.
Channel widths are not given in the paper; the defaults (feat=28, reduction=4) give
83,836 parameters, close to the reported 80,916.

DnCNN and SRCNN are the paper's baselines and also serve as surrogates for transfer
attacks.
"""
import torch
from torch import nn
import torch.nn.functional as F


def lrelu():
    return nn.LeakyReLU(0.2, inplace=False)


class SRSubNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, feat=32, mode="bicubic"):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, feat, 9, padding=4)
        self.c2 = nn.Conv2d(feat, feat, 1)
        self.c3 = nn.Conv2d(feat, out_ch, 5, padding=2)
        self.mode = mode

    def forward(self, x):
        f = F.leaky_relu(self.c1(x), 0.2)
        f = F.interpolate(f, size=(12, 12), mode=self.mode, align_corners=False)
        f = F.leaky_relu(self.c2(f), 0.2)
        f = F.interpolate(f, size=(36, 36), mode=self.mode, align_corners=False)
        return self.c3(f)


class SFEB(nn.Module):
    """Spatial feature extraction: parallel dilated 3x3 convs, concatenated with input."""

    def __init__(self, ch, dilations=(1, 2, 3, 4)):
        super().__init__()
        b = ch // len(dilations)
        self.branches = nn.ModuleList(
            [nn.Conv2d(ch, b, 3, padding=d, dilation=d) for d in dilations])
        self.fuse = nn.Conv2d(ch + b * len(dilations), ch, 1)

    def forward(self, x):
        y = torch.cat([F.leaky_relu(br(x), 0.2) for br in self.branches] + [x], dim=1)
        return self.fuse(y)


class LRB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.leaky_relu(self.c1(x), 0.2))


class CAB(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.down = nn.Conv2d(ch, ch // reduction, 1)
        self.up = nn.Conv2d(ch // reduction, ch, 1)

    def forward(self, f, x_in):
        w = F.adaptive_avg_pool2d(f, 1)
        w = torch.sigmoid(self.up(F.relu(self.down(w))))
        return x_in * w


class SAM(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.sfeb = SFEB(ch)
        self.lrb1 = LRB(ch)
        self.lrb2 = LRB(ch)
        self.cab = CAB(ch, reduction)

    def forward(self, x):
        f = self.lrb2(self.lrb1(self.sfeb(x)))
        return x + f + self.cab(f, x)   # long + short residuals


class DSRnet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, feat=28, sr_feat=32, reduction=4, n_sam=2):
        super().__init__()
        self.sr = SRSubNet(in_ch, out_ch, sr_feat)
        self.head = nn.Conv2d(out_ch, feat, 3, padding=1)
        self.sams = nn.ModuleList([SAM(feat, reduction) for _ in range(n_sam)])
        self.tail = nn.Conv2d(feat, out_ch, 3, padding=1)
        self.final = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x0 = self.sr(x)                  # initial full-CSI estimate (2,36,36)
        f0 = self.head(x0)
        f = self.sams[0](f0)
        for sam in self.sams[1:]:
            f = sam(f + f0)              # SAM1 output added to SR features, then SAM2
        y = self.tail(f) + x0
        return self.final(y) + y


class DnCNN(nn.Module):
    """DnCNN [Zhang et al. 2017] on a bicubic-upsampled 36x36 input (residual learning)."""

    def __init__(self, in_ch=2, out_ch=2, feat=40, depth=10):
        super().__init__()
        self.inp = nn.Conv2d(in_ch, out_ch, 1)
        layers = [nn.Conv2d(out_ch, feat, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(feat, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU()]
        layers += [nn.Conv2d(feat, out_ch, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        x = self.inp(F.interpolate(x, size=(36, 36), mode="bicubic", align_corners=False))
        return x - self.body(x)


class SRCNN(nn.Module):
    """SRCNN [Dong et al. 2014] on a bicubic-upsampled 36x36 input."""

    def __init__(self, in_ch=2, out_ch=2):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 64, 9, padding=4)
        self.c2 = nn.Conv2d(64, 32, 1)
        self.c3 = nn.Conv2d(32, out_ch, 5, padding=2)

    def forward(self, x):
        x = F.interpolate(x, size=(36, 36), mode="bicubic", align_corners=False)
        return self.c3(F.relu(self.c2(F.relu(self.c1(x)))))


MODELS = {"dsrnet": DSRnet, "dncnn": DnCNN, "srcnn": SRCNN}


def build(name, **kw):
    return MODELS[name](**kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
