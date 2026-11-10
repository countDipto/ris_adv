"""Data loading for the RISnet DeepMIMO channel set (Peng et al., GLOBECOM 2023).

Files expected in DATA_DIR (from the RISnet Google Drive link):
    channels_ris_rx_training.pt   (20480, 1, 1296) complex  RIS -> user channel per location
    channels_ris_rx_testing.pt    (1024, 1, 1296)  complex
    channel_tx_ris.pt             (1296, 9)        complex  RIS <- BS channel
    channels_direct_*.pt          direct BS -> user channels
    group_definition_4users_*.npy indices of the 4-user groups

DSRnet setting (Alam et al.): 16 active sensing elements on a 4x4 grid of the 36x36 RIS,
at row/column indices [4, 13, 22, 31] (the RISnet `indices_sensors`).

Released tensors are (N, 1, 1296) complex128; reshaped row-major to (N, 36, 36).
channel_tx_ris.pt is (1296, 9). Each sample here is one user's 36x36 RIS->user channel. The partial CSI is the 4x4
sub-grid, given as (amplitude, phase) features as in DSRnet Eq. (2). The target is the
full 36x36 channel as (real, imag), divided by a global scale so values are O(1).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

SENSOR_IDX = [4, 13, 22, 31]


def load_pt(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    return obj


def sensor_mask(n=36, idx=SENSOR_IDX):
    m = torch.zeros(n, n, dtype=torch.bool)
    ii = torch.tensor(idx)
    m[ii[:, None], ii[None, :]] = True
    return m


def partial_from_full(h, idx=SENSOR_IDX):
    """h: (..., 36, 36) complex -> (..., 4, 4) complex at the sensing elements."""
    ii = torch.as_tensor(idx, device=h.device)
    return h[..., ii[:, None], ii[None, :]]


def complex_to_amp_phase(hp, amp_scale):
    """(..., 4, 4) complex -> (..., 2, 4, 4) real features [|h|/scale, angle/pi]."""
    return torch.stack([hp.abs() / amp_scale, hp.angle() / np.pi], dim=-3)


def complex_to_ri(h, scale):
    return torch.stack([h.real / scale, h.imag / scale], dim=-3)


def ri_to_complex(x, scale):
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :]) * scale


def add_pilot_noise(hp, snr_db, generator=None):
    """Complex AWGN on the sensed partial CSI at a per-sample SNR.
    snr_db: None (noiseless), a number, or a (lo, hi) tuple -> SNR drawn uniformly per sample."""
    if snr_db is None:
        return hp
    if isinstance(snr_db, (tuple, list)):
        lo, hi = snr_db
        snr_db = lo + (hi - lo) * torch.rand(hp.shape[:-2] + (1, 1), generator=generator)
    snr_db = torch.as_tensor(snr_db, dtype=torch.float32)
    p = hp.abs().pow(2).mean(dim=(-2, -1), keepdim=True)
    sigma = torch.sqrt(p / (10 ** (snr_db.to(p.device) / 10)) / 2)
    n = torch.complex(torch.randn(hp.shape, generator=generator),
                      torch.randn(hp.shape, generator=generator)).to(hp.device)
    return hp + sigma * n


class RISChannelDataset(Dataset):
    """Returns (x_partial_features (2,4,4), y_full_ri (2,36,36), h_full_complex (36,36))."""

    def __init__(self, channels, scale=None, pilot_snr_db=None, idx=SENSOR_IDX,
                 input_repr="ap"):
        channels = torch.as_tensor(channels).to(torch.complex64)
        # RISnet files store (N, 1, 1296); element order is row-major on the 36x36 grid
        # (checked: adjacent-element phase increments are smooth along both axes).
        assert channels.shape[-1] == 1296 or channels.shape[-2:] == (36, 36), channels.shape
        self.h = channels.reshape(-1, 36, 36)
        self.scale = float(scale if scale is not None else self.h.abs().pow(2).mean().sqrt())
        self.pilot_snr_db = pilot_snr_db
        self.idx = idx
        self.input_repr = input_repr      # 'ap' (amp/phase, as in DSRnet) or 'ri'

    def __len__(self):
        return self.h.shape[0]

    def features(self, hp):
        return encode_partial(hp, self.scale, self.input_repr)

    def __getitem__(self, i):
        h = self.h[i]
        hp = add_pilot_noise(partial_from_full(h, self.idx), self.pilot_snr_db)
        return self.features(hp), complex_to_ri(h, self.scale), h


def encode_partial(hp, scale, input_repr="ap"):
    """Differentiable encoder of partial CSI; used by training and by any physical
    perturbation model that changes hp, so both see the same input pipeline."""
    if input_repr == "ap":
        return complex_to_amp_phase(hp, scale)
    return complex_to_ri(hp, scale)


def load_split(data_dir, split="training", variant=""):
    """variant: '' (LoS ray tracing), '_5e-5', or '_iid' as in the RISnet release."""
    path = os.path.join(data_dir, f"channels_ris_rx_{split}{variant}.pt")
    return load_pt(path)


def load_tx_ris(data_dir):
    return load_pt(os.path.join(data_dir, "channel_tx_ris.pt"))


def split_indices(locations, val_frac=0.05, holdout_frac=0.10, seed=0):
    """Split the training file into train / val (random) / far (spatially disjoint).

    far: users in the last `holdout_frac` of the street's x-range. They are never trained
    on, so they measure generalisation to an unseen part of the street. The released test
    file is interleaved with the training grid (every test user is 0.2 m from a training
    user), so on its own it measures interpolation, not generalisation.
    """
    loc = torch.as_tensor(locations)[:, :2]
    x = loc[:, 0]
    cut = x.min() + (1 - holdout_frac) * (x.max() - x.min())
    far = torch.nonzero(x > cut).flatten()
    rest = torch.nonzero(x <= cut).flatten()
    g = torch.Generator().manual_seed(seed)
    rest = rest[torch.randperm(len(rest), generator=g)]
    n_val = int(round(val_frac * len(rest)))
    return {"train": rest[n_val:], "val": rest[:n_val], "far": far}
