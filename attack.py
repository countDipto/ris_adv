"""Phase-constrained malicious-RIS attacks on partial-CSI channel estimators.

No synthetic adversarial tensor is added to the estimator input. Instead, a
second, passive RIS changes the *physical* RIS-to-user channel:

    h(phi) = h0 + G_ar @ (phi * g_ua)

G_ar and g_ua must come from a geometry-aware or measured propagation model,
in the SAME complex-channel units as h0. This module does not invent them.

The primary attack is universal: one phase vector is optimized on TRAIN users,
then kept fixed for the unseen TEST users. ``universal=False`` supplies a
white-box, per-user upper bound. Phase quantization uses a straight-through
gradient and tracks the best actually quantized candidate.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch

from data import encode_partial, partial_from_full, ri_to_complex


@dataclass(frozen=True)
class AttackConfig:
    steps: int = 150
    lr: float = 0.08
    restarts: int = 3
    phase_bits: Optional[int] = 1  # None means continuous unit-modulus phases
    pilot_snr_db: Optional[float] = 20.0
    eot_samples: int = 3
    universal: bool = True
    seed: int = 0

    def validate(self):
        if self.steps < 1 or self.restarts < 1 or self.eot_samples < 1:
            raise ValueError("steps, restarts and eot_samples must be positive")
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")
        if self.phase_bits is not None and not 1 <= self.phase_bits <= 8:
            raise ValueError("phase_bits must be 1..8 or None")


@dataclass
class AttackResult:
    phi: torch.Tensor           # (A,) universal, otherwise (B,A), complex
    h_perturbed: torch.Tensor   # (B,36,36), complex
    nmse_linear: torch.Tensor   # (B,) on optimization noise realizations
    relative_change_db: torch.Tensor  # (B,) ||h_phi-h0||^2 / ||h0||^2 in dB
    config: AttackConfig


def _check_links(h0, G_ar, g_ua):
    if h0.ndim != 3 or tuple(h0.shape[-2:]) != (36, 36):
        raise ValueError("h0 must be complex (B,36,36)")
    if not h0.is_complex() or not G_ar.is_complex() or not g_ua.is_complex():
        raise TypeError("h0, G_ar and g_ua must all be complex tensors")
    b = h0.shape[0]
    if G_ar.ndim not in (2, 3) or G_ar.shape[-2] != 1296:
        raise ValueError("G_ar must have shape (1296,A) or (B,1296,A)")
    if G_ar.ndim == 3 and G_ar.shape[0] != b:
        raise ValueError("batched G_ar must have B rows")
    a = G_ar.shape[-1]
    if g_ua.ndim == 1:
        if g_ua.shape[0] != a:
            raise ValueError("g_ua must have A entries")
    elif g_ua.ndim == 2:
        if g_ua.shape != (b, a):
            raise ValueError("g_ua must have shape (B,A)")
    else:
        raise ValueError("g_ua must have shape (A,) or (B,A)")
    if h0.device != G_ar.device or h0.device != g_ua.device:
        raise ValueError("all physical channel tensors must be on the same device")
    return b, a


def channel_with_malicious_ris(h0, G_ar, g_ua, phi):
    """Construct the true perturbed channel, with no arbitrary epsilon budget.

    G_ar: legit-RIS elements x attacker-RIS elements; g_ua: user -> attacker.
    ``phi`` may be shared (A,) or per user (B,A). Phase coefficients are
    supplied in linear complex units; this function does not normalize them.
    """
    b, a = _check_links(h0, G_ar, g_ua)
    if phi.shape not in ((a,), (b, a)) or not phi.is_complex():
        raise ValueError("phi must be complex (A,) or (B,A)")
    if phi.device != h0.device:
        raise ValueError("phi and channel tensors must share a device")
    if not torch.allclose(phi.abs().detach(), torch.ones_like(phi.real), atol=2e-5):
        raise ValueError("malicious RIS coefficients must have unit modulus")
    weighted = phi * g_ua
    if weighted.ndim == 1:
        weighted = weighted.unsqueeze(0).expand(b, -1)
    if G_ar.ndim == 2:
        delta = torch.einsum("ma,ba->bm", G_ar, weighted)
    else:
        delta = torch.einsum("bma,ba->bm", G_ar, weighted)
    return h0 + delta.reshape(b, 36, 36)


def phases_from_angles(theta, bits=None, straight_through=False):
    """Unit-modulus continuous or 2**bits-level phase shifts, alphabet starts at 0."""
    phi_soft = torch.polar(torch.ones_like(theta), theta)
    if bits is None:
        return phi_soft
    spacing = 2 * math.pi / (2 ** bits)
    indices = torch.remainder(torch.round(theta / spacing), 2 ** bits)
    phi_hard = torch.polar(torch.ones_like(theta), indices * spacing)
    if straight_through:
        # Forward is genuinely quantized; backwards follows the continuous phase.
        return phi_soft + (phi_hard - phi_soft).detach()
    return phi_hard


def random_phases(n_elements, bits, seed, batch=None, device="cpu"):
    """Reproducible random-phase control with the attack's exact phase alphabet."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (n_elements,) if batch is None else (batch, n_elements)
    if bits is None:
        theta = 2 * math.pi * torch.rand(shape, generator=g)
    else:
        theta = (2 * math.pi / (2 ** bits)) * torch.randint(
            0, 2 ** bits, shape, generator=g).float()
    return phases_from_angles(theta.to(device), None)


def _pilot_observation(h, snr_db, seed):
    hp = partial_from_full(h)
    if snr_db is None:
        return hp
    gen = torch.Generator(device=hp.device).manual_seed(seed)
    real = torch.randn(hp.shape, dtype=hp.real.dtype, device=hp.device, generator=gen)
    imag = torch.randn(hp.shape, dtype=hp.real.dtype, device=hp.device, generator=gen)
    unit_noise = torch.complex(real, imag)
    power = hp.abs().square().mean(dim=(-2, -1), keepdim=True)
    sigma = (power / (2 * 10 ** (snr_db / 10))).sqrt()
    return hp + sigma * unit_noise


def predict_channel(model, h, scale, input_repr="ap", pilot_snr_db=20., seed=123):
    """Differentiable estimator output for an actual channel (not a digital attack)."""
    hp = _pilot_observation(h, pilot_snr_db, seed)
    features = encode_partial(hp, scale, input_repr)
    return ri_to_complex(model(features), scale)


def per_user_nmse(h_hat, h):
    num = (h_hat - h).abs().square().flatten(1).sum(1)
    den = h.abs().square().flatten(1).sum(1).clamp_min(1e-30)
    return num / den


def _scores(model, h0, G_ar, g_ua, phi, scale, input_repr, cfg, seed):
    h = channel_with_malicious_ris(h0, G_ar, g_ua, phi)
    scores = []
    for k in range(cfg.eot_samples):
        pred = predict_channel(model, h, scale, input_repr, cfg.pilot_snr_db,
                               seed + k)
        scores.append(per_user_nmse(pred, h))
    return torch.stack(scores).mean(0)


def optimize_malicious_ris(model, h0, G_ar, g_ua, scale, input_repr="ap",
                           config=None):
    """Maximize true-channel NMSE by choosing ONLY malicious-RIS phases.

    Universal mode optimizes a single vector across all supplied training users.
    Per-user mode is a powerful, idealized white-box upper bound. Model
    parameters are never updated. Use a surrogate as ``model`` for transfer.
    For reproducibility the EOT noise draws are common to every phase candidate.
    """
    cfg = config or AttackConfig()
    cfg.validate()
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("scale must be a positive, finite training-set scale")
    device = next(model.parameters()).device
    h0 = h0.to(device=device, dtype=torch.complex64)
    G_ar = G_ar.to(device=device, dtype=torch.complex64)
    g_ua = g_ua.to(device=device, dtype=torch.complex64)
    b, a = _check_links(h0, G_ar, g_ua)
    shape = (a,) if cfg.universal else (b, a)
    best_scores = torch.full((1 if cfg.universal else b,), -float("inf"), device=device)
    best_phi = None
    model.eval()

    for restart in range(cfg.restarts):
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + 1237 * restart)
        theta = (2 * math.pi * torch.rand(shape, generator=g) - math.pi).to(device)
        theta.requires_grad_()
        optimizer = torch.optim.Adam([theta], lr=cfg.lr)

        def record_candidate():
            nonlocal best_scores, best_phi
            with torch.no_grad():
                hard_phi = phases_from_angles(theta, cfg.phase_bits).detach()
                score = _scores(model, h0, G_ar, g_ua, hard_phi, scale,
                                input_repr, cfg, cfg.seed + 5000)
                compared = score.mean().view(1) if cfg.universal else score
                improved = compared > best_scores
                if best_phi is None:
                    best_phi = hard_phi.clone()
                elif cfg.universal:
                    if improved.item():
                        best_phi.copy_(hard_phi)
                else:
                    best_phi[improved] = hard_phi[improved]
                best_scores = torch.where(improved, compared, best_scores)

        record_candidate()
        for step in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            phi = phases_from_angles(theta, cfg.phase_bits, straight_through=True)
            error = _scores(model, h0, G_ar, g_ua, phi, scale, input_repr,
                            cfg, cfg.seed + 5000)
            # Autograd only for theta: no gradient is accumulated on victim weights.
            theta.grad = torch.autograd.grad(-error.mean(), theta)[0]
            optimizer.step()
            if (step + 1) % 10 == 0 or step + 1 == cfg.steps:
                record_candidate()

    with torch.no_grad():
        h_attack = channel_with_malicious_ris(h0, G_ar, g_ua, best_phi)
        score = _scores(model, h0, G_ar, g_ua, best_phi, scale, input_repr,
                        cfg, cfg.seed + 5000)
        ratio = ((h_attack - h0).abs().square().flatten(1).sum(1) /
                 h0.abs().square().flatten(1).sum(1).clamp_min(1e-30))
        change_db = 10 * torch.log10(ratio.clamp_min(1e-30))
    return AttackResult(best_phi.detach(), h_attack.detach(), score.detach(),
                        change_db.detach(), cfg)
