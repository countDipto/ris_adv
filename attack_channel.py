"""Channel model for a rogue, zero-power ("adversarial") reflecting surface.

Threat model
------------
A passive second surface is placed in the scene near the legitimate 36x36 RIS.
It transmits no power of its own; it only re-radiates the ambient pilot that the
legitimate RIS's sensing elements already measure. By choosing its per-element
phase configuration ``phi`` it adds a structured term ``delta`` to the 16 sensed
channel samples, pushing DSRnet off its learned, site-specific prior.

This module provides only the *physics*: the cascaded
``source -> adversarial element -> sensing element`` channel ``delta(phi)``, its
power calibration, and the matched random-phase control. The phase-optimisation
loop that chooses ``phi`` lives in ``attack.py``.

Geometry and units
------------------
Everything is in wavelengths (lambda), so the carrier drops out of every steering
phase and nothing depends on an absolute path loss we cannot read from the data.
Half-wavelength element spacing (the RISnet / DeepMIMO convention) is the default.

  * Legitimate RIS: 36x36 planar array in the z=0 plane. Element (row r, col c)
    sits at (c*ris_spacing, r*ris_spacing, 0). Row-major reshape of the released
    1296-vectors to (36, 36) matches ``data.partial_from_full``: the sensed (4,4)
    grid has entry [a, b] = element (row=SENSOR_IDX[a], col=SENSOR_IDX[b]).
  * Adversarial surface: ``n_adv`` x ``n_adv`` planar array centred at ``center``,
    lying in the x-z plane (facing the RIS along y). It re-radiates a plane wave
    arriving from direction ``incidence`` (the pilot source).

Cascaded model
--------------
For sensing element s and adversarial element k,

    delta_s(phi) = c * sum_k  A[s,k] * exp( j * ( phi_k + kappa_k + 2*pi*d[s,k] ) )
                 = c * (D @ exp(j*phi))_s ,      D[s,k] = A[s,k]*exp(j(kappa_k+2*pi*d[s,k]))

  kappa_k   = 2*pi * <p_k, incidence>          incidence steering on the surface
  d[s,k]    = || p_k - q_s ||  (in lambda)      near-field spherical departure phase
  A[s,k]    = 1/d[s,k] (near_field) or 1         amplitude spreading to the sensor
  c         scalar set by :meth:`calibrate`      physical reflection strength

``phi`` enters only through ``exp(j*phi)``, so ``delta`` is differentiable in
``phi`` and the attack can backprop the estimator's error onto the surface.

Power budget
------------
``c`` is calibrated so the *expected* adversarial power per sensor under a random
configuration equals ``alpha`` times the clean sensed power ``P_clean``:

    E_phi[ |delta_s|^2 ] = c^2 * sum_k A[s,k]^2      (random phi, cross terms vanish)
    mean_s E_phi[...] = alpha * P_clean      =>      c = sqrt(alpha*P_clean / S2)

with ``S2 = mean_s sum_k A[s,k]^2``. So ``alpha`` (swept in dB) is the honest
"relative interference power of the rogue surface in a random config"; an
optimised ``phi`` then shows how much *coherent* damage structure adds on top at
the same physical surface strength ``c``. The matched control in
:meth:`random_phi` uses the same ``c``, so it is the correct same-power,
same-geometry baseline.
"""
import numpy as np
import torch

from data import SENSOR_IDX, partial_from_full

TWO_PI = 2.0 * np.pi


def _sensor_positions(sensor_idx=SENSOR_IDX, ris_spacing=0.5):
    """(16, 3) sensor positions, flattened in the same [a, b] row-major order as
    ``data.partial_from_full`` (a over rows SENSOR_IDX, b over cols SENSOR_IDX)."""
    idx = torch.as_tensor(sensor_idx, dtype=torch.float32)
    rows = idx[:, None].expand(len(idx), len(idx))      # a -> row
    cols = idx[None, :].expand(len(idx), len(idx))       # b -> col
    x = cols.reshape(-1) * ris_spacing                   # col -> x
    y = rows.reshape(-1) * ris_spacing                   # row -> y
    z = torch.zeros_like(x)
    return torch.stack([x, y, z], dim=-1)                # (16, 3)


def _surface_positions(n_adv, spacing, center):
    """(M, 3) adversarial element positions, planar grid in the x-z plane."""
    a = (torch.arange(n_adv, dtype=torch.float32) - (n_adv - 1) / 2.0) * spacing
    gx, gz = torch.meshgrid(a, a, indexing="ij")
    p = torch.stack([gx.reshape(-1), torch.zeros(n_adv * n_adv), gz.reshape(-1)], dim=-1)
    return p + torch.as_tensor(center, dtype=torch.float32)


class AdversarialSurface:
    """Differentiable ``delta(phi)`` for a rogue passive surface (see module docstring).

    Parameters
    ----------
    n_adv : int           side length; the surface has M = n_adv**2 elements.
    spacing : float       adversarial element spacing, in lambda (default 0.5).
    center : (3,) float   surface centre, in lambda, relative to RIS element (0,0).
    incidence : (3,)      unit direction of the arriving pilot plane wave.
    sensor_idx : list     legitimate sensing-element row/col indices.
    ris_spacing : float   legitimate RIS element spacing, in lambda.
    near_field : bool     True -> 1/d amplitude spreading to each sensor; False -> equal.
    """

    def __init__(self, n_adv=8, spacing=0.5, center=(10.0, 8.0, 2.0),
                 incidence=(0.0, -1.0, 0.0), sensor_idx=SENSOR_IDX, ris_spacing=0.5,
                 near_field=True, device="cpu", dtype=torch.complex64):
        self.n_adv = int(n_adv)
        self.M = self.n_adv * self.n_adv
        self.device = device
        self.dtype = dtype
        self.sensor_idx = list(sensor_idx)
        self.near_field = near_field

        q = _sensor_positions(sensor_idx, ris_spacing).to(device)          # (16, 3)
        p = _surface_positions(self.n_adv, spacing, center).to(device)      # (M, 3)
        u = torch.as_tensor(incidence, dtype=torch.float32, device=device)
        u = u / u.norm().clamp_min(1e-9)

        d = torch.cdist(q, p)                                               # (16, M) in lambda
        kappa = TWO_PI * (p @ u)                                            # (M,)
        amp = (1.0 / d.clamp_min(1e-6)) if near_field else torch.ones_like(d)
        phase = kappa[None, :] + TWO_PI * d                                 # (16, M)
        D = amp * torch.exp(1j * phase.to(torch.float64))                   # (16, M) complex128
        self.D = D.to(dtype)                                               # fixed dictionary
        self.S2 = float((amp.pow(2).sum(dim=1)).mean().item())             # mean_s sum_k A^2
        self.c = None                                                      # set by calibrate()

    # ------------------------------------------------------------------ calibration
    def calibrate(self, hp_clean, alpha_db=-10.0):
        """Set the surface strength ``c`` from a batch of clean sensed CSI.

        hp_clean : (..., 4, 4) or (..., 16) complex   clean sensed samples.
        alpha_db : float                              budget in dB (relative power).
        Returns the scalar ``c`` and also stores it on the surface.
        """
        hp = torch.as_tensor(hp_clean)
        p_clean = hp.abs().pow(2).mean().item()
        alpha = 10.0 ** (alpha_db / 10.0)
        self.c = float(np.sqrt(alpha * p_clean / max(self.S2, 1e-12)))
        self.alpha_db = alpha_db
        self.p_clean = p_clean
        return self.c

    # ------------------------------------------------------------------ forward model
    def delta(self, phi):
        """delta(phi) added to the sensed (4,4) grid.

        phi : (..., M) real   per-element phases (radians).
        returns (..., 4, 4) complex, aligned with ``data.partial_from_full``.
        """
        if self.c is None:
            raise RuntimeError("call calibrate() before delta()")
        w = torch.exp(1j * phi.to(torch.float32))                 # (..., M) unit modulus
        w = w.to(self.D.dtype)
        d = self.c * torch.einsum("sk,...k->...s", self.D, w)      # (..., 16)
        return d.reshape(*d.shape[:-1], 4, 4)

    def delivered_power(self, phi):
        """Mean |delta|^2 per sensor for configuration ``phi`` (for power-matched reporting)."""
        return self.delta(phi).abs().pow(2).mean().item()

    # ------------------------------------------------------------------ random control
    def random_phi(self, batch_shape=(), generator=None):
        """Matched random-phase control: phi ~ U[0, 2*pi), same surface strength ``c``."""
        shape = tuple(batch_shape) + (self.M,)
        return TWO_PI * torch.rand(shape, generator=generator, device=self.device)

    def zero_phi(self, batch_shape=()):
        return torch.zeros(tuple(batch_shape) + (self.M,), device=self.device)


def apply_delta(h_full, delta, sensor_idx=SENSOR_IDX):
    """Clean sensed grid from a full channel, plus the adversarial term.

    h_full : (..., 36, 36) complex ground-truth channel.
    delta  : (..., 4, 4) complex adversarial contribution from :meth:`AdversarialSurface.delta`.
    returns the corrupted sensed grid (..., 4, 4) complex = partial_from_full(h) + delta.
    """
    return partial_from_full(h_full, sensor_idx) + delta
