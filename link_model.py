"""First-order, geometry-derived links for an additional *reflective* malicious RIS.

Numerical geometry uses NumPy only. All distances are in meters. The model
uses spherical free-space voltage responses and a simple cosine half-space
element pattern; it is NOT a ray-tracing replacement or a mutual-coupling EM
solver. No data-dependent rescaling is performed.
"""

from dataclasses import dataclass
import math

import numpy as np


SPEED_OF_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class LinkGeometry:
    frequency_hz: float = 3.5e9
    legitimate_center_xyz: tuple = (278.42, 576.97, 2.0)
    legitimate_rows: int = 36
    legitimate_cols: int = 36
    legitimate_spacing_lambda: float = 0.25
    malicious_rows: int = 8
    malicious_cols: int = 8
    malicious_spacing_lambda: float = 0.5
    attacker_forward_m: float = 4.0
    attacker_lateral_m: float = 8.0
    attacker_height_offset_m: float = 0.0
    reflection_efficiency: float = 1.0  # idealized amplitude efficiency
    relative_phase_offset_rad: float = 0.0

    @property
    def wavelength_m(self):
        return SPEED_OF_LIGHT / self.frequency_hz

    def validate(self):
        if self.frequency_hz <= 0 or not math.isfinite(self.frequency_hz):
            raise ValueError("frequency_hz must be finite and positive")
        if any(v <= 0 for v in (self.legitimate_rows, self.legitimate_cols,
                                self.malicious_rows, self.malicious_cols)):
            raise ValueError("array dimensions must be positive")
        if self.legitimate_rows * self.legitimate_cols != 1296:
            raise ValueError("victim DSRnet requires 1296 legitimate RIS elements")
        if self.legitimate_spacing_lambda <= 0 or self.malicious_spacing_lambda <= 0:
            raise ValueError("array spacings must be positive")
        if not (0 < self.reflection_efficiency <= 1):
            raise ValueError("reflection_efficiency must lie in (0,1]")
        if self.attacker_lateral_m == 0:
            raise ValueError("lateral offset must be nonzero for the default reflector geometry")


def unit(v):
    x = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(x)
    if n < 1e-9 or not np.isfinite(n):
        raise ValueError("cannot normalize a zero/non-finite direction")
    return x / n


def infer_street_direction(legitimate_center_xyz, user_positions_xyz):
    """Assumption: face legitimate RIS horizontally toward median user position."""
    users = np.asarray(user_positions_xyz, dtype=np.float64)
    if users.ndim != 2 or users.shape[1] != 3 or len(users) == 0:
        raise ValueError("user locations must be (N,3)")
    center = np.asarray(legitimate_center_xyz, dtype=np.float64)
    displacement = np.median(users, axis=0) - center
    displacement[2] = 0  # vertical facade array
    return unit(displacement)


def vertical_upa(center_xyz, rows, cols, spacing_m, outward_normal_xyz):
    """Return row-major positions: row index vertical, column horizontal."""
    normal = unit(outward_normal_xyz)
    if abs(normal[2]) > 1e-7:
        raise ValueError("this initial generator supports vertical facades only")
    up = np.array([0.0, 0.0, 1.0])
    horizontal = unit(np.cross(up, normal))
    vertical = unit(np.cross(normal, horizontal))
    cc = (np.arange(cols) - (cols - 1) / 2) * spacing_m
    rr = (np.arange(rows) - (rows - 1) / 2) * spacing_m
    grid = (np.asarray(center_xyz)[None, None, :] +
            rr[:, None, None] * vertical[None, None, :] +
            cc[None, :, None] * horizontal[None, None, :])
    return grid.reshape(rows * cols, 3)


def _isotropic_voltage_gain(distance_m, wavelength_m):
    if np.any(distance_m < 1e-5):
        raise ValueError("propagation distance too close to zero")
    k = 2 * np.pi / wavelength_m
    return (wavelength_m / (4 * np.pi * distance_m) *
            np.exp(-1j * k * distance_m))


def build_reflective_links(user_train_xyz, user_test_xyz, cfg=LinkGeometry()):
    """Create G_ar=(1296,A), g_train=(N,A), g_test=(T,A).

    Assumed placement: malicious panel offset forward toward the street and
    laterally from the legitimate panel; it faces back toward the street.
    This ensures both the legitimate RIS and typical street users can be on
    the same reflective side. Non-visible user/element links are attenuated
    to zero by the cosine half-space model (no STAR-RIS transmission).
    """
    cfg.validate()
    train = np.asarray(user_train_xyz, dtype=np.float64)
    test = np.asarray(user_test_xyz, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != 3 or test.shape[1] != 3:
        raise ValueError("training and test positions must both be (N,3)")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("user positions cannot contain non-finite values")

    lam = cfg.wavelength_m
    center_l = np.asarray(cfg.legitimate_center_xyz, dtype=np.float64)
    normal_l = infer_street_direction(center_l, train)
    lateral = unit(np.cross(np.array([0., 0., 1.]), normal_l))
    center_a = (center_l + cfg.attacker_forward_m * normal_l +
                cfg.attacker_lateral_m * lateral +
                np.array([0, 0, cfg.attacker_height_offset_m]))
    normal_a = -np.sign(cfg.attacker_lateral_m) * lateral

    legitimate_xyz = vertical_upa(
        center_l, cfg.legitimate_rows, cfg.legitimate_cols,
        cfg.legitimate_spacing_lambda * lam, normal_l)
    malicious_xyz = vertical_upa(
        center_a, cfg.malicious_rows, cfg.malicious_cols,
        cfg.malicious_spacing_lambda * lam, normal_a)

    # Element-to-element link from attacker -> legitimate RIS. Spherical phase
    # preserves near-field geometry; half-space pattern disallows back-face RF.
    leg_to_att = malicious_xyz[None, :, :] - legitimate_xyz[:, None, :]
    d_ar = np.linalg.norm(leg_to_att, axis=-1)
    dir_la = leg_to_att / d_ar[:, :, None]
    cos_legit = np.maximum(0., np.einsum("mac,c->ma", dir_la, normal_l))
    cos_att = np.maximum(0., np.einsum("mac,c->ma", -dir_la, normal_a))
    G_ar = (_isotropic_voltage_gain(d_ar, lam) *
            np.sqrt(cos_legit * cos_att) * cfg.reflection_efficiency *
            np.exp(1j * cfg.relative_phase_offset_rad)).astype(np.complex64)

    def user_to_attacker(users, batch_size=1024):
        g = np.empty((len(users), len(malicious_xyz)), dtype=np.complex64)
        backside = 0
        for start in range(0, len(users), batch_size):
            block = users[start:start + batch_size]
            att_to_user = block[:, None, :] - malicious_xyz[None, :, :]
            d = np.linalg.norm(att_to_user, axis=-1)
            directions = att_to_user / d[:, :, None]
            cos = np.maximum(0., np.einsum("bac,c->ba", directions, normal_a))
            backside += int(np.count_nonzero(cos.max(axis=1) <= 1e-8))
            g[start:start + len(block)] = (_isotropic_voltage_gain(d, lam) *
                                             np.sqrt(cos)).astype(np.complex64)
        return g, backside

    g_train, rear_train = user_to_attacker(train)
    g_test, rear_test = user_to_attacker(test)
    info = {
        "frequency_hz": cfg.frequency_hz,
        "wavelength_m": lam,
        "legitimate_center_xyz": center_l.tolist(),
        "legitimate_normal_xyz": normal_l.tolist(),
        "malicious_center_xyz": center_a.tolist(),
        "malicious_normal_xyz": normal_a.tolist(),
        "legitimate_spacing_m": cfg.legitimate_spacing_lambda * lam,
        "malicious_spacing_m": cfg.malicious_spacing_lambda * lam,
        "reflector_element_count": cfg.malicious_rows * cfg.malicious_cols,
        "rear_side_training_users": rear_train,
        "rear_side_testing_users": rear_test,
        "separation_between_centers_m": float(np.linalg.norm(center_a - center_l)),
    }
    return G_ar, g_train, g_test, info
