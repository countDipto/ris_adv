"""TRAIN-ONLY placement/orientation sweep for a passive malicious RIS.

Ranks hypothetical vertical-facade placements by a deliberately optimistic
upper bound on the change to DSRnet's 16 sensed channel coefficients. It does
NOT optimize DSRnet or inspect the released testing / spatial far sets.

Run on Pine from ris_adv/:
    python sweep_geometry.py --data_dir ../data --checkpoint runs/dsrnet_snr.pt

The output prints a ready-to-run ``generate_links.py`` command for the chosen
candidate. Actual buildings/wall availability and RISnet coherent phase
alignment are not yet verified; do not treat these as deployable sites.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from generate_links import load_positions
from link_model import LinkGeometry, SPEED_OF_LIGHT, infer_street_direction, unit, vertical_upa


# A geometry-only example for standalone tests. Actual experiment indices are
# derived below from the project's data.partial_from_full implementation.
SENSOR_IDX = np.array([r * 36 + c for r in (4, 13, 22, 31)
                       for c in (4, 13, 22, 31)], dtype=np.int64)


@dataclass(frozen=True)
class Candidate:
    rows: int
    cols: int
    forward_m: float
    lateral_m: float
    rotation_degrees: float
    normal_xyz: tuple


def load_training_samples(data_dir, checkpoint, n_select, n_validate, seed):
    """Read ONLY the frozen victim checkpoint's train IDs, never val/far/test."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required on Pine to read RISnet .pt channels") from exc
    # Do not assume which 16 sensing elements the victim uses: reproduce the
    # exact selection operator from the existing project source code.
    from data import partial_from_full
    tag = torch.arange(1296, dtype=torch.float32).reshape(1, 36, 36)
    sensor_values = partial_from_full(tag.to(torch.complex64)).reshape(-1)
    sensor_idx = np.asarray(sensor_values.real.detach().cpu().numpy(), dtype=np.int64)
    if (sensor_idx.size != 16 or len(np.unique(sensor_idx)) != 16 or
            not np.all((0 <= sensor_idx) & (sensor_idx < 1296)) or
            not np.allclose(sensor_values.imag.detach().cpu().numpy(), 0) or
            not np.allclose(sensor_values.real.detach().cpu().numpy(), sensor_idx)):
        raise ValueError("partial_from_full is not a simple 16-element selector; "
                         "the sweep needs its exact measurement operator")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ids = np.asarray(ck["split"]["train"], dtype=np.int64)
    if len(ids) < n_select + n_validate:
        raise ValueError("Requested more users than the victim's train split contains")
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    select_ids, validation_ids = ids[:n_select], ids[n_select:n_select + n_validate]

    locations = load_positions(Path(data_dir) / "locations_training.pt")
    if ids.max() >= len(locations):
        raise ValueError("Checkpoint split indices do not match training locations")
    raw = torch.load(Path(data_dir) / "channels_ris_rx_training.pt",
                     map_location="cpu", weights_only=False)
    if isinstance(raw, torch.Tensor):
        selected = raw[torch.as_tensor(select_ids)].reshape(-1, 1296).numpy()
        validated = raw[torch.as_tensor(validation_ids)].reshape(-1, 1296).numpy()
    else:
        raw = np.asarray(raw).reshape(-1, 1296)
        selected, validated = raw[select_ids], raw[validation_ids]
    if raw.shape[0] != len(locations):
        raise ValueError("Training channel count differs from training locations")
    del raw
    return (locations[select_ids], np.asarray(selected[:, sensor_idx], dtype=np.complex64),
            locations[validation_ids], np.asarray(validated[:, sensor_idx], dtype=np.complex64),
            locations[ids], sensor_idx)


def _rotate_horizontal(v, angle_deg):
    a = math.radians(angle_deg)
    x, y = v[:2]
    return unit([math.cos(a) * x - math.sin(a) * y,
                 math.sin(a) * x + math.cos(a) * y, 0.])


def candidate_geometry(center_l, normal_l, median_train_xyz, forward, lateral,
                       rotation_deg, rows, cols):
    lateral_axis = unit(np.cross(np.array([0., 0., 1.]), normal_l))
    attacker_center = center_l + forward * normal_l + lateral * lateral_axis
    to_legitimate = unit(center_l - attacker_center)
    to_users = unit(median_train_xyz - attacker_center)
    bisector = to_legitimate + to_users
    if np.linalg.norm(bisector) < 1e-4:
        bisector = -np.sign(lateral) * lateral_axis
    normal_a = _rotate_horizontal(unit([bisector[0], bisector[1], 0.]), rotation_deg)
    candidate = Candidate(rows, cols, forward, lateral, rotation_deg,
                          tuple(float(v) for v in normal_a))
    return attacker_center, normal_a, candidate


def optimistic_sensing_bound(legitimate_sensor_xyz, attacker_xyz, normal_l,
                             normal_a, users_xyz, sensed_h, wavelength,
                             reflection_efficiency=1.0):
    """Per-user upper bound: each sensor may independently align ALL RIS phases.

    The common phase vector of a real RIS cannot generally achieve all these
    sensor-specific maxima simultaneously, so this overestimates the attacker.
    Only distances and front-side cosines matter for the bound (phase-free).
    """
    leg_to_att = attacker_xyz[None, :, :] - legitimate_sensor_xyz[:, None, :]
    d = np.linalg.norm(leg_to_att, axis=2)
    directions = leg_to_att / d[:, :, None]
    cos_leg = np.clip(np.einsum("sac,c->sa", directions, normal_l), 0., 1.)
    cos_att = np.clip(np.einsum("sac,c->sa", -directions, normal_a), 0., 1.)
    G_abs = (wavelength / (4 * np.pi * d) * np.sqrt(cos_leg * cos_att) *
             reflection_efficiency)
    coupled_elements = (G_abs > 0).any(axis=0)

    upper_power = np.empty(len(users_xyz), dtype=np.float64)
    rear = 0
    for start in range(0, len(users_xyz), 512):
        chunk = users_xyz[start:start + 512]
        att_to_user = chunk[:, None, :] - attacker_xyz[None, :, :]
        du = np.linalg.norm(att_to_user, axis=2)
        dir_u = att_to_user / du[:, :, None]
        cosine = np.clip(np.einsum("bac,c->ba", dir_u, normal_a), 0., 1.)
        if coupled_elements.any():
            rear += int(np.count_nonzero(
                cosine[:, coupled_elements].max(axis=1) <= 1e-8))
        else:
            rear += len(chunk)
        g_abs = wavelength / (4 * np.pi * du) * np.sqrt(cosine)
        # Every sensing element receives its own independent phase-alignment
        # advantage, yielding a safe upper bound on the physical contribution.
        upper_amp = g_abs @ G_abs.T
        upper_power[start:start + len(chunk)] = (upper_amp ** 2).sum(axis=1)

    baseline_power = np.maximum((np.abs(sensed_h).astype(np.float64) ** 2).sum(1), 1e-30)
    bound_db = 10 * np.log10(np.maximum(upper_power / baseline_power, 1e-30))
    return {
        "coverage": (len(users_xyz) - rear) / len(users_xyz),
        "rear_users": rear,
        "p10_db": float(np.percentile(bound_db, 10)),
        "median_db": float(np.percentile(bound_db, 50)),
        "p90_db": float(np.percentile(bound_db, 90)),
        "pct_above_minus20db": float(100 * np.mean(bound_db > -20.)),
    }


def evaluate_candidate(candidate, pos, sensed_h, center_l, normal_l,
                       element_pos_l, lam, efficiency):
    lateral_axis = unit(np.cross(np.array([0., 0., 1.]), normal_l))
    center_a = (center_l + candidate.forward_m * normal_l +
                candidate.lateral_m * lateral_axis)
    normal_a = np.asarray(candidate.normal_xyz, dtype=np.float64)
    pos_a = vertical_upa(center_a, candidate.rows, candidate.cols,
                         0.5 * lam, normal_a)
    return optimistic_sensing_bound(element_pos_l, pos_a, normal_l, normal_a,
                                    pos, sensed_h, lam, efficiency)


def _rank_key(item, min_coverage):
    metric = item["validation"] if "validation" in item else item["selection"]
    eligible = metric["coverage"] >= min_coverage
    if eligible:
        return (1, metric["median_db"], metric["p10_db"], metric["coverage"])
    return (0, metric["coverage"], metric["median_db"], metric["p10_db"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", default="runs/dsrnet_snr.pt")
    ap.add_argument("--out", default="runs/attack/geometry_sweep.json")
    ap.add_argument("--n_select", type=int, default=512)
    ap.add_argument("--n_validate", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_coverage", type=float, default=0.90)
    ap.add_argument("--forward_m", type=float, nargs="+", default=[2., 4., 8.])
    ap.add_argument("--lateral_m", type=float, nargs="+",
                    default=[-12., -8., -4., -2., 2., 4., 8., 12.])
    ap.add_argument("--normal_rotations_deg", type=float, nargs="+",
                    default=[-45., 0., 45.])
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16],
                    help="square attacker side lengths, in elements")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not (0 < args.min_coverage <= 1) or args.n_select <= 0 or args.n_validate <= 0:
        raise ValueError("Require min_coverage in (0,1], and positive sample sizes")
    if args.top_k < 1 or any(size < 1 for size in args.sizes):
        raise ValueError("Require positive top_k and attacker array sizes")
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {out}; pass --overwrite deliberately")

    cfg = LinkGeometry()
    cfg.validate()
    (sel_pos, sel_h, val_pos, val_h, all_train_pos, sensor_idx) = load_training_samples(
        args.data_dir, args.checkpoint, args.n_select, args.n_validate, args.seed)
    lam = SPEED_OF_LIGHT / cfg.frequency_hz
    center_l = np.asarray(cfg.legitimate_center_xyz, dtype=np.float64)
    normal_l = infer_street_direction(center_l, all_train_pos)
    median_train = np.median(all_train_pos, axis=0)
    legitimate_sensors = vertical_upa(
        center_l, cfg.legitimate_rows, cfg.legitimate_cols,
        cfg.legitimate_spacing_lambda * lam, normal_l)[sensor_idx]

    candidates = []
    for size in args.sizes:
        for forward in args.forward_m:
            if forward <= 0:
                raise ValueError("forward distances must be positive for front-side coupling")
            for lateral in args.lateral_m:
                if lateral == 0:
                    raise ValueError("lateral offsets cannot be zero")
                for rot in args.normal_rotations_deg:
                    _, _, c = candidate_geometry(center_l, normal_l, median_train,
                                                 forward, lateral, rot, size, size)
                    metrics = evaluate_candidate(c, sel_pos, sel_h, center_l,
                                                 normal_l, legitimate_sensors, lam,
                                                 cfg.reflection_efficiency)
                    candidates.append({"candidate": asdict(c), "selection": metrics})
    candidates.sort(key=lambda x: _rank_key(x, args.min_coverage), reverse=True)

    # Only finalists are evaluated on disjoint training-only validation users.
    finalists = candidates[:max(args.top_k, 1)]
    for item in finalists:
        c = Candidate(**item["candidate"])
        item["validation"] = evaluate_candidate(
            c, val_pos, val_h, center_l, normal_l, legitimate_sensors, lam,
            cfg.reflection_efficiency)
    finalists.sort(key=lambda x: _rank_key(x, args.min_coverage), reverse=True)
    chosen = finalists[0]

    output = {
        "method": "train-only loose elementwise triangle-inequality upper bound",
        "not_publication_ready": True,
        "limitations": ["hypothetical free-space facades", "unknown exact RIS orientation",
                        "unverified coherent phase and amplitude reference",
                        "upper bound ignores one common 1-bit phase vector"],
        "config": vars(args),
        "victim_train_ids_only": True,
        "victim_sensing_element_indices_row_major": sensor_idx.tolist(),
        "n_selection_users": len(sel_pos), "n_validation_users": len(val_pos),
        "legitimate_ris_center": center_l.tolist(),
        "legitimate_ris_normal_assumed": normal_l.tolist(),
        "frequency_hz": cfg.frequency_hz,
        "candidate_count": len(candidates),
        "chosen": chosen, "finalists": finalists,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Searched {len(candidates)} placements using only victim TRAIN users")
    print("Rank  Size  Forward  Lateral  Rotation  Coverage  Median upper  p10 upper  >-20dB")
    for i, item in enumerate(finalists, 1):
        c, m = item["candidate"], item["validation"]
        print(f"{i:4d} {c['rows']:5d} {c['forward_m']:8.1f} {c['lateral_m']:8.1f} "
              f"{c['rotation_degrees']:9.0f} {100*m['coverage']:8.1f}% "
              f"{m['median_db']:11.2f} {m['p10_db']:10.2f} "
              f"{m['pct_above_minus20db']:8.1f}%")
    c = chosen["candidate"]
    print(f"Saved {out}; chosen TRAIN-only candidate:")
    if chosen["validation"]["coverage"] < args.min_coverage:
        print("WARNING: No finalist meets the specified coverage threshold; "
              "do not assume this geometry can produce a useful attack.")
    if chosen["validation"]["pct_above_minus20db"] == 0:
        print("WARNING: Even this generous bound is below -20 dB for all sampled "
              "train-validation users; attack optimization may be unproductive.")
    print("python generate_links.py --data_dir ../data "
          "--out ../data/malicious_links_sweep_selected.npz "
          f"--attacker_rows {c['rows']} --attacker_cols {c['cols']} "
          f"--forward_m {c['forward_m']} --lateral_m {c['lateral_m']} "
          "--legitimate_normal " + " ".join(f"{x:.9f}" for x in normal_l) + " "
          "--attacker_normal " + " ".join(f"{x:.9f}" for x in c["normal_xyz"]))


if __name__ == "__main__":
    main()
