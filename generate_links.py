"""Build explicit-geometry malicious-RIS links for the RISnet 3.5 GHz scenario.

Run from the existing ris_adv folder on Pine (where torch loads .pt files):

  python generate_links.py --data_dir ../data \
      --out ../data/malicious_links_geom_v0.npz

The generated links are EXPLORATORY: the exact RIS panel orientation and the
coherent phase origin of the published DeepMIMO data remain unverified. The
benchmark refuses to label these links calibrated without further checks.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from link_model import LinkGeometry, build_reflective_links


def load_positions(path):
    path = Path(path)
    if path.suffix == ".npy":
        positions = np.load(path, allow_pickle=False)
    elif path.suffix == ".pt":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is needed only to load RISnet .pt locations; run on Pine") from exc
        positions = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
    else:
        raise ValueError("locations must be .pt or .npy files")
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected (N,3) user locations, got {positions.shape}")
    return positions


def diagnostic_against_reference(G_ar, g_test, reference_channel_path, max_users=128):
    """Magnitude-only diagnostic, NOT a coherent-phase calibration."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect RISnet channel .pt files") from exc
    full = torch.load(reference_channel_path, map_location="cpu", weights_only=False)
    if hasattr(full, "detach"):
        full = full.detach().cpu().numpy()
    n = min(max_users, len(full), len(g_test))
    if n < 1:
        raise ValueError("reference dataset has no channel samples")
    base = np.asarray(full[:n]).reshape(n, 1296)
    rng = np.random.default_rng(0)
    phi = rng.choice(np.array([-1., 1.]), size=G_ar.shape[1]).astype(np.complex64)
    delta = (g_test[:n] * phi) @ G_ar.T
    ratio = ((np.abs(delta) ** 2).sum(axis=1) /
             np.maximum((np.abs(base) ** 2).sum(axis=1), 1e-30))
    return {
        "median_absolute_h0": float(np.median(np.abs(base))),
        "median_absolute_added_path": float(np.median(np.abs(delta))),
        "median_random_phase_added_path_power_db_relative_to_h0": float(
            10 * np.log10(max(float(np.median(ratio)), 1e-30))),
        "reference_users": n,
        "note": "Magnitude check only; does not verify complex phase reference or channel normalization",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_locations", default=None,
                    help="override training locations .pt/.npy for validation")
    ap.add_argument("--test_locations", default=None,
                    help="override testing locations .pt/.npy for validation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frequency_hz", type=float, default=3.5e9)
    ap.add_argument("--legitimate_center", type=float, nargs=3,
                    default=(278.42, 576.97, 2.0))
    ap.add_argument("--legitimate_normal", type=float, nargs=3, default=None,
                    help="freeze the legitimate RIS orientation from train-only sweep")
    ap.add_argument("--attacker_rows", type=int, default=8)
    ap.add_argument("--attacker_cols", type=int, default=8)
    ap.add_argument("--attacker_spacing_lambda", type=float, default=0.5)
    ap.add_argument("--forward_m", type=float, default=4.0)
    ap.add_argument("--lateral_m", type=float, default=8.0)
    ap.add_argument("--height_offset_m", type=float, default=0.0)
    ap.add_argument("--attacker_normal", type=float, nargs=3, default=None,
                    help="explicit unit normal of vertical malicious panel (from placement sweep)")
    ap.add_argument("--efficiency", type=float, default=1.0)
    ap.add_argument("--phase_offset_rad", type=float, default=0.0)
    ap.add_argument("--reference_channels", default=None,
                    help="optional RISnet TEST .pt for magnitude diagnostics")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing link file: {out}")
    train_loc = load_positions(args.train_locations or
                               Path(args.data_dir) / "locations_training.pt")
    test_loc = load_positions(args.test_locations or
                              Path(args.data_dir) / "locations_testing.pt")
    geometry = LinkGeometry(
        frequency_hz=args.frequency_hz,
        legitimate_center_xyz=tuple(args.legitimate_center),
        legitimate_normal_xyz=(tuple(args.legitimate_normal)
                               if args.legitimate_normal is not None else None),
        malicious_rows=args.attacker_rows, malicious_cols=args.attacker_cols,
        malicious_spacing_lambda=args.attacker_spacing_lambda,
        attacker_forward_m=args.forward_m,
        attacker_lateral_m=args.lateral_m,
        attacker_height_offset_m=args.height_offset_m,
        malicious_normal_xyz=(tuple(args.attacker_normal)
                              if args.attacker_normal is not None else None),
        reflection_efficiency=args.efficiency,
        relative_phase_offset_rad=args.phase_offset_rad)

    G_ar, gt, ge, info = build_reflective_links(train_loc, test_loc, geometry)
    meta = {
        "provenance": "first-order spherical free-space two-hop model; see link_model.py",
        "carrier_frequency_hz": args.frequency_hz,
        "units_aligned_with_h0": False,
        "phase_reference_verified": False,
        "geometry_is_assumed": True,
        "calibration": "NOT verified against source DeepMIMO ray tracing phase convention",
        "generator_parameters": asdict(geometry),
        **info,
    }
    if args.reference_channels:
        meta["magnitude_diagnostic"] = diagnostic_against_reference(
            G_ar, ge, args.reference_channels)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, G_ar=G_ar, g_ua_training=gt, g_ua_testing=ge,
                        metadata=json.dumps(meta))
    print(f"Wrote {out} | G_ar {G_ar.shape}, training {gt.shape}, testing {ge.shape}")
    print(f"legitimate RIS: {info['legitimate_center_xyz']}; "
          f"attacker: {info['malicious_center_xyz']}; "
          f"separation {info['separation_between_centers_m']:.2f} m")
    print(f"Users on rear side: train {info['rear_side_training_users']}, "
          f"test {info['rear_side_testing_users']}")
    if args.reference_channels:
        print(json.dumps(meta["magnitude_diagnostic"], indent=2))
    print("UNVERIFIED geometry/phase alignment. For exploratory evaluation only.")


if __name__ == "__main__":
    main()
