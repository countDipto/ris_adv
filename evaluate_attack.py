"""Reproducible universal malicious-RIS benchmark on the existing RISnet split.

Requires an independently generated, physically calibrated ``links.pt`` (see
ATTACK_SPEC.md). Refuses to substitute random complex link tensors. This stage
reports CSI metrics only; an oracle-vs-DL rate comparison needs the future WSR
controller and must not be inferred from NMSE.

Run from ris_adv/:
  python evaluate_attack.py --data_dir ../data --links ../data/malicious_links.pt \
      --victim runs/dsrnet_snr.pt --attacker runs/dncnn_snr.pt \
      --phase_bits 1 --steps 150 --seed 0
"""

import argparse
import json
from pathlib import Path

import torch

from attack import (AttackConfig, channel_with_malicious_ris,
                    optimize_malicious_ris, predict_channel, random_phases)
from data import load_pt, load_split
from metrics import complex_corr, nmse_db, psnr_db
from models import build


def load_estimator(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build(ck["model"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def load_links(path, n_train, n_test):
    links = torch.load(path, map_location="cpu", weights_only=False)
    required = ("G_ar", "g_ua_training", "g_ua_testing", "metadata")
    missing = [key for key in required if key not in links]
    if missing:
        raise ValueError(f"Missing link-bundle fields: {missing}. See ATTACK_SPEC.md")
    meta = links["metadata"]
    if not isinstance(meta, dict) or meta.get("units_aligned_with_h0") is not True:
        raise ValueError("Link metadata must explicitly affirm units_aligned_with_h0=True")
    if not meta.get("provenance") or not meta.get("carrier_frequency_hz"):
        raise ValueError("Link metadata needs provenance and carrier_frequency_hz")
    G = links["G_ar"]
    gt, ge = links["g_ua_training"], links["g_ua_testing"]
    if gt.shape[0] != n_train or ge.shape[0] != n_test:
        raise ValueError("g_ua arrays must follow the original training/testing user order")
    if G.ndim != 2 or G.shape[0] != 1296 or G.shape[1] != gt.shape[1] or G.shape[1] != ge.shape[1]:
        raise ValueError("This harness expects one fixed G_ar of shape (1296,A)")
    for name, value in (("G_ar", G), ("g_ua_training", gt), ("g_ua_testing", ge)):
        if not isinstance(value, torch.Tensor) or not value.is_complex():
            raise TypeError(f"{name} must be a complex torch tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite entries")
    return G, gt, ge, meta


@torch.no_grad()
def evaluate_condition(model, ck, h0, G, g, phase_vector, is_far, snr, device,
                       batch_size=128, seed=123):
    """Evaluate model against the true physical channel under common pilot noise."""
    true_all, pred_all, change_all = [], [], []
    for start in range(0, h0.shape[0], batch_size):
        end = min(start + batch_size, h0.shape[0])
        base = h0[start:end].to(device=device, dtype=torch.complex64)
        if phase_vector is None:
            actual = base
        else:
            actual = channel_with_malicious_ris(
                base, G.to(device=device, dtype=torch.complex64),
                g[start:end].to(device=device, dtype=torch.complex64),
                phase_vector.to(device=device, dtype=torch.complex64))
        predicted = predict_channel(model, actual, ck["scale"],
                                    ck.get("input_repr", "ap"), snr,
                                    seed=seed + start)
        true_all.append(actual.cpu())
        pred_all.append(predicted.cpu())
        numerator = (actual - base).abs().square().flatten(1).sum(1)
        denominator = base.abs().square().flatten(1).sum(1).clamp_min(1e-30)
        change_all.append((numerator / denominator).cpu())

    h = torch.cat(true_all)
    h_hat = torch.cat(pred_all)
    change = torch.cat(change_all)
    out = {}
    for region, mask in (("trained_area", ~is_far), ("far", is_far)):
        if not mask.any():
            raise ValueError(f"No users in {region}; verify the location split")
        out[region] = {
            "n_users": int(mask.sum()),
            "nmse_db": float(nmse_db(h_hat[mask], h[mask])),
            "rho": float(complex_corr(h_hat[mask], h[mask])),
            "psnr_db": float(psnr_db(h_hat[mask], h[mask])),
            "physical_change_power_ratio_db": (
                None if phase_vector is None else float(
                    10 * torch.log10(change[mask].mean().clamp_min(1e-30)))),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--links", required=True, help="physically calibrated links.pt")
    parser.add_argument("--victim", default="runs/dsrnet_snr.pt")
    parser.add_argument("--attacker", default=None,
                        help="surrogate checkpoint, or omit for white-box victim gradients")
    parser.add_argument("--phase_bits", type=int, default=1)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--eot_samples", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--pilot_snr_db", type=float, default=20)
    parser.add_argument("--opt_users", type=int, default=256,
                        help="number of training users to optimize universal phases on")
    parser.add_argument("--eval_batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None,
                        help="defaults to runs/attack/universal_seed<seed>.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    victim, victim_ck = load_estimator(args.victim, device)
    attacker, attacker_ck = load_estimator(args.attacker or args.victim, device)

    train_h = load_split(args.data_dir, "training")
    test_h = load_split(args.data_dir, "testing")
    G, g_train, g_test, meta = load_links(args.links, len(train_h), len(test_h))
    loc_train = load_pt(str(Path(args.data_dir) / "locations_training.pt"))
    loc_test = load_pt(str(Path(args.data_dir) / "locations_testing.pt"))
    frac = victim_ck.get("args", {}).get("holdout_frac", 0.10)
    x_train = torch.as_tensor(loc_train)[:, 0]
    cut = x_train.min() + (1 - frac) * (x_train.max() - x_train.min())
    is_far = torch.as_tensor(loc_test)[:, 0] > cut

    train_ids = torch.as_tensor(victim_ck["split"]["train"], dtype=torch.long)
    gen = torch.Generator().manual_seed(args.seed)
    n_opt = min(args.opt_users, len(train_ids))
    train_ids = train_ids[torch.randperm(len(train_ids), generator=gen)[:n_opt]]
    if n_opt < 1:
        raise ValueError("opt_users must be positive")

    cfg = AttackConfig(steps=args.steps, lr=args.lr, restarts=args.restarts,
                       phase_bits=args.phase_bits, pilot_snr_db=args.pilot_snr_db,
                       eot_samples=args.eot_samples, universal=True, seed=args.seed)
    optimized = optimize_malicious_ris(
        attacker, train_h[train_ids].reshape(-1, 36, 36), G,
        g_train[train_ids], attacker_ck["scale"],
        attacker_ck.get("input_repr", "ap"), cfg)
    random_control = random_phases(G.shape[1], args.phase_bits,
                                   args.seed + 100, device=device)
    results = {}
    for condition, phase in (("no_attack", None),
                             ("random_control", random_control),
                             ("optimized", optimized.phi)):
        for snr in (None, 30., 20., 10.):
            label = "noiseless" if snr is None else f"{snr:g}dB"
            results[f"{condition}@{label}"] = evaluate_condition(
                victim, victim_ck, test_h.reshape(-1, 36, 36), G, g_test,
                phase, is_far, snr, device, args.eval_batch)

    payload = {
        "name": "universal_malicious_ris", "victim": Path(args.victim).stem,
        "optimizer_model": Path(args.attacker or args.victim).stem,
        "config": vars(args), "n_attack_elements": G.shape[1],
        "optimize_train_users": n_opt, "heldout_test_users": len(test_h),
        "link_metadata": meta,
        "phase_angles_radians": optimized.phi.angle().cpu().tolist(),
        "random_control_angles_radians": random_control.angle().cpu().tolist(),
        "optimization_nmse_linear": float(optimized.nmse_linear.mean()),
        "results": results,
        "important": "CSI-only results; oracle-rate penalty and receiver rate not yet implemented",
    }
    destination = Path(args.out or f"runs/attack/universal_seed{args.seed}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved {destination} | optimized on {n_opt} TRAIN users, evaluated on TEST")
    for condition in ("no_attack", "random_control", "optimized"):
        values = results[f"{condition}@20dB"]
        trained = values["trained_area"]["nmse_db"]
        far = values["far"]["nmse_db"]
        print(f"{condition:16s} 20dB NMSE: trained {trained:.2f}, far {far:.2f} dB")


if __name__ == "__main__":
    main()
