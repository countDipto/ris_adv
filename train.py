"""Train a partial-CSI -> full-CSI RIS channel estimator on the RISnet DeepMIMO data.

Example (V100):
    python train.py --data_dir ../data --model dsrnet --epochs 300 --pilot_snr_range 10 30
    python train.py --data_dir ../data --model dncnn  --epochs 300 --pilot_snr_range 10 30

Splits (see data.split_indices):
    train / val : random split of the training file, minus the far block
    far         : last 10% of the street's x-range, never trained on (unseen area)
    test        : the released test file (interleaved with the training grid)
The checkpoint is selected on val, not on test.

Outputs (in --out_dir): <name>.pt (weights, scale, config, split indices, final metrics),
<name>_log.csv (per epoch) and <name>_final.json (test/far NMSE at fixed pilot SNRs).
DSRnet paper reference: NMSE -13.77 dB, rho 0.934.
"""
import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from data import RISChannelDataset, load_split, load_pt, ri_to_complex, split_indices
from metrics import nmse_db, psnr_db, complex_corr
from models import build, n_params


def evaluate(model, ds, scale, device, snr="dataset", seed=123):
    """NMSE/PSNR/rho on ds. snr='dataset' uses the dataset's own pilot-noise setting."""
    base = ds.dataset if isinstance(ds, Subset) else ds
    old = base.pilot_snr_db
    if snr != "dataset":
        base.pilot_snr_db = snr
    torch.manual_seed(seed)                      # same noise draw every evaluation
    model.eval()
    hs, hh = [], []
    with torch.no_grad():
        for x, _, h in DataLoader(ds, batch_size=1024):
            hh.append(ri_to_complex(model(x.to(device)), scale).cpu())
            hs.append(h)
    base.pilot_snr_db = old
    h, h_hat = torch.cat(hs), torch.cat(hh)
    return {"nmse_db": nmse_db(h_hat, h).item(), "psnr_db": psnr_db(h_hat, h).item(),
            "rho": complex_corr(h_hat, h).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--variant", default="", help="'', '_5e-5' or '_iid'")
    ap.add_argument("--model", default="dsrnet", choices=["dsrnet", "dncnn", "srcnn"])
    ap.add_argument("--name", default=None, help="run name (default: model name)")
    ap.add_argument("--input_repr", default="ap", choices=["ap", "ri"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=9e-4)
    ap.add_argument("--pilot_snr_db", type=float, default=None, help="fixed training SNR")
    ap.add_argument("--pilot_snr_range", type=float, nargs=2, default=None,
                    help="train with SNR ~ U[lo, hi] dB per sample (overrides --pilot_snr_db)")
    ap.add_argument("--holdout_frac", type=float, default=0.10)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--eval_snrs", type=float, nargs="*", default=[30, 20, 10])
    ap.add_argument("--max_train", type=int, default=None, help="subset for smoke tests")
    ap.add_argument("--train_split", default="training",
                    help="use 'testing' only for a CPU smoke test without the large file")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    name = args.name or args.model

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    train_snr = tuple(args.pilot_snr_range) if args.pilot_snr_range else args.pilot_snr_db

    h_all = load_split(args.data_dir, args.train_split, args.variant)
    loc = load_pt(os.path.join(args.data_dir, f"locations_{args.train_split}.pt"))
    if args.max_train:
        h_all, loc = h_all[: args.max_train], loc[: args.max_train]
    sp = split_indices(loc, args.val_frac, args.holdout_frac, args.seed)

    full = RISChannelDataset(h_all, pilot_snr_db=train_snr, input_repr=args.input_repr)
    # normalisation scale from the training indices only
    full.scale = float(full.h[sp["train"]].abs().pow(2).mean().sqrt())
    train_ds, val_ds, far_ds = (Subset(full, sp[k]) for k in ("train", "val", "far"))
    test_ds = RISChannelDataset(load_split(args.data_dir, "testing", args.variant),
                                scale=full.scale, pilot_snr_db=train_snr,
                                input_repr=args.input_repr)
    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    model = build(args.model).to(device)
    print(f"{name} ({args.model}): {n_params(model):,} params | scale={full.scale:.3e} | "
          f"train={len(train_ds)} val={len(val_ds)} far={len(far_ds)} test={len(test_ds)} | "
          f"train SNR={train_snr} | device={device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ckpt = os.path.join(args.out_dir, f"{name}.pt")
    best = 1e9
    with open(os.path.join(args.out_dir, f"{name}_log.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_mse", "val_nmse_db", "far_nmse_db", "test_nmse_db",
                    "test_rho", "sec"])
        for ep in range(1, args.epochs + 1):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for x, y, _ in tr:
                x, y = x.to(device), y.to(device)
                loss = torch.mean((model(x) - y) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * x.shape[0]
                n += x.shape[0]
            sched.step()
            mv = evaluate(model, val_ds, full.scale, device)
            mf = evaluate(model, far_ds, full.scale, device) if len(far_ds) else {"nmse_db": float("nan")}
            mt = evaluate(model, test_ds, full.scale, device)
            w.writerow([ep, tot / n, mv["nmse_db"], mf["nmse_db"], mt["nmse_db"], mt["rho"],
                        time.time() - t0])
            fh.flush()
            if mv["nmse_db"] < best:
                best = mv["nmse_db"]
                torch.save({"state_dict": model.state_dict(), "model": args.model,
                            "scale": full.scale, "input_repr": args.input_repr,
                            "train_snr": train_snr, "split": {k: v.tolist() for k, v in sp.items()},
                            "epoch": ep, "args": vars(args)}, ckpt)
            if ep == 1 or ep % 10 == 0 or ep == args.epochs:
                print(f"ep {ep:4d} | train {tot / n:.4f} | val {mv['nmse_db']:.2f} | "
                      f"far {mf['nmse_db']:.2f} | test {mt['nmse_db']:.2f} dB | best val {best:.2f}")

    # final report with the selected checkpoint
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    final = {"epoch": state["epoch"], "test": {}, "far": {}}
    for snr in [None] + list(args.eval_snrs):
        key = "noiseless" if snr is None else f"{snr:g}dB"
        final["test"][key] = evaluate(model, test_ds, full.scale, device, snr=snr)
        if len(far_ds):
            final["far"][key] = evaluate(model, far_ds, full.scale, device, snr=snr)
    with open(os.path.join(args.out_dir, f"{name}_final.json"), "w") as fh:
        json.dump(final, fh, indent=2)
    print("final (selected on val):")
    for split in ("test", "far"):
        print(f"  {split}: " + " | ".join(f"{k} {v['nmse_db']:.2f} dB"
                                         for k, v in final[split].items()))


if __name__ == "__main__":
    main()
