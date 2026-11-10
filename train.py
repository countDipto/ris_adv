"""Train a partial-CSI -> full-CSI RIS channel estimator on the RISnet DeepMIMO data.

Example (V100):
    python train.py --data_dir C:/ris_smart_city_2/data --model dsrnet --epochs 300
    python train.py --data_dir ... --model dncnn --epochs 300    # surrogate / baseline

Outputs (in --out_dir): <model>.pt checkpoint (weights + scale + config), log.csv.
DSRnet paper reference: NMSE -13.77 dB, PSNR 26.39 dB, rho 0.934 (2 SAMs, 80,916 params).
"""
import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from data import RISChannelDataset, load_split, ri_to_complex
from metrics import nmse_db, psnr_db, complex_corr
from models import build, n_params


def evaluate(model, loader, scale, device):
    model.eval()
    hs, hh = [], []
    with torch.no_grad():
        for x, _, h in loader:
            y = model(x.to(device))
            hh.append(ri_to_complex(y, scale).cpu())
            hs.append(h)
    h, h_hat = torch.cat(hs), torch.cat(hh)
    return {"nmse_db": nmse_db(h_hat, h).item(), "psnr_db": psnr_db(h_hat, h).item(),
            "rho": complex_corr(h_hat, h).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--variant", default="", help="'', '_5e-5' or '_iid'")
    ap.add_argument("--model", default="dsrnet", choices=["dsrnet", "dncnn", "srcnn"])
    ap.add_argument("--input_repr", default="ap", choices=["ap", "ri"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=9e-4)
    ap.add_argument("--pilot_snr_db", type=float, default=None)
    ap.add_argument("--max_train", type=int, default=None, help="subset for smoke tests")
    ap.add_argument("--train_split", default="training",
                    help="use 'testing' only for a CPU smoke test without the large file")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    h_train = load_split(args.data_dir, args.train_split, args.variant)
    h_test = load_split(args.data_dir, "testing", args.variant)
    if args.max_train:
        h_train = h_train[: args.max_train]
    train_ds = RISChannelDataset(h_train, pilot_snr_db=args.pilot_snr_db,
                                 input_repr=args.input_repr)
    test_ds = RISChannelDataset(h_test, scale=train_ds.scale, pilot_snr_db=args.pilot_snr_db,
                                input_repr=args.input_repr)
    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    te = DataLoader(test_ds, batch_size=512)

    model = build(args.model).to(device)
    print(f"{args.model}: {n_params(model):,} params | scale={train_ds.scale:.3e} | "
          f"train={len(train_ds)} test={len(test_ds)} | device={device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    log_path = os.path.join(args.out_dir, f"{args.model}_log.csv")
    best = 1e9
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_mse", "nmse_db", "psnr_db", "rho", "sec"])
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
            m = evaluate(model, te, train_ds.scale, device)
            w.writerow([ep, tot / n, m["nmse_db"], m["psnr_db"], m["rho"], time.time() - t0])
            fh.flush()
            if m["nmse_db"] < best:
                best = m["nmse_db"]
                torch.save({"state_dict": model.state_dict(), "model": args.model,
                            "scale": train_ds.scale, "input_repr": args.input_repr,
                            "metrics": m, "epoch": ep},
                           os.path.join(args.out_dir, f"{args.model}.pt"))
            if ep == 1 or ep % 10 == 0 or ep == args.epochs:
                print(f"ep {ep:4d} | train MSE {tot / n:.4f} | NMSE {m['nmse_db']:.2f} dB | "
                      f"PSNR {m['psnr_db']:.2f} | rho {m['rho']:.3f} | best {best:.2f} dB")


if __name__ == "__main__":
    main()
