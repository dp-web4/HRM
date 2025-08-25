#!/usr/bin/env python3
import argparse, csv, os, time
import numpy as np
import torch

from tinyvae_irp_plugin import create_tinyvae_irp

def synthetic_crop(rgb=True, size=64):
    if rgb:
        arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    else:
        arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)
        arr = np.stack([arr]*3, axis=-1)
    t = torch.from_numpy(arr).float() / 255.0
    t = t.permute(2,0,1).unsqueeze(0)
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration_s", type=float, default=30.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--mean", type=float, nargs=3, default=[0.5,0.5,0.5])
    ap.add_argument("--std", type=float, nargs=3, default=[0.5,0.5,0.5])
    ap.add_argument("--crop_size", type=int, default=64)
    ap.add_argument("--out_csv", type=str, default="/tmp/tinyvae/bench_results.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = create_tinyvae_irp(
        device=device,
        input_channels=3,
        latent_dim=args.latent_dim,
        use_fp16=args.fp16,
        normalize=args.normalize,
        mean=args.mean,
        std=args.std
    )

    fields = ["t_ms","reconstruction_error","kl_divergence","trust","latent_norm","fp16","latent_dim","normalize"]
    t_end = time.time() + args.duration_s

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        n = 0
        times = []

        while time.time() < t_end:
            # TODO: replace with real crop from your attention pipeline
            x = synthetic_crop(size=args.crop_size).to(vae.device)

            z, telem = vae.refine(x)
            row = {
                "t_ms": telem.get("time_ms", 0.0),
                "reconstruction_error": telem.get("reconstruction_error", 0.0),
                "kl_divergence": telem.get("kl_divergence", 0.0),
                "trust": telem.get("trust", 0.0),
                "latent_norm": telem.get("latent_norm", 0.0),
                "fp16": int(args.fp16),
                "latent_dim": args.latent_dim,
                "normalize": int(args.normalize)
            }
            w.writerow(row)
            times.append(row["t_ms"])
            n += 1

        if times:
            times_sorted = sorted(times)
            p50 = times_sorted[int(0.5*len(times_sorted))]
            p90 = times_sorted[int(0.9*len(times_sorted))]
            p99 = times_sorted[int(0.99*len(times_sorted))-1]
            print(f"Frames: {n}  time_ms p50/p90/p99 = {p50:.2f}/{p90:.2f}/{p99:.2f}  (fp16={args.fp16})")
        else:
            print("No frames recorded.")

if __name__ == "__main__":
    main()
