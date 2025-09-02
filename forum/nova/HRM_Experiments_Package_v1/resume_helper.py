#!/usr/bin/env python3
"""
resume_helper.py — Standardized resume protocol for HRM experiments.

Usage pattern (inside your train script):

    from resume_helper import load_resume_bundle, heartbeat_writer, make_run_id

    args = ...  # argparse from your train script
    run_id = make_run_id(args.exp_name, parent_run_id=args.parent_run_id, resume_step=args.resume_step)

    # Construct your model/optimizer/scheduler/scaler first:
    model = build_model(args)
    optimizer = build_optimizer(model.parameters(), kind=args.optimizer)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    bundle = load_resume_bundle(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ckpt_path=args.resume,             # checkpoints/hrm_arc_step_XXXX.pt or *_best.pt
        allow_partial_weights=args.allow_partial,
        optimizer_policy=args.optimizer_policy,    # "load", "reset", or "swap_to_lion"
        schedule_policy=args.schedule_policy,      # "load" or "reset"
        log_fn=print,
    )

    # Then continue training using bundle["global_step"], bundle["epoch"], bundle["best_val"]
    # and call heartbeat_writer(...) every minute in your training loop.

CLI (standalone):
    python resume_helper.py --resume checkpoints/hrm_arc_step_40000.pt --dry-run

This prints what would be loaded without touching model code (useful for validation).
"""

import math
import json
import time
import datetime
import hashlib
from pathlib import Path

try:
    import torch
except Exception as e:
    torch = None

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def make_run_id(exp_name: str, parent_run_id: str = "", resume_step: int = -1) -> str:
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    tail = f"_resume@{resume_step}" if resume_step >= 0 else ""
    parent = f"{parent_run_id}_" if parent_run_id else ""
    return f"{exp_name}_{parent}{stamp}{tail}"

def choose_latest_step_ckpt(ckpt_dir: str):
    p = Path(ckpt_dir)
    if not p.exists():
        return None
    cands = sorted(p.glob("hrm_arc_step_*.pt"), key=lambda x: int(x.stem.split('_')[-1]))
    return cands[-1] if cands else None

def load_resume_bundle(
    model,
    optimizer,
    scheduler,
    scaler,
    ckpt_path: str,
    allow_partial_weights: bool = False,
    optimizer_policy: str = "load",   # "load" | "reset" | "swap_to_lion"
    schedule_policy: str = "load",    # "load" | "reset"
    log_fn=print,
) -> dict:
    """
    Returns dict with: epoch, global_step, best_val, loaded, missing, unexpected.
    Side effects: loads state_dicts into model/optimizer/scheduler/scaler as per policy.
    """
    out = dict(epoch=0, global_step=0, best_val=float("inf"),
               loaded=False, missing=[], unexpected=[])

    if ckpt_path == "" or ckpt_path is None:
        log_fn("[resume] No checkpoint path provided; starting fresh.")
        return out

    if torch is None:
        log_fn("[resume] WARNING: torch not available in this environment; dry-load only.")
        if not Path(ckpt_path).exists():
            log_fn(f"[resume] ERROR: ckpt not found: {ckpt_path}")
        return out

    ckpt = torch.load(ckpt_path, map_location="cpu")
    log_fn(f"[resume] Loading checkpoint: {ckpt_path}")

    # 1) model weights
    try:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=not allow_partial_weights)
    except TypeError:
        # some torch versions return None on success
        missing, unexpected = [], []
    if allow_partial_weights:
        # torch may not return these lists when strict=False; compute diffs manually
        try:
            cur = set(dict(model.state_dict()).keys())
            saved = set(dict(ckpt["model_state_dict"]).keys())
            missing = sorted(list(cur - saved))
            unexpected = sorted(list(saved - cur))
        except Exception:
            pass
    out["missing"], out["unexpected"] = missing, unexpected
    if missing:
        log_fn(f"[resume] warning: missing params: {len(missing)}")
    if unexpected:
        log_fn(f"[resume] warning: unexpected params: {len(unexpected)}")

    # 2) optimizer policy
    if optimizer_policy == "load" and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        log_fn("[resume] optimizer: loaded")
    else:
        log_fn(f"[resume] optimizer: reset ({optimizer_policy})")

    # 3) scheduler policy
    if schedule_policy == "load" and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            log_fn("[resume] scheduler: loaded")
        except Exception as e:
            log_fn(f"[resume] scheduler: load failed ({e}); resetting.")
    else:
        log_fn(f"[resume] scheduler: reset ({schedule_policy})")

    # 4) scaler (AMP)
    if "scaler_state_dict" in ckpt and scaler is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            log_fn("[resume] amp scaler: loaded")
        except Exception as e:
            log_fn(f"[resume] amp scaler: load failed ({e}); resetting.")
    else:
        log_fn("[resume] amp scaler: reset")

    out["epoch"] = int(ckpt.get("epoch", 0))
    out["global_step"] = int(ckpt.get("global_step", 0))
    out["best_val"] = float(ckpt.get("best_val_loss", float("inf")))
    out["loaded"] = True
    log_fn(f"[resume] step={out['global_step']} epoch={out['epoch']} best_val={out['best_val']:.4f}")
    return out

def heartbeat_writer(status_path: str, payload_fn, interval_sec: int = 60, log_fn=print):
    """
    Return a callable you can invoke each iteration to write status.json.
    Example:
        write_status = heartbeat_writer(args.status_json_path, lambda: payload, 60)
        # in loop: write_status()
    """
    def write_once():
        try:
            payload = payload_fn()
            Path(status_path).parent.mkdir(parents=True, exist_ok=True)
            with open(status_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            log_fn(f"[heartbeat] write failed: {e}")
    return write_once

def make_status_payload(run_id: str,
                        epoch: int,
                        global_step: int,
                        steps_per_sec: float,
                        lr: float,
                        grad_norm: float,
                        last_fast_eval_step: int,
                        last_full_eval_step: int,
                        best_eval_acc: float,
                        best_eval_step: int,
                        eta_to_next_fast_eval_sec: float,
                        eta_to_next_full_eval_sec: float):
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "epoch": epoch,
        "global_step": global_step,
        "steps_per_sec": steps_per_sec,
        "lr": lr,
        "grad_norm": grad_norm,
        "last_fast_eval_step": last_fast_eval_step,
        "last_full_eval_step": last_full_eval_step,
        "best_eval_acc": best_eval_acc,
        "best_eval_step": best_eval_step,
        "eta_to_next_fast_eval_sec": eta_to_next_fast_eval_sec,
        "eta_to_next_full_eval_sec": eta_to_next_full_eval_sec
    }

if __name__ == "__main__":
    import argparse, os, sys
    ap = argparse.ArgumentParser(description="Validate or use resume policy (dry-run friendly).")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume.")
    ap.add_argument("--ckpt_dir", type=str, default="", help="If set, will pick latest hrm_arc_step_*.pt from this dir.")
    ap.add_argument("--optimizer_policy", type=str, default="load", choices=["load","reset","swap_to_lion"])
    ap.add_argument("--schedule_policy", type=str, default="load", choices=["load","reset"])
    ap.add_argument("--allow_partial", action="store_true", help="Allow partial model weight loading (strict=False).")
    ap.add_argument("--dry-run", action="store_true", help="Parse/inspect checkpoint without touching torch modules.")
    args = ap.parse_args()

    ckpt = args.resume
    if args.ckpt_dir and not ckpt:
        latest = choose_latest_step_ckpt(args.ckpt_dir)
        if latest is None:
            print(f"[resume] no step ckpts found in: {args.ckpt_dir}")
            sys.exit(1)
        ckpt = str(latest)

    if not ckpt:
        print("[resume] No checkpoint provided. Use --resume or --ckpt_dir.")
        sys.exit(1)

    print(f"[resume] Selected checkpoint: {ckpt}")
    if args.dry_run or (torch is None):
        try:
            size = os.path.getsize(ckpt)
            print(f"[resume] (dry) file exists, size={size} bytes")
        except Exception as e:
            print(f"[resume] (dry) failed to read file: {e}")
        sys.exit(0)

    # Demo minimal load with dummy containers (for smoke testing)
    class _Dummy:
        def state_dict(self): return {}
        def load_state_dict(self, *a, **k): return ([], [])
    model = _Dummy()
    optimizer = _Dummy()
    scheduler = _Dummy()
    scaler = _Dummy()

    _ = load_resume_bundle(model, optimizer, scheduler, scaler,
                           ckpt_path=ckpt,
                           allow_partial_weights=args.allow_partial,
                           optimizer_policy=args.optimizer_policy,
                           schedule_policy=args.schedule_policy,
                           log_fn=print)
