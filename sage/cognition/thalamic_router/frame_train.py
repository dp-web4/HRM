"""frame_train — data pipeline + trainer for the frame-based router.

Data pipeline:
  - Iterate each trace (from ARC-SAGE/knowledge/visual-memory/)
  - Replay env step-by-step with teacher-forced known_good actions
  - At each step, emit (prev_frame, curr_frame, scalar_bookkeeping,
    action_label, invoke_label) tuples
  - Cache per-game to .npz for fast reuse across epochs

Training:
  - FrameRouter model (see frame_router.py)
  - Losses:
      L_action  = CE(action_logits, known_good_action)
      L_invoke  = BCE(invoke_logit, invoke_label)
      L_dyn     = MSE(next_visual_pred, actual_next_visual_pool.detach())
  - Auxiliary dynamics loss shapes the embedding (surprise signal at infer time)

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-2-world-model-embedding-sprint.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sage.cognition.thalamic_router.gameplay_capture import (
    load_trace, _discover_trace, ARC_AGI_EXPERIMENTS,
)
from sage.cognition.thalamic_router.frame_router import (
    FrameRouter, FrameRouterConfig, save_frame_router,
    onehot_frame, build_scalar_vector, compute_invoke_label,
    N_ACTIONS, N_COLORS, FRAME_H, FRAME_W, RECENT_ACTIONS_K, ACTION_NAMES,
)


# ───────────────────────────────────────────────────────────────────
# Trace replay → frame tuples
# ───────────────────────────────────────────────────────────────────

def _make_env(game_family: str, game_id: str):
    """Same version-fallback pattern as gameplay_capture."""
    if str(ARC_AGI_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(ARC_AGI_EXPERIMENTS))
    from arc_agi import Arcade
    arc = Arcade(operation_mode="offline")
    env = arc.make(game_id)
    if env is None:
        env = arc.make(game_family)
    if env is None:
        raise RuntimeError(f"arc.make None for {game_id} / {game_family}")
    return env


def replay_trace_to_tuples(
    trace, game_idx: int, n_games: int, n_levels: int = 10,
) -> List[Dict[str, Any]]:
    """Replay a trace on the env, yield training tuples per step.

    Teacher-forced: env always advances on trace.step.action, so the NN
    observes the solver's actual frame sequence at every tick.
    """
    env = _make_env(trace.game, trace.game_id)
    try:
        from arcengine import GameAction
        int_to_action = {ga.value: ga for ga in GameAction}
    except Exception:
        int_to_action = {}

    fd = env.reset()
    # First-step convention: prev_frame is all zeros (user: "at game start it's all zero")
    zero_frame = np.zeros((N_COLORS, FRAME_H, FRAME_W), dtype=np.float32)
    prev_frame_oh = zero_frame
    prev_frame_raw = None      # for invoke-label pixel diff
    prev_level: Optional[int] = None
    recent: deque = deque([0] * RECENT_ACTIONS_K, maxlen=RECENT_ACTIONS_K)
    total_steps = max(1, len(trace.steps))

    tuples: List[Dict[str, Any]] = []
    for step in trace.steps:
        curr_frame_raw = np.asarray(getattr(fd, "frame", None))
        curr_frame_oh = onehot_frame(curr_frame_raw) if curr_frame_raw is not None else zero_frame

        level = step.level if step.level is not None else 0
        step_frac = min(1.0, step.index / total_steps)
        budget_remaining = max(0.0, 1.0 - step.index / total_steps)
        available_actions = [1] * N_ACTIONS          # placeholder — all allowed
        batch_state = [0.0, 0.0, 0.0]                # placeholder — no batched play yet

        invoke = compute_invoke_label(
            prev_frame=prev_frame_raw if prev_frame_raw is not None else np.zeros_like(curr_frame_raw),
            curr_frame=curr_frame_raw,
            step_index=step.index, level=level, prev_level=prev_level,
        )

        scalar = build_scalar_vector(
            game_idx=game_idx, n_games=n_games,
            level=level, n_levels=n_levels,
            step_frac=step_frac, budget_remaining=budget_remaining,
            recent_actions=list(recent),
            available_actions=available_actions,
            batch_state=batch_state,
        )

        tuples.append({
            "prev_frame": prev_frame_oh,
            "curr_frame": curr_frame_oh,
            "scalar": np.array(scalar, dtype=np.float32),
            "action": int(step.action),
            "invoke": float(invoke),
        })

        # Teacher-forced advance
        ga = int_to_action.get(step.action)
        try:
            if ga is not None:
                fd = env.step(ga, data=step.data) if step.data else env.step(ga)
        except Exception:
            break

        recent.append(step.action)
        prev_frame_oh = curr_frame_oh
        prev_frame_raw = curr_frame_raw
        prev_level = level

    return tuples


def build_dataset_from_traces(
    games: List[str], arc_sage_root: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build the full (prev, curr, scalar, action, invoke) corpus across games."""
    # Resolve game_ids from coordination file
    coord_path = arc_sage_root / "knowledge" / "game_coordination.json"
    coord = json.loads(coord_path.read_text())
    id_by_family = {g["family"]: g["id"] for g in coord.get("games", [])}

    # Stable game-slug ordering
    game_slugs = sorted(games)

    all_tuples: List[Dict[str, Any]] = []
    for family in game_slugs:
        game_id = id_by_family.get(family)
        if not game_id:
            print(f"  [skip] no coord entry for {family}")
            continue
        trace_path = _discover_trace(family, game_id)
        if not trace_path or not trace_path.exists():
            print(f"  [skip] no trace for {family}")
            continue
        try:
            trace = load_trace(trace_path, family, game_id)
        except Exception as e:
            print(f"  [skip] {family}: load failed: {e!r}")
            continue
        gi = game_slugs.index(family)
        try:
            tuples = replay_trace_to_tuples(trace, gi, len(game_slugs))
        except Exception as e:
            print(f"  [skip] {family}: replay failed: {e!r}")
            continue
        # Tag game on each tuple for diagnostic filtering
        for t in tuples:
            t["game"] = family
        all_tuples.extend(tuples)
        print(f"  {family:8s}: {len(tuples)} tuples")

    return all_tuples, game_slugs


# ───────────────────────────────────────────────────────────────────
# Torch Dataset
# ───────────────────────────────────────────────────────────────────

class FrameDataset(Dataset):
    def __init__(self, tuples: List[Dict[str, Any]]):
        self.tuples = tuples

    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.tuples[idx]
        return {
            "prev_frame": torch.from_numpy(t["prev_frame"]),
            "curr_frame": torch.from_numpy(t["curr_frame"]),
            "scalar": torch.from_numpy(t["scalar"]),
            "action": torch.tensor(t["action"], dtype=torch.long),
            "invoke": torch.tensor(t["invoke"], dtype=torch.float32),
        }


def _pair_next(tuples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach next-step visual (the actual frame at step t+1) to each tuple.
    Used as dynamics-head target. Last step of each trace drops its dynamics
    target (nothing comes after). Traces are detected by game-identity boundary.
    """
    out: List[Dict[str, Any]] = []
    for i in range(len(tuples) - 1):
        t = tuples[i]
        nxt = tuples[i + 1]
        # Same game = same trace (we built traces contiguously per-game)
        if nxt.get("game") != t.get("game"):
            continue
        t = dict(t)
        t["next_curr_frame"] = nxt["curr_frame"]
        out.append(t)
    return out


class FramePairDataset(Dataset):
    """Like FrameDataset but also carries the actual next-step visual for
    dynamics-head supervision."""
    def __init__(self, tuples: List[Dict[str, Any]]):
        self.tuples = tuples

    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.tuples[idx]
        return {
            "prev_frame": torch.from_numpy(t["prev_frame"]),
            "curr_frame": torch.from_numpy(t["curr_frame"]),
            "next_curr_frame": torch.from_numpy(t["next_curr_frame"]),
            "scalar": torch.from_numpy(t["scalar"]),
            "action": torch.tensor(t["action"], dtype=torch.long),
            "invoke": torch.tensor(t["invoke"], dtype=torch.float32),
        }


# ───────────────────────────────────────────────────────────────────
# Training + evaluation
# ───────────────────────────────────────────────────────────────────

def train(
    train_ds: FramePairDataset, val_ds: FramePairDataset,
    n_games: int, n_levels: int,
    epochs: int = 40, batch_size: int = 32, lr: float = 1e-3,
    alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.2,
    device: str = "cpu", verbose: bool = True,
) -> Tuple[FrameRouter, Dict[str, Any]]:
    model = FrameRouter(n_games=n_games, n_levels=n_levels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history: Dict[str, List[float]] = {}
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            prev = batch["prev_frame"].to(device)
            curr = batch["curr_frame"].to(device)
            nxt = batch["next_curr_frame"].to(device)
            scalar = batch["scalar"].to(device)
            a = batch["action"].to(device)
            inv = batch["invoke"].to(device)
            a_oh = F.one_hot(a, num_classes=N_ACTIONS).float()

            out = model(prev, curr, scalar, a_oh)
            L_a = F.cross_entropy(out["action_logits"], a)
            L_i = F.binary_cross_entropy_with_logits(out["invoke_logit"], inv)

            # Dynamics target = visual pool of (curr, next) — what the next step's
            # CNN would produce. Detach target so gradient only flows into dynamics_head.
            with torch.no_grad():
                # next step has prev=curr, curr=next: approximate target
                next_out = model(curr, nxt, scalar)
                next_visual = next_out["visual_pool"]
            L_d = F.mse_loss(out["next_visual_pred"], next_visual)

            loss = alpha * L_a + beta * L_i + gamma * L_d
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        val = evaluate(model, val_loader, device)
        history.setdefault("train_loss", []).append(train_loss)
        for k, v in val.items():
            history.setdefault(f"val_{k}", []).append(v)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  ep{epoch:3d}  loss={train_loss:.3f}  "
                  f"act={val['action_acc']:.3f}  "
                  f"inv_PR={val['invoke_precision']:.2f}/{val['invoke_recall']:.2f}  "
                  f"inv_auroc={val['invoke_auroc']:.3f}  "
                  f"dyn={val['dynamics_mse']:.3f}")

    return model, history


def evaluate(model: FrameRouter, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    action_correct = 0
    action_total = 0
    inv_preds = []
    inv_true = []
    dyn_mse_sum = 0.0
    dyn_n = 0
    with torch.no_grad():
        for batch in loader:
            prev = batch["prev_frame"].to(device)
            curr = batch["curr_frame"].to(device)
            nxt = batch["next_curr_frame"].to(device)
            scalar = batch["scalar"].to(device)
            a = batch["action"].to(device)
            inv = batch["invoke"].to(device)
            a_oh = F.one_hot(a, num_classes=N_ACTIONS).float()

            out = model(prev, curr, scalar, a_oh)
            action_correct += (out["action_logits"].argmax(dim=-1) == a).sum().item()
            action_total += a.numel()
            inv_preds.extend(torch.sigmoid(out["invoke_logit"]).cpu().tolist())
            inv_true.extend(inv.cpu().tolist())

            next_out = model(curr, nxt, scalar)
            next_visual = next_out["visual_pool"]
            dyn_mse_sum += F.mse_loss(out["next_visual_pred"], next_visual, reduction="sum").item()
            dyn_n += next_visual.numel()

    # AUROC via Mann-Whitney U
    pos = [p for p, t in zip(inv_preds, inv_true) if t >= 0.5]
    neg = [p for p, t in zip(inv_preds, inv_true) if t < 0.5]
    if pos and neg:
        wins = sum(1 for p in pos for n in neg if p > n) + \
               0.5 * sum(1 for p in pos for n in neg if p == n)
        auroc = wins / (len(pos) * len(neg))
    else:
        auroc = 0.5

    tp = sum(1 for p, t in zip(inv_preds, inv_true) if p >= 0.5 and t >= 0.5)
    fp = sum(1 for p, t in zip(inv_preds, inv_true) if p >= 0.5 and t < 0.5)
    fn = sum(1 for p, t in zip(inv_preds, inv_true) if p < 0.5 and t >= 0.5)
    return {
        "action_acc": action_correct / max(action_total, 1),
        "invoke_auroc": auroc,
        "invoke_precision": tp / max(tp + fp, 1),
        "invoke_recall": tp / max(tp + fn, 1),
        "invoke_positive_rate": sum(inv_true) / max(len(inv_true), 1),
        "dynamics_mse": dyn_mse_sum / max(dyn_n, 1),
    }


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

def _resolve_arc_sage() -> Path:
    for root in [
        Path(os.environ.get("ARC_SAGE_DIR", "")),
        Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE"),
        Path.home() / "ai-workspace" / "ARC-SAGE",
        Path.home() / "repos" / "ARC-SAGE",
    ]:
        if root.exists() and (root / "knowledge" / "game_coordination.json").exists():
            return root
    raise FileNotFoundError("ARC-SAGE not found; set ARC_SAGE_DIR env var")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="Comma-separated game slugs or 'all'.")
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1.0, help="action CE weight")
    p.add_argument("--beta", type=float, default=0.5, help="invoke BCE weight")
    p.add_argument("--gamma", type=float, default=0.2, help="dynamics MSE weight")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True, help="Adapter base path (no ext).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    arc_sage = _resolve_arc_sage()
    print(f"ARC-SAGE root: {arc_sage}")

    if args.games == "all":
        coord = json.loads((arc_sage / "knowledge" / "game_coordination.json").read_text())
        games = [g["family"] for g in coord.get("games", [])]
    else:
        games = [g.strip() for g in args.games.split(",") if g.strip()]

    print(f"Games: {games}")
    t0 = time.time()
    tuples, game_slugs = build_dataset_from_traces(games, arc_sage)
    print(f"Built {len(tuples)} tuples across {len(game_slugs)} games "
          f"in {time.time()-t0:.1f}s")

    pair_tuples = _pair_next(tuples)
    print(f"Paired tuples (with next-frame target): {len(pair_tuples)}")
    if len(pair_tuples) < 100:
        print("ERROR: need ≥100 pairs for training")
        return 2

    # Stratified by game — shuffle within each game, then split
    by_game: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in pair_tuples:
        by_game[t.get("game", "?")].append(t)
    train_pairs: List[Dict[str, Any]] = []
    val_pairs: List[Dict[str, Any]] = []
    for g, items in by_game.items():
        random.shuffle(items)
        split = int((1 - args.val_frac) * len(items))
        train_pairs.extend(items[:split])
        val_pairs.extend(items[split:])
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    train_ds = FramePairDataset(train_pairs)
    val_ds = FramePairDataset(val_pairs)

    print(f"\nTraining on {args.device}  epochs={args.epochs}  "
          f"(α={args.alpha} β={args.beta} γ={args.gamma})")
    model, history = train(
        train_ds, val_ds, n_games=len(game_slugs), n_levels=10,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        device=args.device,
    )

    final_val = {k: v[-1] for k, v in history.items() if v}
    print(f"\nFinal validation metrics:")
    for k, v in final_val.items():
        print(f"  {k}: {v:.4f}")

    cfg = FrameRouterConfig(
        n_games=len(game_slugs), game_slugs=game_slugs, n_levels=10,
        trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        machine=args.machine,
        train_record_count=len(train_pairs),
        val_metrics=final_val,
    )
    save_frame_router(model.cpu(), cfg, Path(args.out))
    print(f"\nSaved adapter: {args.out}.pt + {args.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
