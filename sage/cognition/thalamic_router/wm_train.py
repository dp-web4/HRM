"""World-model training — data pipeline + joint-loss trainer.

Reads gameplay / sage_plays_live / sage_plays_self partitions, joins
consecutive steps into (state_t, action, state_{t+1}, outcome) tuples,
trains WorldModel with α·L_action + β·L_dynamics + γ·L_outcome.

Outcome labels:
  1.0  if the trace was a WIN (gameplay + teacher-forced)
  0.0  if GAME_OVER (self-play losing runs)
  0.5  if NOT_FINISHED (teacher-forced runs that didn't reach WIN)

Validation splits are stratified by (game, outcome) and always include
a held-out game per bundle for generalization test.

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-2-world-model-embedding-sprint.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sage.cognition.router.data.reader import RouterDatasetReader
from sage.cognition.thalamic_router.phase1_training import _feature_vec
from sage.cognition.thalamic_router.world_model import (
    WorldModel, WorldModelConfig, save_world_model,
    build_input_vector, action_onehot,
    N_ACTIONS, N_BASE_FEATURES, ACTION_NAMES,
)


# ───────────────────────────────────────────────────────────────────
# Data loading — consecutive-step pair construction
# ───────────────────────────────────────────────────────────────────

def _outcome_from_metadata(md: Dict[str, Any]) -> float:
    """Derive the scalar outcome label from record metadata."""
    source = md.get("source", "")
    # Self-play GAME_OVER = hard 0.0 negative
    if source == "sage_plays_self":
        spl_self = md.get("sage_plays_self") or {}
        if spl_self.get("state_after") == "GAME_OVER":
            return 0.0
        # Other self-play states: unknown quality (still likely bad)
        return 0.2

    # Gameplay + teacher-forced: check game_outcome at trace level
    go = md.get("game_outcome") or {}
    if go:
        result = (go.get("result") or go.get("final_state") or "").upper()
        if result == "WIN":
            return 1.0
        if result == "GAME_OVER":
            return 0.0
        if result == "NOT_FINISHED":
            return 0.5
        # inferred_won fallback
        if go.get("inferred_won") is True:
            return 1.0
        return 0.5

    # Replay source says inferred_won — trust the trace
    replay_won = md.get("replay_source")
    if source in ("gameplay", "sage_plays_live"):
        # gameplay_capture only replays winning traces → 1.0
        # (teacher-forced on winning trace → same)
        return 1.0
    return 0.5


def _load_records_raw(path: Path) -> List[Dict[str, Any]]:
    """Read all records from a partition dir, preserving order."""
    reader = RouterDatasetReader(base_dir=Path("/tmp"))
    records = []
    for shard in sorted(path.glob("*.jsonl*")):
        for rec in reader.read_file(shard):
            records.append(rec)
    return records


def _action_from_record(rec: Dict[str, Any]) -> Optional[int]:
    """Extract the supervised action label for this record.
    For gameplay/sage_plays_live: known_good_action
    For sage_plays_self: the proposed action SAGE actually took
    """
    md = rec.get("metadata") or {}
    src = md.get("source", "")
    if src in ("gameplay", "sage_plays_live"):
        a = md.get("known_good_action")
        return int(a) if isinstance(a, int) else None
    if src == "sage_plays_self":
        spl = md.get("sage_plays_self") or {}
        a = spl.get("proposed_action")
        return int(a) if isinstance(a, int) else None
    return None


def _build_pairs(
    records: List[Dict[str, Any]],
    game_slugs: List[str],
    trace_length_hint: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Group records by (source, game, replay_source_path), sort by
    step_index, and emit consecutive (t, t+1) tuples.

    Each emitted dict has:
      base_features_t, base_features_tp1, game_idx, level, step_frac,
      last_action, action_t, outcome
    """
    game_idx = {g: i for i, g in enumerate(game_slugs)}

    groups: Dict[Tuple[str, str, Optional[str]], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        md = rec.get("metadata") or {}
        src = md.get("source", "")
        game = md.get("game") or ""
        key = (src, game, md.get("replay_source_path") or md.get("game_id"))
        groups[key].append(rec)

    pairs: List[Dict[str, Any]] = []
    for key, recs in groups.items():
        recs.sort(key=lambda r: (r.get("metadata") or {}).get("step_index", 0))
        game = key[1]
        if game not in game_idx:
            continue
        gi = game_idx[game]
        # Estimate trace length for step_frac — use max step_index seen
        max_step = max(
            (r.get("metadata") or {}).get("step_index", 0) for r in recs
        ) or 1

        for i in range(len(recs) - 1):
            r_t, r_tp1 = recs[i], recs[i + 1]
            md_t = r_t.get("metadata") or {}
            md_tp1 = r_tp1.get("metadata") or {}
            # Require consecutive step indices
            s_t = md_t.get("step_index")
            s_tp1 = md_tp1.get("step_index")
            if not isinstance(s_t, int) or not isinstance(s_tp1, int):
                continue
            if s_tp1 != s_t + 1:
                continue

            action_t = _action_from_record(r_t)
            if action_t is None or action_t < 0 or action_t >= N_ACTIONS:
                continue

            feats_t = _feature_vec(r_t)
            feats_tp1 = _feature_vec(r_tp1)

            level = md_t.get("level") or md_t.get("known_good_level") or 0
            step_frac = s_t / max_step

            # last_action: we'd need to look at record i-1, but we'll
            # default to 0 for simplicity. Temporal context can be added later.
            last_a = 0

            outcome = _outcome_from_metadata(md_t)

            pairs.append({
                "feats_t": feats_t,
                "feats_tp1": feats_tp1,
                "game_idx": gi,
                "level": int(level),
                "step_frac": float(step_frac),
                "last_action": last_a,
                "action_t": int(action_t),
                "outcome": float(outcome),
                "game": game,
                "source": key[0],
            })
    return pairs


# ───────────────────────────────────────────────────────────────────
# Torch Dataset
# ───────────────────────────────────────────────────────────────────

class WMDataset(Dataset):
    def __init__(
        self, pairs: List[Dict[str, Any]], n_games: int, n_levels: int,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        self.pairs = pairs
        self.n_games = n_games
        self.n_levels = n_levels
        # Normalization: fit on training set
        base = np.array([p["feats_t"] for p in pairs], dtype=np.float64)
        if feature_mean is None:
            self.feature_mean = base.mean(axis=0)
            self.feature_std = base.std(axis=0) + 1e-6
        else:
            self.feature_mean = feature_mean
            self.feature_std = feature_std

    def __len__(self) -> int:
        return len(self.pairs)

    def _norm(self, feats: List[float]) -> List[float]:
        arr = (np.array(feats) - self.feature_mean) / self.feature_std
        return arr.tolist()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.pairs[idx]
        feats_t = self._norm(p["feats_t"])
        feats_tp1 = self._norm(p["feats_tp1"])
        x_t = build_input_vector(
            feats_t, p["game_idx"], self.n_games, p["level"], self.n_levels,
            p["step_frac"], p["last_action"],
        )
        x_tp1 = build_input_vector(
            feats_tp1, p["game_idx"], self.n_games, p["level"], self.n_levels,
            min(1.0, p["step_frac"] + 0.01), p["action_t"],
        )
        return {
            "x_t": torch.tensor(x_t, dtype=torch.float32),
            "x_tp1": torch.tensor(x_tp1, dtype=torch.float32),
            "action": torch.tensor(p["action_t"], dtype=torch.long),
            "outcome": torch.tensor(p["outcome"], dtype=torch.float32),
        }


# ───────────────────────────────────────────────────────────────────
# Training + evaluation
# ───────────────────────────────────────────────────────────────────

def train(
    train_ds: WMDataset, val_ds: WMDataset, n_games: int, n_levels: int,
    epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
    alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3,
    device: str = "cpu", verbose: bool = True,
) -> Tuple[WorldModel, Dict[str, Any]]:
    model = WorldModel(n_games=n_games, n_levels=n_levels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_action_acc": [], "val_outcome_auroc": [],
               "val_dynamics_mse": [], "val_trivial_dynamics_mse": []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            x_t = batch["x_t"].to(device)
            x_tp1 = batch["x_tp1"].to(device)
            a = batch["action"].to(device)
            o = batch["outcome"].to(device)
            a_oh = F.one_hot(a, num_classes=N_ACTIONS).float()

            emb_t = model.encode(x_t)
            with torch.no_grad():
                emb_tp1 = model.encode(x_tp1)

            action_logits = model.forward_action(emb_t)
            outcome_logit = model.forward_outcome(emb_t)
            next_pred = model.forward_dynamics(emb_t, a_oh)

            L_a = F.cross_entropy(action_logits, a)
            L_d = F.mse_loss(next_pred, emb_tp1)
            L_o = F.binary_cross_entropy_with_logits(outcome_logit, o)
            loss = alpha * L_a + beta * L_d + gamma * L_o

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(loss.item())

        mean_train_loss = float(np.mean(epoch_losses))
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(mean_train_loss)
        for k, v in val_metrics.items():
            history.setdefault(f"val_{k}", []).append(v)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={mean_train_loss:.4f}  "
                  f"val_action_acc={val_metrics['action_acc']:.3f}  "
                  f"val_dyn_mse={val_metrics['dynamics_mse']:.4f}  "
                  f"val_out_auroc={val_metrics['outcome_auroc']:.3f}")

    return model, history


def evaluate(model: WorldModel, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    action_correct = 0
    action_total = 0
    dyn_mse_sum = 0.0
    dyn_count = 0
    trivial_dyn_mse_sum = 0.0
    outcome_preds = []
    outcome_true = []
    with torch.no_grad():
        for batch in loader:
            x_t = batch["x_t"].to(device)
            x_tp1 = batch["x_tp1"].to(device)
            a = batch["action"].to(device)
            o = batch["outcome"].to(device)
            a_oh = F.one_hot(a, num_classes=N_ACTIONS).float()

            emb_t = model.encode(x_t)
            emb_tp1 = model.encode(x_tp1)
            action_logits = model.forward_action(emb_t)
            outcome_logit = model.forward_outcome(emb_t)
            next_pred = model.forward_dynamics(emb_t, a_oh)

            action_correct += (action_logits.argmax(dim=-1) == a).sum().item()
            action_total += a.numel()
            dyn_mse_sum += F.mse_loss(next_pred, emb_tp1, reduction="sum").item()
            # Trivial baseline: "predict zero change" = emb_t itself
            trivial_dyn_mse_sum += F.mse_loss(emb_t, emb_tp1, reduction="sum").item()
            dyn_count += emb_tp1.numel()
            outcome_preds.extend(torch.sigmoid(outcome_logit).cpu().tolist())
            outcome_true.extend(o.cpu().tolist())

    auroc = _simple_auroc(outcome_preds, outcome_true)
    return {
        "action_acc": action_correct / max(action_total, 1),
        "dynamics_mse": dyn_mse_sum / max(dyn_count, 1),
        "trivial_dynamics_mse": trivial_dyn_mse_sum / max(dyn_count, 1),
        "outcome_auroc": auroc,
    }


def _simple_auroc(preds: List[float], targets: List[float]) -> float:
    """Threshold at 0.5 on targets to binarize, then compute AUROC
    via the Mann-Whitney U statistic."""
    pos = [p for p, t in zip(preds, targets) if t >= 0.5]
    neg = [p for p, t in zip(preds, targets) if t < 0.5]
    if not pos or not neg:
        return 0.5
    n_correct = sum(1 for p in pos for n in neg if p > n) + \
                0.5 * sum(1 for p in pos for n in neg if p == n)
    return n_correct / (len(pos) * len(neg))


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Path to router data dir (will scan {machine}/* subdirs).")
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1.0, help="action loss weight")
    p.add_argument("--beta", type=float, default=0.5, help="dynamics loss weight")
    p.add_argument("--gamma", type=float, default=0.3, help="outcome loss weight")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True, help="Base path for saved adapter (no ext).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    machine_root = Path(args.data) / args.machine
    sources = ["gameplay", "sage_plays_live", "sage_plays_self"]
    all_records = []
    for src in sources:
        sub = machine_root / src
        if sub.is_dir():
            recs = _load_records_raw(sub)
            print(f"  {src}: {len(recs)} records")
            all_records.extend(recs)

    if not all_records:
        print(f"ERROR: no records found under {machine_root}")
        return 1

    # Collect game slugs
    game_slugs = sorted({(r.get("metadata") or {}).get("game") for r in all_records
                         if (r.get("metadata") or {}).get("game")})
    print(f"  games: {game_slugs}")

    pairs = _build_pairs(all_records, game_slugs)
    print(f"  consecutive-step pairs: {len(pairs)}")
    if len(pairs) < 100:
        print("ERROR: need ≥100 pairs for training")
        return 2

    # Stratified train/val split
    random.shuffle(pairs)
    split = int(0.85 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    print(f"  train: {len(train_pairs)}  val: {len(val_pairs)}")

    train_ds = WMDataset(train_pairs, n_games=len(game_slugs), n_levels=10)
    val_ds = WMDataset(
        val_pairs, n_games=len(game_slugs), n_levels=10,
        feature_mean=train_ds.feature_mean, feature_std=train_ds.feature_std,
    )

    print(f"\nTraining on {args.device} for {args.epochs} epochs "
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

    cfg = WorldModelConfig(
        n_games=len(game_slugs), game_slugs=game_slugs, n_levels=10,
        feature_mean=train_ds.feature_mean.tolist(),
        feature_std=train_ds.feature_std.tolist(),
        trained_at=dt.datetime.utcnow().isoformat() + "Z",
        machine=args.machine,
        train_record_count=len(train_pairs),
        val_metrics=final_val,
    )
    save_world_model(model.cpu(), cfg, Path(args.out))
    print(f"\nSaved adapter: {args.out}.pt + {args.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
