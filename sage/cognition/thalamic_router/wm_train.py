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


# v2 invoke-label threshold. Self-supervised dispatch signal.
#
# IMPORTANT: snarc_novelty as currently computed in synth_router_input
# (gameplay_capture.py:196) is NOT frame-to-frame — it's a linearly-decaying
# tick proxy: `max(0, 1 - tick/200) if delta>0.01 else 0.1`. Using it as an
# invoke trigger labels all early-trace records as "should invoke" regardless
# of what's actually happening visually. That teaches the NN to invoke on
# ticks < 100, which is wrong.
#
# The fields that ARE frame-to-frame:
#   snarc_arousal  = frame_delta          (full delta, clamped to [0,1])
#   snarc_surprise = frame_delta * 0.8    (attenuated delta)
#
# Self-supervised "invoke" label should fire on frame-to-frame change, not
# tick position. We use arousal (the full delta) with a moderate threshold.
# This matches the user's frame: "after the LLM told SAGE what to do, only
# invoke again if novelty/surprise on the resulting frame justifies it."
FRAME_DELTA_INVOKE_THRESHOLD = 0.5   # delta ≥ 0.5 = "something significant changed"


def _invoke_label(
    rec_t: Dict[str, Any],
    is_first_of_trace: bool,
    level_changed: bool,
) -> float:
    """Self-supervised label: should the NN have invoked the LLM at this state?

    Fires 1.0 on:
      - first frame of a trace (no context yet)
      - level just changed (new geometry/rules)
      - frame_delta from previous frame ≥ threshold (significant visual event)

    Returns 0.0 otherwise. NOT a soft label — obvious-case supervision.
    Generalization to non-obvious "invoke" moments is what the NN learns
    from the input features (e.g., low confidence on the play head).

    NB: uses snarc_arousal (frame-delta) not snarc_novelty (tick proxy).
    See comment above the threshold constant for why.
    """
    if is_first_of_trace or level_changed:
        return 1.0
    ri = rec_t.get("router_input") or {}
    arousal = float(ri.get("snarc_arousal") or 0.0)
    if arousal >= FRAME_DELTA_INVOKE_THRESHOLD:
        return 1.0
    return 0.0


def _build_pairs(
    records: List[Dict[str, Any]],
    game_slugs: List[str],
    trace_length_hint: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Group records by (source, game, replay_source_path), sort by
    step_index, and emit consecutive (t, t+1) tuples.

    Each emitted dict has:
      base_features_t, base_features_tp1, game_idx, level, step_frac,
      last_action, action_t, outcome, invoke
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
        max_step = max(
            (r.get("metadata") or {}).get("step_index", 0) for r in recs
        ) or 1

        prev_level = None
        for i in range(len(recs) - 1):
            r_t, r_tp1 = recs[i], recs[i + 1]
            md_t = r_t.get("metadata") or {}
            md_tp1 = r_tp1.get("metadata") or {}
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
            last_a = 0
            outcome = _outcome_from_metadata(md_t)

            # v2 invoke label — self-supervised from SNARC + first-frame
            is_first = (i == 0) or (s_t == 1)
            level_changed = (prev_level is not None and int(level) != prev_level)
            invoke = _invoke_label(r_t, is_first, level_changed)
            prev_level = int(level)

            pairs.append({
                "feats_t": feats_t,
                "feats_tp1": feats_tp1,
                "game_idx": gi,
                "level": int(level),
                "step_frac": float(step_frac),
                "last_action": last_a,
                "action_t": int(action_t),
                "outcome": float(outcome),
                "invoke": float(invoke),
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
            "invoke": torch.tensor(p.get("invoke", 0.0), dtype=torch.float32),
        }


# ───────────────────────────────────────────────────────────────────
# Training + evaluation
# ───────────────────────────────────────────────────────────────────

def train(
    train_ds: WMDataset, val_ds: WMDataset, n_games: int, n_levels: int,
    epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
    alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5,
    delta: float = 0.3, epsilon: float = 0.3,
    device: str = "cpu", verbose: bool = True,
) -> Tuple[WorldModel, Dict[str, Any]]:
    """v1 training with action-conditional outcome + contrastive loss.

    Losses:
      L_a: action-head CE on known_good (supervised)
      L_d: dynamics MSE (predict next embedding)
      L_o: outcome BCE — outcome_head(emb, KNOWN action) → trace outcome
      L_c: contrastive outcome — outcome_head(emb, RANDOM other action) →
           0.0 if anchor is winning, else 0.5.
           Forces the outcome head to differentiate by action, preventing
           source-discrimination shortcut.
    """
    model = WorldModel(
        n_games=n_games, n_levels=n_levels,
        outcome_action_conditional=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history: Dict[str, List[float]] = {}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_L_a, epoch_L_d, epoch_L_o, epoch_L_c = [], [], [], []
        epoch_L_i = []
        for batch in train_loader:
            x_t = batch["x_t"].to(device)
            x_tp1 = batch["x_tp1"].to(device)
            a = batch["action"].to(device)
            o = batch["outcome"].to(device)
            inv = batch["invoke"].to(device)
            a_oh = F.one_hot(a, num_classes=N_ACTIONS).float()

            emb_t = model.encode(x_t)
            with torch.no_grad():
                emb_tp1 = model.encode(x_tp1)

            action_logits = model.forward_action(emb_t)
            L_a = F.cross_entropy(action_logits, a)

            next_pred = model.forward_dynamics(emb_t, a_oh)
            L_d = F.mse_loss(next_pred, emb_tp1)

            outcome_logit_pos = model.forward_outcome(emb_t, a_oh)
            L_o = F.binary_cross_entropy_with_logits(outcome_logit_pos, o)

            with torch.no_grad():
                neg_actions = torch.randint(
                    0, N_ACTIONS, size=a.shape, device=device
                )
                same = (neg_actions == a)
                while same.any():
                    neg_actions = torch.where(
                        same,
                        torch.randint(0, N_ACTIONS, size=a.shape, device=device),
                        neg_actions,
                    )
                    same = (neg_actions == a)
                neg_a_oh = F.one_hot(neg_actions, num_classes=N_ACTIONS).float()

            outcome_logit_neg = model.forward_outcome(emb_t, neg_a_oh)
            neg_target = torch.where(o >= 0.7, torch.full_like(o, 0.3),
                                     torch.full_like(o, 0.5))
            L_c = F.binary_cross_entropy_with_logits(outcome_logit_neg, neg_target)

            # v2: invoke head — BCE on self-supervised invoke label
            invoke_logit = model.forward_invoke(emb_t)
            L_i = F.binary_cross_entropy_with_logits(invoke_logit, inv)

            loss = alpha * L_a + beta * L_d + gamma * L_o + delta * L_c + epsilon * L_i
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(loss.item())
            epoch_L_a.append(L_a.item()); epoch_L_d.append(L_d.item())
            epoch_L_o.append(L_o.item()); epoch_L_c.append(L_c.item())
            epoch_L_i.append(L_i.item())

        mean_train_loss = float(np.mean(epoch_losses))
        val_metrics = evaluate(model, val_loader, device)
        history.setdefault("train_loss", []).append(mean_train_loss)
        history.setdefault("L_action", []).append(float(np.mean(epoch_L_a)))
        history.setdefault("L_dynamics", []).append(float(np.mean(epoch_L_d)))
        history.setdefault("L_outcome", []).append(float(np.mean(epoch_L_o)))
        history.setdefault("L_contrastive", []).append(float(np.mean(epoch_L_c)))
        history.setdefault("L_invoke", []).append(float(np.mean(epoch_L_i)))
        for k, v in val_metrics.items():
            history.setdefault(f"val_{k}", []).append(v)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  ep{epoch:3d}  loss={mean_train_loss:.3f}  "
                  f"acc={val_metrics['action_acc']:.3f}  "
                  f"dyn={val_metrics['dynamics_mse']:.3f}  "
                  f"out_au={val_metrics['outcome_auroc']:.3f}  "
                  f"spread={val_metrics['outcome_spread']:.3f}  "
                  f"inv_au={val_metrics['invoke_auroc']:.3f}  "
                  f"inv_PR={val_metrics['invoke_precision']:.2f}/{val_metrics['invoke_recall']:.2f}")

    return model, history


def evaluate(model: WorldModel, loader: DataLoader, device: str) -> Dict[str, float]:
    """v1 eval. Reports outcome_spread — the mean std of outcome_head
    predictions across all 7 actions for each state. High spread means
    the outcome head differentiates actions (planning signal is real).
    Low spread means the head collapses to a per-state constant
    (v0's pathology)."""
    model.eval()
    action_correct = 0
    action_total = 0
    dyn_mse_sum = 0.0
    dyn_count = 0
    trivial_dyn_mse_sum = 0.0
    outcome_preds_pos = []
    outcome_true = []
    outcome_spreads = []
    invoke_preds = []
    invoke_true = []
    with torch.no_grad():
        for batch in loader:
            x_t = batch["x_t"].to(device)
            x_tp1 = batch["x_tp1"].to(device)
            a = batch["action"].to(device)
            o = batch["outcome"].to(device)
            inv = batch["invoke"].to(device)
            a_oh = F.one_hot(a, num_classes=N_ACTIONS).float()
            B = x_t.size(0)

            emb_t = model.encode(x_t)
            emb_tp1 = model.encode(x_tp1)
            action_logits = model.forward_action(emb_t)
            next_pred = model.forward_dynamics(emb_t, a_oh)

            action_correct += (action_logits.argmax(dim=-1) == a).sum().item()
            action_total += a.numel()
            dyn_mse_sum += F.mse_loss(next_pred, emb_tp1, reduction="sum").item()
            trivial_dyn_mse_sum += F.mse_loss(emb_t, emb_tp1, reduction="sum").item()
            dyn_count += emb_tp1.numel()

            if model.outcome_action_conditional:
                outcome_logit_pos = model.forward_outcome(emb_t, a_oh)
                emb_rep = emb_t.unsqueeze(1).expand(-1, N_ACTIONS, -1)
                all_a_oh = torch.eye(N_ACTIONS, device=device).unsqueeze(0).expand(B, -1, -1)
                flat_emb = emb_rep.reshape(-1, emb_rep.size(-1))
                flat_a = all_a_oh.reshape(-1, N_ACTIONS)
                all_logits = model.forward_outcome(flat_emb, flat_a).reshape(B, N_ACTIONS)
                all_probs = torch.sigmoid(all_logits)
                spread = all_probs.std(dim=1)
                outcome_spreads.extend(spread.cpu().tolist())
            else:
                outcome_logit_pos = model.forward_outcome(emb_t)

            outcome_preds_pos.extend(torch.sigmoid(outcome_logit_pos).cpu().tolist())
            outcome_true.extend(o.cpu().tolist())

            # v2: invoke eval
            invoke_logit = model.forward_invoke(emb_t)
            invoke_preds.extend(torch.sigmoid(invoke_logit).cpu().tolist())
            invoke_true.extend(inv.cpu().tolist())

    auroc = _simple_auroc(outcome_preds_pos, outcome_true)
    # Invoke metrics — precision/recall at 0.5 threshold + AUROC
    inv_auroc = _simple_auroc(invoke_preds, invoke_true)
    tp = sum(1 for p, t in zip(invoke_preds, invoke_true) if p >= 0.5 and t >= 0.5)
    fp = sum(1 for p, t in zip(invoke_preds, invoke_true) if p >= 0.5 and t < 0.5)
    fn = sum(1 for p, t in zip(invoke_preds, invoke_true) if p < 0.5 and t >= 0.5)
    inv_precision = tp / max(tp + fp, 1)
    inv_recall = tp / max(tp + fn, 1)

    return {
        "action_acc": action_correct / max(action_total, 1),
        "dynamics_mse": dyn_mse_sum / max(dyn_count, 1),
        "trivial_dynamics_mse": trivial_dyn_mse_sum / max(dyn_count, 1),
        "outcome_auroc": auroc,
        "outcome_spread": float(np.mean(outcome_spreads)) if outcome_spreads else 0.0,
        "invoke_auroc": inv_auroc,
        "invoke_precision": inv_precision,
        "invoke_recall": inv_recall,
        "invoke_positive_rate": sum(invoke_true) / max(len(invoke_true), 1),
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
    p.add_argument("--gamma", type=float, default=0.5, help="outcome (positive) loss weight")
    p.add_argument("--delta", type=float, default=0.3, help="outcome contrastive loss weight (v1)")
    p.add_argument("--epsilon", type=float, default=0.3, help="invoke-head loss weight (v2)")
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
          f"(α={args.alpha} β={args.beta} γ={args.gamma} δ={args.delta} ε={args.epsilon})")
    model, history = train(
        train_ds, val_ds, n_games=len(game_slugs), n_levels=10,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, epsilon=args.epsilon,
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
        trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        machine=args.machine,
        train_record_count=len(train_pairs),
        val_metrics=final_val,
        architecture_version=2,
        outcome_action_conditional=True,
    )
    save_world_model(model.cpu(), cfg, Path(args.out))
    print(f"\nSaved adapter: {args.out}.pt + {args.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
