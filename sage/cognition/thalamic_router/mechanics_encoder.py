"""Mechanics encoder — learned per-game embedding organized by structural similarity.

Concept: one shared dynamics network across all games, with a per-game learned
embedding as the ONLY per-game parameter. The embedding must compress game
mechanics because it's the only way the shared network can predict different
games' next states accurately.

Architecture:
    Shared frame CNN (reuses FrameRouter's CNN structure) → 64d visual pool
    Per-game embedding:  game_id → ℝ^32 (nn.Embedding)

    Dynamics network:
        (visual_pool_t, action_onehot, game_embedding) → visual_pool_{t+1}

    Text-anchor projection (regularization):
        game_embedding → Linear → 768d  —   MSE against  nomic_embed(game_text)

Training:
    L = α · MSE(dynamics_pred, pool_{t+1}.detach())
      + β · MSE(proj(game_embedding), nomic(text[game]))

    where text[game] = concatenated cartridge entries OR world-model markdown
    fallback.

Inference:
    game_embedding[g] is a 32d vector per trained game.
    nearest_games(g, k=3): cosine similarity over all game embeddings.

    At invoke time the router's LLM prompt gets: "this game's mechanics are
    near {g1, g2, g3}" → structural prior, not 10KB of prose per-invocation.

Novel games:
    Initialize new embedding as centroid, few-shot-adapt on K trajectories,
    keep everything else frozen.

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-3-mechanics-encoder-and-stack.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sage.cognition.thalamic_router.frame_router import (
    FrameCNN, onehot_frame,
    N_ACTIONS, FRAME_H, FRAME_W, N_COLORS,
)
from sage.cognition.thalamic_router.gameplay_capture import (
    load_trace, _discover_trace, ARC_AGI_EXPERIMENTS,
)


DEFAULT_EMB_DIM = 32
DEFAULT_POOL_DIM = 64
NOMIC_EMB_DIM = 768   # nomic-embed-text output dimension


# ───────────────────────────────────────────────────────────────────
# The model
# ───────────────────────────────────────────────────────────────────

class MechanicsEncoder(nn.Module):
    """Shared dynamics network + per-game embedding + text-anchor projection."""

    def __init__(
        self,
        n_games: int,
        emb_dim: int = DEFAULT_EMB_DIM,
        pool_dim: int = DEFAULT_POOL_DIM,
        hidden: int = 128,
        text_emb_dim: int = NOMIC_EMB_DIM,
    ):
        super().__init__()
        self.n_games = n_games
        self.emb_dim = emb_dim
        self.pool_dim = pool_dim

        # Reuse the frame CNN from FrameRouter
        self.cnn = FrameCNN(pool_dim=pool_dim)

        # Per-game embedding — the structural signature
        self.game_embedding = nn.Embedding(n_games, emb_dim)
        nn.init.normal_(self.game_embedding.weight, std=0.02)

        # Dynamics: (pool_t, action_oh, game_emb) → pool_{t+1}
        self.dynamics = nn.Sequential(
            nn.Linear(pool_dim + N_ACTIONS + emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pool_dim),
        )

        # Text-anchor projection
        self.text_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, text_emb_dim),
        )

    def encode_frames(self, prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        return self.cnn(prev, curr)

    def predict_next_pool(
        self, pool_t: torch.Tensor, action_oh: torch.Tensor,
        game_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([pool_t, action_oh, game_emb], dim=-1)
        return self.dynamics(x)

    def game_text_projection(self, game_emb: torch.Tensor) -> torch.Tensor:
        return self.text_proj(game_emb)


# ───────────────────────────────────────────────────────────────────
# Text anchor — nomic-embed-text via Ollama
# ───────────────────────────────────────────────────────────────────

def nomic_embed_text(text: str, base_url: str = "http://localhost:11434",
                    model: str = "nomic-embed-text") -> Optional[np.ndarray]:
    """Query Ollama's nomic-embed-text endpoint. Returns 768d vector or None."""
    import urllib.request, urllib.error
    payload = {"model": model, "prompt": text}
    req = urllib.request.Request(
        f"{base_url}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as r:
            body = json.loads(r.read().decode("utf-8"))
        emb = body.get("embedding") or body.get("embeddings", [None])[0]
        if emb is None:
            return None
        return np.array(emb, dtype=np.float32)
    except Exception:
        return None


def load_world_model_text(game_family: str, max_chars: int = 3000) -> str:
    """Load full world-model markdown for text-anchor. More text than the invoke-
    time summary (which was 1500 chars) — we want to capture more semantic
    detail for the anchor.
    """
    for root in [
        Path(os.environ.get("SHARED_CONTEXT_DIR", "")),
        Path("/mnt/c/exe/projects/ai-agents/shared-context"),
        Path.home() / "ai-workspace" / "shared-context",
        Path.home() / "repos" / "shared-context",
    ]:
        wm_path = root / "arc-agi-3" / "world-models" / f"{game_family}.md"
        if wm_path.exists():
            break
    else:
        return ""
    try:
        text = wm_path.read_text()
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception:
        return ""


# ───────────────────────────────────────────────────────────────────
# Trajectory data — reuses frame_train's replay logic
# ───────────────────────────────────────────────────────────────────

def replay_trace_for_mechanics(
    trace, game_idx: int,
) -> List[Dict[str, Any]]:
    """Replay a trace and emit (prev_frame_oh, curr_frame_oh, action, next_curr_oh, game_idx)
    tuples. Simpler than frame_train's output — no scalar bookkeeping.
    """
    if str(ARC_AGI_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(ARC_AGI_EXPERIMENTS))
    from arc_agi import Arcade
    arc = Arcade(operation_mode="offline")
    env = arc.make(trace.game_id) or arc.make(trace.game)
    if env is None:
        return []
    try:
        from arcengine import GameAction
        int_to_action = {ga.value: ga for ga in GameAction}
    except Exception:
        int_to_action = {}

    fd = env.reset()
    zero_oh = np.zeros((N_COLORS, FRAME_H, FRAME_W), dtype=np.float32)
    prev_oh = zero_oh
    tuples: List[Dict[str, Any]] = []
    for step in trace.steps:
        curr_raw = getattr(fd, "frame", None)
        curr_oh = onehot_frame(curr_raw) if curr_raw is not None else zero_oh
        # Advance
        ga = int_to_action.get(step.action)
        try:
            if ga is not None:
                fd = env.step(ga, data=step.data) if step.data else env.step(ga)
        except Exception:
            break
        next_raw = getattr(fd, "frame", None)
        next_oh = onehot_frame(next_raw) if next_raw is not None else zero_oh

        if 0 <= step.action < N_ACTIONS:
            tuples.append({
                "prev_oh": prev_oh, "curr_oh": curr_oh,
                "next_oh": next_oh,
                "action": int(step.action),
                "game_idx": int(game_idx),
            })
        prev_oh = curr_oh
    return tuples


def build_mechanics_dataset(
    games: List[str], arc_sage_root: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    coord = json.loads((arc_sage_root / "knowledge" / "game_coordination.json").read_text())
    id_by_family = {g["family"]: g["id"] for g in coord.get("games", [])}

    game_slugs = sorted(games)
    all_tuples: List[Dict[str, Any]] = []
    for family in game_slugs:
        gid = id_by_family.get(family)
        if not gid:
            print(f"  [skip] no coord entry for {family}"); continue
        tp = _discover_trace(family, gid)
        if not tp or not tp.exists():
            print(f"  [skip] no trace for {family}"); continue
        try:
            trace = load_trace(tp, family, gid)
        except Exception as e:
            print(f"  [skip] {family}: {e!r}"); continue
        gi = game_slugs.index(family)
        try:
            tuples = replay_trace_for_mechanics(trace, gi)
        except Exception as e:
            print(f"  [skip] {family} replay: {e!r}"); continue
        all_tuples.extend(tuples)
        print(f"  {family:8s}: {len(tuples)} tuples")
    return all_tuples, game_slugs


# ───────────────────────────────────────────────────────────────────
# Torch Dataset + training
# ───────────────────────────────────────────────────────────────────

class MechanicsDataset(Dataset):
    def __init__(self, tuples: List[Dict[str, Any]]):
        self.tuples = tuples

    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.tuples[idx]
        return {
            "prev_oh": torch.from_numpy(t["prev_oh"]),
            "curr_oh": torch.from_numpy(t["curr_oh"]),
            "next_oh": torch.from_numpy(t["next_oh"]),
            "action": torch.tensor(t["action"], dtype=torch.long),
            "game_idx": torch.tensor(t["game_idx"], dtype=torch.long),
        }


def train(
    dataset: MechanicsDataset, n_games: int,
    text_anchors: Optional[np.ndarray],
    epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
    alpha: float = 1.0, beta: float = 0.3,
    device: str = "cpu", verbose: bool = True,
) -> Tuple[MechanicsEncoder, Dict[str, Any]]:
    """Train the mechanics encoder. text_anchors is (n_games, 768) or None."""
    model = MechanicsEncoder(n_games=n_games).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    text_anchors_t: Optional[torch.Tensor] = None
    anchor_mask: Optional[torch.Tensor] = None
    if text_anchors is not None:
        text_anchors_t = torch.from_numpy(text_anchors).to(device)
        # Mask out games without an anchor (zero vector convention)
        anchor_mask = (text_anchors_t.norm(dim=-1) > 1e-6).float()

    history: Dict[str, List[float]] = {}
    for epoch in range(epochs):
        model.train()
        losses, L_d_sum, L_a_sum = [], [], []
        for batch in loader:
            prev = batch["prev_oh"].to(device)
            curr = batch["curr_oh"].to(device)
            nxt = batch["next_oh"].to(device)
            a = batch["action"].to(device)
            g = batch["game_idx"].to(device)
            a_oh = F.one_hot(a, num_classes=N_ACTIONS).float()

            pool_t = model.encode_frames(prev, curr)
            with torch.no_grad():
                pool_tp1 = model.encode_frames(curr, nxt)

            game_emb = model.game_embedding(g)
            pred = model.predict_next_pool(pool_t, a_oh, game_emb)
            L_d = F.mse_loss(pred, pool_tp1)

            loss = alpha * L_d
            L_a_val = 0.0
            if text_anchors_t is not None:
                proj = model.game_text_projection(model.game_embedding.weight)  # (n_games, 768)
                diff = (proj - text_anchors_t).pow(2).mean(dim=-1)               # (n_games,)
                if anchor_mask is not None:
                    L_a = (diff * anchor_mask).sum() / (anchor_mask.sum() + 1e-8)
                else:
                    L_a = diff.mean()
                loss = loss + beta * L_a
                L_a_val = float(L_a.item())

            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            L_d_sum.append(float(L_d.item()))
            L_a_sum.append(L_a_val)

        mean_loss = float(np.mean(losses))
        mean_ld = float(np.mean(L_d_sum))
        mean_la = float(np.mean(L_a_sum)) if text_anchors is not None else 0.0
        history.setdefault("loss", []).append(mean_loss)
        history.setdefault("L_dynamics", []).append(mean_ld)
        history.setdefault("L_anchor", []).append(mean_la)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  ep{epoch:3d}  loss={mean_loss:.3f}  L_dyn={mean_ld:.3f}  L_anchor={mean_la:.3f}")

    return model, history


# ───────────────────────────────────────────────────────────────────
# Similarity lookup
# ───────────────────────────────────────────────────────────────────

def nearest_games(
    query_idx: int, embeddings: np.ndarray, game_slugs: List[str], k: int = 3,
) -> List[Tuple[str, float]]:
    """Return top-k nearest games to `query_idx` by cosine similarity."""
    q = embeddings[query_idx]
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    all_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = all_norms @ q_norm
    sims[query_idx] = -np.inf   # exclude self
    top = np.argsort(sims)[::-1][:k]
    return [(game_slugs[i], float(sims[i])) for i in top]


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
    raise FileNotFoundError("ARC-SAGE not found")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="all",
                   help="Comma-separated game slugs or 'all'")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1.0, help="dynamics MSE weight")
    p.add_argument("--beta", type=float, default=0.3, help="text-anchor MSE weight")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True, help="Output base path (no ext)")
    p.add_argument("--no-text-anchor", action="store_true",
                   help="Skip text-anchor term (no Ollama dependency)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    arc_sage = _resolve_arc_sage()
    if args.games == "all":
        coord = json.loads((arc_sage / "knowledge" / "game_coordination.json").read_text())
        games = [g["family"] for g in coord.get("games", [])]
    else:
        games = [g.strip() for g in args.games.split(",") if g.strip()]

    print(f"Games requested: {games}")
    t0 = time.time()
    tuples, game_slugs = build_mechanics_dataset(games, arc_sage)
    print(f"Built {len(tuples)} tuples across {len(game_slugs)} games in {time.time()-t0:.1f}s")
    if len(tuples) < 100:
        print("ERROR: need ≥100 tuples"); return 2

    # Text anchors
    text_anchors = None
    if not args.no_text_anchor:
        print("Computing text anchors via nomic-embed-text...")
        anchors = []
        for g in game_slugs:
            wm_text = load_world_model_text(g)
            if wm_text:
                emb = nomic_embed_text(wm_text)
                if emb is not None:
                    anchors.append(emb); continue
            anchors.append(np.zeros(NOMIC_EMB_DIM, dtype=np.float32))
        text_anchors = np.array(anchors)
        n_with_anchor = sum(1 for a in anchors if np.linalg.norm(a) > 1e-6)
        print(f"  anchors computed: {n_with_anchor}/{len(game_slugs)} games")

    dataset = MechanicsDataset(tuples)
    print(f"\nTraining on {args.device} — {args.epochs} epochs (α={args.alpha} β={args.beta})")
    model, history = train(
        dataset, n_games=len(game_slugs), text_anchors=text_anchors,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        alpha=args.alpha, beta=args.beta, device=args.device,
    )

    # Export embeddings + metadata
    model.cpu().eval()
    with torch.no_grad():
        embeddings = model.game_embedding.weight.detach().cpu().numpy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), str(out) + ".pt")

    # Save embeddings + neighbors as JSON
    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    for i, slug in enumerate(game_slugs):
        neighbors[slug] = nearest_games(i, embeddings, game_slugs, k=3)

    meta = {
        "architecture": "mechanics_encoder_v1",
        "n_games": len(game_slugs),
        "game_slugs": game_slugs,
        "emb_dim": DEFAULT_EMB_DIM,
        "pool_dim": DEFAULT_POOL_DIM,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_record_count": len(tuples),
        "final_losses": {k: v[-1] for k, v in history.items()},
        "embeddings": embeddings.tolist(),
        "nearest_neighbors": neighbors,
        "has_text_anchor": text_anchors is not None,
    }
    with open(str(out) + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved: {out}.pt + {out}.json")

    print("\n=== Nearest-neighbor map (top-3 per game) ===")
    for slug in game_slugs:
        ngs = neighbors[slug]
        neigh_str = ", ".join(f"{n} ({s:.2f})" for n, s in ngs)
        print(f"  {slug:8s} ↔ {neigh_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
