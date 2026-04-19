"""Frame-router — the thalamic router as a small CNN + bookkeeping MLP.

Inputs (router's MRH — "where we are / where we just were / bookkeeping"):
  - prev_frame:         one-hot (16, H, W); zeros at game start
  - curr_frame:         one-hot (16, H, W)
  - game_id:            one-hot
  - level:              one-hot
  - step_frac:          float in [0, 1]
  - action_budget:      remaining, normalized
  - recent_actions:     last K actions, one-hot sequence
  - available_actions:  per-game/level mask (binary)
  - batch_state:        placeholder {in_batch, progress, goal_dummy} — zeros
                        for now, reserved for future LLM-delegated action batches

Outputs:
  - action_head:    softmax over 7 actions (the hint)
  - invoke_head:    sigmoid (the dispatch decision)
  - dynamics_head:  auxiliary — predicts next CNN pool, used as surprise signal
                    (predicted-vs-actual divergence → elevated invoke next tick)

NOT inputs (deliberately excluded — these are LLM MRH, not router MRH):
  - SNARC scalars (CNN rediscovers them from raw frames)
  - membot cartridge / episodic memory / world models
  - cross-game knowledge

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-2-world-model-embedding-sprint.md
      (this module iterates past v0/v1/v2 world_model.py with the user-validated
      frame-based architecture)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical action space — matches GameAction enum
N_ACTIONS = 7
ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]

# Frame grid — pad all games to this. ARC-AGI-3 standard is 64×64.
FRAME_H = 64
FRAME_W = 64
N_COLORS = 16

# How many recent actions carry into the bookkeeping context
RECENT_ACTIONS_K = 8


def onehot_frame(frame: Any) -> np.ndarray:
    """Convert a frame (int array, shape (L,H,W) or (H,W)) to one-hot (16,H,W).
    If multi-layer, use the LAST layer (topmost render). Pads to FRAME_H × FRAME_W.

    Handles degenerate cases (empty array, 1-D, 0-sized) by returning all-zero
    one-hot — honest "I don't have a valid frame" signal. Observed on cn04/re86
    under SDK version-drift where env.step returns empty frames for some actions.
    """
    zero = np.zeros((N_COLORS, FRAME_H, FRAME_W), dtype=np.float32)
    if frame is None:
        return zero
    arr = np.asarray(frame, dtype=np.int64)
    if arr.size == 0:
        return zero
    if arr.ndim == 3:
        if arr.shape[0] == 0:
            return zero
        arr = arr[-1]   # last layer (topmost visual)
    if arr.ndim == 1:
        # 1-D degenerate frame — can't meaningfully reshape without knowing H,W.
        # Pad into a row-vector at the top of the canonical grid.
        row = arr[:FRAME_W]
        padded = np.zeros((FRAME_H, FRAME_W), dtype=np.int64)
        padded[0, :len(row)] = row
        arr = padded
    elif arr.ndim != 2:
        # Unknown dimensionality — be conservative
        return zero
    h, w = arr.shape
    if h == 0 or w == 0:
        return zero
    # Pad to canonical shape
    padded = np.zeros((FRAME_H, FRAME_W), dtype=np.int64)
    padded[:min(h, FRAME_H), :min(w, FRAME_W)] = arr[:FRAME_H, :FRAME_W]
    oh = np.zeros((N_COLORS, FRAME_H, FRAME_W), dtype=np.float32)
    padded = np.clip(padded, 0, N_COLORS - 1)
    for c in range(N_COLORS):
        oh[c] = (padded == c).astype(np.float32)
    return oh


class FrameCNN(nn.Module):
    """Small CNN on stacked (prev + curr) one-hot frames.

    Input:  (B, 32, H, W)   — 16 colors × 2 frames
    Output: (B, pool_dim)   — pooled visual feature
    """
    def __init__(self, pool_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(N_COLORS * 2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),   # 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                  # 64 × 4 × 4
        )
        self.fc = nn.Linear(64 * 4 * 4, pool_dim)

    def forward(self, prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        # prev, curr: (B, 16, H, W). Stack along channel dim.
        x = torch.cat([prev, curr], dim=1)
        z = self.net(x)
        z = z.reshape(z.size(0), -1)
        return self.fc(z)


class FrameRouter(nn.Module):
    """Frame-driven router.

    - CNN processes (prev + curr) frames
    - MLP processes scalar bookkeeping
    - Concat → trunk → embedding
    - Heads: action (7-way), invoke (1), dynamics (predicts next CNN pool)
    """
    def __init__(
        self,
        n_games: int,
        n_levels: int = 10,
        pool_dim: int = 64,
        scalar_hidden: int = 32,
        emb_dim: int = 32,
        trunk_hidden: int = 64,
        include_last_action: bool = True,   # v4+ default; v3 configs → False
        use_mech_embedding: bool = False,   # v6+ — game context from mechanics encoder
        mech_emb_dim: int = 32,             # mechanics encoder output dim
    ):
        super().__init__()
        self.n_games = n_games
        self.n_levels = n_levels
        self.emb_dim = emb_dim
        self.pool_dim = pool_dim
        self.include_last_action = include_last_action
        self.use_mech_embedding = use_mech_embedding
        self.mech_emb_dim = mech_emb_dim

        # Scalar bookkeeping input dim:
        #   game_context (n_games one-hot OR mech_emb_dim mechanics-encoder embedding)
        #   + level (n_levels) + step_frac(1) + budget(1)
        #   + last_action (N_ACTIONS) ← action that caused curr_frame [v4+]
        #   + recent_actions (K * N_ACTIONS) ← older context
        #   + available_actions (N_ACTIONS)
        #   + batch_state (3: in_batch, progress, goal_dummy)
        last_action_dim = N_ACTIONS if include_last_action else 0
        game_context_dim = mech_emb_dim if use_mech_embedding else n_games
        self.game_context_dim = game_context_dim
        self.scalar_dim = (
            game_context_dim + n_levels + 1 + 1 +
            last_action_dim +
            RECENT_ACTIONS_K * N_ACTIONS +
            N_ACTIONS + 3
        )

        # Mechanics-encoder embedding table — frozen buffer, loaded at construct time.
        # If use_mech_embedding=True, expected to be populated via load_mech_embedding_table().
        # Kept as a buffer (not Parameter) so gradient doesn't flow — preserves the
        # cross-game transfer semantics learned by the mechanics encoder.
        if use_mech_embedding:
            self.register_buffer(
                "mech_embedding_table",
                torch.zeros(n_games, mech_emb_dim, dtype=torch.float32),
            )

        self.cnn = FrameCNN(pool_dim=pool_dim)
        self.scalar_net = nn.Sequential(
            nn.Linear(self.scalar_dim, scalar_hidden),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(pool_dim + scalar_hidden, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, emb_dim),
        )

        self.action_head = nn.Linear(emb_dim, N_ACTIONS)
        self.invoke_head = nn.Linear(emb_dim, 1)
        # Dynamics: (emb, action_onehot) → predicted next CNN pool
        self.dynamics_head = nn.Sequential(
            nn.Linear(emb_dim + N_ACTIONS, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, pool_dim),
        )

    def encode(
        self, prev_frame: torch.Tensor, curr_frame: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, visual_pool) — visual_pool is used as dynamics target."""
        visual = self.cnn(prev_frame, curr_frame)
        scalar = self.scalar_net(scalars)
        emb = self.trunk(torch.cat([visual, scalar], dim=-1))
        return emb, visual

    def forward(
        self, prev_frame: torch.Tensor, curr_frame: torch.Tensor,
        scalars: torch.Tensor, action_onehot: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        emb, visual = self.encode(prev_frame, curr_frame, scalars)
        out = {
            "embedding": emb,
            "visual_pool": visual,
            "action_logits": self.action_head(emb),
            "invoke_logit": self.invoke_head(emb).squeeze(-1),
        }
        if action_onehot is not None:
            out["next_visual_pred"] = self.dynamics_head(
                torch.cat([emb, action_onehot], dim=-1)
            )
        return out


def build_scalar_vector(
    game_idx: int, n_games: int,
    level: int, n_levels: int,
    step_frac: float,
    budget_remaining: float,
    last_action: int,
    recent_actions: List[int],
    available_actions: List[int],
    batch_state: List[float],
    include_last_action: bool = True,
    mech_embedding: Optional[List[float]] = None,
) -> List[float]:
    """Construct the scalar bookkeeping vector for one example.

    `last_action` is the action that caused the transition prev_frame →
    curr_frame. At step 1 there is no last action → pass 0 (sentinel);
    combined with prev_frame=zeros this gives the NN a consistent
    "no-prior-context" fingerprint.

    `include_last_action=False` reproduces the v3 layout (for loading
    Thor's v1 adapter trained before the explicit last_action field).

    `mech_embedding` — if provided, replaces the n_games one-hot with the
    mechanics-encoder embedding (v6+). Caller is responsible for looking up
    the embedding by game_idx (typically from FrameRouter.mech_embedding_table).
    """
    if mech_embedding is not None:
        game_oh = list(mech_embedding)
    else:
        game_oh = [1.0 if i == game_idx else 0.0 for i in range(n_games)]
    level_c = max(0, min(n_levels - 1, level if level is not None else 0))
    level_oh = [1.0 if i == level_c else 0.0 for i in range(n_levels)]

    # recent_actions: pad/trim to K. Older actions come first, newest last.
    padded = list(recent_actions[-RECENT_ACTIONS_K:])
    while len(padded) < RECENT_ACTIONS_K:
        padded.insert(0, 0)
    recent_oh: List[float] = []
    for a in padded:
        recent_oh.extend(
            [1.0 if i == a else 0.0 for i in range(N_ACTIONS)]
        )
    avail = [float(b) for b in available_actions[:N_ACTIONS]]
    while len(avail) < N_ACTIONS:
        avail.append(1.0)
    batch = [float(v) for v in batch_state[:3]]
    while len(batch) < 3:
        batch.append(0.0)

    out = game_oh + level_oh + [float(step_frac), float(budget_remaining)]
    if include_last_action:
        last_action_oh = [1.0 if i == last_action else 0.0 for i in range(N_ACTIONS)]
        out = out + last_action_oh
    out = out + recent_oh + avail + batch
    return out


@dataclass
class FrameRouterConfig:
    """Serialize-able config; saved alongside weights."""
    n_games: int
    game_slugs: List[str]
    n_levels: int = 10
    pool_dim: int = 64
    scalar_hidden: int = 32
    emb_dim: int = 32
    trunk_hidden: int = 64
    trained_at: Optional[str] = None
    train_commit: Optional[str] = None
    machine: Optional[str] = None
    train_record_count: Optional[int] = None
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    architecture_version: int = 4   # v3 = frame-based; v4 = + explicit last_action; v6 = + mech_embedding
    include_last_action: bool = True  # v3 adapters → False; v4+ → True
    use_mech_embedding: bool = False  # v6+ — game context from mechanics encoder
    mech_emb_dim: int = 32
    mech_encoder_path: Optional[str] = None   # path to mechanics encoder .json (for provenance)
    frame_h: int = FRAME_H
    frame_w: int = FRAME_W
    n_colors: int = N_COLORS
    recent_actions_k: int = RECENT_ACTIONS_K


def save_frame_router(model: FrameRouter, cfg: FrameRouterConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path) + ".pt")
    with open(str(path) + ".json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_frame_router(path: Path) -> Tuple[FrameRouter, FrameRouterConfig]:
    path = Path(path)
    with open(str(path) + ".json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Forward-compat for v3 configs that predate include_last_action / mech_embedding
    if "include_last_action" not in raw:
        raw["include_last_action"] = raw.get("architecture_version", 3) >= 4
    raw.setdefault("use_mech_embedding", False)
    raw.setdefault("mech_emb_dim", 32)
    raw.setdefault("mech_encoder_path", None)
    cfg = FrameRouterConfig(**raw)
    model = FrameRouter(
        n_games=cfg.n_games, n_levels=cfg.n_levels,
        pool_dim=cfg.pool_dim, scalar_hidden=cfg.scalar_hidden,
        emb_dim=cfg.emb_dim, trunk_hidden=cfg.trunk_hidden,
        include_last_action=cfg.include_last_action,
        use_mech_embedding=cfg.use_mech_embedding,
        mech_emb_dim=cfg.mech_emb_dim,
    )
    state = torch.load(str(path) + ".pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_mech_embedding_table(
    mech_encoder_json_path: Path, game_slugs: List[str], mech_emb_dim: int = 32,
) -> np.ndarray:
    """Load the mechanics-encoder embedding table, aligned to the router's
    `game_slugs` order. Missing games get zero embeddings (silently — they're
    rare and the CNN can still drive action choice).

    Returns (n_games, mech_emb_dim) float32 numpy array.
    """
    with open(mech_encoder_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    mech_slugs = meta["game_slugs"]
    mech_embs = np.array(meta["embeddings"], dtype=np.float32)
    assert mech_embs.shape == (len(mech_slugs), mech_emb_dim), (
        f"mechanics encoder shape mismatch: {mech_embs.shape} vs "
        f"({len(mech_slugs)}, {mech_emb_dim})"
    )
    slug_to_emb = {s: mech_embs[i] for i, s in enumerate(mech_slugs)}
    table = np.zeros((len(game_slugs), mech_emb_dim), dtype=np.float32)
    for i, slug in enumerate(game_slugs):
        if slug in slug_to_emb:
            table[i] = slug_to_emb[slug]
    return table


# ───────────────────────────────────────────────────────────────────
# Invoke label (self-supervised, frame-to-frame)
# ───────────────────────────────────────────────────────────────────

PIXEL_DIFF_INVOKE_THRESHOLD = 0.1   # 10% of cells changed → invoke


def compute_invoke_label(
    prev_frame: np.ndarray, curr_frame: np.ndarray,
    step_index: int, level: int, prev_level: Optional[int],
    pixel_diff_threshold: float = PIXEL_DIFF_INVOKE_THRESHOLD,
) -> float:
    """Self-supervised label: should the NN invoke the LLM at this state?

    Fires on:
      - first frame of trace (step_index == 1; prev_frame is all-zeros)
      - level just changed (new geometry/rules)
      - frame-to-frame pixel diff rate ≥ threshold

    All three are observable from the two frames + bookkeeping — no cheating.
    """
    if step_index == 1:
        return 1.0
    if prev_level is not None and level != prev_level:
        return 1.0
    # Raw pixel diff; works on multi-layer frames by using last layer.
    a = np.asarray(prev_frame)
    b = np.asarray(curr_frame)
    if a.ndim == 3:
        a = a[-1] if a.shape[0] > 0 else a.reshape(0, 0)
    if b.ndim == 3:
        b = b[-1] if b.shape[0] > 0 else b.reshape(0, 0)
    if a.size == 0 and b.size == 0:
        return 0.0     # both empty — no signal either way
    if a.size == 0 or b.size == 0:
        return 1.0     # one empty, one not — shape/state change is a huge signal
    if a.shape != b.shape:
        return 1.0
    diff_rate = float((a != b).mean())
    return 1.0 if diff_rate >= pixel_diff_threshold else 0.0
