"""World-model embedding encoder + three-head architecture.

The embedding IS the world-model core. Three heads trained jointly
force it to be simultaneously (a) action-predictive, (b) predictable
under actions (simulator), (c) outcome-discriminating (winning vs
losing contexts).

Inference: forward-simulate action candidates via dynamics_head, score
via outcome_head, argmax. Consciousness loop in miniature.

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-2-world-model-embedding-sprint.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Action space — matches Head B
N_ACTIONS = 7                     # 0=A0, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=SEL, 6=CLICK
ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]

# Feature layout — must match what _feature_vec produces from a RouterInput
BASE_FEATURE_NAMES = [
    "snarc_surprise", "snarc_novelty", "snarc_arousal",
    "snarc_reward", "snarc_conflict",
    "sensory_novelty", "sensory_urgency", "atp_norm",
    "wm_goal_active", "wm_pressure",
    "habit_available", "habit_confidence",
    "has_audio", "has_message", "has_vision",
    "metabolic_level",
]
N_BASE_FEATURES = len(BASE_FEATURE_NAMES)    # 16

# Default embedding dim — 32 is enough for this data scale (~12k records)
DEFAULT_EMB_DIM = 32
DEFAULT_HIDDEN = 64


class WorldModel(nn.Module):
    """Encoder + three heads. Shared embedding trained jointly.

    v1 change: outcome head is ACTION-CONDITIONAL.
      outcome_head(embedding, action_onehot) → P(win | state, action)
    This breaks the source-discrimination shortcut v0 fell into (where
    outcome_head(embedding) could simply learn "which source was this
    record" because gameplay records all have outcome=1 and self-play
    GAME_OVER records all have outcome=0).

    v0 backward-compat: if outcome_action_conditional=False, uses the
    old outcome_head(embedding) signature.
    """

    def __init__(
        self,
        n_games: int,
        n_levels: int = 10,
        n_base_features: int = N_BASE_FEATURES,
        emb_dim: int = DEFAULT_EMB_DIM,
        hidden: int = DEFAULT_HIDDEN,
        n_actions: int = N_ACTIONS,
        outcome_action_conditional: bool = True,
    ):
        super().__init__()
        self.n_games = n_games
        self.n_levels = n_levels
        self.emb_dim = emb_dim
        self.n_actions = n_actions
        self.outcome_action_conditional = outcome_action_conditional

        # Input dimension: base features + game one-hot + level one-hot
        #   + step_frac (1) + last_action one-hot
        self.input_dim = (
            n_base_features + n_games + n_levels + 1 + n_actions
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

        self.action_head = nn.Linear(emb_dim, n_actions)

        if outcome_action_conditional:
            # v1: outcome depends on the specific action taken
            self.outcome_head = nn.Sequential(
                nn.Linear(emb_dim + n_actions, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            # v0 backward-compat
            self.outcome_head = nn.Linear(emb_dim, 1)

        # Dynamics: (emb + action one-hot) → next embedding
        self.dynamics_head = nn.Sequential(
            nn.Linear(emb_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_action(self, emb: torch.Tensor) -> torch.Tensor:
        return self.action_head(emb)

    def forward_dynamics(
        self, emb: torch.Tensor, action_onehot: torch.Tensor
    ) -> torch.Tensor:
        return self.dynamics_head(torch.cat([emb, action_onehot], dim=-1))

    def forward_outcome(
        self, emb: torch.Tensor, action_onehot: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns logits (pre-sigmoid). In v1 (action-conditional), requires
        action_onehot. In v0 mode, action_onehot is ignored."""
        if self.outcome_action_conditional:
            if action_onehot is None:
                raise ValueError("v1 outcome head requires action_onehot")
            return self.outcome_head(
                torch.cat([emb, action_onehot], dim=-1)
            ).squeeze(-1)
        return self.outcome_head(emb).squeeze(-1)

    def forward(
        self, x: torch.Tensor, action_onehot: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass. action_onehot required for outcome in v1."""
        emb = self.encode(x)
        out = {
            "embedding": emb,
            "action_logits": self.forward_action(emb),
        }
        if action_onehot is not None:
            out["outcome_logit"] = self.forward_outcome(emb, action_onehot)
            out["next_emb_pred"] = self.forward_dynamics(emb, action_onehot)
        elif not self.outcome_action_conditional:
            out["outcome_logit"] = self.forward_outcome(emb)
        return out


def build_input_vector(
    base_features: List[float],
    game_idx: int, n_games: int,
    level: int, n_levels: int,
    step_frac: float,
    last_action: int, n_actions: int = N_ACTIONS,
) -> List[float]:
    """Build the input vector by concatenating all inputs. Pure Python
    so tests + inference don't need torch-on-numpy for data prep."""
    game_oh = [1.0 if i == game_idx else 0.0 for i in range(n_games)]
    level_clamped = max(0, min(n_levels - 1, level if level is not None else 0))
    level_oh = [1.0 if i == level_clamped else 0.0 for i in range(n_levels)]
    action_oh = [1.0 if i == last_action else 0.0 for i in range(n_actions)]
    return list(base_features) + game_oh + level_oh + [float(step_frac)] + action_oh


def action_onehot(action: int, n_actions: int = N_ACTIONS) -> List[float]:
    return [1.0 if i == action else 0.0 for i in range(n_actions)]


@dataclass
class WorldModelConfig:
    """Serialize-able model config. Saved alongside weights."""
    n_games: int
    game_slugs: List[str]
    n_levels: int = 10
    n_base_features: int = N_BASE_FEATURES
    emb_dim: int = DEFAULT_EMB_DIM
    hidden: int = DEFAULT_HIDDEN
    n_actions: int = N_ACTIONS
    feature_mean: List[float] = field(default_factory=list)
    feature_std: List[float] = field(default_factory=list)
    trained_at: Optional[str] = None
    train_commit: Optional[str] = None
    machine: Optional[str] = None
    train_record_count: Optional[int] = None
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    # Architecture version. 0 = outcome_head(emb). 1 = outcome_head(emb, action)
    architecture_version: int = 1
    outcome_action_conditional: bool = True


def save_world_model(
    model: WorldModel, cfg: WorldModelConfig, path: Path
) -> None:
    """Persist weights + config. Two files: {path}.pt + {path}.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path) + ".pt")
    with open(str(path) + ".json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_world_model(path: Path) -> Tuple[WorldModel, WorldModelConfig]:
    """Load model + config. `path` is the base (no .pt/.json suffix)."""
    path = Path(path)
    with open(str(path) + ".json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Forward-compat: v0 configs don't have these fields
    raw.setdefault("architecture_version", 0)
    raw.setdefault("outcome_action_conditional", False)
    cfg = WorldModelConfig(**raw)
    model = WorldModel(
        n_games=cfg.n_games, n_levels=cfg.n_levels,
        n_base_features=cfg.n_base_features,
        emb_dim=cfg.emb_dim, hidden=cfg.hidden,
        n_actions=cfg.n_actions,
        outcome_action_conditional=cfg.outcome_action_conditional,
    )
    state = torch.load(str(path) + ".pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg
