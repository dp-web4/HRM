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

        # v2: invoke head — dispatch decision. Outputs P(invoke LLM | state).
        # Trained with self-supervised labels derived from SNARC + frame-delta
        # + first-frame signals. The NN is NOT the thing that plays in the end;
        # it's the triage layer that decides when to escalate to the LLM with
        # a context+hint package (see choose_dispatch in this module).
        self.invoke_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_action(self, emb: torch.Tensor) -> torch.Tensor:
        return self.action_head(emb)

    def forward_dynamics(
        self, emb: torch.Tensor, action_onehot: torch.Tensor
    ) -> torch.Tensor:
        return self.dynamics_head(torch.cat([emb, action_onehot], dim=-1))

    def forward_invoke(self, emb: torch.Tensor) -> torch.Tensor:
        """Returns logit for invoke decision (pre-sigmoid)."""
        return self.invoke_head(emb).squeeze(-1)

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


# ───────────────────────────────────────────────────────────────────
# v2: Dispatch decision — the actual router output
# ───────────────────────────────────────────────────────────────────

INVOKE_THRESHOLD = 0.5                # invoke_head probability threshold
PLAY_CONFIDENCE_THRESHOLD = 0.65      # top-action outcome floor; below this, force invoke
PLAY_MARGIN_THRESHOLD = 0.08          # gap between top and 2nd action; below this, uncertain → invoke


def choose_dispatch(
    model: "WorldModel",
    x: torch.Tensor,
    game: Optional[str] = None,
    level: Optional[int] = None,
    step_index: Optional[int] = None,
    invoke_threshold: float = INVOKE_THRESHOLD,
    play_confidence_threshold: float = PLAY_CONFIDENCE_THRESHOLD,
    play_margin_threshold: float = PLAY_MARGIN_THRESHOLD,
) -> Dict[str, Any]:
    """Produce the router's dispatch output: invoke OR play, with context + hint.

    Two-gate invoke decision:
      1. Structural: invoke_head says "novelty / first-frame / large diff"
      2. Confidence: top-action outcome < play_confidence_threshold, OR
                     margin between top and 2nd action < play_margin_threshold.
                     Low confidence means the NN isn't sure what to do — dispatch.

    If either gate fires → invoke (emit context+hint package for LLM).
    Otherwise → play (NN commits to its top action).

    Returns a dict with:
      - decision:      "invoke" | "play"
      - invoke_reasons: list of triggers (["structural"], ["confidence"], both, or [])
      - invoke_prob:    float — invoke_head's raw probability
      - play_action:    int — action class the NN would play (always populated)
      - play_confidence: float — top action's outcome score
      - play_margin:    float — gap between top and 2nd action
      - action_ranking: sorted list of (idx, outcome_score)
      - action_prior:   supervised action-head softmax
      - context:        what to hand to LLM when invoking
    """
    model.eval()
    with torch.no_grad():
        emb = model.encode(x.unsqueeze(0))                           # (1, D)
        invoke_prob = torch.sigmoid(model.forward_invoke(emb)).item()
        action_prior = F.softmax(model.forward_action(emb), dim=-1)[0]

        if model.outcome_action_conditional:
            emb_rep = emb.expand(model.n_actions, -1)
            a_oh = torch.eye(model.n_actions, device=x.device)
            action_outcomes = torch.sigmoid(model.forward_outcome(emb_rep, a_oh))
        else:
            state_outcome = torch.sigmoid(model.forward_outcome(emb))[0]
            action_outcomes = state_outcome.expand(model.n_actions)

    # Sort actions by planning score
    sorted_idx = torch.argsort(action_outcomes, descending=True)
    top_action = int(sorted_idx[0].item())
    top_conf = float(action_outcomes[top_action].item())
    second_conf = float(action_outcomes[int(sorted_idx[1].item())].item())
    margin = top_conf - second_conf

    ranking = [(int(i.item()), float(action_outcomes[i].item())) for i in sorted_idx]

    reasons = []
    if invoke_prob > invoke_threshold:
        reasons.append("structural")        # novelty / first-frame / level-change signal
    if top_conf < play_confidence_threshold:
        reasons.append("low_confidence")    # top action isn't confident enough to commit
    if margin < play_margin_threshold:
        reasons.append("tight_margin")      # top and 2nd action too close

    decision = "invoke" if reasons else "play"

    return {
        "decision": decision,
        "invoke_reasons": reasons,
        "invoke_prob": invoke_prob,
        "play_action": top_action,
        "play_confidence": top_conf,
        "play_margin": margin,
        "action_ranking": ranking,
        "action_prior": [float(p) for p in action_prior.tolist()],
        "context": {
            "embedding": [float(v) for v in emb[0].cpu().tolist()],
            "game": game, "level": level, "step_index": step_index,
        },
    }


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
