"""Tests for world_model.py — encoder, three heads, invoke head, dispatch.

Covers the v2 architecture (action-conditional outcome + invoke head for
dispatch decisions). v0 backward-compat path (outcome on state alone)
is exercised via an explicit construction, since existing saved v0
models must still load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.cognition.thalamic_router.world_model import (
    ACTION_NAMES,
    BASE_FEATURE_NAMES,
    DEFAULT_EMB_DIM,
    DEFAULT_HIDDEN,
    N_ACTIONS,
    N_BASE_FEATURES,
    WorldModel,
    WorldModelConfig,
    action_onehot,
    build_input_vector,
    choose_dispatch,
    load_world_model,
    save_world_model,
)


# ───────────────────────────────────────────────────────────────────
# Pure-Python helpers
# ───────────────────────────────────────────────────────────────────


def test_build_input_vector_length_matches_model_input_dim():
    n_games = 4
    n_levels = 10
    base = [0.0] * N_BASE_FEATURES
    vec = build_input_vector(
        base_features=base, game_idx=0, n_games=n_games,
        level=0, n_levels=n_levels, step_frac=0.0,
        last_action=0, n_actions=N_ACTIONS,
    )
    expected_len = N_BASE_FEATURES + n_games + n_levels + 1 + N_ACTIONS
    assert len(vec) == expected_len

    # And: a freshly-constructed WorldModel with the same sizing accepts
    # exactly this width as its encoder input.
    model = WorldModel(n_games=n_games, n_levels=n_levels)
    assert model.input_dim == expected_len


def test_build_input_vector_one_hot_segments_are_correct():
    n_games, n_levels = 3, 5
    base = [0.5] * N_BASE_FEATURES
    vec = build_input_vector(
        base_features=base, game_idx=2, n_games=n_games,
        level=1, n_levels=n_levels, step_frac=0.7,
        last_action=3, n_actions=N_ACTIONS,
    )
    # Segment offsets
    game_start = N_BASE_FEATURES
    level_start = game_start + n_games
    step_frac_idx = level_start + n_levels
    action_start = step_frac_idx + 1

    # Base features preserved
    assert vec[:N_BASE_FEATURES] == base
    # Game one-hot: index 2 hot
    assert vec[game_start:game_start + n_games] == [0.0, 0.0, 1.0]
    # Level one-hot: index 1 hot
    assert vec[level_start:level_start + n_levels] == [0.0, 1.0, 0.0, 0.0, 0.0]
    # step_frac passed through
    assert vec[step_frac_idx] == 0.7
    # Action one-hot: index 3 hot
    expected_action = [0.0] * N_ACTIONS
    expected_action[3] = 1.0
    assert vec[action_start:action_start + N_ACTIONS] == expected_action


def test_build_input_vector_level_clamped_to_valid_range():
    # Level above n_levels-1 clamps to n_levels-1; negative clamps to 0;
    # None clamps to 0.
    base = [0.0] * N_BASE_FEATURES
    high = build_input_vector(
        base_features=base, game_idx=0, n_games=1,
        level=999, n_levels=10, step_frac=0.0, last_action=0,
    )
    # Last level bucket (index 9) should be hot
    level_seg = high[N_BASE_FEATURES + 1:N_BASE_FEATURES + 1 + 10]
    assert level_seg[9] == 1.0 and sum(level_seg) == 1.0

    low = build_input_vector(
        base_features=base, game_idx=0, n_games=1,
        level=-5, n_levels=10, step_frac=0.0, last_action=0,
    )
    level_seg = low[N_BASE_FEATURES + 1:N_BASE_FEATURES + 1 + 10]
    assert level_seg[0] == 1.0 and sum(level_seg) == 1.0

    nn = build_input_vector(
        base_features=base, game_idx=0, n_games=1,
        level=None, n_levels=10, step_frac=0.0, last_action=0,
    )
    level_seg = nn[N_BASE_FEATURES + 1:N_BASE_FEATURES + 1 + 10]
    assert level_seg[0] == 1.0 and sum(level_seg) == 1.0


def test_action_onehot_is_one_hot():
    for a in range(N_ACTIONS):
        oh = action_onehot(a)
        assert len(oh) == N_ACTIONS
        assert oh[a] == 1.0
        assert sum(oh) == 1.0


# ───────────────────────────────────────────────────────────────────
# WorldModel forward shapes
# ───────────────────────────────────────────────────────────────────


def _batch_input(model: WorldModel, batch: int = 4) -> torch.Tensor:
    return torch.randn(batch, model.input_dim)


def test_world_model_v1_forward_shapes():
    m = WorldModel(n_games=3, n_levels=10, outcome_action_conditional=True)
    x = _batch_input(m, batch=4)
    a_oh = torch.eye(m.n_actions)[torch.randint(0, m.n_actions, (4,))]

    out = m(x, action_onehot=a_oh)

    assert out["embedding"].shape == (4, m.emb_dim)
    assert out["action_logits"].shape == (4, m.n_actions)
    assert out["outcome_logit"].shape == (4,)
    assert out["next_emb_pred"].shape == (4, m.emb_dim)


def test_world_model_v0_backward_compat_outcome_without_action():
    # v0 mode: outcome depends only on embedding. No action_onehot needed.
    m = WorldModel(n_games=2, n_levels=3, outcome_action_conditional=False)
    x = _batch_input(m, batch=2)

    out = m(x)   # no action_onehot

    assert "outcome_logit" in out
    assert out["outcome_logit"].shape == (2,)
    # Dynamics requires action, so it shouldn't be emitted without one.
    assert "next_emb_pred" not in out


def test_world_model_v1_outcome_requires_action():
    m = WorldModel(n_games=2, n_levels=3, outcome_action_conditional=True)
    emb = torch.randn(2, m.emb_dim)
    with pytest.raises(ValueError):
        m.forward_outcome(emb, action_onehot=None)


def test_world_model_invoke_head_returns_logit_per_row():
    m = WorldModel(n_games=2, n_levels=3)
    emb = torch.randn(5, m.emb_dim)
    logits = m.forward_invoke(emb)
    assert logits.shape == (5,)


# ───────────────────────────────────────────────────────────────────
# choose_dispatch — the three-gate invoke decision
# ───────────────────────────────────────────────────────────────────


def _stubbed_model(
    *,
    invoke_logit: float,
    outcome_scores: list[float],
    action_prior_logits: list[float] | None = None,
    outcome_action_conditional: bool = True,
) -> WorldModel:
    """Build a WorldModel whose three heads return deterministic tensors
    regardless of input, so we can isolate choose_dispatch's gate logic."""
    m = WorldModel(n_games=2, n_levels=3,
                   outcome_action_conditional=outcome_action_conditional)
    m.eval()
    if action_prior_logits is None:
        action_prior_logits = [0.0] * N_ACTIONS
    prior = torch.tensor(action_prior_logits, dtype=torch.float32)
    scores = torch.tensor(outcome_scores, dtype=torch.float32)

    # Use closures over tensors to override head outputs.
    def fake_encode(x):  # x has leading batch
        return torch.zeros(x.shape[0], m.emb_dim)

    def fake_forward_action(emb):
        return prior.unsqueeze(0).expand(emb.shape[0], -1)

    def fake_forward_invoke(emb):
        return torch.full((emb.shape[0],), invoke_logit, dtype=torch.float32)

    def fake_forward_outcome(emb, a_oh=None):
        # In v1, a_oh is a one-hot tensor (n_actions, n_actions) or similar.
        # We return the sigmoid-inverse of each action's desired score,
        # indexed by argmax(a_oh).
        if a_oh is None:
            # v0: return a single logit for the state's outcome
            return torch.logit(torch.full((emb.shape[0],), float(scores[0])))
        idx = a_oh.argmax(dim=-1)
        out = torch.logit(scores.clamp(1e-6, 1 - 1e-6))[idx]
        return out

    m.encode = fake_encode
    m.forward_action = fake_forward_action
    m.forward_invoke = fake_forward_invoke
    m.forward_outcome = fake_forward_outcome
    return m


def test_choose_dispatch_play_when_confident_and_margin_wide():
    # All outcomes: action 0 highly confident, others low. invoke_head cold.
    scores = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    m = _stubbed_model(invoke_logit=-5.0, outcome_scores=scores)
    x = torch.zeros(m.input_dim)

    d = choose_dispatch(m, x)

    assert d["decision"] == "play"
    assert d["play_action"] == 0
    assert d["invoke_reasons"] == []
    assert d["play_confidence"] == pytest.approx(0.9, abs=1e-4)
    assert d["play_margin"] == pytest.approx(0.8, abs=1e-4)


def test_choose_dispatch_invokes_on_structural_signal():
    # High-confidence play available, but invoke_head fires.
    scores = [0.95, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    m = _stubbed_model(invoke_logit=5.0, outcome_scores=scores)
    x = torch.zeros(m.input_dim)

    d = choose_dispatch(m, x)

    assert d["decision"] == "invoke"
    assert "structural" in d["invoke_reasons"]
    # Structural invoke still populates play_action for fall-through.
    assert d["play_action"] == 0


def test_choose_dispatch_invokes_on_low_confidence():
    # Top action is below play_confidence_threshold (0.65).
    scores = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    m = _stubbed_model(invoke_logit=-5.0, outcome_scores=scores)
    x = torch.zeros(m.input_dim)

    d = choose_dispatch(m, x)

    assert d["decision"] == "invoke"
    assert "low_confidence" in d["invoke_reasons"]
    assert "structural" not in d["invoke_reasons"]


def test_choose_dispatch_invokes_on_tight_margin():
    # Top action confident enough, but 2nd is right behind it.
    scores = [0.80, 0.77, 0.05, 0.05, 0.05, 0.05, 0.05]
    m = _stubbed_model(invoke_logit=-5.0, outcome_scores=scores)
    x = torch.zeros(m.input_dim)

    d = choose_dispatch(m, x)

    assert d["decision"] == "invoke"
    assert "tight_margin" in d["invoke_reasons"]
    assert "low_confidence" not in d["invoke_reasons"]


def test_choose_dispatch_stacks_multiple_reasons():
    # Structural AND low confidence AND tight margin all fire.
    scores = [0.5, 0.48, 0.05, 0.05, 0.05, 0.05, 0.05]
    m = _stubbed_model(invoke_logit=5.0, outcome_scores=scores)
    x = torch.zeros(m.input_dim)

    d = choose_dispatch(m, x)

    assert d["decision"] == "invoke"
    assert set(d["invoke_reasons"]) == {"structural", "low_confidence", "tight_margin"}


def test_choose_dispatch_action_ranking_is_sorted_descending():
    scores = [0.1, 0.9, 0.3, 0.5, 0.2, 0.8, 0.4]
    m = _stubbed_model(invoke_logit=-5.0, outcome_scores=scores)
    d = choose_dispatch(m, torch.zeros(m.input_dim))

    ranked = d["action_ranking"]
    assert [i for i, _ in ranked] == [1, 5, 3, 6, 2, 4, 0]
    # Monotonically non-increasing confidence
    confs = [c for _, c in ranked]
    assert all(confs[i] >= confs[i + 1] for i in range(len(confs) - 1))


def test_choose_dispatch_context_contains_embedding_and_metadata():
    scores = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    m = _stubbed_model(invoke_logit=-5.0, outcome_scores=scores)
    d = choose_dispatch(
        m, torch.zeros(m.input_dim),
        game="cd82", level=3, step_index=42,
    )
    ctx = d["context"]
    assert ctx["game"] == "cd82"
    assert ctx["level"] == 3
    assert ctx["step_index"] == 42
    assert isinstance(ctx["embedding"], list)
    assert len(ctx["embedding"]) == m.emb_dim


# ───────────────────────────────────────────────────────────────────
# Save / load roundtrip
# ───────────────────────────────────────────────────────────────────


def test_save_load_roundtrip_preserves_weights_and_architecture(tmp_path):
    m = WorldModel(
        n_games=3, n_levels=7,
        emb_dim=16, hidden=24,
        outcome_action_conditional=True,
    )
    cfg = WorldModelConfig(
        n_games=3, game_slugs=["a", "b", "c"],
        n_levels=7, emb_dim=16, hidden=24,
        outcome_action_conditional=True,
    )
    path = tmp_path / "wm"
    save_world_model(m, cfg, path)

    loaded, loaded_cfg = load_world_model(path)

    # Config roundtrip
    assert loaded_cfg.n_games == 3
    assert loaded_cfg.game_slugs == ["a", "b", "c"]
    assert loaded_cfg.outcome_action_conditional is True
    assert loaded_cfg.emb_dim == 16

    # Weights roundtrip: identical output for same input
    x = torch.randn(2, m.input_dim)
    a_oh = torch.eye(m.n_actions)[[0, 1]]
    m.eval(); loaded.eval()
    with torch.no_grad():
        orig = m(x, action_onehot=a_oh)
        new = loaded(x, action_onehot=a_oh)
    for k in ("embedding", "action_logits", "outcome_logit", "next_emb_pred"):
        assert torch.allclose(orig[k], new[k], atol=1e-6), f"{k} diverged"


def test_load_world_model_forward_compat_for_v0_configs(tmp_path):
    # v0 configs predate `architecture_version` and `outcome_action_conditional`.
    # load_world_model must default them to the v0 values (0 / False).
    import json
    path = tmp_path / "wm_v0"
    m = WorldModel(
        n_games=2, n_levels=3,
        emb_dim=16, hidden=24,
        outcome_action_conditional=False,
    )
    torch.save(m.state_dict(), str(path) + ".pt")
    # Write a minimal v0-shaped config without the new fields.
    v0_cfg = {
        "n_games": 2, "game_slugs": ["a", "b"],
        "n_levels": 3, "n_base_features": N_BASE_FEATURES,
        "emb_dim": 16, "hidden": 24, "n_actions": N_ACTIONS,
        "feature_mean": [], "feature_std": [],
        "trained_at": None, "train_commit": None, "machine": None,
        "train_record_count": None, "val_metrics": {},
    }
    with open(str(path) + ".json", "w", encoding="utf-8") as f:
        json.dump(v0_cfg, f)

    loaded, cfg = load_world_model(path)
    assert cfg.architecture_version == 0
    assert cfg.outcome_action_conditional is False
    # v0 forward path works without action_onehot
    with torch.no_grad():
        out = loaded(torch.randn(1, loaded.input_dim))
    assert "outcome_logit" in out
