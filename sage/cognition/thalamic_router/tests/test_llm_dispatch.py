"""Tests for llm_dispatch — parser, prompt composition, mechanics-neighbor loading.

No network calls. Ollama / claude CLI / Anthropic clients not exercised here;
their behavior is end-to-end verified by live runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.cognition.thalamic_router.llm_dispatch import (
    parse_llm_response, render_frame_png, render_frame_pair_png,
    build_prompt, _load_world_model_summary, _load_mechanics_neighbors,
    _load_level_annotations, _load_level_bboxes, _nn_click_coords,
    _nn_click_rotation,
    ACTION_NAMES,
)


# ───────────────────────────────────────────────────────────────────
# Parser — literal format
# ───────────────────────────────────────────────────────────────────

def test_parser_literal_format():
    a, c, r = parse_llm_response("ACTION=6 X=48 Y=36\nclick there", fallback_action=99)
    assert a == 6
    assert c == {"x": 48, "y": 36}
    assert "click" in r


def test_parser_literal_with_brackets():
    a, c, r = parse_llm_response("ACTION=6[ X=59 Y=59 ]\nrationale", fallback_action=99)
    assert a == 6
    assert c == {"x": 59, "y": 59}


def test_parser_literal_non_click_no_coords():
    a, c, r = parse_llm_response("ACTION=1\nmove up", fallback_action=99)
    assert a == 1
    assert c is None


# ───────────────────────────────────────────────────────────────────
# Parser — natural-language fallbacks
# ───────────────────────────────────────────────────────────────────

def test_parser_natural_i_choose():
    a, c, r = parse_llm_response("I choose CLICK at (32, 12).", fallback_action=99)
    assert a == 6
    assert c == {"x": 32, "y": 12}


def test_parser_natural_action_is():
    a, c, r = parse_llm_response("My action is UP to move upward.", fallback_action=99)
    assert a == 1
    assert c is None


def test_parser_natural_going_with():
    a, c, r = parse_llm_response("Going with LEFT — the target is on the left.", fallback_action=99)
    assert a == 3


def test_parser_natural_lets_try():
    a, c, r = parse_llm_response("Let's try DOWN this time.", fallback_action=99)
    assert a == 2


def test_parser_naked_action_name_fallback():
    a, c, r = parse_llm_response("We should RIGHT and see what happens", fallback_action=99)
    assert a == 4


def test_parser_naked_click_with_paren_coords():
    a, c, r = parse_llm_response("Looking at the board, I'd CLICK the cell at (45, 20).",
                                 fallback_action=99)
    assert a == 6
    assert c == {"x": 45, "y": 20}


def test_parser_paren_coords_x_separator():
    # "(32x12)" format sometimes seen
    a, c, r = parse_llm_response("ACTION=6\nclick at (32x12)", fallback_action=99)
    assert a == 6
    assert c == {"x": 32, "y": 12}


def test_parser_click_fallback_coords_when_missing():
    a, c, r = parse_llm_response("ACTION=6\nno coords here",
                                 fallback_action=99, fallback_coords={"x": 32, "y": 32})
    assert a == 6
    assert c == {"x": 32, "y": 32}


def test_parser_select_alias():
    """SELECT should map to SEL (action 5) via the name alias."""
    a, c, r = parse_llm_response("I'll SELECT the yellow piece.", fallback_action=99)
    assert a == 5


def test_parser_pure_noise_falls_through():
    a, c, r = parse_llm_response("word soup without any hint", fallback_action=42)
    assert a == 42
    assert "parse_failed" in r


def test_parser_out_of_range_falls_through():
    a, c, r = parse_llm_response("ACTION=99", fallback_action=3)
    assert a == 3
    assert "out_of_range" in r


# ───────────────────────────────────────────────────────────────────
# Coord clamping
# ───────────────────────────────────────────────────────────────────

def test_parser_clamps_coords_to_grid():
    # Claude sometimes emits larger-than-grid coords
    a, c, r = parse_llm_response("ACTION=6 X=300 Y=-5", fallback_action=99)
    assert a == 6
    assert 0 <= c["x"] <= 63
    assert 0 <= c["y"] <= 63
    assert c["x"] == 63   # 300 clamped to max
    assert c["y"] == 0    # -5 clamped to 0


# ───────────────────────────────────────────────────────────────────
# Frame rendering
# ───────────────────────────────────────────────────────────────────

def test_render_frame_png_produces_png_bytes():
    grid = np.random.randint(0, 16, (64, 64))
    png = render_frame_png(grid)
    assert png.startswith(b"\x89PNG")


def test_render_frame_pair_png_produces_png_bytes():
    a = np.random.randint(0, 16, (64, 64))
    b = np.random.randint(0, 16, (64, 64))
    png = render_frame_pair_png(a, b)
    assert png.startswith(b"\x89PNG")


# ───────────────────────────────────────────────────────────────────
# Prompt composition
# ───────────────────────────────────────────────────────────────────

def test_build_prompt_includes_required_sections():
    p = build_prompt(
        game="ft09", level=0, step_index=1,
        play_action_idx=6, play_confidence=0.94,
        action_ranking=[(6, 0.94), (3, 0.03)],
        recent_actions=[0, 0],
        invoke_reasons=["novelty"],
    )
    # Always has current-situation block
    assert "Game: ft09" in p
    assert "Level: 0" in p
    assert "Step: 1" in p
    # Has action map
    assert "A0=0" in p and "CLICK=6" in p
    # Has NN hint block
    assert "top pick: CLICK" in p
    # Has response format instruction
    assert "ACTION=" in p


def test_build_prompt_with_trajectory():
    p = build_prompt(
        game="ft09", level=0, step_index=5,
        play_action_idx=6, play_confidence=0.9,
        action_ranking=[(6, 0.9)],
        recent_actions=[6, 6],
        invoke_reasons=["stuck"],
        recent_trajectory=[
            {"step": 3, "action": 6, "coords": {"x": 32, "y": 32},
             "level": 0, "new_level": 0, "frame_delta_pct": 0.0},
            {"step": 4, "action": 6, "coords": {"x": 40, "y": 40},
             "level": 0, "new_level": 0, "frame_delta_pct": 5.0},
        ],
    )
    assert "Recent trajectory" in p
    assert "CLICK(32,32)" in p
    assert "0% frame change" in p
    assert "CLICK(40,40)" in p
    assert "5% frame change" in p


# ───────────────────────────────────────────────────────────────────
# World-model loader
# ───────────────────────────────────────────────────────────────────

def test_world_model_loader_returns_empty_for_unknown_game():
    txt = _load_world_model_summary("unknown_game_family_xyz", max_chars=500)
    assert txt == ""


def test_world_model_loader_returns_sections_for_known_game():
    # Requires shared-context/arc-agi-3/world-models/ft09.md to exist
    txt = _load_world_model_summary("ft09", max_chars=2000)
    if not txt:
        # Skip gracefully if shared-context unavailable
        return
    # Expected sections if found
    assert "## Objects" in txt or "## Rules" in txt


# ───────────────────────────────────────────────────────────────────
# Mechanics-neighbor loader
# ───────────────────────────────────────────────────────────────────

def test_mechanics_neighbor_loader_returns_dict():
    # Always returns dict even when no adapter exists (graceful degradation)
    nn = _load_mechanics_neighbors()
    assert isinstance(nn, dict)


# ───────────────────────────────────────────────────────────────────
# Level annotations + NN click rotation
# ───────────────────────────────────────────────────────────────────

def test_level_annotations_returns_dict():
    """Always returns dict even for missing games."""
    ann = _load_level_annotations("unknown_game_xyz")
    assert isinstance(ann, dict)


def test_level_annotations_real_game_if_shared_available():
    """If shared-context is present, ft09 should have annotations."""
    ann = _load_level_annotations("ft09")
    if not ann:
        return  # shared-context unavailable, skip gracefully
    assert 0 in ann
    hint = ann[0]
    # Hint should mention solver step count + bbox
    assert "step" in hint.lower()


def test_nn_click_coords_fallback_when_no_bbox():
    """Unknown game → center-of-frame fallback."""
    _nn_click_rotation.clear()
    c = _nn_click_coords("unknown_game_xyz", 0)
    assert c == {"x": 32, "y": 32}


def test_nn_click_coords_rotates_through_bbox_grid():
    """ft09 L0 bbox=(38-54, 38-54) stride=(16,8) → grid:
    (38,38), (54,38), (38,46), (54,46), (38,54), (54,54).
    After the 6-point cycle, function returns None to signal "exhausted
    grid — force invoke." (Self-inverse games like ft09 would toggle
    earlier clicks off if we kept cycling.)
    """
    bboxes = _load_level_bboxes("ft09")
    if 0 not in bboxes:
        return  # shared-context unavailable
    _nn_click_rotation.clear()
    seen = []
    for _ in range(6):
        c = _nn_click_coords("ft09", 0)
        assert c is not None
        seen.append((c["x"], c["y"]))
    # 6 grid points should be unique
    assert len(set(seen)) == 6
    # All should fall in bbox
    b = bboxes[0]
    for x, y in seen:
        assert b["x_min"] <= x <= b["x_max"]
        assert b["y_min"] <= y <= b["y_max"]
    # 7th call → None (grid exhausted)
    assert _nn_click_coords("ft09", 0) is None


def test_nn_click_coords_rotation_per_level_independent():
    """Rotation counter is per (game, level) — different levels have
    independent counters."""
    bboxes = _load_level_bboxes("ft09")
    if 0 not in bboxes or 1 not in bboxes:
        return
    _nn_click_rotation.clear()
    # Advance L0 several times
    for _ in range(3):
        _nn_click_coords("ft09", 0)
    # L1's first call should start at L1's own index 0
    l1_first = _nn_click_coords("ft09", 1)
    # Construct expected: first grid point = (x_min, y_min) of L1 bbox
    b1 = bboxes[1]
    assert l1_first == {"x": b1["x_min"], "y": b1["y_min"]}


def test_level_annotation_hint_does_not_leak_winning_coords():
    """Critical: the hint field must NOT contain specific winning click
    coords. It should carry bbox + stride + step count, not the sequence."""
    ann = _load_level_annotations("ft09")
    if not ann:
        return
    # Known ft09 L0 winning coords: (38,54), (38,46), (38,38), (54,46)
    # The hint mentions the BBOX but shouldn't enumerate these specific pairs.
    hint = ann[0]
    # Hint can mention range 38-54 (that's the bbox, not a winning coord)
    # but shouldn't list the full sequence
    # Simple check: does it have a list of 3+ coord-like tuples?
    import re
    coord_tuples = re.findall(r"\(\s*\d+\s*,\s*\d+\s*\)", hint)
    # A hint should have 0 or maybe 1 coord-like mention, not 3+
    assert len(coord_tuples) < 3, f"Hint appears to leak winning coords: {hint}"


# ───────────────────────────────────────────────────────────────────
# Mechanics-encoder input swap (v6 plumbing)
# ───────────────────────────────────────────────────────────────────

def test_frame_router_v5_backcompat_loads_with_use_mech_false():
    """Existing v5 adapter must load with default use_mech_embedding=False."""
    from pathlib import Path
    from sage.cognition.thalamic_router.frame_router import load_frame_router
    v5_path = Path("/mnt/c/exe/projects/ai-agents/private-context/training-data/"
                    "router/_adapters/cbp-framerouter-v5nodyn-2026-04-18")
    if not (v5_path.parent.exists() and (v5_path.with_suffix(".pt")).exists()):
        return  # adapter not on this machine
    model, cfg = load_frame_router(v5_path)
    assert cfg.use_mech_embedding is False
    # scalar_dim = n_games(6) + n_levels(10) + 2 + last_action(7)
    #            + 8*7 + 7 + 3 = 91
    assert model.scalar_dim == 91


def test_load_mech_embedding_table_aligns_to_router_slugs():
    """Mechanics embeddings should align to the router's game_slugs order."""
    from pathlib import Path
    from sage.cognition.thalamic_router.frame_router import (
        load_frame_router, load_mech_embedding_table,
    )
    v5 = Path("/mnt/c/exe/projects/ai-agents/private-context/training-data/"
              "router/_adapters/cbp-framerouter-v5nodyn-2026-04-18")
    mech = Path("/mnt/c/exe/projects/ai-agents/private-context/training-data/"
                "router/_adapters/cbp-mechanics-2026-04-18.json")
    if not (v5.with_suffix(".pt").exists() and mech.exists()):
        return
    _, cfg = load_frame_router(v5)
    table = load_mech_embedding_table(mech, cfg.game_slugs, mech_emb_dim=32)
    assert table.shape == (cfg.n_games, 32)
    # All 6 CBP games should be in the 25-game mechanics encoder
    nonzero = int((table.sum(1) != 0).sum())
    assert nonzero == cfg.n_games


def test_build_scalar_vector_with_mech_embedding():
    """mech_embedding parameter replaces n_games one-hot with 32d vector."""
    from sage.cognition.thalamic_router.frame_router import build_scalar_vector
    n_games = 25
    vec_onehot = build_scalar_vector(
        game_idx=3, n_games=n_games, level=0, n_levels=10,
        step_frac=0.1, budget_remaining=0.5, last_action=0,
        recent_actions=[0]*8, available_actions=[1]*7, batch_state=[0, 0, 0],
    )
    vec_mech = build_scalar_vector(
        game_idx=3, n_games=n_games, level=0, n_levels=10,
        step_frac=0.1, budget_remaining=0.5, last_action=0,
        recent_actions=[0]*8, available_actions=[1]*7, batch_state=[0, 0, 0],
        mech_embedding=[0.1] * 32,
    )
    # n_games(25) → mech_emb_dim(32) replaces the game_context slot
    assert len(vec_onehot) - n_games == len(vec_mech) - 32


if __name__ == "__main__":
    failures = 0
    for name in list(globals()):
        if name.startswith("test_"):
            try:
                globals()[name]()
                print(f"  ✓ {name}")
            except AssertionError as e:
                print(f"  ✗ {name}: {e}")
                failures += 1
            except Exception as e:
                print(f"  ✗ {name}: {type(e).__name__}: {e}")
                failures += 1
    if failures:
        print(f"\n{failures} failures")
        sys.exit(1)
    print("\nAll tests passed")
