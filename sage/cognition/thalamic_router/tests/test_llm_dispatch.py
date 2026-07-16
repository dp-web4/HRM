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
    _nn_click_rotation, _detect_trajectory_patterns,
    _load_gameplay_system_prompt, SYSTEM_PROMPT,
    build_session_system_prompt, compose_cnn_narration,
    summarize_game_progress, trim_conversation_history,
    MAX_CONVERSATION_TURNS, KEEP_VERBATIM_TURNS,
    LLMClient, OllamaClient, ClaudeCLIClient,
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

# ───────────────────────────────────────────────────────────────────
# World-model loader
# ───────────────────────────────────────────────────────────────────

def test_world_model_loader_returns_empty_for_unknown_game():
    txt = _load_world_model_summary("unknown_game_family_xyz", max_chars=500)
    assert txt == ""


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


def test_nn_click_coords_fallback_when_no_bbox():
    """Unknown game → center-of-frame fallback."""
    _nn_click_rotation.clear()
    c = _nn_click_coords("unknown_game_xyz", 0)
    assert c == {"x": 32, "y": 32}


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


# ───────────────────────────────────────────────────────────────────
# Gameplay system prompt loader + trajectory pattern detection
# ───────────────────────────────────────────────────────────────────

def test_system_prompt_loads_nonempty():
    """Either from the canonical MD file or the fallback — must be non-empty."""
    p = _load_gameplay_system_prompt()
    assert p and len(p) > 50


def test_system_prompt_from_file_if_available():
    """If shared-context is on disk, the loaded prompt should be the long one."""
    from sage.cognition.thalamic_router.llm_dispatch import _resolve_gameplay_system_prompt_path
    path = _resolve_gameplay_system_prompt_path()
    if path is None:
        return  # graceful skip if shared-context unavailable
    p = _load_gameplay_system_prompt()
    assert len(p) > 1000    # the canonical MD is ~10KB; the fallback is ~400 chars
    assert "R6" in p or "Perceive" in p or "deliberation tier" in p


def test_trajectory_stuck_flag():
    """3+ consecutive low-delta steps at the tail → STUCK flag."""
    w = [
        {"step": 1, "action": 4, "frame_delta_pct": 12.0, "level": 0, "new_level": 0},
        {"step": 2, "action": 6, "coords": {"x": 32, "y": 32}, "frame_delta_pct": 0.0, "level": 0, "new_level": 0},
        {"step": 3, "action": 6, "coords": {"x": 32, "y": 32}, "frame_delta_pct": 0.0, "level": 0, "new_level": 0},
        {"step": 4, "action": 6, "coords": {"x": 32, "y": 32}, "frame_delta_pct": 0.0, "level": 0, "new_level": 0},
    ]
    flags = _detect_trajectory_patterns(w)
    assert any("STUCK" in f for f in flags)
    assert any("PERSEVERATION" in f for f in flags)


def test_trajectory_level_advance_flag():
    w = [{"step": 5, "action": 1, "frame_delta_pct": 8.0, "level": 0, "new_level": 1}]
    flags = _detect_trajectory_patterns(w)
    assert any("Level advanced" in f for f in flags)


def test_trajectory_oscillation_flag():
    """UP-DOWN-UP-DOWN alternation → oscillation flag."""
    w = [
        {"step": 1, "action": 1, "frame_delta_pct": 5.0, "level": 0, "new_level": 0},
        {"step": 2, "action": 2, "frame_delta_pct": 5.0, "level": 0, "new_level": 0},
        {"step": 3, "action": 1, "frame_delta_pct": 5.0, "level": 0, "new_level": 0},
        {"step": 4, "action": 2, "frame_delta_pct": 5.0, "level": 0, "new_level": 0},
    ]
    flags = _detect_trajectory_patterns(w)
    assert any("OSCILLATION" in f for f in flags)


def test_trajectory_empty_returns_empty():
    assert _detect_trajectory_patterns([]) == []


# ───────────────────────────────────────────────────────────────────
# Three-party conversation composers (Phase 4 B4)
# ───────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────
# LLMClient multi-turn signature
# ───────────────────────────────────────────────────────────────────

def test_llm_client_chat_signature_has_history_and_system_prompt():
    """Multi-turn params are optional for backward compat."""
    import inspect
    sig = inspect.signature(LLMClient.chat)
    params = list(sig.parameters.keys())
    assert "history" in params
    assert "system_prompt" in params
    # Defaults make them optional
    assert sig.parameters["history"].default is None
    assert sig.parameters["system_prompt"].default is None


def test_ollama_client_builds_messages_with_history():
    """OllamaClient puts history turns between system and new user."""
    # No network call — just verify the payload shape builder logic
    # by patching urlopen at the module level.
    import urllib.request
    from unittest.mock import patch
    client = OllamaClient(model="test-model")
    captured = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self):
            return b'{"message": {"content": "ACTION=1"}}'

    def fake_urlopen(req, timeout=None):
        captured["data"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        client.chat(
            "new turn",
            history=[
                {"role": "user", "content": "old user"},
                {"role": "assistant", "content": "old response"},
            ],
            system_prompt="fleet canonical",
        )

    messages = captured["data"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "fleet canonical"
    assert messages[1]["role"] == "user" and messages[1]["content"] == "old user"
    assert messages[2]["role"] == "assistant" and messages[2]["content"] == "old response"
    assert messages[3]["role"] == "user" and messages[3]["content"] == "new turn"


def test_claude_cli_client_serializes_history_as_prose():
    """ClaudeCLIClient builds tagged prose when history is provided."""
    from unittest.mock import patch, MagicMock
    # Instantiation needs claude in PATH; mock shutil.which
    with patch("shutil.which", return_value="/fake/claude"):
        client = ClaudeCLIClient()
    captured = {}

    class FakeSubproc:
        def run(self, args, input=None, capture_output=None, timeout=None):
            captured["input"] = input.decode("utf-8")
            m = MagicMock(); m.returncode = 0
            m.stdout = b"ACTION=1\n"
            m.stderr = b""
            return m
        TimeoutExpired = Exception

    client._subprocess = FakeSubproc()

    client.chat(
        "new turn",
        history=[
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old response"},
        ],
        system_prompt="fleet canonical",
    )
    # Expect [System], [User], [Assistant], final [User]:, trailing [Assistant]:
    prompt = captured["input"]
    assert "[System]" in prompt
    assert "fleet canonical" in prompt
    assert "[User]:" in prompt
    assert "old user" in prompt
    assert "[Assistant]:" in prompt
    assert "old response" in prompt
    assert "new turn" in prompt
    # Final "[Assistant]:" with no content to trigger completion
    assert prompt.rstrip().endswith("[Assistant]:")


# ───────────────────────────────────────────────────────────────────
# Memory trim + game-progress summary (Phase 4 B5)
# ───────────────────────────────────────────────────────────────────

def test_summarize_game_progress_empty_trajectory():
    s = summarize_game_progress([])
    assert "none" in s.lower() or "no earlier" in s.lower()


def test_summarize_game_progress_captures_level_changes():
    traj = [
        {"step": 1, "action": 4, "coords": None, "frame_delta_pct": 8.0,
         "level": 0, "new_level": 0},
        {"step": 5, "action": 4, "coords": None, "frame_delta_pct": 12.0,
         "level": 0, "new_level": 1},  # level advance
        {"step": 12, "action": 1, "coords": None, "frame_delta_pct": 3.0,
         "level": 1, "new_level": 1},
    ]
    s = summarize_game_progress(traj)
    assert "L0→L1" in s
    assert "step 5" in s


def test_summarize_game_progress_buckets_repeated_actions():
    traj = [
        {"step": i, "action": 6, "coords": {"x": 32, "y": 32},
         "frame_delta_pct": 0.0, "level": 0, "new_level": 0}
        for i in range(1, 6)
    ]
    s = summarize_game_progress(traj)
    assert "CLICK(32,32)" in s
    assert "×5" in s
    assert "no effect" in s


def test_trim_conversation_history_under_threshold_no_trim():
    history = [{"role": "user", "content": f"Step {i}"} for i in range(5)]
    trimmed, preamble = trim_conversation_history(history, trajectory=[])
    assert trimmed == history
    assert preamble is None


def test_trim_conversation_history_over_threshold_trims_and_summarizes():
    # Build 24 turns (12 user, 12 assistant) — over the 20 threshold
    history = []
    for i in range(12):
        history.append({"role": "user", "content": f"Step {i+1}. invoke."})
        history.append({"role": "assistant", "content": f"ACTION=1. rationale {i}"})
    trajectory = [
        {"step": i+1, "action": 1, "coords": None, "frame_delta_pct": 5.0,
         "level": 0, "new_level": 0}
        for i in range(12)
    ]
    trimmed, preamble = trim_conversation_history(history, trajectory)
    assert len(trimmed) <= KEEP_VERBATIM_TURNS
    assert preamble is not None
    assert "Prior context" in preamble
    # The preamble should not contain LLM rationales verbatim
    assert "rationale" not in preamble


def test_trim_conversation_history_preserves_recent_verbatim():
    history = []
    for i in range(12):
        history.append({"role": "user", "content": f"Step {i+1}. invoke."})
        history.append({"role": "assistant", "content": f"ACTION=1."})
    trajectory = [
        {"step": i+1, "action": 1, "coords": None, "frame_delta_pct": 5.0,
         "level": 0, "new_level": 0}
        for i in range(12)
    ]
    trimmed, _ = trim_conversation_history(history, trajectory)
    # Kept turns should be the latest ones
    last_user = [t for t in trimmed if t["role"] == "user"][-1]
    assert "Step 12" in last_user["content"]


# ───────────────────────────────────────────────────────────────────
# Grounded reasoning metrics (Phase 4 B6)
# ───────────────────────────────────────────────────────────────────

def test_entity_validity_gibberish_entities_score_low():
    from sage.cognition.thalamic_router.grounded_metrics import entity_validity_rate
    r = entity_validity_rate(
        "The xqzt is positioned near the blargh; fiddlewick must align.",
        world_model_text="",
    )
    # None of xqzt/blargh/fiddlewick are valid
    assert r < 0.3


def test_entity_validity_world_model_vocabulary_counted():
    from sage.cognition.thalamic_router.grounded_metrics import entity_validity_rate
    wm = "The basket collects eggs and launches them toward the canvas."
    r = entity_validity_rate(
        "The basket is positioned to launch toward the canvas target.",
        world_model_text=wm,
    )
    # basket, launch, canvas all appear in WM — high validity
    assert r > 0.5


def test_vocabulary_correctness_overlap_with_world_model():
    from sage.cognition.thalamic_router.grounded_metrics import vocabulary_correctness
    wm = "The basket collects eggs and launches paint toward the canvas."
    r = vocabulary_correctness(
        "Clicking the basket to collect eggs.",
        world_model_text=wm,
    )
    # "basket", "collect", "eggs" overlap with WM
    assert r > 0.3


def test_vocabulary_correctness_no_domain_terms_neutral():
    from sage.cognition.thalamic_router.grounded_metrics import vocabulary_correctness
    r = vocabulary_correctness(
        "Try a different approach because the current one isn't working.",
        world_model_text="basket collects eggs",
    )
    # Generic rationale, no domain terms — neutral 0.5
    assert 0.3 <= r <= 0.7


def test_mechanics_alignment_prediction_matches_observation():
    from sage.cognition.thalamic_router.grounded_metrics import mechanics_alignment
    # Predicted advance, observed level change
    r = mechanics_alignment(
        "This action will advance to the next level.",
        observed_frame_delta_pct=30.0,
        level_advanced=True,
    )
    assert r == 1.0


def test_mechanics_alignment_prediction_opposite_observation():
    from sage.cognition.thalamic_router.grounded_metrics import mechanics_alignment
    # Predicted change, observed nothing
    r = mechanics_alignment(
        "This will launch the basket and paint the canvas.",
        observed_frame_delta_pct=0.0,
        level_advanced=False,
    )
    assert r == 0.0


def test_mechanics_alignment_no_prediction_returns_none():
    from sage.cognition.thalamic_router.grounded_metrics import mechanics_alignment
    r = mechanics_alignment(
        "Picking RIGHT.",
        observed_frame_delta_pct=5.0,
    )
    assert r is None


def test_mechanics_alignment_testing_language_scores_low_delta():
    from sage.cognition.thalamic_router.grounded_metrics import mechanics_alignment
    # "Testing" / "exploring" = low-change prediction. Low delta = match.
    r = mechanics_alignment(
        "Testing whether this coordinate responds at all.",
        observed_frame_delta_pct=0.5,
    )
    assert r == 1.0


def test_aggregate_grounded_metrics_computes_means():
    from sage.cognition.thalamic_router.grounded_metrics import aggregate_grounded_metrics
    per = [
        {"entity_validity_rate": 0.8, "vocabulary_correctness": 0.7,
         "mechanics_alignment": 1.0, "grounded_pass": True},
        {"entity_validity_rate": 0.4, "vocabulary_correctness": 0.3,
         "mechanics_alignment": 0.0, "grounded_pass": False},
        {"entity_validity_rate": 0.6, "vocabulary_correctness": 0.5,
         "mechanics_alignment": None, "grounded_pass": False},
    ]
    agg = aggregate_grounded_metrics(per)
    assert abs(agg["mean_entity_validity_rate"] - 0.6) < 1e-6
    assert abs(agg["mean_vocabulary_correctness"] - 0.5) < 1e-6
    # mean alignment excludes None
    assert abs(agg["mean_mechanics_alignment"] - 0.5) < 1e-6
    assert agg["grounded_pass_count"] == 1
    assert agg["n_invokes_measured"] == 3


def test_aggregate_grounded_metrics_empty():
    from sage.cognition.thalamic_router.grounded_metrics import aggregate_grounded_metrics
    agg = aggregate_grounded_metrics([])
    assert agg["n_invokes_measured"] == 0
    assert agg["mean_entity_validity_rate"] == 0.0


# ───────────────────────────────────────────────────────────────────
# MRH round-trip (Phase 5 C)
# ───────────────────────────────────────────────────────────────────

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
