"""Tests for the lean prompt builder."""

from sage.cognition.thalamic_router.lean_prompt import (
    GAME_SUMMARIES,
    build_lean_prompt,
    build_lean_retry_prompt,
)


class TestBuildLeanPrompt:
    def test_minimal_prompt_under_100_tokens(self):
        """Bare minimum prompt should be very short."""
        p = build_lean_prompt(game="cd82", level=0, step=1)
        tokens_est = len(p) // 4
        assert tokens_est < 100, f"minimal prompt ~{tokens_est} tokens, want <100"
        assert "ACTION=" in p

    def test_full_prompt_under_300_tokens(self):
        """Full prompt with all fields should stay under 300 tokens."""
        p = build_lean_prompt(
            game="cd82", level=0, step=42,
            world_model_summary=GAME_SUMMARIES.get("cd82", ""),
            predictions="  UP: blocked\n  DOWN: blocked\n  LEFT: 201px top-right (+13 red)\n  RIGHT: 201px bottom-left (+8 blue)\n  SEL: 51px center (paint event)",
            frame_state_summary="60% black, 15% red top-right, target needs more red",
            nn_hint="NN suggests LEFT (conf 0.72)",
            stuck_info="3 steps no progress",
            goal_hint="match target color pattern",
        )
        tokens_est = len(p) // 4
        assert tokens_est < 300, f"full prompt ~{tokens_est} tokens, want <300"

    def test_predictions_are_central(self):
        """Predictions should come before hints and format instructions."""
        p = build_lean_prompt(
            game="tr87", level=0, step=10,
            predictions="  UP: 13px bottom-left\n  DOWN: 15px bottom-left",
            nn_hint="NN suggests DOWN",
        )
        pred_pos = p.find("I tested what each action does")
        hint_pos = p.find("Hint:")
        format_pos = p.find("Reply with ONLY")
        assert pred_pos < hint_pos < format_pos, "predictions should come before hint and format"

    def test_format_instruction_always_present(self):
        p = build_lean_prompt(game="ft09", level=0, step=1)
        assert "ACTION=" in p
        assert "CLICK" in p

    def test_stuck_warning_included(self):
        p = build_lean_prompt(
            game="sb26", level=0, step=50,
            stuck_info="stuck 5 steps, no frame change",
        )
        assert "WARNING" in p
        assert "stuck" in p

    def test_no_predictions_shows_action_list(self):
        p = build_lean_prompt(game="cd82", level=0, step=1)
        assert "Available actions:" in p

    def test_with_predictions_no_action_list(self):
        p = build_lean_prompt(
            game="cd82", level=0, step=1,
            predictions="  UP: blocked\n  LEFT: 201px",
        )
        assert "Available actions:" not in p
        assert "Use the predictions" in p


class TestRetryPrompt:
    def test_retry_is_ultra_short(self):
        p = build_lean_retry_prompt()
        tokens_est = len(p) // 4
        assert tokens_est < 30, f"retry ~{tokens_est} tokens, want <30"
        assert "ACTION=" in p

    def test_retry_lists_options(self):
        p = build_lean_retry_prompt()
        assert "UP" in p
        assert "CLICK" in p


class TestGameSummaries:
    def test_all_25_games_covered(self):
        expected = {"cd82", "sb26", "ft09", "tr87", "sc25", "tu93", "r11l",
                    "cn04", "sp80", "vc33", "ar25", "lp85", "ka59", "wa30",
                    "su15", "sk48", "g50t", "tn36", "s5i5", "m0r0", "bp35",
                    "dc22", "lf52", "re86", "ls20"}
        assert set(GAME_SUMMARIES.keys()) == expected

    def test_summaries_under_100_tokens_each(self):
        for game, summary in GAME_SUMMARIES.items():
            tokens_est = len(summary) // 4
            assert tokens_est < 100, f"{game} summary ~{tokens_est} tokens, want <100"
