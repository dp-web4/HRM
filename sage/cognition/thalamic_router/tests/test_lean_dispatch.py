"""Tests for lean prompt shape and budget.

Tests the build_lean_prompt function in isolation via source extraction,
avoiding the heavy torch/numpy import chain from llm_dispatch.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path


def _load_build_lean_prompt():
    """Extract build_lean_prompt from lean_dispatch.py without triggering
    module-level imports (torch, numpy, llm_dispatch)."""
    src = (Path(__file__).parent.parent / "lean_dispatch.py").read_text()
    tree = ast.parse(src)

    # Find the function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_lean_prompt":
            func_src = ast.get_source_segment(src, node)
            break
    else:
        raise RuntimeError("build_lean_prompt not found in lean_dispatch.py")

    # Compile and exec in a clean namespace
    ns = {"List": list}
    exec(f"from typing import List\n{func_src}", ns)
    return ns["build_lean_prompt"]


build_lean_prompt = _load_build_lean_prompt()


class TestBuildLeanPrompt:
    def _make_prompt(self, **overrides):
        defaults = dict(
            game="cd82", level=0, step=42,
            lookahead_text="  UP: blocked\n  DOWN: blocked\n  LEFT: 201px top-right\n  RIGHT: 201px bottom-left\n  SEL: 51px center",
            click_targets="",
            wm_summary="Paint regions to match target pattern. Basket navigates octagonal ring. SEL launches paint.",
            nn_hint="LEFT", nn_confidence=0.72,
            recent_actions=["RIGHT", "LEFT", "RIGHT", "LEFT", "SEL"],
            special_signals=[],
        )
        defaults.update(overrides)
        return build_lean_prompt(**defaults)

    def test_full_prompt_under_400_tokens(self):
        p = self._make_prompt()
        tokens_est = len(p) // 4
        assert tokens_est < 400, f"full prompt ~{tokens_est} tokens, want <400"

    def test_minimal_prompt_under_200_tokens(self):
        p = self._make_prompt(
            lookahead_text="  UP: blocked\n  LEFT: 13px",
            wm_summary="Match pattern.",
            recent_actions=[],
        )
        tokens_est = len(p) // 4
        assert tokens_est < 200, f"minimal ~{tokens_est} tokens, want <200"

    def test_lookahead_is_prominent(self):
        p = self._make_prompt()
        la_pos = p.find("Actions NOW")
        hint_pos = p.find("NN hint")
        action_pos = p.find("ACTION=")
        assert la_pos < hint_pos < action_pos

    def test_format_instruction_present(self):
        assert "ACTION=" in self._make_prompt()

    def test_special_signals_included(self):
        p = self._make_prompt(special_signals=["STUCK 5 steps", "LEVEL ADVANCE possible"])
        assert "STUCK" in p
        assert "LEVEL ADVANCE" in p

    def test_no_signals_clean(self):
        assert "⚠" not in self._make_prompt(special_signals=[])

    def test_recent_actions_shown(self):
        assert "UP → DOWN → LEFT" in self._make_prompt(recent_actions=["UP", "DOWN", "LEFT"])

    def test_click_targets_appended(self):
        assert "CLICK(32,48)" in self._make_prompt(click_targets="  CLICK(32,48): 150px")

    def test_world_model_truncated(self):
        assert "x" * 301 not in self._make_prompt(wm_summary="x" * 500)

    def test_nn_confidence_formatted(self):
        p = self._make_prompt(nn_hint="SEL", nn_confidence=0.85)
        assert "85%" in p

    def test_asks_about_progress(self):
        p = self._make_prompt()
        assert "PROGRESS" in p or "progress" in p.lower()
