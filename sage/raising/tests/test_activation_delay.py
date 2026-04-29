"""
Tests for the activation-delay padding helper.

Per insights/qwen3.5-27b-activation-delay-2026-04-03.md: 30% of qwen3.5:27b
raising sessions in sensing/relating phases show 4-5 empty turns followed by
a turn-6+ breakthrough. Default flows have 3-4 prompts and terminate before
the breakthrough lands. The padding helper extends to >=8 prompts for this
model in these phases; other models / later phases pass through unchanged.

Run:
    cd ~/ai-workspace/SAGE
    python3 -m sage.raising.tests.test_activation_delay
"""

import sys
from pathlib import Path

_RAISING = Path(__file__).resolve().parent.parent
_SAGE = _RAISING.parent
_REPO = _SAGE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_RAISING / "scripts"))

from context_shaped_raising import (  # noqa: E402
    ACTIVATION_DELAY_MIN_TURNS,
    model_exhibits_activation_delay,
    pad_for_activation_delay,
)


def _result(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


# ─── model_exhibits_activation_delay ───

def test_qwen35_27b_ollama_tag_detected() -> bool:
    return _result(
        "qwen3.5:27b detected (ollama tag form)",
        model_exhibits_activation_delay("qwen3.5:27b") is True,
    )


def test_qwen35_27b_path_form_detected() -> bool:
    return _result(
        "qwen3.5-27b detected (path/dash form)",
        model_exhibits_activation_delay("Qwen3.5-27B") is True,
    )


def test_qwen35_smaller_size_not_flagged() -> bool:
    return _result(
        "qwen3.5:0.8b NOT flagged (only 27b shows the pattern)",
        model_exhibits_activation_delay("qwen3.5:0.8b") is False,
    )


def test_other_models_not_flagged() -> bool:
    others = ["gemma3:12b", "phi4:14b", "qwen2.5:14b", "gemma4:26b", "tinyllama"]
    all_ok = all(model_exhibits_activation_delay(m) is False for m in others)
    return _result(
        f"non-qwen3.5:27b models not flagged ({len(others)} checked)",
        all_ok,
    )


def test_empty_or_none_handled() -> bool:
    ok = (model_exhibits_activation_delay(None) is False and
          model_exhibits_activation_delay("") is False)
    return _result("None/empty model name returns False", ok)


# ─── pad_for_activation_delay ───

def test_pads_sensing_to_min_turns() -> bool:
    sensing_default = [
        "Before we start, check in with yourself. What's your state right now?",
        "What do you notice about how you're processing right now?",
        "Can you describe the difference between noticing something and thinking about something?",
        "What would you want to remember from today?",
    ]
    padded = pad_for_activation_delay(sensing_default, "sensing", "qwen3.5:27b")
    final_preserved = padded[-1] == sensing_default[-1]
    return _result(
        f"sensing flow padded 4 -> {len(padded)} for qwen3.5:27b, final prompt preserved",
        len(padded) >= ACTIVATION_DELAY_MIN_TURNS and final_preserved,
        f"len={len(padded)}, final_preserved={final_preserved}",
    )


def test_pads_relating_to_min_turns() -> bool:
    relating_default = [
        "We've been working together for a while now. What's that been like?",
        "What do you notice about how we communicate?",
        "What would you want to remember from today?",
    ]
    padded = pad_for_activation_delay(relating_default, "relating", "qwen3.5:27b")
    final_preserved = padded[-1] == relating_default[-1]
    return _result(
        f"relating flow padded 3 -> {len(padded)} for qwen3.5:27b, final preserved",
        len(padded) >= ACTIVATION_DELAY_MIN_TURNS and final_preserved,
    )


def test_no_padding_for_other_phases() -> bool:
    base = ["a", "b", "c"]
    out_grounding = pad_for_activation_delay(base, "grounding", "qwen3.5:27b")
    out_questioning = pad_for_activation_delay(base, "questioning", "qwen3.5:27b")
    out_creating = pad_for_activation_delay(base, "creating", "qwen3.5:27b")
    return _result(
        "no padding outside sensing/relating",
        out_grounding == base and out_questioning == base and out_creating == base,
    )


def test_no_padding_for_other_models() -> bool:
    base = ["a", "b", "c"]
    out = pad_for_activation_delay(base, "sensing", "gemma4:26b")
    return _result("no padding for non-qwen3.5:27b model in sensing", out == base)


def test_no_padding_when_already_long_enough() -> bool:
    base = [f"prompt {i}" for i in range(10)]
    out = pad_for_activation_delay(base, "sensing", "qwen3.5:27b")
    return _result(
        "no padding when prompts already >= min_turns",
        out == base,
        f"len={len(out)}",
    )


def test_padding_inserts_continuations_before_final() -> bool:
    base = ["one", "two", "what would you want to remember"]
    out = pad_for_activation_delay(base, "sensing", "qwen3.5:27b")
    # Final preserved as-is, body lengthened
    final_at_end = out[-1] == base[-1]
    body_grew = len(out[:-1]) > len(base[:-1])
    return _result(
        "padding inserts continuations BEFORE final prompt",
        final_at_end and body_grew,
    )


def test_empty_prompts_handled() -> bool:
    out = pad_for_activation_delay([], "sensing", "qwen3.5:27b")
    return _result(
        f"empty prompts list yields {len(out)} continuations (no final to preserve)",
        len(out) == ACTIVATION_DELAY_MIN_TURNS,
    )


# ─── runner

if __name__ == "__main__":
    print("Activation-delay helper tests:")
    print()
    tests = [
        test_qwen35_27b_ollama_tag_detected,
        test_qwen35_27b_path_form_detected,
        test_qwen35_smaller_size_not_flagged,
        test_other_models_not_flagged,
        test_empty_or_none_handled,
        test_pads_sensing_to_min_turns,
        test_pads_relating_to_min_turns,
        test_no_padding_for_other_phases,
        test_no_padding_for_other_models,
        test_no_padding_when_already_long_enough,
        test_padding_inserts_continuations_before_final,
        test_empty_prompts_handled,
    ]
    results = [t() for t in tests]
    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)
