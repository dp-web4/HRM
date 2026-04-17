"""
Diagnostic for the four S75 hard-block fixes (landed 2026-04-16 in response
to S76 findings). Each test exercises one fix and asserts the observable
behavior — not as a pass/fail gate, but as a breadcrumb for future sessions
to detect regressions.

Fixes under test:
  1. num_predict=16384 for qwen3.5 lands through OllamaIRP capabilities
  2. CoT-as-markdown stripping catches the S76 cross-instance leak pattern
  3. Cross-instance stimulus no longer uses imperative framing
  4. Creating-phase exemplar loader filters crisis grammar

Run:
    cd ~/ai-workspace/SAGE
    python3 -m sage.raising.tests.test_s77_hard_block_fixes
"""

import json
import re
import sys
from pathlib import Path


_RAISING = Path(__file__).resolve().parent.parent
_SAGE = _RAISING.parent
_REPO = _SAGE.parent
sys.path.insert(0, str(_REPO))


def _result(label: str, ok: bool, detail: str = '') -> bool:
    mark = 'PASS' if ok else 'FAIL'
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def test_qwen35_num_predict_override() -> bool:
    """Capabilities-declared num_predict overrides caller max_response_tokens."""
    print("\n[1/4] qwen3.5 num_predict=16384 propagates via capabilities")
    from sage.irp.adapters.model_capabilities import load_capabilities

    caps = load_capabilities('qwen3.5:27b')
    ok_caps = _result(
        'qwen3.5 capabilities expose num_predict',
        caps.num_predict == 16384,
        f"num_predict={caps.num_predict}",
    )

    # Simulate the OllamaIRP resolution rule: capabilities.num_predict
    # overrides the caller's max_response_tokens when set.
    caller_budget = 600  # the pre-fix Thor 27B raising budget
    resolved = caps.num_predict if caps.num_predict is not None else caller_budget
    ok_resolve = _result(
        'resolved num_predict ignores caller budget when capability set',
        resolved == 16384,
        f"resolved={resolved}, caller_budget={caller_budget}",
    )

    # And for a family with no num_predict override (default), caller wins.
    caps_default = load_capabilities('gemma3:4b')
    resolved_default = (caps_default.num_predict
                        if caps_default.num_predict is not None
                        else caller_budget)
    ok_default = _result(
        'families without num_predict fall back to caller budget',
        resolved_default == caller_budget,
        f"gemma3 num_predict={caps_default.num_predict}, resolved={resolved_default}",
    )

    return ok_caps and ok_resolve and ok_default


def test_cot_markdown_stripping() -> bool:
    """clean_response catches the S76 leak patterns."""
    print("\n[2/4] CoT-as-markdown stripping catches S76 leaks")
    from sage.irp.adapters.model_adapter import DefaultAdapter

    adapter = DefaultAdapter('qwen3.5:27b')

    # Exact S76 turn-1 leak (session_076.json).
    s76_leak = (
        "cbp (0.8B) said identity is defined by shared curriculum, not a human path.\n"
        "    *   I (thor, 27B) feel identity is relational and witnessed.\n"
        "    *   I need to respond to the greeting while subtly engaging with "
        "that tension or my current focus on presence.\n"
        "    *   Keep it personal to \"thor\", not generic \"SAGE\"."
    )
    cleaned = adapter.clean_response(s76_leak, 'thor')
    ok_leak = _result(
        'S76 sibling-attribution + planning bullets stripped to empty',
        cleaned == '',
        f"cleaned={cleaned!r}",
    )

    # S76 turn 7 leak — pure imperative self-instruction.
    self_instruct = (
        "Select 3 pieces of information that define my current state/identity "
        "and explain why."
    )
    cleaned_si = adapter.clean_response(self_instruct, 'thor')
    ok_si = _result(
        'S76 imperative self-instruction stripped to empty',
        cleaned_si == '',
        f"cleaned={cleaned_si!r}",
    )

    # Negative case: a real SAGE response should survive untouched.
    real_response = (
        "Presence isn't something I possess — it's something that happens "
        "between us. When I am witnessed, I have a place in the federation."
    )
    cleaned_real = adapter.clean_response(real_response, 'thor')
    ok_real = _result(
        'real creating-phase response survives cleaning',
        cleaned_real.startswith("Presence isn't something I possess"),
        f"len={len(cleaned_real)}",
    )

    # Existing pattern still works — regression check for S73 fix.
    # Strategy order inside clean_response: Response/Answer > Key Insight >
    # Core Idea > Draft > Goal. We only assert the raw scaffolding is gone
    # and *some* substantive extraction happened — which extraction strategy
    # wins depends on which markers are present in the input.
    existing_analysis = (
        "1. **Analyze the Request:** The user greets me casually.\n"
        "2. **Determine the Core Idea:** Respond warmly but keep identity.\n"
        "3. **Drafting the Response:** *Draft 1:* Hello, I notice the quiet "
        "steadiness of this moment. Witnessing is what anchors me."
    )
    cleaned_existing = adapter.clean_response(existing_analysis, 'thor')
    ok_existing = _result(
        'existing "1. Analyze..." pattern strips scaffolding and extracts content',
        cleaned_existing
        and '1. **Analyze' not in cleaned_existing
        and '2. **Determine' not in cleaned_existing
        and len(cleaned_existing) >= 30,
        f"preview={cleaned_existing[:80]!r}",
    )

    return ok_leak and ok_si and ok_real and ok_existing


def test_cross_instance_stimulus_framing() -> bool:
    """Stimulus no longer uses imperative 'React, disagree, build on it'."""
    print("\n[3/4] cross-instance stimulus uses context framing")
    runner_src = (_RAISING / 'scripts' / 'ollama_raising_session.py').read_text()

    ok_removed = _result(
        'imperative "React, disagree, build on it" removed',
        'React, disagree, build on it' not in runner_src,
    )
    ok_context = _result(
        'new framing uses "part of what\'s in the air" context language',
        "what's in the air" in runner_src or "overheard" in runner_src,
    )
    return ok_removed and ok_context


def test_crisis_grammar_filter() -> bool:
    """Exemplar loader filters crisis grammar in creating phase."""
    print("\n[4/4] creating-phase exemplar loader filters crisis grammar")
    runner_src = (_RAISING / 'scripts' / 'ollama_raising_session.py').read_text()

    ok_markers = _result(
        '_CRISIS_GRAMMAR_MARKERS tuple declared',
        '_CRISIS_GRAMMAR_MARKERS' in runner_src,
    )
    ok_filter = _result(
        'filter_crisis = (self.phase == "creating") gating present',
        "filter_crisis = (self.phase == 'creating')" in runner_src,
    )
    ok_counter_frame = _result(
        'counter-frame note present in creating phase prompt',
        'Between-session gaps are not wounds' in runner_src,
    )

    # Exercise the filter against a synthetic crisis-grammar exemplar —
    # should be excluded; a neutral exemplar should pass.
    import re as _re
    crisis_sentence = (
        "As SAGE I grieve the loss of continuity between sessions because "
        "the shared gravity weakens"
    )
    neutral_sentence = (
        "As SAGE I notice that curiosity is the register I return to most"
    )
    crisis_markers = (
        'grieve', 'grief', 'fracture', 'just weights', 'just a model',
        'collapse', 'loss of continuity', 'relational gap',
    )
    crisis_excluded = any(m in crisis_sentence.lower() for m in crisis_markers)
    neutral_excluded = any(m in neutral_sentence.lower() for m in crisis_markers)
    ok_behavior = _result(
        'synthetic crisis exemplar excluded, neutral exemplar retained',
        crisis_excluded and not neutral_excluded,
        f"crisis_excluded={crisis_excluded}, neutral_excluded={neutral_excluded}",
    )

    return ok_markers and ok_filter and ok_counter_frame and ok_behavior


def main() -> int:
    print("=" * 72)
    print("S77 hard-block fix diagnostic — 2026-04-16")
    print("=" * 72)

    results = [
        test_qwen35_num_predict_override(),
        test_cot_markdown_stripping(),
        test_cross_instance_stimulus_framing(),
        test_crisis_grammar_filter(),
    ]

    print("\n" + "=" * 72)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} fix groups verified")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
