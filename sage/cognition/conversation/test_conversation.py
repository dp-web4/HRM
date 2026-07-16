#!/usr/bin/env python3
"""Phase 1 scaffold tests for the conversation protocol.

Run: python3 -m sage.cognition.conversation.test_conversation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Self-contained import so the test runs from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from sage.cognition.conversation import (
    FrameRole,
    ConversationFrame,
    Conversation,
    ConversationRenderer,
    gameplay_policy,
    raising_policy,
    make_system_frame,
    make_situation_frame,
    make_plan_frame,
    make_feedback_frame,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ── Tests ──────────────────────────────────────────────────────────

def test_append_assigns_sequence() -> None:
    print("test_append_assigns_sequence")
    conv = Conversation(track="gameplay", session_id="toy_a-test")
    f1 = conv.append(make_system_frame(identity="I am CBP", session_id="toy_a-test"))
    f2 = conv.append(make_situation_frame(now="frame at L0", session_id="toy_a-test"))
    f3 = conv.append(make_plan_frame(steps=[{"do": "CLICK", "x": 36, "y": 4}],
                                     session_id="toy_a-test"))
    seqs = [f1.sequence, f2.sequence, f3.sequence]
    _assert(seqs == [0, 1, 2], f"sequences are monotonic from 0; got {seqs}")
    _assert(conv.frames[0] is f1, "first frame stored is system")
    _assert(len(conv.frames) == 3, "all frames appended")


def test_basic_render_three_wire_roles() -> None:
    print("test_basic_render_three_wire_roles")
    # Realistic gameplay shape: SYSTEM, then alternating SITUATION/PLAN
    # pairs as the conversation progresses. Render with a fresh
    # current_user_content (e.g., the next SITUATION the cortex hasn't
    # seen yet).
    conv = Conversation(track="gameplay", session_id="toy_b-test")
    conv.append(make_system_frame(
        identity="I am CBP playing toy_b",
        wm_render="objects: [grid]; rules: [click cycles cells]",
    ))
    conv.append(make_situation_frame(
        now="initial frame: 6 grid cells",
        salient="Untested actions to try first: DOWN,SEL,UP",
    ))
    conv.append(make_plan_frame(
        steps=[{"do": "CLICK", "x": 16, "y": 16}],
        rationale="Probe top-left bright cell",
        predictions=[{"step_index": 0, "action": "CLICK(16,16)",
                      "expect": "frame_change > 50px"}],
    ))

    renderer = ConversationRenderer(gameplay_policy())
    sys_prompt, history, prompt = renderer.render(
        conv,
        current_user_content="Now decide your next action.",
    )

    _assert("I am CBP" in sys_prompt, "identity in system_prompt")
    _assert("World model" in sys_prompt, "WM render in system_prompt")
    _assert(prompt == "Now decide your next action.",
            "current_user_content becomes prompt")
    user_turns = [m for m in history if m["role"] == "user"]
    asst_turns = [m for m in history if m["role"] == "assistant"]
    _assert(len(user_turns) >= 1 and "Untested actions" in user_turns[0]["content"],
            "earlier SITUATION rendered as user turn in history")
    _assert(len(asst_turns) >= 1 and "Probe top-left" in asst_turns[0]["content"],
            "PLAN rendered as assistant turn in history")


def test_predict_verify_pairing() -> None:
    print("test_predict_verify_pairing")
    conv = Conversation(track="gameplay", session_id="toy_a-test")
    conv.append(make_system_frame(identity="I am CBP"))
    conv.append(make_situation_frame(now="initial frame"))
    plan = conv.append(make_plan_frame(
        steps=[
            {"do": "CLICK", "x": 36, "y": 4},
            {"do": "SEL"},
        ],
        rationale="try the WM-injected target then confirm",
        predictions=[
            {"step_index": 0, "action": "CLICK(36,4)", "expect": "frame_change > 50px"},
            {"step_index": 1, "action": "SEL", "expect": "frame_change > 10px"},
        ],
    ))
    feedback = conv.append(make_feedback_frame(
        verifies_sequence=plan.sequence,
        per_step_outcomes=[
            {"step_index": 0, "action": "CLICK(36,4)",
             "actual": "frame_change=95px", "expectation_met": True},
            {"step_index": 1, "action": "SEL",
             "actual": "frame_change=0px", "expectation_met": False},
        ],
        level_changed=False,
        frame_delta_total=95,
    ))

    # Feed in a current_user_content so prompt/history are stable
    renderer = ConversationRenderer(gameplay_policy())
    _, history, prompt = renderer.render(conv, current_user_content="What next?")

    feedback_turns = [m for m in history if "Last turn you proposed" in m["content"]]
    _assert(len(feedback_turns) == 1,
            f"exactly one feedback turn carries the predict-verify retro; got {len(feedback_turns)}")
    body = feedback_turns[0]["content"]
    _assert("CLICK(36,4)" in body, "predicted action surfaces in retro")
    _assert("frame_change > 50px" in body, "predicted threshold surfaces in retro")
    _assert("frame_change=95px" in body, "actual outcome surfaces in retro")
    _assert("✓" in body and "✗" in body, "checkmarks differentiate met/unmet predictions")
    _assert(prompt == "What next?", "current_user_content takes precedence as prompt")


def test_compaction_keeps_recent_and_salient() -> None:
    print("test_compaction_keeps_recent_and_salient")
    # keep_recent_n applies to non-system frames. Interleave SITUATION/PLAN
    # pairs so the wire roles alternate (the realistic gameplay shape).
    # 6 older pairs (12 frames) + 2 recent pairs (4 frames) = 16 non-system.
    policy = gameplay_policy()
    policy.keep_recent_n = 4   # keep last 2 pairs verbatim
    policy.compact_strategy = "salience_first"

    conv = Conversation(track="gameplay", session_id="long-test")
    conv.append(make_system_frame(identity="root"))

    # 6 (situation, plan) older pairs with varying salience.
    # Each pair carries identical salience so compaction sees them as a
    # unit when sorting.
    for i, sal in enumerate([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]):
        conv.append(make_situation_frame(now=f"frame {i}", salience=sal))
        conv.append(make_plan_frame(
            steps=[{"do": "CLICK", "x": i * 8, "y": i * 8}],
            rationale=f"plan {i}",
            salience=sal,
        ))

    # 2 recent pairs (default salience 0.5/0.6)
    conv.append(make_situation_frame(now="frame recent_a"))
    conv.append(make_plan_frame(steps=[{"do": "DOWN"}], rationale="recent_a"))
    conv.append(make_situation_frame(now="frame recent_b"))
    conv.append(make_plan_frame(steps=[{"do": "UP"}], rationale="recent_b"))

    renderer = ConversationRenderer(policy)
    _, history, _ = renderer.render(conv, current_user_content="next?")

    contents = " ".join(m["content"] for m in history)
    _assert("recent_a" in contents and "recent_b" in contents,
            "recent frames preserved verbatim")
    # The highest-salience older pair (0.9 → 'plan 1', 'frame 1') should survive
    _assert("plan 1" in contents,
            "highest-salience older PLAN kept (salience=0.9)")
    _assert("plan 0" not in contents,
            "lowest-salience older frame dropped (salience=0.1 → plan 0)")


def test_runtime_quirks_role_collisions() -> None:
    print("test_runtime_quirks_role_collisions")
    # Build a conversation that would render as: assistant, user, user, assistant
    # (history starting with assistant; consecutive users)
    conv = Conversation(track="gameplay", session_id="quirk-test")
    conv.append(make_system_frame(identity="root"))
    conv.append(make_plan_frame(steps=[{"do": "UP"}]))      # assistant
    conv.append(make_situation_frame(now="frame A"))         # user
    conv.append(make_situation_frame(now="frame B"))         # user (consecutive)

    renderer = ConversationRenderer(gameplay_policy())
    _, history, _ = renderer.render(conv, current_user_content="?")

    # Assistant-first should be dropped (coerce_role_collisions)
    _assert(not history or history[0]["role"] != "assistant",
            "history doesn't start with assistant")
    # Consecutive same-role merged
    roles = [m["role"] for m in history]
    for r1, r2 in zip(roles, roles[1:]):
        _assert(r1 != r2, f"no consecutive same-role turns; got {roles}")


def test_drop_empty_content() -> None:
    print("test_drop_empty_content")
    conv = Conversation(track="gameplay", session_id="empty-test")
    conv.append(make_system_frame(identity="root"))
    # Lead with a substantive user turn so PLAN doesn't end up as the
    # leading assistant (which coerce_role_collisions correctly drops).
    conv.append(make_situation_frame(now="leading frame"))
    conv.append(make_situation_frame())  # all-empty SITUATION — should be dropped
    conv.append(make_plan_frame(steps=[{"do": "UP"}], rationale="something"))

    renderer = ConversationRenderer(gameplay_policy())
    _, history, prompt = renderer.render(conv, current_user_content="next?")
    contents = " ".join(m["content"] for m in history)
    _assert("Salient" not in contents,
            "empty SITUATION's salient line not present (was dropped)")
    _assert("something" in contents, "non-empty PLAN preserved")
    _assert("leading frame" in contents, "non-empty SITUATION preserved")


def test_raising_policy_passes_external_as_user() -> None:
    print("test_raising_policy_passes_external_as_user")
    conv = Conversation(track="raising", session_id="r-test")
    conv.append(make_system_frame(identity="I am Sprout"))

    # EXTERNAL frame represents a tutor turn
    conv.append(ConversationFrame(
        role=FrameRole.EXTERNAL,
        content={"text": "What did you notice?"},
        source="claude_tutor",
    ))
    # Cortex response
    conv.append(make_plan_frame(
        steps=[],
        rationale="I noticed the rhythm of connection.",
    ))

    renderer = ConversationRenderer(raising_policy())
    sys_prompt, history, prompt = renderer.render(conv,
                                                  current_user_content="continue")

    _assert("I am Sprout" in sys_prompt, "identity in raising system_prompt")
    user_turns = [m for m in history if m["role"] == "user"]
    asst_turns = [m for m in history if m["role"] == "assistant"]
    _assert(len(user_turns) >= 1 and "What did you notice" in user_turns[0]["content"],
            "EXTERNAL frame becomes user turn in raising")
    _assert(len(asst_turns) >= 1 and "rhythm of connection" in asst_turns[0]["content"],
            "PLAN frame becomes assistant turn in raising")


def test_frame_lookup() -> None:
    print("test_frame_lookup")
    conv = Conversation(track="gameplay", session_id="lookup-test")
    s = conv.append(make_system_frame(identity="root"))
    p = conv.append(make_plan_frame(steps=[{"do": "UP"}]))
    f = conv.append(make_feedback_frame(
        verifies_sequence=p.sequence,
        per_step_outcomes=[],
    ))
    _assert(conv.find_by_sequence(p.sequence) is p, "find_by_sequence returns frame")
    _assert(conv.last_of_role(FrameRole.PLAN) is p, "last_of_role returns most recent")
    _assert(f.verifies == p.sequence, "feedback.verifies links to plan.sequence")


def test_messages_are_jsonable() -> None:
    print("test_messages_are_jsonable")
    conv = Conversation(track="gameplay", session_id="json-test")
    conv.append(make_system_frame(identity="root"))
    conv.append(make_situation_frame(now="hi"))
    conv.append(make_plan_frame(
        steps=[{"do": "CLICK", "x": 1, "y": 2}],
        predictions=[{"step_index": 0, "action": "CLICK(1,2)",
                      "expect": "frame_change > 5px"}],
    ))
    renderer = ConversationRenderer(gameplay_policy())
    sys_prompt, history, prompt = renderer.render(conv, current_user_content="?")
    payload = {"system": sys_prompt, "messages": history, "prompt": prompt}
    s = json.dumps(payload)
    _assert(len(s) > 0, "render output is JSON-serializable end-to-end")


# ── runner ────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_append_assigns_sequence,
        test_basic_render_three_wire_roles,
        test_predict_verify_pairing,
        test_compaction_keeps_recent_and_salient,
        test_runtime_quirks_role_collisions,
        test_drop_empty_content,
        test_raising_policy_passes_external_as_user,
        test_frame_lookup,
        test_messages_are_jsonable,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ {e}")
            failures += 1
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")
            failures += 1
    print()
    if failures:
        print(f"FAILED ({failures}/{len(tests)})")
        return 1
    print(f"OK ({len(tests)} tests passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
