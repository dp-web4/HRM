"""Tests for the MRH context architecture (Phase 5 B).

Covers:
  - Each block type renders, summarizes, and gives a minimal floor line
  - estimate_cost returns positive int
  - Priority defaults respect fleet consensus (Task 100, Sensors 95,
    Effectors 90, Identity 85, Mechanics 75, Metabolic 70, Experiential 50)
  - should_drop behavior at pressure thresholds
  - MRHContext.compose() produces system + user halves
  - System half routes Identity + Mechanics + Effectors; user half
    routes Task + Sensors + Metabolic + Experiential
  - Pure rendering: no I/O or state mutation during render/summarize
  - swap_recommendations flow from MetabolicBlock to MRHContext
  - Image attachments collected via collect_image_attachments
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.context.mrh import (
    MRHBlock, MRHContext, BlockRenderMode,
    IdentityBlock, SensorsBlock, EffectorsBlock,
    MechanicsBlock, ExperientialCacheBlock,
    MetabolicBlock, TaskBlock,
)
from sage.context.mrh.base import ImageAttachment, collect_image_attachments
from sage.context.mrh.effectors import HabitMatch, MotorSkill
from sage.context.mrh.experiential import TrajectoryEntry, EpisodicMatch
from sage.context.mrh.metabolic import MetacogSignalSummary


# ───────────────────────────────────────────────────────────────────
# Block basics — each type renders cleanly
# ───────────────────────────────────────────────────────────────────

def test_identity_block_renders_lens_not_description():
    """Sprout's rule: identity must be a lens, not a description.

    The preset lenses should orient toward the task, not describe SAGE.
    """
    b = IdentityBlock(mode="solo_gladiator")
    text = b.render(budget_tokens=500)
    assert "game" in text.lower()
    assert "goal" in text.lower() or "action" in text.lower() or "observe" in text.lower()
    # Anti-check: no "I am a" / "I am the" self-description patterns
    assert "I am a " not in text
    assert "I am the " not in text
    # Priority default
    assert b.priority == 85


def test_identity_block_mode_variants_differ():
    solo = IdentityBlock(mode="solo_gladiator").render(500)
    partnered = IdentityBlock(mode="partnered").render(500)
    assert solo != partnered
    assert "partner" in partnered.lower() or "witness" in partnered.lower()
    # solo should NOT include partnership language
    assert "partner" not in solo.lower()


def test_identity_block_addendum_appended():
    b = IdentityBlock(mode="solo_gladiator", addendum="Session note: be curious.")
    text = b.render(500)
    assert "Session note: be curious." in text


def test_sensors_block_renders_images_and_text():
    img = ImageAttachment(png_bytes=b"fake", label="current_frame")
    b = SensorsBlock(
        description="Two frames side by side",
        image_attachments=[img],
        text_snippets=[("note", "player at left")],
    )
    text = b.render(500)
    assert "Perception" in text
    assert "side by side" in text
    assert "current_frame" in text
    assert "player at left" in text
    assert b.priority == 95


def test_sensors_block_minimal_fallback_under_pressure():
    img = ImageAttachment(png_bytes=b"x", label="frame")
    b = SensorsBlock(description="", image_attachments=[img])
    m = b.minimal()
    assert "1 image" in m


def test_effectors_block_game_actions_default():
    b = EffectorsBlock(kind_profile="game_actions")
    text = b.render(500)
    assert "CLICK" in text
    assert "UP" in text
    assert "ACTION=" in text   # response format included
    assert b.priority == 90


def test_effectors_block_with_habit_match():
    h = HabitMatch(
        habit_id="h-toy_a-L1-abc",
        confidence=0.95,
        reliability=1.0,
        sequence=["LEFT", "DOWN", "CLICK"],
        source_count=3,
    )
    b = EffectorsBlock(kind_profile="game_actions", habit_match=h)
    text = b.render(500)
    assert "h-toy_a-L1-abc" in text
    assert "ACTION=HABIT" in text
    assert "LEFT" in text


def test_effectors_block_with_motor_skills():
    skill = MotorSkill(
        name="navigate_to",
        signature="SKILL=navigate_to X=<int> Y=<int>",
        description="moves to target",
    )
    b = EffectorsBlock(motor_skills=[skill])
    text = b.render(500)
    assert "navigate_to" in text
    assert "SKILL=" in text


def test_effectors_block_text_profile_differs_from_game():
    game = EffectorsBlock(kind_profile="game_actions").render(500)
    text = EffectorsBlock(kind_profile="text").render(500)
    assert game != text


def test_mechanics_block_renders_world_model():
    b = MechanicsBlock(
        world_model_text="Clicking destroys colored tiles. Gravity pulls up.",
        mechanics_cluster="toy_f (0.42), toy_g (0.38)",
        game_family="toy_b",
    )
    text = b.render(1000)
    assert "Clicking destroys colored tiles" in text
    assert "toy_f" in text
    assert b.priority == 75


def test_mechanics_block_stuck_escape_profile():
    b = MechanicsBlock(
        world_model_text="world",
        profile="stuck_escape",
    )
    text = b.render(500)
    assert "Stuck-escape" in text or "stuck" in text.lower()


def test_mechanics_block_truncates_long_world_model_under_pressure():
    huge = "\n\n".join(["paragraph " + str(i) + " " * 200 for i in range(50)])
    b = MechanicsBlock(world_model_text=huge)
    # tiny budget
    text = b.render(budget_tokens=64)
    # Should include a truncation marker or be smaller
    assert len(text) < len(huge)


def test_experiential_block_trajectory_summary_no_llm_quotes():
    """Sprout's rule: summary never quotes the LLM back to itself.

    ExperientialCacheBlock compresses trajectory entries, not LLM
    rationales. The summary is derived from action+outcome data only.
    """
    traj = [
        TrajectoryEntry(step=i, action_name="CLICK", coords={"x": 32, "y": 32},
                        frame_delta_pct=0.0, level_before=0, level_after=0)
        for i in range(1, 12)
    ]
    b = ExperientialCacheBlock(recent_trajectory=traj, verbatim_tail=3)
    text = b.render(1000)
    assert "Earlier:" in text
    assert "Most recent steps" in text
    # Only last 3 should appear verbatim
    assert "step 11" in text
    assert "step 10" in text
    assert b.priority == 50


def test_experiential_block_with_episodic_and_patterns():
    ep = EpisodicMatch(formatted_text="toy_b L0: CLICK(12,12) → advance", similarity=0.8)
    b = ExperientialCacheBlock(
        recent_trajectory=[
            TrajectoryEntry(step=1, action_name="CLICK", frame_delta_pct=5.0),
        ],
        pattern_flags=["⚠️ STUCK: last 3 steps no progress"],
        episodic_matches=[ep],
        retrieved_patterns=["From Andy's filter: 'basket launches paint'"],
    )
    text = b.render(1000)
    assert "CLICK(12,12)" in text
    assert "STUCK" in text
    assert "basket launches paint" in text


def test_metabolic_block_renders_crisp_numerical_signals():
    """Sprout's rule: metabolic is numerical, not narrative."""
    sig = MetacogSignalSummary(
        signal="perseveration",
        severity=0.8,
        evidence={"repeats": 3, "action": "CLICK"},
        suggestion="try a different action",
    )
    b = MetabolicBlock(
        metacog_signals=[sig],
        metabolic_state="focus",
        atp_balance=847.0,
        atp_trend="falling",
        confidence=0.31,
        stuck_duration=5,
        actions_since_last_invoke=12,
    )
    text = b.render(500)
    # Crisp status line with numbers
    assert "ATP=847" in text
    assert "confidence=0.31" in text
    assert "stuck=5" in text
    # Metacog signal surfaced
    assert "perseveration" in text
    assert "severity 0.8" in text
    assert "try a different action" in text
    assert b.priority == 70


def test_metabolic_block_swap_recommendations():
    b = MetabolicBlock(swap_recommendations=["mechanics:stuck_escape"])
    recs = b.get_swap_recommendations()
    assert recs == ["mechanics:stuck_escape"]


def test_task_block_renders_game_level_step():
    b = TaskBlock(
        goal="Pick the best next action",
        invoke_reasons=["stuck"],
        step_index=14,
        level=0,
        game_family="toy_b",
    )
    text = b.render(200)
    assert "toy_b" in text
    assert "Level: 0" in text
    assert "Step: 14" in text
    assert "stuck" in text
    assert b.priority == 100   # Irreducible


# ───────────────────────────────────────────────────────────────────
# Render purity (Thor's rule)
# ───────────────────────────────────────────────────────────────────

def test_render_is_pure_no_state_mutation():
    """Block render must not mutate block state (population is dispatcher's job)."""
    b = MechanicsBlock(world_model_text="world", game_family="toy_b")
    before = (b.world_model_text, b.game_family)
    _ = b.render(500)
    _ = b.render(1000)
    _ = b.summarize(100)
    after = (b.world_model_text, b.game_family)
    assert before == after


def test_render_idempotent():
    b = TaskBlock(goal="test", invoke_reasons=["probe"])
    a = b.render(200)
    b2 = b.render(200)
    assert a == b2


# ───────────────────────────────────────────────────────────────────
# Priority + pressure
# ───────────────────────────────────────────────────────────────────

def test_priority_defaults_match_fleet_consensus():
    assert TaskBlock().priority == 100
    assert SensorsBlock().priority == 95
    assert EffectorsBlock().priority == 90
    assert IdentityBlock().priority == 85
    assert MechanicsBlock().priority == 75
    assert MetabolicBlock().priority == 70
    assert ExperientialCacheBlock().priority == 50


def test_should_drop_threshold():
    low = ExperientialCacheBlock()       # priority 50
    high = TaskBlock()                    # priority 100
    # Under mild pressure, low-priority drops, high stays
    assert low.should_drop(pressure=0.6) is True
    assert high.should_drop(pressure=0.6) is False
    # Task should never drop
    assert high.should_drop(pressure=0.99) is False


# ───────────────────────────────────────────────────────────────────
# MRHContext.compose
# ───────────────────────────────────────────────────────────────────

def _build_minimal_context() -> MRHContext:
    """Build a minimal-but-complete context for composition tests."""
    return MRHContext(
        identity=IdentityBlock(mode="solo_gladiator"),
        sensors=SensorsBlock(description="prev/curr frame pair"),
        effectors=EffectorsBlock(kind_profile="game_actions"),
        mechanics=MechanicsBlock(
            world_model_text="test mechanics",
            game_family="toy_b",
        ),
        experiential=ExperientialCacheBlock(),
        metabolic=MetabolicBlock(metabolic_state="active"),
        task=TaskBlock(
            game_family="toy_b",
            level=0,
            step_index=5,
            invoke_reasons=["probe"],
        ),
    )


def test_compose_returns_system_and_user_halves():
    ctx = _build_minimal_context()
    system, user = ctx.compose()
    assert isinstance(system, str)
    assert isinstance(user, str)
    assert system
    assert user


def test_compose_system_has_identity_mechanics_effectors():
    ctx = _build_minimal_context()
    system, _ = ctx.compose()
    # Identity lens content
    assert "game" in system.lower()
    # Effectors content
    assert "CLICK" in system
    # Mechanics content
    assert "test mechanics" in system


def test_compose_user_has_task_sensors_metabolic_experiential():
    ctx = _build_minimal_context()
    _, user = ctx.compose()
    # Task
    assert "toy_b" in user
    assert "Step: 5" in user
    # Sensors
    assert "frame pair" in user


def test_compose_respects_budget():
    """With a very small budget, blocks should compress/fall through to minimal."""
    ctx = _build_minimal_context()
    # Tiny budget — force compression path
    system, user = ctx.compose(system_budget_tokens=100, user_budget_tokens=100)
    # Should still have content (never-omit floor)
    assert system
    assert user


def test_swap_recommendations_flow_from_metabolic_to_context():
    ctx = _build_minimal_context()
    ctx.metabolic.swap_recommendations = ["mechanics:stuck_escape", "identity:escalate"]
    recs = ctx.swap_recommendations()
    assert "mechanics:stuck_escape" in recs
    assert "identity:escalate" in recs


def test_collect_image_attachments_from_sensors():
    ctx = _build_minimal_context()
    img = ImageAttachment(png_bytes=b"xyz", label="current")
    ctx.sensors.image_attachments.append(img)
    images = collect_image_attachments(ctx)
    assert len(images) == 1
    assert images[0].label == "current"


def test_blocks_returned_in_priority_order():
    ctx = _build_minimal_context()
    blocks = ctx.blocks()
    priorities = [b.priority for b in blocks]
    assert priorities == sorted(priorities, reverse=True)


# ───────────────────────────────────────────────────────────────────
# estimate_cost sanity
# ───────────────────────────────────────────────────────────────────

def test_estimate_cost_positive_for_every_block_type():
    blocks: list[MRHBlock] = [
        IdentityBlock(mode="solo_gladiator"),
        SensorsBlock(description="x"),
        EffectorsBlock(),
        MechanicsBlock(world_model_text="mechanics"),
        ExperientialCacheBlock(recent_trajectory=[TrajectoryEntry(step=1, action_name="UP")]),
        MetabolicBlock(),
        TaskBlock(goal="g"),
    ]
    for b in blocks:
        cost = b.estimate_cost()
        assert cost > 0, f"{b.kind} returned non-positive cost"


# ───────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────
# API aliases (Sprout's raising-adoption uses shorter field names)
# ───────────────────────────────────────────────────────────────────

def test_effectors_profile_aliases_kind_profile():
    """`profile` is an alias for `kind_profile` — Sprout raising usage."""
    b = EffectorsBlock(profile="text", addendum="50-100 words.")
    assert b.kind_profile == "text"
    assert b.addendum == "50-100 words."


def test_effectors_addendum_renders_without_response_format_for_text():
    """Text profile: addendum carries format guidance; response_format
    (ACTION=...) omitted since text profiles don't emit game actions."""
    b = EffectorsBlock(profile="text", addendum="50-100 words. One idea.")
    text = b.render(500)
    assert "50-100 words" in text
    # No ACTION= since this isn't game_actions
    assert "ACTION=" not in text


def test_experiential_trajectory_summary_aliases_conversation_summary():
    b = ExperientialCacheBlock(trajectory_summary="prev session summary")
    assert b.conversation_summary == "prev session summary"


def test_metabolic_phase_aliases_metabolic_state():
    b = MetabolicBlock(phase="wake")
    assert b.metabolic_state == "wake"


def test_task_description_aliases_goal():
    b = TaskBlock(description="Raising session 99 — phase: creating")
    assert b.goal == "Raising session 99 — phase: creating"


def test_compose_accepts_max_tokens_shorthand():
    """compose(max_tokens=N) splits 75/25 system/user for single-knob callers."""
    ctx = _build_minimal_context()
    system, user = ctx.compose(max_tokens=4000)
    assert system
    assert user


def test_sprout_raising_idiom_end_to_end():
    """Exact usage pattern from sage/raising/scripts/ollama_raising_session.py
    (Sprout's _build_system_prompt_mrh). Must not raise TypeError."""
    ctx = MRHContext(
        identity=IdentityBlock(mode="partnered", addendum="Express freshly."),
        sensors=SensorsBlock(),
        effectors=EffectorsBlock(profile="text", addendum="50-100 words."),
        mechanics=MechanicsBlock(world_model_text="phase=creating"),
        experiential=ExperientialCacheBlock(trajectory_summary="prev summary"),
        metabolic=MetabolicBlock(metacog_signals=[], confidence=0.7, phase="wake"),
        task=TaskBlock(description="Raising session 99 — phase: creating"),
    )
    system, user = ctx.compose(max_tokens=30000)
    assert isinstance(system, str) and system
    assert isinstance(user, str)   # user may be empty if no user-side content


if __name__ == "__main__":
    failures = 0
    for name in list(globals()):
        if name.startswith("test_"):
            try:
                globals()[name]()
                print(f"  \u2713 {name}")
            except AssertionError as e:
                print(f"  \u2717 {name}: {e}")
                failures += 1
            except Exception as e:
                print(f"  \u2717 {name}: {type(e).__name__}: {e}")
                failures += 1
    if failures:
        print(f"\n{failures} failures")
        sys.exit(1)
    print("\nAll MRH tests passed")
