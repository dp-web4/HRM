#!/usr/bin/env python3
"""
Solver Context — 4-layer context assembly for LLM planning prompts.

Part of SAGE Solver v11 modular architecture.

Layers:
  L4 (metacognitive)  — stable principles from fleet raising (~500 tokens)
  L3 (membot/fleet)   — adaptive situation-relevant memories (~2K tokens)
  L2 (game KB)        — scene, action model, objects, mechanics (~3K tokens)
  L1 (narrative)      — session history with compression (~5K tokens)

Total: ~10.5K tokens, leaving ~120K for reasoning.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from solver_actions import obj_name, obj_catalog, ACTION_NAMES


# ─── Layer 4: Metacognitive Principles (fixed, from fleet raising) ───

METACOGNITIVE = """METACOGNITIVE PRINCIPLES (from fleet raising experience):
- Track state transitions. When actions have periodic effects, count the period.
- Hold multiple hypotheses. Test the most distinguishing one first.
- Uncertainty is information. Not knowing what a button does = click it ONCE.
- Persistence != perseveration. If an approach isn't producing new signal, change approach.
- Count before planning. Find cycle lengths before attempting solutions.
- Distinguish correlation from causation. Large pixel change != progress toward goal.
- Look for STRUCTURE, not just objects. Connectors between boxes = parent-child hierarchy.
  Border colors that match between a slot and a group = "this slot expands into that group."
- When stuck, look at the visual structure more carefully -- don't enumerate all possibilities.
  The answer is usually in what you're NOT seeing, not in what you haven't tried.
- Carry patterns across levels. If a rule worked on the previous level, it's the first
  hypothesis for the current level. Test it before exploring alternatives.
- WHERE you click matters as much as WHAT you click. Placement position determines slot assignment.
- Border colors ARE semantic cues. A colored border on a slot that matches an indicator color
  means "this indicator goes HERE." Color is information, not decoration.
- Rules EVOLVE between levels. The same game can change its rule from one level to the next.
  Don't assume the pattern is fixed -- test it, and if it fails, look for what changed.
- FIRST verify your clicks work. Click one visible object and confirm pixels change AT that spot.
  If only a counter changes far away, your coordinate mapping is wrong -- recalibrate.
- Zoom out: pixels -> objects -> groups -> constraints. Solve at the highest abstraction level.
  Small multi-colored patterns ARE instructions. Decode them before acting.
- Objects have ROLES: buttons (cause remote changes), pieces (change themselves), indicators
  (encode constraints), targets (show where pieces go). Identify roles first, then plan."""


# ─── Context Budget Tracking ───

try:
    _raising_scripts = os.path.join(
        os.path.dirname(__file__), '..', '..', 'sage', 'raising', 'scripts')
    sys.path.insert(0, _raising_scripts)
    from context_shaped_raising import ContextBudget
except ImportError:
    class ContextBudget:
        """Fallback if raising module not available."""
        def __init__(self, max_tokens=131072):
            self.max_tokens = max_tokens
            self.layers = {}

        def record(self, name, text):
            self.layers[name] = len(text)

        def total_chars(self):
            return sum(self.layers.values())

        def utilization(self):
            return (self.total_chars() / 4) / self.max_tokens

        def report(self):
            total = self.total_chars() // 4
            lines = [
                f"Context: ~{total} tokens "
                f"({self.utilization():.1%} of {self.max_tokens})"]
            for n, c in sorted(self.layers.items(), key=lambda x: -x[1]):
                lines.append(f"  {n}: ~{c // 4} tokens")
            lines.append(f"  FREE: ~{self.max_tokens - total} tokens")
            return "\n".join(lines)


def assemble_context_and_plan(backend, narrative, tracker, action_model,
                              available, levels, win_levels, grid=None,
                              ctx=None, attempt_num=1, verbose=False,
                              fleet_context="", vision_insight="",
                              mechanics=None):
    """Assemble all 4 layers with budget tracking, ask backend for next actions.

    Args:
        backend: ModelBackend instance (for querying and capability detection)
        narrative: SessionNarrative instance
        tracker: SpatialTracker instance
        action_model: ActionEffectModel instance
        available: list of available action ints
        levels: current levels completed
        win_levels: total levels to win
        grid: current game grid (for vision, optional)
        ctx: ContextConstructor instance (optional)
        attempt_num: current attempt number
        verbose: print budget info
        fleet_context: pre-built federation context string
        vision_insight: pre-computed vision analysis string
        mechanics: GameMechanics instance (optional)

    Returns:
        LLM response text with next actions.
    """
    budget = ContextBudget()
    has_move = any(a in available for a in [1, 2, 3, 4])

    # Layer 4: Metacognitive principles (fixed, ~500 tokens)
    layer4 = METACOGNITIVE
    budget.record("L4_metacognitive", layer4)

    # Layer 3: Situation-relevant membot memories (adaptive, target ~2K)
    patterns = narrative.detect_patterns()
    interactive_names = [obj_name(o) for o in tracker.get_interactive_objects()]
    situation = f"Playing game level {levels}/{win_levels}. "
    if interactive_names:
        situation += f"Interactive objects: {', '.join(interactive_names)}. "
    if "STABLE" in patterns:
        situation += "Grid similarity stable -- not progressing. "
    if "WARNING" in patterns:
        situation += "Repeated clicking without level-up. Need different approach. "
    layer3 = ctx.build_layer3(situation) if ctx else ""

    n_events = len(narrative.events)
    if n_events > 8 and not any(e.leveled_up for e in narrative.events):
        stuck_advice = ctx.on_stuck(
            n_events, interactive_names, patterns) if ctx else ""
        if stuck_advice:
            layer3 = (layer3 + "\n\n" + stuck_advice) if layer3 else stuck_advice
    budget.record("L3_membot", layer3)

    # Layer 2: Game KB — scene + action model + objects + mechanics
    layer2_scene = ""
    try:
        from sage.irp.plugins.grid_vision_irp import GridVisionIRP, GridObservation
        gv = GridVisionIRP.__new__(GridVisionIRP)
        gv._buffer = []
        gv._frame_count = 0
        gv._prev_frame = None
        g = (tracker.prev_grid
             if hasattr(tracker, 'prev_grid') and tracker.prev_grid is not None
             else None)
        if g is not None:
            objects_list = [
                {"id": o.id, "color": o.color,
                 "bbox": [o.y, o.x, o.y + o.h, o.x + o.w],
                 "centroid": [o.cy, o.cx], "size": o.size}
                for o in tracker.objects.values()]
            obs = GridObservation(
                frame_raw=g, objects=objects_list,
                changes=[], moved=[], step_number=0, level_id="")
            layer2_scene = ("SCENE (what the screen looks like):\n"
                            + gv.describe_scene(obs))
    except Exception:
        pass  # scene description is additive, not required

    layer2_model = action_model.describe()
    layer2_catalog = obj_catalog(tracker)

    # Layer 2.5: Discovered mechanics (highest-value for solving)
    layer2_mechanics = mechanics.to_context() if mechanics else ""
    if layer2_mechanics:
        budget.record("L2.5_mechanics", layer2_mechanics)

    layer2 = ""
    if layer2_mechanics:
        layer2 += layer2_mechanics + "\n\n"
    if layer2_scene:
        layer2 += layer2_scene + "\n\n"
    layer2 += f"ACTION MODEL:\n{layer2_model}\n\nOBJECTS:\n{layer2_catalog}"
    budget.record("L2_game_kb", layer2)

    # Layer 1: Session narrative (expanding with compression, target ~5K)
    layer1 = narrative.to_context()
    budget.record("L1_narrative", layer1)

    if verbose and (n_events % 10 == 0 or n_events <= 3):
        print(f"    {budget.report()}")

    # Fleet knowledge (federated, from all machines)
    if fleet_context:
        budget.record("L3_federation", fleet_context)

    # Assemble prompt
    vision_section = ""
    if vision_insight:
        vision_section = (
            "VISUAL ANALYSIS (from looking at the game screen):\n"
            + vision_insight + "\n")

    prompt = f"""{layer4}

{layer3}

{fleet_context}

{vision_section}GAME STATE: Level {levels}/{win_levels}, Attempt {attempt_num}

{layer2}

{layer1}

{"MOVEMENT available: UP/DOWN/LEFT/RIGHT." if has_move else "Click-only game."}

Based on EVERYTHING above -- your metacognitive principles, cross-game insights,
what you've learned about this game's objects, and your recent action history --
what are the NEXT 3 actions?

Use exact object names: "CLICK cyan_23" (not "CLICK cyan")
Use REPEAT for repetition: "REPEAT 5 CLICK cyan_23"
Use CLICK INTERACTIVE to click all known-working objects.

NEXT 3 ACTIONS:"""

    budget.record("prompt_framing",
                  prompt[len(layer4) + len(layer3) + len(layer2) + len(layer1):])

    # Optionally attach grid image for vision-capable backends
    image_b64 = None
    if grid is not None and backend.supports_vision():
        try:
            from arc_vision import grid_to_image_b64
            image_b64 = grid_to_image_b64(grid)
        except Exception:
            pass

    return backend.query(prompt, image_b64=image_b64)
