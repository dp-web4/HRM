#!/usr/bin/env python3
"""
ARC-AGI-3 World Model + Plan State

Living documents that persist in the context window and get updated
after every action. The LLM both reads and writes these.

WorldModel: What we know about the game
PlanState: What we're doing about it

The feedback loop is explicit:
  1. Model reads world model + plan + LAST RESULT
  2. Model proposes action + expected outcome
  3. We execute and observe
  4. We tell the model EXACTLY what happened vs what it expected
  5. We flag when its hypothesis is failing
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObjectKnowledge:
    """What we know about one object."""
    name: str
    position: tuple  # (cx, cy)
    size: int
    role: str = "unknown"  # button, indicator, canvas, decoration, tool, target
    known_effect: str = ""  # "rotates ring clockwise", "selects color", etc.
    click_count: int = 0
    total_px_changed: int = 0
    caused_level_up: bool = False


class WorldModel:
    """What we know about the game — updated after every action."""

    def __init__(self):
        self.objects: dict[str, ObjectKnowledge] = {}
        self.mechanic_hypothesis: str = ""
        self.goal_hypothesis: str = ""
        self.constraints: list[str] = []
        self.level_rules: list[str] = []
        self.prior_level_rules: list[str] = []
        self.current_level: int = 0
        self.total_actions: int = 0
        self.total_level_ups: int = 0

    def add_object(self, name: str, cx: int, cy: int, size: int,
                   role: str = "unknown", effect: str = ""):
        self.objects[name] = ObjectKnowledge(
            name=name, position=(cx, cy), size=size,
            role=role, known_effect=effect)

    def record_action(self, obj_name: str, px_changed: int, level_up: bool):
        """Record what happened when we interacted with an object."""
        self.total_actions += 1
        if obj_name in self.objects:
            obj = self.objects[obj_name]
            obj.click_count += 1
            obj.total_px_changed += px_changed
            if level_up:
                obj.caused_level_up = True
        if level_up:
            self.total_level_ups += 1

    def update_from_probe(self, tracker, action_model):
        """Populate world model from probe results."""
        for obj in tracker.get_interactive_objects():
            from arc_perception import color_name
            name = f"{color_name(obj.color)}_{obj.id}"
            avg_px = 0
            if hasattr(obj, 'click_responses') and obj.click_responses:
                # click_responses can be list of dicts or list of ints
                vals = []
                for cr in obj.click_responses:
                    if isinstance(cr, dict):
                        vals.append(cr.get('n_pixels', 0))
                    else:
                        vals.append(cr)
                avg_px = sum(vals) / len(vals) if vals else 0
            self.add_object(name, obj.cx, obj.cy, obj.size,
                           role="interactive",
                           effect=f"~{avg_px:.0f}px change per click")

        for obj in tracker.get_static_objects():
            from arc_perception import color_name
            name = f"{color_name(obj.color)}_{obj.id}"
            self.add_object(name, obj.cx, obj.cy, obj.size, role="decoration")

        game_type = action_model.infer_game_type()
        if game_type and game_type != "unknown":
            self.mechanic_hypothesis = game_type

    def on_level_up(self, new_level: int):
        if self.level_rules:
            self.prior_level_rules = self.level_rules.copy()
        self.level_rules = []
        self.current_level = new_level

    def get_overused_objects(self, threshold: int = 4) -> list:
        """Objects clicked many times with only tiny effects."""
        overused = []
        for obj in self.objects.values():
            if obj.click_count >= threshold and not obj.caused_level_up:
                avg = obj.total_px_changed / obj.click_count if obj.click_count else 0
                if avg < 5:  # tiny effect
                    overused.append(obj)
        return overused

    def get_untried_objects(self) -> list:
        """Interactive objects not yet clicked in planning phase."""
        return [obj for obj in self.objects.values()
                if obj.role == "interactive" and obj.click_count == 0]

    def suggest_next_object(self) -> ObjectKnowledge:
        """Suggest the most promising object to try next.

        Priority: untried > least-clicked > largest-effect.
        Returns None if no interactive objects exist.
        """
        interactive = [o for o in self.objects.values() if o.role == "interactive"]
        if not interactive:
            return None

        # First: any untried objects
        untried = [o for o in interactive if o.click_count == 0]
        if untried:
            return untried[0]

        # Second: least-clicked (explore underexplored)
        by_clicks = sorted(interactive, key=lambda o: o.click_count)
        least_clicked = by_clicks[0]

        # Third: if all roughly equal clicks, pick largest avg effect
        by_effect = sorted(interactive,
                          key=lambda o: o.total_px_changed / max(o.click_count, 1),
                          reverse=True)
        best_effect = by_effect[0]

        # Prefer least-clicked unless best_effect is dramatically better
        best_avg = best_effect.total_px_changed / max(best_effect.click_count, 1)
        least_avg = least_clicked.total_px_changed / max(least_clicked.click_count, 1)
        if best_avg > least_avg * 3 and best_avg > 5:
            return best_effect
        return least_clicked

    def to_context(self) -> str:
        parts = ["=== WORLD MODEL (what we know) ==="]

        # Discovered mechanics (from observation-based discovery protocol)
        if hasattr(self, 'mechanics_context') and self.mechanics_context:
            parts.append(self.mechanics_context)
            parts.append("")

        if self.mechanic_hypothesis:
            parts.append(f"GAME TYPE: {self.mechanic_hypothesis}")

        if self.goal_hypothesis:
            parts.append(f"GOAL: {self.goal_hypothesis}")
        else:
            parts.append("GOAL: (unknown — form a hypothesis)")

        parts.append(f"PROGRESS: {self.total_level_ups} levels solved in {self.total_actions} actions")

        # Objects with usage stats
        interactive = [o for o in self.objects.values() if o.role == "interactive"]
        if interactive:
            parts.append("INTERACTIVE OBJECTS:")
            for obj in interactive:
                avg_px = obj.total_px_changed / obj.click_count if obj.click_count else 0
                if obj.click_count == 0:
                    status = "NOT YET TRIED"
                elif obj.caused_level_up:
                    status = "CAUSED LEVEL UP"
                elif avg_px < 3:
                    status = f"clicked {obj.click_count}x, ~{avg_px:.0f}px avg (TINY effect)"
                else:
                    status = f"clicked {obj.click_count}x, ~{avg_px:.0f}px avg"
                parts.append(f"  {obj.name} at ({obj.position[0]},{obj.position[1]}) — {status}")

        # Positive direction: suggest what to try next (not what to stop)
        suggestion = self.suggest_next_object()
        if suggestion:
            parts.append(f"\nRECOMMENDED NEXT ACTION: Click {suggestion.name} at ({suggestion.position[0]},{suggestion.position[1]})")
            if suggestion.click_count == 0:
                parts.append(f"  Reason: You haven't tried {suggestion.name} yet — it might be the key to this level.")
            else:
                parts.append(f"  Reason: {suggestion.name} has the most promising effect pattern so far.")

        decoration_count = sum(1 for o in self.objects.values() if o.role == "decoration")
        if decoration_count:
            parts.append(f"DECORATION: {decoration_count} objects (ignore)")

        if self.constraints:
            parts.append("CONSTRAINTS:")
            for c in self.constraints[-5:]:
                parts.append(f"  - {c}")

        if self.prior_level_rules:
            parts.append("PRIOR LEVEL RULES (test first):")
            for r in self.prior_level_rules[-3:]:
                parts.append(f"  - {r}")

        if self.level_rules:
            parts.append("CURRENT LEVEL RULES:")
            for r in self.level_rules[-3:]:
                parts.append(f"  - {r}")

        return "\n".join(parts)


class PlanState:
    """What we're doing — updated after every observation."""

    def __init__(self):
        self.hypothesis: str = ""
        self.subgoal: str = ""
        self.next_action: str = ""
        self.expected_outcome: str = ""
        self.confidence: str = "low"
        self.actions_on_current_plan: int = 0
        self.max_actions_before_replan: int = 5
        self.history: list[dict] = []
        self.consecutive_failures: int = 0
        self.last_action_obj: str = ""
        self.same_obj_streak: int = 0

    def record_outcome(self, action: str, expected: str, actual: str,
                       px_changed: int, level_up: bool):
        """Record what happened vs what was expected."""
        # Determine if expectation was met meaningfully
        # (not just "something changed" but "the predicted thing happened")
        matched = level_up or px_changed > 10  # Only large changes count as "matched"

        self.history.append({
            "action": action,
            "expected": expected,
            "actual": actual,
            "px_changed": px_changed,
            "level_up": level_up,
            "matched": matched,
        })
        if len(self.history) > 10:
            self.history = self.history[-10:]

        self.actions_on_current_plan += 1

        if not matched:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        # Track same-object streaks
        if action == self.last_action_obj:
            self.same_obj_streak += 1
        else:
            self.same_obj_streak = 0
            self.last_action_obj = action

    def needs_replan(self) -> bool:
        if not self.hypothesis:
            return True
        if self.actions_on_current_plan >= self.max_actions_before_replan:
            return True
        if self.consecutive_failures >= 3:
            return True
        if self.same_obj_streak >= 3:
            return True
        return False

    def to_context(self) -> str:
        parts = ["=== CURRENT PLAN ==="]

        if self.hypothesis:
            parts.append(f"HYPOTHESIS: {self.hypothesis}")
        else:
            parts.append("HYPOTHESIS: (none yet — form one from the world model)")

        if self.subgoal:
            parts.append(f"SUBGOAL: {self.subgoal}")

        parts.append(f"CONFIDENCE: {self.confidence}")

        # Last result — EXPLICIT FEEDBACK
        if self.history:
            last = self.history[-1]
            parts.append("")
            parts.append("LAST ACTION RESULT:")
            parts.append(f"  You did: {last['action']}")
            parts.append(f"  You expected: {last['expected']}")
            parts.append(f"  What happened: {last['actual']}")
            parts.append(f"  Pixels changed: {last['px_changed']}")
            if last['level_up']:
                parts.append(f"  ★ LEVEL UP! Your action solved the level!")
            elif last['matched']:
                parts.append(f"  Result: SIGNIFICANT change — your action did something meaningful.")
            else:
                parts.append(f"  Result: MINIMAL change ({last['px_changed']}px). Your prediction was WRONG.")
                parts.append(f"  → Your expectation did NOT match reality. Reconsider your hypothesis.")

        # Streak warnings
        if self.consecutive_failures >= 3:
            parts.append(f"\n⚠ {self.consecutive_failures} consecutive actions with minimal effect.")
            parts.append(f"  Your current approach is NOT WORKING. Change your hypothesis.")
            parts.append(f"  Try a DIFFERENT object or a DIFFERENT strategy.")

        if self.same_obj_streak >= 2:
            parts.append(f"\n⚠ You clicked {self.last_action_obj} {self.same_obj_streak + 1} times in a row.")
            parts.append(f"  Repeating the same action won't give different results. Try something else.")

        # Recent history summary
        if len(self.history) > 1:
            parts.append("")
            parts.append("RECENT HISTORY (last 5):")
            for h in self.history[-5:]:
                px = h['px_changed']
                indicator = "★" if h['level_up'] else ("✓" if h['matched'] else "✗")
                parts.append(f"  {indicator} {h['action']}: {px}px changed")

        if self.needs_replan():
            parts.append("\n→ REPLAN NEEDED. Your approach isn't working. Form a NEW hypothesis.")

        return "\n".join(parts)


# ─── LLM Interaction Prompts ───

PLAN_UPDATE_PROMPT = """You are solving a puzzle game. Read everything below carefully.

{fleet_context}

{scene_description}

{world_model}

{plan_state}

INSTRUCTIONS:
- If your last prediction was WRONG, your hypothesis is probably wrong too. CHANGE IT.
- If you clicked the same object repeatedly with tiny effects, STOP. Try a DIFFERENT one.
- Tiny effects (1-2px) mean state toggle. Large effects (50+px) mean you moved something big. Focus on big effects.
- Look at the SCENE — what spatial structure do you see? Where are the large regions?
- Look at FLEET KNOWLEDGE above — other machines may know what works on this game.
- Think: what would "solved" look like? What needs to change in the grid?

Respond in this EXACT format (one line each, no extra text):
HYPOTHESIS: [your theory — MUST be different from before if your prediction was wrong]
GOAL: [what winning looks like]
SUBGOAL: [specific next thing to test]
ACTION: [exactly ONE: CLICK object_name / UP / DOWN / LEFT / RIGHT / SELECT]
EXPECT: [specific prediction like "green region shifts 50px left" NOT "state changes"]
CONFIDENCE: [low/medium/high]
RULE: [any rule discovered, or "none"]"""


def parse_plan_update(response: str, world_model: WorldModel, plan: PlanState):
    """Parse LLM response and update world model + plan state."""
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        upper = line.upper()

        if upper.startswith("HYPOTHESIS:"):
            plan.hypothesis = line.split(":", 1)[1].strip()
        elif upper.startswith("GOAL:"):
            world_model.goal_hypothesis = line.split(":", 1)[1].strip()
        elif upper.startswith("SUBGOAL:"):
            plan.subgoal = line.split(":", 1)[1].strip()
        elif upper.startswith("ACTION:"):
            plan.next_action = line.split(":", 1)[1].strip()
        elif upper.startswith("EXPECT:"):
            plan.expected_outcome = line.split(":", 1)[1].strip()
        elif upper.startswith("CONFIDENCE:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("low", "medium", "high"):
                plan.confidence = val
        elif upper.startswith("RULE:"):
            rule = line.split(":", 1)[1].strip()
            if rule.lower() not in ("none", "n/a", ""):
                if rule not in world_model.level_rules:
                    world_model.level_rules.append(rule)

    plan.actions_on_current_plan = 0
