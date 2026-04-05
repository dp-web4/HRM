#!/usr/bin/env python3
"""
ARC-AGI-3 Session Narrative Builder

Transforms raw action logs into structured narratives that make
sequential patterns visible to the language model.

This is Layer 1 of the context-shaped intelligence architecture:
working memory that presents the model's own history as a
story it can reason about.

Design: not just "CLICK(4,32): 293px changed" but
"Clicked the left rotation button (3rd time). Region shifted
counterclockwise again. After 3 clicks, the pattern in rows
10-20 is 75% of the way through a full cycle."
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from arc_perception import background_color, color_name, visual_similarity


class ActionEvent:
    """One action and its observed effect."""
    __slots__ = ("step", "action", "target", "data",
                 "changed", "n_pixels", "grid_sim_to_initial",
                 "grid_sim_to_previous", "level_before", "level_after",
                 "observation")

    def __init__(self, step: int, action: int, target: str = "",
                 data: dict = None):
        self.step = step
        self.action = action
        self.target = target  # e.g. "cyan_23" or "UP"
        self.data = data
        self.changed = False
        self.n_pixels = 0
        self.grid_sim_to_initial = 1.0
        self.grid_sim_to_previous = 1.0
        self.level_before = 0
        self.level_after = 0
        self.observation = ""  # human-readable observation

    @property
    def leveled_up(self) -> bool:
        return self.level_after > self.level_before


class SessionNarrative:
    """Builds a structured narrative from game actions.

    The narrative is designed to make patterns visible to the LLM:
    - Groups repeated actions and counts them
    - Tracks similarity to initial state (are we progressing?)
    - Detects cycles (did the grid return to a prior state?)
    - Summarizes what each object DOES, not just that it was clicked
    """

    def __init__(self, initial_grid: np.ndarray):
        self.initial_grid = initial_grid.copy()
        self.events: List[ActionEvent] = []
        self.prev_grid: np.ndarray = initial_grid.copy()

        # Object behavior tracking
        self.object_effects: Dict[str, list] = defaultdict(list)
        # {target_name: [{"changed": bool, "n_pixels": int, "sim_delta": float}]}

        # State tracking for cycle detection
        self.grid_hashes: Dict[str, int] = {}  # hash → step when first seen
        self._hash_grid(initial_grid, 0)

        # Similarity trajectory
        self.similarity_history: List[float] = [1.0]

    def _hash_grid(self, grid, step):
        import hashlib
        h = hashlib.blake2b(grid.tobytes(), digest_size=8).hexdigest()
        if h not in self.grid_hashes:
            self.grid_hashes[h] = step
        return h

    def record(self, step: int, action: int, target: str,
               data: dict, grid_after: np.ndarray,
               level_before: int, level_after: int) -> ActionEvent:
        """Record an action and build the narrative event."""
        event = ActionEvent(step, action, target, data)
        event.changed = not np.array_equal(self.prev_grid, grid_after)
        event.n_pixels = int(np.sum(self.prev_grid != grid_after))
        event.grid_sim_to_initial = visual_similarity(grid_after, self.initial_grid)
        event.grid_sim_to_previous = visual_similarity(grid_after, self.prev_grid)
        event.level_before = level_before
        event.level_after = level_after

        # Track object behavior
        sim_delta = event.grid_sim_to_initial - (self.similarity_history[-1] if self.similarity_history else 1.0)
        self.object_effects[target].append({
            "changed": event.changed,
            "n_pixels": event.n_pixels,
            "sim_delta": sim_delta,
        })

        # Build observation text
        event.observation = self._describe_event(event, sim_delta)

        # Update state
        self.similarity_history.append(event.grid_sim_to_initial)
        self.events.append(event)

        # Cycle detection
        h = self._hash_grid(grid_after, step)
        if h in self.grid_hashes and self.grid_hashes[h] != step:
            prior_step = self.grid_hashes[h]
            cycle_len = step - prior_step
            event.observation += f" ⟳ Grid returned to state from step {prior_step} (cycle length: {cycle_len})"

        self.prev_grid = grid_after.copy()
        return event

    def _describe_event(self, event: ActionEvent, sim_delta: float) -> str:
        """Generate human-readable observation for one action."""
        parts = []

        # What was done
        parts.append(f"{event.target}")

        # What happened
        if event.leveled_up:
            parts.append(f"★ LEVEL UP to {event.level_after}!")
        elif event.changed:
            # Describe the change meaningfully
            if event.n_pixels > 200:
                parts.append(f"large change ({event.n_pixels}px)")
            elif event.n_pixels > 50:
                parts.append(f"moderate change ({event.n_pixels}px)")
            else:
                parts.append(f"small change ({event.n_pixels}px)")

            # Describe trajectory
            if sim_delta < -0.05:
                parts.append("moving AWAY from initial state")
            elif sim_delta > 0.05:
                parts.append("moving TOWARD initial state")
            else:
                parts.append("similarity stable")
        else:
            parts.append("no effect")

        return " → ".join(parts)

    def object_summary(self) -> str:
        """Summarize what each object DOES based on all observations."""
        if not self.object_effects:
            return "No actions taken yet."

        lines = []
        for target, effects in sorted(self.object_effects.items()):
            n = len(effects)
            hits = sum(1 for e in effects if e["changed"])
            if n == 0:
                continue

            rate = hits / n
            avg_px = sum(e["n_pixels"] for e in effects) / n
            avg_sim = sum(e["sim_delta"] for e in effects) / n

            if rate == 0:
                lines.append(f"  {target}: no effect ({n} tries)")
            else:
                direction = ""
                if avg_sim < -0.01:
                    direction = ", moves grid AWAY from initial"
                elif avg_sim > 0.01:
                    direction = ", moves grid TOWARD initial"
                lines.append(f"  {target}: {rate:.0%} effective, ~{avg_px:.0f}px{direction} ({n} tries)")

        return "\n".join(lines) if lines else "No actions taken yet."

    def detect_patterns(self) -> str:
        """Extract sequential patterns from the action history."""
        if len(self.events) < 3:
            return ""

        patterns = []

        # Pattern 1: Cycle detection per object
        for target, effects in self.object_effects.items():
            if len(effects) < 2:
                continue
            changed_effects = [e for e in effects if e["changed"]]
            if not changed_effects:
                continue

            # Are all effects the same magnitude? (consistent behavior)
            pixels = [e["n_pixels"] for e in changed_effects]
            if pixels and max(pixels) > 0:
                if max(pixels) - min(pixels) < max(pixels) * 0.1:
                    patterns.append(f"{target}: consistent effect (~{sum(pixels)//len(pixels)}px each time)")

        # Pattern 2: Similarity trajectory
        if len(self.similarity_history) >= 5:
            recent = self.similarity_history[-5:]
            trend = recent[-1] - recent[0]
            if abs(trend) < 0.02:
                patterns.append(f"Grid similarity STABLE around {recent[-1]:.1%} (not progressing)")
            elif trend < -0.05:
                patterns.append(f"Grid similarity DECREASING ({recent[0]:.1%} → {recent[-1]:.1%}) — changing significantly")
            elif trend > 0.05:
                patterns.append(f"Grid similarity INCREASING ({recent[0]:.1%} → {recent[-1]:.1%}) — returning toward initial")

        # Pattern 3: Same object clicked repeatedly with no level-up
        for target, effects in self.object_effects.items():
            if len(effects) >= 5 and all(e["changed"] for e in effects[-5:]):
                # 5+ effective clicks, no level-up
                if not any(ev.leveled_up for ev in self.events if ev.target == target):
                    patterns.append(f"WARNING: {target} clicked {len(effects)}× with changes but NO level-up — may need different approach")

        return "\n".join(patterns) if patterns else ""

    def to_context(self, max_events: int = 15) -> str:
        """Build the full narrative context for the LLM prompt.

        This is Layer 1 of the context window — working memory.
        """
        sections = []

        # Object behavior summary
        obj_sum = self.object_summary()
        if obj_sum:
            sections.append(f"WHAT I'VE LEARNED ABOUT OBJECTS:\n{obj_sum}")

        # Detected patterns
        patterns = self.detect_patterns()
        if patterns:
            sections.append(f"PATTERNS OBSERVED:\n{patterns}")

        # Similarity trajectory
        if len(self.similarity_history) > 1:
            current = self.similarity_history[-1]
            sections.append(f"GRID STATE: {current:.1%} similar to initial (1.0 = unchanged, 0.0 = completely different)")

        # Recent action narrative (last N events)
        recent = self.events[-max_events:]
        if recent:
            lines = []
            for ev in recent:
                lines.append(f"  Step {ev.step}: {ev.observation}")
            sections.append(f"RECENT ACTIONS:\n" + "\n".join(lines))

        # Level-up events (always included)
        level_ups = [ev for ev in self.events if ev.leveled_up]
        if level_ups:
            lines = [f"  Step {ev.step}: {ev.target} → Level {ev.level_after}" for ev in level_ups]
            sections.append(f"LEVEL-UP EVENTS:\n" + "\n".join(lines))

        return "\n\n".join(sections) if sections else "No actions taken yet."
