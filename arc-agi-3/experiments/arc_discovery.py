#!/usr/bin/env python3
"""
Mechanic Discovery Protocol — extract game abstractions from observation.

Replaces the flat probe phase with structured discovery:
  Phase 0: Calibrate  (1-2 actions)  — verify clicks hit targets
  Phase 1: Classify   (5-10 actions) — button vs piece vs indicator vs static
  Phase 2: Correlate  (5-10 actions) — click A → which objects change?
  Phase 3: Structure  (0 actions)    — grids, rings, rows, groups
  Phase 4: Constraints (0 actions)   — decode indicator sprites as rules

Output: GameMechanics dataclass with LLM-readable to_context() method.
Drop-in replacement for probe() in sage_solver_v7.py.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from arc_perception import (
    find_color_regions, background_color, color_name, get_frame
)
from arc_spatial import SpatialTracker, SpatialObject
from arc_action_model import ActionEffectModel
from arcengine import GameAction


# ─── Data Structures ───

@dataclass
class ObjectRole:
    name: str
    obj_id: int
    classification: str  # button, piece, indicator, target, decoration, unknown
    position: Tuple[int, int]
    size: int
    color: int
    evidence: str

@dataclass
class EffectCorrelation:
    trigger: str
    affected: List[str]
    effect_type: str  # toggle, cycle, shift, remote, self-modify
    position_sensitive: bool = False
    avg_pixel_change: float = 0.0

@dataclass
class ObjectGroup:
    group_id: int
    members: List[str]
    structure: str  # ring, grid, row, column, pair, cluster
    regularity: float = 0.0
    shared_color: Optional[int] = None
    center: Optional[Tuple[int, int]] = None

@dataclass
class ConstraintPattern:
    indicator_name: str
    position: Tuple[int, int]
    center_color: int
    pattern_matrix: List[List[int]]
    interpretation: str
    nearby_pieces: List[str] = field(default_factory=list)

@dataclass
class GameMechanics:
    game_type: str = "unknown"
    calibrated: bool = False
    available_actions: List[int] = field(default_factory=list)
    objects: Dict[str, ObjectRole] = field(default_factory=dict)
    correlations: List[EffectCorrelation] = field(default_factory=list)
    groups: List[ObjectGroup] = field(default_factory=list)
    constraints: List[ConstraintPattern] = field(default_factory=list)
    interaction_model: str = "unknown"
    discovery_budget_used: int = 0
    confidence: str = "low"
    discovery_log: List[str] = field(default_factory=list)

    def to_context(self) -> str:
        """Render as LLM-readable text for context window."""
        ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                        5: "SELECT", 6: "CLICK", 7: "UNDO"}

        lines = ["GAME MECHANICS DISCOVERED:"]

        # Toolset — what verbs the game gives us
        verbs = [ACTION_NAMES.get(a, f"ACTION{a}") for a in self.available_actions]
        lines.append(f"  TOOLSET: {', '.join(verbs)}")
        has_move = any(a in self.available_actions for a in [1, 2, 3, 4])
        has_click = 6 in self.available_actions
        has_select = 5 in self.available_actions
        has_undo = 7 in self.available_actions
        if has_click and not has_move:
            lines.append("    → Click-only: no movement. This is a PUZZLE (toggle, rotate, place).")
        elif has_move and not has_click:
            lines.append("    → Move-only: no clicking. This is NAVIGATION (maze, sokoban).")
        elif has_move and has_click:
            lines.append("    → Move+Click: navigate AND interact. May have a cursor/character.")
        if has_select:
            lines.append("    → SELECT available: game has a selection/confirmation mechanic.")
        if has_undo:
            lines.append("    → UNDO available: can reverse mistakes. Use it if stuck.")

        lines.append(f"  Type: {self.game_type}")
        lines.append(f"  Interaction: {self.interaction_model}")
        if not self.calibrated:
            lines.append("  ⚠ CALIBRATION FAILED — clicks may not hit intended targets!")
        lines.append(f"  Confidence: {self.confidence}")
        lines.append(f"  Discovery cost: {self.discovery_budget_used} actions")

        # Objects by role
        by_role = defaultdict(list)
        for obj in self.objects.values():
            by_role[obj.classification].append(obj)

        for role in ["button", "piece", "indicator", "target", "decoration", "unknown"]:
            objs = by_role.get(role, [])
            if not objs:
                continue
            lines.append(f"\n  {role.upper()}S ({len(objs)}):")
            for obj in objs[:10]:
                lines.append(f"    {obj.name} at ({obj.position[0]},{obj.position[1]}) "
                             f"size={obj.size} — {obj.evidence}")

        # Effect correlations
        if self.correlations:
            lines.append(f"\n  EFFECT MAP:")
            for corr in self.correlations:
                affected_str = ", ".join(corr.affected[:5])
                extra = ""
                if corr.position_sensitive:
                    extra = " (click position within object matters!)"
                lines.append(f"    Click {corr.trigger} → {corr.effect_type}: "
                             f"{affected_str}{extra}")

        # Structural groups
        if self.groups:
            lines.append(f"\n  STRUCTURE:")
            for g in self.groups:
                members_str = ", ".join(g.members[:6])
                if len(g.members) > 6:
                    members_str += f" +{len(g.members)-6} more"
                lines.append(f"    {g.structure} (group {g.group_id}): {members_str}")

        # Constraints
        if self.constraints:
            lines.append(f"\n  CONSTRAINT PATTERNS:")
            for c in self.constraints:
                lines.append(f"    {c.indicator_name}: {c.interpretation}")

        return "\n".join(lines)


# ─── Helpers ───

def _obj_name(color: int, obj_id: int) -> str:
    return f"{color_name(color)}_{obj_id}"


def _regions_overlap(r1_bbox, r2_bbox, tolerance=4) -> bool:
    """Check if two bounding boxes overlap within tolerance."""
    x1, y1, x2, y2 = r1_bbox
    x3, y3, x4, y4 = r2_bbox
    return not (x2 + tolerance < x3 or x4 + tolerance < x1 or
                y2 + tolerance < y3 or y4 + tolerance < y1)


def _size_bucket(size: int) -> str:
    if size < 20:
        return "small"
    elif size < 100:
        return "medium"
    return "large"


def _find_changed_objects(tracker: SpatialTracker,
                          before_positions: Dict[int, Tuple[int, int, int]],
                          grid_after: np.ndarray) -> List[Tuple[SpatialObject, str]]:
    """Compare tracked objects before/after to find which changed.
    Returns [(obj, change_type)] where change_type is 'moved' or 'color_changed'.
    """
    changed = []
    for obj in tracker.objects.values():
        before = before_positions.get(obj.id)
        if before is None:
            continue
        old_cx, old_cy, old_color = before
        if obj.cx != old_cx or obj.cy != old_cy:
            changed.append((obj, "moved"))
        elif obj.color != old_color:
            changed.append((obj, "color_changed"))
        else:
            # Check if pixels at object location changed
            x, y, x2, y2 = obj.bbox
            x, y = max(0, x), max(0, y)
            x2 = min(grid_after.shape[1], x2)
            y2 = min(grid_after.shape[0], y2)
            # We can't easily check pixel changes without before grid,
            # so skip this case
            pass
    return changed


# ─── Discovery Engine ───

class MechanicDiscovery:
    """Orchestrates the 5-phase mechanic discovery protocol."""

    INT_TO_GA = {a.value: a for a in GameAction}

    def __init__(self, env, fd, grid: np.ndarray,
                 available_actions: List[int], total_budget: int = 300,
                 verbose: bool = False):
        self.env = env
        self.fd = fd
        self.grid = grid
        self.available = available_actions
        self.total_budget = total_budget
        self.verbose = verbose

        self.tracker = SpatialTracker()
        self.action_model = ActionEffectModel()
        self.mechanics = GameMechanics()

        self.discovery_budget = min(int(total_budget * 0.15), 45)
        self.steps_used = 0
        self.level_ups = 0
        self.bg = background_color(grid)
        self.mechanics.available_actions = list(available_actions)

    def _log(self, msg: str):
        self.mechanics.discovery_log.append(msg)
        if self.verbose:
            print(f"  [discovery] {msg}")

    def _step(self, action: int, x: int = 0, y: int = 0):
        """Execute one game action, update tracking, return new grid."""
        ga = self.INT_TO_GA.get(action)
        if ga is None:
            return self.grid

        grid_before = self.grid.copy()
        level_before = self.fd.levels_completed

        if action == 6 and (x > 0 or y > 0):
            self.fd = self.env.step(ga, data={"x": x, "y": y})
        else:
            self.fd = self.env.step(ga)

        self.grid = get_frame(self.fd)
        self.steps_used += 1

        # Update models
        spatial_diff = self.tracker.update(self.grid)
        self.action_model.observe(action, grid_before, self.grid, spatial_diff)

        if self.fd.levels_completed > level_before:
            self.level_ups += 1
            self._log(f"Level up during discovery! Now at {self.fd.levels_completed}")

        return self.grid

    def _click(self, x: int, y: int) -> Tuple[np.ndarray, int]:
        """Click at (x, y). Returns (new_grid, n_pixels_changed)."""
        grid_before = self.grid.copy()
        self._step(6, x, y)
        n_changed = int((grid_before != self.grid).sum())
        return self.grid, n_changed

    def discover(self):
        """Run all discovery phases. Returns probe-compatible tuple + mechanics."""
        # Initialize tracker with first frame
        self.tracker.update(self.grid)
        regions = find_color_regions(self.grid, min_size=3)
        self._log(f"Initial scan: {len(regions)} regions, "
                  f"{len(self.tracker.objects)} tracked objects")

        has_click = 6 in self.available
        has_move = any(a in self.available for a in [1, 2, 3, 4])

        self._log(f"Toolset: {self.available} "
                  f"({'click' if has_click else ''}"
                  f"{'+' if has_click and has_move else ''}"
                  f"{'move' if has_move else ''})")

        # Phase 0: Calibrate (only if we can click)
        if has_click:
            self._phase_calibrate(regions)

        # Phase 1: Classify (test each interaction verb)
        if self.steps_used < self.discovery_budget:
            self._phase_classify(regions)

        # Phase 2: Correlate (only for click games with buttons)
        if has_click and self.steps_used < self.discovery_budget - 2:
            self._phase_correlate()

        # Phase 3: Structure (free — no clicks)
        self._phase_structure()

        # Phase 4: Constraints (free — no clicks)
        self._phase_constraints()

        # Synthesize
        self._synthesize()

        self.mechanics.discovery_budget_used = self.steps_used
        self._log(f"Discovery complete: {self.mechanics.game_type}, "
                  f"confidence={self.mechanics.confidence}, "
                  f"cost={self.steps_used} actions")

        return (self.action_model, self.tracker, self.fd, self.grid,
                self.steps_used, self.level_ups, self.mechanics)

    # ─── Phase 0: Calibrate ───

    def _phase_calibrate(self, regions):
        """Verify coordinate mapping by clicking the largest visible object."""
        if not regions or 6 not in self.available:
            self.mechanics.calibrated = True  # can't test, assume ok
            return

        # Pick a mid-sized region (not background-huge, not indicator-tiny)
        # Sort by size, pick the median-sized one
        sorted_regions = sorted(regions, key=lambda r: r["size"])
        # Prefer medium-sized objects — they're most likely interactive
        mid_idx = len(sorted_regions) // 2
        target = sorted_regions[min(mid_idx, len(sorted_regions) - 1)]
        cx, cy = target["cx"], target["cy"]
        target_bbox = (target["x"], target["y"],
                       target["x"] + target["w"], target["y"] + target["h"])

        self._log(f"Phase 0: calibrating at ({cx},{cy}) on "
                  f"{color_name(target['color'])} ({target['size']}px)")

        grid_before = self.grid.copy()
        self._click(cx, cy)
        diff_mask = grid_before != self.grid
        n_changed = int(diff_mask.sum())

        if n_changed == 0:
            self._log("Phase 0: NO pixels changed at all — game may not respond to clicks")
            self.mechanics.calibrated = False
            return

        # Check if changes happened NEAR the click point
        changed_coords = np.argwhere(diff_mask)
        if len(changed_coords) == 0:
            self.mechanics.calibrated = False
            return

        changed_bbox = (int(changed_coords[:, 1].min()),
                        int(changed_coords[:, 0].min()),
                        int(changed_coords[:, 1].max()),
                        int(changed_coords[:, 0].max()))

        if _regions_overlap(target_bbox, changed_bbox, tolerance=8):
            self.mechanics.calibrated = True
            self._log(f"Phase 0: CALIBRATED — {n_changed}px changed near click point")
        else:
            # Changes happened far from click — only counter?
            if n_changed <= 4:
                self._log(f"Phase 0: only {n_changed}px changed FAR from click — "
                          f"likely just a counter. Coordinates may be wrong.")
                self.mechanics.calibrated = False
            else:
                self._log(f"Phase 0: {n_changed}px changed but at "
                          f"({changed_bbox[0]},{changed_bbox[1]}), not near "
                          f"({cx},{cy}). Remote effect or wrong coords.")
                self.mechanics.calibrated = True  # remote effect is still valid

    # ─── Phase 1: Classify ───

    def _phase_classify(self, regions):
        """Click one per object type, classify by response."""
        ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                        5: "SELECT", 6: "CLICK"}
        has_move = any(a in self.available for a in [1, 2, 3, 4])
        has_click = 6 in self.available

        # Test movement first (if available) — reveals cursor, navigation type
        if has_move:
            for a in [1, 2, 3, 4]:
                if a in self.available and self.steps_used < self.discovery_budget:
                    grid_before = self.grid.copy()
                    self._step(a)
                    n_changed = int((grid_before != self.grid).sum())
                    name = ACTION_NAMES.get(a, f"A{a}")
                    if n_changed > 5:
                        self._log(f"Phase 1: {name} moves {n_changed}px — game has navigation")
                    else:
                        self._log(f"Phase 1: {name} no visible effect")

        if not has_click:
            self.mechanics.interaction_model = "move-only"
            return

        self.mechanics.interaction_model = "move+click" if has_move else "click"

        # Group regions by (color, size_bucket)
        type_groups = defaultdict(list)
        for r in regions:
            key = (r["color"], _size_bucket(r["size"]))
            type_groups[key].append(r)

        self._log(f"Phase 1: {len(type_groups)} distinct object types from "
                  f"{len(regions)} regions")

        # Click one representative per type
        for (color, bucket), group in sorted(type_groups.items(),
                                              key=lambda x: x[0][1]):
            if self.steps_used >= self.discovery_budget:
                break

            # Pick the representative closest to center of grid
            rep = min(group, key=lambda r: abs(r["cx"] - 32) + abs(r["cy"] - 32))
            cx, cy = rep["cx"], rep["cy"]
            name = _obj_name(color, rep.get("id", id(rep) % 1000))

            # Snapshot object positions before click
            before_positions = {
                obj.id: (obj.cx, obj.cy, obj.color)
                for obj in self.tracker.objects.values()
            }

            grid_before = self.grid.copy()
            self._click(cx, cy)
            n_changed = int((grid_before != self.grid).sum())

            # Find the tracked object at the click point
            clicked_obj = None
            for obj in self.tracker.objects.values():
                if (obj.x <= cx <= obj.x + obj.w and
                        obj.y <= cy <= obj.y + obj.h):
                    clicked_obj = obj
                    break

            # Classify
            if n_changed <= 2:
                classification = "decoration"
                evidence = "no response to click"
            else:
                # Check if change was local (on clicked object) or remote
                diff_coords = np.argwhere(grid_before != self.grid)
                change_bbox = (int(diff_coords[:, 1].min()),
                               int(diff_coords[:, 0].min()),
                               int(diff_coords[:, 1].max()),
                               int(diff_coords[:, 0].max()))
                click_bbox = (rep["x"], rep["y"],
                              rep["x"] + rep["w"], rep["y"] + rep["h"])

                local_change = _regions_overlap(click_bbox, change_bbox, tolerance=4)

                if bucket == "small" and not local_change:
                    classification = "button"
                    evidence = f"small ({rep['size']}px), click caused remote change ({n_changed}px)"
                elif local_change and n_changed < 50:
                    classification = "piece"
                    evidence = f"click changed self ({n_changed}px)"
                elif not local_change:
                    classification = "button"
                    evidence = f"click caused change elsewhere ({n_changed}px)"
                else:
                    classification = "piece"
                    evidence = f"click changed {n_changed}px including self"

            # Check for multi-colored indicator pattern (no click needed)
            if classification == "decoration" and rep["w"] <= 8 and rep["h"] <= 8:
                x0, y0 = rep["x"], rep["y"]
                x1, y1 = x0 + rep["w"], y0 + rep["h"]
                sub = self.grid[y0:y1, x0:x1]
                unique_colors = set(int(v) for v in np.unique(sub) if v != self.bg)
                if len(unique_colors) > 1:
                    classification = "indicator"
                    evidence = f"multi-colored ({len(unique_colors)} colors), no click response"

            # Record
            obj_name = _obj_name(color, clicked_obj.id if clicked_obj else 0)
            role = ObjectRole(
                name=obj_name,
                obj_id=clicked_obj.id if clicked_obj else 0,
                classification=classification,
                position=(cx, cy),
                size=rep["size"],
                color=color,
                evidence=evidence,
            )
            self.mechanics.objects[obj_name] = role

            # Also classify all other objects of same type
            for other in group:
                if other is not rep:
                    other_obj = None
                    for obj in self.tracker.objects.values():
                        if (obj.x <= other["cx"] <= obj.x + obj.w and
                                obj.y <= other["cy"] <= obj.y + other["h"]):
                            other_obj = obj
                            break
                    other_name = _obj_name(color,
                                           other_obj.id if other_obj else id(other) % 1000)
                    self.mechanics.objects[other_name] = ObjectRole(
                        name=other_name,
                        obj_id=other_obj.id if other_obj else 0,
                        classification=classification,
                        position=(other["cx"], other["cy"]),
                        size=other["size"],
                        color=color,
                        evidence=f"same type as {obj_name}",
                    )

            self._log(f"Phase 1: {obj_name} = {classification} ({evidence})")
            self.tracker.record_click(cx, cy, n_changed > 2, n_changed)

    # ─── Phase 2: Correlate ───

    def _phase_correlate(self):
        """For each button, track which OTHER objects change when clicked."""
        buttons = [obj for obj in self.mechanics.objects.values()
                   if obj.classification == "button"]

        if not buttons:
            self._log("Phase 2: no buttons found, skipping correlation")
            return

        # Cap at 5 buttons, prioritize smallest
        buttons = sorted(buttons, key=lambda o: o.size)[:5]

        for btn in buttons:
            if self.steps_used >= self.discovery_budget:
                break

            # Snapshot all tracked objects
            before_positions = {
                obj.id: (obj.cx, obj.cy, obj.color)
                for obj in self.tracker.objects.values()
            }
            grid_before = self.grid.copy()

            cx, cy = btn.position
            self._click(cx, cy)
            n_changed = int((grid_before != self.grid).sum())

            if n_changed <= 2:
                self._log(f"Phase 2: {btn.name} no effect on re-click")
                continue

            # Find which tracked objects changed
            affected_names = []
            effect_type = "unknown"
            for obj in self.tracker.objects.values():
                before = before_positions.get(obj.id)
                if before is None:
                    continue
                old_cx, old_cy, old_color = before
                if obj.cx != old_cx or obj.cy != old_cy:
                    affected_names.append(_obj_name(obj.color, obj.id))
                    effect_type = "shift"

            # If no tracked objects moved, check for color changes in grid
            if not affected_names:
                diff = grid_before != self.grid
                if diff.sum() > 10:
                    effect_type = "toggle" if n_changed < 50 else "cycle"
                    # Find regions where changes occurred
                    changed_coords = np.argwhere(diff)
                    for obj in self.tracker.objects.values():
                        if obj.id == btn.obj_id:
                            continue
                        obj_region = (changed_coords[:, 1] >= obj.x) & \
                                     (changed_coords[:, 1] < obj.x + obj.w) & \
                                     (changed_coords[:, 0] >= obj.y) & \
                                     (changed_coords[:, 0] < obj.y + obj.h)
                        if obj_region.sum() > 0:
                            affected_names.append(_obj_name(obj.color, obj.id))

            if affected_names:
                corr = EffectCorrelation(
                    trigger=btn.name,
                    affected=affected_names,
                    effect_type=effect_type,
                    avg_pixel_change=float(n_changed),
                )
                self.mechanics.correlations.append(corr)
                self._log(f"Phase 2: {btn.name} → {effect_type}: "
                          f"{', '.join(affected_names[:3])}")

    # ─── Phase 3: Structure ───

    def _phase_structure(self):
        """Pure geometric analysis — no clicks needed."""
        objects = list(self.tracker.objects.values())
        if len(objects) < 3:
            return

        # Group by color
        by_color = defaultdict(list)
        for obj in objects:
            by_color[obj.color].append(obj)

        group_id = 0
        for color, color_objs in by_color.items():
            if len(color_objs) < 3:
                continue

            # Check for grid structure
            xs = sorted(set(obj.cx for obj in color_objs))
            ys = sorted(set(obj.cy for obj in color_objs))

            if len(xs) >= 2 and len(ys) >= 2:
                # Check regularity of spacing
                x_diffs = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
                y_diffs = [ys[i+1] - ys[i] for i in range(len(ys)-1)]

                x_regular = (max(x_diffs) - min(x_diffs) <= 3) if x_diffs else False
                y_regular = (max(y_diffs) - min(y_diffs) <= 3) if y_diffs else False

                if x_regular and y_regular and len(xs) >= 2 and len(ys) >= 2:
                    members = [_obj_name(obj.color, obj.id) for obj in color_objs]
                    reg = 1.0 - (max(x_diffs) - min(x_diffs)) / max(max(x_diffs), 1)
                    self.mechanics.groups.append(ObjectGroup(
                        group_id=group_id,
                        members=members,
                        structure="grid",
                        regularity=reg,
                        shared_color=color,
                        center=(sum(xs) // len(xs), sum(ys) // len(ys)),
                    ))
                    group_id += 1
                    self._log(f"Phase 3: grid of {len(members)} "
                              f"{color_name(color)} objects ({len(xs)}x{len(ys)})")
                    continue

            # Check for row/column
            for obj_set, axis in [(xs, "column"), (ys, "row")]:
                if len(obj_set) <= 1:
                    continue
                # All objects have same x (column) or same y (row)?
                if axis == "row":
                    # Group by cy
                    for target_y in ys:
                        row_objs = [o for o in color_objs if abs(o.cy - target_y) <= 2]
                        if len(row_objs) >= 3:
                            members = [_obj_name(o.color, o.id) for o in row_objs]
                            self.mechanics.groups.append(ObjectGroup(
                                group_id=group_id,
                                members=members,
                                structure="row",
                                regularity=0.8,
                                shared_color=color,
                            ))
                            group_id += 1

        # Check for ring structure via effect correlations
        if self.mechanics.correlations:
            # Build effect graph: trigger → affected
            effect_graph = {}
            for corr in self.mechanics.correlations:
                effect_graph[corr.trigger] = set(corr.affected)

            # Look for cycles
            for start in effect_graph:
                visited = [start]
                current = start
                for _ in range(20):  # max ring size
                    neighbors = effect_graph.get(current, set())
                    # Find next unvisited
                    next_obj = None
                    for n in neighbors:
                        if n == start and len(visited) >= 3:
                            # Cycle found!
                            self.mechanics.groups.append(ObjectGroup(
                                group_id=group_id,
                                members=visited,
                                structure="ring",
                                regularity=1.0,
                            ))
                            group_id += 1
                            self._log(f"Phase 3: ring of {len(visited)} objects")
                            break
                        if n not in visited and n in effect_graph:
                            next_obj = n
                            break
                    if next_obj:
                        visited.append(next_obj)
                        current = next_obj
                    else:
                        break

        self._log(f"Phase 3: found {len(self.mechanics.groups)} structural groups")

    # ─── Phase 4: Constraints ───

    def _phase_constraints(self):
        """Find multi-colored indicator sprites and decode them as constraints."""
        regions = find_color_regions(self.grid, min_size=2)

        for region in regions:
            # Only small regions (likely indicators, not large pieces)
            if region["size"] > 30 or region["w"] > 10 or region["h"] > 10:
                continue

            x0, y0 = region["x"], region["y"]
            w, h = region["w"], region["h"]
            sub = self.grid[y0:y0+h, x0:x0+w]

            # Must have multiple non-background colors
            unique = set(int(v) for v in np.unique(sub))
            unique.discard(self.bg)
            unique.discard(-1)
            if len(unique) < 2:
                continue

            # This is a multi-colored indicator!
            # Try to decode as constraint pattern
            # Assume pixels are doubled (each "game pixel" = 2x2 display pixels)
            pattern = []
            step = 2 if w >= 6 and h >= 6 else 1
            rows_decoded = 0
            for pr in range(0, h, step):
                row = []
                for pc in range(0, w, step):
                    if pr < h and pc < w:
                        row.append(int(sub[pr, pc]))
                pattern.append(row)
                rows_decoded += 1
                if rows_decoded >= 5:
                    break

            if len(pattern) < 3 or len(pattern[0]) < 3:
                continue

            # Center pixel = target color
            cr, cc = len(pattern) // 2, len(pattern[0]) // 2
            center_color = pattern[cr][cc]
            if center_color == 0 or center_color == self.bg:
                continue

            # Interpret: black(0) = must match center, non-black = must NOT match
            interpretations = []
            for pr, row in enumerate(pattern):
                for pc, val in enumerate(row):
                    if pr == cr and pc == cc:
                        continue  # skip center
                    dr, dc = pr - cr, pc - cc
                    if val == 0:
                        interpretations.append(
                            f"neighbor at ({dc},{dr}) must be {color_name(center_color)}")
                    elif val != self.bg:
                        interpretations.append(
                            f"neighbor at ({dc},{dr}) must NOT be {color_name(center_color)}")

            if not interpretations:
                continue

            # Find nearby piece objects
            nearby = []
            for obj_name, obj_role in self.mechanics.objects.items():
                if obj_role.classification == "piece":
                    dist = abs(obj_role.position[0] - region["cx"]) + \
                           abs(obj_role.position[1] - region["cy"])
                    if dist < 30:
                        nearby.append(obj_name)

            constraint = ConstraintPattern(
                indicator_name=_obj_name(region["color"], id(region) % 1000),
                position=(region["cx"], region["cy"]),
                center_color=center_color,
                pattern_matrix=pattern,
                interpretation="; ".join(interpretations[:8]),
                nearby_pieces=nearby,
            )
            self.mechanics.constraints.append(constraint)

            # Also register as indicator in objects
            ind_name = constraint.indicator_name
            if ind_name not in self.mechanics.objects:
                self.mechanics.objects[ind_name] = ObjectRole(
                    name=ind_name,
                    obj_id=0,
                    classification="indicator",
                    position=constraint.position,
                    size=region["size"],
                    color=region["color"],
                    evidence=f"constraint pattern: {constraint.interpretation[:60]}",
                )

        self._log(f"Phase 4: found {len(self.mechanics.constraints)} constraint patterns")

    # ─── Synthesis ───

    def _synthesize(self):
        """Combine all findings into a coherent game type assessment."""
        n_buttons = sum(1 for o in self.mechanics.objects.values()
                        if o.classification == "button")
        n_pieces = sum(1 for o in self.mechanics.objects.values()
                       if o.classification == "piece")
        n_indicators = sum(1 for o in self.mechanics.objects.values()
                           if o.classification == "indicator")
        has_grid = any(g.structure == "grid" for g in self.mechanics.groups)
        has_ring = any(g.structure == "ring" for g in self.mechanics.groups)
        has_constraints = len(self.mechanics.constraints) > 0

        # Determine game type
        if has_constraints and has_grid:
            self.mechanics.game_type = "color-constraint"
        elif has_ring:
            self.mechanics.game_type = "ring-rotation"
        elif n_buttons > 0 and n_pieces > 0:
            self.mechanics.game_type = "button-piece"
        elif n_pieces > 0 and not n_buttons:
            self.mechanics.game_type = "direct-manipulation"
        elif self.action_model.infer_game_type() != "unknown":
            self.mechanics.game_type = self.action_model.infer_game_type()
        else:
            self.mechanics.game_type = "unknown"

        # Confidence
        if self.mechanics.calibrated and (n_buttons + n_pieces) > 0:
            self.mechanics.confidence = "high" if self.mechanics.correlations else "medium"
        elif not self.mechanics.calibrated:
            self.mechanics.confidence = "low"
        else:
            self.mechanics.confidence = "medium"
