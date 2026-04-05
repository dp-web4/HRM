#!/usr/bin/env python3
"""
ARC-AGI-3 Spatial Reasoning Layer

Builds on arc_perception.py to provide higher-level spatial understanding:
- Object identity tracking across frames (which region is which after movement)
- Interactive object detection (which regions respond to clicks/actions)
- Spatial relationship primitives (alignment, adjacency, containment, paths)
- Movement vector extraction (what moved, how far, in what direction)

This is SAGE's "spatial cortex" — it understands WHERE things are relative
to each other and HOW they change, not just WHAT colors exist.

Design:
- Stateful: maintains object registry across frames
- Builds on arc_perception.py (uses find_color_regions, grid_diff, etc.)
- Returns structured data + text summaries for LLM consumption
- No game-specific knowledge
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from arc_perception import (
    find_color_regions, background_color, color_name,
    detect_sections, grid_diff,
)


# ─────────────────────────────────────────────────────────────
# Object tracking
# ─────────────────────────────────────────────────────────────

class SpatialObject:
    """A tracked object on the grid — persists across frames."""

    __slots__ = ("id", "color", "x", "y", "w", "h", "cx", "cy", "size",
                 "is_interactive", "click_responses", "move_history")

    def __init__(self, obj_id: int, region: dict):
        self.id = obj_id
        self.color = region["color"]
        self.x = region["x"]
        self.y = region["y"]
        self.w = region["w"]
        self.h = region["h"]
        self.cx = region["cx"]
        self.cy = region["cy"]
        self.size = region["size"]
        self.is_interactive = None  # None=unknown, True/False after testing
        self.click_responses = []   # list of {changed: bool, n_pixels: int}
        self.move_history = []      # list of (dx, dy) movements

    @property
    def center(self) -> Tuple[int, int]:
        return (self.cx, self.cy)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def click_effectiveness(self) -> float:
        if not self.click_responses:
            return -1.0  # unknown
        hits = sum(1 for r in self.click_responses if r["changed"])
        return hits / len(self.click_responses)

    def update_position(self, region: dict):
        dx = region["cx"] - self.cx
        dy = region["cy"] - self.cy
        if dx != 0 or dy != 0:
            self.move_history.append((dx, dy))
        self.x = region["x"]
        self.y = region["y"]
        self.w = region["w"]
        self.h = region["h"]
        self.cx = region["cx"]
        self.cy = region["cy"]
        self.size = region["size"]

    def describe(self) -> str:
        name = color_name(self.color)
        eff = ""
        if self.is_interactive is True:
            eff = " [INTERACTIVE]"
        elif self.is_interactive is False:
            eff = " [static]"
        moves = ""
        if self.move_history:
            last = self.move_history[-1]
            moves = f" moved({last[0]:+d},{last[1]:+d})"
        return f"obj{self.id}:{name}({self.w}x{self.h})@({self.cx},{self.cy}){eff}{moves}"


# ─────────────────────────────────────────────────────────────
# Spatial tracker — maintains state across frames
# ─────────────────────────────────────────────────────────────

class SpatialTracker:
    """Tracks objects and their spatial relationships across frames.

    Usage:
        tracker = SpatialTracker()
        tracker.update(grid)            # First frame
        tracker.update(grid_after)      # After action
        print(tracker.describe())       # What changed spatially
    """

    def __init__(self, min_region_size: int = 4):
        self.min_region_size = min_region_size
        self.objects: Dict[int, SpatialObject] = {}
        self.next_id = 0
        self.frame_count = 0
        self.prev_grid: Optional[np.ndarray] = None

        # Spatial relationship cache (updated each frame)
        self._alignments: List[dict] = []
        self._adjacencies: List[dict] = []

        # Action-outcome tracking
        self.action_outcomes: List[dict] = []  # {action, objects_moved, objects_changed}

    def update(self, grid: np.ndarray) -> dict:
        """Process a new frame. Returns spatial diff summary."""
        regions = find_color_regions(grid, min_size=self.min_region_size)
        self.frame_count += 1

        if self.frame_count == 1:
            # First frame — initialize all objects
            for r in regions:
                obj = SpatialObject(self.next_id, r)
                self.objects[self.next_id] = obj
                self.next_id += 1
            self._update_relationships()
            self.prev_grid = grid.copy()
            return {"type": "init", "n_objects": len(self.objects)}

        # Match regions to existing objects (by color + proximity)
        matched, new, lost = self._match_objects(regions)

        # Update matched objects
        movements = []
        for obj_id, region in matched:
            obj = self.objects[obj_id]
            old_cx, old_cy = obj.cx, obj.cy
            obj.update_position(region)
            if obj.cx != old_cx or obj.cy != old_cy:
                movements.append({
                    "obj": obj.id,
                    "color": color_name(obj.color),
                    "from": (old_cx, old_cy),
                    "to": (obj.cx, obj.cy),
                    "dx": obj.cx - old_cx,
                    "dy": obj.cy - old_cy,
                })

        # Create new objects
        for region in new:
            obj = SpatialObject(self.next_id, region)
            self.objects[self.next_id] = obj
            self.next_id += 1

        # Mark lost objects (don't delete — they might reappear)
        for obj_id in lost:
            pass  # Keep in registry, just note they're not visible

        self._update_relationships()
        self.prev_grid = grid.copy()

        return {
            "type": "update",
            "movements": movements,
            "new_objects": len(new),
            "lost_objects": len(lost),
            "n_objects": len(self.objects),
        }

    def record_click(self, x: int, y: int, changed: bool, n_pixels: int = 0):
        """Record a click and which object it hit."""
        for obj in self.objects.values():
            if (obj.x <= x < obj.x + obj.w and obj.y <= y < obj.y + obj.h):
                obj.click_responses.append({"changed": changed, "n_pixels": n_pixels})
                # Mark interactive on first positive response (don't wait for 2)
                if changed:
                    obj.is_interactive = True
                elif len(obj.click_responses) >= 2 and not any(r["changed"] for r in obj.click_responses):
                    obj.is_interactive = False
                return obj.id
        return None

    def record_action_outcome(self, action: int, action_data: Optional[dict],
                              diff_summary: dict):
        """Record what happened after an action."""
        self.action_outcomes.append({
            "action": action,
            "data": action_data,
            "movements": diff_summary.get("movements", []),
            "new_objects": diff_summary.get("new_objects", 0),
            "lost_objects": diff_summary.get("lost_objects", 0),
        })

    def _match_objects(self, regions: list) -> Tuple[list, list, list]:
        """Match detected regions to existing objects.

        Uses color + proximity. Returns (matched, new, lost) where:
        - matched: list of (obj_id, region)
        - new: list of unmatched regions
        - lost: list of obj_ids not matched to any region
        """
        available_objects = {oid: obj for oid, obj in self.objects.items()}
        matched = []
        used_objects = set()

        for region in regions:
            best_obj = None
            best_dist = float('inf')
            for oid, obj in available_objects.items():
                if oid in used_objects:
                    continue
                if obj.color != region["color"]:
                    continue
                dist = abs(obj.cx - region["cx"]) + abs(obj.cy - region["cy"])
                if dist < best_dist:
                    best_dist = dist
                    best_obj = oid

            if best_obj is not None and best_dist < 32:  # max 32 pixel movement
                matched.append((best_obj, region))
                used_objects.add(best_obj)
            else:
                # Unmatched region — could be new object or color-changed
                pass  # Will be added as new

        new_regions = [r for i, r in enumerate(regions)
                       if not any(m[1] is r for m in matched)]
        lost_ids = [oid for oid in available_objects if oid not in used_objects]

        return matched, new_regions, lost_ids

    def _update_relationships(self):
        """Compute spatial relationships between all object pairs."""
        objs = list(self.objects.values())
        self._alignments = []
        self._adjacencies = []

        for i, a in enumerate(objs):
            for b in objs[i+1:]:
                # Check alignment
                if abs(a.cy - b.cy) <= 2:
                    self._alignments.append({
                        "type": "horizontal",
                        "objects": (a.id, b.id),
                        "y": a.cy,
                    })
                if abs(a.cx - b.cx) <= 2:
                    self._alignments.append({
                        "type": "vertical",
                        "objects": (a.id, b.id),
                        "x": a.cx,
                    })

                # Check adjacency (bounding boxes touch or overlap)
                gap_x = max(0, max(a.x, b.x) - min(a.x + a.w, b.x + b.w))
                gap_y = max(0, max(a.y, b.y) - min(a.y + a.h, b.y + b.h))
                if gap_x <= 2 and gap_y <= 2:
                    self._adjacencies.append({
                        "objects": (a.id, b.id),
                        "gap": (gap_x, gap_y),
                    })

    # ─── Query methods ───

    def get_interactive_objects(self) -> List[SpatialObject]:
        """Objects confirmed to respond to clicks."""
        return [o for o in self.objects.values() if o.is_interactive is True]

    def get_static_objects(self) -> List[SpatialObject]:
        """Objects confirmed to NOT respond to clicks."""
        return [o for o in self.objects.values() if o.is_interactive is False]

    def get_untested_objects(self) -> List[SpatialObject]:
        """Objects we haven't clicked yet."""
        return [o for o in self.objects.values() if o.is_interactive is None]

    def get_aligned_objects(self, axis: str = "horizontal") -> List[List[SpatialObject]]:
        """Groups of objects aligned on the same row or column."""
        groups = defaultdict(list)
        for al in self._alignments:
            if al["type"] == axis:
                key = al.get("y" if axis == "horizontal" else "x")
                for oid in al["objects"]:
                    if oid in self.objects:
                        groups[key].append(self.objects[oid])
        return [list(set(g)) for g in groups.values() if len(set(g)) >= 2]

    def get_movement_pattern(self) -> Optional[str]:
        """Detect if objects move in a consistent pattern."""
        all_moves = []
        for obj in self.objects.values():
            all_moves.extend(obj.move_history)
        if not all_moves:
            return None
        dx_avg = sum(m[0] for m in all_moves) / len(all_moves)
        dy_avg = sum(m[1] for m in all_moves) / len(all_moves)
        if abs(dx_avg) > abs(dy_avg):
            return f"horizontal (avg dx={dx_avg:+.1f})"
        elif abs(dy_avg) > abs(dx_avg):
            return f"vertical (avg dy={dy_avg:+.1f})"
        return None

    def suggest_click_targets(self, grid: np.ndarray, n: int = 3) -> List[Tuple[int, int, str]]:
        """Suggest the best click targets based on spatial reasoning.

        Priority:
        1. Interactive objects we know work (exploit)
        2. Untested objects near interactive ones (spatial inference)
        3. Objects with unique spatial properties (corners, isolated)

        Returns list of (x, y, reason) tuples.
        """
        targets = []

        # Priority 1: known interactive objects
        for obj in self.get_interactive_objects():
            targets.append((obj.cx, obj.cy,
                           f"interactive {color_name(obj.color)} "
                           f"({obj.click_effectiveness:.0%} effective)"))

        # Priority 2: untested objects adjacent to interactive ones
        interactive_ids = {o.id for o in self.get_interactive_objects()}
        for adj in self._adjacencies:
            a, b = adj["objects"]
            if a in interactive_ids and b in self.objects:
                obj = self.objects[b]
                if obj.is_interactive is None:
                    targets.append((obj.cx, obj.cy,
                                   f"untested {color_name(obj.color)}, adjacent to interactive"))
            elif b in interactive_ids and a in self.objects:
                obj = self.objects[a]
                if obj.is_interactive is None:
                    targets.append((obj.cx, obj.cy,
                                   f"untested {color_name(obj.color)}, adjacent to interactive"))

        # Priority 3: untested objects with unique properties
        untested = self.get_untested_objects()
        # Small objects are more likely buttons
        for obj in sorted(untested, key=lambda o: o.size):
            if len(targets) >= n * 2:
                break
            targets.append((obj.cx, obj.cy,
                           f"untested {color_name(obj.color)} "
                           f"({obj.w}x{obj.h}, size={obj.size})"))

        return targets[:n]

    # ─── Text descriptions for LLM ───

    def describe(self) -> str:
        """Full spatial description for LLM consumption."""
        lines = [f"Objects tracked: {len(self.objects)}"]

        # Interactive objects
        interactive = self.get_interactive_objects()
        if interactive:
            lines.append(f"\nInteractive ({len(interactive)}):")
            for obj in interactive:
                lines.append(f"  {obj.describe()}")

        # Static objects
        static = self.get_static_objects()
        if static:
            lines.append(f"\nStatic ({len(static)}):")
            for obj in static[:5]:
                lines.append(f"  {obj.describe()}")
            if len(static) > 5:
                lines.append(f"  ... and {len(static)-5} more")

        # Untested
        untested = self.get_untested_objects()
        if untested:
            lines.append(f"\nUntested ({len(untested)}):")
            for obj in sorted(untested, key=lambda o: o.size)[:5]:
                lines.append(f"  {obj.describe()}")

        # Spatial relationships
        h_groups = self.get_aligned_objects("horizontal")
        v_groups = self.get_aligned_objects("vertical")
        if h_groups or v_groups:
            lines.append(f"\nAlignments:")
            for group in h_groups[:3]:
                names = [f"{color_name(o.color)}@{o.cx}" for o in group]
                lines.append(f"  Row y={group[0].cy}: {', '.join(names)}")
            for group in v_groups[:3]:
                names = [f"{color_name(o.color)}@{o.cy}" for o in group]
                lines.append(f"  Col x={group[0].cx}: {', '.join(names)}")

        # Movement pattern
        pattern = self.get_movement_pattern()
        if pattern:
            lines.append(f"\nMovement pattern: {pattern}")

        return "\n".join(lines)

    def describe_diff(self, diff: dict) -> str:
        """Describe a spatial diff in human terms."""
        if diff["type"] == "init":
            return f"Initial frame: {diff['n_objects']} objects detected"

        parts = []
        if diff["movements"]:
            for m in diff["movements"]:
                direction = ""
                if abs(m["dx"]) > abs(m["dy"]):
                    direction = "right" if m["dx"] > 0 else "left"
                elif abs(m["dy"]) > abs(m["dx"]):
                    direction = "down" if m["dy"] > 0 else "up"
                else:
                    direction = f"({m['dx']:+d},{m['dy']:+d})"
                parts.append(f"{m['color']} moved {direction} by {max(abs(m['dx']),abs(m['dy']))}px")

        if diff["new_objects"]:
            parts.append(f"{diff['new_objects']} new object(s) appeared")
        if diff["lost_objects"]:
            parts.append(f"{diff['lost_objects']} object(s) disappeared")
        if not parts:
            parts.append("no spatial changes detected")

        return "; ".join(parts)
