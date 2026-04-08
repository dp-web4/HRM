#!/usr/bin/env python3
"""
Solver Probe — lightweight game exploration and mechanic discovery.

Part of SAGE Solver v11 modular architecture.

Two probe modes:
  1. Basic probe (from v7): test each action type, click visible regions
  2. MechanicDiscovery (from arc_discovery.py): structured 5-phase protocol

The basic probe is the fallback when MechanicDiscovery is unavailable or fails.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from arcengine import GameAction
from arc_perception import get_frame, find_color_regions

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def probe(env, fd, grid, available, budget=20):
    """Basic probe: test each available action, click visible regions.

    Returns (action_model, tracker, fd, grid, steps_used, level_ups).
    """
    from arc_spatial import SpatialTracker
    from arc_action_model import ActionEffectModel

    model = ActionEffectModel()
    tracker = SpatialTracker()
    tracker.update(grid)
    steps = 0
    start_levels = fd.levels_completed
    level_ups = 0

    # Test non-click actions (UP, DOWN, LEFT, RIGHT, SELECT, UNDO)
    for action in [a for a in available if a != 6]:
        for _ in range(2):
            if steps >= budget:
                break
            prev = grid.copy()
            try:
                fd = env.step(INT_TO_GAME_ACTION[action])
            except Exception:
                continue
            if fd is None:
                return model, tracker, fd, grid, steps, level_ups
            grid = get_frame(fd)
            steps += 1
            diff = tracker.update(grid)
            model.observe(action, prev, grid, diff)
            if fd.levels_completed > start_levels + level_ups:
                level_ups = fd.levels_completed - start_levels
            new_avail = [
                a.value if hasattr(a, "value") else int(a)
                for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                return model, tracker, fd, grid, steps, level_ups

    # Test click on visible regions (interleaved small/large)
    if 6 in available:
        regions = find_color_regions(grid, min_size=4)
        by_size = sorted(regions, key=lambda x: x["size"])
        interleaved = []
        left, right = 0, len(by_size) - 1
        while left <= right:
            interleaved.append(by_size[left])
            if left != right:
                interleaved.append(by_size[right])
            left += 1
            right -= 1

        seen = set()
        for r in interleaved:
            key = (r["color"], r["w"], r["h"])
            if key in seen:
                continue
            seen.add(key)
            if steps >= budget:
                break
            prev = grid.copy()
            data = {'x': r["cx"], 'y': r["cy"]}
            try:
                fd = env.step(GameAction.ACTION6, data=data)
            except Exception:
                continue
            if fd is None:
                return model, tracker, fd, grid, steps, level_ups
            grid = get_frame(fd)
            steps += 1
            diff = tracker.update(grid)
            model.observe(6, prev, grid, diff)
            changed = not np.array_equal(prev, grid)
            tracker.record_click(
                data['x'], data['y'], changed, int(np.sum(prev != grid)))
            if fd.levels_completed > start_levels + level_ups:
                level_ups = fd.levels_completed - start_levels
            new_avail = [
                a.value if hasattr(a, "value") else int(a)
                for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                break

    return model, tracker, fd, grid, steps, level_ups


def discover_mechanics(env, fd, grid, available, budget=300, verbose=False):
    """Structured 5-phase mechanic discovery via MechanicDiscovery.

    Returns (action_model, tracker, fd, grid, steps_used, level_ups, mechanics)
    or None if MechanicDiscovery is unavailable.
    """
    try:
        from arc_discovery import MechanicDiscovery
    except ImportError:
        return None

    discovery = MechanicDiscovery(
        env, fd, grid, available,
        total_budget=budget, verbose=verbose)
    action_model, tracker, fd, grid, probe_steps, probe_levels, mechanics = \
        discovery.discover()
    return (action_model, tracker, fd, grid, probe_steps, probe_levels,
            mechanics)
