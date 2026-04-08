#!/usr/bin/env python3
"""
Solver Actions — action name mapping, object naming, plan text parsing.

Part of SAGE Solver v11 modular architecture.

Extracted from sage_solver_v7.py with the REPEAT-aware parse_actions fix
and color-name-to-coordinate resolution.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from arc_perception import color_name


ACTION_NAMES = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SELECT", 6: "CLICK", 7: "UNDO",
}


def obj_name(obj):
    """Canonical name for a tracked spatial object: '<color>_<id>'."""
    return f"{color_name(obj.color)}_{obj.id}"


def obj_catalog(tracker):
    """Build a text catalog of all tracked objects, grouped by role.

    Returns a multi-line string suitable for LLM context.
    """
    lines = []
    interactive = tracker.get_interactive_objects()
    if interactive:
        lines.append("INTERACTIVE (proven — use these):")
        for o in interactive:
            lines.append(
                f"  {obj_name(o)} ({o.w}x{o.h}) at ({o.cx},{o.cy})"
                f" — {o.click_effectiveness:.0%}")
    static = tracker.get_static_objects()
    if static:
        lines.append(f"NO EFFECT ({len(static)} objects — skip)")
    untested = tracker.get_untested_objects()
    if untested:
        small = [o for o in untested if o.size <= 64]
        if small:
            lines.append(
                f"UNTESTED ({len(small)} small objects — explore if needed):")
            for o in sorted(small, key=lambda x: x.size)[:3]:
                lines.append(
                    f"  {obj_name(o)} ({o.w}x{o.h}) at ({o.cx},{o.cy})")
    return "\n".join(lines)


def parse_actions(plan_text, tracker, available):
    """Parse LLM plan text into a list of executable action tuples.

    Supports:
      CLICK <color>_<id>     — click named object
      CLICK <color>          — click first object of that color
      CLICK ALL <color>      — click all objects of that color
      CLICK INTERACTIVE      — click all proven-interactive objects
      REPEAT N <action>      — repeat an action N times (max 30)
      UP / DOWN / LEFT / RIGHT / SELECT / UNDO

    Returns list of (action_int, data_dict_or_None, target_name).
    """
    # Build name→object map from tracker
    name_map = {}
    for obj in tracker.objects.values():
        name_map[obj_name(obj).upper()] = obj
        cname = color_name(obj.color).upper()
        if cname not in name_map:
            name_map[cname] = []
        if isinstance(name_map.get(cname), list):
            name_map[cname].append(obj)

    actions = []
    for line in plan_text.split("\n"):
        # Strip leading numbering like "1. " or "2) "
        line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
        upper = line.upper().strip()
        if not upper:
            continue

        # REPEAT N <sub-action>
        m = re.match(r'REPEAT\s+(\d+)\s+(.+)', upper)
        if m:
            n = min(int(m.group(1)), 30)
            sub = parse_actions(m.group(2), tracker, available)
            actions.extend(sub * n)
            continue

        # CLICK ALL <color>
        m = re.match(r'CLICK\s+ALL\s+(\w+)', upper)
        if m and 6 in available:
            objs = name_map.get(m.group(1), [])
            if isinstance(objs, list):
                for o in objs:
                    actions.append(
                        (6, {'x': o.cx, 'y': o.cy}, obj_name(o)))
            continue

        # CLICK INTERACTIVE
        if 'CLICK INTERACTIVE' in upper and 6 in available:
            for o in tracker.get_interactive_objects():
                actions.append(
                    (6, {'x': o.cx, 'y': o.cy}, obj_name(o)))
            continue

        # CLICK <color>_<id> (exact named object)
        m = re.match(r'CLICK\s+(\w+_\d+)', upper)
        if m and 6 in available:
            obj = name_map.get(m.group(1))
            if obj and not isinstance(obj, list):
                actions.append(
                    (6, {'x': obj.cx, 'y': obj.cy}, obj_name(obj)))
            continue

        # CLICK <color> (first object of that color)
        m = re.match(r'CLICK\s+(\w+)', upper)
        if m and 6 in available:
            objs = name_map.get(m.group(1), [])
            if isinstance(objs, list) and objs:
                o = objs[0]
                actions.append(
                    (6, {'x': o.cx, 'y': o.cy}, obj_name(o)))
            continue

        # Directional actions
        for name, num in [("UP", 1), ("DOWN", 2), ("LEFT", 3), ("RIGHT", 4)]:
            if name in upper and num in available:
                actions.append((num, None, name))
                break
        else:
            # SELECT / SUBMIT
            if any(w in upper for w in ["SELECT", "SUBMIT"]) and 5 in available:
                actions.append((5, None, "SELECT"))
            # UNDO
            elif "UNDO" in upper and 7 in available:
                actions.append((7, None, "UNDO"))

    return actions
