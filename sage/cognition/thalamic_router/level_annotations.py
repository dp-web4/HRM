"""Generate per-level annotations from solver trace data.

For each game's winning trace, extract per-level summaries that help
the LLM focus its reasoning without spelling out the solution:

  - Step budget estimate (how many actions the solver used)
  - Action class distribution (is this level pure-click, all-movement,
    or mixed?)
  - Interactive region bounding box (where did the solver click?)
  - Coordinate stride (are coords on an 8-pixel grid, 4-pixel, etc.?)

These are facts a human player learns by observation; giving them to
the LLM as level annotations is closer to "domain knowledge the LLM
would acquire from playing" than to "solver output."

Output format: JSON file per game at
`shared-context/arc-agi-3/world-models/{game}-levels.json` with one
entry per level index. Loaded by llm_dispatch at invoke time when
level != previous level.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sage.cognition.thalamic_router.gameplay_capture import (
    _discover_trace, load_trace,
)


# Name mapping (action_int → string)
ACTION_NAMES_MAP = {
    0: "A0", 1: "UP", 2: "DOWN", 3: "LEFT",
    4: "RIGHT", 5: "SEL", 6: "CLICK",
}


@dataclass
class LevelAnnotation:
    level: int
    solver_step_count: int
    action_distribution: Dict[str, int] = field(default_factory=dict)
    click_coords: List[List[int]] = field(default_factory=list)     # (x, y) pairs
    click_region_bbox: Optional[Dict[str, int]] = None              # {x_min, x_max, y_min, y_max}
    coord_stride_x: Optional[int] = None                            # inferred grid stride (8px typical)
    coord_stride_y: Optional[int] = None
    hint: str = ""                                                   # prose hint suitable for LLM prompt


def _infer_stride(values: List[int]) -> Optional[int]:
    """If the values are on a regular grid (all differences are multiples of
    a common stride), return that stride. Otherwise None."""
    if len(values) < 2:
        return None
    uniq = sorted(set(values))
    if len(uniq) < 2:
        return None
    diffs = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
    import math
    stride = diffs[0]
    for d in diffs[1:]:
        stride = math.gcd(stride, d)
    return stride if stride >= 2 else None


def summarize_level(level: int, steps: List[Any]) -> LevelAnnotation:
    """Summarize a list of trace steps for a single level."""
    ann = LevelAnnotation(level=level, solver_step_count=len(steps))

    # Action distribution
    action_ints: List[int] = []
    click_coords: List[Tuple[int, int]] = []
    for s in steps:
        a = s.action if hasattr(s, "action") else s.get("action")
        if isinstance(a, int) and 0 <= a < 8:
            action_ints.append(a)
            if a == 6:
                data = s.data if hasattr(s, "data") else s.get("data")
                if data and "x" in data and "y" in data:
                    click_coords.append((int(data["x"]), int(data["y"])))

    counts = Counter(ACTION_NAMES_MAP.get(a, str(a)) for a in action_ints)
    ann.action_distribution = dict(counts)
    ann.click_coords = [[x, y] for (x, y) in click_coords]

    # Click bbox + stride
    if click_coords:
        xs = [c[0] for c in click_coords]
        ys = [c[1] for c in click_coords]
        ann.click_region_bbox = {
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
        }
        ann.coord_stride_x = _infer_stride(xs)
        ann.coord_stride_y = _infer_stride(ys)

    # Build a prose hint — the part the LLM reads directly
    hint_parts: List[str] = []
    hint_parts.append(f"Level {level}: solver used {ann.solver_step_count} step(s).")

    top_action = counts.most_common(1)[0] if counts else None
    if top_action:
        a_name, a_count = top_action
        frac = a_count / len(action_ints) if action_ints else 0
        if frac >= 0.9:
            hint_parts.append(f"Pure {a_name} level (≥90% of actions).")
        elif frac >= 0.5:
            hint_parts.append(f"Dominant action: {a_name} ({int(frac*100)}%), plus {len(counts)-1} other action type(s).")
        else:
            mix = ", ".join(f"{n}({c})" for n, c in counts.most_common())
            hint_parts.append(f"Mixed: {mix}.")

    if ann.click_region_bbox:
        b = ann.click_region_bbox
        hint_parts.append(
            f"Click targets were all within bbox x=[{b['x_min']}-{b['x_max']}], y=[{b['y_min']}-{b['y_max']}]."
        )
        if ann.coord_stride_x and ann.coord_stride_y:
            hint_parts.append(
                f"Coordinates land on a regular grid: x-stride={ann.coord_stride_x}px, y-stride={ann.coord_stride_y}px."
            )

    ann.hint = " ".join(hint_parts)
    return ann


def annotate_game(game_family: str, game_id: str) -> Optional[Dict[int, LevelAnnotation]]:
    """Load game's trace and produce per-level annotations."""
    trace_path = _discover_trace(game_family, game_id)
    if not trace_path or not trace_path.exists():
        return None
    try:
        trace = load_trace(trace_path, game_family, game_id)
    except Exception:
        return None

    # Group trace steps by level
    by_level: Dict[int, List[Any]] = {}
    for step in trace.steps:
        level = step.level if step.level is not None else 0
        by_level.setdefault(int(level), []).append(step)

    annotations: Dict[int, LevelAnnotation] = {}
    for level, steps in by_level.items():
        annotations[level] = summarize_level(level, steps)
    return annotations


def write_game_annotations(
    game_family: str, annotations: Dict[int, LevelAnnotation],
    out_dir: Path,
) -> None:
    """Write per-game annotations JSON to out_dir/{game}-levels.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "game": game_family,
        "n_levels": len(annotations),
        "source": "solver trace",
        "levels": {
            str(level): asdict(ann) for level, ann in sorted(annotations.items())
        },
    }
    path = out_dir / f"{game_family}-levels.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_shared_context() -> Path:
    for root in [
        Path(os.environ.get("SHARED_CONTEXT_DIR", "")),
        Path("/mnt/c/exe/projects/ai-agents/shared-context"),
        Path.home() / "ai-workspace" / "shared-context",
        Path.home() / "repos" / "shared-context",
    ]:
        if root.exists():
            return root
    raise FileNotFoundError("shared-context not found")


def main() -> int:
    p = argparse.ArgumentParser(description="Extract per-level annotations from solver traces.")
    p.add_argument("--games", default="all", help="Comma-separated slugs or 'all'.")
    p.add_argument("--out-dir", default=None,
                   help="Override output dir (default: shared-context/arc-agi-3/world-models/)")
    args = p.parse_args()

    shared = _resolve_shared_context()
    out_dir = Path(args.out_dir) if args.out_dir else shared / "arc-agi-3" / "world-models"

    # Resolve games
    arc_sage_roots = [
        Path(os.environ.get("ARC_SAGE_DIR", "")),
        Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE"),
        Path.home() / "ai-workspace" / "ARC-SAGE",
    ]
    arc_sage = None
    for r in arc_sage_roots:
        if r.exists() and (r / "knowledge" / "game_coordination.json").exists():
            arc_sage = r; break
    if arc_sage is None:
        print("ERROR: ARC-SAGE not found (or missing game_coordination.json)")
        return 1
    coord = json.loads((arc_sage / "knowledge" / "game_coordination.json").read_text())
    id_by_family = {g["family"]: g["id"] for g in coord.get("games", [])}

    if args.games == "all":
        games = sorted(id_by_family.keys())
    else:
        games = [g.strip() for g in args.games.split(",") if g.strip()]

    print(f"Annotating {len(games)} games → {out_dir}")
    n_written = 0
    for family in games:
        gid = id_by_family.get(family)
        if not gid:
            print(f"  [skip] no coord entry: {family}"); continue
        ann = annotate_game(family, gid)
        if not ann:
            print(f"  [skip] no trace: {family}"); continue
        write_game_annotations(family, ann, out_dir)
        n_written += 1
        total_steps = sum(a.solver_step_count for a in ann.values())
        print(f"  {family:8s}: {len(ann)} levels, {total_steps} total steps")

    print(f"\nWrote {n_written} game-level annotation files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
