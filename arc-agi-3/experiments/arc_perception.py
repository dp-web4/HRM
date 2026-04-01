#!/usr/bin/env python3
"""
ARC-AGI-3 Grid Perception Toolkit

Pure numpy grid analysis that converts 64x64 color grids into structured
text descriptions a small LLM (0.8B) can reason over.

This is SAGE's "eyes" — it doesn't decide what to do, it just reports
what it sees in a format the LLM can understand.

Design principles:
- No game-specific knowledge baked in
- Returns text descriptions, not action plans
- Composable: each function extracts one type of feature
- Deterministic: same grid → same description
"""

import numpy as np
from collections import Counter


COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "cyan", 9: "brown",
    10: "pink", 11: "maroon", 12: "olive", 13: "navy", 14: "teal",
    15: "white",
}


def get_frame(fd):
    """Extract 2D grid from frame data."""
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


def background_color(grid):
    """Identify the most common color (background)."""
    return int(np.bincount(grid.astype(int).flatten()).argmax())


def color_name(c):
    """Human-readable color name."""
    return COLOR_NAMES.get(int(c), f"color{c}")


def grid_summary(grid):
    """High-level grid summary: size, background, non-bg colors and counts."""
    bg = background_color(grid)
    flat = grid.astype(int).flatten()
    counts = Counter(flat)
    non_bg = {c: n for c, n in counts.items() if c != bg}

    lines = [f"Grid: {grid.shape[0]}x{grid.shape[1]}, background={color_name(bg)}"]
    if non_bg:
        color_list = ", ".join(f"{color_name(c)}({n}px)" for c, n in
                               sorted(non_bg.items(), key=lambda x: -x[1]))
        lines.append(f"Colors: {color_list}")
    else:
        lines.append("Grid is entirely background color")
    return "\n".join(lines)


def find_color_regions(grid, min_size=4):
    """Find contiguous rectangular regions of non-background color.

    Returns list of {color, x, y, w, h, size} dicts, sorted by y then x.
    """
    bg = background_color(grid)
    visited = np.zeros_like(grid, dtype=bool)
    regions = []

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            # Flood-fill to find extent
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if (cr < 0 or cr >= grid.shape[0] or cc < 0 or cc >= grid.shape[1]):
                    continue
                if visited[cr, cc] or int(grid[cr, cc]) != color:
                    continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                stack.extend([(cr+1, cc), (cr-1, cc), (cr, cc+1), (cr, cc-1)])

            if len(cells) >= min_size:
                rows = [p[0] for p in cells]
                cols = [p[1] for p in cells]
                regions.append({
                    "color": color,
                    "color_name": color_name(color),
                    "x": min(cols),
                    "y": min(rows),
                    "w": max(cols) - min(cols) + 1,
                    "h": max(rows) - min(rows) + 1,
                    "cx": (min(cols) + max(cols)) // 2,
                    "cy": (min(rows) + max(rows)) // 2,
                    "size": len(cells),
                })

    return sorted(regions, key=lambda r: (r["y"], r["x"]))


def describe_spatial_layout(regions):
    """Describe spatial arrangement of color regions.

    Groups regions into top/middle/bottom thirds and describes arrangement.
    """
    if not regions:
        return "No colored regions found"

    # Determine grid extent from regions
    max_y = max(r["y"] + r["h"] for r in regions)
    third = max_y // 3

    top = [r for r in regions if r["cy"] < third]
    mid = [r for r in regions if third <= r["cy"] < 2 * third]
    bot = [r for r in regions if r["cy"] >= 2 * third]

    lines = []
    for label, group in [("Top", top), ("Middle", mid), ("Bottom", bot)]:
        if group:
            items = ", ".join(f"{r['color_name']}@({r['cx']},{r['cy']})"
                              for r in sorted(group, key=lambda r: r["x"]))
            lines.append(f"{label}: {items}")

    return "\n".join(lines) if lines else "No spatial structure detected"


def find_row_patterns(grid, row_idx, min_run=3):
    """Scan a row for contiguous color runs.

    Returns list of {color, color_name, x_start, x_end, center_x, width}.
    """
    bg = background_color(grid)
    row = grid[row_idx, :].astype(int)
    runs = []
    c = 0
    while c < row.shape[0]:
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        start = c
        while c < row.shape[0] and int(row[c]) == color:
            c += 1
        width = c - start
        if width >= min_run:
            runs.append({
                "color": color,
                "color_name": color_name(color),
                "x_start": start,
                "x_end": c - 1,
                "center_x": (start + c - 1) // 2,
                "width": width,
            })
    return runs


def find_markers(grid, target_color, min_cluster=2):
    """Find clusters of a specific color (e.g., empty slot markers).

    Returns list of {x, y} center positions.
    """
    positions = np.argwhere(grid.astype(int) == target_color)
    if len(positions) == 0:
        return []

    # Cluster nearby positions
    clusters = []
    used = set()
    for i, (r, c) in enumerate(positions):
        if i in used:
            continue
        cluster = [(int(r), int(c))]
        used.add(i)
        for j, (r2, c2) in enumerate(positions):
            if j in used:
                continue
            if abs(r2 - r) < 4 and abs(c2 - c) < 4:
                cluster.append((int(r2), int(c2)))
                used.add(j)
        if len(cluster) >= min_cluster:
            cy = sum(p[0] for p in cluster) // len(cluster)
            cx = sum(p[1] for p in cluster) // len(cluster)
            clusters.append({"x": cx, "y": cy})

    return sorted(clusters, key=lambda c: (c["y"], c["x"]))


def detect_sections(grid):
    """Detect horizontal sections separated by background or border rows.

    Returns list of {y_start, y_end, label} describing grid sections.
    """
    bg = background_color(grid)
    row_diversity = []
    for r in range(grid.shape[0]):
        unique = len(set(int(x) for x in grid[r, :]) - {bg})
        row_diversity.append(unique)

    # Find section boundaries (rows with only background)
    sections = []
    in_section = False
    start = 0
    for r in range(grid.shape[0]):
        if row_diversity[r] > 0 and not in_section:
            start = r
            in_section = True
        elif row_diversity[r] == 0 and in_section:
            sections.append({"y_start": start, "y_end": r - 1})
            in_section = False
    if in_section:
        sections.append({"y_start": start, "y_end": grid.shape[0] - 1})

    return sections


def grid_diff(grid_before, grid_after):
    """Describe what changed between two grid states.

    Returns text description of changes.
    """
    if grid_before.shape != grid_after.shape:
        return "Grid size changed"

    diff = grid_before.astype(int) != grid_after.astype(int)
    n_changed = int(diff.sum())

    if n_changed == 0:
        return "No change"

    # Find changed region bounding box
    changed_pos = np.argwhere(diff)
    min_r, min_c = changed_pos.min(axis=0)
    max_r, max_c = changed_pos.max(axis=0)

    # What colors appeared/disappeared
    before_colors = Counter(grid_before[diff].astype(int).tolist())
    after_colors = Counter(grid_after[diff].astype(int).tolist())

    appeared = {color_name(c): n for c, n in after_colors.items()
                if c not in before_colors or after_colors[c] > before_colors.get(c, 0)}
    disappeared = {color_name(c): n for c, n in before_colors.items()
                   if c not in after_colors or before_colors[c] > after_colors.get(c, 0)}

    lines = [f"{n_changed} pixels changed in region ({min_c},{min_r})-({max_c},{max_r})"]
    if appeared:
        lines.append(f"Appeared: {appeared}")
    if disappeared:
        lines.append(f"Disappeared: {disappeared}")
    return "\n".join(lines)


def full_perception(grid):
    """Complete grid perception — returns structured text for LLM consumption.

    This is the main entry point. Call this, pass result to LLM.
    """
    lines = [grid_summary(grid)]

    # Detect sections
    sections = detect_sections(grid)
    if sections:
        lines.append(f"\nSections ({len(sections)}):")
        for i, s in enumerate(sections):
            lines.append(f"  Section {i}: rows {s['y_start']}-{s['y_end']}")

    # Find color regions
    regions = find_color_regions(grid, min_size=6)
    if regions:
        lines.append(f"\nColor regions ({len(regions)}):")
        # Group by section
        for i, s in enumerate(sections if sections else [{"y_start": 0, "y_end": grid.shape[0]}]):
            sec_regions = [r for r in regions
                          if s["y_start"] <= r["cy"] <= s["y_end"]]
            if sec_regions:
                items = ", ".join(
                    f"{r['color_name']}({r['w']}x{r['h']})@({r['cx']},{r['cy']})"
                    for r in sec_regions[:10])  # cap at 10 per section
                lines.append(f"  Section {i}: {items}")

    # Spatial layout summary
    lines.append(f"\nLayout:\n{describe_spatial_layout(regions)}")

    return "\n".join(lines)


def perception_for_action(grid, action_type, click_pos=None):
    """Focused perception around a planned action.

    Returns description of what's at/near the action target.
    """
    if action_type == 6 and click_pos:
        x, y = click_pos
        # Describe what's at and near the click position
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            color = int(grid[y, x])
            # 5x5 neighborhood
            y0, y1 = max(0, y-2), min(grid.shape[0], y+3)
            x0, x1 = max(0, x-2), min(grid.shape[1], x+3)
            neighborhood = grid[y0:y1, x0:x1].astype(int)
            neighbors = Counter(neighborhood.flatten().tolist())
            return (f"Click ({x},{y}): {color_name(color)}, "
                    f"neighborhood: {dict(neighbors)}")
    return f"Action {action_type}"


# ─── Visual Memory Integration ───

def visual_similarity(grid_a, grid_b):
    """Compute pixel-wise visual similarity between two grids [0-1].
    
    Returns 1.0 for identical grids, 0.0 for completely different.
    Handles different shapes by returning 0.0.
    """
    if grid_a.shape != grid_b.shape:
        return 0.0
    total_cells = grid_a.size
    matching_cells = np.sum(grid_a == grid_b)
    return float(matching_cells / total_cells)


def color_effectiveness_summary(color_tries, color_changes):
    """Format color effectiveness learning for LLM.
    
    Args:
        color_tries: dict {color: num_tries}
        color_changes: dict {color: num_changes}
        
    Returns:
        Text summary of which colors cause grid changes.
    """
    if not color_tries:
        return "No color effectiveness data yet."
    
    lines = []
    effective = []  # >30% rate
    neutral = []    # 1-30% rate
    ineffective = []  # 0% rate
    
    for color in sorted(color_tries.keys()):
        tries = color_tries[color]
        changes = color_changes.get(color, 0)
        rate = changes / max(tries, 1)
        
        entry = f"{color_name(color)}({changes}/{tries}={rate:.0%})"
        if rate > 0.3:
            effective.append(entry)
        elif rate > 0:
            neutral.append(entry)
        else:
            ineffective.append(entry)
    
    if effective:
        lines.append(f"Effective colors (click causes change): {', '.join(effective)}")
    if neutral:
        lines.append(f"Sometimes effective: {', '.join(neutral)}")
    if ineffective:
        lines.append(f"Ineffective (no change): {', '.join(ineffective)}")
    
    return "\n".join(lines) if lines else "All colors untested."


def visual_memory_context(grid, cartridge=None):
    """Check visual memory for similar past states.
    
    Args:
        grid: Current grid
        cartridge: Membot cartridge with visual_memory
        
    Returns:
        Text description of visual similarities to known states.
    """
    if not cartridge or "visual_memory" not in cartridge.data:
        return None
    
    snapshots = cartridge.data["visual_memory"].get("snapshots", {})
    if not snapshots:
        return None
    
    # Find similar snapshots
    similar = []
    for label, snapshot in snapshots.items():
        try:
            stored_frame = cartridge._base64_to_frame(snapshot["frame_b64"])
            sim = visual_similarity(grid, stored_frame)
            if sim > 0.7:  # >70% similar
                meta = snapshot.get("metadata", {})
                desc = meta.get("description", label)
                similar.append(f"{desc} (similarity: {sim:.1%})")
        except Exception:
            continue
    
    if similar:
        return "Visual memory matches:\n- " + "\n- ".join(similar[:3])
    
    return None
