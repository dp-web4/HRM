#!/usr/bin/env python3
"""
Frame State Extraction — compute what pixels can't tell the LLM.

Takes (prev_frame, curr_frame) and produces structured text describing
the game state: color distribution, spatial layout, change location,
object positions, progress signals. All game-generic, all numpy, <1ms.

Injected into the LLM prompt so the model reasons from COMPUTED FACTS
instead of trying to extract them from a 64x64 pixel image.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# ARC-AGI-3 color names
COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "cyan", 9: "brown",
    10: "purple", 11: "teal", 12: "pink", 13: "lime",
    14: "dark-gray", 15: "white",
}


def extract_state(
    curr_frame: np.ndarray,
    prev_frame: Optional[np.ndarray] = None,
    history_frames: Optional[List[np.ndarray]] = None,
) -> str:
    """Extract structured state description from game frames.

    Args:
        curr_frame: current frame (H, W) int array, values 0-15
        prev_frame: previous frame, same shape (for change detection)
        history_frames: older frames for progress tracking

    Returns:
        Multi-line text block for prompt injection
    """
    # Normalize to 2D
    curr = _to_2d(curr_frame)
    prev = _to_2d(prev_frame) if prev_frame is not None else None
    h, w = curr.shape

    parts = []

    # 1. Color distribution
    parts.append(_color_distribution(curr))

    # 2. Spatial layout (4x4 grid summary)
    parts.append(_spatial_layout(curr))

    # 3. Change from last action
    if prev is not None and prev.shape == curr.shape:
        parts.append(_change_description(prev, curr))

    # 4. Non-background object regions
    parts.append(_object_regions(curr))

    # 5. Progress signal (similarity to N steps ago)
    if history_frames and len(history_frames) >= 5:
        old = _to_2d(history_frames[-5])
        if old.shape == curr.shape:
            sim = float(np.mean(old == curr))
            if sim > 0.95:
                parts.append(f"Progress: frame is {sim:.0%} similar to 5 steps ago — very little has changed.")
            elif sim > 0.80:
                parts.append(f"Progress: frame is {sim:.0%} similar to 5 steps ago — moderate change.")
            else:
                parts.append(f"Progress: frame is {sim:.0%} similar to 5 steps ago — significant change.")

    return "\n".join(parts)


def _to_2d(frame: np.ndarray) -> np.ndarray:
    """Normalize frame to (H, W) int array."""
    if frame is None:
        return np.zeros((64, 64), dtype=np.int32)
    f = np.asarray(frame)
    if f.ndim == 3:
        if f.shape[0] <= 16:  # one-hot (C, H, W)
            return f.argmax(axis=0).astype(np.int32)
        return f[-1].astype(np.int32)  # multi-layer, take last
    return f.astype(np.int32)


def _color_distribution(frame: np.ndarray) -> str:
    """Color histogram as text."""
    h, w = frame.shape
    total = h * w
    counts = np.bincount(frame.ravel(), minlength=16)

    # Sort by frequency, skip zero-count
    entries = []
    for idx in np.argsort(-counts):
        if counts[idx] == 0:
            break
        pct = counts[idx] / total * 100
        if pct >= 1.0:  # skip <1% colors
            name = COLOR_NAMES.get(int(idx), f"c{idx}")
            entries.append(f"{name} {pct:.0f}%")

    return f"Colors: {', '.join(entries)}"


def _spatial_layout(frame: np.ndarray, grid_n: int = 4) -> str:
    """Divide frame into grid, report dominant color per cell."""
    h, w = frame.shape
    bh, bw = max(h // grid_n, 1), max(w // grid_n, 1)

    labels = {
        (0, 0): "top-left", (0, 1): "top-ctr-left", (0, 2): "top-ctr-right", (0, 3): "top-right",
        (1, 0): "mid-left", (1, 1): "mid-center-L", (1, 2): "mid-center-R", (1, 3): "mid-right",
        (2, 0): "low-left", (2, 1): "low-center-L", (2, 2): "low-center-R", (2, 3): "low-right",
        (3, 0): "bot-left", (3, 1): "bot-ctr-left", (3, 2): "bot-ctr-right", (3, 3): "bot-right",
    }

    lines = ["Spatial layout:"]
    for gy in range(grid_n):
        row_parts = []
        for gx in range(grid_n):
            block = frame[gy*bh:(gy+1)*bh, gx*bw:(gx+1)*bw]
            if block.size == 0:
                continue
            counts = np.bincount(block.ravel(), minlength=16)
            top1 = int(np.argmax(counts))
            top1_pct = counts[top1] / max(block.size, 1) * 100
            name = COLOR_NAMES.get(top1, f"c{top1}")
            # Second color if significant
            counts_copy = counts.copy()
            counts_copy[top1] = 0
            top2 = int(np.argmax(counts_copy))
            top2_pct = counts_copy[top2] / max(block.size, 1) * 100
            if top2_pct >= 15:
                name2 = COLOR_NAMES.get(top2, f"c{top2}")
                row_parts.append(f"{name}({top1_pct:.0f}%)+{name2}({top2_pct:.0f}%)")
            else:
                row_parts.append(f"{name}({top1_pct:.0f}%)")
        lines.append("  " + " | ".join(row_parts))

    return "\n".join(lines)


def _change_description(prev: np.ndarray, curr: np.ndarray) -> str:
    """Describe what changed between frames."""
    changed = prev != curr
    n_changed = int(np.sum(changed))
    total = prev.size

    if n_changed == 0:
        return "Last action: NO visible change. The action had no effect."

    pct = n_changed / total * 100

    # Where did changes happen?
    ys, xs = np.where(changed)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    h, w = prev.shape
    loc_y = "top" if cy < h // 3 else ("middle" if cy < 2 * h // 3 else "bottom")
    loc_x = "left" if cx < w // 3 else ("center" if cx < 2 * w // 3 else "right")

    # What colors appeared/disappeared?
    old_at_changed = prev[changed]
    new_at_changed = curr[changed]
    old_colors = np.bincount(old_at_changed, minlength=16)
    new_colors = np.bincount(new_at_changed, minlength=16)
    diff = new_colors.astype(int) - old_colors.astype(int)

    grew = [(COLOR_NAMES.get(i, f"c{i}"), int(diff[i])) for i in range(16) if diff[i] > 5]
    shrunk = [(COLOR_NAMES.get(i, f"c{i}"), int(-diff[i])) for i in range(16) if diff[i] < -5]

    grew.sort(key=lambda x: -x[1])
    shrunk.sort(key=lambda x: -x[1])

    desc = f"Last action: {n_changed}px changed ({pct:.1f}%) in {loc_y}-{loc_x} region (x:{x_min}-{x_max}, y:{y_min}-{y_max})."
    if grew:
        desc += f" Grew: {', '.join(f'{name}(+{n})' for name, n in grew[:3])}."
    if shrunk:
        desc += f" Shrunk: {', '.join(f'{name}(-{n})' for name, n in shrunk[:3])}."

    return desc


def _object_regions(frame: np.ndarray, min_size: int = 20) -> str:
    """Find distinct non-background colored regions."""
    h, w = frame.shape
    # Background = most common color
    bg = int(np.bincount(frame.ravel(), minlength=16).argmax())

    # Find connected regions of non-background
    visited = np.zeros_like(frame, dtype=bool)
    regions = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or frame[y, x] == bg:
                continue
            # BFS flood fill
            color = int(frame[y, x])
            queue = [(y, x)]
            pixels = []
            while queue:
                cy, cx = queue.pop(0)
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx]:
                    continue
                if frame[cy, cx] != color:
                    continue
                visited[cy, cx] = True
                pixels.append((cy, cx))
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((cy + dy, cx + dx))

            if len(pixels) >= min_size:
                pys = [p[0] for p in pixels]
                pxs = [p[1] for p in pixels]
                regions.append({
                    "color": COLOR_NAMES.get(color, f"c{color}"),
                    "size": len(pixels),
                    "center": ((min(pys) + max(pys)) // 2, (min(pxs) + max(pxs)) // 2),
                    "bbox": (min(pxs), min(pys), max(pxs), max(pys)),
                })

    if not regions:
        return "Objects: no distinct colored regions detected (frame may be uniform)."

    # Sort by size, limit to top 6
    regions.sort(key=lambda r: -r["size"])
    lines = [f"Objects ({len(regions)} regions, bg={COLOR_NAMES.get(bg, f'c{bg}')}):"]
    for r in regions[:6]:
        cx, cy = r["center"][1], r["center"][0]
        x1, y1, x2, y2 = r["bbox"]
        lines.append(
            f"  {r['color']}: {r['size']}px at center({cx},{cy}), "
            f"bbox({x1},{y1})-({x2},{y2})"
        )

    return "\n".join(lines)
