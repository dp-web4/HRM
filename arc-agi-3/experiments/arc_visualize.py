#!/usr/bin/env python3
"""
ARC-AGI-3 Grid Visualization

Renders palette-indexed grids (0-15) to PNG files with proper ARC colors.
Optionally annotates cursor position, clicked objects, and KB highlights.

Usage:
    from arc_visualize import save_frame, save_annotated_frame

    save_frame(grid, "output/initial.png")
    save_annotated_frame(grid, "output/step_42.png",
                         cursor=(19, 28, "teal"),
                         clicked=[(35, 31), (19, 28)],
                         label="Level 0 | Step 42 | click(35,31) → 12px")
"""

import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ARC-AGI palette — 16 colors matching the game engine
ARC_PALETTE = {
    0:  (0,   0,   0),    # black
    1:  (0,   116, 217),  # blue
    2:  (255, 65,  54),   # red
    3:  (46,  204, 64),   # green
    4:  (255, 220, 0),    # yellow
    5:  (170, 170, 170),  # gray
    6:  (240, 18,  190),  # magenta
    7:  (255, 133, 27),   # orange
    8:  (0,   210, 211),  # cyan/teal
    9:  (127, 96,  0),    # brown
    10: (255, 182, 193),  # pink
    11: (128, 0,   0),    # maroon
    12: (128, 128, 0),    # olive
    13: (0,   0,   128),  # navy
    14: (0,   128, 128),  # teal
    15: (255, 255, 255),  # white
}

SCALE = 8  # Pixels per grid cell (64x64 grid → 512x512 image)


def grid_to_rgb(grid, scale=SCALE):
    """Convert palette-indexed grid to RGB numpy array.

    Args:
        grid: (H, W) array with values 0-15
        scale: upscale factor per cell

    Returns:
        (H*scale, W*scale, 3) uint8 RGB array
    """
    h, w = grid.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in ARC_PALETTE.items():
        mask = grid.astype(int) == idx
        rgb[mask] = color

    # Upscale with nearest-neighbor
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


def save_frame(grid, path, scale=SCALE):
    """Save grid as PNG with ARC colors.

    Args:
        grid: (H, W) palette-indexed array
        path: output file path
        scale: pixels per cell
    """
    if not HAS_PIL:
        return False
    rgb = grid_to_rgb(grid, scale)
    img = Image.fromarray(rgb)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return True


def save_annotated_frame(grid, path, cursor=None, clicked=None,
                         highlights=None, label=None, scale=SCALE):
    """Save grid as PNG with annotations overlaid.

    Args:
        grid: (H, W) palette-indexed array
        path: output file path
        cursor: (x, y, color_name) or None
        clicked: list of (x, y) positions that were just clicked
        highlights: list of (x, y, color) tuples for KB high-impact objects
        label: text label for top of image
        scale: pixels per cell
    """
    if not HAS_PIL:
        return False

    rgb = grid_to_rgb(grid, scale)

    # Add label bar at top if provided
    label_height = 20 if label else 0
    h, w = rgb.shape[:2]
    canvas = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    canvas[label_height:] = rgb
    if label:
        canvas[:label_height] = (40, 40, 40)  # dark gray bar

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    if label:
        try:
            draw.text((4, 2), label, fill=(255, 255, 255))
        except Exception:
            pass

    # Draw cursor crosshair
    if cursor:
        cx, cy = int(cursor[0] * scale), int(cursor[1] * scale) + label_height
        size = max(3, scale // 2)
        # Crosshair lines
        draw.line([(cx - size*3, cy), (cx - size, cy)], fill=(255, 255, 0), width=2)
        draw.line([(cx + size, cy), (cx + size*3, cy)], fill=(255, 255, 0), width=2)
        draw.line([(cx, cy - size*3), (cx, cy - size)], fill=(255, 255, 0), width=2)
        draw.line([(cx, cy + size), (cx, cy + size*3)], fill=(255, 255, 0), width=2)

    # Draw click markers (red circles)
    if clicked:
        for x, y in clicked:
            px, py = int(x * scale), int(y * scale) + label_height
            r = max(2, scale // 2)
            draw.ellipse([(px-r, py-r), (px+r, py+r)], outline=(255, 0, 0), width=2)

    # Draw KB highlight boxes (green)
    if highlights:
        for x, y, color in highlights:
            px, py = int(x * scale), int(y * scale) + label_height
            r = max(3, scale)
            draw.rectangle([(px-r, py-r), (px+r, py+r)], outline=(0, 255, 0), width=2)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return True


def save_game_summary(frames, path, labels=None, scale=4):
    """Save a grid of frames as one summary image.

    Args:
        frames: list of (H, W) grids
        path: output file path
        labels: optional list of labels per frame
        scale: pixels per cell (smaller for summaries)
    """
    if not HAS_PIL or not frames:
        return False

    n = len(frames)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    h, w = frames[0].shape[:2]
    ph, pw = h * scale, w * scale
    gap = 2
    label_h = 16

    canvas_w = cols * pw + (cols - 1) * gap
    canvas_h = rows * (ph + label_h) + (rows - 1) * gap
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, grid in enumerate(frames):
        row, col = divmod(i, cols)
        x0 = col * (pw + gap)
        y0 = row * (ph + label_h + gap) + label_h
        rgb = grid_to_rgb(grid, scale)
        canvas[y0:y0+ph, x0:x0+pw] = rgb

    img = Image.fromarray(canvas)
    if labels:
        draw = ImageDraw.Draw(img)
        for i, lbl in enumerate(labels):
            if not lbl:
                continue
            row, col = divmod(i, cols)
            x0 = col * (pw + gap)
            y0 = row * (ph + label_h + gap)
            try:
                draw.text((x0 + 2, y0), lbl[:30], fill=(200, 200, 200))
            except Exception:
                pass

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return True


if __name__ == "__main__":
    # Quick test: generate a random grid and save it
    test_grid = np.random.randint(0, 16, (64, 64), dtype=np.uint8)
    if save_frame(test_grid, "/tmp/arc_viz_test.png"):
        print("Test image saved to /tmp/arc_viz_test.png")
    else:
        print("PIL not available — install with: pip install Pillow")
