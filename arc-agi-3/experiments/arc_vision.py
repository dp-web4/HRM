#!/usr/bin/env python3
"""
ARC-AGI-3 Vision: Send game grids as images to multimodal models.

Converts 64x64 game grids to PNG images and sends them via Ollama's
multimodal API. Gemma 4 E4B can SEE the puzzle instead of reading
text descriptions of it.

ARC-AGI-3 uses 16 colors (0-15). We map them to distinct RGB values.
Grid is scaled up 8x (64→512) for model visibility.
"""

import io
import base64
import numpy as np

# ARC-AGI-3 color palette (16 colors, distinct RGB values)
ARC_PALETTE = {
    0:  (0, 0, 0),        # black
    1:  (0, 116, 217),    # blue
    2:  (255, 65, 54),    # red
    3:  (46, 204, 64),    # green
    4:  (255, 220, 0),    # yellow
    5:  (170, 170, 170),  # gray
    6:  (240, 18, 190),   # magenta
    7:  (255, 133, 27),   # orange
    8:  (0, 220, 220),    # cyan
    9:  (165, 103, 63),   # brown
    10: (255, 175, 200),  # pink
    11: (128, 0, 0),      # maroon
    12: (128, 128, 0),    # olive
    13: (0, 0, 128),      # navy
    14: (0, 128, 128),    # teal
    15: (255, 255, 255),  # white
}


def grid_to_image_b64(grid: np.ndarray, scale: int = 8) -> str:
    """Convert a 64x64 game grid to a base64-encoded PNG.

    Args:
        grid: 2D numpy array with values 0-15
        scale: Upscale factor (8 = 512x512 output)

    Returns:
        Base64-encoded PNG string for Ollama's images field
    """
    from PIL import Image

    h, w = grid.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for color_idx, rgb_val in ARC_PALETTE.items():
        mask = (grid.astype(int) == color_idx)
        rgb[mask] = rgb_val

    img = Image.fromarray(rgb)
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def grid_to_diff_image_b64(grid_before: np.ndarray, grid_after: np.ndarray,
                            scale: int = 8) -> str:
    """Create a side-by-side before/after comparison image.

    Left half: before state. Right half: after state.
    Changed cells highlighted with a red border.
    """
    from PIL import Image, ImageDraw

    h, w = grid_before.shape[:2]

    # Convert both grids to RGB
    def to_rgb(grid):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for ci, rv in ARC_PALETTE.items():
            rgb[grid.astype(int) == ci] = rv
        return rgb

    rgb_before = to_rgb(grid_before)
    rgb_after = to_rgb(grid_after)

    # Side by side with 4px gap
    gap = 4
    combined = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 128
    combined[:, :w] = rgb_before
    combined[:, w + gap:] = rgb_after

    img = Image.fromarray(combined)
    if scale > 1:
        img = img.resize(((w * 2 + gap) * scale, h * scale), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()
