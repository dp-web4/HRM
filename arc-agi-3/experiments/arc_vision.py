#!/usr/bin/env python3
"""
ARC-AGI-3 Vision: Send game grids as images to multimodal models.

Converts 64x64 game grids to PNG images and sends them via Ollama's
multimodal API. Gemma 4 E4B can SEE the puzzle instead of reading
text descriptions of it.

ARC-AGI-3 uses 16 colors (0-15). We map them to distinct RGB values.
Grid is scaled up 4x (64→256) for model visibility. Most vision models
resize to 224-336px internally — 256x256 is sufficient.
"""

import io
import base64
import numpy as np

# ARC-AGI-3 SDK official palette (from arc_agi/rendering.py COLOR_MAP)
# There is ONE palette — the SDK's. All files use this. No exceptions.
ARC_PALETTE = {
    0:  (255, 255, 255),  # white
    1:  (204, 204, 204),  # off-white
    2:  (153, 153, 153),  # light-gray
    3:  (102, 102, 102),  # neutral
    4:  (51, 51, 51),     # off-black
    5:  (0, 0, 0),        # black
    6:  (229, 58, 163),   # magenta
    7:  (255, 123, 204),  # pink
    8:  (249, 60, 49),    # red
    9:  (30, 147, 255),   # blue
    10: (136, 216, 241),  # light-blue
    11: (255, 220, 0),    # yellow
    12: (255, 133, 27),   # orange
    13: (146, 18, 49),    # maroon
    14: (79, 204, 48),    # green
    15: (163, 86, 214),   # purple
}


def grid_to_image_b64(grid: np.ndarray, scale: int = 4) -> str:
    """Convert a 64x64 game grid to a base64-encoded PNG.

    Args:
        grid: 2D numpy array with values 0-15
        scale: Upscale factor (4 = 256x256 output)

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
                            scale: int = 4) -> str:
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
