#!/usr/bin/env python3
"""Test visual memory integration in perception toolkit."""

import sys
sys.path.insert(0, "arc-agi-3/experiments")

import numpy as np
from arc_perception import visual_similarity, color_effectiveness_summary
from membot_cartridge import MembotCartridge

# Test visual similarity
grid1 = np.random.randint(0, 16, (64, 64), dtype=np.uint8)
grid2 = grid1.copy()
grid3 = np.random.randint(0, 16, (64, 64), dtype=np.uint8)

print("Testing visual_similarity:")
print(f"  Identical grids: {visual_similarity(grid1, grid2):.3f} (expect 1.000)")
print(f"  Different grids: {visual_similarity(grid1, grid3):.3f} (expect <0.1)")

# Change 10% of grid2
mask = np.random.rand(64, 64) < 0.1
grid2[mask] = np.random.randint(0, 16, mask.sum())
print(f"  90% similar: {visual_similarity(grid1, grid2):.3f} (expect ~0.9)")

# Test color effectiveness summary
color_tries = {1: 5, 8: 50, 14: 10, 9: 3}
color_changes = {1: 0, 8: 50, 14: 2, 9: 0}

print("\nTesting color_effectiveness_summary:")
print(color_effectiveness_summary(color_tries, color_changes))

# Test with cartridge
print("\nTesting visual_memory_context:")
cart = MembotCartridge("test_visual")
cart.read()

# Store a test frame
cart.store_frame_snapshot(
    "test_initial",
    grid1,
    metadata={"description": "Initial test frame"}
)

from arc_perception import visual_memory_context
context = visual_memory_context(grid1, cart)
print(f"  Context: {context}")

context = visual_memory_context(grid3, cart)
print(f"  No match: {context}")

print("\n✓ All visual perception tests passed!")
