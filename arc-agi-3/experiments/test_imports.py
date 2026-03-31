#!/usr/bin/env python3
"""Test import speed."""
import sys
import time
sys.path.insert(0, ".")

print("Testing imports...")

t0 = time.time()
from sage.irp.plugins.grid_vision_irp import GridVisionIRP
print(f"GridVisionIRP import: {time.time() - t0:.3f}s")

t0 = time.time()
from sage.irp.plugins.game_action_effector import GameActionEffector
print(f"GameActionEffector import: {time.time() - t0:.3f}s")

print("\nNow testing GridVisionIRP initialization...")
t0 = time.time()
gv = GridVisionIRP({"entity_id": "test", "buffer_size": 20})
print(f"GridVisionIRP init: {time.time() - t0:.3f}s")

print("\nAll imports and init completed successfully!")
