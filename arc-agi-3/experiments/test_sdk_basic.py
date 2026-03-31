#!/usr/bin/env python3
"""Basic SDK test without SAGE components."""
import time
import numpy as np
from arc_agi import Arcade

print("Testing ARC-AGI-3 SDK on Thor...")
print("=" * 60)

# Initialize
arcade = Arcade()
envs = arcade.get_environments()
print(f"Available environments: {len(envs)}")

# Pick first game
env_info = envs[0]
game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
print(f"Testing game: {game_id}")

# Make environment
env = arcade.make(game_id)
print(f"Environment created")

# Reset and get first frame
t0 = time.time()
frame_data = env.reset()
reset_time = time.time() - t0

grid = np.array(frame_data.frame)
print(f"Reset time: {reset_time:.3f}s")
print(f"Frame shape: {grid.shape}")
print(f"Available actions: {frame_data.available_actions}")
print(f"Levels: {frame_data.levels_completed}/{frame_data.win_levels}")

# Take one action
if frame_data.available_actions:
    action = frame_data.available_actions[0]
    t0 = time.time()
    frame_data = env.step(action)
    step_time = time.time() - t0
    grid = np.array(frame_data.frame)
    print(f"Step time: {step_time:.3f}s")
    print(f"New frame shape: {grid.shape}")

print("\nSDK test complete - environment works!")
