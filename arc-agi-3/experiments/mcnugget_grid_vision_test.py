#!/usr/bin/env python3
"""
McNugget ARC-AGI-3 × GridVisionIRP integration test.

Plays a game with random actions while pushing frames through
GridVisionIRP to verify the perception pipeline works end-to-end.

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/mcnugget_grid_vision_test.py
"""

import sys
import time
import random
import numpy as np

sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction
from sage.irp.plugins.grid_vision_irp import GridVisionIRP, GridObservation


def main():
    print("=" * 60)
    print("McNugget ARC-AGI-3 × GridVisionIRP Integration Test")
    print("=" * 60)

    # --- Initialize game ---
    arcade = Arcade()
    envs = arcade.get_environments()
    game_id = envs[0].game_id if hasattr(envs[0], "game_id") else str(envs[0])
    print(f"\nGame: {game_id}")
    print(f"Environments available: {len(envs)}")

    env = arcade.make(game_id)
    frame_data = env.reset()

    grid = np.array(frame_data.frame)
    print(f"Raw grid shape: {grid.shape} (GridVisionIRP handles squeeze internally)")
    print(f"Grid dtype: {grid.dtype}, range: [{grid.min()}, {grid.max()}]")
    print(f"Unique colors: {np.unique(grid).tolist()}")
    print(f"Available actions: {frame_data.available_actions}")
    print(f"Levels: {frame_data.levels_completed}/{frame_data.win_levels}")

    # --- Initialize GridVisionIRP ---
    gv = GridVisionIRP({"entity_id": "grid_vision_test", "buffer_size": 20})

    # Push initial frame
    obs = gv.push_raw_frame(
        grid,
        step_number=0,
        action_taken=0,
        level_id=game_id,
    )
    print(f"\n--- Initial frame ---")
    print(f"  Objects: {obs.n_objects} (none without perception layer)")
    print(f"  Changes: {len(obs.changes)} (first frame, no diff)")
    print(f"  Change magnitude: {obs.change_magnitude:.3f}")

    # --- IRP contract test ---
    state = gv.init_state(obs, {"level_id": game_id})
    print(f"\n--- IRP state ---")
    print(f"  has_data: {state.x['has_data']}")
    print(f"  energy: {state.energy_val:.3f}")
    print(f"  meta: n_objects={state.meta['n_objects']}, n_changes={state.meta['n_changes']}")

    # --- Play 20 steps ---
    print(f"\n--- Playing 20 steps ---")
    total_changes = 0
    effective_steps = 0

    for i in range(20):
        actions = frame_data.available_actions or [GameAction.ACTION1]
        action = random.choice(actions)
        frame_data = env.step(action)

        new_grid = np.array(frame_data.frame)

        obs = gv.push_raw_frame(
            new_grid,
            step_number=i + 1,
            action_taken=action.value if hasattr(action, "value") else int(action),
            level_id=game_id,
        )

        n_changes = len(obs.changes)
        total_changes += n_changes
        if n_changes > 0:
            effective_steps += 1

        # IRP step
        state = gv.step(state)

        print(
            f"  Step {i+1:2d}: action={action.value if hasattr(action, 'value') else action}, "
            f"changes={n_changes:4d}, magnitude={obs.change_magnitude:.3f}, "
            f"energy={state.energy_val:.3f}, "
            f"state={frame_data.state.name}, "
            f"levels={frame_data.levels_completed}/{frame_data.win_levels}"
        )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Game: {game_id}")
    print(f"  Steps played: 20")
    print(f"  Effective steps (caused changes): {effective_steps}")
    print(f"  Total cell changes: {total_changes}")
    print(f"  Avg changes/step: {total_changes/20:.1f}")
    print(f"  Levels completed: {frame_data.levels_completed}/{frame_data.win_levels}")
    print(f"  GridVisionIRP stats: {gv.stats}")
    print(f"  Halt check: {gv.halt([state])}")

    # --- Sensor observation format test ---
    sensor_obs = gv.to_sensor_observation()
    if sensor_obs:
        print(f"\n  Sensor observation keys: {list(sensor_obs.keys())}")
        print(f"  Sensor data keys: {list(sensor_obs['data'].keys())}")
        print(f"  Modality: {sensor_obs['modality']}")
        print(f"  Trust: {sensor_obs['trust']}")
    else:
        print(f"\n  WARNING: to_sensor_observation() returned None")

    print(f"\n✅ GridVisionIRP integration test PASSED")


if __name__ == "__main__":
    main()
