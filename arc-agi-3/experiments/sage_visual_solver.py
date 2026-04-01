#!/usr/bin/env python3
"""
Visual Rotation Solver for lp85
================================

Uses visual memory to:
1. Store initial frame when level starts
2. Store winning frames when level completes
3. Compare current frame to known states
4. Guide actions based on visual similarity
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction
from membot_cartridge import MembotCartridge
from sage_driver import SageDriver, grid_hash, ACTION_LABELS
import time
import argparse
import random

# Map integers to GameAction enums
INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


class VisualRotationSolver:
    """Solver that uses visual pattern matching for rotation puzzles."""

    def __init__(self, game_id: str, verbose: bool = False):
        self.game_id = game_id
        self.verbose = verbose
        arcade = Arcade()
        self.env = arcade.make(game_id)
        self.frame_data = None
        self.current_grid = None

        # Initialize driver with available actions
        self.driver = None  # Will init after first frame

        # Initialize cartridge
        self.cart = MembotCartridge(game_id)
        self.cart.read()

        # Visual memory
        self.level_initial_frames = {}  # level → initial frame
        self.level_frames_history = {}  # level → list of (step, frame, similarity_to_initial)

        # Performance tracking
        self.total_steps = 0
        self.max_level_reached = 0
        self.current_level = 1

    def get_current_frame(self) -> np.ndarray:
        """Extract current frame from arcade frame_data."""
        if self.frame_data is None:
            return np.zeros((64, 64), dtype=np.uint8)

        grid = np.array(self.frame_data.frame)
        if grid.ndim == 3:
            grid = grid[-1]  # Take last channel if 3D

        return grid

    def store_initial_frame(self, level: int, frame: np.ndarray):
        """Store the initial state of a level visually."""
        label = f"level_{level}_initial"
        self.level_initial_frames[level] = frame
        self.cart.store_frame_snapshot(
            label,
            frame,
            metadata={
                "level": level,
                "step": 0,
                "description": f"Initial state of level {level}"
            }
        )
        if self.verbose:
            print(f"   📸 Stored initial frame for level {level}")

    def store_winning_frame(self, level: int, frame: np.ndarray, step: int):
        """Store a winning frame when level completes."""
        label = f"level_{level}_win_step_{step}"
        self.cart.store_frame_snapshot(
            label,
            frame,
            metadata={
                "level": level,
                "step": step,
                "description": f"Winning state for level {level}",
                "is_win": True
            }
        )
        if self.verbose:
            print(f"   🏆 Stored winning frame for level {level} at step {step}")

    def compare_to_initial(self, level: int, current_frame: np.ndarray) -> float:
        """Compare current frame to initial frame of this level."""
        if level not in self.level_initial_frames:
            return 0.0

        initial = self.level_initial_frames[level]
        return self.cart.compute_visual_similarity(initial, current_frame)

    def find_similar_winning_frames(self, current_frame: np.ndarray, threshold: float = 0.8) -> list:
        """Find winning frames similar to current state."""
        results = self.cart.find_similar_snapshots(current_frame, threshold=threshold)

        # Filter to only winning frames
        winning = []
        for label, score in results:
            if "win" in label:
                winning.append((label, score))

        return winning

    def choose_action_visually(self, level: int, step: int, state_hash: str) -> int:
        """Choose action based on visual analysis."""
        current_frame = self.get_current_frame()

        # Compare to initial frame
        similarity_to_initial = self.compare_to_initial(level, current_frame)

        # Store frame in history
        if level not in self.level_frames_history:
            self.level_frames_history[level] = []
        self.level_frames_history[level].append((step, current_frame.copy(), similarity_to_initial))

        # Check if we're close to a known winning state
        similar_wins = self.find_similar_winning_frames(current_frame, threshold=0.85)

        if similar_wins and self.verbose:
            print(f"   🎯 Found {len(similar_wins)} similar winning states:")
            for label, score in similar_wins[:3]:
                print(f"      - {label}: {score:.3f}")

        # Strategy:
        # 1. If we've seen this exact configuration before, try a different action
        # 2. If we're very similar to initial (>0.95), we're going in circles
        # 3. Otherwise, use driver's action selection (exploration + exploitation)

        if similarity_to_initial > 0.95 and step > 10:
            # We're going in circles, force exploration with coordinates
            if self.verbose:
                print(f"   🔄 Too similar to initial ({similarity_to_initial:.3f}), forcing exploration")
            import random
            action = random.choice(self.driver.available)
            return self.driver._with_coordinates(action, current_frame)

        # Use driver's normal selection (which includes exhaustion/stagnation logic)
        return self.driver.select_action(state_hash, current_frame)

    def run(self, steps: int = 1000):
        """Run visual solver for specified steps."""
        print("=" * 70)
        print("VISUAL ROTATION SOLVER - lp85")
        print("=" * 70)
        print(f"\nGame: {self.game_id}")
        print(f"Max steps: {steps}")
        print(f"\nCartridge insights:")
        for insight in self.cart.data.get("strategic_insights", [])[:3]:
            print(f"  - {insight[:80]}...")
        print()

        # Initialize game
        self.frame_data = self.env.reset()
        self.current_grid = self.get_current_frame()

        # Initialize driver with available actions
        available = [a.value if hasattr(a, "value") else int(a)
                    for a in (self.frame_data.available_actions or [])]
        self.driver = SageDriver(available, self.game_id)

        # Detect initial level from frame_data
        self.current_level = self.frame_data.level if hasattr(self.frame_data, 'level') else 1

        # Store initial frame
        self.store_initial_frame(self.current_level, self.current_grid)

        # Main game loop
        for step in range(steps):
            # Detect level from frame_data
            level = self.frame_data.level if hasattr(self.frame_data, 'level') else 1

            # Check if level changed
            if level != self.current_level:
                print(f"\n⭐ LEVEL UP: {self.current_level} → {level}")

                # Store winning frame from previous level
                self.store_winning_frame(self.current_level, self.current_grid, step)

                # Store initial frame for new level
                self.store_initial_frame(level, self.current_grid)

                self.current_level = level
                self.max_level_reached = max(self.max_level_reached, level)

                # Update cartridge
                self.cart.update_best_score(level, step)

            # Get state hash for driver
            state_hash = grid_hash(self.current_grid)

            # Store before frame for action outcome
            before_frame = self.current_grid.copy()

            # Choose action using visual analysis
            action_result = self.choose_action_visually(level, step, state_hash)

            # Execute action — handle both simple (int) and coordinate ((int, dict)) formats
            if isinstance(action_result, tuple):
                action, data = action_result
                game_action = INT_TO_GAME_ACTION[action]
                self.frame_data = self.env.step(game_action, data=data)
            else:
                action = action_result
                game_action = INT_TO_GAME_ACTION[action]
                self.frame_data = self.env.step(game_action)

            # Get new grid
            self.current_grid = self.get_current_frame()
            new_state_hash = grid_hash(self.current_grid)

            # Check if grid changed
            changed = not np.array_equal(before_frame, self.current_grid)

            # Update driver with outcome
            self.driver.record(action, state_hash, changed)

            # Store visual outcome
            self.cart.store_action_visual_outcome(
                action,
                before_frame,
                self.current_grid,
                level,
                step
            )

            # Progress logging
            if step % 50 == 0 or step < 10:
                sim = self.compare_to_initial(level, self.current_grid)
                action_name = ACTION_LABELS.get(action, f"A{action}")
                print(f"Step {step:4d}: Level {level}, Action {action_name}, Similarity to initial: {sim:.3f}, Changed: {changed}")

            self.total_steps += 1

            # Check for stagnation
            if step > 200 and self.current_level == 1:
                print(f"\n⚠️  Still on level 1 after {step} steps - may need manual strategy")

        # Final report
        print("\n" + "=" * 70)
        print("VISUAL SOLVER COMPLETE")
        print("=" * 70)
        print(f"Total steps: {self.total_steps}")
        print(f"Max level: {self.max_level_reached}")
        print(f"\nVisual memory stored:")
        print(f"  - Snapshots: {len(self.cart.data['visual_memory']['snapshots'])}")
        print(f"  - Action outcomes: {len(self.cart.data['visual_memory']['action_outcomes'])}")

        # Update driver effectiveness in cartridge
        self.cart.update_action_effectiveness(self.driver)

        return self.max_level_reached, self.total_steps


def main():
    parser = argparse.ArgumentParser(description="Visual rotation solver for ARC-AGI-3")
    parser.add_argument("--game", default="lp85", help="Game ID (default: lp85)")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps (default: 1000)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    solver = VisualRotationSolver(args.game, verbose=args.verbose)
    max_level, total_steps = solver.run(steps=args.steps)

    print(f"\n📊 Final Score: Level {max_level} in {total_steps} steps")


if __name__ == "__main__":
    main()
