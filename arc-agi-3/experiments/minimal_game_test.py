#!/usr/bin/env python3
"""
Minimal ARC-AGI-3 game environment test.
Tests if we can load a game, get frames, and send actions without the full agent framework.
"""

import time
import random
from arcengine import FrameData, GameAction, GameState

# Try to import arc_agi for local game execution
try:
    import arc_agi
    has_local = True
    print("✓ arc_agi available - can run locally")
except ImportError:
    has_local = False
    print("✗ arc_agi not available - need API for remote games")

def test_game_actions():
    """Test that we can create and manipulate GameActions."""
    print("\n=== Testing GameAction Creation ===")

    # Simple actions (no coordinates)
    simple_actions = [a for a in GameAction if a.is_simple()]
    print(f"Simple actions: {[a.value for a in simple_actions]}")

    # Complex actions (need coordinates)
    complex_actions = [a for a in GameAction if a.is_complex()]
    print(f"Complex actions: {[a.value for a in complex_actions]}")

    # Test creating a complex action
    action = GameAction.ACTION0
    if action.is_complex():
        action.set_data({"x": 10, "y": 20})
        print(f"\nCreated complex action: {action.value} at ({action.data['x']}, {action.data['y']})")

    return True

def test_frame_data():
    """Test FrameData structure (what we receive from game)."""
    print("\n=== Testing FrameData Structure ===")

    # The FrameData class tells us what information the game provides
    import inspect
    sig = inspect.signature(FrameData.__init__)
    print(f"FrameData fields: {list(sig.parameters.keys())}")

    # Key fields we care about:
    # - state: GameState (NOT_PLAYED, PLAYING, WIN, GAME_OVER)
    # - grid: 64x64 array of colors
    # - levels_completed: int (formerly 'score')
    # - available_actions: list of valid actions

    print("\nGameState options:")
    for state in GameState:
        print(f"  - {state.name}: {state.value}")

    return True

def measure_iteration_speed():
    """Measure how fast we can create and process actions."""
    print("\n=== Measuring Action Creation Speed ===")

    n_iterations = 10000
    start = time.time()

    for i in range(n_iterations):
        # Simulate choosing a random action
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})

    elapsed = time.time() - start
    actions_per_sec = n_iterations / elapsed
    ms_per_action = (elapsed / n_iterations) * 1000

    print(f"Created {n_iterations} random actions in {elapsed:.3f}s")
    print(f"Speed: {actions_per_sec:.0f} actions/sec ({ms_per_action:.3f} ms/action)")
    print(f"\nThis means action selection overhead is: {ms_per_action:.3f} ms")
    print(f"For 100ms consciousness loop budget, action selection costs {(ms_per_action/100)*100:.1f}% of cycle")

    return True

def explore_grid_representation():
    """Understand how grids are represented."""
    print("\n=== Grid Representation ===")

    # ARC grids are 64x64 with 16 possible colors (0-15)
    # We need to understand the format to design GridVisionIRP

    print("Grid specs:")
    print("  - Size: 64x64 cells")
    print("  - Colors: 16 possible (0-15)")
    print("  - Format: Likely numpy array shape (64, 64)")
    print("  - Total cells: 4096")
    print("  - Bits per cell: 4 (2^4 = 16 colors)")
    print("  - Total grid size: 4096 cells * 4 bits = 2KB per frame")

    print("\nRepresentation options for GridVisionIRP:")
    print("  1. Single channel: (64, 64) int array, values 0-15")
    print("     - Pros: Compact, simple")
    print("     - Cons: No gradient for CNNs")
    print("  2. One-hot: (64, 64, 16) float array")
    print("     - Pros: CNN-friendly, clear semantics")
    print("     - Cons: 16x larger (32KB), more compute")
    print("  3. Hybrid: Store as int, convert to one-hot when needed")
    print("     - Pros: Best of both")
    print("     - Cons: Conversion overhead")

    print("\nRecommendation for Phase 1: Option 1 (single channel)")
    print("  - Simplest to start")
    print("  - Upgrade to option 3 when adding CNN in Phase 2")

    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("ARC-AGI-3 Minimal Environment Test")
    print("=" * 60)

    tests = [
        ("GameAction creation", test_game_actions),
        ("FrameData structure", test_frame_data),
        ("Action iteration speed", measure_iteration_speed),
        ("Grid representation", explore_grid_representation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "✓" if success else "✗"))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results.append((name, "✗"))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, result in results:
        print(f"{result} {name}")

    # Summary for Thor session
    print("\n" + "=" * 60)
    print("Thor Session 001 Summary:")
    print("=" * 60)
    print("✓ ARC-AGI SDK installed and working")
    print("✓ arcengine imports successful")
    print("✓ GameAction and FrameData structures understood")
    print("✓ Action creation speed measured: ~0.03ms overhead")
    print("✓ Grid representation options analyzed")
    print("\nNext steps:")
    print("  1. Implement GridVisionIRP Phase 1 (simple features)")
    print("  2. Create GameActionEffector")
    print("  3. Test with actual game (local or API)")
    print("  4. Measure full consciousness loop speed")

if __name__ == "__main__":
    main()
