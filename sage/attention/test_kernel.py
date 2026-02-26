#!/usr/bin/env python3
"""
Test script for Attention Kernel

Validates basic kernel functionality:
- State machine transitions
- ATP budget allocation
- Logging infrastructure
- Experience capture
- Sleep triggers
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.attention.kernel import AttentionKernel, ExperienceBuffer
from sage.attention.state import AttentionState
from sage.attention.atp_budget import ATPBudget


def test_state_machine():
    """Test state enumeration and transitions"""
    print("\n=== Testing State Machine ===")

    states = list(AttentionState)
    print(f"Available states: {[str(s) for s in states]}")

    assert AttentionState.IDLE in states
    assert AttentionState.FOCUS in states
    assert AttentionState.THINK in states
    assert AttentionState.ACT in states
    assert AttentionState.SLEEP in states
    assert AttentionState.RECOVER in states

    # Test string conversion
    assert str(AttentionState.IDLE) == "IDLE"
    assert AttentionState.from_string("FOCUS") == AttentionState.FOCUS

    print("✓ State machine working")


def test_atp_budget():
    """Test ATP budget allocation"""
    print("\n=== Testing ATP Budget ===")

    budget = ATPBudget(total_budget=1000.0)

    # Register plugins
    budget.register_plugin("vision", initial_trust=0.8)
    budget.register_plugin("language", initial_trust=0.6)
    budget.register_plugin("memory", initial_trust=0.7)

    # Allocate
    allocations = budget.allocate(["vision", "language", "memory"])
    print(f"Allocations: {allocations}")

    total_allocated = sum(allocations.values())
    print(f"Total allocated: {total_allocated:.2f} / {budget.total_budget}")

    assert abs(total_allocated - budget.total_budget) < 0.01  # Should allocate all budget

    # Vision should get most (highest trust)
    assert allocations["vision"] > allocations["language"]
    assert allocations["vision"] > allocations["memory"]

    print("✓ ATP budget working")


def test_experience_buffer():
    """Test experience capture and buffer"""
    print("\n=== Testing Experience Buffer ===")

    buffer = ExperienceBuffer(max_size=10)

    # Add experiences
    for i in range(15):
        buffer.add({
            'ts': i,
            'source': 'test',
            'salience': i * 0.1
        })

    print(f"Buffer size: {buffer.size} (max: 10)")
    print(f"Total salience: {buffer.salience_sum:.2f}")

    assert buffer.size == 10  # Should cap at max_size

    # Get top-k
    top_5 = buffer.get_top_k(5)
    print(f"Top 5 saliences: {[e['salience'] for e in top_5]}")

    assert len(top_5) == 5
    assert top_5[0]['salience'] >= top_5[1]['salience']  # Descending order

    print("✓ Experience buffer working")


async def test_kernel_basic():
    """Test basic kernel initialization and tick"""
    print("\n=== Testing Kernel Basics ===")

    # Clean up any previous test logs
    import shutil
    shutil.rmtree('test_logs', ignore_errors=True)

    config = {
        'atp_budget': 1000.0,
        'buffer_size': 100,
        'log_dir': 'test_logs',
        'tick_interval': 0.1,
        'sleep_policy': {
            'buffer_size': 10,
            'salience_sum': 5.0,
            'time_hours': 24,
            'idle_minutes': 1
        }
    }

    kernel = AttentionKernel(config)

    assert kernel.state == AttentionState.IDLE
    assert kernel.tick_count == 0

    # Run a few ticks
    print("Running 5 ticks...")
    for i in range(5):
        await kernel.tick()

    print(f"Completed {kernel.tick_count} ticks")
    assert kernel.tick_count == 5

    # Check logs were created
    log_path = Path('test_logs/attention_tick.jsonl')
    assert log_path.exists()

    # Read and validate log
    with open(log_path) as f:
        lines = f.readlines()
        assert len(lines) == 5  # One per tick

        # Parse first tick
        first_tick = json.loads(lines[0])
        print(f"First tick: {first_tick['state']}, duration: {first_tick['duration_ms']:.2f}ms")

        assert 'ts' in first_tick
        assert 'state' in first_tick
        assert 'duration_ms' in first_tick

    print("✓ Kernel basics working")

    # Cleanup
    import shutil
    shutil.rmtree('test_logs', ignore_errors=True)


async def test_kernel_run_limited():
    """Test kernel running for limited time"""
    print("\n=== Testing Kernel Run Loop ===")

    config = {
        'tick_interval': 0.05,  # Fast ticks for testing
        'log_dir': 'test_logs'
    }

    kernel = AttentionKernel(config)

    # Run for 1 second
    print("Running kernel for 1 second...")

    async def stop_after_delay():
        await asyncio.sleep(1.0)
        kernel.stop()

    # Start both tasks
    await asyncio.gather(
        kernel.run_forever(),
        stop_after_delay()
    )

    print(f"Kernel stopped after {kernel.tick_count} ticks")
    assert kernel.tick_count >= 15  # Should complete ~20 ticks in 1 second at 0.05s interval

    print("✓ Kernel run loop working")

    # Cleanup
    import shutil
    shutil.rmtree('test_logs', ignore_errors=True)


def main():
    """Run all tests"""
    print("=" * 60)
    print("SAGE Attention Kernel Test Suite")
    print("=" * 60)

    try:
        test_state_machine()
        test_atp_budget()
        test_experience_buffer()
        asyncio.run(test_kernel_basic())
        asyncio.run(test_kernel_run_limited())

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
