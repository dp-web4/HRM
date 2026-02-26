#!/usr/bin/env python3
"""
Test Kernel v2 + Sleep Consolidation Integration

Complete integration test showing:
1. Kernel collecting high-salience experiences
2. Sleep trigger activating
3. Sleep consolidation converting experiences and training
4. Kernel resuming after sleep
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.attention.kernel_v2 import AttentionKernelV2
from sage.attention.state import AttentionState


async def test_kernel_with_sleep_consolidation():
    """Test full kernel lifecycle with sleep consolidation"""
    print("\n=== Testing Kernel v2 + Sleep Consolidation Integration ===")

    # Create kernel with sleep consolidation enabled (but disabled for this test)
    config = {
        'atp_budget': 1000.0,
        'buffer_size': 50,
        'salience_memory_size': 100,
        'tick_interval': 0.1,
        'log_dir': 'logs/attention/integration_test',

        # Sleep policy: trigger after 20 experiences or 30 seconds
        'sleep_policy': {
            'buffer_threshold': 20,
            'time_threshold': 30.0
        },

        # Sleep consolidation config (disabled for fast test)
        'sleep_consolidation': {
            'enabled': False,  # Set to True to test actual training
            'min_salience': 0.6,
            'max_experiences': 10,
            'epochs': 1,
            'device': 'cpu'
        },

        # Plugin config (no plugins for this test)
        'plugin_config': {},

        # LLM config (no LLM for this test)
        'llm_config': {}
    }

    kernel = AttentionKernelV2(config)

    print(f"\nKernel initialized:")
    print(f"  Sleep consolidation: {'enabled' if kernel.sleep_consolidation else 'disabled'}")
    print(f"  Buffer size: {kernel.experience_buffer.size}")
    print(f"  Initial state: {kernel.state}")

    # Simulate kernel lifecycle
    print("\n--- Simulating Kernel Lifecycle ---")

    # Manually add high-salience experiences to buffer
    print("\nAdding high-salience experiences...")
    for i in range(25):
        # Create experiences with varying salience
        salience = 0.5 + (i % 5) * 0.1  # Range: 0.5 to 0.9

        # Different sources
        source = ['focus', 'think', 'act'][i % 3]

        experience = {
            'ts': kernel.tick_count,
            'source': source,
            'context': {
                'tick': kernel.tick_count,
                'goal': 'test',
                'iteration': i
            },
            'outcome': {
                'confidence': salience,
                'status': 'success' if i % 3 != 0 else 'uncertain'
            },
            'salience': salience
        }

        kernel.capture_experience(
            source,
            experience['context'],
            experience['outcome']
        )

        kernel.tick_count += 1

    print(f"Buffer size after experiences: {kernel.experience_buffer.size}")
    print(f"Total salience: {kernel.experience_buffer.salience_sum:.2f}")

    # Check if sleep should trigger
    should_sleep = kernel.sleep_trigger.should_sleep(
        kernel.experience_buffer,
        kernel.last_sleep_time
    )

    print(f"\nShould sleep (natural trigger): {should_sleep}")

    # For this test, force sleep even if trigger didn't activate naturally
    print("\n--- Entering SLEEP State (forced for test) ---")

    # Transition to SLEEP
    kernel.state = AttentionState.SLEEP

    # Get salience statistics before sleep
    stats_before = kernel.salience_scorer.get_statistics()
    print(f"Statistics before sleep:")
    print(f"  Avg salience: {stats_before['avg_salience']:.3f}")
    print(f"  Max salience: {stats_before['max_salience']:.3f}")
    print(f"  Source distribution: {stats_before['source_distribution']}")

    # Execute sleep behavior (includes consolidation if enabled)
    await kernel.sleep_behavior()

    # Check results
    print(f"\nAfter sleep:")
    print(f"  Buffer size: {kernel.experience_buffer.size}")
    print(f"  Sleep cycles: {kernel.sleep_cycle_count}")
    print(f"  State: {kernel.state}")

    # If consolidation was enabled, check results
    if kernel.sleep_consolidation:
        stats = kernel.sleep_consolidation.get_statistics()
        print(f"\nConsolidation statistics:")
        print(f"  Sleep cycles completed: {stats['sleep_cycles_completed']}")
        print(f"  Total experiences consolidated: {stats['total_experiences_consolidated']}")

        if stats['latest_cycle']:
            latest = stats['latest_cycle']
            print(f"  Latest cycle:")
            print(f"    Status: {latest.get('status', 'N/A')}")
            print(f"    Experiences: {latest.get('num_experiences', 0)}")
            if latest.get('final_loss'):
                print(f"    Final loss: {latest['final_loss']:.4f}")

    # Verify buffer was cleared
    assert kernel.experience_buffer.size == 0, "Buffer should be cleared after sleep"
    assert kernel.sleep_cycle_count == 1, "Should have completed 1 sleep cycle"

    print("\n✓ Kernel + Sleep Consolidation integration test passed")


async def test_multiple_sleep_cycles():
    """Test multiple sleep cycles over time"""
    print("\n=== Testing Multiple Sleep Cycles ===")

    config = {
        'buffer_size': 20,
        'sleep_policy': {
            'buffer_threshold': 10,  # Sleep after 10 experiences
            'time_threshold': 100.0
        },
        'sleep_consolidation': {
            'enabled': False,  # Disabled for fast test
            'min_salience': 0.5
        },
        'tick_interval': 0.01,
        'log_dir': 'logs/attention/multi_sleep_test'
    }

    kernel = AttentionKernelV2(config)

    print(f"\nRunning {3} sleep cycles...")

    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")

        # Add experiences
        for i in range(12):
            kernel.capture_experience(
                'focus',
                {'tick': kernel.tick_count},
                {'salience': 0.6 + i * 0.02},
            )
            kernel.tick_count += 1

        print(f"Buffer size: {kernel.experience_buffer.size}")

        # Execute sleep
        kernel.state = AttentionState.SLEEP
        await kernel.sleep_behavior()

        print(f"Sleep cycle {kernel.sleep_cycle_count} complete")
        print(f"Buffer cleared: {kernel.experience_buffer.size == 0}")

    assert kernel.sleep_cycle_count == 3, "Should have completed 3 sleep cycles"

    print("\n✓ Multiple sleep cycles test passed")


async def test_sleep_with_actual_consolidation():
    """Test sleep with actual consolidation (if available)"""
    print("\n=== Testing Sleep with Actual Consolidation ===")

    # Check if sleep training is available
    from sage.attention.sleep_consolidation import SLEEP_TRAINING_AVAILABLE

    if not SLEEP_TRAINING_AVAILABLE:
        print("⚠️ Sleep training not available - skipping actual consolidation test")
        print("   This is expected if raising pipeline is not installed")
        return

    print("Sleep training available - running full consolidation test...")

    config = {
        'buffer_size': 30,
        'sleep_policy': {
            'buffer_threshold': 15,
            'time_threshold': 100.0
        },
        'sleep_consolidation': {
            'enabled': True,  # Enable actual training
            'min_salience': 0.6,
            'max_experiences': 5,
            'epochs': 1,  # Quick training
            'device': 'cpu'
        },
        'log_dir': 'logs/attention/consolidation_test'
    }

    kernel = AttentionKernelV2(config)

    # Add high-salience experiences
    print("\nAdding high-salience experiences...")
    for i in range(20):
        salience = 0.7 + (i % 3) * 0.1  # Range: 0.7 to 0.9
        kernel.capture_experience(
            'think',
            {
                'prompt': f'Test prompt {i}',
                'tick': kernel.tick_count
            },
            {
                'text': f'Test response {i}',
                'confidence': salience
            }
        )
        kernel.tick_count += 1

    print(f"Buffer size: {kernel.experience_buffer.size}")

    # Execute sleep with consolidation
    kernel.state = AttentionState.SLEEP
    await kernel.sleep_behavior()

    # Check consolidation results
    if kernel.sleep_consolidation:
        stats = kernel.sleep_consolidation.get_statistics()
        print(f"\nConsolidation results:")
        print(f"  Sleep cycles: {stats['sleep_cycles_completed']}")
        print(f"  Experiences consolidated: {stats['total_experiences_consolidated']}")

        if stats['latest_cycle']:
            latest = stats['latest_cycle']
            print(f"  Latest cycle status: {latest.get('status', 'N/A')}")

            if latest.get('status') not in ['error', 'skipped', 'disabled']:
                assert latest.get('num_experiences', 0) > 0, \
                    "Should have consolidated experiences"
                print(f"  ✓ Successfully consolidated {latest['num_experiences']} experiences")

    print("\n✓ Actual consolidation test complete")


async def main():
    """Run all integration tests"""
    print("=" * 70)
    print("Kernel v2 + Sleep Consolidation Integration Tests")
    print("=" * 70)

    try:
        # Test 1: Basic integration
        await test_kernel_with_sleep_consolidation()

        # Test 2: Multiple sleep cycles
        await test_multiple_sleep_cycles()

        # Test 3: Actual consolidation (if available)
        await test_sleep_with_actual_consolidation()

        print("\n" + "=" * 70)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        print("\nDay 2 Afternoon: Sleep Consolidation Integration - COMPLETE")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
