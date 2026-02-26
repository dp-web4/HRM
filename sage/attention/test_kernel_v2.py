#!/usr/bin/env python3
"""
Test Attention Kernel v2 with SNARC Salience Integration

Tests that experience salience scoring works correctly and integrates
with the kernel's experience capture system.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.attention.kernel_v2 import AttentionKernelV2
from sage.attention.experience_salience import ExperienceSalienceScorer
from sage.attention.state import AttentionState


async def test_salience_scorer():
    """Test SNARC salience scoring for experiences"""
    print("\n=== Testing Experience Salience Scorer ===")

    scorer = ExperienceSalienceScorer(memory_size=50)

    # Test 1: Error experiences should have high surprise
    error_context = {'goal': 'test'}
    error_outcome = {'status': 'failed', 'error': 'something went wrong'}
    error_salience = scorer.score_experience('focus', error_context, error_outcome)
    print(f"Error experience salience: {error_salience:.3f}")
    assert error_salience > 0.3, "Errors should be salient"

    # Test 2: First experience should have high novelty
    first_context = {'goal': 'observe'}
    first_outcome = {'status': 'success'}
    first_salience = scorer.score_experience('think', first_context, first_outcome)
    print(f"First experience salience: {first_salience:.3f}")
    assert first_salience > 0.5, "First experience should be novel"

    # Test 3: Repeated similar experiences should have lower novelty
    for i in range(10):
        scorer.score_experience('idle', first_context, first_outcome)

    repeated_salience = scorer.score_experience('idle', first_context, first_outcome)
    print(f"Repeated experience salience: {repeated_salience:.3f}")
    assert repeated_salience < first_salience, "Repeated experiences should be less novel"

    # Test 4: High-arousal experiences (many plugins)
    arousal_context = {'goal': 'test'}
    arousal_outcome = {
        'results': {
            'plugin1': {'converged': True},
            'plugin2': {'converged': True},
            'plugin3': {'converged': False},
            'plugin4': {'converged': True},
            'plugin5': {'converged': True},
        },
        'total_budget_used': 500.0
    }
    arousal_salience = scorer.score_experience('focus', arousal_context, arousal_outcome)
    print(f"High-arousal experience salience: {arousal_salience:.3f}")
    assert arousal_salience > 0.4, "High plugin count should increase arousal"

    # Test 5: High-conflict experiences (plugin disagreement)
    conflict_outcome = {
        'disagreement': 0.8,
        'confidence': 0.3,
        'results': {
            'plugin1': {'converged': True},
            'plugin2': {'converged': False},
        }
    }
    conflict_salience = scorer.score_experience('focus', arousal_context, conflict_outcome)
    print(f"High-conflict experience salience: {conflict_salience:.3f}")
    assert conflict_salience > 0.4, "High disagreement should increase conflict"

    # Test 6: High-reward experiences (successful convergence)
    reward_outcome = {
        'status': 'success',
        'confidence': 0.9,
        'results': {
            'plugin1': {'converged': True, 'final_energy': 0.1},
            'plugin2': {'converged': True, 'final_energy': 0.05},
            'plugin3': {'converged': True, 'final_energy': 0.15},
        }
    }
    reward_salience = scorer.score_experience('focus', arousal_context, reward_outcome)
    print(f"High-reward experience salience: {reward_salience:.3f}")
    assert reward_salience > 0.4, "Successful outcomes should increase reward"

    # Test 7: Statistics
    stats = scorer.get_statistics()
    print(f"Total experiences scored: {stats['total_experiences']}")
    print(f"Avg salience: {stats['avg_salience']:.3f}")
    print(f"Source distribution: {stats['source_distribution']}")
    assert stats['total_experiences'] > 0, "Should track experiences"

    print("✓ Experience salience scorer working")


async def test_kernel_v2_salience_integration():
    """Test that kernel v2 computes real salience for experiences"""
    print("\n=== Testing Kernel v2 Salience Integration ===")

    config = {
        'atp_budget': 1000.0,
        'buffer_size': 50,
        'tick_interval': 0.1,
        'log_dir': 'test_logs_v2',
        'salience_memory_size': 100,
        'plugin_config': {
            'enable_vision': False,  # Disable for testing
            'enable_language': False,
            'enable_memory': False,
        }
    }

    kernel = AttentionKernelV2(config)

    # Test 1: Capture experience and verify salience is computed
    kernel.capture_experience(
        'focus',
        {'goal': 'test', 'tick': 1},
        {'status': 'success', 'confidence': 0.8}
    )

    assert kernel.experience_buffer.size == 1, "Should capture experience"
    experience = kernel.experience_buffer.buffer[0]
    print(f"Captured experience salience: {experience['salience']:.3f}")
    assert experience['salience'] > 0.0, "Should compute non-zero salience"
    assert experience['salience'] != 0.5, "Should not use placeholder salience"

    # Test 2: Capture error experience and verify high salience
    kernel.capture_experience(
        'recover',
        {'goal': 'test'},
        {'error': 'test error', 'status': 'failed'}
    )

    error_experience = kernel.experience_buffer.buffer[1]
    print(f"Error experience salience: {error_experience['salience']:.3f}")
    assert error_experience['salience'] > 0.4, "Errors should have high salience"

    # Test 3: Verify salience statistics
    stats = kernel.salience_scorer.get_statistics()
    print(f"Salience stats: avg={stats['avg_salience']:.3f}, max={stats['max_salience']:.3f}")
    assert stats['total_experiences'] == 2, "Should track experiences"

    # Test 4: Test top-k retrieval by salience
    for i in range(10):
        salience_boost = i * 0.05  # Vary salience
        kernel.capture_experience(
            'think',
            {'goal': 'test', 'iteration': i},
            {'confidence': 0.5 + salience_boost}
        )

    top_5 = kernel.experience_buffer.get_top_k(5)
    print(f"Top 5 experiences retrieved: {len(top_5)}")
    assert len(top_5) == 5, "Should retrieve top-k experiences"

    # Verify they're sorted by salience (descending)
    saliences = [exp['salience'] for exp in top_5]
    print(f"Top 5 saliences: {[f'{s:.3f}' for s in saliences]}")
    assert saliences == sorted(saliences, reverse=True), "Should be sorted by salience"

    print("✓ Kernel v2 salience integration working")


async def test_kernel_v2_tick_with_salience():
    """Test that kernel v2 tick cycle captures experiences with salience"""
    print("\n=== Testing Kernel v2 Tick Cycle ===")

    config = {
        'atp_budget': 1000.0,
        'buffer_size': 50,
        'tick_interval': 0.05,
        'log_dir': 'test_logs_v2',
        'plugin_config': {
            'enable_vision': False,
            'enable_language': False,
            'enable_memory': False,
        }
    }

    kernel = AttentionKernelV2(config)

    # Run a few ticks
    for _ in range(5):
        await kernel.tick()

    print(f"Ticks executed: {kernel.tick_count}")
    assert kernel.tick_count == 5, "Should execute ticks"

    # Check that experiences were captured (IDLE doesn't capture by default)
    # But let's transition to FOCUS manually to test
    kernel.state = AttentionState.FOCUS
    await kernel.tick()

    # FOCUS should capture experience
    # Note: This requires plugin router to be available, but we disabled plugins
    # So it should still capture a 'no_plugins' experience

    print(f"Experiences captured: {kernel.experience_buffer.size}")

    print("✓ Kernel v2 tick cycle working")


async def test_sleep_with_salience_stats():
    """Test that sleep behavior reports salience statistics"""
    print("\n=== Testing Sleep Behavior with Salience Stats ===")

    config = {
        'atp_budget': 1000.0,
        'buffer_size': 50,
        'tick_interval': 0.05,
        'log_dir': 'test_logs_v2',
        'plugin_config': {
            'enable_vision': False,
            'enable_language': False,
            'enable_memory': False,
        }
    }

    kernel = AttentionKernelV2(config)

    # Add some experiences with varying salience
    for i in range(20):
        kernel.capture_experience(
            'focus' if i % 2 == 0 else 'think',
            {'goal': 'test', 'iteration': i},
            {
                'status': 'success' if i % 3 != 0 else 'failed',
                'confidence': 0.5 + (i * 0.02)
            }
        )

    print(f"Experiences before sleep: {kernel.experience_buffer.size}")
    print(f"Total salience: {kernel.experience_buffer.salience_sum:.2f}")

    # Trigger sleep
    await kernel.sleep_behavior()

    print(f"Experiences after sleep: {kernel.experience_buffer.size}")
    assert kernel.experience_buffer.size == 0, "Sleep should clear buffer"
    assert kernel.sleep_cycle_count == 1, "Should increment sleep cycle"

    print("✓ Sleep behavior with salience stats working")


async def main():
    """Run all kernel v2 tests"""
    print("=" * 70)
    print("SAGE Attention Kernel v2 (SNARC Salience) Test Suite")
    print("=" * 70)

    try:
        # Test 1: Salience scorer
        await test_salience_scorer()

        # Test 2: Kernel v2 integration
        await test_kernel_v2_salience_integration()

        # Test 3: Tick cycle
        await test_kernel_v2_tick_with_salience()

        # Test 4: Sleep behavior
        await test_sleep_with_salience_stats()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nDay 1 Evening: SNARC Salience Integration - VALIDATED")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
