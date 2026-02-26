#!/usr/bin/env python3
"""
Test SAGE unified entry point.

Validates that SAGE.create() and sage.run() work as expected.
"""

import asyncio
import sys
from pathlib import Path

# Add HRM to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_basic_import():
    """Test that SAGE can be imported"""
    print("\n=== Test 1: Basic Import ===")

    from sage import SAGE

    print("✓ SAGE import successful")
    print(f"  Version: {SAGE.__module__}")


async def test_create_default():
    """Test creating SAGE with defaults"""
    print("\n=== Test 2: Create with Defaults ===")

    from sage import SAGE

    # Create with all defaults (mock sensors, mock LLM)
    sage = SAGE.create()

    print("✓ SAGE.create() successful")
    print(f"  Type: {type(sage).__name__}")
    print(f"  Has run method: {hasattr(sage, 'run')}")
    print(f"  Has state property: {hasattr(sage, 'state')}")


async def test_create_with_options():
    """Test creating SAGE with various options"""
    print("\n=== Test 3: Create with Options ===")

    from sage import SAGE

    # Test different configurations
    configs = [
        {"name": "Default", "kwargs": {}},
        {"name": "PolicyGate enabled", "kwargs": {"use_policy_gate": True}},
        {"name": "Real sensors", "kwargs": {"use_real_sensors": True}},
        {"name": "Custom config", "kwargs": {"config": {"metabolic_params": {"base_atp": 2000.0}}}}
    ]

    for cfg in configs:
        try:
            sage = SAGE.create(**cfg["kwargs"])
            print(f"✓ {cfg['name']}: Created successfully")
        except Exception as e:
            print(f"✗ {cfg['name']}: Failed - {e}")
            raise


async def test_run_one_cycle():
    """Test running SAGE for 1 cycle"""
    print("\n=== Test 4: Run One Cycle ===")

    from sage import SAGE

    sage = SAGE.create()

    # Run for just 1 cycle
    stats = await sage.run(max_cycles=1)

    print("✓ sage.run(max_cycles=1) successful")
    print(f"  Cycles completed: {stats['cycles_completed']}")
    print(f"  Duration: {stats['duration_seconds']:.2f}s")
    print(f"  Final state: {stats['final_state']}")

    assert stats['cycles_completed'] == 1, "Should complete exactly 1 cycle"


async def test_run_multiple_cycles():
    """Test running SAGE for multiple cycles"""
    print("\n=== Test 5: Run Multiple Cycles ===")

    from sage import SAGE

    sage = SAGE.create()

    # Run for 5 cycles
    stats = await sage.run(max_cycles=5)

    print("✓ sage.run(max_cycles=5) successful")
    print(f"  Cycles completed: {stats['cycles_completed']}")
    print(f"  Duration: {stats['duration_seconds']:.2f}s")
    print(f"  Final state: {stats['final_state']}")

    assert stats['cycles_completed'] == 5, "Should complete exactly 5 cycles"


async def test_get_statistics():
    """Test getting SAGE statistics"""
    print("\n=== Test 6: Get Statistics ===")

    from sage import SAGE

    sage = SAGE.create()

    # Run a few cycles
    await sage.run(max_cycles=3)

    # Get statistics
    stats = sage.get_statistics()

    print("✓ sage.get_statistics() successful")
    print(f"  Keys: {list(stats.keys())}")
    if 'metabolic_state' in stats:
        print(f"  Metabolic state: {stats['metabolic_state']}")
    if 'experience_count' in stats:
        print(f"  Experiences: {stats['experience_count']}")


async def test_readme_example():
    """Test the example from the README/docstring"""
    print("\n=== Test 7: README Example ===")

    from sage import SAGE

    # Example from docstring
    sage = SAGE.create()
    stats = await sage.run(max_cycles=2)

    print("✓ README example works")
    print(f"  Cycles: {stats['cycles_completed']}")


async def main():
    """Run all tests"""
    print("=" * 70)
    print("SAGE Unified Entry Point Tests")
    print("=" * 70)

    tests = [
        test_basic_import,
        test_create_default,
        test_create_with_options,
        test_run_one_cycle,
        test_run_multiple_cycles,
        test_get_statistics,
        test_readme_example
    ]

    failed = 0

    for test in tests:
        try:
            await test()
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    if failed == 0:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nSAGE unified entry point: OPERATIONAL")
    else:
        print(f"✗ {failed} TEST(S) FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
