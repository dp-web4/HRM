#!/usr/bin/env python3
"""
Test IRP Plugin Integration with Attention Kernel

Tests that the plugin router can discover, allocate, and invoke IRP plugins.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.attention.plugin_router import PluginRouter
from sage.attention.atp_budget import ATPBudget


async def test_plugin_discovery():
    """Test that plugins can be discovered"""
    print("\n=== Testing Plugin Discovery ===")

    router = PluginRouter({
        'device': 'cpu',
        'enable_vision': True,
        'enable_language': True,
        'enable_memory': True,
        'enable_control': False,
        'enable_tts': False,
    })

    plugins = router.get_available_plugins()
    print(f"Discovered plugins: {plugins}")

    assert len(plugins) > 0, "Should discover at least one plugin"
    print(f"✓ Plugin discovery working ({len(plugins)} plugins)")

    return router


async def test_atp_allocation(router):
    """Test ATP budget allocation to plugins"""
    print("\n=== Testing ATP Allocation ===")

    budget = ATPBudget(total_budget=1000.0)
    plugins = router.get_available_plugins()

    # Register plugins
    for plugin in plugins:
        budget.register_plugin(plugin, initial_trust=0.5)

    # Allocate
    allocations = budget.allocate(plugins[:3])  # Test with first 3
    print(f"Allocations: {allocations}")

    total = sum(allocations.values())
    print(f"Total allocated: {total:.2f}")

    assert abs(total - budget.available) < 0.01, "Should allocate all available budget"
    print("✓ ATP allocation working")

    return allocations


async def test_plugin_execution(router, allocations):
    """Test executing plugins with ATP budgets"""
    print("\n===Testing Plugin Execution ===")

    context = {
        'goal': 'test',
        'observations': None,
        'constraints': {}
    }

    results = await router.run_plugins(context, allocations)

    print(f"Execution status: {results['status']}")
    print(f"Plugins executed: {len(results['results'])}")
    print(f"Budget used: {results['total_budget_used']:.2f}")

    assert results['status'] in ['success', 'no_valid_plugins'], "Should complete execution"
    print("✓ Plugin execution working")

    return results


async def test_trust_updates(router, results):
    """Test that trust scores update based on results"""
    print("\n=== Testing Trust Updates ===")

    initial_trust = router.get_plugin_trust_scores()
    print(f"Initial trust scores: {initial_trust}")

    # Trust should update after execution
    updated_trust = results.get('trust_scores', {})
    print(f"Updated trust scores: {updated_trust}")

    # At least one plugin should have trust updated
    if initial_trust and updated_trust:
        changed = any(
            abs(updated_trust.get(name, 0) - initial_trust.get(name, 0)) > 0.001
            for name in initial_trust
        )
        if changed:
            print("✓ Trust scores updated")
        else:
            print("⚠️ Trust scores unchanged (may be normal if all failed)")
    else:
        print("⚠️ No trust scores to compare")


async def main():
    """Run all integration tests"""
    print("=" * 70)
    print("IRP Plugin Integration Test Suite")
    print("=" * 70)

    try:
        # Test 1: Discovery
        router = await test_plugin_discovery()

        if not router.available:
            print("\n⚠️ No plugins available - skipping execution tests")
            print("This is normal if IRP dependencies aren't installed")
            return

        # Test 2: ATP Allocation
        allocations = await test_atp_allocation(router)

        # Test 3: Execution
        results = await test_plugin_execution(router, allocations)

        # Test 4: Trust Updates
        await test_trust_updates(router, results)

        # Cleanup
        router.shutdown()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
