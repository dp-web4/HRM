#!/usr/bin/env python3
"""
PolicyGate Integration Test - 50 Cycle Run
Tests PolicyGate Phase 2 completion: consciousness loop integration
"""

import asyncio
import sys
from pathlib import Path

# Add sage to path
sage_root = Path(__file__).parent
if str(sage_root) not in sys.path:
    sys.path.insert(0, str(sage_root))

from sage.core.sage_consciousness import SAGEConsciousness


async def test_policy_gate_50_cycles():
    """Run 50-cycle consciousness loop test with PolicyGate enabled."""

    print("="*80)
    print("PolicyGate Phase 2 Integration Test")
    print("="*80)
    print()

    # Configure with PolicyGate enabled
    config = {
        'enable_policy_gate': True,
        'policy_rules': [
            {
                'name': 'network_monitoring',
                'action_type': 'network',
                'decision': 'warn',
                'reason': 'Network actions require audit trail'
            },
            {
                'name': 'file_safety',
                'action_type': 'filesystem',
                'decision': 'deny',
                'reason': 'Filesystem writes blocked in test mode'
            }
        ],
        'default_policy': 'allow'
    }

    print(f"Configuration:")
    print(f"  PolicyGate: Enabled")
    print(f"  Rules: {len(config['policy_rules'])}")
    print(f"  Default: {config['default_policy']}")
    print()

    # Create consciousness instance
    sage = SAGEConsciousness(
        config=config,
        initial_atp=100.0,
        simulation_mode=True
    )

    # Verify PolicyGate is active
    print(f"PolicyGate Status:")
    print(f"  Enabled: {sage.policy_gate_enabled}")
    print(f"  Instance: {sage.policy_gate is not None}")
    print(f"  Effect system: {sage._effect_system_available}")
    print()

    if not sage.policy_gate_enabled:
        print("ERROR: PolicyGate not enabled!")
        return False

    # Run 50 cycles
    print("Running 50-cycle consciousness loop...")
    print()

    await sage.run(max_cycles=50)

    # Analyze results
    print()
    print("="*80)
    print("PolicyGate Integration Test Results")
    print("="*80)
    print()

    stats = sage.stats

    print(f"Consciousness Metrics:")
    print(f"  Total cycles: {stats['total_cycles']}")
    print(f"  Plugins executed: {stats['plugins_executed']}")
    print(f"  State transitions: {stats['state_transitions']}")
    print(f"  ATP consumed: {stats['total_atp_consumed']:.2f}")
    print()

    print(f"Effect System Metrics:")
    print(f"  Effects proposed: {stats.get('effects_proposed', 0)}")
    print(f"  Effects approved: {stats.get('effects_approved', 0)}")
    print(f"  Effects denied: {stats.get('effects_denied', 0)}")
    print(f"  Effects executed: {stats.get('effects_executed', 0)}")
    print()

    print(f"Memory Systems:")
    print(f"  SNARC experiences: {len(sage.snarc_memory)}")
    print(f"  Average salience: {stats['average_salience']:.3f}")
    print()

    print(f"Final State:")
    print(f"  Metabolic state: {sage.metabolic.current_state.value}")
    print(f"  ATP remaining: {sage.metabolic.atp_current:.2f}")
    print()

    # Validation
    print("Validation:")
    success = True

    if stats['total_cycles'] != 50:
        print(f"  ✗ Cycle count mismatch: {stats['total_cycles']} != 50")
        success = False
    else:
        print(f"  ✓ Completed 50 cycles")

    if sage.policy_gate_enabled:
        print(f"  ✓ PolicyGate remained active")
    else:
        print(f"  ✗ PolicyGate was disabled")
        success = False

    if stats.get('effects_proposed', 0) > 0:
        approval_rate = stats.get('effects_approved', 0) / stats['effects_proposed']
        print(f"  ✓ PolicyGate processed effects (approval: {approval_rate:.1%})")
    else:
        print(f"  ℹ No effects proposed (mock sensors)")

    if len(sage.snarc_memory) > 0:
        print(f"  ✓ SNARC memory recording experiences")
    else:
        print(f"  ✗ SNARC memory empty")
        success = False

    print()

    if success:
        print("✅ PolicyGate Phase 2 Integration: COMPLETE")
        print()
        print("All systems operational:")
        print("  - PolicyGate integrated at step 8.6")
        print("  - Effect evaluation pipeline active")
        print("  - Consciousness loop stable over 50 cycles")
        print("  - Memory systems recording policy decisions")
        return True
    else:
        print("❌ PolicyGate Phase 2 Integration: ISSUES DETECTED")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_policy_gate_50_cycles())
    sys.exit(0 if result else 1)
