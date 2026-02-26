#!/usr/bin/env python3
"""
SAGE with PolicyGate Example

Shows how to enable PolicyGate for accountability.

PolicyGate is an IRP plugin that evaluates proposed actions at step 8.5
of the consciousness loop, providing on-device policy compliance.
"""

import asyncio
import sys
from pathlib import Path

# Add HRM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sage import SAGE


async def main():
    """Run SAGE with PolicyGate enabled"""

    print("Creating SAGE with PolicyGate enabled...")
    print()

    # Create SAGE with PolicyGate
    sage = SAGE.create(use_policy_gate=True)

    print("Running SAGE for 20 consciousness cycles...")
    print("(PolicyGate will evaluate all proposed actions)")
    print()

    # Run for 20 cycles
    stats = await sage.run(max_cycles=20)

    print()
    print("SAGE run complete!")
    print(f"  Cycles: {stats['cycles_completed']}")
    print(f"  Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"  Final state: {stats['final_state']}")

    # Get detailed statistics
    detailed_stats = sage.get_statistics()
    print()
    print("Detailed statistics:")
    for key, value in detailed_stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    asyncio.run(main())
