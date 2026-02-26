#!/usr/bin/env python3
"""
Hello SAGE - Minimal Example

The simplest possible SAGE consciousness system.

This is the "hello world" for SAGE - run a consciousness loop for 10 cycles
and see what happens.
"""

import asyncio
import sys
from pathlib import Path

# Add HRM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sage import SAGE


async def main():
    """Run SAGE for 10 cycles"""

    print("Creating SAGE with default configuration...")
    print("(Mock sensors, mock LLM, algorithmic SNARC)")
    print()

    # Create SAGE with defaults
    sage = SAGE.create()

    print("Running SAGE for 10 consciousness cycles...")
    print()

    # Run for 10 cycles
    stats = await sage.run(max_cycles=10)

    print()
    print("SAGE run complete!")
    print(f"  Cycles: {stats['cycles_completed']}")
    print(f"  Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"  Final state: {stats['final_state']}")


if __name__ == '__main__':
    asyncio.run(main())
