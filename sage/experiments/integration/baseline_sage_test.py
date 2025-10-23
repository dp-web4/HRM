#!/usr/bin/env python3
"""
Baseline SAGE Loop Test
Experiment: What happens in sensory deprivation?

Question: How does SNARC behave with no sensors?
- Does it handle empty observations gracefully?
- What does "salience" mean when there's nothing to sense?
- Does trust evolve or stay flat?
- Do we see any emergent patterns?

Approach: Run kernel for 20 cycles, observe, document surprises.
"""

import sys
import os
from pathlib import Path

# Add sage to path - need to be at HRM level
hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from sage.core.sage_kernel import SAGEKernel, MetabolicState, ExecutionResult

def main():
    print("=" * 70)
    print("BASELINE SAGE LOOP TEST: Sensory Deprivation")
    print("=" * 70)
    print("\nExperiment: SAGE kernel with NO sensors")
    print("Question: What does consciousness do when it can't sense?\n")

    # Configuration
    sensor_sources = {}  # Empty - no sensors
    action_handlers = {}  # Empty - no actions

    # Create kernel
    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True
    )

    print("Starting kernel with zero sensors...")
    print("Let's see what happens...\n")

    # Run for 20 cycles
    try:
        kernel.run(max_cycles=20, cycle_delay=0.5)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nInteresting! The system breaks without sensors.")
        print("What does this tell us about consciousness requirements?")

    print("\n" + "=" * 70)
    print("BASELINE TEST COMPLETE")
    print("=" * 70)

    # Observations section
    print("\n## OBSERVATIONS\n")
    print("(Document what actually happened, not what was expected)")
    print("\n1. Did the kernel run?")
    print("2. How did SNARC handle empty observations?")
    print("3. What was the cycle behavior?")
    print("4. Any surprises?")
    print("\n")

if __name__ == "__main__":
    main()
