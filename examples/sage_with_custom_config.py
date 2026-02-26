#!/usr/bin/env python3
"""
SAGE with Custom Configuration Example

Shows how to customize SAGE's metabolic parameters, ATP budget,
and other settings.
"""

import asyncio
import sys
from pathlib import Path

# Add HRM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sage import SAGE


async def main():
    """Run SAGE with custom configuration"""

    print("Creating SAGE with custom configuration...")
    print()

    # Custom configuration
    config = {
        # Hardware-bound identity
        'lct_identity': {
            'hardware_id': 'thor-example-001',
            'entity_name': 'SAGE-Thor-Demo'
        },

        # Metabolic parameters
        'metabolic_params': {
            'base_atp': 2000.0,        # Higher starting ATP
            'atp_recovery_rate': 50.0,  # Faster recovery
            'crisis_threshold': 100.0   # Lower crisis threshold
        },

        # SNARC weights (5D salience)
        'snarc_weights': {
            'surprise': 0.25,
            'novelty': 0.20,
            'arousal': 0.20,
            'reward': 0.20,
            'conflict': 0.15
        }
    }

    # Create SAGE with custom config
    sage = SAGE.create(config=config)

    print("Configuration:")
    print(f"  Hardware ID: {config['lct_identity']['hardware_id']}")
    print(f"  Base ATP: {config['metabolic_params']['base_atp']}")
    print(f"  Recovery rate: {config['metabolic_params']['atp_recovery_rate']}")
    print()

    print("Running SAGE for 15 consciousness cycles...")
    print()

    # Run for 15 cycles
    stats = await sage.run(max_cycles=15)

    print()
    print("SAGE run complete!")
    print(f"  Cycles: {stats['cycles_completed']}")
    print(f"  Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"  Final state: {stats['final_state']}")


if __name__ == '__main__':
    asyncio.run(main())
