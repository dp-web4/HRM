#!/usr/bin/env python3
"""
SAGE Attention Kernel Demo

Demonstrates the continuous attention loop running with state machine,
ATP budgeting, experience capture, and auditable logging.

This is the v1 kernel - Tier 0 (always-on) without full IRP/LLM integration yet.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.attention import AttentionKernel, AttentionState


async def demo_basic_kernel():
    """Demo: Basic kernel running for a short duration"""
    print("=" * 70)
    print("SAGE Attention Kernel v1 - Demo")
    print("=" * 70)
    print()
    print("This demonstrates the foundational Tier 0 (always-on) attention")
    print("kernel running continuously with:")
    print("  - 6-state machine (IDLE/FOCUS/THINK/ACT/SLEEP/RECOVER)")
    print("  - ATP budget allocation")
    print("  - Experience capture")
    print("  - Auditable tick logging")
    print()
    print("=" * 70)
    print()

    # Configure kernel
    config = {
        'atp_budget': 1000.0,
        'buffer_size': 100,
        'log_dir': 'demo_logs',
        'tick_interval': 0.5,  # Half-second ticks for demo
        'sleep_policy': {
            'buffer_size': 50,
            'salience_sum': 25.0,
            'time_hours': 1,
            'idle_minutes': 5
        }
    }

    kernel = AttentionKernel(config)

    print("Kernel initialized in state:", kernel.state)
    print("Running for 10 seconds...")
    print()

    # Run for 10 seconds
    async def stop_after_delay():
        await asyncio.sleep(10.0)
        kernel.stop()

    await asyncio.gather(
        kernel.run_forever(),
        stop_after_delay()
    )

    print()
    print("=" * 70)
    print("Demo Complete")
    print("=" * 70)
    print()
    print(f"Total ticks: {kernel.tick_count}")
    print(f"Final state: {kernel.state}")
    print(f"Experience buffer size: {kernel.experience_buffer.size}")
    print(f"Total salience: {kernel.experience_buffer.salience_sum:.2f}")
    print()
    print("Logs available in:")
    print(f"  demo_logs/attention_tick.jsonl")
    print(f"  demo_logs/action_log.jsonl")
    print(f"  demo_logs/context_manifest.jsonl")
    print()

    # Show sample of logs
    log_path = Path('demo_logs/attention_tick.jsonl')
    if log_path.exists():
        import json
        with open(log_path) as f:
            lines = f.readlines()

        print(f"Sample tick log (first tick):")
        first_tick = json.loads(lines[0])
        print(json.dumps(first_tick, indent=2))


async def demo_with_state_transitions():
    """Demo: Kernel with manual state transitions"""
    print("=" * 70)
    print("SAGE Attention Kernel - State Transition Demo")
    print("=" * 70)
    print()

    config = {
        'log_dir': 'demo_logs_states',
        'tick_interval': 1.0,
    }

    kernel = AttentionKernel(config)

    print("Demonstrating state transitions...")
    print()

    # Manually trigger a few ticks and observe state
    for i in range(5):
        print(f"Tick {i+1}: State = {kernel.state}")
        await kernel.tick()
        await asyncio.sleep(0.5)

    print()
    print(f"Completed {kernel.tick_count} ticks")
    print(f"Final state: {kernel.state}")


def main():
    """Run demos"""
    import argparse

    parser = argparse.ArgumentParser(description='SAGE Attention Kernel Demo')
    parser.add_argument('--mode', choices=['basic', 'states', 'both'], default='basic',
                        help='Demo mode to run')

    args = parser.parse_args()

    if args.mode == 'basic':
        asyncio.run(demo_basic_kernel())
    elif args.mode == 'states':
        asyncio.run(demo_with_state_transitions())
    elif args.mode == 'both':
        asyncio.run(demo_basic_kernel())
        print("\n" + "=" * 70 + "\n")
        asyncio.run(demo_with_state_transitions())


if __name__ == '__main__':
    main()
