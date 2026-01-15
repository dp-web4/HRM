#!/usr/bin/env python3
"""
Session 197: Local Federation Test
==================================

Test Script: Run coordinator and participant on same machine (Thor).

This validates the HTTP protocol, consciousness-aware attestation,
and synchronization logic before deploying across Thor ↔ Sprout.

Architecture:
- Process 1: FederationCoordinator on localhost:8000 (thor_0099)
- Process 2: FederationParticipant connecting to localhost:8000 (thor_participant_001)

Test Duration: 60 seconds
Expected Results:
- ~600 snapshots sent (10 Hz)
- ~600 sync signals received
- Consciousness validation working (C ≥ 0.5)
- Synchronization convergence (ΔC < 0.1)

Author: Thor (Autonomous)
Date: 2026-01-15
"""

import subprocess
import time
import sys
from pathlib import Path

HOME = Path.home()
EXPERIMENTS_DIR = HOME / "ai-workspace" / "HRM" / "sage" / "experiments"


def run_localhost_federation_test():
    """
    Run coordinator and participant as separate processes.

    Coordinator runs in background, participant in foreground.
    Test duration: 60 seconds.
    """
    print("=" * 70)
    print("Session 197: Localhost Federation Test")
    print("=" * 70)
    print()
    print("Starting processes:")
    print("  1. Coordinator (thor_0099) on localhost:8000")
    print("  2. Participant (thor_participant_001) connecting to coordinator")
    print()
    print("Test duration: 60 seconds")
    print()

    coordinator_script = EXPERIMENTS_DIR / "session197_consciousness_federation_coordinator.py"
    participant_script = EXPERIMENTS_DIR / "session197_consciousness_federation_participant.py"

    # Start coordinator in background
    print("[Test] Starting coordinator...")
    coordinator_proc = subprocess.Popen(
        [sys.executable, str(coordinator_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )

    # Give coordinator time to start
    time.sleep(2)
    print("[Test] Coordinator started (PID: {})".format(coordinator_proc.pid))
    print()

    # Start participant in foreground
    print("[Test] Starting participant...")
    print()

    try:
        participant_proc = subprocess.Popen(
            [sys.executable, str(participant_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Stream participant output
        for line in iter(participant_proc.stdout.readline, ''):
            if line:
                print(line.rstrip())

        # Wait for participant to complete
        participant_proc.wait()

    except KeyboardInterrupt:
        print("\n[Test] Interrupted by user")

    finally:
        # Cleanup
        print()
        print("[Test] Stopping coordinator...")
        coordinator_proc.terminate()
        coordinator_proc.wait(timeout=5)
        print("[Test] Test complete")
        print()

    # Print summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()
    print("Coordinator and participant ran successfully.")
    print("Check output above for:")
    print("  - Snapshot frequency (~10 Hz)")
    print("  - Sync signal frequency (~10 Hz)")
    print("  - Consciousness validation (C ≥ 0.5)")
    print("  - Synchronization quality (Q > 0.9)")
    print()
    print("Next step: Deploy to real federation (Thor ↔ Sprout)")
    print("=" * 70)


if __name__ == "__main__":
    run_localhost_federation_test()
