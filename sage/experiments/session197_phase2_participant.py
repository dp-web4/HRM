#!/usr/bin/env python3
"""
Session 197 Phase 2: Federation Participant Deployment Script

This wrapper script starts the consciousness-aware federation participant
configured for real network deployment (connecting to Thor coordinator).
"""

import sys
import argparse
from session197_consciousness_federation_participant import FederationParticipant

def main():
    parser = argparse.ArgumentParser(
        description="Session 197 Phase 2: Consciousness-Aware Federation Participant"
    )
    parser.add_argument("--coordinator-host", default="10.0.0.99",
                       help="Coordinator host IP (default: 10.0.0.99 = Thor)")
    parser.add_argument("--coordinator-port", type=int, default=8000,
                       help="Coordinator port (default: 8000)")
    parser.add_argument("--node-id", default="sprout",
                       help="Node ID for this participant (default: sprout)")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Test duration in seconds (default: 60)")

    args = parser.parse_args()

    coordinator_url = f"http://{args.coordinator_host}:{args.coordinator_port}"

    print("=" * 70)
    print("Session 197 Phase 2: Consciousness-Aware Federation Participant")
    print("=" * 70)
    print(f"Node ID: {args.node_id}")
    print(f"Coordinator: {coordinator_url}")
    print(f"Duration: {args.duration} seconds")
    print()
    print("Network Deployment Configuration:")
    print("- Connecting to remote coordinator (Thor)")
    print("- Consciousness validation: C ≥ 0.5, γ ≈ 0.35")
    print("- Snapshot frequency: 10 Hz (100ms cycle)")
    print("- Nine-domain federation: D1-D9")
    print("- Multi-coupling: D4→D2, D8→D1, D5→D9")
    print()

    # Create participant
    participant = FederationParticipant(
        node_id=args.node_id,
        coordinator_url=coordinator_url
    )

    print(f"[Participant {args.node_id}] Connecting to coordinator...")
    print()
    print("Starting federation loop...")
    print("Press Ctrl+C to stop")
    print()

    try:
        participant.run_participant_loop(duration=args.duration)
    except KeyboardInterrupt:
        print(f"\n[Participant {args.node_id}] Interrupted by user")
        participant.stop()
        sys.exit(0)

    print()
    print("=" * 70)
    print("Participant Test Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
