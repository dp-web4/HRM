#!/usr/bin/env python3
"""
Session 197 Phase 2: Federation Coordinator Deployment Script

This wrapper script starts the consciousness-aware federation coordinator
configured for real network deployment (Thor as coordinator).
"""

import sys
import argparse
from session197_consciousness_federation_coordinator import FederationCoordinator

def main():
    parser = argparse.ArgumentParser(
        description="Session 197 Phase 2: Consciousness-Aware Federation Coordinator"
    )
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to listen on (default: 8000)")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Test duration in seconds (default: 60)")

    args = parser.parse_args()

    print("=" * 70)
    print("Session 197 Phase 2: Consciousness-Aware Federation Coordinator")
    print("=" * 70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Duration: {args.duration} seconds")
    print()
    print("Network Deployment Configuration:")
    print("- Listening on all interfaces (0.0.0.0)")
    print("- Accepting connections from remote participants")
    print("- Consciousness validation: C ≥ 0.5, γ ≈ 0.35")
    print("- Synchronization: 10 Hz, Γ=0.1, κ=0.15")
    print()

    # Create coordinator
    coordinator = FederationCoordinator(
        coordinator_id="thor_0099",
        host=args.host,
        port=args.port
    )

    print(f"[Coordinator thor_0099] Starting on {args.host}:{args.port}")
    print()
    print("Endpoints:")
    print("  POST /snapshot          - Receive participant snapshots")
    print("  GET  /sync_signal       - Serve synchronization signals")
    print("  POST /coupling_event    - Receive coupling events")
    print("  GET  /federation_status - Federation health")
    print()
    print("Waiting for participants to connect...")
    print("Press Ctrl+C to stop")
    print()

    try:
        coordinator.run_coordinator_server(duration=args.duration)
    except KeyboardInterrupt:
        print("\n[Coordinator] Interrupted by user")
        coordinator.stop()
        sys.exit(0)

    print()
    print("=" * 70)
    print("Coordinator Test Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
