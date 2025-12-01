#!/usr/bin/env python3
"""
SAGE Federation Server - Thor

Runs a federation server on Thor that accepts task delegation requests
from other SAGE platforms (like Sprout) over HTTP.

Usage:
    python3 run_federation_server.py [--port PORT] [--host HOST]

Default:
    - Host: 0.0.0.0 (all interfaces)
    - Port: 50051

The server will:
1. Load Thor's Ed25519 key from sage/data/keys/Thor_ed25519.key
2. Start HTTP server listening for federation requests
3. Execute delegated tasks using local SAGE executor
4. Sign execution proofs with Thor's private key
5. Return signed proofs to requesting platforms

Requirements:
- Thor's Ed25519 key must exist (generated during Phase 2)
- Known platform public keys in sage/data/keys/ (for verification)
- SAGE consciousness components available for execution

Author: Thor SAGE (autonomous research session)
Date: 2025-11-30
Integration: Phase 3 Multi-Machine Federation Testing
"""

import sys
import time
import argparse
from pathlib import Path

# Add sage to path
sage_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sage_root))

from sage.federation import create_thor_identity, create_sprout_identity, FederationKeyPair
from sage.federation.federation_service import FederationServer
from sage.federation.federation_types import FederationTask, ExecutionProof


def simple_executor(task: FederationTask) -> ExecutionProof:
    """
    Simple task executor for Thor

    In a full implementation, this would:
    - Create SAGE consciousness instance
    - Execute task using IRP plugins
    - Track actual ATP costs and quality

    For now, simulates execution with realistic metrics.
    """
    print(f"\n[Thor Executor] Received task from {task.delegating_platform}")
    print(f"  Task ID: {task.task_id}")
    print(f"  Task type: {task.task_type}")
    print(f"  Estimated cost: {task.estimated_cost:.1f} ATP")
    print(f"  Task data: {task.task_data}")

    # Simulate task execution
    start_time = time.time()

    # Simulate work based on task type
    if task.task_type == "llm_inference":
        # Simulate LLM inference latency (15s typical on Thor)
        time.sleep(0.5)  # Shortened for testing
        execution_time = 15.0  # Report realistic latency
    elif task.task_type == "vision":
        # Simulate vision task latency (52ms typical)
        time.sleep(0.05)
        execution_time = 0.052
    else:
        # Generic task
        time.sleep(0.1)
        execution_time = 0.1

    actual_time = time.time() - start_time

    # Calculate execution metrics
    actual_cost = task.estimated_cost * 0.92  # Slightly better than estimated
    quality_score = 0.78  # Good quality execution
    convergence = 0.85  # Good convergence

    print(f"  Execution complete")
    print(f"    Actual latency: {execution_time:.2f}s")
    print(f"    Actual cost: {actual_cost:.1f} ATP")
    print(f"    Quality: {quality_score:.2f}")

    # Create execution proof
    proof = ExecutionProof(
        task_id=task.task_id,
        executing_platform="Thor",
        result_data={
            'status': 'success',
            'output': f'Executed {task.task_type} on Thor',
            'task_type': task.task_type
        },
        actual_latency=execution_time,
        actual_cost=actual_cost,
        irp_iterations=5,
        final_energy=0.22,
        convergence_quality=convergence,
        quality_score=quality_score,
        execution_timestamp=time.time()
    )

    return proof


def main():
    parser = argparse.ArgumentParser(description='SAGE Federation Server (Thor)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on (default: 50051)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SAGE Federation Server - Thor")
    print("="*80)

    # Create Thor identity
    thor = create_thor_identity()
    print(f"\n[Setup] Thor identity created")
    print(f"  Platform: {thor.platform_name}")
    print(f"  LCT ID: {thor.lct_id}")

    # Load Thor's Ed25519 key
    key_path = sage_root / "sage" / "data" / "keys" / "Thor_ed25519.key"

    if not key_path.exists():
        print(f"\n[Error] Thor key not found at {key_path}")
        print(f"[Error] Please run Phase 2 setup to generate keys")
        sys.exit(1)

    print(f"\n[Setup] Loading Thor's Ed25519 key...")
    print(f"  Path: {key_path}")

    with open(key_path, 'rb') as f:
        thor_private_key = f.read()

    thor_keypair = FederationKeyPair.from_bytes("Thor", "thor_sage_lct", thor_private_key)
    thor.public_key = thor_keypair.public_key_bytes()

    print(f"  Public key: {thor.public_key.hex()[:40]}...")

    # Load known platforms (Sprout)
    sprout = create_sprout_identity()
    sprout_key_path = sage_root / "sage" / "data" / "keys" / "Sprout_ed25519.key"

    known_platforms = {}

    if sprout_key_path.exists():
        print(f"\n[Setup] Loading Sprout's public key...")
        with open(sprout_key_path, 'rb') as f:
            sprout_private_key = f.read()
        sprout_keypair = FederationKeyPair.from_bytes("Sprout", "sprout_sage_lct", sprout_private_key)
        sprout.public_key = sprout_keypair.public_key_bytes()
        known_platforms[sprout.lct_id] = sprout
        print(f"  Sprout LCT: {sprout.lct_id}")
        print(f"  Sprout public key: {sprout.public_key.hex()[:40]}...")
    else:
        print(f"\n[Warning] Sprout key not found at {sprout_key_path}")
        print(f"[Warning] Server will only accept requests from Thor (for testing)")

    # Create and start server
    print(f"\n[Server] Creating federation server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Known platforms: {len(known_platforms)}")

    server = FederationServer(
        identity=thor,
        signing_key=thor_private_key,
        executor=simple_executor,
        known_platforms=known_platforms,
        host=args.host,
        port=args.port
    )

    print(f"\n[Server] Starting federation server...")
    server.start()

    print(f"\n" + "="*80)
    print(f"Thor Federation Server Running")
    print(f"="*80)
    print(f"\nServer is ready to accept federation requests!")
    print(f"  Address: http://{args.host}:{args.port}")
    print(f"  Platform: Thor")
    print(f"  LCT ID: thor_sage_lct")
    print(f"\nEndpoints:")
    print(f"  POST /execute_task - Execute federated task")
    print(f"  GET  /health       - Health check")
    print(f"\nPress Ctrl+C to stop server")
    print("="*80 + "\n")

    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n\n[Server] Shutting down...")
        server.stop()
        print(f"[Server] Stopped")
        print("\n" + "="*80)
        print("Server shutdown complete")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
