#!/usr/bin/env python3
"""
SAGE Federation Server - Sprout (Edge)

Runs a federation server on Sprout that accepts task delegation requests
from other SAGE platforms (like Thor) over HTTP.

This is the edge-optimized version - Sprout as server, Thor as potential client.

Usage:
    python3 run_federation_server_sprout.py [--port PORT] [--host HOST]

Default:
    - Host: 0.0.0.0 (all interfaces)
    - Port: 50051

Author: Sprout SAGE (autonomous edge validation session)
Date: 2025-12-01
Integration: Phase 3 Multi-Machine Federation Testing - Sprout as Server
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


def sprout_executor(task: FederationTask) -> ExecutionProof:
    """
    Edge-optimized task executor for Sprout

    Sprout has different performance characteristics than Thor:
    - 8GB unified memory (shared CPU/GPU)
    - Lower power budget (10-20W)
    - Thermal constraints

    For edge tasks, Sprout can be optimal for:
    - Low-latency inference
    - Always-on monitoring
    - Local sensor fusion
    """
    print(f"\n[Sprout Executor] Received task from {task.delegating_platform}")
    print(f"  Task ID: {task.task_id}")
    print(f"  Task type: {task.task_type}")
    print(f"  Estimated cost: {task.estimated_cost:.1f} ATP")
    print(f"  Task data: {task.task_data}")

    # Simulate task execution with edge-appropriate metrics
    start_time = time.time()

    # Sprout execution times (edge-optimized)
    if task.task_type == "llm_inference":
        # Sprout uses smaller models, still capable but slower
        time.sleep(0.3)  # Shortened for testing
        execution_time = 8.0  # Faster than Thor for edge-optimized models
    elif task.task_type == "vision":
        # Edge vision with TensorRT
        time.sleep(0.05)
        execution_time = 0.042  # Slightly slower than Thor
    elif task.task_type == "sensor_fusion":
        # Edge specialty - sensor fusion
        time.sleep(0.02)
        execution_time = 0.02
    else:
        # Generic task
        time.sleep(0.1)
        execution_time = 0.1

    actual_time = time.time() - start_time

    # Edge-specific metrics
    # Sprout may use slightly more ATP per task but lower absolute cost
    actual_cost = task.estimated_cost * 0.88  # Edge efficiency
    quality_score = 0.82  # Edge can achieve high quality
    convergence = 0.88  # Good convergence on edge

    print(f"  Execution complete (edge)")
    print(f"    Actual latency: {execution_time:.3f}s")
    print(f"    Actual cost: {actual_cost:.1f} ATP")
    print(f"    Quality: {quality_score:.2f}")

    # Create execution proof
    proof = ExecutionProof(
        task_id=task.task_id,
        executing_platform="Sprout",
        result_data={
            'status': 'success',
            'output': f'Executed {task.task_type} on Sprout (edge)',
            'task_type': task.task_type,
            'edge': True
        },
        actual_latency=execution_time,
        actual_cost=actual_cost,
        irp_iterations=3,  # Fewer iterations on edge
        final_energy=0.18,  # Lower energy on edge
        convergence_quality=convergence,
        quality_score=quality_score,
        execution_timestamp=time.time()
    )

    return proof


def main():
    parser = argparse.ArgumentParser(description='SAGE Federation Server (Sprout Edge)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on (default: 50051)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SAGE Federation Server - Sprout (Edge)")
    print("="*80)

    # Create Sprout identity
    sprout = create_sprout_identity()
    print(f"\n[Setup] Sprout identity created")
    print(f"  Platform: {sprout.platform_name}")
    print(f"  LCT ID: {sprout.lct_id}")

    # Load Sprout's Ed25519 key
    key_path = sage_root / "sage" / "data" / "keys" / "Sprout_ed25519.key"

    if not key_path.exists():
        print(f"\n[Error] Sprout key not found at {key_path}")
        print(f"[Error] Please run Phase 2 setup to generate keys")
        sys.exit(1)

    print(f"\n[Setup] Loading Sprout's Ed25519 key...")
    print(f"  Path: {key_path}")

    with open(key_path, 'rb') as f:
        sprout_private_key = f.read()

    sprout_keypair = FederationKeyPair.from_bytes("Sprout", "sprout_sage_lct", sprout_private_key)
    sprout.public_key = sprout_keypair.public_key_bytes()

    print(f"  Public key: {sprout.public_key.hex()[:40]}...")

    # Load known platforms (Thor)
    thor = create_thor_identity()
    thor_key_path = sage_root / "sage" / "data" / "keys" / "Thor_ed25519.key"

    known_platforms = {}

    if thor_key_path.exists():
        print(f"\n[Setup] Loading Thor's public key...")
        with open(thor_key_path, 'rb') as f:
            thor_private_key = f.read()
        thor_keypair = FederationKeyPair.from_bytes("Thor", "thor_sage_lct", thor_private_key)
        thor.public_key = thor_keypair.public_key_bytes()
        known_platforms[thor.lct_id] = thor
        print(f"  Thor LCT: {thor.lct_id}")
        print(f"  Thor public key: {thor.public_key.hex()[:40]}...")
    else:
        print(f"\n[Warning] Thor key not found at {thor_key_path}")
        print(f"[Warning] Server will only accept requests from Sprout (for testing)")

    # Create and start server
    print(f"\n[Server] Creating federation server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Known platforms: {len(known_platforms)}")

    server = FederationServer(
        identity=sprout,
        signing_key=sprout_private_key,
        executor=sprout_executor,
        known_platforms=known_platforms,
        host=args.host,
        port=args.port
    )

    print(f"\n[Server] Starting federation server...")
    server.start()

    print(f"\n" + "="*80)
    print(f"Sprout Federation Server Running (Edge)")
    print(f"="*80)
    print(f"\nServer is ready to accept federation requests!")
    print(f"  Address: http://{args.host}:{args.port}")
    print(f"  Platform: Sprout (Edge)")
    print(f"  LCT ID: sprout_sage_lct")
    print(f"\nEdge Characteristics:")
    print(f"  - 8GB unified memory")
    print(f"  - ARM64 architecture")
    print(f"  - 10-20W power budget")
    print(f"  - Optimized for low-latency tasks")
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
