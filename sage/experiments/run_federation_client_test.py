#!/usr/bin/env python3
"""
SAGE Federation Client Test

Tests federation task delegation by sending tasks to a remote SAGE platform.

Usage:
    # Test Thor server from local client (simulating Sprout):
    python3 run_federation_client_test.py --target thor --host localhost --port 50051

    # Test from actual Sprout to Thor:
    python3 run_federation_client_test.py --target thor --host thor.local --port 50051

The client will:
1. Load local platform's Ed25519 key (Sprout or Thor)
2. Create test federation task
3. Send task to target platform via HTTP
4. Verify execution proof signature
5. Display results

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
from sage.federation.federation_service import FederationClient
from sage.federation.federation_types import FederationTask, QualityRequirements
from sage.core.mrh_profile import PROFILE_REFLEXIVE
from sage.core.attention_manager import MetabolicState


def main():
    parser = argparse.ArgumentParser(description='SAGE Federation Client Test')
    parser.add_argument('--local', choices=['thor', 'sprout'], default='sprout',
                        help='Local platform (default: sprout)')
    parser.add_argument('--target', choices=['thor', 'sprout'], default='thor',
                        help='Target platform to delegate to (default: thor)')
    parser.add_argument('--host', default='localhost',
                        help='Target host (default: localhost)')
    parser.add_argument('--port', type=int, default=50051,
                        help='Target port (default: 50051)')
    parser.add_argument('--task-type', default='llm_inference',
                        help='Task type to test (default: llm_inference)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SAGE Federation Client Test")
    print("="*80)

    # Create local identity
    if args.local == 'thor':
        local_identity = create_thor_identity()
        local_key_path = sage_root / "sage" / "data" / "keys" / "Thor_ed25519.key"
    else:
        local_identity = create_sprout_identity()
        local_key_path = sage_root / "sage" / "data" / "keys" / "Sprout_ed25519.key"

    print(f"\n[Setup] Local platform: {local_identity.platform_name}")
    print(f"  LCT ID: {local_identity.lct_id}")

    if not local_key_path.exists():
        print(f"\n[Error] Local key not found at {local_key_path}")
        print(f"[Error] Please run Phase 2 setup to generate keys")
        sys.exit(1)

    print(f"\n[Setup] Loading local Ed25519 key...")
    print(f"  Path: {local_key_path}")

    with open(local_key_path, 'rb') as f:
        local_private_key = f.read()

    local_keypair = FederationKeyPair.from_bytes(
        local_identity.platform_name,
        local_identity.lct_id,
        local_private_key
    )
    local_identity.public_key = local_keypair.public_key_bytes()

    print(f"  Public key: {local_identity.public_key.hex()[:40]}...")

    # Create target identity
    if args.target == 'thor':
        target_identity = create_thor_identity()
        target_key_path = sage_root / "sage" / "data" / "keys" / "Thor_ed25519.key"
    else:
        target_identity = create_sprout_identity()
        target_key_path = sage_root / "sage" / "data" / "keys" / "Sprout_ed25519.key"

    print(f"\n[Setup] Target platform: {target_identity.platform_name}")
    print(f"  LCT ID: {target_identity.lct_id}")
    print(f"  Address: http://{args.host}:{args.port}")

    if not target_key_path.exists():
        print(f"\n[Error] Target key not found at {target_key_path}")
        print(f"[Error] Cannot verify proof signatures without target's public key")
        sys.exit(1)

    print(f"\n[Setup] Loading target's public key...")
    with open(target_key_path, 'rb') as f:
        target_private_key = f.read()

    target_keypair = FederationKeyPair.from_bytes(
        target_identity.platform_name,
        target_identity.lct_id,
        target_private_key
    )
    target_identity.public_key = target_keypair.public_key_bytes()

    print(f"  Public key: {target_identity.public_key.hex()[:40]}...")

    # Create federation client
    print(f"\n[Client] Creating federation client...")
    client = FederationClient(
        local_identity=local_identity,
        signing_key=local_private_key,
        platform_registry={target_identity.lct_id: (args.host, args.port)}
    )

    # Health check
    print(f"\n[Client] Health checking target platform...")
    healthy = client.health_check(target_identity.lct_id, timeout=5.0)

    if not healthy:
        print(f"  ✗ Target platform unreachable")
        print(f"\n[Error] Cannot reach {target_identity.platform_name} at {args.host}:{args.port}")
        print(f"[Error] Make sure the federation server is running")
        sys.exit(1)

    print(f"  ✓ Target platform healthy")

    # Create test task
    print(f"\n[Client] Creating test task...")
    print(f"  Task type: {args.task_type}")

    task = FederationTask(
        task_id=f"test_task_{int(time.time())}",
        task_type=args.task_type,
        task_data={
            'query': "What is SAGE consciousness architecture?",
            'test': True
        },
        estimated_cost=50.0,
        task_horizon=PROFILE_REFLEXIVE,
        complexity="low",
        delegating_platform=local_identity.lct_id,
        delegating_state=MetabolicState.FOCUS,
        quality_requirements=QualityRequirements(
            min_quality=0.6,
            min_convergence=0.7,
            max_energy=100.0
        ),
        max_latency=30.0,
        deadline=time.time() + 60.0
    )

    print(f"  Task ID: {task.task_id}")
    print(f"  Estimated cost: {task.estimated_cost:.1f} ATP")

    # Delegate task
    print(f"\n[Client] Delegating task to {target_identity.platform_name}...")
    print(f"  " + "-"*76)

    start_time = time.time()

    proof = client.delegate_task(
        task,
        target_platform_id=target_identity.lct_id,
        target_public_key=target_identity.public_key,
        timeout=60.0
    )

    delegation_time = time.time() - start_time

    print(f"  " + "-"*76)

    # Check result
    if proof:
        print(f"\n" + "="*80)
        print("✓ Task Delegation Successful")
        print("="*80)

        print(f"\nExecution Proof:")
        print(f"  Task ID: {proof.task_id}")
        print(f"  Executing platform: {proof.executing_platform}")
        print(f"  Actual latency: {proof.actual_latency:.2f}s")
        print(f"  Actual cost: {proof.actual_cost:.1f} ATP")
        print(f"  Quality score: {proof.quality_score:.2f}")
        print(f"  Convergence: {proof.convergence_quality:.2f}")
        print(f"  IRP iterations: {proof.irp_iterations}")

        print(f"\nResult Data:")
        for key, value in proof.result_data.items():
            print(f"  {key}: {value}")

        print(f"\nPerformance:")
        print(f"  Total delegation time: {delegation_time:.2f}s")
        print(f"  Network overhead: {delegation_time - proof.actual_latency:.2f}s")

        print(f"\nSecurity:")
        print(f"  ✓ Task signed with {local_identity.platform_name}'s Ed25519 key")
        print(f"  ✓ Proof verified with {target_identity.platform_name}'s Ed25519 key")
        print(f"  ✓ Cryptographic chain of trust maintained")

        print("\n" + "="*80)
        print("Federation Test Complete - SUCCESS")
        print("="*80 + "\n")

        sys.exit(0)

    else:
        print(f"\n" + "="*80)
        print("✗ Task Delegation Failed")
        print("="*80)
        print(f"\nPossible causes:")
        print(f"  - Network connectivity issues")
        print(f"  - Signature verification failed")
        print(f"  - Target platform rejected task")
        print(f"  - Timeout exceeded")
        print("\n" + "="*80 + "\n")

        sys.exit(1)


if __name__ == "__main__":
    main()
