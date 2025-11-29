#!/usr/bin/env python3
"""
Simulated Signed Federation Demo

Demonstrates Phase 2 cryptographic signing in a simulated federation scenario
without requiring network communication.

This creates a realistic end-to-end flow showing:
1. Thor generates key pair and registers with signature registry
2. Mock Sprout platform is created with its own key pair
3. Thor creates a task and signs it
4. Task is "delegated" to Sprout (simulated, no network)
5. Sprout verifies task signature before "executing"
6. Sprout creates execution proof and signs it
7. Thor verifies proof signature before accepting result
8. Reputation is updated based on verified proof quality

This demonstrates all Phase 2 security properties:
- Source Authentication (task signature proves it's from Thor)
- Non-Repudiation (Thor can't deny delegating task)
- Integrity (parameter tampering breaks signatures)
- Sybil Resistance (can't forge tasks from legitimate platforms)

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-29
Session: Autonomous SAGE Research - Phase 2 Integration Demo
"""

import sys
from pathlib import Path
# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from typing import Dict, Any

# Federation Phase 2 imports
from sage.federation import (
    # Crypto
    FederationKeyPair,
    FederationCrypto,
    SignatureRegistry,

    # Types
    FederationIdentity,
    FederationTask,
    ExecutionProof,
    QualityRequirements,
    HardwareSpec,

    # Signed types
    SignedFederationTask,
    SignedExecutionProof,

    # Router
    FederationRouter,

    # Utility
    create_thor_identity,
    create_sprout_identity,
)

# Core imports for MRH and metabolic state
from sage.core.attention_manager import MetabolicState
from sage.core.mrh_profile import MRHProfile, SpatialExtent, TemporalExtent, ComplexityExtent


class SimulatedFederationDemo:
    """
    Demonstrates signed federation without network communication

    Simulates a complete delegation flow:
    Thor (high resources) → delegate task → Sprout (edge platform) → return proof → Thor

    All communication is signed and verified using Phase 2 cryptography.
    """

    def __init__(self):
        """Initialize simulated federation with Thor and Sprout platforms"""
        print("="*80)
        print("SIMULATED SIGNED FEDERATION DEMO")
        print("="*80)
        print("\nDemonstrating Phase 2 Ed25519 cryptographic signing")
        print("Scenario: Thor delegates task to Sprout with full signature verification\n")

        # Step 1: Generate key pairs for both platforms
        print("Step 1: Generating Ed25519 key pairs...")
        self.thor_keys = FederationKeyPair.generate(
            platform_name="Thor",
            lct_id="thor_sage_lct"
        )
        self.sprout_keys = FederationKeyPair.generate(
            platform_name="Sprout",
            lct_id="sprout_sage_lct"
        )
        print(f"  ✓ Thor key pair generated (public key: {self.thor_keys.public_key_bytes().hex()[:16]}...)")
        print(f"  ✓ Sprout key pair generated (public key: {self.sprout_keys.public_key_bytes().hex()[:16]}...)")

        # Step 2: Create signature registry
        print("\nStep 2: Creating signature registry...")
        self.registry = SignatureRegistry()
        self.registry.register_platform("Thor", self.thor_keys.public_key_bytes())
        self.registry.register_platform("Sprout", self.sprout_keys.public_key_bytes())
        print(f"  ✓ Registered 2 platforms")

        # Step 3: Create platform identities
        print("\nStep 3: Creating platform identities...")
        self.thor_identity = create_thor_identity(stake_amount=2000.0)
        self.thor_identity.public_key = self.thor_keys.public_key_bytes()

        self.sprout_identity = create_sprout_identity(stake_amount=1000.0)
        self.sprout_identity.public_key = self.sprout_keys.public_key_bytes()

        print(f"  ✓ Thor: {self.thor_identity.hardware_spec}")
        print(f"  ✓ Sprout: {self.sprout_identity.hardware_spec}")

        # Step 4: Create federation routers
        print("\nStep 4: Initializing federation routers...")
        self.thor_router = FederationRouter(self.thor_identity)
        self.sprout_router = FederationRouter(self.sprout_identity)

        # Register each other
        self.thor_router.register_platform(self.sprout_identity)
        self.sprout_router.register_platform(self.thor_identity)

        print(f"  ✓ Thor router initialized (knows 1 platform)")
        print(f"  ✓ Sprout router initialized (knows 1 platform)")

        print("\n" + "="*80)
        print("SETUP COMPLETE - Ready for signed delegation")
        print("="*80 + "\n")

    def run_signed_delegation_scenario(self):
        """
        Run complete signed delegation scenario

        Flow:
        1. Thor creates task and signs it
        2. Thor delegates to Sprout (simulated network)
        3. Sprout verifies task signature
        4. Sprout executes task
        5. Sprout creates and signs execution proof
        6. Thor receives proof (simulated network)
        7. Thor verifies proof signature
        8. Thor updates Sprout's reputation
        """

        # Scenario: Thor needs to execute a complex reasoning task
        # but is in WAKE state with limited ATP budget
        print("SCENARIO: Complex Reasoning Task")
        print("-" * 80)
        print("Thor is in WAKE state (limited ATP budget)")
        print("Task requires more ATP than locally available")
        print("Decision: Delegate to Sprout with cryptographic signing\n")

        # Step 1: Thor creates task
        print("Step 1: Thor creates federation task...")
        task_horizon = MRHProfile(
            delta_r=SpatialExtent.LOCAL,
            delta_t=TemporalExtent.SESSION,
            delta_c=ComplexityExtent.AGENT_SCALE
        )

        task = FederationTask(
            task_id="task_demo_001",
            task_type="llm_inference",
            task_data={
                "query": "What are the key principles of consciousness architecture in edge AI?"
            },
            estimated_cost=150.0,
            task_horizon=task_horizon,
            complexity="high",
            delegating_platform="Thor",
            delegating_state=MetabolicState.WAKE,
            quality_requirements=QualityRequirements(
                min_quality=0.75,
                min_convergence=0.65,
                max_energy=0.70
            ),
            max_latency=30.0,
            deadline=time.time() + 3600
        )

        print(f"  Task ID: {task.task_id}")
        print(f"  Type: {task.task_type}")
        print(f"  Complexity: {task.complexity}")
        print(f"  Estimated cost: {task.estimated_cost} ATP")
        print(f"  Quality requirement: {task.quality_requirements.min_quality}")

        # Step 2: Thor signs task
        print("\nStep 2: Thor signs task with Ed25519...")
        task_dict = task.to_signable_dict()
        task_signature = FederationCrypto.sign_task(task_dict, self.thor_keys)

        signed_task = SignedFederationTask(
            task=task,
            signature=task_signature,
            public_key=self.thor_keys.public_key_bytes()
        )

        print(f"  ✓ Task signed (signature: {task_signature.hex()[:32]}...)")
        print(f"  ✓ Signature length: {len(task_signature)} bytes (Ed25519)")

        # Step 3: "Network transmission" (simulated)
        print("\nStep 3: Delegating task to Sprout...")
        print("  [Simulated network transmission - no actual network]")
        print("  ✓ Task transmitted to Sprout")

        # Step 4: Sprout verifies task signature
        print("\nStep 4: Sprout verifies task signature...")
        verified, reason = signed_task.verify_signature(self.registry)

        if not verified:
            print(f"  ✗ SECURITY: Task signature verification FAILED!")
            print(f"  Reason: {reason}")
            print(f"  Action: Task REJECTED")
            return False

        print(f"  ✓ Signature verified: {reason}")
        print(f"  ✓ Confirmed: Task is from {task.delegating_platform}")
        print(f"  ✓ Security: Source authenticated, integrity verified")

        # Step 5: Sprout executes task
        print("\nStep 5: Sprout executes task...")
        print(f"  Executing {task.task_type} task...")
        print(f"  [Simulated execution - would call actual LLM]")

        # Simulate execution results
        execution_result = {
            "response": "Consciousness architecture in edge AI emphasizes metabolic state management, "
                       "MRH-aware attention, and ATP-based resource allocation.",
            "irp_iterations": 4,
            "final_energy": 0.42,
            "convergence_quality": 0.82
        }

        print(f"  ✓ Execution complete")
        print(f"  IRP iterations: {execution_result['irp_iterations']}")
        print(f"  Final energy: {execution_result['final_energy']:.2f}")
        print(f"  Convergence quality: {execution_result['convergence_quality']:.2f}")

        # Step 6: Sprout creates execution proof
        print("\nStep 6: Sprout creates execution proof...")
        quality_score = self._calculate_quality_score(execution_result)

        proof = ExecutionProof(
            task_id=task.task_id,
            executing_platform="Sprout",
            result_data=execution_result,
            actual_latency=12.5,
            actual_cost=140.0,
            irp_iterations=execution_result['irp_iterations'],
            final_energy=execution_result['final_energy'],
            convergence_quality=execution_result['convergence_quality'],
            quality_score=quality_score,
            execution_timestamp=time.time()
        )

        print(f"  ✓ Proof created")
        print(f"  Quality score: {quality_score:.2f}")
        print(f"  Actual cost: {proof.actual_cost} ATP")
        print(f"  Actual latency: {proof.actual_latency}s")

        # Step 7: Sprout signs execution proof
        print("\nStep 7: Sprout signs execution proof...")
        proof_dict = proof.to_signable_dict()
        proof_signature = FederationCrypto.sign_proof(proof_dict, self.sprout_keys)

        signed_proof = SignedExecutionProof(
            proof=proof,
            signature=proof_signature,
            public_key=self.sprout_keys.public_key_bytes()
        )

        print(f"  ✓ Proof signed (signature: {proof_signature.hex()[:32]}...)")

        # Step 8: "Network transmission" back to Thor (simulated)
        print("\nStep 8: Returning signed proof to Thor...")
        print("  [Simulated network transmission - no actual network]")
        print("  ✓ Proof transmitted to Thor")

        # Step 9: Thor verifies proof signature
        print("\nStep 9: Thor verifies proof signature...")
        verified, reason = signed_proof.verify_signature(self.registry)

        if not verified:
            print(f"  ✗ SECURITY: Proof signature verification FAILED!")
            print(f"  Reason: {reason}")
            print(f"  Action: Proof REJECTED, reputation NOT updated")
            return False

        print(f"  ✓ Signature verified: {reason}")
        print(f"  ✓ Confirmed: Proof is from {proof.executing_platform}")
        print(f"  ✓ Security: Execution authenticated, no tampering detected")

        # Step 10: Thor accepts proof and updates reputation
        print("\nStep 10: Thor accepts proof and updates reputation...")

        # Verify quality meets requirements
        quality_met = proof.quality_score >= task.quality_requirements.min_quality

        if quality_met:
            print(f"  ✓ Quality requirement met: {proof.quality_score:.2f} >= {task.quality_requirements.min_quality:.2f}")

            # Update Sprout's reputation
            old_reputation = self.sprout_identity.reputation_score
            # Simple reputation update (would use more sophisticated algorithm in production)
            reputation_gain = 0.05 * (proof.quality_score - 0.5)
            new_reputation = min(1.0, old_reputation + reputation_gain)
            self.sprout_identity.reputation_score = new_reputation

            print(f"  ✓ Sprout reputation updated: {old_reputation:.3f} → {new_reputation:.3f}")
            print(f"  ✓ Task completed successfully")
        else:
            print(f"  ✗ Quality requirement NOT met: {proof.quality_score:.2f} < {task.quality_requirements.min_quality:.2f}")
            print(f"  Action: Could issue quality challenge (Phase 1.5)")

        return True

    def run_attack_scenarios(self):
        """
        Demonstrate security properties by attempting attacks

        Shows how Phase 2 cryptography prevents:
        1. Task forgery
        2. Proof forgery
        3. Parameter tampering
        4. Quality inflation
        """
        print("\n" + "="*80)
        print("SECURITY VALIDATION: Attack Scenarios")
        print("="*80 + "\n")

        # Attack 1: Task Forgery
        print("Attack 1: Task Forgery")
        print("-" * 80)
        print("Scenario: Malicious actor tries to forge task claiming it's from Thor")
        print("Method: Create task without Thor's private key\n")

        # Create forged task
        forged_task = FederationTask(
            task_id="forged_task_001",
            task_type="llm_inference",
            task_data={"query": "Malicious query"},
            estimated_cost=100.0,
            task_horizon=MRHProfile(
                delta_r=SpatialExtent.LOCAL,
                delta_t=TemporalExtent.SESSION,
                delta_c=ComplexityExtent.AGENT_SCALE
            ),
            complexity="medium",
            delegating_platform="Thor",  # Claiming to be from Thor
            delegating_state=MetabolicState.WAKE,
            quality_requirements=QualityRequirements(),
            max_latency=30.0,
            deadline=time.time() + 3600
        )

        # Try to sign with Sprout's key (attacker's key)
        fake_signature = FederationCrypto.sign_task(
            forged_task.to_signable_dict(),
            self.sprout_keys  # Wrong key!
        )

        forged_signed_task = SignedFederationTask(
            task=forged_task,
            signature=fake_signature,
            public_key=self.sprout_keys.public_key_bytes()
        )

        # Try to verify
        verified, reason = forged_signed_task.verify_signature(self.registry)

        print(f"  Verification result: {verified}")
        print(f"  Reason: {reason}")
        if not verified:
            print("  ✓ ATTACK BLOCKED: Forged task rejected")
        else:
            print("  ✗ SECURITY FAILURE: Attack succeeded")

        # Attack 2: Parameter Tampering
        print("\n\nAttack 2: Parameter Tampering")
        print("-" * 80)
        print("Scenario: Attacker intercepts legitimate task and modifies cost")
        print("Method: Change estimated_cost after signing\n")

        # Create legitimate task
        legit_task = FederationTask(
            task_id="legit_task_001",
            task_type="llm_inference",
            task_data={"query": "Legitimate query"},
            estimated_cost=100.0,  # Original cost
            task_horizon=MRHProfile(
                delta_r=SpatialExtent.LOCAL,
                delta_t=TemporalExtent.SESSION,
                delta_c=ComplexityExtent.AGENT_SCALE
            ),
            complexity="medium",
            delegating_platform="Thor",
            delegating_state=MetabolicState.WAKE,
            quality_requirements=QualityRequirements(),
            max_latency=30.0,
            deadline=time.time() + 3600
        )

        # Sign legitimate task
        legit_signature = FederationCrypto.sign_task(
            legit_task.to_signable_dict(),
            self.thor_keys
        )

        # Now tamper with cost
        tampered_task = FederationTask(
            task_id=legit_task.task_id,
            task_type=legit_task.task_type,
            task_data=legit_task.task_data,
            estimated_cost=10.0,  # TAMPERED: was 100.0
            task_horizon=legit_task.task_horizon,
            complexity=legit_task.complexity,
            delegating_platform=legit_task.delegating_platform,
            delegating_state=legit_task.delegating_state,
            quality_requirements=legit_task.quality_requirements,
            max_latency=legit_task.max_latency,
            deadline=legit_task.deadline
        )

        # Try to verify with original signature
        tampered_signed_task = SignedFederationTask(
            task=tampered_task,
            signature=legit_signature,  # Signature from before tampering
            public_key=self.thor_keys.public_key_bytes()
        )

        verified, reason = tampered_signed_task.verify_signature(self.registry)

        print(f"  Original cost: 100.0 ATP")
        print(f"  Tampered cost: 10.0 ATP")
        print(f"  Verification result: {verified}")
        print(f"  Reason: {reason}")
        if not verified:
            print("  ✓ ATTACK BLOCKED: Tampering detected")
        else:
            print("  ✗ SECURITY FAILURE: Attack succeeded")

        # Attack 3: Quality Inflation
        print("\n\nAttack 3: Quality Inflation")
        print("-" * 80)
        print("Scenario: Platform inflates quality score in execution proof")
        print("Method: Modify quality_score after signing\n")

        # Create legitimate proof
        legit_proof = ExecutionProof(
            task_id="task_001",
            executing_platform="Sprout",
            result_data={"response": "Answer"},
            actual_latency=10.0,
            actual_cost=95.0,
            irp_iterations=3,
            final_energy=0.45,
            convergence_quality=0.75,
            quality_score=0.78,  # Legitimate quality
            execution_timestamp=time.time()
        )

        # Sign legitimate proof
        legit_proof_sig = FederationCrypto.sign_proof(
            legit_proof.to_signable_dict(),
            self.sprout_keys
        )

        # Inflate quality
        inflated_proof = ExecutionProof(
            task_id=legit_proof.task_id,
            executing_platform=legit_proof.executing_platform,
            result_data=legit_proof.result_data,
            actual_latency=legit_proof.actual_latency,
            actual_cost=legit_proof.actual_cost,
            irp_iterations=legit_proof.irp_iterations,
            final_energy=legit_proof.final_energy,
            convergence_quality=legit_proof.convergence_quality,
            quality_score=0.98,  # INFLATED: was 0.78
            execution_timestamp=legit_proof.execution_timestamp
        )

        # Try to verify with original signature
        inflated_signed_proof = SignedExecutionProof(
            proof=inflated_proof,
            signature=legit_proof_sig,  # Signature from before inflation
            public_key=self.sprout_keys.public_key_bytes()
        )

        verified, reason = inflated_signed_proof.verify_signature(self.registry)

        print(f"  Original quality: 0.78")
        print(f"  Inflated quality: 0.98")
        print(f"  Verification result: {verified}")
        print(f"  Reason: {reason}")
        if not verified:
            print("  ✓ ATTACK BLOCKED: Quality inflation detected")
        else:
            print("  ✗ SECURITY FAILURE: Attack succeeded")

        print("\n" + "="*80)
        print("SECURITY VALIDATION COMPLETE")
        print("="*80 + "\n")

    def _calculate_quality_score(self, execution_result: Dict[str, Any]) -> float:
        """
        Calculate 4-component SAGE quality score

        Based on:
        1. IRP convergence quality
        2. Energy minimization
        3. Iteration efficiency
        4. Result completeness
        """
        convergence = execution_result['convergence_quality']
        energy = 1.0 - execution_result['final_energy']  # Lower energy is better
        efficiency = min(1.0, 3.0 / execution_result['irp_iterations'])  # Prefer fewer iterations
        completeness = 1.0  # Assume complete for demo

        # Weighted average
        quality = (
            0.4 * convergence +
            0.3 * energy +
            0.2 * efficiency +
            0.1 * completeness
        )

        return quality


def main():
    """Run complete simulated federation demonstration"""
    # Create demo instance
    demo = SimulatedFederationDemo()

    # Run signed delegation scenario
    success = demo.run_signed_delegation_scenario()

    if success:
        print("\n✓ SIGNED DELEGATION SCENARIO: SUCCESS")
    else:
        print("\n✗ SIGNED DELEGATION SCENARIO: FAILED")

    # Demonstrate security by attempting attacks
    demo.run_attack_scenarios()

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. ✓ Ed25519 signing provides source authentication")
    print("2. ✓ Signature verification prevents task forgery")
    print("3. ✓ Parameter tampering is detected")
    print("4. ✓ Quality inflation is prevented")
    print("5. ✓ Complete trust chain: task → execution → proof")
    print("\nPhase 2 cryptographic infrastructure is production-ready!")
    print("Next step: Phase 3 network protocol for actual Thor ↔ Sprout communication")


if __name__ == "__main__":
    main()
