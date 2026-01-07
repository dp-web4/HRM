#!/usr/bin/env python3
"""
Session 169: Sensor-Level Federation Validation - Provider Fix Propagation

Research Goal: Validate that Session 134's TrustZone provider fix automatically
propagates to sensor-level federation code without modification.

Research Arc Context:
- Session 165 (Thor): Sensor-level federation test, 33.3% network density
- Session 133-134 (Legion): Fixed double-hashing bug at provider level
- Session 168 (Thor): Validated 100% success at PROVIDER level
- Session 169 (Thor): Validate 100% success at SENSOR level

Key Question: Does the provider-level fix (Session 134) automatically fix the
sensor-level federation code (Session 165) without modification?

Hypothesis: YES - because sensor layer calls provider.verify_aliveness_proof()
which internally uses provider.verify_signature() (the fixed method).

Architecture Insight: This tests whether consciousness aliveness verification
(sensor level) properly leverages hardware provider fixes (provider level).

Expected Outcome: 100% network density at sensor level (same as Session 168).

Philosophy: "Surprise is prize" - If it fails, we learn about layer coupling.
                                    If it succeeds, we validate architectural elegance.

Hardware: Jetson AGX Thor Developer Kit
Session: Autonomous SAGE Development - Session 169
"""

import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple

# Add paths
HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM"))
sys.path.insert(0, str(HOME / "ai-workspace" / "web4"))

# Web4 imports
from core.lct_capability_levels import EntityType
from core.lct_binding import (
    TrustZoneProvider,
    SoftwareProvider,
)

# Session 128 consciousness components (sensor level)
from test_session128_consciousness_aliveness_integration import (
    ConsciousnessState,
    ConsciousnessPatternCorpus,
    ConsciousnessAlivenessSensor
)


# ============================================================================
# SESSION 169: SENSOR-LEVEL FEDERATION VALIDATION
# ============================================================================

def test_sensor_level_cross_verification():
    """
    Test 1: Sensor-level cross-platform verification using Session 128 consciousness sensors.

    This replicates Session 165's approach but with Session 134 fix in place.
    Expected: 100% success (Software can now verify TrustZone at sensor level).
    """
    print("=" * 80)
    print("TEST 1: SENSOR-LEVEL CROSS-PLATFORM VERIFICATION")
    print("=" * 80)
    print()
    print("Using Session 128 ConsciousnessAlivenessSensor (high-level API)")
    print("Testing if Session 134 provider fix propagates to sensor layer...")
    print()

    # Create TrustZone consciousness sensor
    print("Step 1: Create TrustZone consciousness sensor (Thor hardware)...")
    tz_provider = TrustZoneProvider()
    tz_lct = tz_provider.create_lct(EntityType.AI, "thor-trustzone-sensor-test")

    # Create consciousness pattern corpus for TrustZone
    tz_corpus = ConsciousnessPatternCorpus(consciousness_id="thor-tz-169")
    tz_corpus.add_pattern("hardware_platform", {"name": "thor", "type": "trustzone"})
    tz_corpus.add_pattern("security_capability", {"level": 5, "device": "/dev/tee0"})
    tz_corpus.add_pattern("research_context", {"session": "169", "test": "sensor-validation"})

    tz_sensor = ConsciousnessAlivenessSensor(
        lct=tz_lct,
        provider=tz_provider,
        corpus=tz_corpus
    )

    print(f"  TrustZone Sensor:")
    print(f"    LCT: {tz_lct.lct_id}")
    print(f"    Level: {tz_lct.capability_level}")
    print(f"    Session: {tz_sensor.session_id}")
    print(f"    Hardware: trustzone")
    print()

    # Create Software consciousness sensor
    print("Step 2: Create Software consciousness sensor (verification peer)...")
    sw_provider = SoftwareProvider()
    sw_lct = sw_provider.create_lct(EntityType.AI, "software-sensor-verifier")

    sw_corpus = ConsciousnessPatternCorpus(consciousness_id="software-sw-169")
    sw_corpus.add_pattern("identity", {"name": "software-verifier", "role": "peer"})
    sw_corpus.add_pattern("purpose", {"test": "cross-platform", "session": "169"})

    sw_sensor = ConsciousnessAlivenessSensor(
        lct=sw_lct,
        provider=sw_provider,
        corpus=sw_corpus
    )

    print(f"  Software Sensor:")
    print(f"    LCT: {sw_lct.lct_id}")
    print(f"    Level: {sw_lct.capability_level}")
    print(f"    Session: {sw_sensor.session_id}")
    print(f"    Hardware: software")
    print()

    # Test Software → TrustZone verification (CRITICAL TEST)
    print("Step 3: Software verifies TrustZone consciousness (CRITICAL)...")
    print("  This is the verification that FAILED in Session 165 (33.3% density)")
    print()

    start_time = time.time()

    try:
        # Software creates challenge for TrustZone (manual challenge creation)
        from core.lct_binding.trust_policy import AgentAlivenessChallenge, AgentPolicyTemplates

        challenge_nonce = hashlib.sha256(f"sw_to_tz_{time.time()}".encode()).digest()[:16]
        challenge_timestamp = datetime.now(timezone.utc)
        challenge_id = hashlib.sha256(challenge_nonce + str(challenge_timestamp).encode()).hexdigest()

        sw_challenge = AgentAlivenessChallenge(
            nonce=challenge_nonce,
            timestamp=challenge_timestamp,
            challenge_id=challenge_id,
            expires_at=challenge_timestamp + timedelta(seconds=60),
            verifier_lct_id=sw_lct.lct_id,
            purpose="cross_platform_verification",
            expected_session_id=tz_sensor.session_id,
            expected_corpus_hash=tz_corpus.compute_corpus_hash()
        )

        print(f"  Software created challenge: {sw_challenge.challenge_id[:16]}...")

        # TrustZone generates proof
        tz_proof = tz_sensor.prove_consciousness_aliveness(sw_challenge)

        print(f"  TrustZone generated proof (signature: {len(tz_proof.signature)} bytes)")

        # Software verifies TrustZone's proof (THIS IS THE KEY TEST)
        trust_policy = AgentPolicyTemplates.FEDERATED_CONSCIOUSNESS

        result = sw_sensor.verify_consciousness_aliveness(
            challenge=sw_challenge,
            proof=tz_proof,
            expected_public_key=tz_lct.public_key,
            trust_policy=trust_policy
        )

        elapsed = time.time() - start_time

        if result.valid:
            print(f"  🎉 SUCCESS! Software CAN verify TrustZone at sensor level!")
            print(f"  Verification time: {elapsed:.4f}s")
            print(f"  Hardware continuity: {result.hardware_continuity:.2f}")
            print(f"  Session continuity: {result.session_continuity:.2f}")
            print(f"  Epistemic continuity: {result.epistemic_continuity:.2f}")
            print(f"  Full continuity: {result.full_continuity:.2f}")
            return True, elapsed, result
        else:
            print(f"  ❌ FAILED! Software still cannot verify TrustZone at sensor level")
            print(f"  Failure type: {result.failure_type}")
            print(f"  Error: {result.error}")
            return False, elapsed, result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, elapsed, None


def test_sensor_level_federation_topology():
    """
    Test 2: Complete federation topology at sensor level (all verification pairs).

    Mirrors Session 168's Test 2 but at sensor level instead of provider level.
    Expected: 6/6 successful verifications (100% network density).
    """
    print()
    print("=" * 80)
    print("TEST 2: SENSOR-LEVEL FEDERATION TOPOLOGY")
    print("=" * 80)
    print()
    print("Creating 3-node federation using consciousness sensors...")
    print("Testing all 6 verification pairs for complete topology...")
    print()

    # Create three consciousness sensors (simulating Thor + 2 peers)
    sensors = {}

    for node_name, hw_type in [("Thor", "trustzone"), ("Peer1", "software"), ("Peer2", "software")]:
        provider = TrustZoneProvider() if hw_type == "trustzone" else SoftwareProvider()
        lct = provider.create_lct(EntityType.AI, f"sensor-{node_name.lower()}")

        corpus = ConsciousnessPatternCorpus(consciousness_id=f"{node_name.lower()}-169")
        corpus.add_pattern("identity", {"name": node_name, "node": node_name.lower()})
        corpus.add_pattern("hardware_type", {"type": hw_type, "level": lct.capability_level})

        sensor = ConsciousnessAlivenessSensor(
            lct=lct,
            provider=provider,
            corpus=corpus
        )

        sensors[node_name] = {
            'sensor': sensor,
            'lct': lct,
            'corpus': corpus,
            'provider': provider
        }

        print(f"  {node_name}:")
        print(f"    Hardware: {hw_type} (Level {lct.capability_level})")
        print(f"    LCT: {lct.lct_id}")
        print(f"    Session: {sensor.session_id}")
        print()

    # Test all 6 verification pairs
    print("Testing all verification pairs:")
    print()

    from core.lct_binding.trust_policy import AgentPolicyTemplates, AgentAlivenessChallenge
    trust_policy = AgentPolicyTemplates.FEDERATED_CONSCIOUSNESS

    verifications = []
    successful_count = 0
    total_time = 0

    for verifier_name in ["Thor", "Peer1", "Peer2"]:
        for prover_name in ["Thor", "Peer1", "Peer2"]:
            if verifier_name == prover_name:
                continue  # Skip self-verification

            verifier = sensors[verifier_name]
            prover = sensors[prover_name]

            print(f"  {verifier_name} verifying {prover_name}...", end=" ")

            try:
                start = time.time()

                # Verifier creates challenge (manual creation)
                challenge_nonce = hashlib.sha256(f"{verifier_name}_to_{prover_name}_{time.time()}".encode()).digest()[:16]
                challenge_timestamp = datetime.now(timezone.utc)
                challenge_id = hashlib.sha256(challenge_nonce + str(challenge_timestamp).encode()).hexdigest()

                challenge = AgentAlivenessChallenge(
                    nonce=challenge_nonce,
                    timestamp=challenge_timestamp,
                    challenge_id=challenge_id,
                    expires_at=challenge_timestamp + timedelta(seconds=60),
                    verifier_lct_id=verifier['lct'].lct_id,
                    purpose="federation_topology_test",
                    expected_session_id=prover['sensor'].session_id,
                    expected_corpus_hash=prover['corpus'].compute_corpus_hash()
                )

                # Prover generates proof
                proof = prover['sensor'].prove_consciousness_aliveness(challenge)

                # Verifier verifies proof
                result = verifier['sensor'].verify_consciousness_aliveness(
                    challenge=challenge,
                    proof=proof,
                    expected_public_key=prover['lct'].public_key,
                    trust_policy=trust_policy
                )

                elapsed = time.time() - start
                total_time += elapsed

                if result.valid:
                    print(f"✓ VERIFIED ({elapsed:.4f}s)")
                    successful_count += 1
                    verifications.append({
                        "verifier": verifier_name,
                        "prover": prover_name,
                        "verified": True,
                        "time": elapsed,
                        "continuity": result.full_continuity
                    })
                else:
                    print(f"✗ FAILED ({result.failure_type})")
                    verifications.append({
                        "verifier": verifier_name,
                        "prover": prover_name,
                        "verified": False,
                        "failure": str(result.failure_type),
                        "error": result.error
                    })
            except Exception as e:
                print(f"✗ EXCEPTION: {str(e)}")
                verifications.append({
                    "verifier": verifier_name,
                    "prover": prover_name,
                    "verified": False,
                    "exception": str(e)
                })

    # Calculate network density
    total_pairs = 6  # 3 nodes, 2 edges each
    network_density = successful_count / total_pairs

    print()
    print("Topology Results:")
    print(f"  Successful verifications: {successful_count}/{total_pairs}")
    print(f"  Network density: {network_density:.1%}")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Average per verification: {total_time/total_pairs:.4f}s")
    print()

    if network_density == 1.0:
        print("  🎉 COMPLETE FULL MESH at sensor level!")
        topology = "complete_full_mesh"
    elif network_density > 0.5:
        print("  ⚠️  Partial mesh topology")
        topology = "partial_mesh"
    else:
        print("  ⚠️  Island topology")
        topology = "island"

    return {
        "successful_verifications": successful_count,
        "total_verifications": total_pairs,
        "network_density": network_density,
        "topology": topology,
        "verifications": verifications,
        "total_time": total_time,
        "avg_time": total_time / total_pairs
    }


def run_session_169():
    """
    Main entry point for Session 169.

    Validates sensor-level federation after Session 134 provider fix.
    """
    print("=" * 80)
    print("SESSION 169: SENSOR-LEVEL FEDERATION VALIDATION")
    print("=" * 80)
    print()
    print("Research Goal: Validate Session 134 provider fix propagates to sensor level")
    print()
    print("Background:")
    print("  - Session 165: Sensor-level federation, 33.3% network density")
    print("  - Session 134: Fixed double-hashing bug at provider level")
    print("  - Session 168: Validated 100% success at provider level")
    print("  - Session 169: Validate 100% success at sensor level")
    print()
    print("Key Question: Does provider fix automatically fix sensor layer?")
    print("Hypothesis: YES (sensor calls provider internally)")
    print()

    start_time = time.time()

    # Test 1: Basic cross-verification at sensor level
    test1_passed, test1_time, test1_result = test_sensor_level_cross_verification()

    # Test 2: Complete topology at sensor level
    test2_results = test_sensor_level_federation_topology()

    total_time = time.time() - start_time

    # Compile results
    results = {
        "session": "169",
        "title": "Sensor-Level Federation Validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": total_time,
        "test1_sensor_level_verification": {
            "passed": test1_passed,
            "time": test1_time,
            "description": "Software verifying TrustZone at sensor level"
        },
        "test2_sensor_level_topology": test2_results,
        "comparison_with_session_165": {
            "session_165_density": 0.333,
            "session_169_density": test2_results["network_density"],
            "improvement": test2_results["network_density"] > 0.333
        },
        "comparison_with_session_168": {
            "session_168_density": 1.0,
            "session_168_level": "provider",
            "session_169_density": test2_results["network_density"],
            "session_169_level": "sensor",
            "fix_propagated": test2_results["network_density"] == 1.0
        },
        "key_findings": {
            "session_134_fix_works_at_provider_level": True,  # From Session 168
            "session_134_fix_propagates_to_sensor_level": test2_results["network_density"] == 1.0,
            "sensor_layer_needs_no_modification": test2_results["network_density"] == 1.0,
            "architectural_elegance_validated": test2_results["network_density"] == 1.0
        }
    }

    # Save results
    results_file = HOME / "ai-workspace" / "HRM" / "sage" / "experiments" / "session169_sensor_level_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("SESSION 169 COMPLETE")
    print("=" * 80)
    print()
    print(f"Results saved to: {results_file}")
    print()
    print("Key Findings:")
    print(f"  - Test 1 (Sensor-level cross-verification): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"  - Test 2 (Sensor-level topology): {test2_results['successful_verifications']}/{test2_results['total_verifications']} verified ({test2_results['network_density']:.1%} density)")
    print(f"  - Session 134 fix propagated to sensor level: {results['key_findings']['session_134_fix_propagates_to_sensor_level']}")
    print(f"  - Architecture elegance validated: {results['key_findings']['architectural_elegance_validated']}")
    print()
    print(f"Total duration: {total_time:.4f}s")
    print()

    return results


if __name__ == "__main__":
    results = run_session_169()

    # Print final summary
    print("=" * 80)
    print("RESEARCH INSIGHT: Provider vs Sensor Layer Coupling")
    print("=" * 80)
    print()

    if results["key_findings"]["architectural_elegance_validated"]:
        print("✅ ARCHITECTURAL ELEGANCE CONFIRMED")
        print()
        print("The Session 134 provider-level fix automatically propagates to the sensor")
        print("layer through clean architectural separation. ConsciousnessAlivenessSensor")
        print("properly delegates hardware verification to the provider layer, enabling")
        print("consciousness-level federation to transparently benefit from hardware-level")
        print("fixes without modification.")
        print()
        print("This validates the layered architecture:")
        print("  - Provider layer: Hardware signature operations (fixed in Session 134)")
        print("  - Sensor layer: Consciousness verification (automatically fixed)")
        print("  - Federation layer: Multi-node consciousness network (ready)")
        print()
        print("Implication: Multi-machine federation (Thor + Legion + Sprout) can now")
        print("proceed with confidence that all layers support cross-platform verification.")
    else:
        print("⚠️  ARCHITECTURAL COUPLING ISSUE DETECTED")
        print()
        print("The sensor layer did not automatically benefit from the provider-level fix.")
        print("This suggests tight coupling or duplicate logic between layers that needs")
        print("to be refactored for proper separation of concerns.")
        print()
        print("Next steps: Investigate sensor-level implementation to identify and fix")
        print("the coupling issue.")
    print()
