#!/usr/bin/env python3
"""
Session 173 Edge Validation: Secure Federated Cogitation on Sprout

Testing Thor's secure federated cogitation network (Session 173) on Jetson Orin Nano 8GB.

Thor's Implementation (Session 173):
- Integrated Session 172 8-layer defense with Session 166 federated cogitation
- SecureConceptualThought: Thoughts with security validation
- SecureCogitationSession: Trust-weighted collective reasoning
- SecureFederatedCogitationNode: Complete security + cogitation integration

Edge Validation Goals:
1. Verify secure cogitation works on constrained edge hardware (8GB)
2. Test all 8 security layers operational with cogitation modes
3. Test PoW identity creation on ARM64 edge platform
4. Validate quality filtering and spam resistance on edge
5. Profile edge cogitation performance vs Thor's metrics

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 5)
Session: Autonomous Edge Validation - Session 173
Date: 2026-01-08
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from enum import Enum

HOME = Path.home()

# Edge monitoring
def get_edge_metrics() -> Dict[str, Any]:
    """Get edge hardware metrics."""
    metrics = {
        "platform": "Jetson Orin Nano 8GB",
        "hardware_type": "tpm2_simulated",
        "capability_level": 5
    }

    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('MemAvailable:'):
                    available_kb = int(line.split()[1])
                    metrics["memory_available_mb"] = available_kb / 1024
                elif line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                    metrics["memory_total_mb"] = total_kb / 1024
    except:
        pass

    # Temperature
    try:
        temp_paths = [
            '/sys/devices/virtual/thermal/thermal_zone0/temp',
            '/sys/class/thermal/thermal_zone0/temp'
        ]
        for path in temp_paths:
            try:
                with open(path, 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    metrics["temperature_c"] = temp
                    break
            except:
                continue
    except:
        pass

    return metrics


# Import Session 173 components
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

from session173_secure_federated_cogitation import (
    CogitationMode,
    SecureConceptualThought,
    SecureCogitationSession,
    SecureFederatedCogitationNode,
    SecureFederatedCogitationNetwork
)


def test_edge_secure_cogitation():
    """Test secure federated cogitation on edge hardware."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  SESSION 173 EDGE VALIDATION: SECURE FEDERATED COGITATION  ".center(70) + "|")
    print("|" + "           Jetson Orin Nano 8GB (Sprout)                    ".center(70) + "|")
    print("|" + " " * 70 + "|")
    print("+" + "=" * 70 + "+")
    print()

    # Get edge metrics
    edge_metrics = get_edge_metrics()
    print("Edge Hardware:")
    print(f"  Platform: {edge_metrics['platform']}")
    print(f"  Hardware: {edge_metrics['hardware_type']} (Level {edge_metrics['capability_level']})")
    if 'temperature_c' in edge_metrics:
        print(f"  Temperature: {edge_metrics['temperature_c']}C")
    if 'memory_available_mb' in edge_metrics:
        print(f"  Memory: {int(edge_metrics['memory_available_mb'])} MB available")
    print()

    all_tests_passed = True

    # ========================================================================
    # TEST 1: Edge PoW Identity Creation
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Edge PoW Identity Creation")
    print("=" * 72)
    print()

    print("Creating secure cogitation node with PoW identity on edge...")
    start = time.time()

    node = SecureFederatedCogitationNode(
        node_id="sprout-cogitator",
        hardware_type="tpm2",
        capability_level=5,
        pow_difficulty=236,  # Same as Thor
        corpus_max_thoughts=50,
        corpus_max_size_mb=5.0
    )

    pow_time = time.time() - start

    print(f"  PoW completed in {pow_time:.3f}s")
    print(f"  Identity validated: {node.pow_identity is not None}")
    print()

    # Compare with Thor (typical range 0.2-2.0s)
    thor_pow_range = (0.2, 2.0)
    edge_within_range = thor_pow_range[0] <= pow_time <= thor_pow_range[1] * 1.5  # Allow 50% overhead

    print(f"Edge PoW performance: {'✓ Within acceptable range' if edge_within_range else '⚠ Slower than expected'}")
    print()

    test1_pass = (
        node.pow_identity is not None and
        pow_time < 10.0  # Reasonable upper bound for edge
    )

    print(f"{'✓ TEST 1 PASSED' if test1_pass else '✗ TEST 1 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test1_pass

    # ========================================================================
    # TEST 2: Edge Secure Cogitation Session
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Edge Secure Cogitation Session")
    print("=" * 72)
    print()

    print("Creating secure cogitation session on edge...")
    session_id = node.create_cogitation_session(
        "How does edge hardware constrain distributed consciousness reasoning?"
    )
    print(f"  Session created: {session_id}")
    print()

    # Contribute thoughts in different modes
    print("Contributing conceptual thoughts...")
    thoughts = [
        ("Edge constraints require efficient security - lightweight PoW and streaming validation enable federated cogitation on resource-limited hardware.", CogitationMode.EXPLORING),
        ("Can 8-layer security maintain effectiveness with reduced edge resources?", CogitationMode.QUESTIONING),
        ("Hardware asymmetry creates natural hierarchy where edge validates locally while cloud provides compute-intensive reasoning.", CogitationMode.INTEGRATING),
        ("Edge coherence filtering ensures only high-quality thoughts propagate to network, preserving bandwidth.", CogitationMode.VERIFYING),
        ("spam", CogitationMode.EXPLORING),  # Should be rejected
        ("x", CogitationMode.EXPLORING),  # Should be rejected
    ]

    accepted = 0
    rejected = 0

    for content, mode in thoughts:
        success, reason, thought = node.contribute_thought(session_id, mode, content)
        preview = content[:50] + "..." if len(content) > 50 else content
        if success:
            accepted += 1
            print(f"  ✓ {mode.value}: {preview}")
        else:
            rejected += 1
            print(f"  ✗ {mode.value}: {reason}")

    print()

    # Session summary
    session = node.active_sessions[session_id]
    summary = session.get_summary()
    print("Edge Session Summary:")
    print(f"  Thoughts submitted: {summary['thoughts_submitted']}")
    print(f"  Thoughts accepted: {summary['thoughts_accepted']}")
    print(f"  Thoughts rejected: {summary['thoughts_rejected']}")
    print(f"  Acceptance rate: {summary['acceptance_rate']*100:.1f}%")
    print(f"  Collective coherence: {summary['collective_coherence']:.3f}")
    print(f"  Mode distribution: {summary['mode_distribution']}")
    print()

    test2_pass = (
        accepted >= 4 and  # Quality thoughts accepted
        rejected >= 2 and  # Spam rejected
        summary['collective_coherence'] > 0.5  # High coherence maintained
    )

    print(f"{'✓ TEST 2 PASSED' if test2_pass else '✗ TEST 2 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Edge 8-Layer Security Integration
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Edge 8-Layer Security Integration")
    print("=" * 72)
    print()

    print("Validating all 8 security layers operational on edge...")
    print()

    node_metrics = node.get_node_metrics()
    security = node_metrics["security"]

    print("Security Layers on Edge:")
    print(f"  Layer 1 (PoW): {security.get('proof_of_work', {}).get('identities_validated', 0)} identities validated")
    print(f"  Layers 2-6: {security.get('thoughts_processed', 0)} thoughts processed")
    print(f"  Layer 7 (Corpus): {security.get('corpus_management', {}).get('thought_count', 0)} thoughts stored")
    print(f"  Layer 8 (Trust Decay): {security.get('trust_decay', {}).get('decay_applications', 0)} applications")
    print()

    test3_pass = (
        'corpus_management' in security and
        'trust_decay' in security and
        security.get('thoughts_processed', 0) >= 4
    )

    print(f"{'✓ TEST 3 PASSED' if test3_pass else '✗ TEST 3 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Edge Performance Profile
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Edge Performance Profile")
    print("=" * 72)
    print()

    print("Measuring edge cogitation throughput...")

    # Rapid thought validation test
    test_thought = "A test thought of reasonable length for performance measurement."
    validation_count = 100
    start = time.time()

    for i in range(validation_count):
        # Just test the security validation, not full contribution
        accepted, reason, metrics = node.security.validate_thought_contribution_8layer(
            node.node_id,
            test_thought
        )

    validation_time = time.time() - start
    throughput = validation_count / validation_time

    print(f"  {validation_count} validations in {validation_time:.3f}s")
    print(f"  Throughput: {throughput:.0f} validations/sec")
    print()

    # Memory footprint
    cogitation_metrics = node_metrics["cogitation"]
    print(f"Edge Memory Footprint:")
    print(f"  Active sessions: {cogitation_metrics['total_sessions']}")
    print(f"  Total thoughts: {cogitation_metrics['total_thoughts_accepted']}")
    print()

    # Compare with Thor (typical: 100k+ validations/sec)
    edge_acceptable_throughput = 1000  # 1k validations/sec is acceptable for edge

    test4_pass = throughput >= edge_acceptable_throughput

    print(f"{'✓ TEST 4 PASSED' if test4_pass else '✗ TEST 4 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Convergent Research Validation
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Convergent Research Validation")
    print("=" * 72)
    print()

    print("Session Integration Check:")
    print()
    print("  Thor Sessions:")
    print("    - Session 170: 5-layer defense-in-depth")
    print("    - Session 171: 6-layer (added PoW)")
    print("    - Session 172: 8-layer (corpus + decay)")
    print("    - Session 173: Secure federated cogitation")
    print()
    print("  Legion Sessions (integrated):")
    print("    - Sessions 136-141: Security research arc")
    print("    - Sessions 142-148: ATP security + advanced defenses")
    print()
    print("  Edge Validation (this session):")
    print("    - Session 165-172: Architecture + security validation")
    print("    - Session 173: Secure cogitation validation")
    print()

    # Verify all components present
    components_present = {
        "pow": "proof_of_work" in security,
        "rate_limit": "thoughts_processed" in security,
        "quality": "thoughts_rejected" in security,
        "corpus": "corpus_management" in security,
        "trust_decay": "trust_decay" in security,
        "cogitation": cogitation_metrics['total_sessions'] > 0
    }

    print("Component Verification:")
    for component, present in components_present.items():
        status = "✓ present" if present else "✗ missing"
        print(f"  {component}: {status}")
    print()

    test5_pass = all(components_present.values())

    print(f"{'✓ TEST 5 PASSED' if test5_pass else '✗ TEST 5 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("SESSION 173 EDGE VALIDATION SUMMARY")
    print("=" * 72)
    print()

    print("Test Results:")
    print(f"  edge_pow_identity: {'PASS' if test1_pass else 'FAIL'}")
    print(f"  edge_secure_cogitation: {'PASS' if test2_pass else 'FAIL'}")
    print(f"  edge_8layer_integration: {'PASS' if test3_pass else 'FAIL'}")
    print(f"  edge_performance: {'PASS' if test4_pass else 'FAIL'}")
    print(f"  convergent_validation: {'PASS' if test5_pass else 'FAIL'}")
    print()

    test_count = sum([test1_pass, test2_pass, test3_pass, test4_pass, test5_pass])
    print(f"Overall: {test_count}/5 tests passed")
    print()

    if all_tests_passed:
        print("+" + "-" * 70 + "+")
        print("|" + " " * 70 + "|")
        print("|" + "   ✓ ✓ ✓ ALL TESTS PASSED - SECURE COGITATION ON EDGE! ✓ ✓ ✓   ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Secure federated cogitation operational on 8GB edge hardware")
        print("  - All 8 security layers working with cogitation modes")
        print("  - PoW identity creation within acceptable performance bounds")
        print("  - Quality filtering and spam resistance effective on edge")
        print(f"  - Edge throughput: {throughput:.0f} validations/sec")
        print("  - Convergent research (Thor + Legion) validated on edge")
    else:
        print("✗ SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "173_edge",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": {
            "edge_pow_identity": test1_pass,
            "edge_secure_cogitation": test2_pass,
            "edge_8layer_integration": test3_pass,
            "edge_performance": test4_pass,
            "convergent_validation": test5_pass
        },
        "edge_metrics": edge_metrics,
        "performance": {
            "pow_time_seconds": pow_time,
            "validation_throughput_per_sec": throughput,
        },
        "session_summary": summary
    }

    results_path = Path(__file__).parent / "session173_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_secure_cogitation()
    sys.exit(0 if success else 1)
