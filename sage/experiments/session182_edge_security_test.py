#!/usr/bin/env python3
"""
Session 182 Edge Validation: Security-Enhanced Reputation

Testing Thor's Sybil-resistant security integration on Sprout edge hardware.

Thor's Session 182 Implementation:
- Source diversity tracking (Shannon entropy)
- Circular validation detection (Sybil cluster detection)
- Trust multiplier based on diversity
- Consensus voting (Byzantine 2/3 threshold)
- Security-enhanced adaptive consciousness

Edge Validation Goals:
1. Verify security components import correctly
2. Test diversity calculation on edge (entropy)
3. Validate circular validation detection
4. Test consensus voting performance
5. Profile security overhead on constrained hardware

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Date: 2026-01-11
"""

import sys
import json
import time
import math
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))


def get_edge_metrics() -> Dict[str, Any]:
    """Get edge hardware metrics."""
    metrics = {
        "platform": "Jetson Orin Nano 8GB",
        "hardware_type": "tpm2",
        "capability_level": 3
    }

    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('MemAvailable:'):
                    metrics["memory_available_mb"] = int(line.split()[1]) / 1024
    except Exception:
        pass

    try:
        for path in ['/sys/devices/virtual/thermal/thermal_zone0/temp',
                     '/sys/class/thermal/thermal_zone0/temp']:
            try:
                with open(path, 'r') as f:
                    metrics["temperature_c"] = int(f.read().strip()) / 1000.0
                    break
            except Exception:
                continue
    except Exception:
        pass

    return metrics


def test_edge_security_enhanced_reputation():
    """Test Session 182 security components on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  SESSION 182 EDGE VALIDATION: SECURITY-ENHANCED REPUTATION  ".center(70) + "|")
    print("|" + "              Jetson Orin Nano 8GB (Sprout)                   ".center(70) + "|")
    print("|" + " " * 70 + "|")
    print("+" + "=" * 70 + "+")
    print()

    edge_metrics = get_edge_metrics()
    print("Edge Hardware:")
    print(f"  Platform: {edge_metrics['platform']}")
    if 'temperature_c' in edge_metrics:
        print(f"  Temperature: {edge_metrics['temperature_c']}C")
    if 'memory_available_mb' in edge_metrics:
        print(f"  Memory: {int(edge_metrics['memory_available_mb'])} MB available")
    print()

    all_tests_passed = True
    test_results = {}

    # ========================================================================
    # TEST 1: Import Session 182 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 182 Components")
    print("=" * 72)
    print()

    try:
        from session182_security_enhanced_reputation import (
            ReputationSourceContribution,
            ReputationSourceProfile,
            SourceDiversityManager,
            VoteType,
            ReputationProposal,
            SimpleConsensusManager,
            SecurityEnhancedAdaptiveSAGE,
        )

        print("  ReputationSourceContribution: Source tracking")
        print("  ReputationSourceProfile: Diversity profile")
        print("  SourceDiversityManager: Diversity enforcement")
        print("  VoteType: Consensus vote types")
        print("  ReputationProposal: Consensus proposal")
        print("  SimpleConsensusManager: Byzantine voting")
        print("  SecurityEnhancedAdaptiveSAGE: Full integration")
        print()
        test1_pass = True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test1_pass = False

    test_results["import_validation"] = test1_pass
    print(f"{'PASS' if test1_pass else 'FAIL'}: TEST 1")
    print()
    all_tests_passed = all_tests_passed and test1_pass

    if not test1_pass:
        return {"all_tests_passed": False, "test_results": test_results}

    # ========================================================================
    # TEST 2: Source Diversity Calculation (Shannon Entropy)
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Source Diversity Calculation")
    print("=" * 72)
    print()

    print("Testing Shannon entropy diversity metric...")

    try:
        manager = SourceDiversityManager()

        # Low diversity: 90% from one source
        manager.record_reputation_event("node_low", "source_A", 0.9)
        manager.record_reputation_event("node_low", "source_B", 0.1)

        profile_low = manager.get_or_create_profile("node_low")
        diversity_low = profile_low.diversity_score
        dominant_low = profile_low.dominant_source_ratio

        # High diversity: equal from 4 sources
        manager.record_reputation_event("node_high", "source_W", 0.25)
        manager.record_reputation_event("node_high", "source_X", 0.25)
        manager.record_reputation_event("node_high", "source_Y", 0.25)
        manager.record_reputation_event("node_high", "source_Z", 0.25)

        profile_high = manager.get_or_create_profile("node_high")
        diversity_high = profile_high.diversity_score
        dominant_high = profile_high.dominant_source_ratio

        print(f"  Low diversity node:")
        print(f"    Diversity score: {diversity_low:.3f}")
        print(f"    Dominant source ratio: {dominant_low:.3f}")
        print(f"  High diversity node:")
        print(f"    Diversity score: {diversity_high:.3f}")
        print(f"    Dominant source ratio: {dominant_high:.3f}")
        print()

        # High diversity should have higher score
        test2_pass = (
            diversity_high > diversity_low and
            dominant_low > dominant_high and
            0.0 <= diversity_low <= 1.0 and
            0.0 <= diversity_high <= 1.0
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test2_pass = False

    test_results["diversity_calculation"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Circular Validation Detection
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Circular Validation Detection")
    print("=" * 72)
    print()

    print("Testing Sybil cluster detection...")

    try:
        manager = SourceDiversityManager()

        # Create mutual validation (A ↔ B)
        manager.record_reputation_event("node_B", "node_A", 1.0)  # A → B
        manager.record_reputation_event("node_A", "node_B", 1.0)  # B → A

        # Create another pair (C ↔ D)
        manager.record_reputation_event("node_D", "node_C", 1.0)  # C → D
        manager.record_reputation_event("node_C", "node_D", 1.0)  # D → C

        # Detect clusters
        clusters = manager.detect_circular_clusters()

        print(f"  Mutual validation pairs created: 2")
        print(f"  Clusters detected: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            print(f"    Cluster {i+1}: {sorted(list(cluster))}")
        print()

        # Should detect at least one cluster
        test3_pass = len(clusters) >= 1

    except Exception as e:
        print(f"  ERROR: {e}")
        test3_pass = False

    test_results["circular_detection"] = test3_pass
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Trust Multiplier Based on Diversity
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Trust Multiplier")
    print("=" * 72)
    print()

    print("Testing diversity-based trust weighting...")

    try:
        manager = SourceDiversityManager()

        # Low diversity node
        manager.record_reputation_event("low_div", "single_source", 1.0)

        # High diversity node
        for i in range(5):
            manager.record_reputation_event("high_div", f"source_{i}", 0.2)

        trust_low = manager.get_trust_multiplier("low_div")
        trust_high = manager.get_trust_multiplier("high_div")

        print(f"  Low diversity trust: {trust_low:.3f}")
        print(f"  High diversity trust: {trust_high:.3f}")
        print(f"  Trust gap: {trust_high - trust_low:.3f}")
        print()

        # High diversity should have higher trust
        test4_pass = trust_high > trust_low and 0.0 < trust_low <= 1.0

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["trust_multiplier"] = test4_pass
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Consensus Voting (Byzantine 2/3 Threshold)
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Consensus Voting")
    print("=" * 72)
    print()

    print("Testing Byzantine fault-tolerant voting...")

    try:
        consensus = SimpleConsensusManager(consensus_threshold=0.67)

        # Create proposal
        proposal_id = consensus.create_proposal(
            target_node="target",
            source_node="source",
            quality=0.8
        )

        # High-weight approvals
        consensus.vote_on_proposal(proposal_id, "voter_1", VoteType.APPROVE,
                                  voter_reputation=0.9, voter_diversity=0.9)
        consensus.vote_on_proposal(proposal_id, "voter_2", VoteType.APPROVE,
                                  voter_reputation=0.8, voter_diversity=0.85)

        # Low-weight rejection (low diversity = Sybil suspect)
        consensus.vote_on_proposal(proposal_id, "voter_3", VoteType.REJECT,
                                  voter_reputation=0.9, voter_diversity=0.2)

        has_consensus, result = consensus.check_consensus(proposal_id)

        proposal = consensus.proposals[proposal_id]
        total_weight = sum(proposal.vote_weights.values())

        print(f"  Votes: 2 approve (high weight), 1 reject (low weight)")
        print(f"  Total weight: {total_weight:.3f}")
        print(f"  Consensus reached: {has_consensus}")
        print(f"  Result: {result.value if result else 'None'}")
        print()

        # Should approve (high-weight approvals exceed 2/3)
        test5_pass = has_consensus and result == VoteType.APPROVE

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["consensus_voting"] = test5_pass
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: Security-Enhanced SAGE Integration
    # ========================================================================
    print("=" * 72)
    print("TEST 6: Security-Enhanced SAGE Integration")
    print("=" * 72)
    print()

    print("Testing full security integration...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sage = SecurityEnhancedAdaptiveSAGE(
                node_id="sprout_edge",
                hardware_type="jetson_orin_nano",
                capability_level=3,
                storage_path=Path(tmpdir)
            )

            # Record peer verifications
            sage.record_peer_verification("peer_A", quality=0.8, use_consensus=False)
            sage.record_peer_verification("peer_B", quality=0.7, use_consensus=False)
            sage.record_peer_verification("peer_C", quality=0.9, use_consensus=False)

            # Get security-enhanced reputation
            rep = sage.get_security_enhanced_reputation("peer_A")

            # Select security-aware depth
            depth = sage.select_security_aware_depth()

            # Detect threats
            threats = sage.detect_security_threats()

            print(f"  Recorded 3 peer verifications")
            print(f"  Peer A reputation: {rep:.3f}")
            print(f"  Security-aware depth: {depth.name}")
            print(f"  Threat status: {threats['overall_status']}")
            print(f"  Circular clusters: {len(threats['circular_clusters'])}")
            print(f"  Diversity violations: {threats['diversity_violations']}")
            print()

            # Security system should flag low diversity
            test6_pass = threats['diversity_violations'] > 0

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test6_pass = False

    test_results["sage_integration"] = test6_pass
    print(f"{'PASS' if test6_pass else 'FAIL'}: TEST 6")
    print()
    all_tests_passed = all_tests_passed and test6_pass

    # ========================================================================
    # TEST 7: Edge Performance Profile
    # ========================================================================
    print("=" * 72)
    print("TEST 7: Edge Performance Profile")
    print("=" * 72)
    print()

    print("Profiling security operations on edge...")

    try:
        manager = SourceDiversityManager()

        # Profile diversity recording
        iterations = 1000
        start = time.time()
        for i in range(iterations):
            manager.record_reputation_event(f"node_{i % 10}", f"source_{i % 5}", 0.5)
        record_time = time.time() - start

        record_ops_per_sec = iterations / record_time

        # Profile diversity calculation
        start = time.time()
        for i in range(iterations):
            _ = manager.get_trust_multiplier(f"node_{i % 10}")
        calc_time = time.time() - start

        calc_ops_per_sec = iterations / calc_time

        # Profile consensus voting
        consensus = SimpleConsensusManager()
        start = time.time()
        for i in range(100):
            pid = consensus.create_proposal("target", f"source_{i}", 0.5)
            consensus.vote_on_proposal(pid, "voter_1", VoteType.APPROVE, 0.8, 0.8)
            consensus.vote_on_proposal(pid, "voter_2", VoteType.APPROVE, 0.7, 0.7)
            _, _ = consensus.check_consensus(pid)
        consensus_time = time.time() - start

        consensus_ops_per_sec = 100 / consensus_time

        print(f"  Diversity recording: {record_ops_per_sec:,.0f} ops/sec")
        print(f"  Trust calculation: {calc_ops_per_sec:,.0f} ops/sec")
        print(f"  Consensus cycles: {consensus_ops_per_sec:,.0f} ops/sec")
        print()

        # Should be fast enough for edge
        test7_pass = record_ops_per_sec > 1000 and calc_ops_per_sec > 1000

    except Exception as e:
        print(f"  ERROR: {e}")
        test7_pass = False

    test_results["edge_performance"] = test7_pass
    print(f"{'PASS' if test7_pass else 'FAIL'}: TEST 7")
    print()
    all_tests_passed = all_tests_passed and test7_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("SESSION 182 EDGE VALIDATION SUMMARY")
    print("=" * 72)
    print()

    print("Test Results:")
    for test_name, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
    print()

    test_count = sum(test_results.values())
    total_tests = len(test_results)
    print(f"Overall: {test_count}/{total_tests} tests passed")
    print()

    if all_tests_passed:
        print("+" + "-" * 70 + "+")
        print("|" + " " * 70 + "|")
        print("|" + "  SECURITY-ENHANCED REPUTATION VALIDATED ON EDGE!  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Shannon entropy diversity calculation fast on ARM64")
        print("  - Circular validation (Sybil cluster) detection working")
        print("  - Trust multiplier correctly penalizes low diversity")
        print("  - Byzantine consensus voting operational")
        print("  - Security integrated with adaptive consciousness")
        print()
        print("Security Stack Validated:")
        print("  Session 177: ATP-adaptive depth")
        print("  Session 178: Federated coordination")
        print("  Session 179: Reputation-aware depth")
        print("  Session 180: Persistent reputation")
        print("  Session 181: Meta-learning depth")
        print("  Session 182: Security-enhanced (NOW VALIDATED)")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "182_edge",
        "title": "Security-Enhanced Reputation - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": test_results,
        "edge_metrics": edge_metrics,
        "security_features": {
            "source_diversity_tracking": True,
            "circular_validation_detection": True,
            "trust_multipliers": True,
            "consensus_voting": True,
            "sybil_resistance": True
        }
    }

    results_path = Path(__file__).parent / "session182_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_security_enhanced_reputation()
    sys.exit(0 if success else 1)
