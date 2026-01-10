#!/usr/bin/env python3
"""
Session 178 Edge Validation: Federated SAGE Verification on Sprout

Testing Thor's federated adaptive consciousness on Jetson Orin Nano 8GB.

Thor's Implementation (Session 178):
- Extends Session 177's individual adaptive depth to federated multi-node
- Network ATP economics influence individual node depth selection
- Nodes with low ATP leverage high-ATP peers for verification
- Emergent collective behavior balancing quality and sustainability

Edge Validation Goals:
1. Verify federated depth coordination on constrained edge hardware
2. Test network state computation on ARM64
3. Validate peer verification delegation logic
4. Profile network analytics on edge
5. Test collective resource adaptation

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Session: Autonomous Edge Validation - Session 178
Date: 2026-01-10
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage"))


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
                    available_kb = int(line.split()[1])
                    metrics["memory_available_mb"] = available_kb / 1024
    except Exception:
        pass

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
            except Exception:
                continue
    except Exception:
        pass

    return metrics


def test_edge_federated_sage():
    """Test federated SAGE verification on edge hardware."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  SESSION 178 EDGE VALIDATION: FEDERATED SAGE VERIFICATION  ".center(70) + "|")
    print("|" + "           Jetson Orin Nano 8GB (Sprout)                     ".center(70) + "|")
    print("|" + " " * 70 + "|")
    print("+" + "=" * 70 + "+")
    print()

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
    test_results = {}

    # ========================================================================
    # TEST 1: Import Session 178 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 178 Components")
    print("=" * 72)
    print()

    print("Testing federated SAGE module imports...")

    try:
        from session178_federated_sage_verification import (
            NetworkDepthState,
            FederatedAdaptiveSAGE,
            FederatedSAGENetwork,
        )
        from session177_sage_adaptive_depth import (
            CognitiveDepth,
            DEPTH_CONFIGS,
        )

        print("  NetworkDepthState: OK")
        print("  FederatedAdaptiveSAGE: OK")
        print("  FederatedSAGENetwork: OK")
        print("  CognitiveDepth: OK")
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
        return {
            "all_tests_passed": False,
            "test_results": test_results,
            "error": "Import failed"
        }

    # ========================================================================
    # TEST 2: Network State Computation
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Network State Computation")
    print("=" * 72)
    print()

    print("Testing network depth state logic...")

    try:
        # Create network depth state
        node_depths = {
            "node_a": CognitiveDepth.THOROUGH,
            "node_b": CognitiveDepth.STANDARD,
            "node_c": CognitiveDepth.MINIMAL,
        }
        node_atp = {
            "node_a": 130.0,
            "node_b": 90.0,
            "node_c": 40.0,
        }

        state = NetworkDepthState(
            node_depths=node_depths,
            node_atp=node_atp,
            network_avg_depth=2.0,  # (4+2+0)/3
            network_avg_atp=86.67,  # (130+90+40)/3
            network_health=0.867,
            timestamp=time.time(),
        )

        # Test methods
        distribution = state.get_depth_distribution()
        quality = state.compute_collective_quality()

        print(f"  Node depths: {len(node_depths)} nodes")
        print(f"  Depth distribution: {distribution}")
        print(f"  Collective quality: {quality:.3f}")
        print(f"  Network health: {state.network_health:.3f}")
        print()

        # Validate
        test2_pass = (
            len(distribution) == 3 and
            0.0 < quality <= 1.0 and
            state.network_health > 0
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["network_state"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Federated Network Simulation
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Federated Network Simulation")
    print("=" * 72)
    print()

    print("Creating 3-node federated network on edge...")

    try:
        start = time.time()

        network = FederatedSAGENetwork()

        # Add nodes (simulating Legion, Thor, Sprout)
        network.add_node(
            node_id="legion_sim",
            hardware_type="tpm2",
            capability_level=5,
            initial_atp=130.0
        )
        network.add_node(
            node_id="thor_sim",
            hardware_type="trustzone",
            capability_level=5,
            initial_atp=100.0
        )
        network.add_node(
            node_id="sprout_sim",
            hardware_type="tpm2",
            capability_level=3,
            initial_atp=60.0
        )

        creation_time = time.time() - start

        print(f"  3 nodes created in {creation_time:.3f}s")
        print(f"  Nodes: {list(network.nodes.keys())}")
        print()

        # Compute initial network state
        state = network.update_all_nodes()

        print(f"  Initial network state:")
        print(f"    Avg depth: {state.network_avg_depth:.2f}")
        print(f"    Avg ATP: {state.network_avg_atp:.2f}")
        print(f"    Health: {state.network_health:.2f}")
        print(f"    Distribution: {state.get_depth_distribution()}")
        print()

        test3_pass = (
            len(network.nodes) == 3 and
            state.network_avg_atp > 0 and
            creation_time < 60.0  # Reasonable for edge
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test3_pass = False

    test_results["network_simulation"] = test3_pass
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Peer Verification Delegation
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Peer Verification Delegation")
    print("=" * 72)
    print()

    print("Testing peer verification logic...")

    try:
        # Get Sprout node (low ATP)
        sprout_node = network.nodes["sprout_sim"]

        # Drain Sprout's ATP
        sprout_node.attention_manager.total_atp = 35.0
        network.update_all_nodes()

        # Request verification from peer
        verification = sprout_node.request_peer_verification(
            thought_content="Test thought requiring peer verification",
            my_depth=CognitiveDepth.MINIMAL,
        )

        print(f"  Sprout ATP: {sprout_node.attention_manager.total_atp}")
        print(f"  Verification peer: {verification['peer_id']}")
        print(f"  Peer depth: {verification['peer_depth']}")
        print(f"  Verification quality: {verification['verification_quality']:.2f}")
        print(f"  Trust established: {verification['trust_established']}")
        print()

        # Validate verification logic
        test4_pass = (
            verification['verified'] and
            verification['peer_id'] is not None and
            verification['verification_quality'] > 0
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["peer_verification"] = test4_pass
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Collective Resource Adaptation
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Collective Resource Adaptation")
    print("=" * 72)
    print()

    print("Testing network-wide ATP stress and recovery...")

    try:
        # Stress all nodes
        for node_id, node in network.nodes.items():
            node.attention_manager.total_atp = 45.0

        state_stressed = network.update_all_nodes()

        print(f"  Stressed state:")
        print(f"    Avg ATP: {state_stressed.network_avg_atp:.2f}")
        print(f"    Health: {state_stressed.network_health:.2f}")
        print(f"    Distribution: {state_stressed.get_depth_distribution()}")
        print()

        # Recover Legion
        network.nodes["legion_sim"].attention_manager.total_atp = 140.0
        state_recovered = network.update_all_nodes()

        print(f"  Recovered state (Legion high ATP):")
        print(f"    Avg ATP: {state_recovered.network_avg_atp:.2f}")
        print(f"    Health: {state_recovered.network_health:.2f}")
        print(f"    Distribution: {state_recovered.get_depth_distribution()}")
        print()

        # Network should adapt - Legion should go deeper to help
        legion_depth = state_recovered.node_depths["legion_sim"]
        test5_pass = (
            state_recovered.network_health > state_stressed.network_health or
            legion_depth in [CognitiveDepth.DEEP, CognitiveDepth.THOROUGH]
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["collective_adaptation"] = test5_pass
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: Network Analytics
    # ========================================================================
    print("=" * 72)
    print("TEST 6: Network Analytics")
    print("=" * 72)
    print()

    print("Analyzing network history...")

    try:
        analytics = network.get_network_analytics()

        print(f"  Total states: {analytics['total_states']}")
        print(f"  Avg depth (mean): {analytics['avg_depth_mean']:.2f}")
        print(f"  Avg ATP (mean): {analytics['avg_atp_mean']:.2f}")
        print(f"  Avg health (mean): {analytics['avg_health_mean']:.2f}")
        print(f"  Collective quality (mean): {analytics['collective_quality_mean']:.2f}")
        print()
        print(f"  Final state:")
        print(f"    Depth: {analytics['final_state']['depth']:.2f}")
        print(f"    ATP: {analytics['final_state']['atp']:.2f}")
        print(f"    Health: {analytics['final_state']['health']:.2f}")
        print()

        test6_pass = (
            analytics['total_states'] >= 3 and
            'final_state' in analytics
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test6_pass = False

    test_results["network_analytics"] = test6_pass
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

    print("Profiling federated operations on edge...")

    try:
        iterations = 100
        start = time.time()

        for _ in range(iterations):
            network.update_all_nodes()

        elapsed = time.time() - start
        ops_per_sec = iterations / elapsed

        print(f"  Network updates: {iterations}")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Updates/sec: {ops_per_sec:.1f}")
        print()

        # Edge should handle at least 10 updates/sec
        test7_pass = ops_per_sec > 10

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
    print("SESSION 178 EDGE VALIDATION SUMMARY")
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
        print("|" + "  FEDERATED SAGE VERIFICATION VALIDATED ON EDGE!  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Multi-node federation operational on 8GB edge hardware")
        print("  - Network depth coordination working on ARM64")
        print("  - Peer verification delegation functional")
        print("  - Collective resource adaptation observed")
        print("  - Network analytics tracking properly")
        print("  - Emergent collective behavior validated")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "178_edge",
        "title": "Federated SAGE Verification - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": test_results,
        "edge_metrics": edge_metrics,
        "convergent_research": {
            "thor_session_178": "Federated SAGE Verification",
            "thor_session_177": "SAGE Adaptive Depth",
            "thor_session_175": "Network Economic Federation",
            "legion_session_158": "Dynamic verification depth",
            "edge_validation": "Multi-node adaptive consciousness validated"
        }
    }

    results_path = Path(__file__).parent / "session178_edge_federated_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_federated_sage()
    sys.exit(0 if success else 1)
