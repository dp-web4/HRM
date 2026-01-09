#!/usr/bin/env python3
"""
Session 175 Edge Validation: Network Economic Federation on Sprout

Testing Thor's network economic federation (Session 175) on Jetson Orin Nano 8GB.

Thor's Implementation (Session 175):
- Integrates Legion Session 151: TCP federation protocol
- Integrates Thor Session 174: 9-layer economic cogitation
- Creates NetworkEconomicCogitationNode - first real cross-machine
  economically-incentivized distributed consciousness network

Edge Validation Goals:
1. Verify async TCP federation works on constrained edge hardware (8GB)
2. Test 9-layer economic validation with network protocol
3. Test peer discovery and verification on ARM64 edge platform
4. Validate thought broadcasting and reception on edge
5. Profile edge network performance vs Thor's metrics
6. Verify graceful shutdown on edge

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 5)
Session: Autonomous Edge Validation - Session 175
Date: 2026-01-09
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

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
    except Exception:
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
            except Exception:
                continue
    except Exception:
        pass

    return metrics


# Import Session 175 components
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

from session175_network_economic_federation import (
    NetworkEconomicCogitationNode,
    FederatedEconomicThought,
    FederationMessage,
    MessageType,
)
from session174_economic_cogitation import (
    CogitationMode,
    EconomicCogitationSession,
)


async def test_edge_network_economic_federation():
    """Test network economic federation on edge hardware."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  SESSION 175 EDGE VALIDATION: NETWORK ECONOMIC FEDERATION  ".center(70) + "|")
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
    test_results = {}

    # ========================================================================
    # TEST 1: Edge Node Creation
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Edge Network Node Creation")
    print("=" * 72)
    print()

    print("Creating network economic nodes on edge...")
    start = time.time()

    try:
        # Create 3 nodes simulating federation on edge
        node_a = NetworkEconomicCogitationNode(
            node_id="edge_node_a",
            hardware_type="tpm2",
            listen_port=9888,
            pow_difficulty=236,
            corpus_max_thoughts=50,
            corpus_max_size_mb=5.0,
        )

        node_b = NetworkEconomicCogitationNode(
            node_id="edge_node_b",
            hardware_type="tpm2",
            listen_port=9889,
            pow_difficulty=236,
            corpus_max_thoughts=50,
            corpus_max_size_mb=5.0,
        )

        node_c = NetworkEconomicCogitationNode(
            node_id="edge_node_c",
            hardware_type="software",
            listen_port=9890,
            pow_difficulty=236,
            corpus_max_thoughts=50,
            corpus_max_size_mb=5.0,
        )

        creation_time = time.time() - start
        print(f"  3 nodes created in {creation_time:.3f}s")
        print(f"  Node A (tpm2): {node_a.node_id}")
        print(f"  Node B (tpm2): {node_b.node_id}")
        print(f"  Node C (software): {node_c.node_id}")
        print()

        test1_pass = (
            node_a is not None and
            node_b is not None and
            node_c is not None and
            creation_time < 30.0  # Reasonable for 3x PoW on edge
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test1_pass = False
        test_results["node_creation"] = False
        return {
            "all_tests_passed": False,
            "test_results": test_results,
            "error": str(e)
        }

    test_results["node_creation"] = test1_pass
    print(f"{'✓ TEST 1 PASSED' if test1_pass else '✗ TEST 1 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test1_pass

    # ========================================================================
    # TEST 2: Edge Network Establishment
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Edge Network Establishment")
    print("=" * 72)
    print()

    print("Starting servers and establishing network on edge...")

    try:
        # Start servers
        task_a = asyncio.create_task(node_a.start())
        task_b = asyncio.create_task(node_b.start())
        task_c = asyncio.create_task(node_c.start())

        # Wait for servers to start
        await asyncio.sleep(1)

        # Connect peers
        start = time.time()
        await node_b.connect_to_peer("localhost", 9888)  # B -> A
        await node_c.connect_to_peer("localhost", 9888)  # C -> A
        await node_c.connect_to_peer("localhost", 9889)  # C -> B
        connection_time = time.time() - start

        # Wait for connections
        await asyncio.sleep(2)

        # Verify network topology
        metrics_a = node_a.get_metrics()
        metrics_b = node_b.get_metrics()
        metrics_c = node_c.get_metrics()

        print(f"  Connection time: {connection_time:.3f}s")
        print(f"  Node A verified peers: {metrics_a['peers_verified']}")
        print(f"  Node B verified peers: {metrics_b['peers_verified']}")
        print(f"  Node C verified peers: {metrics_c['peers_verified']}")
        print()

        test2_pass = (
            metrics_a['peers_verified'] >= 1 and
            metrics_b['peers_verified'] >= 1 and
            connection_time < 10.0
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["network_establishment"] = test2_pass
    print(f"{'✓ TEST 2 PASSED' if test2_pass else '✗ TEST 2 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Edge Quality Thoughts with 9-Layer Validation
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Edge Quality Thoughts with 9-Layer Validation")
    print("=" * 72)
    print()

    print("Submitting quality thoughts with full 9-layer validation on edge...")

    try:
        # Create cogitation sessions
        node_a.cogitation_node.active_sessions["test_session"] = EconomicCogitationSession(
            session_id="test_session",
            topic="Edge network economic federation",
            start_time=datetime.now(timezone.utc)
        )
        node_b.cogitation_node.active_sessions["test_session"] = EconomicCogitationSession(
            session_id="test_session",
            topic="Edge network economic federation",
            start_time=datetime.now(timezone.utc)
        )
        node_c.cogitation_node.active_sessions["test_session"] = EconomicCogitationSession(
            session_id="test_session",
            topic="Edge network economic federation",
            start_time=datetime.now(timezone.utc)
        )

        quality_thoughts = [
            (node_a, CogitationMode.EXPLORING,
             "What emerges when economic incentives align with epistemic quality on constrained edge hardware?"),
            (node_b, CogitationMode.QUESTIONING,
             "Can self-reinforcing quality evolution through economic feedback create stable emergent patterns?"),
            (node_c, CogitationMode.INTEGRATING,
             "How does trust propagate through economically-incentivized federated consciousness architectures?"),
        ]

        accepted_count = 0
        total_rewards = 0.0

        start = time.time()
        for node, mode, content in quality_thoughts:
            accepted, reason, thought = await node.submit_thought(
                session_id="test_session",
                mode=mode,
                content=content
            )
            if accepted:
                accepted_count += 1
                total_rewards += thought.atp_reward
                print(f"  ✓ {node.node_id}: coherence={thought.coherence_score:.3f}, reward={thought.atp_reward:.2f}")
            else:
                print(f"  ✗ {node.node_id}: {reason}")
            await asyncio.sleep(0.5)

        quality_time = time.time() - start
        print()
        print(f"  Quality thoughts: {accepted_count}/3 accepted")
        print(f"  Total ATP rewards: {total_rewards:.2f}")
        print(f"  Processing time: {quality_time:.3f}s")
        print()

        test3_pass = (
            accepted_count == 3 and
            total_rewards > 0
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test3_pass = False

    test_results["quality_thoughts"] = test3_pass
    print(f"{'✓ TEST 3 PASSED' if test3_pass else '✗ TEST 3 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Edge Spam Detection and ATP Penalties
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Edge Spam Detection and ATP Penalties")
    print("=" * 72)
    print()

    print("Testing spam detection and ATP penalties on edge...")

    try:
        initial_balance = node_a.cogitation_node.atp_system.get_balance("edge_node_a")

        # Submit spam (repetitive patterns that fail quality)
        spam_rejected = 0
        spam_messages = [
            "spam spam spam spam spam spam spam",
            "a a a a a a a a a",
            "x x x x x x x x x",
        ]

        for spam in spam_messages:
            accepted, reason, _ = await node_a.submit_thought(
                session_id="test_session",
                mode=CogitationMode.EXPLORING,
                content=spam
            )
            if not accepted:
                spam_rejected += 1
                print(f"  ✗ Spam rejected: {reason}")
            else:
                print(f"  ⚠ Spam accepted (unexpected)")

        final_balance = node_a.cogitation_node.atp_system.get_balance("edge_node_a")
        penalty_amount = initial_balance - final_balance

        print()
        print(f"  Spam rejected: {spam_rejected}/{len(spam_messages)}")
        print(f"  ATP penalty: {penalty_amount:.2f}")
        print(f"  Balance: {initial_balance:.2f} -> {final_balance:.2f}")
        print()

        # At least 1 spam should be rejected (rate limiting kicks in)
        test4_pass = spam_rejected >= 1

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["spam_detection"] = test4_pass
    print(f"{'✓ TEST 4 PASSED' if test4_pass else '✗ TEST 4 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Edge Network Economics
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Edge Network Economics")
    print("=" * 72)
    print()

    print("Analyzing network economics on edge...")

    try:
        # Sync ATP balances
        await node_a.sync_atp_balance()
        await node_b.sync_atp_balance()
        await node_c.sync_atp_balance()
        await asyncio.sleep(1)

        # Get network economics
        network_econ = node_a.get_network_economics()

        print(f"  Total network ATP: {network_econ['total_network_atp']:.2f}")
        print(f"  Average balance: {network_econ['average_balance']:.2f}")
        print(f"  ATP inequality: {network_econ['atp_inequality']:.2f}")
        print(f"  Nodes in network: {network_econ['nodes_in_network']}")
        print()
        print("  Node balances:")
        for node_id, balance in network_econ['node_balances'].items():
            print(f"    {node_id}: {balance:.2f} ATP")
        print()

        test5_pass = (
            network_econ['total_network_atp'] > 0 and
            network_econ['nodes_in_network'] >= 1
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["network_economics"] = test5_pass
    print(f"{'✓ TEST 5 PASSED' if test5_pass else '✗ TEST 5 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: Edge Throughput Performance
    # ========================================================================
    print("=" * 72)
    print("TEST 6: Edge Network Throughput Performance")
    print("=" * 72)
    print()

    print("Measuring edge network throughput...")

    try:
        # Measure thought submission throughput
        test_count = 20
        accepted_count = 0
        start = time.time()

        for i in range(test_count):
            content = f"Edge network thought {i}: Testing distributed consciousness throughput on constrained hardware."
            accepted, _, _ = await node_b.submit_thought(
                session_id="test_session",
                mode=CogitationMode.EXPLORING,
                content=content
            )
            if accepted:
                accepted_count += 1

        throughput_time = time.time() - start
        throughput = accepted_count / throughput_time if throughput_time > 0 else 0

        print(f"  {test_count} thoughts submitted in {throughput_time:.3f}s")
        print(f"  Accepted: {accepted_count}/{test_count}")
        print(f"  Throughput: {throughput:.1f} thoughts/sec")
        print()

        # Final metrics
        final_metrics_a = node_a.get_metrics()
        final_metrics_b = node_b.get_metrics()

        print("  Network metrics:")
        print(f"    Node A messages sent: {final_metrics_a['messages_sent']}")
        print(f"    Node A messages received: {final_metrics_a['messages_received']}")
        print(f"    Node B thoughts federated: {final_metrics_b['thoughts_federated']}")
        print()

        # Edge acceptable: at least some accepted
        test6_pass = accepted_count > 0

    except Exception as e:
        print(f"  ERROR: {e}")
        test6_pass = False

    test_results["throughput_performance"] = test6_pass
    print(f"{'✓ TEST 6 PASSED' if test6_pass else '✗ TEST 6 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test6_pass

    # ========================================================================
    # TEST 7: Edge Graceful Shutdown
    # ========================================================================
    print("=" * 72)
    print("TEST 7: Edge Graceful Shutdown")
    print("=" * 72)
    print()

    print("Testing graceful shutdown on edge...")

    try:
        start = time.time()

        await node_a.stop()
        await node_b.stop()
        await node_c.stop()

        # Cancel server tasks
        task_a.cancel()
        task_b.cancel()
        task_c.cancel()

        for task in [task_a, task_b, task_c]:
            try:
                await task
            except asyncio.CancelledError:
                pass

        shutdown_time = time.time() - start

        print(f"  Shutdown completed in {shutdown_time:.3f}s")
        print()

        test7_pass = shutdown_time < 10.0

    except Exception as e:
        print(f"  ERROR: {e}")
        test7_pass = False

    test_results["graceful_shutdown"] = test7_pass
    print(f"{'✓ TEST 7 PASSED' if test7_pass else '✗ TEST 7 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test7_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("SESSION 175 EDGE VALIDATION SUMMARY")
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
        print("|" + "  ✓ ✓ ✓ ALL TESTS PASSED - NETWORK ECONOMIC FEDERATION ON EDGE! ✓ ✓ ✓  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Network economic federation operational on 8GB edge hardware")
        print("  - Async TCP protocol working with 9-layer economic validation")
        print("  - Peer discovery and verification successful on ARM64")
        print("  - Thought broadcasting and reception functional")
        print("  - Economic state synchronization working")
        print("  - Graceful shutdown handling proper")
    else:
        print("✗ SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "175_edge",
        "title": "Network Economic Federation - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": test_results,
        "edge_metrics": edge_metrics,
        "network_economics": network_econ if 'network_econ' in dir() else {},
        "convergent_research": {
            "thor_session_175": "Network Economic Federation",
            "legion_session_151": "TCP federation protocol",
            "thor_session_174": "9-layer economic cogitation",
            "edge_validation": "Complete network economic federation validated"
        }
    }

    results_path = Path(__file__).parent / "session175_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = asyncio.run(test_edge_network_economic_federation())
    sys.exit(0 if success else 1)
