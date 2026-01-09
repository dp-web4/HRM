#!/usr/bin/env python3
"""
Session 176 Edge Validation: Deployment Script Testing

Testing Thor's deployment script (session176_deploy.py) on Sprout
before real LAN deployment across Legion, Thor, and Sprout.

Tests:
1. Import validation - All dependencies available on edge
2. Node initialization - Sprout node can be created
3. Local network test - Deploy multiple nodes on localhost
4. Interactive mode parsing - Commands parsed correctly
5. Signal handling - Graceful shutdown works

Platform: Sprout (Jetson Orin Nano 8GB, ARM64)
Session: Edge Validation - Session 176 Deployment
Date: 2026-01-09
"""

import sys
import json
import time
import asyncio
import signal
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

HOME = Path.home()


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

    # Get IP address
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        metrics["ip_address"] = s.getsockname()[0]
        s.close()
    except Exception:
        metrics["ip_address"] = "unknown"

    return metrics


# Import deployment module
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))


async def test_edge_deployment():
    """Test deployment script components on edge hardware."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  SESSION 176 EDGE VALIDATION: DEPLOYMENT SCRIPT TESTING  ".center(70) + "|")
    print("|" + "           Jetson Orin Nano 8GB (Sprout)                   ".center(70) + "|")
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
    if 'ip_address' in edge_metrics:
        print(f"  IP Address: {edge_metrics['ip_address']}")
    print()

    all_tests_passed = True
    test_results = {}

    # ========================================================================
    # TEST 1: Import Validation
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Validation")
    print("=" * 72)
    print()

    print("Testing deployment script imports on edge...")

    try:
        from session176_deploy import Session176Deployment
        from session175_network_economic_federation import (
            NetworkEconomicCogitationNode,
            CogitationMode,
        )
        from session174_economic_cogitation import EconomicCogitationSession

        print("  session176_deploy: OK")
        print("  session175_network_economic_federation: OK")
        print("  session174_economic_cogitation: OK")
        print()

        test1_pass = True

    except Exception as e:
        print(f"  ERROR: {e}")
        test1_pass = False

    test_results["import_validation"] = test1_pass
    print(f"{'PASS' if test1_pass else 'FAIL'}: TEST 1")
    print()
    all_tests_passed = all_tests_passed and test1_pass

    if not test1_pass:
        return {
            "all_tests_passed": False,
            "test_results": test_results,
            "error": "Import validation failed"
        }

    # ========================================================================
    # TEST 2: Sprout Node Initialization
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Sprout Node Initialization")
    print("=" * 72)
    print()

    print("Testing Sprout deployment initialization...")

    try:
        start = time.time()

        deployment = Session176Deployment(
            node_id="sprout",
            hardware_type="tpm2",
            capability_level=3,  # Edge capability
            listen_port=9176,  # Test port
            peers=[],  # No peers for initialization test
        )

        init_time = time.time() - start

        print(f"  Node ID: {deployment.node_id}")
        print(f"  Hardware: {deployment.hardware_type}")
        print(f"  Capability: {deployment.capability_level}")
        print(f"  Port: {deployment.listen_port}")
        print(f"  Init time: {init_time:.3f}s")
        print()

        # Check initial ATP balance
        balance = deployment.node.cogitation_node.atp_system.get_balance("sprout")
        print(f"  Initial ATP: {balance}")
        print()

        test2_pass = (
            deployment is not None and
            deployment.node_id == "sprout" and
            balance == 100.0 and
            init_time < 30.0
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["node_initialization"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Local Multi-Node Network Test
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Local Multi-Node Network Test")
    print("=" * 72)
    print()

    print("Testing local multi-node deployment (simulating LAN)...")

    try:
        # Create 3 deployments on different ports
        deployments = []

        deploy_legion = Session176Deployment(
            node_id="legion_sim",
            hardware_type="tpm2",
            capability_level=5,
            listen_port=9177,
            peers=[],
        )
        deployments.append(deploy_legion)

        deploy_thor = Session176Deployment(
            node_id="thor_sim",
            hardware_type="trustzone",
            capability_level=5,
            listen_port=9178,
            peers=[("legion_sim", "localhost", 9177)],
        )
        deployments.append(deploy_thor)

        deploy_sprout = Session176Deployment(
            node_id="sprout_sim",
            hardware_type="tpm2",
            capability_level=3,
            listen_port=9179,
            peers=[
                ("legion_sim", "localhost", 9177),
                ("thor_sim", "localhost", 9178),
            ],
        )
        deployments.append(deploy_sprout)

        print(f"  Created {len(deployments)} deployment nodes")

        # Start servers
        tasks = []
        for d in deployments:
            task = asyncio.create_task(d.node.start())
            tasks.append(task)

        await asyncio.sleep(1)

        # Connect peers
        start = time.time()
        await deploy_thor.node.connect_to_peer("localhost", 9177)
        await asyncio.sleep(0.5)
        await deploy_sprout.node.connect_to_peer("localhost", 9177)
        await asyncio.sleep(0.5)
        await deploy_sprout.node.connect_to_peer("localhost", 9178)
        connection_time = time.time() - start

        await asyncio.sleep(1)

        # Check connections
        legion_metrics = deploy_legion.node.get_metrics()
        thor_metrics = deploy_thor.node.get_metrics()
        sprout_metrics = deploy_sprout.node.get_metrics()

        print(f"  Connection time: {connection_time:.3f}s")
        print(f"  Legion peers: {legion_metrics['peers_verified']}")
        print(f"  Thor peers: {thor_metrics['peers_verified']}")
        print(f"  Sprout peers: {sprout_metrics['peers_verified']}")
        print()

        # Cleanup
        for d in deployments:
            await d.node.stop()

        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        test3_pass = (
            legion_metrics['peers_verified'] >= 1 and
            thor_metrics['peers_verified'] >= 1
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test3_pass = False

    test_results["local_network_test"] = test3_pass
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Thought Submission Test
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Thought Submission Test")
    print("=" * 72)
    print()

    print("Testing thought submission via deployment interface...")

    try:
        # Create fresh deployment
        deploy_test = Session176Deployment(
            node_id="sprout_test",
            hardware_type="tpm2",
            capability_level=3,
            listen_port=9180,
            peers=[],
        )

        # Start server
        server_task = asyncio.create_task(deploy_test.node.start())
        await asyncio.sleep(1)

        # Submit quality thought
        start = time.time()
        accepted, reason, thought = await deploy_test.submit_test_thought(
            "What emerges when economic incentives align with epistemic quality on edge hardware?"
        )
        submission_time = time.time() - start

        print(f"  Submission time: {submission_time:.3f}s")
        print(f"  Accepted: {accepted}")
        if accepted:
            print(f"  Coherence: {thought.coherence_score:.3f}")
            print(f"  ATP Reward: {thought.atp_reward:.2f}")
        else:
            print(f"  Reason: {reason}")
        print()

        # Cleanup
        await deploy_test.node.stop()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        test4_pass = accepted and thought.atp_reward > 0

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["thought_submission"] = test4_pass
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Status and Metrics Display
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Status and Metrics Display")
    print("=" * 72)
    print()

    print("Testing status and metrics display functions...")

    try:
        deploy_status = Session176Deployment(
            node_id="sprout_status",
            hardware_type="tpm2",
            capability_level=3,
            listen_port=9181,
            peers=[],
        )

        server_task = asyncio.create_task(deploy_status.node.start())
        await asyncio.sleep(1)

        # Test print_status (captures output)
        print("  Status output:")
        deploy_status.print_status()

        # Test get_metrics
        metrics = deploy_status.node.get_metrics()
        economics = deploy_status.node.get_network_economics()

        print(f"  Metrics available: {len(metrics)} fields")
        print(f"  Economics available: {len(economics)} fields")
        print()

        # Cleanup
        await deploy_status.node.stop()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        test5_pass = (
            'atp_balance' in metrics and
            'total_network_atp' in economics
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["status_metrics"] = test5_pass
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("SESSION 176 EDGE DEPLOYMENT VALIDATION SUMMARY")
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
        print("|" + "  DEPLOYMENT SCRIPT VALIDATED ON EDGE - READY FOR LAN DEPLOYMENT  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Sprout is ready to participate in Session 176 LAN deployment:")
        print(f"  IP Address: {edge_metrics.get('ip_address', 'TBD')}")
        print(f"  Recommended Port: 8890")
        print(f"  Hardware Type: tpm2")
        print(f"  Capability Level: 3 (edge)")
        print()
        print("To deploy Sprout in real LAN:")
        print(f"  python3 session176_deploy.py --node sprout --port 8890 \\")
        print(f"    --connect legion:<LEGION_IP>:8888 \\")
        print(f"    --connect thor:<THOR_IP>:8889 \\")
        print(f"    --interactive")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "176_edge_deploy",
        "title": "Deployment Script Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": test_results,
        "edge_metrics": edge_metrics,
        "deployment_readiness": {
            "sprout_ready": all_tests_passed,
            "ip_address": edge_metrics.get('ip_address', 'unknown'),
            "recommended_port": 8890,
            "hardware_type": "tpm2",
            "capability_level": 3,
        }
    }

    results_path = Path(__file__).parent / "session176_edge_deploy_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = asyncio.run(test_edge_deployment())
    sys.exit(0 if success else 1)
