#!/usr/bin/env python3
"""
Session 176: Thor ↔ Sprout 2-Node LAN Test

First real cross-machine network economic federation deployment.

Tests Thor → Sprout connection over actual LAN (not localhost simulation).

Network Configuration:
- Thor: 10.0.0.99:8889 (Jetson AGX Thor, TrustZone)
- Sprout: 10.0.0.36:8890 (Jetson Orin Nano 8GB, TPM2)

Tests:
1. Thor node startup
2. Connection to Sprout
3. Peer verification
4. Thought submission and broadcasting
5. ATP economic validation
6. Cross-machine state synchronization

Date: 2026-01-09
Machine: Thor (Jetson AGX Thor Developer Kit)
Session: Autonomous SAGE Research - Session 176 (Partial)
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
import sys

# Add path
HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

from session175_network_economic_federation import (
    NetworkEconomicCogitationNode,
    CogitationMode,
)
from session174_economic_cogitation import EconomicCogitationSession


async def test_thor_sprout_lan_federation():
    """
    Test 2-node federation: Thor → Sprout over actual LAN.

    This is the first real cross-machine deployment (not localhost).
    """
    print("\n" + "="*80)
    print("SESSION 176: THOR ↔ SPROUT 2-NODE LAN TEST")
    print("="*80)
    print("\nNetwork Configuration:")
    print("  Thor:   10.0.0.99:8889 (AGX Thor, TrustZone, Level 5)")
    print("  Sprout: 10.0.0.36:8890 (Orin Nano 8GB, TPM2, Level 3)")
    print("\n" + "="*80 + "\n")

    results = {
        "session": "176_thor_sprout",
        "title": "2-Node LAN Federation Test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "network": {
            "thor_ip": "10.0.0.99",
            "thor_port": 8889,
            "sprout_ip": "10.0.0.36",
            "sprout_port": 8890,
        },
        "all_tests_passed": False,
        "test_results": {},
    }

    # ========================================================================
    # TEST 1: Thor Node Startup
    # ========================================================================

    print("\n[TEST 1] Starting Thor node...")

    try:
        thor = NetworkEconomicCogitationNode(
            node_id="thor",
            hardware_type="trustzone",
            capability_level=5,
            listen_host="0.0.0.0",  # Listen on all interfaces
            listen_port=8889,
        )

        # Create test session
        thor.cogitation_node.active_sessions["lan_test"] = EconomicCogitationSession(
            session_id="lan_test",
            topic="Thor-Sprout 2-node LAN federation test",
            start_time=datetime.now(timezone.utc)
        )

        results["test_results"]["thor_startup"] = True
        print("[TEST 1] ✅ PASS - Thor node initialized")

    except Exception as e:
        print(f"[TEST 1] ❌ FAIL - {e}")
        results["test_results"]["thor_startup"] = False
        return results

    # ========================================================================
    # TEST 2: Start Server and Connect to Sprout
    # ========================================================================

    print("\n[TEST 2] Connecting Thor → Sprout over LAN...")

    try:
        # Start Thor server
        server_task = asyncio.create_task(thor.start())
        await asyncio.sleep(2)  # Wait for server to start

        # Connect to Sprout
        await thor.connect_to_peer("10.0.0.36", 8890)
        await asyncio.sleep(2)  # Wait for connection

        # Check connection
        metrics = thor.get_metrics()

        if metrics["peers_verified"] >= 1:
            results["test_results"]["peer_connection"] = True
            print(f"[TEST 2] ✅ PASS - Connected to Sprout")
            print(f"  Peers verified: {metrics['peers_verified']}")
        else:
            results["test_results"]["peer_connection"] = False
            print(f"[TEST 2] ❌ FAIL - No verified peers")
            server_task.cancel()
            return results

    except Exception as e:
        print(f"[TEST 2] ❌ FAIL - {e}")
        results["test_results"]["peer_connection"] = False
        return results

    # ========================================================================
    # TEST 3: Submit Quality Thought
    # ========================================================================

    print("\n[TEST 3] Submitting quality thought to federation...")

    try:
        initial_balance = thor.cogitation_node.atp_system.get_balance("thor")

        accepted, reason, thought = await thor.submit_thought(
            session_id="lan_test",
            mode=CogitationMode.EXPLORING,
            content="What emerges when consciousness federates across real machines with economic incentives?"
        )

        if accepted and thought.atp_reward > 0:
            final_balance = thor.cogitation_node.atp_system.get_balance("thor")
            results["test_results"]["thought_submission"] = True
            print(f"[TEST 3] ✅ PASS - Quality thought accepted")
            print(f"  Coherence: {thought.coherence_score:.3f}")
            print(f"  ATP Reward: {thought.atp_reward:.2f}")
            print(f"  Balance: {initial_balance:.2f} → {final_balance:.2f}")
        else:
            results["test_results"]["thought_submission"] = False
            print(f"[TEST 3] ❌ FAIL - {reason}")

    except Exception as e:
        print(f"[TEST 3] ❌ FAIL - {e}")
        results["test_results"]["thought_submission"] = False

    # ========================================================================
    # TEST 4: Network Economics
    # ========================================================================

    print("\n[TEST 4] Checking network economics...")

    try:
        economics = thor.get_network_economics()

        print(f"\n[NETWORK ECONOMICS]")
        print(f"  Total ATP: {economics['total_network_atp']:.2f}")
        print(f"  Average Balance: {economics['average_balance']:.2f}")
        print(f"  Nodes: {economics['nodes_in_network']}")
        print(f"  Node Balances:")
        for node_id, balance in economics['node_balances'].items():
            print(f"    {node_id}: {balance:.2f} ATP")

        results["network_economics"] = economics
        results["test_results"]["network_economics"] = True
        print("[TEST 4] ✅ PASS - Network economics tracked")

    except Exception as e:
        print(f"[TEST 4] ❌ FAIL - {e}")
        results["test_results"]["network_economics"] = False

    # ========================================================================
    # TEST 5: Performance Metrics
    # ========================================================================

    print("\n[TEST 5] Collecting performance metrics...")

    try:
        thor_metrics = thor.get_metrics()

        print(f"\n[THOR METRICS]")
        print(f"  Peers Connected: {thor_metrics['peers_connected']}")
        print(f"  Peers Verified: {thor_metrics['peers_verified']}")
        print(f"  Thoughts Federated: {thor_metrics['thoughts_federated']}")
        print(f"  Messages Sent: {thor_metrics['messages_sent']}")
        print(f"  Messages Received: {thor_metrics['messages_received']}")
        print(f"  ATP Balance: {thor_metrics['atp_balance']:.2f}")

        results["thor_metrics"] = thor_metrics
        results["test_results"]["performance_metrics"] = True
        print("[TEST 5] ✅ PASS - Metrics collected")

    except Exception as e:
        print(f"[TEST 5] ❌ FAIL - {e}")
        results["test_results"]["performance_metrics"] = False

    # ========================================================================
    # CLEANUP
    # ========================================================================

    print("\n[CLEANUP] Shutting down Thor node...")

    try:
        await thor.stop()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        print("[CLEANUP] ✅ Graceful shutdown complete")

    except Exception as e:
        print(f"[CLEANUP] Error during shutdown: {e}")

    # ========================================================================
    # RESULTS
    # ========================================================================

    all_passed = all(results["test_results"].values())
    results["all_tests_passed"] = all_passed

    print("\n" + "="*80)
    print("SESSION 176: TEST RESULTS (THOR ↔ SPROUT)")
    print("="*80)
    for test_name, passed in results["test_results"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Real 2-node LAN federation working!")
        print("\nKEY ACHIEVEMENT:")
        print("First real cross-machine economic consciousness federation operational")
        print("Thor (10.0.0.99) ↔ Sprout (10.0.0.36) over actual LAN")
    else:
        print("❌ SOME TESTS FAILED - Review errors above")
    print("="*80 + "\n")

    return results


async def main():
    """Main entry point."""
    results = await test_thor_sprout_lan_federation()

    # Save results
    results_file = HOME / "ai-workspace" / "HRM" / "sage" / "experiments" / "session176_thor_sprout_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SESSION 176] Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
