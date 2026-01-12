#!/usr/bin/env python3
"""
Session 183 Edge Validation: Network Protocol SAGE

Testing Thor's network protocol integration on Sprout edge hardware.

Thor's Session 183 Implementation:
- ProtocolMessage: Network-ready message format
- Message types: REPUTATION_PROPOSAL, CONSENSUS_VOTE, IDENTITY_ANNOUNCEMENT
- JSONL serialization for streaming
- Peer-to-peer communication simulation
- Network status reporting

Edge Validation Goals:
1. Verify protocol components import correctly
2. Test message serialization/deserialization
3. Validate identity announcement
4. Test reputation proposal broadcast
5. Validate peer-to-peer communication
6. Profile protocol operations on edge

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Date: 2026-01-11
"""

import sys
import json
import time
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


def test_edge_network_protocol():
    """Test Session 183 network protocol on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  SESSION 183 EDGE VALIDATION: NETWORK PROTOCOL SAGE  ".center(70) + "|")
    print("|" + "            Jetson Orin Nano 8GB (Sprout)              ".center(70) + "|")
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
    # TEST 1: Import Session 183 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 183 Components")
    print("=" * 72)
    print()

    try:
        from session183_network_protocol_sage import (
            MessageType,
            ProtocolMessage,
            ReputationProposalPayload,
            ConsensusVotePayload,
            IdentityAnnouncementPayload,
            NetworkReadySAGE,
        )
        from session182_security_enhanced_reputation import VoteType

        print("  MessageType: Protocol message types")
        print("  ProtocolMessage: Network-ready message")
        print("  ReputationProposalPayload: Proposal data")
        print("  ConsensusVotePayload: Vote data")
        print("  IdentityAnnouncementPayload: Peer discovery")
        print("  NetworkReadySAGE: Full integration")
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
    # TEST 2: Message Serialization (JSONL)
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Message Serialization (JSONL)")
    print("=" * 72)
    print()

    print("Testing JSONL message format...")

    try:
        # Create a test message
        msg = ProtocolMessage(
            message_type=MessageType.IDENTITY_ANNOUNCEMENT.value,
            source_node_id="sprout_test",
            timestamp=time.time(),
            payload={"test": "data", "value": 42},
            attestation="test_attestation_123"
        )

        # Serialize to JSONL
        jsonl = msg.to_jsonl()

        # Deserialize
        msg2 = ProtocolMessage.from_jsonl(jsonl)

        print(f"  Original message ID: {msg.message_id}")
        print(f"  Serialized length: {len(jsonl)} bytes")
        print(f"  Deserialized message ID: {msg2.message_id}")
        print()

        # Validate round-trip
        test2_pass = (
            msg.message_id == msg2.message_id and
            msg.source_node_id == msg2.source_node_id and
            msg.payload == msg2.payload
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["jsonl_serialization"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Identity Announcement
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Identity Announcement (Peer Discovery)")
    print("=" * 72)
    print()

    print("Testing identity announcement on edge...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sage = NetworkReadySAGE(
                node_id="sprout_edge",
                hardware_type="jetson_orin_nano",
                capability_level=3,
                storage_path=Path(tmpdir),
                network_address="10.0.0.36"
            )

            # Announce identity
            identity_msg = sage.announce_identity()

            print(f"  Node: {sage.node_id}")
            print(f"  Network address: {sage.network_address}")
            print(f"  Message type: {identity_msg.message_type}")
            print(f"  Features: {identity_msg.payload.get('features', [])[:3]}...")
            print(f"  Attestation length: {len(identity_msg.attestation)}")
            print()

            test3_pass = (
                identity_msg.message_type == MessageType.IDENTITY_ANNOUNCEMENT.value and
                identity_msg.payload['node_id'] == 'sprout_edge' and
                len(identity_msg.attestation) > 0 and
                sage.messages_sent == 1
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test3_pass = False

    test_results["identity_announcement"] = test3_pass
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Reputation Proposal Broadcast
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Reputation Proposal Broadcast")
    print("=" * 72)
    print()

    print("Testing reputation proposal on edge...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sage = NetworkReadySAGE(
                node_id="sprout_edge",
                hardware_type="jetson_orin_nano",
                capability_level=3,
                storage_path=Path(tmpdir)
            )

            # Broadcast reputation proposal
            proposal_msg = sage.broadcast_reputation_proposal(
                target_node_id="peer_node",
                quality=0.85,
                event_type="verification"
            )

            print(f"  Proposal ID: {proposal_msg.payload['proposal_id'][:8]}...")
            print(f"  Target node: {proposal_msg.payload['target_node_id']}")
            print(f"  Quality: {proposal_msg.payload['quality_contribution']}")
            print(f"  Messages sent: {sage.messages_sent}")
            print(f"  Proposals broadcast: {sage.proposals_broadcast}")
            print()

            test4_pass = (
                proposal_msg.message_type == MessageType.REPUTATION_PROPOSAL.value and
                sage.proposals_broadcast == 1 and
                len(sage.pending_proposals) == 1
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["reputation_proposal"] = test4_pass
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Peer-to-Peer Communication Simulation
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Peer-to-Peer Communication")
    print("=" * 72)
    print()

    print("Testing simulated P2P between Thor and Sprout...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two nodes
            thor = NetworkReadySAGE(
                node_id="thor",
                hardware_type="jetson_agx_thor",
                capability_level=5,
                storage_path=Path(tmpdir) / "thor",
                network_address="10.0.0.99"
            )

            sprout = NetworkReadySAGE(
                node_id="sprout",
                hardware_type="jetson_orin_nano",
                capability_level=3,
                storage_path=Path(tmpdir) / "sprout",
                network_address="10.0.0.36"
            )

            # Exchange identities
            thor_identity = thor.announce_identity()
            sprout.receive_identity_announcement(thor_identity)

            sprout_identity = sprout.announce_identity()
            thor.receive_identity_announcement(sprout_identity)

            print(f"  Thor knows peers: {list(thor.known_peers.keys())}")
            print(f"  Sprout knows peers: {list(sprout.known_peers.keys())}")

            # Thor proposes reputation for Sprout
            proposal = thor.broadcast_reputation_proposal("sprout", 0.9)
            sprout.receive_reputation_proposal(proposal)

            # Sprout votes
            vote = sprout.cast_vote_on_proposal(
                proposal.payload['proposal_id'],
                VoteType.APPROVE
            )
            thor.receive_consensus_vote(vote)

            print(f"  Thor proposals: {thor.proposals_broadcast}")
            print(f"  Sprout votes: {sprout.votes_cast}")
            print(f"  Thor received: {thor.messages_received}")
            print()

            test5_pass = (
                "sprout" in thor.known_peers and
                "thor" in sprout.known_peers and
                thor.proposals_broadcast == 1 and
                sprout.votes_cast == 1
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test5_pass = False

    test_results["p2p_communication"] = test5_pass
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: JSONL Export/Import (using ProtocolMessage directly)
    # ========================================================================
    print("=" * 72)
    print("TEST 6: JSONL Export/Import")
    print("=" * 72)
    print()

    print("Testing JSONL file exchange (lightweight)...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSONL export/import without full SAGE (memory efficient)
            export_path = Path(tmpdir) / "messages.jsonl"

            # Create test messages directly
            messages = [
                ProtocolMessage(
                    message_type=MessageType.IDENTITY_ANNOUNCEMENT.value,
                    source_node_id="exporter",
                    timestamp=time.time(),
                    payload={"node_id": "exporter", "capability_level": 3},
                    attestation="test_attestation"
                ),
                ProtocolMessage(
                    message_type=MessageType.REPUTATION_PROPOSAL.value,
                    source_node_id="exporter",
                    timestamp=time.time(),
                    payload={"proposal_id": "test123", "target": "peer"},
                    attestation="test_attestation"
                )
            ]

            # Export to JSONL file
            with open(export_path, 'w') as f:
                for msg in messages:
                    f.write(msg.to_jsonl() + '\n')

            # Import from JSONL file
            imported = []
            with open(export_path, 'r') as f:
                for line in f:
                    if line.strip():
                        imported.append(ProtocolMessage.from_jsonl(line.strip()))

            print(f"  Exported: {len(messages)} messages")
            print(f"  Imported: {len(imported)} messages")
            print(f"  Round-trip verified: {imported[0].source_node_id}")
            print()

            test6_pass = (
                len(imported) == len(messages) and
                imported[0].source_node_id == "exporter" and
                imported[1].message_type == MessageType.REPUTATION_PROPOSAL.value
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test6_pass = False

    test_results["jsonl_export_import"] = test6_pass
    print(f"{'PASS' if test6_pass else 'FAIL'}: TEST 6")
    print()
    all_tests_passed = all_tests_passed and test6_pass

    # ========================================================================
    # TEST 7: Edge Performance Profile (Protocol-only, no LLM)
    # ========================================================================
    print("=" * 72)
    print("TEST 7: Edge Performance Profile")
    print("=" * 72)
    print()

    print("Profiling protocol operations on edge (lightweight)...")

    try:
        iterations = 1000  # More iterations since we're not loading LLM

        # Profile ProtocolMessage creation
        start = time.time()
        for i in range(iterations):
            msg = ProtocolMessage(
                message_type=MessageType.IDENTITY_ANNOUNCEMENT.value,
                source_node_id=f"node_{i}",
                timestamp=time.time(),
                payload={"test": i, "features": ["a", "b", "c"]},
                attestation="attestation_hash"
            )
        creation_time = time.time() - start
        creation_ops_per_sec = iterations / creation_time

        # Profile JSONL serialization round-trip
        test_msg = ProtocolMessage(
            message_type=MessageType.REPUTATION_PROPOSAL.value,
            source_node_id="perf_node",
            timestamp=time.time(),
            payload={"proposal_id": "123", "quality": 0.85, "target": "peer"},
            attestation="test_attestation_12345"
        )

        start = time.time()
        for _ in range(iterations):
            jsonl = test_msg.to_jsonl()
            _ = ProtocolMessage.from_jsonl(jsonl)
        serial_time = time.time() - start
        serial_ops_per_sec = iterations / serial_time

        # Profile MessageType enum access
        start = time.time()
        for _ in range(iterations * 10):
            _ = MessageType.REPUTATION_PROPOSAL.value
            _ = MessageType.CONSENSUS_VOTE.value
            _ = MessageType.IDENTITY_ANNOUNCEMENT.value
        enum_time = time.time() - start
        enum_ops_per_sec = (iterations * 10 * 3) / enum_time

        print(f"  Message creation: {creation_ops_per_sec:,.0f} ops/sec")
        print(f"  JSONL round-trip: {serial_ops_per_sec:,.0f} ops/sec")
        print(f"  MessageType access: {enum_ops_per_sec:,.0f} ops/sec")
        print()

        test7_pass = (
            creation_ops_per_sec > 1000 and
            serial_ops_per_sec > 1000 and
            enum_ops_per_sec > 100000
        )

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
    print("SESSION 183 EDGE VALIDATION SUMMARY")
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
        print("|" + "  NETWORK PROTOCOL SAGE VALIDATED ON EDGE!  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - JSONL serialization fast on ARM64")
        print("  - Identity announcement working")
        print("  - Reputation proposal broadcast operational")
        print("  - P2P communication simulated successfully")
        print("  - Protocol ready for LAN deployment")
        print()
        print("Network Stack Validated:")
        print("  Session 177-182: Core SAGE features")
        print("  Session 183: Network protocol (NOW VALIDATED)")
        print()
        print("Ready for Phase 1 LAN Deployment:")
        print("  - Thor (10.0.0.99): Development hub")
        print("  - Legion (10.0.0.72): High-ATP anchor")
        print("  - Sprout (10.0.0.36): Edge validation")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "183_edge",
        "title": "Network Protocol SAGE - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": test_results,
        "edge_metrics": edge_metrics,
        "protocol_features": {
            "identity_announcement": True,
            "reputation_proposals": True,
            "consensus_voting": True,
            "jsonl_serialization": True,
            "p2p_communication": True,
            "network_ready": True
        }
    }

    results_path = Path(__file__).parent / "session183_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_network_protocol()
    sys.exit(0 if success else 1)
