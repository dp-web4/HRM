#!/usr/bin/env python3
"""
Session 176: Real LAN Deployment Script

Deploys Session 175 network economic federation to real distributed machines.

Usage:
    # On Legion (hub):
    python3 session176_deploy.py --node legion --port 8888

    # On Thor:
    python3 session176_deploy.py --node thor --port 8889 --connect legion:<legion-ip>:8888

    # On Sprout:
    python3 session176_deploy.py --node sprout --port 8890 --connect legion:<legion-ip>:8888 thor:<thor-ip>:8889

Date: 2026-01-09
Machine: All (Legion, Thor, Sprout)
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime, timezone
from typing import List, Tuple

from session175_network_economic_federation import (
    NetworkEconomicCogitationNode,
    CogitationMode,
)
from session174_economic_cogitation import EconomicCogitationSession


class Session176Deployment:
    """Manages deployment of network economic federation on real machines."""

    def __init__(
        self,
        node_id: str,
        hardware_type: str,
        capability_level: int,
        listen_port: int,
        peers: List[Tuple[str, str, int]],  # [(node_id, host, port), ...]
    ):
        """
        Initialize deployment.

        Args:
            node_id: Unique node identifier (legion, thor, sprout)
            hardware_type: Hardware security type
            capability_level: Capability level (1-5)
            listen_port: Port to listen on
            peers: List of peers to connect to
        """
        self.node_id = node_id
        self.hardware_type = hardware_type
        self.capability_level = capability_level
        self.listen_port = listen_port
        self.peers = peers

        # Create node
        self.node = NetworkEconomicCogitationNode(
            node_id=node_id,
            hardware_type=hardware_type,
            capability_level=capability_level,
            listen_host="0.0.0.0",  # Listen on all interfaces
            listen_port=listen_port,
        )

        # Create test session
        self.node.cogitation_node.active_sessions["lan_test"] = EconomicCogitationSession(
            session_id="lan_test",
            topic="Real LAN economic federation test",
            start_time=datetime.now(timezone.utc)
        )

        self.running = True

    async def start(self):
        """Start node and connect to peers."""
        print(f"\n{'='*80}")
        print(f"SESSION 176: REAL LAN DEPLOYMENT")
        print(f"{'='*80}")
        print(f"\nNode: {self.node_id}")
        print(f"Hardware: {self.hardware_type}")
        print(f"Capability: {self.capability_level}")
        print(f"Port: {self.listen_port}")
        print(f"Peers: {len(self.peers)}")
        print(f"\n{'='*80}\n")

        # Start server
        server_task = asyncio.create_task(self.node.start())

        # Wait for server to start
        await asyncio.sleep(2)

        # Connect to peers
        for peer_id, host, port in self.peers:
            print(f"[{self.node_id}] Connecting to {peer_id} at {host}:{port}...")
            await self.node.connect_to_peer(host, port)
            await asyncio.sleep(1)

        print(f"\n[{self.node_id}] Node started and connected to {len(self.peers)} peers")
        print(f"[{self.node_id}] Ready for thought submission\n")

        # Print status
        self.print_status()

        # Wait for server
        try:
            await server_task
        except asyncio.CancelledError:
            print(f"\n[{self.node_id}] Shutting down...")
            await self.node.stop()

    def print_status(self):
        """Print current node status."""
        metrics = self.node.get_metrics()

        print(f"\n{'='*80}")
        print(f"[{self.node_id}] STATUS")
        print(f"{'='*80}")
        print(f"Running: {metrics['running']}")
        print(f"Peers Connected: {metrics['peers_connected']}")
        print(f"Peers Verified: {metrics['peers_verified']}")
        print(f"ATP Balance: {metrics['atp_balance']:.2f}")
        print(f"Thoughts Federated: {metrics['thoughts_federated']}")
        print(f"Thoughts Received: {metrics['thoughts_received']}")
        print(f"Messages Sent: {metrics['messages_sent']}")
        print(f"Messages Received: {metrics['messages_received']}")
        print(f"{'='*80}\n")

    async def submit_test_thought(self, content: str):
        """Submit a test thought."""
        print(f"[{self.node_id}] Submitting test thought...")

        accepted, reason, thought = await self.node.submit_thought(
            session_id="lan_test",
            mode=CogitationMode.EXPLORING,
            content=content
        )

        if accepted:
            print(f"[{self.node_id}] ✅ Thought accepted!")
            print(f"[{self.node_id}]   Coherence: {thought.coherence_score:.3f}")
            print(f"[{self.node_id}]   ATP Reward: {thought.atp_reward:.2f}")
            print(f"[{self.node_id}]   New Balance: {thought.contributor_atp_balance:.2f}")
        else:
            print(f"[{self.node_id}] ❌ Thought rejected: {reason}")

        return accepted, reason, thought


async def interactive_mode(deployment: Session176Deployment):
    """Run interactive mode for manual testing."""
    print(f"\n[{deployment.node_id}] Interactive mode - Commands:")
    print(f"  status - Show node status")
    print(f"  submit <text> - Submit a thought")
    print(f"  metrics - Show network metrics")
    print(f"  quit - Exit")
    print()

    while deployment.running:
        try:
            # Get user input (non-blocking)
            cmd = await asyncio.get_event_loop().run_in_executor(
                None, input, f"[{deployment.node_id}]> "
            )

            if cmd.strip() == "quit":
                print(f"[{deployment.node_id}] Exiting...")
                deployment.running = False
                break

            elif cmd.strip() == "status":
                deployment.print_status()

            elif cmd.strip() == "metrics":
                economics = deployment.node.get_network_economics()
                print(f"\n{'='*80}")
                print(f"NETWORK ECONOMICS")
                print(f"{'='*80}")
                print(f"Total Network ATP: {economics['total_network_atp']:.2f}")
                print(f"Average Balance: {economics['average_balance']:.2f}")
                print(f"ATP Inequality: {economics['atp_inequality']:.2f}")
                print(f"Nodes in Network: {economics['nodes_in_network']}")
                print(f"\nNode Balances:")
                for node_id, balance in economics['node_balances'].items():
                    print(f"  {node_id}: {balance:.2f} ATP")
                print(f"{'='*80}\n")

            elif cmd.strip().startswith("submit "):
                content = cmd.strip()[7:]  # Remove "submit "
                await deployment.submit_test_thought(content)

            else:
                print(f"Unknown command: {cmd}")

        except EOFError:
            break
        except KeyboardInterrupt:
            break


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Session 176 network economic federation"
    )
    parser.add_argument(
        "--node",
        required=True,
        choices=["legion", "thor", "sprout"],
        help="Node identifier"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port to listen on (default: 8888)"
    )
    parser.add_argument(
        "--hardware",
        choices=["tpm2", "trustzone", "software"],
        help="Hardware type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--capability",
        type=int,
        default=5,
        help="Capability level 1-5 (default: 5)"
    )
    parser.add_argument(
        "--connect",
        action="append",
        help="Peer to connect to: node_id:host:port (can be specified multiple times)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Auto-detect hardware if not specified
    if args.hardware is None:
        if args.node == "legion":
            hardware = "tpm2"
        elif args.node == "thor":
            hardware = "trustzone"
        elif args.node == "sprout":
            hardware = "tpm2"
        else:
            hardware = "software"
    else:
        hardware = args.hardware

    # Parse peer connections
    peers = []
    if args.connect:
        for peer_spec in args.connect:
            parts = peer_spec.split(":")
            if len(parts) != 3:
                print(f"Error: Invalid peer spec: {peer_spec}")
                print(f"Expected format: node_id:host:port")
                sys.exit(1)

            peer_id, host, port = parts
            peers.append((peer_id, host, int(port)))

    # Create deployment
    deployment = Session176Deployment(
        node_id=args.node,
        hardware_type=hardware,
        capability_level=args.capability,
        listen_port=args.port,
        peers=peers
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n[{args.node}] Received signal {sig}, shutting down...")
        deployment.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start deployment
    start_task = asyncio.create_task(deployment.start())

    # Run interactive mode if requested
    if args.interactive:
        await asyncio.sleep(3)  # Wait for startup
        interact_task = asyncio.create_task(interactive_mode(deployment))

        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [start_task, interact_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

    else:
        # Just wait for server
        await start_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
