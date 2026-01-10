#!/usr/bin/env python3
"""
Session 176: Sprout Server for 2-Node LAN Federation

Sprout-side server that waits for Thor to connect over actual LAN.

Network Configuration:
- Sprout: 10.0.0.36:8890 (Jetson Orin Nano 8GB, TPM2, Level 3)
- Thor:   10.0.0.99:8889 (Jetson AGX Thor, TrustZone, Level 5)

Usage:
1. Start this script on Sprout first
2. Start session176_thor_sprout_test.py on Thor
3. Thor connects to Sprout over LAN
4. Both machines participate in economic federation

Date: 2026-01-09
Machine: Sprout (Jetson Orin Nano 8GB)
"""

import asyncio
import json
import time
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

from session175_network_economic_federation import (
    NetworkEconomicCogitationNode,
    CogitationMode,
)
from session174_economic_cogitation import EconomicCogitationSession


def get_edge_metrics() -> Dict[str, Any]:
    """Get edge hardware metrics."""
    metrics = {
        "platform": "Jetson Orin Nano 8GB",
        "hardware_type": "tpm2",
        "capability_level": 3
    }

    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('MemAvailable:'):
                    available_kb = int(line.split()[1])
                    metrics["memory_available_mb"] = available_kb / 1024
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


class SproutLANServer:
    """Sprout server for 2-node LAN federation with Thor."""

    def __init__(self):
        self.running = True
        self.sprout = None
        self.server_task = None
        self.results = {
            "session": "176_sprout_server",
            "title": "Sprout Server for Thor LAN Federation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "edge_metrics": get_edge_metrics(),
            "events": [],
        }

    def log_event(self, event: str):
        """Log an event with timestamp."""
        ts = datetime.now(timezone.utc).isoformat()
        self.results["events"].append({"time": ts, "event": event})
        print(f"[{ts[:19]}] {event}")

    async def start(self):
        """Start Sprout server and wait for Thor."""
        print()
        print("+" + "=" * 70 + "+")
        print("|" + " " * 70 + "|")
        print("|" + "    SESSION 176: SPROUT SERVER FOR THOR LAN FEDERATION    ".center(70) + "|")
        print("|" + "           Waiting for Thor to connect...                 ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "=" * 70 + "+")
        print()

        edge = self.results["edge_metrics"]
        print("Sprout Configuration:")
        print(f"  IP Address: {edge.get('ip_address', '10.0.0.36')}")
        print(f"  Port: 8890")
        print(f"  Hardware: tpm2 (Level 3)")
        if 'temperature_c' in edge:
            print(f"  Temperature: {edge['temperature_c']}C")
        if 'memory_available_mb' in edge:
            print(f"  Memory: {int(edge['memory_available_mb'])} MB available")
        print()
        print("Expected Thor Configuration:")
        print(f"  IP Address: 10.0.0.99")
        print(f"  Port: 8889")
        print(f"  Hardware: trustzone (Level 5)")
        print()
        print("-" * 72)
        print()

        # Create Sprout node
        self.log_event("Creating Sprout node...")

        self.sprout = NetworkEconomicCogitationNode(
            node_id="sprout",
            hardware_type="tpm2",
            capability_level=3,
            listen_host="0.0.0.0",  # Listen on all interfaces
            listen_port=8890,
        )

        # Create test session
        self.sprout.cogitation_node.active_sessions["lan_test"] = EconomicCogitationSession(
            session_id="lan_test",
            topic="Thor-Sprout 2-node LAN federation test",
            start_time=datetime.now(timezone.utc)
        )

        self.log_event("Sprout node initialized, starting server...")

        # Start server
        self.server_task = asyncio.create_task(self.sprout.start())

        self.log_event("Sprout server started on 0.0.0.0:8890")
        self.log_event("Waiting for Thor (10.0.0.99:8889) to connect...")
        print()

        # Monitor loop - check for connections and log events
        last_peers = 0
        last_thoughts = 0
        iteration = 0

        while self.running:
            await asyncio.sleep(5)  # Check every 5 seconds
            iteration += 1

            metrics = self.sprout.get_metrics()
            current_peers = metrics['peers_verified']
            current_thoughts = metrics['thoughts_received']

            # Log new connections
            if current_peers > last_peers:
                self.log_event(f"NEW: Thor connected! Verified peers: {current_peers}")
                last_peers = current_peers

            # Log received thoughts
            if current_thoughts > last_thoughts:
                new_thoughts = current_thoughts - last_thoughts
                self.log_event(f"RECEIVED: {new_thoughts} thought(s) from Thor")
                last_thoughts = current_thoughts

            # Periodic status (every 60 seconds = 12 iterations)
            if iteration % 12 == 0:
                self.log_event(f"STATUS: Peers={current_peers}, Received={current_thoughts}, ATP={metrics['atp_balance']:.2f}")

    async def submit_thought(self, content: str):
        """Submit a thought from Sprout to federation."""
        if not self.sprout:
            print("ERROR: Sprout not initialized")
            return

        self.log_event(f"Submitting thought: '{content[:50]}...'")

        accepted, reason, thought = await self.sprout.submit_thought(
            session_id="lan_test",
            mode=CogitationMode.EXPLORING,
            content=content
        )

        if accepted:
            self.log_event(f"ACCEPTED: Coherence={thought.coherence_score:.3f}, Reward={thought.atp_reward:.2f}")
        else:
            self.log_event(f"REJECTED: {reason}")

    def show_status(self):
        """Show current status."""
        if not self.sprout:
            print("Sprout not initialized")
            return

        metrics = self.sprout.get_metrics()
        economics = self.sprout.get_network_economics()

        print()
        print("=" * 60)
        print("[SPROUT STATUS]")
        print("=" * 60)
        print(f"  Running: {metrics['running']}")
        print(f"  Peers Connected: {metrics['peers_connected']}")
        print(f"  Peers Verified: {metrics['peers_verified']}")
        print(f"  ATP Balance: {metrics['atp_balance']:.2f}")
        print(f"  Thoughts Federated: {metrics['thoughts_federated']}")
        print(f"  Thoughts Received: {metrics['thoughts_received']}")
        print(f"  Messages Sent: {metrics['messages_sent']}")
        print(f"  Messages Received: {metrics['messages_received']}")
        print()
        print("[NETWORK ECONOMICS]")
        print(f"  Total ATP: {economics['total_network_atp']:.2f}")
        print(f"  Nodes: {economics['nodes_in_network']}")
        for node_id, balance in economics['node_balances'].items():
            print(f"    {node_id}: {balance:.2f} ATP")
        print("=" * 60)
        print()

    async def stop(self):
        """Stop Sprout server."""
        self.running = False
        self.log_event("Shutting down Sprout server...")

        if self.sprout:
            await self.sprout.stop()

        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass

        self.log_event("Sprout server stopped")

        # Save results
        self.results["final_metrics"] = self.sprout.get_metrics() if self.sprout else {}

        results_path = HOME / "ai-workspace" / "HRM" / "sage" / "experiments" / "session176_sprout_server_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved: {results_path}")


async def interactive_mode(server: SproutLANServer):
    """Interactive command mode."""
    print()
    print("Commands:")
    print("  status - Show current status")
    print("  submit <text> - Submit a thought")
    print("  quit - Shutdown server")
    print()

    while server.running:
        try:
            cmd = await asyncio.get_event_loop().run_in_executor(
                None, input, "[sprout]> "
            )

            if cmd.strip() == "quit":
                await server.stop()
                break
            elif cmd.strip() == "status":
                server.show_status()
            elif cmd.strip().startswith("submit "):
                content = cmd.strip()[7:]
                await server.submit_thought(content)
            elif cmd.strip():
                print(f"Unknown command: {cmd}")

        except EOFError:
            break
        except KeyboardInterrupt:
            break


async def main():
    """Main entry point."""
    server = SproutLANServer()

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived signal, shutting down...")
        server.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    server_task = asyncio.create_task(server.start())

    # Wait a moment for startup
    await asyncio.sleep(3)

    # Run interactive mode
    interact_task = asyncio.create_task(interactive_mode(server))

    # Wait for either to complete
    done, pending = await asyncio.wait(
        [server_task, interact_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    # Cleanup
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if server.running:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
