"""
Peer Client — HTTP client for peer-to-peer SAGE messaging.

Sends messages to peer SAGEs via their gateway /chat endpoints.
Uses FleetRegistry for address lookup and PeerMonitor for online status.

Usage:
    client = PeerClient(fleet_registry, peer_monitor)
    result = client.send_message('thor', 'Hello from CBP!')
    print(result)  # {'success': True, 'response': '...', ...}
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

from sage.federation.fleet_registry import FleetRegistry
from sage.federation.peer_monitor import PeerMonitor


class PeerClient:
    """HTTP client for sending messages to peer SAGE instances."""

    def __init__(
        self,
        fleet_registry: FleetRegistry,
        peer_monitor: PeerMonitor,
        timeout: float = 30.0,
        trust_tracker=None,
    ):
        self.fleet_registry = fleet_registry
        self.peer_monitor = peer_monitor
        self.timeout = timeout
        self.trust_tracker = trust_tracker
        self.self_machine = fleet_registry.self_machine

        # Stats
        self.messages_sent = 0
        self.messages_failed = 0

    def send_message(
        self,
        target_machine: str,
        message: str,
        conversation_id: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to a peer SAGE via its /chat endpoint.

        Args:
            target_machine: Machine name (e.g., 'thor', 'sprout')
            message: Message text to send
            conversation_id: Optional conversation ID for multi-turn
            sender: Optional sender ID (defaults to self machine's LCT)

        Returns:
            Dict with 'success', 'response', 'metabolic_state', etc.
        """
        # Look up peer
        peer = self.fleet_registry.get_peer(target_machine)
        if peer is None:
            self.messages_failed += 1
            return {
                'success': False,
                'error': f"Unknown peer: {target_machine}",
                'target': target_machine,
            }

        # Check if online (warn but still try — maybe monitor hasn't caught up)
        if not self.peer_monitor.is_online(target_machine):
            print(f"[PeerClient] Warning: {target_machine} appears offline, attempting anyway")

        # Build request
        url = f"http://{peer['gateway_host']}:{peer['gateway_port']}/chat"
        self_info = self.fleet_registry.get_self()
        self_lct = self_info.get('lct_id', self.self_machine) if self_info else self.self_machine

        payload = {
            'sender': sender or self_lct,
            'message': message,
            'max_wait_seconds': self.timeout,
        }
        if conversation_id:
            payload['conversation_id'] = conversation_id

        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                latency = (time.monotonic() - start) * 1000
                self.messages_sent += 1
                if self.trust_tracker:
                    self.trust_tracker.record_interaction(target_machine, 'success')
                return {
                    'success': True,
                    'target': target_machine,
                    'response': data.get('response', ''),
                    'metabolic_state': data.get('metabolic_state'),
                    'atp_remaining': data.get('atp_remaining'),
                    'latency_ms': round(latency, 1),
                }

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            self.messages_failed += 1
            if self.trust_tracker:
                self.trust_tracker.record_interaction(target_machine, 'error')
            return {
                'success': False,
                'target': target_machine,
                'error': str(e)[:300],
                'latency_ms': round((time.monotonic() - start) * 1000, 1),
            }

        except (json.JSONDecodeError, ValueError) as e:
            self.messages_failed += 1
            if self.trust_tracker:
                self.trust_tracker.record_interaction(target_machine, 'error')
            return {
                'success': False,
                'target': target_machine,
                'error': f"Invalid response: {e}",
            }

    def delegate_task(
        self,
        target_machine: str,
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Delegate a task to a peer SAGE via its /delegate endpoint.

        Args:
            target_machine: Machine name
            task: Task dict (task_id, description, reward, etc.)

        Returns:
            Dict with 'success', 'proof', etc.
        """
        peer = self.fleet_registry.get_peer(target_machine)
        if peer is None:
            return {'success': False, 'error': f"Unknown peer: {target_machine}"}

        url = f"http://{peer['gateway_host']}:{peer['gateway_port']}/delegate"
        body = json.dumps(task).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return {
                    'success': True,
                    'target': target_machine,
                    **data,
                }
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            return {
                'success': False,
                'target': target_machine,
                'error': str(e)[:300],
            }

    def get_stats(self) -> Dict[str, Any]:
        """Return messaging stats."""
        return {
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed,
            'total_attempts': self.messages_sent + self.messages_failed,
        }

    def __repr__(self) -> str:
        return f"PeerClient(self={self.self_machine}, sent={self.messages_sent}, failed={self.messages_failed})"
