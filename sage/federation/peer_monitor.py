"""
Peer Monitor — background health polling for SAGE fleet awareness.

Runs a daemon thread that periodically polls peer /health endpoints.
Maintains a peer_states dict with online status, metabolic state, model,
ATP level, and latency for each known peer.

Usage:
    from sage.federation.fleet_registry import FleetRegistry
    from sage.federation.peer_monitor import PeerMonitor

    registry = FleetRegistry('cbp')
    monitor = PeerMonitor(registry, 'cbp')
    monitor.start()

    # Later...
    states = monitor.get_peer_states()
    online = monitor.get_online_peers()
    monitor.stop()
"""

import json
import time
import threading
import urllib.request
import urllib.error
from typing import Dict, List, Optional

from sage.federation.fleet_registry import FleetRegistry


class PeerMonitor:
    """Background thread that polls peer SAGE health endpoints."""

    def __init__(
        self,
        fleet_registry: FleetRegistry,
        self_machine: str,
        poll_interval: float = 30.0,
        timeout: float = 2.0,
        trust_tracker=None,
    ):
        self.fleet_registry = fleet_registry
        self.self_machine = self_machine
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.trust_tracker = trust_tracker

        self._peer_states: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Initialize all peers as unknown
        for name in fleet_registry.get_peer_names():
            self._peer_states[name] = {
                'online': False,
                'last_seen': None,
                'last_checked': None,
                'metabolic_state': None,
                'model_size': None,
                'atp_level': None,
                'cycle_count': None,
                'latency_ms': None,
                'lct_id': fleet_registry.get_peer(name).get('lct_id', ''),
                'hardware': fleet_registry.get_peer(name).get('hardware', ''),
                'error': None,
            }

    def start(self):
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name='sage-peer-monitor',
        )
        self._thread.start()
        print(f"[PeerMonitor] Started — polling {len(self._peer_states)} peers every {self.poll_interval}s")

    def stop(self):
        """Signal the polling thread to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        print("[PeerMonitor] Stopped")

    def get_peer_states(self) -> Dict[str, dict]:
        """Return copy of current peer states."""
        with self._lock:
            return {name: dict(state) for name, state in self._peer_states.items()}

    def get_online_peers(self) -> List[str]:
        """Return names of currently online peers."""
        with self._lock:
            return [name for name, state in self._peer_states.items() if state['online']]

    def get_peer_state(self, machine_name: str) -> Optional[dict]:
        """Return state for a specific peer."""
        with self._lock:
            state = self._peer_states.get(machine_name)
            return dict(state) if state else None

    def is_online(self, machine_name: str) -> bool:
        """Check if a specific peer is online."""
        with self._lock:
            state = self._peer_states.get(machine_name)
            return state['online'] if state else False

    @property
    def online_count(self) -> int:
        """Number of peers currently online."""
        with self._lock:
            return sum(1 for s in self._peer_states.values() if s['online'])

    def _poll_loop(self):
        """Main polling loop — runs in background thread."""
        # Do a first poll immediately
        self._poll_all_peers()

        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval)
            if not self._stop_event.is_set():
                self._poll_all_peers()

    def _poll_all_peers(self):
        """Poll all peers once."""
        for name in self.fleet_registry.get_peer_names():
            if self._stop_event.is_set():
                break
            self._poll_peer(name)

    def _poll_peer(self, machine_name: str):
        """Poll a single peer's /health endpoint."""
        url = self.fleet_registry.get_gateway_url(machine_name)
        if url is None:
            return

        health_url = f"{url}/health"
        start = time.monotonic()

        try:
            req = urllib.request.Request(health_url, method='GET')
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                latency = (time.monotonic() - start) * 1000

                with self._lock:
                    self._peer_states[machine_name].update({
                        'online': True,
                        'last_seen': time.time(),
                        'last_checked': time.time(),
                        'metabolic_state': data.get('metabolic_state'),
                        'model_size': data.get('model_size'),
                        'atp_level': data.get('atp_current') or data.get('atp_remaining'),
                        'cycle_count': data.get('cycle_count'),
                        'latency_ms': round(latency, 1),
                        'error': None,
                    })

            # Record successful health check in trust tracker
            if self.trust_tracker:
                self.trust_tracker.record_interaction(machine_name, 'success')

        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError, ValueError) as e:
            with self._lock:
                self._peer_states[machine_name].update({
                    'online': False,
                    'last_checked': time.time(),
                    'latency_ms': None,
                    'error': str(e)[:200],
                })

            # Record timeout/error in trust tracker
            if self.trust_tracker:
                self.trust_tracker.record_interaction(machine_name, 'timeout')

    def __repr__(self) -> str:
        return f"PeerMonitor(self={self.self_machine}, online={self.online_count}/{len(self._peer_states)})"
