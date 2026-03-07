"""
Fleet Registry — loads the static fleet manifest and provides peer lookup.

The fleet manifest (fleet.json) is the phone book: it tells each SAGE who
else exists and where to find them. The registry is immutable after load —
runtime state (online/offline, trust scores) lives in PeerMonitor and
PeerTrustTracker, not here.

KNOWN LIMITATION: IPs in fleet.json are hardcoded.  Machines on DHCP
(e.g. McNugget) can change IP between boots.  During testing on 2026-03-07,
McNugget was registered at 10.0.0.150 but actually reachable at 10.0.0.249
(discovered via mDNS: ``mcnugget.local``).

TODO: Replace static IPs with a discovery mechanism:
  - Option A: mDNS/Avahi — resolve ``<machine>.local`` at runtime
  - Option B: Fleet presence files — each machine writes its IP to a
    shared location (NFS, git, or a central registry endpoint)
  - Option C: Broadcast/multicast discovery on the LAN
  mDNS is the simplest path — all machines already advertise via Avahi.

Usage:
    registry = FleetRegistry('cbp')
    peers = registry.get_peers()       # all machines except self
    thor = registry.get_peer('thor')   # specific peer info
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class FleetRegistry:
    """Immutable registry of SAGE fleet members loaded from fleet.json."""

    def __init__(self, self_machine: str, manifest_path: Optional[str] = None):
        self.self_machine = self_machine

        if manifest_path is None:
            manifest_path = str(Path(__file__).parent / 'fleet.json')

        with open(manifest_path, 'r') as f:
            data = json.load(f)

        self._machines: Dict[str, dict] = data.get('machines', {})
        self._version = data.get('fleet_version', 0)

        # Inject machine name into each entry for convenience
        for name, info in self._machines.items():
            info['machine_name'] = name

    @property
    def version(self) -> int:
        return self._version

    @property
    def fleet_size(self) -> int:
        return len(self._machines)

    def get_all(self) -> Dict[str, dict]:
        """Return all machines including self."""
        return dict(self._machines)

    def get_peers(self) -> Dict[str, dict]:
        """Return all machines except self."""
        return {
            name: info for name, info in self._machines.items()
            if name != self.self_machine
        }

    def get_peer(self, machine_name: str) -> Optional[dict]:
        """Return info for a specific machine, or None if not in manifest."""
        return self._machines.get(machine_name)

    def get_self(self) -> Optional[dict]:
        """Return info for self machine."""
        return self._machines.get(self.self_machine)

    def get_peer_names(self) -> List[str]:
        """Return list of peer machine names (excluding self)."""
        return [name for name in self._machines if name != self.self_machine]

    def get_gateway_url(self, machine_name: str) -> Optional[str]:
        """Return HTTP gateway URL for a machine."""
        info = self._machines.get(machine_name)
        if info is None:
            return None
        return f"http://{info['gateway_host']}:{info['gateway_port']}"

    def __repr__(self) -> str:
        peer_count = len(self._machines) - (1 if self.self_machine in self._machines else 0)
        return f"FleetRegistry(self={self.self_machine}, peers={peer_count}, v{self._version})"
