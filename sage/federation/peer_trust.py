"""
Peer Trust Tracker — per-peer T3 trust scores with persistence.

Tracks Talent/Training/Temperament scores for each peer SAGE based on
interaction outcomes. Trust evolves from real interactions via exponential
moving average — recent interactions weight more than old ones.

Trust is directional: CBP's trust in Thor may differ from Thor's trust in CBP.

Usage:
    tracker = PeerTrustTracker('/path/to/state/peer_trust_cbp.json')
    tracker.record_interaction('thor', 'success')
    tracker.record_interaction('thor', 'quality_high')
    print(tracker.get_trust('thor'))
    # {'talent': 0.55, 'training': 0.52, 'temperament': 0.55, ...}
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Any


# Default T3 scores for a new peer (neutral — not trusted, not distrusted)
_DEFAULT_T3 = {
    'talent': 0.5,
    'training': 0.5,
    'temperament': 0.5,
}

# How interaction outcomes affect T3 dimensions
_OUTCOME_DELTAS = {
    'success': {'talent': 0.05, 'training': 0.02, 'temperament': 0.05},
    'timeout': {'talent': 0.0, 'training': 0.0, 'temperament': -0.10},
    'error': {'talent': -0.05, 'training': -0.02, 'temperament': -0.05},
    'quality_high': {'talent': 0.10, 'training': 0.05, 'temperament': 0.02},
    'quality_low': {'talent': -0.08, 'training': -0.03, 'temperament': -0.02},
}


class PeerTrustTracker:
    """Tracks per-peer T3 trust scores with JSON persistence."""

    def __init__(self, state_path: str, alpha: float = 0.1):
        """
        Args:
            state_path: Path to JSON file for persistence
            alpha: EMA smoothing factor (0.1 = 10% weight on new observation)
        """
        self.state_path = Path(state_path)
        self.alpha = alpha
        self._peers: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load trust state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                self._peers = data.get('peers', {})
            except (json.JSONDecodeError, IOError):
                self._peers = {}

    def _save(self):
        """Persist trust state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump({
                    'peers': self._peers,
                    'last_updated': time.time(),
                }, f, indent=2)
        except IOError as e:
            print(f"[PeerTrust] Failed to save: {e}")

    def _ensure_peer(self, machine_name: str):
        """Initialize a peer entry if it doesn't exist."""
        if machine_name not in self._peers:
            self._peers[machine_name] = {
                **_DEFAULT_T3,
                'interactions': 0,
                'last_updated': time.time(),
            }

    def record_interaction(self, machine_name: str, outcome: str):
        """
        Record an interaction outcome and update T3 scores.

        Args:
            machine_name: Peer machine name (e.g., 'thor')
            outcome: One of 'success', 'timeout', 'error', 'quality_high', 'quality_low'
        """
        deltas = _OUTCOME_DELTAS.get(outcome)
        if deltas is None:
            return

        self._ensure_peer(machine_name)
        peer = self._peers[machine_name]

        # EMA update: new = (1 - alpha) * old + alpha * (old + delta)
        # Simplified: new = old + alpha * delta
        for dim, delta in deltas.items():
            old_val = peer[dim]
            new_val = old_val + self.alpha * delta
            peer[dim] = max(0.0, min(1.0, new_val))  # clamp [0, 1]

        peer['interactions'] += 1
        peer['last_updated'] = time.time()

        # Persist every 5 interactions (not every time — avoid I/O thrashing)
        if peer['interactions'] % 5 == 0:
            self._save()

    def get_trust(self, machine_name: str) -> Optional[Dict[str, Any]]:
        """Return T3 trust scores for a peer, or None if unknown."""
        self._ensure_peer(machine_name)
        return dict(self._peers[machine_name])

    def get_reputation(self, machine_name: str) -> float:
        """Return single reputation score (geometric mean of T3)."""
        trust = self.get_trust(machine_name)
        if trust is None:
            return 0.5
        t = trust['talent']
        tr = trust['training']
        te = trust['temperament']
        return (t * tr * te) ** (1.0 / 3.0)

    def get_all_trust(self) -> Dict[str, dict]:
        """Return trust scores for all known peers."""
        return {name: dict(data) for name, data in self._peers.items()}

    def save(self):
        """Force save to disk."""
        self._save()

    def __repr__(self) -> str:
        return f"PeerTrustTracker(peers={len(self._peers)}, path={self.state_path.name})"
