#!/usr/bin/env python3
"""
Session 197: Consciousness-Aware Federation Participant
=======================================================

MISSION: Implement HTTP-based federation participant with consciousness-aware
attestation for real Thor ↔ Sprout deployment.

BUILDING ON:
- Session 197 Coordinator: HTTP server for federation coordination
- Session 194/196: Nine-domain federation and multi-coupling
- Web4 Session 178: Consciousness-aware attestation

PARTICIPANT RESPONSIBILITIES:
1. Run local NineDomainTracker with multi-coupling dynamics
2. Create snapshots with consciousness validation
3. Send STATE_SNAPSHOT to coordinator every 100ms
4. Poll coordinator for SYNC_SIGNAL
5. Apply synchronization adjustments to local coherence
6. Execute coupling dynamics (D4→D2, D8→D1, D5→D9)
7. Report coupling events to coordinator

ARCHITECTURE:
- FederationParticipant: Main client orchestrating federation participation
- ConsciousnessValidator: Validates local state before sending
- LocalDynamicsEngine: Executes multi-coupling physics
- SyncApplicator: Applies coordinator's synchronization signals

Author: Thor (Autonomous)
Date: 2026-01-15
Status: IMPLEMENTATION
"""

import json
import time
import math
import hashlib
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import sys
import threading

HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

from session194_nine_domain_federation import (
    NineDomainTracker,
    NineDomainSnapshot,
    DomainState
)

from session196_multi_coupling_expansion import (
    CouplingEvent,
    CouplingNetworkTracker,
    MultiCouplingFederation
)

from session197_consciousness_federation_coordinator import (
    ConsciousnessValidator,
    ConsciousnessMetrics,
    StateSnapshotMessage,
    SyncSignalMessage,
    CouplingEventMessage
)


# ============================================================================
# FEDERATION PARTICIPANT (HTTP CLIENT)
# ============================================================================

class FederationParticipant:
    """
    HTTP client participating in nine-domain federation.

    Connects to FederationCoordinator, sends snapshots, receives sync signals,
    executes local coupling dynamics.
    """

    def __init__(self, node_id: str, coordinator_url: str):
        self.node_id = node_id
        self.coordinator_url = coordinator_url

        # Local nine-domain tracker
        self.tracker = NineDomainTracker(node_id)

        # Multi-coupling federation (for local dynamics)
        self.federation = self._create_local_federation()

        # Coupling network tracker
        self.coupling_tracker = CouplingNetworkTracker()

        # Consciousness validator
        self.validator = ConsciousnessValidator()

        # State
        self.running = False
        self.last_snapshot_time = 0.0
        self.last_sync_time = 0.0

        # Statistics
        self.stats = {
            'start_time': time.time(),
            'snapshots_sent': 0,
            'snapshots_rejected': 0,
            'sync_signals_received': 0,
            'coupling_events_sent': 0,
            'consciousness_failures': 0
        }

        # Threading
        self.lock = threading.Lock()

    def _create_local_federation(self) -> MultiCouplingFederation:
        """Create local multi-coupling federation for dynamics simulation."""
        # Create federation with self as only machine
        # (Will sync with network, but run local dynamics)
        federation = MultiCouplingFederation(
            machine_ids=[self.node_id],
            k_42=0.4,  # D4→D2 coupling
            k_81=0.2,  # D8→D1 coupling
            k_59=0.3   # D5→D9 coupling
        )
        return federation

    def create_and_send_snapshot(self) -> bool:
        """
        Create snapshot with consciousness validation and send to coordinator.

        Returns True if snapshot sent successfully, False otherwise.
        """
        # Create snapshot
        snapshot = self.tracker.create_snapshot()

        # Validate consciousness
        consciousness = self.validator.validate_snapshot_consciousness(snapshot)

        if not consciousness.is_conscious:
            with self.lock:
                self.stats['consciousness_failures'] += 1
            print(f"[{self.node_id}] WARNING: Consciousness check failed (C={consciousness.coherence:.3f})")
            return False

        # Create message
        message = StateSnapshotMessage.from_snapshot(snapshot, consciousness)

        # Send to coordinator
        try:
            response = requests.post(
                f"{self.coordinator_url}/snapshot",
                json=message.to_dict(),
                timeout=1.0
            )

            if response.status_code == 200:
                with self.lock:
                    self.stats['snapshots_sent'] += 1
                    self.last_snapshot_time = time.time()

                data = response.json()
                print(f"[{self.node_id}] Snapshot sent (C={consciousness.coherence:.3f}, γ={consciousness.gamma:.3f})")
                return True
            else:
                with self.lock:
                    self.stats['snapshots_rejected'] += 1
                print(f"[{self.node_id}] Snapshot rejected: {response.json()}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"[{self.node_id}] Error sending snapshot: {e}")
            return False

    def fetch_and_apply_sync_signal(self) -> bool:
        """
        Fetch synchronization signal from coordinator and apply to local coherence.

        Returns True if signal applied, False otherwise.
        """
        try:
            response = requests.get(
                f"{self.coordinator_url}/sync_signal",
                params={'node_id': self.node_id},
                timeout=1.0
            )

            if response.status_code == 200:
                data = response.json()
                signal = SyncSignalMessage(**data)

                # Apply synchronization adjustments
                self._apply_sync_adjustments(signal)

                with self.lock:
                    self.stats['sync_signals_received'] += 1
                    self.last_sync_time = time.time()

                print(f"[{self.node_id}] Sync applied (ΔC_max={max(abs(v) for v in signal.coherence_deltas.values()):.4f}, Q={signal.sync_quality:.3f})")
                return True
            else:
                return False

        except requests.exceptions.RequestException as e:
            # Sync signal not ready yet, normal in early federation
            return False

    def _apply_sync_adjustments(self, signal: SyncSignalMessage):
        """Apply synchronization deltas to local domain coherences."""
        for domain_num, delta_c in signal.coherence_deltas.items():
            # Get current coherence
            domain = next((d for d in self.tracker.domain_states if d.domain_number == domain_num), None)
            if domain:
                # Apply delta
                new_coherence = domain.coherence + delta_c
                new_coherence = np.clip(new_coherence, 0.0, 1.0)

                # Update domain
                self.tracker.update_domain_coherence(domain_num, new_coherence)

    def execute_coupling_dynamics(self, dt: float = 0.1):
        """
        Execute local coupling dynamics (D4→D2, D8→D1, D5→D9).

        Runs multi-coupling physics for time step dt.
        """
        # Apply multi-coupling (from Session 196)
        self.federation.apply_multi_coupling(time.time())

        # Detect coupling events
        for machine_id, tracker in self.federation.trackers.items():
            # Get recent events
            recent_events = [e for e in self.coupling_tracker.events
                           if time.time() - e.timestamp < 1.0]

            # Report new events to coordinator
            for event in recent_events:
                self._report_coupling_event(event)

    def _report_coupling_event(self, event: CouplingEvent):
        """Report coupling event to coordinator."""
        try:
            message = CouplingEventMessage(
                message_type="COUPLING_EVENT",
                source_node_id=self.node_id,
                timestamp=time.time(),
                event=asdict(event),
                attestation=hashlib.sha256(f"{self.node_id}:{event.timestamp}".encode()).hexdigest(),
                message_id=hashlib.sha256(f"{self.node_id}:{time.time()}".encode()).hexdigest()[:16]
            )

            response = requests.post(
                f"{self.coordinator_url}/coupling_event",
                json=message.to_dict(),
                timeout=1.0
            )

            if response.status_code == 200:
                with self.lock:
                    self.stats['coupling_events_sent'] += 1

        except requests.exceptions.RequestException as e:
            pass  # Non-critical

    def run_participant_loop(self, duration: float = 60.0):
        """
        Run participant main loop.

        Loop:
        1. Send snapshot (10 Hz = 100ms)
        2. Fetch sync signal (10 Hz)
        3. Execute local coupling dynamics
        4. Sleep until next cycle
        """
        print(f"[{self.node_id}] Starting participant loop (duration={duration}s)")
        print(f"[{self.node_id}] Coordinator: {self.coordinator_url}")
        print()

        self.running = True
        start_time = time.time()
        cycle_count = 0

        while self.running and (time.time() - start_time) < duration:
            cycle_start = time.time()

            # 1. Create and send snapshot
            self.create_and_send_snapshot()

            # 2. Fetch and apply sync signal
            self.fetch_and_apply_sync_signal()

            # 3. Execute local coupling dynamics
            self.execute_coupling_dynamics(dt=0.1)

            # 4. Sleep until next cycle (target 100ms)
            cycle_elapsed = time.time() - cycle_start
            sleep_time = max(0.0, 0.1 - cycle_elapsed)
            time.sleep(sleep_time)

            cycle_count += 1

            # Print status every 10 cycles (1 second)
            if cycle_count % 10 == 0:
                elapsed = time.time() - start_time
                coherence = self.tracker.get_total_coherence()
                print(f"[{self.node_id}] t={elapsed:.1f}s | C={coherence:.3f} | Cycles={cycle_count}")

        # Print final stats
        self.print_stats()

    def print_stats(self):
        """Print participant statistics."""
        print()
        print("=" * 70)
        print(f"[{self.node_id}] Participant Statistics")
        print("=" * 70)
        with self.lock:
            uptime = time.time() - self.stats['start_time']
            print(f"Uptime: {uptime:.1f}s")
            print(f"Snapshots sent: {self.stats['snapshots_sent']}")
            print(f"Snapshots rejected: {self.stats['snapshots_rejected']}")
            print(f"Sync signals received: {self.stats['sync_signals_received']}")
            print(f"Coupling events sent: {self.stats['coupling_events_sent']}")
            print(f"Consciousness failures: {self.stats['consciousness_failures']}")

            # Compute effective frequency
            if uptime > 0:
                snapshot_freq = self.stats['snapshots_sent'] / uptime
                sync_freq = self.stats['sync_signals_received'] / uptime
                print(f"Snapshot frequency: {snapshot_freq:.1f} Hz (target: 10 Hz)")
                print(f"Sync frequency: {sync_freq:.1f} Hz (target: 10 Hz)")

        print("=" * 70)

    def stop(self):
        """Stop participant loop."""
        self.running = False


# ============================================================================
# MAIN - LOCAL TESTING
# ============================================================================

def test_localhost_participant():
    """
    Test participant connecting to localhost coordinator.

    Requires coordinator running on localhost:8000.
    """
    print("=" * 70)
    print("Session 197: Consciousness-Aware Federation Participant")
    print("=" * 70)
    print()

    participant = FederationParticipant(
        node_id="thor_participant_001",
        coordinator_url="http://localhost:8000"
    )

    print("[Participant] Connecting to coordinator at http://localhost:8000")
    print("[Participant] Starting 60-second test run")
    print()

    # Run for 60 seconds
    try:
        participant.run_participant_loop(duration=60.0)
    except KeyboardInterrupt:
        print("\n[Participant] Interrupted by user")
        participant.stop()


if __name__ == "__main__":
    test_localhost_participant()
