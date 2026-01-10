#!/usr/bin/env python3
"""
Thor Session 180: Persistent Reputation Storage

Research Goal: Integrate Legion's persistent reputation storage with Session 179's
reputation-aware adaptive depth, enabling reputation that survives across sessions.

Architecture Evolution:
- Session 177: ATP-adaptive depth (ephemeral state)
- Session 178: Federated coordination (ephemeral network state)
- Session 179: Reputation-aware depth (ephemeral reputation)
- Session 180: Persistent reputation (cross-session trust)

Key Innovation: Reputation anchored to hardware identity (LCT), survives restarts,
accumulates over time, creates long-term trust relationships.

Platform: Thor (Jetson AGX Thor, TrustZone L5)
Type: Autonomous Research
Date: 2026-01-10
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Import Session 179 as base
import sys
HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

from session179_reputation_aware_depth import (
    ReputationAwareAdaptiveSAGE,
    SimpleReputation
)

from session178_federated_sage_verification import (
    CognitiveDepth
)


# ============================================================================
# PERSISTENT REPUTATION TYPES
# ============================================================================

@dataclass
class ReputationEvent:
    """A single reputation-affecting event (persisted to disk)."""
    event_id: str
    node_id: str  # Hardware-anchored identity
    event_type: str  # quality_event, verification, violation
    impact: float  # Quality score affecting reputation
    timestamp: float
    context: Dict[str, Any]  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReputationEvent':
        """Create from dict."""
        return cls(**data)


@dataclass
class PersistentReputationScore:
    """
    Persistent reputation score (stored to disk).

    Compatible with Session 179's SimpleReputation but adds persistence,
    event history, and cross-session tracking.
    """
    node_id: str
    total_score: float
    event_count: int
    positive_events: int
    negative_events: int
    last_updated: float
    first_seen: float

    # Session 179 compatibility
    @property
    def reputation_level(self) -> str:
        """Categorical reputation level (Session 179 compatible)."""
        if self.total_score >= 50:
            return "excellent"
        elif self.total_score >= 20:
            return "good"
        elif self.total_score >= 0:
            return "neutral"
        elif self.total_score >= -20:
            return "poor"
        else:
            return "untrusted"

    @property
    def reputation_multiplier(self) -> float:
        """
        Cognitive credit multiplier (Session 179 compatible).

        High reputation → lower multiplier → higher effective ATP
        Low reputation → higher multiplier → lower effective ATP
        """
        if self.total_score >= 50:
            return 0.7   # Excellent: 30% bonus
        elif self.total_score >= 20:
            return 0.85  # Good: 15% bonus
        elif self.total_score >= 0:
            return 1.0   # Neutral: no adjustment
        elif self.total_score >= -20:
            return 1.15  # Poor: 15% penalty
        else:
            return 1.3   # Untrusted: 30% penalty

    @property
    def average_score(self) -> float:
        """Average reputation per event."""
        return self.total_score / self.event_count if self.event_count > 0 else 0.0

    @property
    def positive_ratio(self) -> float:
        """Ratio of positive to total events."""
        return self.positive_events / self.event_count if self.event_count > 0 else 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        data = asdict(self)
        data['reputation_level'] = self.reputation_level
        data['reputation_multiplier'] = self.reputation_multiplier
        data['average_score'] = self.average_score
        data['positive_ratio'] = self.positive_ratio
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistentReputationScore':
        """Create from dict."""
        # Remove calculated properties
        data = {k: v for k, v in data.items()
                if k in ['node_id', 'total_score', 'event_count',
                         'positive_events', 'negative_events',
                         'last_updated', 'first_seen']}
        return cls(**data)

    def to_simple_reputation(self) -> SimpleReputation:
        """Convert to Session 179's SimpleReputation for compatibility."""
        return SimpleReputation(
            node_id=self.node_id,
            total_score=self.total_score,
            event_count=self.event_count,
            positive_events=self.positive_events,
            negative_events=self.negative_events
        )


# ============================================================================
# PERSISTENT REPUTATION MANAGER
# ============================================================================

class PersistentReputationManager:
    """
    Manages persistent reputation storage and retrieval.

    Storage Strategy (inspired by Legion's prototype):
    - Events: Append-only JSONL file (complete history)
    - Scores: JSON file (current aggregated scores)
    - Recovery: Rebuild scores from events if corrupted

    Thread Safety: Single-writer per node (manager owns storage)
    """

    def __init__(self, storage_path: Path, node_id: str):
        """
        Initialize persistent reputation manager.

        Args:
            storage_path: Directory for reputation storage
            node_id: Hardware-anchored node identity
        """
        self.storage_path = storage_path
        self.node_id = node_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Storage files
        self.events_file = self.storage_path / f"{node_id}_reputation_events.jsonl"
        self.scores_file = self.storage_path / f"{node_id}_reputation_scores.json"

        # In-memory caches
        self.scores: Dict[str, PersistentReputationScore] = {}
        self.events: List[ReputationEvent] = []

        # Load existing data
        self._load()

    def _load(self):
        """Load reputation data from disk."""
        # Load events (append-only log)
        if self.events_file.exists():
            with open(self.events_file, 'r') as f:
                for line in f:
                    try:
                        event_data = json.loads(line)
                        event = ReputationEvent.from_dict(event_data)
                        self.events.append(event)
                    except json.JSONDecodeError:
                        continue  # Skip corrupt lines

        # Load scores (aggregated state)
        if self.scores_file.exists():
            with open(self.scores_file, 'r') as f:
                try:
                    scores_data = json.load(f)
                    for node_id, score_data in scores_data.items():
                        self.scores[node_id] = PersistentReputationScore.from_dict(score_data)
                except json.JSONDecodeError:
                    # Rebuild from events if scores corrupted
                    self._rebuild_from_events()

    def _rebuild_from_events(self):
        """Rebuild scores from event log (recovery mechanism)."""
        self.scores = {}
        for event in self.events:
            if event.node_id not in self.scores:
                self.scores[event.node_id] = PersistentReputationScore(
                    node_id=event.node_id,
                    total_score=0.0,
                    event_count=0,
                    positive_events=0,
                    negative_events=0,
                    last_updated=event.timestamp,
                    first_seen=event.timestamp
                )

            score = self.scores[event.node_id]
            score.total_score += event.impact
            score.event_count += 1
            score.last_updated = event.timestamp

            if event.impact > 0:
                score.positive_events += 1
            elif event.impact < 0:
                score.negative_events += 1

        self._save_scores()

    def _save_event(self, event: ReputationEvent):
        """Append event to disk (JSONL format for append-only log)."""
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')

    def _save_scores(self):
        """Save all scores to disk (atomic write for consistency)."""
        scores_data = {node_id: score.to_dict()
                      for node_id, score in self.scores.items()}

        # Atomic write: write to temp file, then rename
        temp_file = self.scores_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(scores_data, f, indent=2)
        temp_file.replace(self.scores_file)

    def record_event(
        self,
        node_id: str,
        event_type: str,
        impact: float,
        context: Optional[Dict[str, Any]] = None
    ) -> ReputationEvent:
        """
        Record a reputation event (persistent to disk).

        Args:
            node_id: Hardware-anchored identity
            event_type: Type of event (quality_event, verification, etc.)
            impact: Impact on reputation
            context: Additional metadata

        Returns:
            Created event
        """
        # Create event with unique ID
        event_id = hashlib.sha256(
            f"{node_id}{time.time()}{impact}".encode()
        ).hexdigest()[:16]

        event = ReputationEvent(
            event_id=event_id,
            node_id=node_id,
            event_type=event_type,
            impact=impact,
            timestamp=time.time(),
            context=context or {}
        )

        # Store event (append-only)
        self.events.append(event)
        self._save_event(event)

        # Update score
        if node_id not in self.scores:
            self.scores[node_id] = PersistentReputationScore(
                node_id=node_id,
                total_score=0.0,
                event_count=0,
                positive_events=0,
                negative_events=0,
                last_updated=time.time(),
                first_seen=time.time()
            )

        score = self.scores[node_id]
        score.total_score += impact
        score.event_count += 1
        score.last_updated = time.time()

        if impact > 0:
            score.positive_events += 1
        elif impact < 0:
            score.negative_events += 1

        self._save_scores()

        return event

    def get_score(self, node_id: str) -> Optional[PersistentReputationScore]:
        """Get reputation score for a node."""
        return self.scores.get(node_id)

    def get_all_scores(self) -> Dict[str, PersistentReputationScore]:
        """Get all reputation scores."""
        return self.scores.copy()

    def get_events(
        self,
        node_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ReputationEvent]:
        """Get recent events, optionally filtered by node."""
        if node_id:
            events = [e for e in self.events if e.node_id == node_id]
        else:
            events = self.events

        return events[-limit:]

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'total_events': len(self.events),
            'total_nodes': len(self.scores),
            'events_file_size': self.events_file.stat().st_size if self.events_file.exists() else 0,
            'scores_file_size': self.scores_file.stat().st_size if self.scores_file.exists() else 0,
            'oldest_event': self.events[0].timestamp if self.events else None,
            'newest_event': self.events[-1].timestamp if self.events else None
        }


# ============================================================================
# PERSISTENT REPUTATION-AWARE ADAPTIVE SAGE
# ============================================================================

class PersistentReputationAwareAdaptiveSAGE(ReputationAwareAdaptiveSAGE):
    """
    Session 180: SAGE with persistent reputation storage.

    Extends Session 179's ReputationAwareAdaptiveSAGE with:
    - Reputation survives across sessions
    - Event history stored to disk
    - Recovery from storage on restart
    - Long-term trust relationships

    Architecture:
    Session 177 (ATP-adaptive)
        → Session 178 (Federated)
        → Session 179 (Reputation-aware)
        → Session 180 (Persistent reputation) ← YOU ARE HERE
    """

    def __init__(
        self,
        node_id: str,
        hardware_type: str,
        capability_level: int,
        storage_path: Optional[Path] = None,
        **kwargs
    ):
        """
        Initialize with persistent reputation.

        Args:
            node_id: Hardware-anchored identity
            hardware_type: Hardware platform (Thor, Legion, Sprout)
            capability_level: Trust capability level (L5, L4, L3)
            storage_path: Path for reputation storage
            **kwargs: Additional parameters for parent classes
        """
        # Initialize persistent reputation manager
        if storage_path is None:
            storage_path = Path.home() / ".sage" / "reputation"

        self.reputation_manager = PersistentReputationManager(
            storage_path=storage_path,
            node_id=node_id
        )

        # Load or create reputation score for this node
        persistent_score = self.reputation_manager.get_score(node_id)
        if persistent_score is None:
            # New node - create initial reputation
            self.reputation_manager.record_event(
                node_id=node_id,
                event_type="initialization",
                impact=0.0,
                context={"hardware_type": hardware_type, "capability_level": capability_level}
            )
            persistent_score = self.reputation_manager.get_score(node_id)

        # Initialize parent with initial reputation score (Session 179 expects float)
        super().__init__(
            node_id=node_id,
            hardware_type=hardware_type,
            capability_level=capability_level,
            initial_reputation=persistent_score.total_score,
            **kwargs
        )

        # Override reputation with persistent-backed version
        self.reputation = persistent_score.to_simple_reputation()
        self.persistent_score = persistent_score

    def record_quality_event_persistent(self, quality_score: float, context: Optional[Dict[str, Any]] = None):
        """
        Record quality event (persisted to disk).

        New method for direct quality event recording with persistence.
        Bypasses Session 179's depth-based quality calculation.

        Args:
            quality_score: Direct quality/reputation impact score
            context: Additional metadata
        """
        # Record to persistent storage
        event = self.reputation_manager.record_event(
            node_id=self.node_id,
            event_type="quality_event",
            impact=quality_score,
            context=context
        )

        # Update in-memory reputation manually (SimpleReputation doesn't have this method)
        self.reputation.total_score += quality_score
        self.reputation.event_count += 1
        if quality_score > 0:
            self.reputation.positive_events += 1
        else:
            self.reputation.negative_events += 1

        # Sync with persistent score
        self.persistent_score = self.reputation_manager.get_score(self.node_id)

        return event

    def get_reputation_summary(self) -> Dict[str, Any]:
        """Get comprehensive reputation summary including persistence."""
        summary = super().get_reputation_summary()

        # Add persistence metadata
        summary['persistent'] = True
        summary['first_seen'] = self.persistent_score.first_seen
        summary['storage_stats'] = self.reputation_manager.get_storage_stats()
        summary['recent_events'] = [
            {
                'event_id': e.event_id,
                'event_type': e.event_type,
                'impact': e.impact,
                'timestamp': e.timestamp
            }
            for e in self.reputation_manager.get_events(node_id=self.node_id, limit=5)
        ]

        return summary


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def test_cross_session_reputation():
    """
    Test that reputation survives across "sessions" (instantiations).

    Simulates:
    1. Session A: Node builds reputation
    2. Session B: Node restarts, reputation recovered
    3. Session C: Reputation continues to accumulate
    """
    storage_path = Path("/tmp/sage_reputation_test")

    # Clean up any existing test data
    import shutil
    if storage_path.exists():
        shutil.rmtree(storage_path)

    print("\n" + "="*70)
    print("TEST: Cross-Session Reputation Survival")
    print("="*70)

    # ========== SESSION A: Initial reputation building ==========
    print("\n--- Session A: Building reputation ---")

    legion_a = PersistentReputationAwareAdaptiveSAGE(
        node_id="legion",
        hardware_type="RTX_4090",
        capability_level=5,
        storage_path=storage_path,
        enable_federation=False
    )

    print(f"Legion A - Initial reputation: {legion_a.reputation.total_score}")

    # Produce 3 high-quality outputs
    for i in range(3):
        legion_a.record_quality_event_persistent(quality_score=5.0, context={"session": "A", "iteration": i})

    print(f"Legion A - After 3 quality events: {legion_a.reputation.total_score}")
    print(f"Legion A - Reputation level: {legion_a.reputation.reputation_level}")
    print(f"Legion A - Multiplier: {legion_a.reputation.reputation_multiplier}")

    # Get depth at 80 ATP
    legion_a.attention_manager.total_atp = 80.0
    depth_a = legion_a.select_reputation_aware_depth()
    print(f"Legion A - Depth at 80 ATP: {depth_a.name}")

    del legion_a  # Simulate session end

    # ========== SESSION B: Reputation recovery ==========
    print("\n--- Session B: Reputation recovery after restart ---")

    legion_b = PersistentReputationAwareAdaptiveSAGE(
        node_id="legion",
        hardware_type="RTX_4090",
        capability_level=5,
        storage_path=storage_path,
        enable_federation=False
    )

    print(f"Legion B - Recovered reputation: {legion_b.reputation.total_score}")
    print(f"Legion B - Reputation level: {legion_b.reputation.reputation_level}")
    print(f"Legion B - Event count: {legion_b.reputation.event_count}")

    # Verify depth matches (same reputation, same depth)
    legion_b.attention_manager.total_atp = 80.0
    depth_b = legion_b.select_reputation_aware_depth()
    print(f"Legion B - Depth at 80 ATP: {depth_b.name}")

    # Continue building reputation
    for i in range(2):
        legion_b.record_quality_event_persistent(quality_score=5.0, context={"session": "B", "iteration": i})

    print(f"Legion B - After 2 more quality events: {legion_b.reputation.total_score}")
    print(f"Legion B - Total events across sessions: {legion_b.reputation.event_count}")

    del legion_b  # Simulate session end

    # ========== SESSION C: Continued accumulation ==========
    print("\n--- Session C: Continued reputation accumulation ---")

    legion_c = PersistentReputationAwareAdaptiveSAGE(
        node_id="legion",
        hardware_type="RTX_4090",
        capability_level=5,
        storage_path=storage_path,
        enable_federation=False
    )

    print(f"Legion C - Recovered reputation: {legion_c.reputation.total_score}")
    print(f"Legion C - Reputation level: {legion_c.reputation.reputation_level}")
    print(f"Legion C - Total events: {legion_c.reputation.event_count}")
    print(f"Legion C - Multiplier: {legion_c.reputation.reputation_multiplier}")

    # Check storage stats
    stats = legion_c.reputation_manager.get_storage_stats()
    print(f"\nStorage Statistics:")
    print(f"  - Total events: {stats['total_events']}")
    print(f"  - Events file size: {stats['events_file_size']} bytes")
    print(f"  - Scores file size: {stats['scores_file_size']} bytes")

    # Verify reputation is cumulative (5 events × 5.0 = 25.0)
    expected_reputation = 25.0
    actual_reputation = legion_c.reputation.total_score

    if abs(actual_reputation - expected_reputation) < 0.1:
        print(f"\n✅ TEST PASSED: Reputation survived across 3 sessions")
        print(f"   Expected: {expected_reputation}, Got: {actual_reputation}")
        return True
    else:
        print(f"\n❌ TEST FAILED: Reputation not cumulative")
        print(f"   Expected: {expected_reputation}, Got: {actual_reputation}")
        return False


def test_multi_node_persistent_reputation():
    """
    Test that multiple nodes can have independent persistent reputations.
    """
    storage_path = Path("/tmp/sage_reputation_multinode")

    # Clean up
    import shutil
    if storage_path.exists():
        shutil.rmtree(storage_path)

    print("\n" + "="*70)
    print("TEST: Multi-Node Persistent Reputation")
    print("="*70)

    # Create 3 nodes
    nodes = [
        PersistentReputationAwareAdaptiveSAGE(
            node_id="legion",
            hardware_type="RTX_4090",
            capability_level=5,
            storage_path=storage_path,
            enable_federation=False
        ),
        PersistentReputationAwareAdaptiveSAGE(
            node_id="thor",
            hardware_type="Jetson_Thor",
            capability_level=5,
            storage_path=storage_path,
            enable_federation=False
        ),
        PersistentReputationAwareAdaptiveSAGE(
            node_id="sprout",
            hardware_type="Jetson_Orin_Nano",
            capability_level=3,
            storage_path=storage_path,
            enable_federation=False
        )
    ]

    # Build different reputations
    print("\n--- Building different reputations ---")

    # Legion: High quality
    for i in range(5):
        nodes[0].record_quality_event_persistent(quality_score=5.0)
    print(f"Legion reputation: {nodes[0].reputation.total_score} ({nodes[0].reputation.reputation_level})")

    # Thor: Neutral
    # (no events)
    print(f"Thor reputation: {nodes[1].reputation.total_score} ({nodes[1].reputation.reputation_level})")

    # Sprout: Poor quality
    for i in range(3):
        nodes[2].record_quality_event_persistent(quality_score=-3.33)
    print(f"Sprout reputation: {nodes[2].reputation.total_score} ({nodes[2].reputation.reputation_level})")

    # Check depths at same ATP
    print("\n--- Depth selection at 80 ATP ---")
    for node in nodes:
        node.attention_manager.total_atp = 80.0
        depth = node.select_reputation_aware_depth()
        print(f"{node.node_id}: {depth.name} (multiplier: {node.reputation.reputation_multiplier}x)")

    # Verify independence
    reputations = [n.reputation.total_score for n in nodes]
    if len(set(reputations)) == 3:  # All different
        print(f"\n✅ TEST PASSED: Each node has independent reputation")
        return True
    else:
        print(f"\n❌ TEST FAILED: Reputations not independent")
        return False


def test_storage_recovery():
    """
    Test that scores can be recovered from events if corrupted.
    """
    storage_path = Path("/tmp/sage_reputation_recovery")

    # Clean up
    import shutil
    if storage_path.exists():
        shutil.rmtree(storage_path)

    print("\n" + "="*70)
    print("TEST: Storage Recovery from Events")
    print("="*70)

    # Create node and build reputation
    node = PersistentReputationAwareAdaptiveSAGE(
        node_id="test_node",
        hardware_type="Test",
        capability_level=5,
        storage_path=storage_path,
        enable_federation=False
    )

    for i in range(5):
        node.record_quality_event_persistent(quality_score=3.0)

    original_score = node.reputation.total_score
    print(f"Original reputation: {original_score}")

    del node  # Close files

    # Corrupt scores file
    scores_file = storage_path / "test_node_reputation_scores.json"
    with open(scores_file, 'w') as f:
        f.write("CORRUPTED DATA!!!")

    print("Scores file corrupted...")

    # Recover by creating new instance (should rebuild from events)
    recovered_node = PersistentReputationAwareAdaptiveSAGE(
        node_id="test_node",
        hardware_type="Test",
        capability_level=5,
        storage_path=storage_path,
        enable_federation=False
    )

    recovered_score = recovered_node.reputation.total_score
    print(f"Recovered reputation: {recovered_score}")

    if abs(original_score - recovered_score) < 0.1:
        print(f"\n✅ TEST PASSED: Reputation recovered from events")
        return True
    else:
        print(f"\n❌ TEST FAILED: Recovery incorrect")
        print(f"   Expected: {original_score}, Got: {recovered_score}")
        return False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║           Thor Session 180: Persistent Reputation Storage         ║
    ║                                                                   ║
    ║  Integrating Legion's persistent storage with Session 179's      ║
    ║  reputation-aware adaptive depth.                                 ║
    ║                                                                   ║
    ║  Reputation now survives across sessions, accumulates over time,  ║
    ║  and creates long-term trust relationships.                       ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    results = {}

    # Test 1: Cross-session reputation survival
    results['cross_session'] = test_cross_session_reputation()

    # Test 2: Multi-node independent reputation
    results['multi_node'] = test_multi_node_persistent_reputation()

    # Test 3: Storage recovery
    results['recovery'] = test_storage_recovery()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if all_passed:
        print("\n🎉 Session 180 Complete: Persistent reputation operational!")
        print("   - Reputation survives across sessions")
        print("   - Multiple nodes have independent reputations")
        print("   - Storage recovery from events working")
        print("   - Ready for real deployment with long-term trust")
