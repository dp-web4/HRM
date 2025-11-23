"""
Blockchain Witness Manager - SAGE Event Witnessing

Provides cryptographic witnessing for all SAGE consciousness events:
- High-salience observations (discoveries)
- Learning sessions (episodes)
- Pattern discoveries
- Skill creations
- Skill applications

Architecture:
    SAGE Events → Witness Manager → Merkle Batching → Blockchain

Features:
    1. Individual witnessing for critical events
    2. Batch witnessing with Merkle tree for efficiency
    3. Witness verification and integrity checking
    4. Cross-machine witness validation
    5. Automatic flush on session end

Author: Thor (SAGE Development Platform)
Date: 2025-11-22
Status: Phase 3 Implementation
"""

import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

# Add memory repo to path for blockchain tools
_memory_root = Path(__file__).parent.parent.parent.parent / "memory"
if str(_memory_root) not in sys.path:
    sys.path.insert(0, str(_memory_root))

try:
    # Import blockchain witnessing tools
    from blockchain.scripts.witness import (
        witness_discovery,
        witness_episode,
        witness_skill,
        witness_contribution,
        Contribution
    )
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    print("[WitnessManager] Warning: Blockchain tools not available. Running in fallback mode.")


@dataclass
class WitnessEvent:
    """
    A SAGE event awaiting witnessing.

    Represents a single event (observation, pattern, skill, etc.) that
    will be witnessed on the blockchain, either individually or in a batch.
    """
    event_type: str  # 'discovery', 'episode', 'skill', 'pattern', 'application'
    entity: str      # Machine creating the event
    timestamp: datetime
    data: Dict[str, Any]
    quality_score: float
    hash: Optional[str] = None

    def __post_init__(self):
        """Calculate hash after initialization"""
        if self.hash is None:
            self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of event for Merkle tree inclusion.

        Returns:
            Hex string of hash
        """
        # Create canonical string representation
        data_str = f"{self.event_type}|{self.entity}|{self.timestamp.isoformat()}"

        # Add sorted data keys for deterministic hashing
        for key in sorted(self.data.keys()):
            data_str += f"|{key}:{self.data[key]}"

        data_str += f"|quality:{self.quality_score}"

        # Hash and return hex
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def to_contribution(self) -> 'Contribution':
        """
        Convert to blockchain Contribution for witnessing.

        Returns:
            Contribution object ready for blockchain
        """
        if not BLOCKCHAIN_AVAILABLE:
            # Return mock contribution in fallback mode
            return None

        return Contribution(
            type=self.event_type,
            entity=self.entity,
            data=self.data,
            quality_score=self.quality_score,
            timestamp=self.timestamp.isoformat().replace('+00:00', 'Z')
        )


@dataclass
class MerkleBatch:
    """
    Batch of events for Merkle tree witnessing.

    Groups multiple events together, creates Merkle tree, and witnesses
    the root hash on blockchain. More efficient than individual witnessing.
    """
    batch_id: str
    events: List[WitnessEvent] = field(default_factory=list)
    merkle_root: Optional[str] = None
    block_hash: Optional[str] = None
    witnessed_at: Optional[datetime] = None

    def add_event(self, event: WitnessEvent):
        """Add event to batch"""
        self.events.append(event)

    def build_merkle_tree(self) -> str:
        """
        Build Merkle tree from events and return root hash.

        Returns:
            Merkle root hash (hex string)
        """
        if not self.events:
            return ""

        # Get leaf hashes
        leaves = [event.hash for event in self.events]

        # Build tree bottom-up
        current_level = leaves

        while len(current_level) > 1:
            next_level = []

            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]

                # If odd number, duplicate last node
                if i + 1 >= len(current_level):
                    right = left
                else:
                    right = current_level[i + 1]

                # Hash pair
                combined = left + right
                parent_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
                next_level.append(parent_hash)

            current_level = next_level

        # Root is the only remaining hash
        self.merkle_root = current_level[0]
        return self.merkle_root

    def witness(self, entity: str) -> Optional[str]:
        """
        Witness this batch on blockchain.

        Creates Merkle root and witnesses it as a validation contribution
        (batches are stored as validation type with Merkle root).

        Args:
            entity: Entity witnessing the batch

        Returns:
            Block hash if successful, None otherwise
        """
        if not self.events:
            return None

        # Build Merkle tree
        self.build_merkle_tree()

        # In test mode or when blockchain unavailable
        if not BLOCKCHAIN_AVAILABLE:
            print(f"[WitnessManager] Would witness batch {self.batch_id} ({len(self.events)} events)")
            print(f"[WitnessManager]    Merkle root: {self.merkle_root[:16]}...")
            return None

        # Create batch contribution as "validation" type with Merkle root
        contrib = Contribution(
            type="validation",  # Use validation type for batch witnessing
            entity=entity,
            data={
                "batch_id": self.batch_id,
                "event_count": len(self.events),
                "merkle_root": self.merkle_root,
                "event_types": [e.event_type for e in self.events],
                "average_quality": sum(e.quality_score for e in self.events) / len(self.events)
            },
            quality_score=sum(e.quality_score for e in self.events) / len(self.events),
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        )

        # Witness on blockchain
        try:
            block_hash = witness_contribution(entity, contrib)
            self.block_hash = block_hash
            self.witnessed_at = datetime.now(timezone.utc)

            print(f"[WitnessManager] ✅ Batch {self.batch_id} witnessed")
            print(f"[WitnessManager]    Events: {len(self.events)}")
            print(f"[WitnessManager]    Merkle root: {self.merkle_root[:16]}...")
            print(f"[WitnessManager]    Block hash: {block_hash[:16]}...")

            return block_hash

        except Exception as e:
            print(f"[WitnessManager] Warning: Failed to witness batch: {e}")
            return None


class WitnessManager:
    """
    Manages blockchain witnessing for SAGE consciousness events.

    Provides:
    - Individual event witnessing (critical events)
    - Batch event witnessing with Merkle trees (efficiency)
    - Witness verification
    - Cross-machine witness validation

    Usage:
        manager = WitnessManager(machine='thor', project='sage')

        # Witness individual discovery
        manager.witness_discovery(discovery_id, title, domain, quality)

        # Add to batch for later witnessing
        manager.add_to_batch(event)

        # Flush batch at session end
        manager.flush_batch()
    """

    def __init__(
        self,
        machine: str = 'thor',
        project: str = 'sage',
        batch_size: int = 10,
        enable_batching: bool = True,
        enable_witnessing: bool = True
    ):
        """
        Initialize witness manager.

        Args:
            machine: Hardware entity
            project: Project context
            batch_size: Number of events before auto-flush
            enable_batching: Use batch witnessing
            enable_witnessing: Actually witness on blockchain
        """
        self.machine = machine
        self.project = project
        self.batch_size = batch_size
        self.enable_batching = enable_batching
        self.enable_witnessing = enable_witnessing

        # Current batch
        self.current_batch: Optional[MerkleBatch] = None
        if enable_batching:
            self.current_batch = self._create_new_batch()

        # Tracking
        self.witnessed_events = []
        self.witnessed_batches = []

        # Statistics
        self.stats = {
            'individual_witnesses': 0,
            'batch_witnesses': 0,
            'total_events_witnessed': 0,
            'failed_witnesses': 0
        }

        if BLOCKCHAIN_AVAILABLE and enable_witnessing:
            print(f"[WitnessManager] Initialized for {machine}/{project}")
            print(f"[WitnessManager] Batching: {'enabled' if enable_batching else 'disabled'}")
            print(f"[WitnessManager] Batch size: {batch_size}")
        else:
            print(f"[WitnessManager] Running in fallback mode")

    def _create_new_batch(self) -> MerkleBatch:
        """Create new batch with unique ID"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        batch_id = f"batch-{self.machine}-{timestamp}"
        return MerkleBatch(batch_id=batch_id)

    def witness_discovery(
        self,
        knowledge_id: str,
        title: str,
        domain: str,
        quality: float,
        tags: List[str] = None
    ) -> Optional[str]:
        """
        Witness a discovery on blockchain.

        Args:
            knowledge_id: Epistemic DB discovery ID
            title: Discovery title
            domain: Knowledge domain
            quality: Quality score (0-1)
            tags: Optional tags

        Returns:
            Block hash if witnessed, None otherwise
        """
        if not self.enable_witnessing:
            return None

        # Create event
        event = WitnessEvent(
            event_type='discovery',
            entity=self.machine,
            timestamp=datetime.now(timezone.utc),
            data={
                'knowledge_id': knowledge_id,
                'title': title,
                'domain': domain,
                'tags': tags or []
            },
            quality_score=quality
        )

        # Decide: individual or batch?
        if self.enable_batching and quality < 0.9:  # Batch medium-quality
            self.add_to_batch(event)
            return None
        else:  # Individual witnessing for high-quality
            return self._witness_individual(event)

    def witness_episode(
        self,
        episode_id: str,
        quality: float,
        discoveries: int = 0,
        failures: int = 0,
        shifts: int = 0
    ) -> Optional[str]:
        """
        Witness a learning episode on blockchain.

        Args:
            episode_id: Epistemic DB episode ID
            quality: Session quality score (0-1)
            discoveries: Number of discoveries
            failures: Number of failures
            shifts: Number of shifts

        Returns:
            Block hash if witnessed, None otherwise
        """
        if not self.enable_witnessing:
            return None

        event = WitnessEvent(
            event_type='episode',
            entity=self.machine,
            timestamp=datetime.now(timezone.utc),
            data={
                'episode_id': episode_id,
                'project': self.project,
                'discoveries': discoveries,
                'failures': failures,
                'shifts': shifts
            },
            quality_score=quality
        )

        # Episodes are always individually witnessed (important)
        return self._witness_individual(event)

    def witness_skill(
        self,
        skill_id: str,
        skill_name: str,
        category: str,
        quality: float,
        success_rate: float
    ) -> Optional[str]:
        """
        Witness a skill creation on blockchain.

        Args:
            skill_id: Epistemic DB skill ID
            skill_name: Skill name
            category: Skill category
            quality: Quality score (0-1)
            success_rate: Success rate (0-1)

        Returns:
            Block hash if witnessed, None otherwise
        """
        if not self.enable_witnessing:
            return None

        event = WitnessEvent(
            event_type='skill',
            entity=self.machine,
            timestamp=datetime.now(timezone.utc),
            data={
                'skill_id': skill_id,
                'skill_name': skill_name,
                'category': category,
                'success_rate': success_rate
            },
            quality_score=quality
        )

        # Skills are always individually witnessed (important)
        return self._witness_individual(event)

    def witness_pattern(
        self,
        pattern_id: str,
        strategy: str,
        success_count: int,
        quality: float
    ) -> Optional[str]:
        """
        Witness a pattern discovery (batch only).

        Patterns are batched because they occur frequently.
        Only witnessed individually if they become skills.

        Args:
            pattern_id: Pattern signature hash
            strategy: Pattern strategy description
            success_count: Times pattern succeeded
            quality: Average quality score

        Returns:
            None (patterns are batched)
        """
        if not self.enable_witnessing or not self.enable_batching:
            return None

        event = WitnessEvent(
            event_type='pattern',
            entity=self.machine,
            timestamp=datetime.now(timezone.utc),
            data={
                'pattern_id': pattern_id,
                'strategy': strategy,
                'success_count': success_count
            },
            quality_score=quality
        )

        self.add_to_batch(event)
        return None

    def witness_skill_application(
        self,
        skill_id: str,
        situation: str,
        success: bool,
        quality: float
    ) -> Optional[str]:
        """
        Witness a skill application (batch only).

        Applications are batched because they occur frequently.

        Args:
            skill_id: Applied skill ID
            situation: Situation description
            success: Whether application succeeded
            quality: Result quality score

        Returns:
            None (applications are batched)
        """
        if not self.enable_witnessing or not self.enable_batching:
            return None

        event = WitnessEvent(
            event_type='application',
            entity=self.machine,
            timestamp=datetime.now(timezone.utc),
            data={
                'skill_id': skill_id,
                'situation': situation,
                'success': success
            },
            quality_score=quality
        )

        self.add_to_batch(event)
        return None

    def add_to_batch(self, event: WitnessEvent):
        """
        Add event to current batch.

        Auto-flushes if batch reaches size limit.
        """
        if not self.enable_batching or self.current_batch is None:
            return

        self.current_batch.add_event(event)

        # Auto-flush if batch full
        if len(self.current_batch.events) >= self.batch_size:
            self.flush_batch()

    def flush_batch(self) -> Optional[str]:
        """
        Flush current batch to blockchain.

        Witnesses all batched events with single Merkle root.
        Creates new batch after flushing.

        Returns:
            Block hash if successful, None otherwise
        """
        if not self.enable_batching or self.current_batch is None:
            return None

        if not self.current_batch.events:
            return None  # Nothing to flush

        # Witness batch
        block_hash = self.current_batch.witness(self.machine)

        if block_hash:
            self.witnessed_batches.append(self.current_batch)
            self.stats['batch_witnesses'] += 1
            self.stats['total_events_witnessed'] += len(self.current_batch.events)
        else:
            self.stats['failed_witnesses'] += 1

        # Create new batch
        self.current_batch = self._create_new_batch()

        return block_hash

    def _witness_individual(self, event: WitnessEvent) -> Optional[str]:
        """
        Witness single event on blockchain.

        Used for critical events (episodes, skills, high-quality discoveries).

        Returns:
            Block hash if successful, None otherwise
        """
        if not BLOCKCHAIN_AVAILABLE:
            print(f"[WitnessManager] Would witness {event.event_type}: {event.data}")
            return None

        try:
            # Route to appropriate witness function
            if event.event_type == 'discovery':
                block_hash = witness_discovery(
                    entity=event.entity,
                    knowledge_id=event.data['knowledge_id'],
                    title=event.data['title'],
                    domain=event.data['domain'],
                    quality=event.quality_score,
                    tags=",".join(event.data.get('tags', []))
                )

            elif event.event_type == 'episode':
                block_hash = witness_episode(
                    entity=event.entity,
                    episode_id=event.data['episode_id'],
                    project=event.data['project'],
                    quality=event.quality_score,
                    discoveries=event.data.get('discoveries', 0),
                    failures=event.data.get('failures', 0),
                    shifts=event.data.get('shifts', 0)
                )

            elif event.event_type == 'skill':
                block_hash = witness_skill(
                    entity=event.entity,
                    skill_id=event.data['skill_id'],
                    skill_name=event.data['skill_name'],
                    category=event.data['category'],
                    quality=event.quality_score,
                    success_rate=event.data['success_rate']
                )

            else:
                # Generic witnessing via contribution
                block_hash = witness_contribution(event.entity, event.to_contribution())

            if block_hash:
                self.witnessed_events.append(event)
                self.stats['individual_witnesses'] += 1
                self.stats['total_events_witnessed'] += 1

                print(f"[WitnessManager] ✅ {event.event_type.capitalize()} witnessed")
                print(f"[WitnessManager]    Block: {block_hash[:16]}...")

            return block_hash

        except Exception as e:
            print(f"[WitnessManager] Warning: Failed to witness {event.event_type}: {e}")
            self.stats['failed_witnesses'] += 1
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get witnessing statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'current_batch_size': len(self.current_batch.events) if self.current_batch else 0,
            'blockchain_available': BLOCKCHAIN_AVAILABLE,
            'batching_enabled': self.enable_batching,
            'witnessing_enabled': self.enable_witnessing
        }

    def verify_event(self, event_hash: str, block_hash: str) -> bool:
        """
        Verify an event was witnessed in a specific block.

        Args:
            event_hash: Hash of the event
            block_hash: Block hash to check

        Returns:
            True if event is in block, False otherwise
        """
        # TODO: Implement verification by reading blockchain
        # This requires querying the blockchain for the block
        # and checking if the event hash is in the Merkle tree
        print(f"[WitnessManager] Verification not yet implemented")
        return False


def create_witness_manager(
    machine: str = 'thor',
    project: str = 'sage',
    **kwargs
) -> WitnessManager:
    """
    Create witness manager with default configuration.

    Args:
        machine: Hardware entity
        project: Project context
        **kwargs: Additional configuration

    Returns:
        Configured WitnessManager instance
    """
    return WitnessManager(
        machine=machine,
        project=project,
        **kwargs
    )
