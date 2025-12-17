#!/usr/bin/env python3
"""
Trust Tensor Sync - SAGE ↔ Web4 Reputation Synchronization

Synchronizes SAGE expert reputation with Web4 trust tensor, enabling:
1. Export: SAGE reputation → Web4 trust observations
2. Import: Web4 trust observations → SAGE reputation
3. Distributed learning: Multiple SAGE instances share expertise
4. Cross-validation: Multiple observers improve trust accuracy

Design Philosophy:
- Bidirectional sync: SAGE ↔ Web4 reputation flows both ways
- Context isolation: Trust is context-specific (code vs math vs creative)
- Confidence weighting: More data → higher confidence
- Observer reputation: Weight observations by observer trust
- Distributed convergence: Trust scores converge across instances

Integration Points:
- ExpertReputationDB: SAGE expert performance tracking
- ExpertIdentityBridge: expert_id ↔ LCT ID mapping
- Web4TrustClient: Trust tensor storage/retrieval (stub for now)
- TrustBasedExpertSelector: Uses synced trust for selection

Created: Session 61 (2025-12-16)
Part of: Web4 ↔ SAGE integration (Session 57 design)
Previous: Session 59 (ExpertIdentityBridge), Session 60 (ATPResourceAllocator)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
import json


# =============================================================================
# TRUST TENSOR DATA STRUCTURES
# =============================================================================

@dataclass
class TrustTensor:
    """
    Web4 trust tensor entry.

    Represents an observation of trust from one entity (observer) about
    another entity (subject) in a specific context.

    This is the fundamental unit of Web4's reputation system, analogous to
    a weighted edge in a trust graph.
    """
    observer_id: str        # LCT ID of observer (e.g., "lct://sage/router")
    subject_id: str         # LCT ID of subject (e.g., "lct://sage_legion/expert/42")
    context: str            # Context identifier (e.g., "code_generation")
    trust_score: float      # Trust value (0-1)
    confidence: float       # Confidence in observation (0-1)
    last_updated: float     # Unix timestamp
    evidence_count: int     # Number of interactions observed
    metadata: Dict = field(default_factory=dict)  # Additional context


@dataclass
class TrustObservation:
    """
    Single trust observation from Web4.

    When importing from Web4, we may receive multiple observations of the
    same expert from different observers. These need to be aggregated.
    """
    observer_id: str
    trust_score: float
    confidence: float
    evidence_count: int
    timestamp: float


# =============================================================================
# WEB4 TRUST CLIENT (STUB)
# =============================================================================

class Web4TrustClient:
    """
    Stub implementation of Web4 trust tensor client.

    In production, this would connect to a distributed trust tensor storage
    (e.g., Redis, distributed DB, blockchain). For now, it stores locally
    to enable testing and development.

    Future implementations:
    - Redis backend for multi-instance sync
    - P2P gossip protocol for federated SAGE
    - Blockchain-based consensus for Byzantine resistance
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize Web4 trust client.

        Args:
            storage_path: Path to JSON storage file (None = in-memory only)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.trust_tensors: Dict[str, List[TrustTensor]] = {}  # subject_id → [observations]

        # Load existing data if available
        if self.storage_path and self.storage_path.exists():
            self._load()

    def update_trust(self, trust_entry: TrustTensor) -> None:
        """
        Update trust tensor with new observation.

        Args:
            trust_entry: Trust observation to record
        """
        subject_id = trust_entry.subject_id

        if subject_id not in self.trust_tensors:
            self.trust_tensors[subject_id] = []

        # Find existing entry from same observer in same context
        existing = None
        for i, entry in enumerate(self.trust_tensors[subject_id]):
            if (entry.observer_id == trust_entry.observer_id and
                entry.context == trust_entry.context):
                existing = i
                break

        if existing is not None:
            # Update existing entry
            self.trust_tensors[subject_id][existing] = trust_entry
        else:
            # Add new entry
            self.trust_tensors[subject_id].append(trust_entry)

        # Persist if storage configured
        if self.storage_path:
            self._save()

    def get_trust_observations(
        self,
        subject_id: str,
        context: Optional[str] = None,
        observer_id: Optional[str] = None
    ) -> List[TrustObservation]:
        """
        Get trust observations for a subject.

        Args:
            subject_id: LCT ID of subject to query
            context: Filter by context (None = all contexts)
            observer_id: Filter by observer (None = all observers)

        Returns:
            List of trust observations
        """
        if subject_id not in self.trust_tensors:
            return []

        # Filter observations
        observations = []
        for tensor in self.trust_tensors[subject_id]:
            # Apply filters
            if context and tensor.context != context:
                continue
            if observer_id and tensor.observer_id != observer_id:
                continue

            # Convert to observation
            obs = TrustObservation(
                observer_id=tensor.observer_id,
                trust_score=tensor.trust_score,
                confidence=tensor.confidence,
                evidence_count=tensor.evidence_count,
                timestamp=tensor.last_updated
            )
            observations.append(obs)

        return observations

    def get_subjects(self, context: Optional[str] = None) -> List[str]:
        """
        Get all subject IDs with trust observations.

        Args:
            context: Filter by context (None = all contexts)

        Returns:
            List of subject LCT IDs
        """
        if context is None:
            return list(self.trust_tensors.keys())

        subjects = set()
        for subject_id, tensors in self.trust_tensors.items():
            for tensor in tensors:
                if tensor.context == context:
                    subjects.add(subject_id)
                    break

        return list(subjects)

    def _save(self) -> None:
        """Save trust tensors to disk."""
        if self.storage_path is None:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            subject_id: [asdict(tensor) for tensor in tensors]
            for subject_id, tensors in self.trust_tensors.items()
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load trust tensors from disk."""
        if self.storage_path is None or not self.storage_path.exists():
            return

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        self.trust_tensors = {
            subject_id: [TrustTensor(**tensor_dict) for tensor_dict in tensors]
            for subject_id, tensors in data.items()
        }


# =============================================================================
# TRUST TENSOR SYNC
# =============================================================================

class TrustTensorSync:
    """
    Synchronizes SAGE expert reputation with Web4 trust tensor.

    Enables bidirectional reputation flow:
    - Export: SAGE expert performance → Web4 trust observations
    - Import: Web4 trust observations → SAGE expert reputation

    Use Cases:
    1. **Single instance**: Export reputation for external visibility
    2. **Multi-instance**: Share expertise across distributed SAGE
    3. **Cross-validation**: Multiple observers improve trust accuracy
    4. **Federation**: Different SAGE deployments learn from each other

    Architecture:
    - TrustTensor: Web4's canonical trust representation
    - ExpertReputationDB: SAGE's internal reputation storage
    - ExpertIdentityBridge: Maps expert_id ↔ LCT ID
    - Web4TrustClient: Trust tensor storage/retrieval

    Design Decisions:
    - Context-specific: Trust varies by task context
    - Confidence-weighted: More evidence → higher confidence
    - Observer-aware: Track who observed what
    - Bidirectional: Both export and import supported
    """

    def __init__(
        self,
        reputation_db,  # ExpertReputationDB
        identity_bridge,  # ExpertIdentityBridge
        web4_trust_client: Optional[Web4TrustClient] = None,
        observer_id: str = "lct://sage/router"
    ):
        """
        Initialize trust tensor synchronization.

        Args:
            reputation_db: SAGE expert reputation database
            identity_bridge: Expert ID ↔ LCT ID mapping
            web4_trust_client: Web4 trust storage (None = create in-memory)
            observer_id: LCT ID of this SAGE instance as observer
        """
        self.reputation_db = reputation_db
        self.identity_bridge = identity_bridge
        self.web4_client = web4_trust_client or Web4TrustClient()
        self.observer_id = observer_id

        # Track sync statistics
        self.exports_count = 0
        self.imports_count = 0
        self.last_sync_time = time.time()

    def export_to_web4(
        self,
        expert_id: int,
        context: Optional[str] = None
    ) -> List[TrustTensor]:
        """
        Export SAGE expert reputation to Web4 trust tensor.

        Args:
            expert_id: Expert to export
            context: Specific context (None = all contexts)

        Returns:
            List of trust tensor entries created
        """
        # Get expert reputation
        rep = self.reputation_db.get_reputation(expert_id, component="sage_moe")
        if rep is None:
            raise ValueError(f"Expert {expert_id} not found in reputation DB")

        # Get or register LCT ID
        lct_id = self.identity_bridge.get_lct(expert_id)
        if lct_id is None:
            lct_id = self.identity_bridge.register_expert(expert_id)

        trust_entries = []

        # Export context-specific trust
        if context:
            # Single context export
            trust_entry = self._create_trust_entry(rep, lct_id, context)
            if trust_entry:
                self.web4_client.update_trust(trust_entry)
                trust_entries.append(trust_entry)
        else:
            # Export all contexts
            for ctx in rep.context_trust.keys():
                trust_entry = self._create_trust_entry(rep, lct_id, ctx)
                if trust_entry:
                    self.web4_client.update_trust(trust_entry)
                    trust_entries.append(trust_entry)

        self.exports_count += len(trust_entries)
        return trust_entries

    def _create_trust_entry(
        self,
        rep,  # ExpertReputation
        lct_id: str,
        context: str
    ) -> Optional[TrustTensor]:
        """Create trust tensor entry from expert reputation."""
        # Get context-specific trust
        trust_score = rep.get_context_trust(context)
        if trust_score is None:
            return None

        # Get context observations count
        observations = rep.context_observations.get(context, 0)

        # Compute confidence (more observations → higher confidence)
        # Sigmoid function: confidence = observations / (observations + 100)
        # At 0 observations: 0% confidence
        # At 100 observations: 50% confidence
        # At 1000 observations: 90.9% confidence
        confidence = observations / (observations + 100.0)

        # Create trust tensor
        trust_entry = TrustTensor(
            observer_id=self.observer_id,
            subject_id=lct_id,
            context=context,
            trust_score=trust_score,
            confidence=confidence,
            last_updated=rep.last_used or time.time(),
            evidence_count=observations,
            metadata={
                'expert_id': rep.expert_id,
                'component': rep.component
            }
        )

        return trust_entry

    def import_from_web4(
        self,
        lct_id: str,
        context: str,
        min_confidence: float = 0.1
    ) -> bool:
        """
        Import Web4 trust observations into SAGE reputation.

        Args:
            lct_id: LCT ID of expert to import
            context: Context to import trust for
            min_confidence: Minimum confidence threshold (filter low-confidence obs)

        Returns:
            True if import successful, False if no data or expert not found
        """
        # Get expert ID
        expert_id = self.identity_bridge.get_expert_id(lct_id)
        if expert_id is None:
            return False

        # Fetch trust observations from Web4
        observations = self.web4_client.get_trust_observations(
            subject_id=lct_id,
            context=context
        )

        if not observations:
            return False

        # Filter by confidence
        observations = [obs for obs in observations if obs.confidence >= min_confidence]

        if not observations:
            return False

        # Get or create expert reputation
        rep = self.reputation_db.get_reputation(expert_id, component="sage_moe")
        if rep is None:
            # Create default reputation if doesn't exist
            # (This happens when importing external observations)
            try:
                from sage.core.expert_reputation import ExpertReputation
            except ImportError:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from core.expert_reputation import ExpertReputation
            rep = ExpertReputation(expert_id=expert_id, component="sage_moe")

        # Aggregate observations (weighted by confidence)
        total_weight = sum(obs.confidence for obs in observations)
        weighted_trust = sum(
            obs.trust_score * obs.confidence for obs in observations
        ) / total_weight

        # Get existing context trust and observations
        existing_trust = rep.get_context_trust(context, default=0.5)
        existing_observations = rep.context_observations.get(context, 0)

        # Blend imported trust with existing trust (weighted average)
        # Give more weight to local observations (they're from this instance)
        local_weight = existing_observations
        remote_weight = sum(obs.evidence_count for obs in observations) * total_weight
        total = local_weight + remote_weight

        if total > 0:
            blended_trust = (
                existing_trust * local_weight + weighted_trust * remote_weight
            ) / total
        else:
            blended_trust = weighted_trust

        # Update trust using the expert's built-in method
        # Set learning_rate to 1.0 to fully replace with blended value
        if context not in rep.context_trust:
            rep.context_trust[context] = blended_trust
            rep.context_observations[context] = int(remote_weight)
        else:
            rep.context_trust[context] = blended_trust
            rep.context_observations[context] = existing_observations + int(remote_weight)

        # Save updated reputation
        self.reputation_db.save(rep)

        self.imports_count += 1
        return True

    def sync_all_experts(
        self,
        context: Optional[str] = None,
        bidirectional: bool = True
    ) -> Dict[str, int]:
        """
        Sync all experts between SAGE and Web4.

        Args:
            context: Specific context (None = all contexts)
            bidirectional: If True, both export and import. If False, export only.

        Returns:
            Statistics: {'exported': N, 'imported': M}
        """
        stats = {'exported': 0, 'imported': 0}

        # Export: SAGE → Web4
        for expert_id in self.identity_bridge.expert_to_lct.keys():
            try:
                entries = self.export_to_web4(expert_id, context)
                stats['exported'] += len(entries)
            except (ValueError, KeyError):
                # Expert not in reputation DB, skip
                continue

        # Import: Web4 → SAGE (if bidirectional)
        if bidirectional:
            # Get all subjects from Web4
            subjects = self.web4_client.get_subjects(context)

            for lct_id in subjects:
                # Skip self-observations (we just exported those)
                if lct_id in self.identity_bridge.lct_to_expert.values():
                    continue

                # Import observations for each context
                if context:
                    contexts = [context]
                else:
                    # Get all contexts for this subject
                    all_obs = self.web4_client.get_trust_observations(lct_id)
                    contexts = list(set(obs.context for obs in all_obs))

                for ctx in contexts:
                    if self.import_from_web4(lct_id, ctx):
                        stats['imported'] += 1

        self.last_sync_time = time.time()
        return stats

    def get_statistics(self) -> Dict:
        """Get synchronization statistics."""
        return {
            'exports_count': self.exports_count,
            'imports_count': self.imports_count,
            'last_sync_time': self.last_sync_time,
            'web4_subjects': len(self.web4_client.get_subjects()),
            'web4_total_observations': sum(
                len(tensors) for tensors in self.web4_client.trust_tensors.values()
            )
        }


# Convenience functions

def create_trust_sync(
    reputation_db,
    identity_bridge,
    storage_path: Optional[Path] = None,
    observer_id: str = "lct://sage/router"
) -> TrustTensorSync:
    """
    Create trust tensor sync with default configuration.

    Args:
        reputation_db: SAGE expert reputation database
        identity_bridge: Expert ID ↔ LCT ID mapping
        storage_path: Path for Web4 trust storage (None = in-memory)
        observer_id: LCT ID of this SAGE instance

    Returns:
        TrustTensorSync instance
    """
    web4_client = Web4TrustClient(storage_path)
    return TrustTensorSync(
        reputation_db=reputation_db,
        identity_bridge=identity_bridge,
        web4_trust_client=web4_client,
        observer_id=observer_id
    )
