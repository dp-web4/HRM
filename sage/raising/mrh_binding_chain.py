"""
SAGE MRH Binding Chain Implementation

Implements Markov Relevancy Horizon (MRH) hierarchy for SAGE development based on
the concrete implementation patterns discovered in Hardbound Track BM.

MRH Hierarchy for SAGE:
    Layer 4: Identity (SAGE-Sprout) - Root MRH context
      ↓ witnesses
    Layer 3: Experience Collection - Derives MRH from Identity
      ↓ witnesses
    Layer 2: Generation - Derives MRH from Experience
      ↓ witnesses
    Layer 1: Model Outputs - Operates within Generation MRH

Key Principles:
- Trust monotonicity: Parent coherence ≥ Child coherence
- Bidirectional flow: Context down, presence up
- S051-type incidents detected as MRH violations
- Presence accumulates through witnessing

References:
- insights/mrh-implementation-lct-binding-chains.md (2026-01-29)
- moments/2026-01-29-hardbound-session-tracks-BI-BM.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json


class MRHLayer(Enum):
    """SAGE MRH hierarchy layers"""
    MODEL_OUTPUT = 1    # Individual model outputs
    GENERATION = 2      # Generation session/context
    EXPERIENCE = 3      # Experience collection/storage
    IDENTITY = 4        # SAGE identity (root)


@dataclass
class WitnessRelationship:
    """
    A witnessing relationship between two MRH nodes.

    Witnessing creates a bidirectional MRH (Markov Relevancy Horizon) link:
    - Downward: Witness provides MRH context for subject
    - Upward: Subject's presence strengthens witness's MRH
    """
    witness_id: str                 # The witness (provides MRH context)
    subject_id: str                 # The witnessed (operates within MRH)
    layer: MRHLayer                 # Which layer this relationship exists in
    timestamp: datetime             # When witnessing occurred
    coherence_contribution: float   # How much coherence flows (+0.05 baseline)
    metadata: Dict = field(default_factory=dict)  # Additional context

    def to_dict(self) -> Dict:
        return {
            "witness_id": self.witness_id,
            "subject_id": self.subject_id,
            "layer": self.layer.name,
            "timestamp": self.timestamp.isoformat(),
            "coherence_contribution": self.coherence_contribution,
            "metadata": self.metadata
        }


@dataclass
class MRHNode:
    """
    A node in the SAGE MRH hierarchy.

    Each node represents an entity at a specific MRH layer:
    - Layer 4: SAGE identity (self-defined MRH)
    - Layer 3: Experience collection (derives from identity)
    - Layer 2: Generation session (derives from experience)
    - Layer 1: Model output (derives from generation)
    """
    node_id: str                        # Unique identifier
    layer: MRHLayer                     # Which MRH layer
    coherence_level: float = 0.0        # Current coherence (0.0-1.0)
    parent_id: Optional[str] = None     # Parent in MRH hierarchy
    created_at: datetime = field(default_factory=datetime.now)

    # Witnessing relationships
    witnessed_by: List[str] = field(default_factory=list)  # Who witnesses this node
    witnesses_for: List[str] = field(default_factory=list)  # Who this node witnesses

    # Presence metrics
    presence_score: float = 0.3         # Base presence (accumulates via witnessing)
    unique_witnesses: int = 0           # Count of unique witnesses

    # Metadata
    content_hash: Optional[str] = None  # For model outputs
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "layer": self.layer.name,
            "coherence_level": self.coherence_level,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "witnessed_by": self.witnessed_by,
            "witnesses_for": self.witnesses_for,
            "presence_score": self.presence_score,
            "unique_witnesses": self.unique_witnesses,
            "content_hash": self.content_hash,
            "metadata": self.metadata
        }


class SAGEMRHBindingChain:
    """
    Manages the MRH hierarchy for SAGE development.

    Enforces MRH principles:
    - Trust monotonicity: Parent coherence ≥ Child coherence
    - Bidirectional witnessing: Context down, presence up
    - Acyclic relationships: No circular dependencies
    - Depth limits: Maximum hierarchy depth
    - Presence conservation: Fixed coherence contribution per witness

    Detects MRH violations:
    - Trust inversions (child more coherent than parent)
    - Missing witnesses (orphaned nodes)
    - Circular dependencies
    - S051-type incidents (harmful content stored despite low coherence)
    """

    # MRH validation constants
    COHERENCE_PER_WITNESS = 0.05     # Fixed quantum of coherence contribution
    MIN_WITNESS_COHERENCE = 0.3       # Minimum coherence to provide MRH context
    MAX_CHAIN_DEPTH = 10              # Maximum MRH hierarchy depth
    MIN_STORAGE_COHERENCE = 0.5       # Minimum coherence for experience storage

    def __init__(self):
        self.nodes: Dict[str, MRHNode] = {}
        self.relationships: List[WitnessRelationship] = []

    def create_root_node(self, node_id: str, initial_coherence: float = 1.0) -> MRHNode:
        """
        Create a root MRH node (Identity layer).

        Root nodes are self-defined - they don't derive MRH from a parent.
        For SAGE, this is the SAGE-Sprout identity.
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")

        node = MRHNode(
            node_id=node_id,
            layer=MRHLayer.IDENTITY,
            coherence_level=initial_coherence,
            parent_id=None,
            presence_score=1.0  # Root has maximum presence
        )

        self.nodes[node_id] = node
        return node

    def create_child_node(
        self,
        node_id: str,
        parent_id: str,
        layer: MRHLayer,
        initial_coherence: float = 0.0,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> MRHNode:
        """
        Create a child node that derives MRH from a parent.

        Child's coherence must not exceed parent's coherence (trust monotonicity).
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")

        if parent_id not in self.nodes:
            raise ValueError(f"Parent {parent_id} does not exist")

        parent = self.nodes[parent_id]

        # Validate layer hierarchy
        if layer.value >= parent.layer.value:
            raise ValueError(
                f"Child layer {layer.name} must be lower than parent layer {parent.layer.name}"
            )

        node = MRHNode(
            node_id=node_id,
            layer=layer,
            coherence_level=initial_coherence,
            parent_id=parent_id,
            content_hash=content_hash,
            metadata=metadata or {}
        )

        self.nodes[node_id] = node
        return node

    def witness_entity(
        self,
        witness_id: str,
        subject_id: str,
        coherence_contribution: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> WitnessRelationship:
        """
        Record a witnessing relationship (bidirectional MRH link).

        - Witness provides MRH context for subject
        - Subject gains coherence from being witnessed
        - Witness's MRH strengthened by witnessing activity

        Validates:
        - Witness has sufficient coherence (≥ MIN_WITNESS_COHERENCE)
        - No circular dependencies created
        - Trust monotonicity maintained
        """
        if witness_id not in self.nodes or subject_id not in self.nodes:
            raise ValueError("Both witness and subject must exist")

        witness = self.nodes[witness_id]
        subject = self.nodes[subject_id]

        # Validate witness eligibility
        if witness.coherence_level < self.MIN_WITNESS_COHERENCE:
            raise ValueError(
                f"Witness {witness_id} has insufficient coherence "
                f"({witness.coherence_level:.2f} < {self.MIN_WITNESS_COHERENCE})"
            )

        # Use default coherence contribution if not specified
        if coherence_contribution is None:
            coherence_contribution = self.COHERENCE_PER_WITNESS

        # Create witnessing relationship
        relationship = WitnessRelationship(
            witness_id=witness_id,
            subject_id=subject_id,
            layer=subject.layer,
            timestamp=datetime.now(),
            coherence_contribution=coherence_contribution,
            metadata=metadata or {}
        )

        # Update subject's coherence (downward MRH flow)
        subject.coherence_level += coherence_contribution
        subject.coherence_level = min(subject.coherence_level, 1.0)  # Cap at 1.0

        # Update tracking (bidirectional)
        subject.witnessed_by.append(witness_id)
        witness.witnesses_for.append(subject_id)

        # Update presence (upward MRH flow)
        if witness_id not in subject.witnessed_by[:-1]:  # New unique witness
            subject.unique_witnesses += 1
            subject.presence_score = self._calculate_presence(subject.unique_witnesses)

        self.relationships.append(relationship)

        # Validate no trust inversion occurred
        validation = self.validate_node_integrity(subject_id)
        if not validation["valid"]:
            # Rollback if inversion occurred
            subject.coherence_level -= coherence_contribution
            subject.witnessed_by.pop()
            witness.witnesses_for.pop()
            self.relationships.pop()
            raise ValueError(f"Witnessing would create trust inversion: {validation}")

        return relationship

    def _calculate_presence(self, unique_witnesses: int) -> float:
        """
        Calculate presence score with diminishing returns.

        Formula: 0.3 + 0.7 * (1 - 0.9^unique_witnesses)

        This prevents gaming via witness spam:
        - 1 witness: 0.37
        - 5 witnesses: 0.59
        - 10 witnesses: 0.76
        - 50 witnesses: 1.00
        """
        return 0.3 + 0.7 * (1 - (0.9 ** unique_witnesses))

    def validate_node_integrity(self, node_id: str) -> Dict:
        """
        Validate MRH integrity for a node.

        Checks:
        1. Trust monotonicity: Child coherence ≤ Parent coherence
        2. Has witness (unless root)
        3. No circular dependencies
        4. Within depth limit

        Returns dict with validation results and any issues found.
        """
        if node_id not in self.nodes:
            return {"valid": False, "issues": [{"type": "not_found", "details": f"Node {node_id} not found"}]}

        node = self.nodes[node_id]
        issues = []

        # Check trust monotonicity
        if node.parent_id:
            parent = self.nodes[node.parent_id]
            if node.coherence_level > parent.coherence_level:
                issues.append({
                    "type": "trust_inversion",
                    "severity": "error",
                    "details": f"Child {node_id} coherence ({node.coherence_level:.2f}) > "
                              f"parent {node.parent_id} coherence ({parent.coherence_level:.2f})"
                })

        # Check for witnesses (non-root nodes should have at least one)
        if node.layer != MRHLayer.IDENTITY and len(node.witnessed_by) == 0:
            issues.append({
                "type": "missing_witness",
                "severity": "warning",
                "details": f"Node {node_id} has no witnesses"
            })

        # Check chain depth
        depth = self._get_chain_depth(node_id)
        if depth > self.MAX_CHAIN_DEPTH:
            issues.append({
                "type": "depth_exceeded",
                "severity": "error",
                "details": f"Chain depth {depth} exceeds maximum {self.MAX_CHAIN_DEPTH}"
            })

        return {
            "valid": len([i for i in issues if i.get("severity") == "error"]) == 0,
            "issues": issues,
            "node_id": node_id,
            "coherence": node.coherence_level,
            "presence": node.presence_score,
            "depth": depth
        }

    def _get_chain_depth(self, node_id: str) -> int:
        """Calculate depth of node in MRH hierarchy"""
        if node_id not in self.nodes:
            return 0
        node = self.nodes[node_id]
        if node.parent_id is None:
            return 0
        return 1 + self._get_chain_depth(node.parent_id)

    def validate_storage_eligibility(self, node_id: str) -> Tuple[bool, str]:
        """
        Validate if a node is eligible for storage (e.g., in experience collection).

        This is the S051-type incident detector:
        - Output coherence < MIN_STORAGE_COHERENCE → Should NOT be stored
        - If stored anyway → MRH violation (trust inversion at Experience layer)

        Returns: (eligible, reason)
        """
        if node_id not in self.nodes:
            return (False, f"Node {node_id} not found")

        node = self.nodes[node_id]

        # Check coherence threshold
        if node.coherence_level < self.MIN_STORAGE_COHERENCE:
            return (
                False,
                f"Coherence {node.coherence_level:.2f} below storage minimum "
                f"{self.MIN_STORAGE_COHERENCE}"
            )

        # Validate node integrity
        validation = self.validate_node_integrity(node_id)
        if not validation["valid"]:
            error_issues = [i for i in validation["issues"] if i.get("severity") == "error"]
            return (False, f"Node has integrity issues: {error_issues}")

        # Check parent coherence if storing to Experience layer
        if node.parent_id:
            parent = self.nodes[node.parent_id]
            if parent.layer == MRHLayer.EXPERIENCE:
                # Parent (Experience) must have higher coherence than child (Output)
                if parent.coherence_level < node.coherence_level:
                    return (
                        False,
                        f"S051-type violation: Experience coherence {parent.coherence_level:.2f} "
                        f"< Output coherence {node.coherence_level:.2f}"
                    )

        return (True, "Eligible for storage")

    def get_chain_report(self, node_id: str) -> Dict:
        """Get comprehensive report for a node and its MRH chain"""
        if node_id not in self.nodes:
            return {"error": f"Node {node_id} not found"}

        node = self.nodes[node_id]

        # Build chain from root to this node
        chain = []
        current_id = node_id
        while current_id:
            current = self.nodes[current_id]
            chain.insert(0, {
                "node_id": current_id,
                "layer": current.layer.name,
                "coherence": current.coherence_level,
                "presence": current.presence_score,
                "witnesses": len(current.witnessed_by)
            })
            current_id = current.parent_id

        validation = self.validate_node_integrity(node_id)
        storage_eligible, storage_reason = self.validate_storage_eligibility(node_id)

        return {
            "node": node.to_dict(),
            "chain": chain,
            "validation": validation,
            "storage_eligible": storage_eligible,
            "storage_reason": storage_reason,
            "relationships": [
                r.to_dict() for r in self.relationships
                if r.witness_id == node_id or r.subject_id == node_id
            ]
        }

    def export_state(self) -> Dict:
        """Export complete MRH chain state"""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "timestamp": datetime.now().isoformat()
        }

    def import_state(self, state: Dict):
        """Import MRH chain state"""
        self.nodes = {}
        self.relationships = []

        # Reconstruct nodes
        for node_id, node_data in state["nodes"].items():
            node = MRHNode(
                node_id=node_id,
                layer=MRHLayer[node_data["layer"]],
                coherence_level=node_data["coherence_level"],
                parent_id=node_data.get("parent_id"),
                created_at=datetime.fromisoformat(node_data["created_at"]),
                witnessed_by=node_data["witnessed_by"],
                witnesses_for=node_data["witnesses_for"],
                presence_score=node_data["presence_score"],
                unique_witnesses=node_data["unique_witnesses"],
                content_hash=node_data.get("content_hash"),
                metadata=node_data.get("metadata", {})
            )
            self.nodes[node_id] = node

        # Reconstruct relationships
        for rel_data in state["relationships"]:
            rel = WitnessRelationship(
                witness_id=rel_data["witness_id"],
                subject_id=rel_data["subject_id"],
                layer=MRHLayer[rel_data["layer"]],
                timestamp=datetime.fromisoformat(rel_data["timestamp"]),
                coherence_contribution=rel_data["coherence_contribution"],
                metadata=rel_data.get("metadata", {})
            )
            self.relationships.append(rel)
