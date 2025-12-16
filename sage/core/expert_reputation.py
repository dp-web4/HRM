#!/usr/bin/env python3
"""
Expert Reputation System - Web4 Trust Pattern Applied to SAGE
============================================================

Applies Web4's contextual trust framework to SAGE expert management.

**Core Insight**: Expert selection IS a trust problem - which expert do we trust
for this input context?

**Web4 Patterns Applied**:
- Trust emerges from interaction history, not assigned scores
- Context matters: Expert trust varies by input domain
- Reputation evolves: New evidence updates trust estimates
- Relationships tracked: Which experts collaborate effectively

**Purpose**: Enable smart expert caching and substitution for edge deployment.

**Architecture**:
- ExpertReputation: Per-expert trust and performance tracking
- ExpertReputationDB: Persistent storage (SQLite)
- TrustBasedExpertSelector: Trust-augmented routing decisions
- ExpertReputationLearner: Evidence-based trust updates

**Author**: Claude (Legion Autonomous Web4 Research Session 55)
**Date**: 2025-12-15
**Cross-Project**: Web4 MRH → SAGE neural architecture
"""

import sqlite3
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Counter
from collections import Counter as CollectionsCounter


@dataclass
class ExpertReputation:
    """
    Expert reputation tracking (Web4 trust pattern applied).

    Tracks:
    - Activation history (Web4: interaction frequency)
    - Performance metrics (Web4: outcome quality)
    - Contextual trust (MRH: context-dependent reliability)
    - Relational data (Web4: entity collaboration patterns)

    All trust scores emerge from observation, not assignment.
    """

    # Identity
    expert_id: int                  # Expert identifier (0-127 for thinker)
    component: str = "thinker"      # "thinker" or "talker"

    # Activation History (Web4: interaction history)
    activation_count: int = 0                   # Total activations
    contexts_seen: Dict[str, int] = field(default_factory=dict)  # {context: count}
    first_seen: Optional[float] = None          # Timestamp of first activation
    last_used: Optional[float] = None           # Timestamp of most recent

    # Performance Metrics (Web4: outcome quality)
    convergence_rate: float = 0.5        # How quickly reduces loss (0-1)
    stability: float = 0.5               # Consistency across similar inputs (0-1)
    efficiency: float = 0.5              # Quality per computation cost (0-1)
    average_confidence: float = 0.5      # Router's confidence when selecting (0-1)

    # Contextual Trust (MRH: context-dependent reliability)
    # Trust = P(expert performs well | context)
    # Learned from observation, not assigned
    context_trust: Dict[str, float] = field(default_factory=dict)  # {context: trust}
    context_observations: Dict[str, int] = field(default_factory=dict)  # Sample counts

    # Relational Data (Web4: entity collaboration patterns)
    co_activated_with: Dict[int, int] = field(default_factory=dict)  # {expert_id: count}
    successful_pairs: Dict[int, float] = field(default_factory=dict)  # {expert_id: quality}
    substituted_for: List[Tuple[int, float, str]] = field(default_factory=list)
    # [(requested_expert, quality_delta, context)]

    # Similarity Metadata (for substitution decisions)
    semantic_cluster: int = -1           # Cluster ID from router analysis
    modality_specialization: str = "unknown"  # "text" | "audio" | "vision" | "fusion"

    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage."""
        data = asdict(self)
        # Convert collections.Counter to dict if needed
        if hasattr(data['co_activated_with'], 'items'):
            data['co_activated_with'] = dict(data['co_activated_with'])
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpertReputation':
        """Deserialize from dictionary."""
        return cls(**data)

    def get_context_trust(self, context: str, default: float = 0.5) -> float:
        """
        Get trust score for specific context.

        Args:
            context: Context identifier
            default: Default trust for unknown context

        Returns:
            Trust score (0-1)
        """
        return self.context_trust.get(context, default)

    def update_context_trust(self, context: str, evidence_quality: float, learning_rate: float = 0.1):
        """
        Update trust for context using Bayesian-style evidence accumulation.

        Web4 Pattern: Trust evolves through evidence, not assignment.

        Args:
            context: Context where expert was activated
            evidence_quality: Observed outcome quality (0-1)
            learning_rate: How quickly to update (0-1)
        """
        prior_trust = self.context_trust.get(context, 0.5)
        observations = self.context_observations.get(context, 0)

        # Bayesian update (simplified EMA)
        posterior_trust = (1 - learning_rate) * prior_trust + learning_rate * evidence_quality

        self.context_trust[context] = posterior_trust
        self.context_observations[context] = observations + 1

    def record_activation(self, context: str, performance: Dict[str, float]):
        """
        Record expert activation and update reputation.

        Args:
            context: Input context
            performance: {
                'convergence': 0-1,
                'stability': 0-1,
                'confidence': 0-1,
                'quality': 0-1 (overall outcome)
            }
        """
        # Update activation history
        self.activation_count += 1
        self.contexts_seen[context] = self.contexts_seen.get(context, 0) + 1

        if self.first_seen is None:
            self.first_seen = time.time()
        self.last_used = time.time()

        # Update performance metrics (EMA)
        α = 0.1  # Learning rate
        self.convergence_rate = (1-α) * self.convergence_rate + α * performance.get('convergence', 0.5)
        self.stability = (1-α) * self.stability + α * performance.get('stability', 0.5)
        self.average_confidence = (1-α) * self.average_confidence + α * performance.get('confidence', 0.5)

        # Update contextual trust
        self.update_context_trust(context, performance.get('quality', 0.5), learning_rate=α)

    def record_co_activation(self, other_expert_id: int, combined_quality: float):
        """
        Record co-activation with another expert (relational trust).

        Web4 Pattern: Track entity collaboration patterns.

        Args:
            other_expert_id: Expert that was co-activated
            combined_quality: Quality of combined output (0-1)
        """
        # Track co-activation frequency
        self.co_activated_with[other_expert_id] = \
            self.co_activated_with.get(other_expert_id, 0) + 1

        # Track pair performance (EMA)
        α = 0.1
        if other_expert_id not in self.successful_pairs:
            self.successful_pairs[other_expert_id] = combined_quality
        else:
            self.successful_pairs[other_expert_id] = \
                (1-α) * self.successful_pairs[other_expert_id] + α * combined_quality

    def record_substitution(self, requested_expert_id: int, quality_delta: float, context: str):
        """
        Record that this expert substituted for another (delegation pattern).

        Web4 Pattern: Track delegation effectiveness.

        Args:
            requested_expert_id: Expert that was originally requested
            quality_delta: Outcome - baseline (positive = good substitution)
            context: Input context
        """
        self.substituted_for.append((requested_expert_id, quality_delta, context))

        # If substitution was successful, boost trust in this context
        if quality_delta > 0:
            current_trust = self.context_trust.get(context, 0.5)
            boost = min(0.1, quality_delta)  # Cap boost at 0.1
            self.context_trust[context] = min(1.0, current_trust + boost)


class ExpertReputationDB:
    """
    SQLite database for expert reputation persistence.

    Survives restarts, accumulates across sessions.
    Designed for future federation (cross-machine expert knowledge).

    Tables:
    - expert_reputation: Core reputation data
    - context_trust: Per-context trust scores
    - co_activation: Expert collaboration patterns
    - substitutions: Delegation effectiveness tracking
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize reputation database.

        Args:
            db_path: Path to SQLite database (default: ~/.sage/expert_reputation.db)
        """
        if db_path is None:
            db_path = Path.home() / ".sage" / "expert_reputation.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._create_tables()

    def _create_tables(self):
        """Create database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS expert_reputation (
                expert_id INTEGER PRIMARY KEY,
                component TEXT,
                activation_count INTEGER,
                first_seen REAL,
                last_used REAL,
                convergence_rate REAL,
                stability REAL,
                efficiency REAL,
                average_confidence REAL,
                semantic_cluster INTEGER,
                modality_specialization TEXT,
                contexts_seen TEXT  -- JSON: {context: count}
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS context_trust (
                expert_id INTEGER,
                context TEXT,
                trust REAL,
                observation_count INTEGER,
                PRIMARY KEY (expert_id, context),
                FOREIGN KEY (expert_id) REFERENCES expert_reputation(expert_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS co_activation (
                expert_a INTEGER,
                expert_b INTEGER,
                count INTEGER,
                combined_quality REAL,
                PRIMARY KEY (expert_a, expert_b),
                FOREIGN KEY (expert_a) REFERENCES expert_reputation(expert_id),
                FOREIGN KEY (expert_b) REFERENCES expert_reputation(expert_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS substitutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                substitute_id INTEGER,
                requested_id INTEGER,
                context TEXT,
                quality_delta REAL,
                timestamp REAL,
                FOREIGN KEY (substitute_id) REFERENCES expert_reputation(expert_id),
                FOREIGN KEY (requested_id) REFERENCES expert_reputation(expert_id)
            )
        """)

        self.conn.commit()

    def get_reputation(self, expert_id: int, component: str = "thinker") -> Optional[ExpertReputation]:
        """
        Load expert reputation from database.

        Args:
            expert_id: Expert identifier
            component: "thinker" or "talker"

        Returns:
            ExpertReputation if exists, None otherwise
        """
        cursor = self.conn.execute(
            "SELECT * FROM expert_reputation WHERE expert_id = ?",
            (expert_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # Reconstruct ExpertReputation
        rep = ExpertReputation(
            expert_id=row['expert_id'],
            component=row['component'],
            activation_count=row['activation_count'],
            first_seen=row['first_seen'],
            last_used=row['last_used'],
            convergence_rate=row['convergence_rate'],
            stability=row['stability'],
            efficiency=row['efficiency'],
            average_confidence=row['average_confidence'],
            semantic_cluster=row['semantic_cluster'],
            modality_specialization=row['modality_specialization'],
            contexts_seen=json.loads(row['contexts_seen']) if row['contexts_seen'] else {}
        )

        # Load context trust
        cursor = self.conn.execute(
            "SELECT context, trust, observation_count FROM context_trust WHERE expert_id = ?",
            (expert_id,)
        )
        for ctx_row in cursor.fetchall():
            rep.context_trust[ctx_row['context']] = ctx_row['trust']
            rep.context_observations[ctx_row['context']] = ctx_row['observation_count']

        # Load co-activations
        cursor = self.conn.execute(
            "SELECT expert_b, count, combined_quality FROM co_activation WHERE expert_a = ?",
            (expert_id,)
        )
        for co_row in cursor.fetchall():
            rep.co_activated_with[co_row['expert_b']] = co_row['count']
            rep.successful_pairs[co_row['expert_b']] = co_row['combined_quality']

        # Load substitutions
        cursor = self.conn.execute(
            "SELECT requested_id, quality_delta, context FROM substitutions WHERE substitute_id = ?",
            (expert_id,)
        )
        for sub_row in cursor.fetchall():
            rep.substituted_for.append((
                sub_row['requested_id'],
                sub_row['quality_delta'],
                sub_row['context']
            ))

        return rep

    def save(self, reputation: ExpertReputation):
        """
        Save expert reputation to database.

        Args:
            reputation: ExpertReputation to save
        """
        # Save core reputation
        self.conn.execute("""
            INSERT OR REPLACE INTO expert_reputation (
                expert_id, component, activation_count, first_seen, last_used,
                convergence_rate, stability, efficiency, average_confidence,
                semantic_cluster, modality_specialization, contexts_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reputation.expert_id,
            reputation.component,
            reputation.activation_count,
            reputation.first_seen,
            reputation.last_used,
            reputation.convergence_rate,
            reputation.stability,
            reputation.efficiency,
            reputation.average_confidence,
            reputation.semantic_cluster,
            reputation.modality_specialization,
            json.dumps(reputation.contexts_seen)
        ))

        # Save context trust
        for context, trust in reputation.context_trust.items():
            obs_count = reputation.context_observations.get(context, 0)
            self.conn.execute("""
                INSERT OR REPLACE INTO context_trust (expert_id, context, trust, observation_count)
                VALUES (?, ?, ?, ?)
            """, (reputation.expert_id, context, trust, obs_count))

        # Save co-activations
        for expert_b, count in reputation.co_activated_with.items():
            quality = reputation.successful_pairs.get(expert_b, 0.5)
            self.conn.execute("""
                INSERT OR REPLACE INTO co_activation (expert_a, expert_b, count, combined_quality)
                VALUES (?, ?, ?, ?)
            """, (reputation.expert_id, expert_b, count, quality))

        # Save substitutions (append only, don't replace)
        for requested_id, quality_delta, context in reputation.substituted_for:
            # Check if already exists
            cursor = self.conn.execute("""
                SELECT id FROM substitutions
                WHERE substitute_id = ? AND requested_id = ? AND context = ? AND quality_delta = ?
            """, (reputation.expert_id, requested_id, context, quality_delta))

            if cursor.fetchone() is None:
                # New substitution
                self.conn.execute("""
                    INSERT INTO substitutions (substitute_id, requested_id, context, quality_delta, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (reputation.expert_id, requested_id, context, quality_delta, time.time()))

        self.conn.commit()

    def get_or_create(self, expert_id: int, component: str = "thinker") -> ExpertReputation:
        """
        Get existing reputation or create new one with defaults.

        Args:
            expert_id: Expert identifier
            component: "thinker" or "talker"

        Returns:
            ExpertReputation (existing or new)
        """
        rep = self.get_reputation(expert_id, component)
        if rep is None:
            rep = ExpertReputation(expert_id=expert_id, component=component)
            self.save(rep)
        return rep

    def get_all_reputations(self, component: str = "thinker") -> List[ExpertReputation]:
        """
        Get all expert reputations for component.

        Args:
            component: "thinker" or "talker"

        Returns:
            List of ExpertReputation objects
        """
        cursor = self.conn.execute(
            "SELECT expert_id FROM expert_reputation WHERE component = ?",
            (component,)
        )
        expert_ids = [row['expert_id'] for row in cursor.fetchall()]

        return [self.get_reputation(eid, component) for eid in expert_ids]

    def get_statistics(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with counts and summary stats
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM expert_reputation")
        total_experts = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(DISTINCT context) FROM context_trust")
        total_contexts = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM co_activation")
        total_pairs = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM substitutions")
        total_substitutions = cursor.fetchone()[0]

        return {
            'total_experts_tracked': total_experts,
            'unique_contexts': total_contexts,
            'expert_pairs_tracked': total_pairs,
            'total_substitutions': total_substitutions,
            'db_path': str(self.db_path)
        }

    def close(self):
        """Close database connection."""
        self.conn.close()


# Convenience functions

def get_default_reputation_db() -> ExpertReputationDB:
    """Get default expert reputation database instance."""
    return ExpertReputationDB()


def record_expert_activation(
    expert_id: int,
    context: str,
    performance: Dict[str, float],
    db: Optional[ExpertReputationDB] = None
):
    """
    Convenience function to record expert activation.

    Args:
        expert_id: Expert that was activated
        context: Input context
        performance: Performance metrics dict
        db: Database instance (creates default if None)
    """
    if db is None:
        db = get_default_reputation_db()

    rep = db.get_or_create(expert_id)
    rep.record_activation(context, performance)
    db.save(rep)


def record_expert_co_activation(
    expert_ids: List[int],
    combined_quality: float,
    db: Optional[ExpertReputationDB] = None
):
    """
    Convenience function to record expert co-activation.

    Args:
        expert_ids: List of experts that were co-activated
        combined_quality: Combined output quality
        db: Database instance (creates default if None)
    """
    if db is None:
        db = get_default_reputation_db()

    for i, expert_a in enumerate(expert_ids):
        rep_a = db.get_or_create(expert_a)

        for expert_b in expert_ids[i+1:]:
            rep_a.record_co_activation(expert_b, combined_quality)

        db.save(rep_a)
