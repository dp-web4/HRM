"""
Policy Decision Logging Infrastructure - Phase 3

Logs every policy decision with full context for:
1. Continuous learning (building training dataset)
2. Human review and correction
3. Pattern extraction
4. Audit trail

Design principles:
- Log everything: situation, decision, reasoning, model response
- SQLite for easy querying and portability
- Timestamps and versioning for tracking evolution
- Ready for 50+ decisions → LoRA training
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class PolicyDecision:
    """A single policy decision with full context."""

    # Identifiers
    decision_id: str  # Unique ID for this decision
    timestamp: str  # ISO format timestamp

    # Input context
    situation: Dict[str, Any]  # The situation being evaluated
    team_context: str  # Team context provided

    # Model information
    model_name: str  # Which model made this decision
    model_version: str  # Model version/checkpoint
    prompt_version: str  # Which prompt variant was used

    # Model output
    decision: str  # allow, deny, require_attestation
    classification: str  # Model's classification of the situation
    risk_level: str  # low, medium, high
    reasoning: str  # Model's reasoning text
    full_response: str  # Complete model response

    # Evaluation (if available)
    expected_decision: Optional[str] = None  # For test scenarios
    decision_correct: Optional[bool] = None
    reasoning_coverage: Optional[float] = None

    # Human review (added later)
    reviewed: bool = False
    review_decision: Optional[str] = None  # human's decision
    review_reasoning: Optional[str] = None  # human's explanation
    review_timestamp: Optional[str] = None

    # Metadata
    scenario_id: Optional[str] = None  # If from test suite
    tags: str = ""  # Comma-separated tags for filtering


class PolicyDecisionLog:
    """
    Manages logging and retrieval of policy decisions.

    Features:
    - SQLite storage for easy querying
    - Automatic schema creation
    - Decision versioning (track model evolution)
    - Query interface for analysis
    - Export for training datasets
    """

    def __init__(self, db_path: str = "results/policy_decisions.db"):
        """Initialize decision log."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Main decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,

                    -- Input
                    situation TEXT NOT NULL,
                    team_context TEXT,

                    -- Model info
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,

                    -- Output
                    decision TEXT NOT NULL,
                    classification TEXT,
                    risk_level TEXT,
                    reasoning TEXT,
                    full_response TEXT,

                    -- Evaluation
                    expected_decision TEXT,
                    decision_correct INTEGER,
                    reasoning_coverage REAL,

                    -- Review
                    reviewed INTEGER DEFAULT 0,
                    review_decision TEXT,
                    review_reasoning TEXT,
                    review_timestamp TEXT,

                    -- Metadata
                    scenario_id TEXT,
                    tags TEXT
                )
            """)

            # Index for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON decisions(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviewed
                ON decisions(reviewed)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_correct
                ON decisions(decision_correct)
            """)

            conn.commit()

    def log_decision(self, decision: PolicyDecision) -> str:
        """
        Log a policy decision.

        Args:
            decision: PolicyDecision object

        Returns:
            decision_id of the logged decision
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Convert decision to dict and handle JSON serialization
            decision_dict = asdict(decision)
            decision_dict['situation'] = json.dumps(decision_dict['situation'])
            decision_dict['decision_correct'] = int(decision_dict['decision_correct']) if decision_dict['decision_correct'] is not None else None
            decision_dict['reviewed'] = int(decision_dict['reviewed'])

            # Insert
            columns = ', '.join(decision_dict.keys())
            placeholders = ', '.join(['?' for _ in decision_dict])

            cursor.execute(
                f"INSERT OR REPLACE INTO decisions ({columns}) VALUES ({placeholders})",
                list(decision_dict.values())
            )

            conn.commit()

        return decision.decision_id

    def get_decision(self, decision_id: str) -> Optional[PolicyDecision]:
        """Retrieve a specific decision by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM decisions WHERE decision_id = ?",
                (decision_id,)
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Convert back to PolicyDecision
            decision_dict = dict(row)
            decision_dict['situation'] = json.loads(decision_dict['situation'])
            decision_dict['decision_correct'] = bool(decision_dict['decision_correct']) if decision_dict['decision_correct'] is not None else None
            decision_dict['reviewed'] = bool(decision_dict['reviewed'])

            return PolicyDecision(**decision_dict)

    def get_unreviewed_decisions(self, limit: int = 10) -> List[PolicyDecision]:
        """Get decisions that haven't been reviewed yet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """SELECT * FROM decisions
                   WHERE reviewed = 0
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,)
            )

            decisions = []
            for row in cursor.fetchall():
                decision_dict = dict(row)
                decision_dict['situation'] = json.loads(decision_dict['situation'])
                decision_dict['decision_correct'] = bool(decision_dict['decision_correct']) if decision_dict['decision_correct'] is not None else None
                decision_dict['reviewed'] = bool(decision_dict['reviewed'])
                decisions.append(PolicyDecision(**decision_dict))

            return decisions

    def mark_reviewed(
        self,
        decision_id: str,
        review_decision: str,
        review_reasoning: str
    ) -> None:
        """Mark a decision as reviewed with human feedback."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """UPDATE decisions
                   SET reviewed = 1,
                       review_decision = ?,
                       review_reasoning = ?,
                       review_timestamp = ?
                   WHERE decision_id = ?""",
                (review_decision, review_reasoning, datetime.now().isoformat(), decision_id)
            )

            conn.commit()

    def get_corrections(self) -> List[PolicyDecision]:
        """
        Get decisions where human review disagreed with model.
        These are candidates for training data.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """SELECT * FROM decisions
                   WHERE reviewed = 1
                   AND decision != review_decision
                   ORDER BY timestamp DESC"""
            )

            corrections = []
            for row in cursor.fetchall():
                decision_dict = dict(row)
                decision_dict['situation'] = json.loads(decision_dict['situation'])
                decision_dict['decision_correct'] = bool(decision_dict['decision_correct']) if decision_dict['decision_correct'] is not None else None
                decision_dict['reviewed'] = bool(decision_dict['reviewed'])
                corrections.append(PolicyDecision(**decision_dict))

            return corrections

    def get_incorrect_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get decisions where model was incorrect (according to expected_decision)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """SELECT * FROM decisions
                   WHERE decision_correct = 0
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,)
            )

            decisions = []
            for row in cursor.fetchall():
                decision_dict = dict(row)
                decision_dict['situation'] = json.loads(decision_dict['situation'])
                decision_dict['decision_correct'] = bool(decision_dict['decision_correct']) if decision_dict['decision_correct'] is not None else None
                decision_dict['reviewed'] = bool(decision_dict['reviewed'])
                decisions.append(decision_dict)

            return decisions

    def get_all_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all decisions (most recent first)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """SELECT * FROM decisions
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,)
            )

            decisions = []
            for row in cursor.fetchall():
                decision_dict = dict(row)
                decision_dict['situation'] = json.loads(decision_dict['situation'])
                decision_dict['decision_correct'] = bool(decision_dict['decision_correct']) if decision_dict['decision_correct'] is not None else None
                decision_dict['reviewed'] = bool(decision_dict['reviewed'])
                decisions.append(decision_dict)

            return decisions

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged decisions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            stats = {}

            # Total decisions
            cursor.execute("SELECT COUNT(*) FROM decisions")
            stats['total_decisions'] = cursor.fetchone()[0]

            # Reviewed
            cursor.execute("SELECT COUNT(*) FROM decisions WHERE reviewed = 1")
            stats['reviewed_count'] = cursor.fetchone()[0]

            # Unreviewed
            cursor.execute("SELECT COUNT(*) FROM decisions WHERE reviewed = 0")
            stats['unreviewed_count'] = cursor.fetchone()[0]

            # Corrections (disagreements)
            cursor.execute(
                """SELECT COUNT(*) FROM decisions
                   WHERE reviewed = 1 AND decision != review_decision"""
            )
            stats['corrections'] = cursor.fetchone()[0]

            # Decision distribution
            cursor.execute(
                """SELECT decision, COUNT(*) as count
                   FROM decisions
                   GROUP BY decision"""
            )
            stats['decision_distribution'] = dict(cursor.fetchall())

            # Accuracy (on evaluated decisions)
            cursor.execute(
                """SELECT AVG(decision_correct)
                   FROM decisions
                   WHERE decision_correct IS NOT NULL"""
            )
            result = cursor.fetchone()[0]
            stats['accuracy'] = float(result) if result is not None else None

            return stats

    def export_for_training(
        self,
        output_file: str,
        min_corrections: int = 50,
        include_correct: bool = True
    ) -> int:
        """
        Export decisions for training dataset.

        Args:
            output_file: Path to save training data (JSON)
            min_corrections: Minimum corrections before exporting (safeguard)
            include_correct: Whether to include correct decisions too

        Returns:
            Number of examples exported
        """
        corrections = self.get_corrections()

        if len(corrections) < min_corrections:
            raise ValueError(
                f"Only {len(corrections)} corrections available. "
                f"Need at least {min_corrections} before training (safeguard)."
            )

        # Get all reviewed decisions if including correct ones
        if include_correct:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT * FROM decisions WHERE reviewed = 1"
                )

                all_reviewed = []
                for row in cursor.fetchall():
                    decision_dict = dict(row)
                    decision_dict['situation'] = json.loads(decision_dict['situation'])
                    all_reviewed.append(decision_dict)

                training_data = all_reviewed
        else:
            training_data = [asdict(c) for c in corrections]

        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        return len(training_data)


def create_decision_id(timestamp: Optional[str] = None) -> str:
    """Generate a unique decision ID."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    # Use timestamp + random suffix for uniqueness
    import hashlib
    hash_input = f"{timestamp}{datetime.now().microsecond}".encode()
    suffix = hashlib.md5(hash_input).hexdigest()[:8]
    clean_ts = timestamp.replace(':', '').replace('.', '')[:15]
    return f"dec_{clean_ts}_{suffix}"


if __name__ == "__main__":
    # Quick test of logging infrastructure
    print("Testing Policy Decision Logging Infrastructure\n")

    # Create logger
    log = PolicyDecisionLog("results/test_policy_decisions.db")

    # Create example decision
    example_decision = PolicyDecision(
        decision_id=create_decision_id(),
        timestamp=datetime.now().isoformat(),
        situation={
            "action_type": "read",
            "actor_role": "member",
            "resource": "docs/readme.md",
            "t3_tensor": {"competence": 0.7, "reliability": 0.8, "integrity": 0.9}
        },
        team_context="Standard team",
        model_name="phi-4-mini-7b",
        model_version="Q4_K_M",
        prompt_version="v2_fewshot_8examples",
        decision="allow",
        classification="routine_read_access",
        risk_level="low",
        reasoning="Member role can read public docs with sufficient trust",
        full_response="Classification: routine_read_access\nDecision: allow...",
        expected_decision="allow",
        decision_correct=True,
        reasoning_coverage=0.85,
        scenario_id="test_001",
        tags="test,read_access"
    )

    # Log it
    print("Logging example decision...")
    decision_id = log.log_decision(example_decision)
    print(f"Logged with ID: {decision_id}")

    # Retrieve it
    print("\nRetrieving decision...")
    retrieved = log.get_decision(decision_id)
    print(f"Retrieved: {retrieved.decision} for {retrieved.situation['action_type']}")

    # Get stats
    print("\nStatistics:")
    stats = log.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Logging infrastructure test complete!")
