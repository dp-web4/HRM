"""
Trust Tracking Database for Hierarchical Cognitive Architecture
Phase 1 Implementation - Trust Evolution System

Based on research from HIERARCHICAL_COGNITIVE_ARCHITECTURE.md
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelTrust:
    """Trust score for a model in a specific context"""
    model_name: str
    context_state: str  # 'stable', 'moving', 'unstable', 'novel'
    trust_score: float  # 0.0 - 1.0, starts at 0.5
    success_count: int
    failure_count: int
    last_updated: str


@dataclass
class TrainingExample:
    """Training example collected during WAKE/FOCUS"""
    id: Optional[int]
    timestamp: str
    input_data: str
    cognitive_layer: str  # 'claude', 'qwen-3b', 'qwen-1.5b', 'qwen-0.5b', 'phi3'
    response: str
    snarc_scores: Dict[str, float]  # Surprise, Novelty, Arousal, Reward, Conflict
    confidence_score: float
    outcome: str  # 'success', 'failure', 'uncertain'
    target_model: str  # Which model should learn from this
    importance: float  # SNARC composite score for prioritization


@dataclass
class ModelPerformance:
    """Performance metrics for model tracking"""
    id: Optional[int]
    model_name: str
    version: str
    test_timestamp: str
    accuracy: float
    avg_confidence: float
    resonance_with_claude: float  # How well aligned with Claude
    deployment_status: str  # 'testing', 'deployed', 'retired'


class TrustTrackingDatabase:
    """
    SQLite database for tracking model trust and training data

    Implements trust evolution patterns from ai-dna-discovery research
    """

    def __init__(self, db_path: str = "hierarchical_cognitive.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()

    def _initialize_database(self):
        """Create database schema"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Model trust table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_trust (
                model_name TEXT NOT NULL,
                context_state TEXT NOT NULL,
                trust_score REAL DEFAULT 0.5,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_updated TEXT,
                PRIMARY KEY (model_name, context_state)
            )
        """)

        # Training examples table (WAKE→DREAM pipeline)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_data TEXT NOT NULL,
                cognitive_layer TEXT NOT NULL,
                response TEXT NOT NULL,
                snarc_scores TEXT NOT NULL,  -- JSON
                confidence_score REAL NOT NULL,
                outcome TEXT NOT NULL,
                target_model TEXT NOT NULL,
                importance REAL NOT NULL,
                metadata TEXT  -- JSON for extensibility
            )
        """)

        # Model performance history
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                test_timestamp TEXT NOT NULL,
                accuracy REAL,
                avg_confidence REAL,
                resonance_with_claude REAL,
                deployment_status TEXT DEFAULT 'testing'
            )
        """)

        # Indexes for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_importance
            ON training_examples(importance DESC)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_target
            ON training_examples(target_model, importance DESC)
        """)

        self.conn.commit()

        # Initialize default trust scores for all models
        self._initialize_default_trust()

    def _initialize_default_trust(self):
        """Initialize trust scores for all models in all contexts"""
        models = ['claude', 'qwen-3b', 'qwen-1.5b', 'qwen-0.5b', 'phi3', 'gemma', 'tinyllama']
        contexts = ['stable', 'moving', 'unstable', 'novel']

        for model in models:
            for context in contexts:
                self.conn.execute("""
                    INSERT OR IGNORE INTO model_trust
                    (model_name, context_state, trust_score, last_updated)
                    VALUES (?, ?, 0.5, ?)
                """, (model, context, datetime.now().isoformat()))

        self.conn.commit()

    def get_trust(self, model_name: str, context_state: str) -> float:
        """Get trust score for model in specific context"""
        cursor = self.conn.execute("""
            SELECT trust_score FROM model_trust
            WHERE model_name = ? AND context_state = ?
        """, (model_name, context_state))

        row = cursor.fetchone()
        return row['trust_score'] if row else 0.5  # Default neutral trust

    def get_all_trust(self, context_state: str) -> Dict[str, float]:
        """Get trust scores for all models in given context"""
        cursor = self.conn.execute("""
            SELECT model_name, trust_score FROM model_trust
            WHERE context_state = ?
            ORDER BY trust_score DESC
        """, (context_state,))

        return {row['model_name']: row['trust_score'] for row in cursor.fetchall()}

    def update_trust(self, model_name: str, context_state: str,
                    outcome: str, learning_rate: float = 0.1):
        """
        Update trust based on outcome (success/failure)

        Uses trust evolution pattern from ai-dna-discovery:
        - Success: trust increases
        - Failure: trust decreases
        - Learning rate controls adaptation speed
        """
        current_trust = self.get_trust(model_name, context_state)

        # Calculate trust delta
        if outcome == 'success':
            # Increase trust, with diminishing returns near 1.0
            delta = learning_rate * (1.0 - current_trust)
            success_delta = 1
            failure_delta = 0
        elif outcome == 'failure':
            # Decrease trust, with diminishing impact near 0.0
            delta = -learning_rate * current_trust
            success_delta = 0
            failure_delta = 1
        else:  # uncertain
            delta = 0
            success_delta = 0
            failure_delta = 0

        new_trust = max(0.0, min(1.0, current_trust + delta))  # Clamp to [0,1]

        self.conn.execute("""
            UPDATE model_trust
            SET trust_score = ?,
                success_count = success_count + ?,
                failure_count = failure_count + ?,
                last_updated = ?
            WHERE model_name = ? AND context_state = ?
        """, (new_trust, success_delta, failure_delta,
              datetime.now().isoformat(), model_name, context_state))

        self.conn.commit()

        return new_trust

    def get_trust_history(self, model_name: str) -> List[ModelTrust]:
        """Get trust scores across all contexts for a model"""
        cursor = self.conn.execute("""
            SELECT * FROM model_trust
            WHERE model_name = ?
            ORDER BY trust_score DESC
        """, (model_name,))

        return [ModelTrust(**dict(row)) for row in cursor.fetchall()]

    def store_training_example(self, example: TrainingExample):
        """Store high-quality training example for DREAM consolidation"""
        self.conn.execute("""
            INSERT INTO training_examples
            (timestamp, input_data, cognitive_layer, response,
             snarc_scores, confidence_score, outcome, target_model, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            example.timestamp,
            example.input_data,
            example.cognitive_layer,
            example.response,
            json.dumps(example.snarc_scores),
            example.confidence_score,
            example.outcome,
            example.target_model,
            example.importance
        ))

        self.conn.commit()

    def get_training_examples(self, target_model: Optional[str] = None,
                            limit: int = 100,
                            min_importance: float = 0.5) -> List[TrainingExample]:
        """
        Get training examples sorted by importance (SNARC scores)

        Used for DREAM consolidation - selective replay of high-salience experiences
        """
        query = """
            SELECT * FROM training_examples
            WHERE importance >= ?
        """
        params = [min_importance]

        if target_model:
            query += " AND target_model = ?"
            params.append(target_model)

        query += " ORDER BY importance DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)

        examples = []
        for row in cursor.fetchall():
            example_dict = dict(row)
            example_dict['snarc_scores'] = json.loads(example_dict['snarc_scores'])
            examples.append(TrainingExample(**example_dict))

        return examples

    def store_performance(self, perf: ModelPerformance):
        """Store model performance metrics"""
        self.conn.execute("""
            INSERT INTO model_performance
            (model_name, version, test_timestamp, accuracy,
             avg_confidence, resonance_with_claude, deployment_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            perf.model_name, perf.version, perf.test_timestamp,
            perf.accuracy, perf.avg_confidence,
            perf.resonance_with_claude, perf.deployment_status
        ))

        self.conn.commit()

    def get_performance_history(self, model_name: str) -> List[ModelPerformance]:
        """Get performance history for model"""
        cursor = self.conn.execute("""
            SELECT * FROM model_performance
            WHERE model_name = ?
            ORDER BY test_timestamp DESC
        """, (model_name,))

        return [ModelPerformance(**dict(row)) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_examples,
                AVG(importance) as avg_importance,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END) as failure_count
            FROM training_examples
        """)

        stats = dict(cursor.fetchone())

        # Get trust scores summary
        cursor = self.conn.execute("""
            SELECT model_name, AVG(trust_score) as avg_trust
            FROM model_trust
            GROUP BY model_name
            ORDER BY avg_trust DESC
        """)

        stats['model_trust'] = {row['model_name']: row['avg_trust']
                               for row in cursor.fetchall()}

        return stats

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test database creation
    print("Creating trust tracking database...")

    with TrustTrackingDatabase("test_hierarchical_cognitive.db") as db:
        print("\n✅ Database initialized")

        # Test trust retrieval
        print("\n📊 Initial trust scores (novel context):")
        trust_scores = db.get_all_trust('novel')
        for model, trust in trust_scores.items():
            print(f"  {model}: {trust:.3f}")

        # Simulate trust evolution
        print("\n🔄 Simulating trust updates...")
        db.update_trust('qwen-1.5b', 'stable', 'success')
        db.update_trust('qwen-1.5b', 'stable', 'success')
        db.update_trust('qwen-0.5b', 'stable', 'failure')

        print("\n📊 Updated trust scores (stable context):")
        trust_scores = db.get_all_trust('stable')
        for model in ['qwen-1.5b', 'qwen-0.5b', 'phi3']:
            print(f"  {model}: {trust_scores[model]:.3f}")

        # Test training example storage
        print("\n💾 Storing training example...")
        example = TrainingExample(
            id=None,
            timestamp=datetime.now().isoformat(),
            input_data="What is 2+2?",
            cognitive_layer="qwen-1.5b",
            response="4",
            snarc_scores={
                'surprise': 0.1,
                'novelty': 0.2,
                'arousal': 0.3,
                'reward': 0.8,
                'conflict': 0.1
            },
            confidence_score=0.95,
            outcome='success',
            target_model='qwen-0.5b',
            importance=0.5
        )
        db.store_training_example(example)

        # Retrieve examples
        examples = db.get_training_examples(limit=10)
        print(f"  Retrieved {len(examples)} training examples")

        # Statistics
        print("\n📈 Database statistics:")
        stats = db.get_statistics()
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Avg importance: {stats.get('avg_importance', 0):.3f}")
        print(f"  Success rate: {stats['success_count']}/{stats['success_count'] + stats['failure_count']}")

        print("\n✅ Trust tracking database ready for Phase 1!")
