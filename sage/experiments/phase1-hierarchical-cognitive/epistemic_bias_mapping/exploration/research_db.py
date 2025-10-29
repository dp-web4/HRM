#!/usr/bin/env python3
"""
Research Experiment Tracking Database

Multi-dimensional exploration of scaffolding suitability threshold.
Tracks all experiments, parameters, results, and metrics.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class ResearchDB:
    """Experiment tracking and analysis database"""

    def __init__(self, db_path: str = "./exploration/research.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema"""

        cursor = self.conn.cursor()

        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_type TEXT NOT NULL,
                model_variant TEXT NOT NULL,
                training_size INTEGER,
                scaffolding_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                duration_seconds REAL,
                notes TEXT
            )
        """)

        # Parameters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                turn_number INTEGER,
                prompt TEXT,
                response TEXT,
                response_length INTEGER,
                energy REAL,
                enhanced_energy REAL,
                coherence_score REAL,
                pattern_collapse BOOLEAN,
                on_topic BOOLEAN,
                epistemic_humility BOOLEAN,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        # Metrics table (aggregated per experiment)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                avg_energy REAL,
                avg_enhanced_energy REAL,
                avg_coherence REAL,
                pattern_collapse_rate REAL,
                on_topic_rate REAL,
                epistemic_humility_rate REAL,
                final_trust_score REAL,
                total_turns INTEGER,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        # Comparisons table (for threshold analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_size INTEGER NOT NULL,
                bare_experiment_id INTEGER,
                scaffolded_experiment_id INTEGER,
                bare_better BOOLEAN,
                scaffolded_better BOOLEAN,
                threshold_crossed BOOLEAN,
                notes TEXT,
                FOREIGN KEY (bare_experiment_id) REFERENCES experiments(id),
                FOREIGN KEY (scaffolded_experiment_id) REFERENCES experiments(id)
            )
        """)

        self.conn.commit()

    def create_experiment(
        self,
        experiment_type: str,
        model_variant: str,
        training_size: int,
        scaffolding_type: str,
        parameters: Dict[str, Any] = None,
        notes: str = None
    ) -> int:
        """Create new experiment record"""

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO experiments (
                experiment_type, model_variant, training_size,
                scaffolding_type, timestamp, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_type,
            model_variant,
            training_size,
            scaffolding_type,
            datetime.now().isoformat(),
            'running',
            notes
        ))

        experiment_id = cursor.lastrowid

        # Store parameters
        if parameters:
            for name, value in parameters.items():
                cursor.execute("""
                    INSERT INTO parameters (experiment_id, param_name, param_value)
                    VALUES (?, ?, ?)
                """, (experiment_id, name, json.dumps(value)))

        self.conn.commit()
        return experiment_id

    def add_result(
        self,
        experiment_id: int,
        turn_number: int,
        prompt: str,
        response: str,
        energy: float = None,
        enhanced_energy: float = None,
        coherence_score: float = None,
        pattern_collapse: bool = False,
        on_topic: bool = True,
        epistemic_humility: bool = True
    ):
        """Add result for a turn"""

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO results (
                experiment_id, turn_number, prompt, response,
                response_length, energy, enhanced_energy, coherence_score,
                pattern_collapse, on_topic, epistemic_humility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, turn_number, prompt, response,
            len(response.split()), energy, enhanced_energy, coherence_score,
            pattern_collapse, on_topic, epistemic_humility
        ))

        self.conn.commit()

    def complete_experiment(
        self,
        experiment_id: int,
        duration_seconds: float,
        metrics: Dict[str, Any]
    ):
        """Mark experiment complete and store metrics"""

        cursor = self.conn.cursor()

        # Update experiment status
        cursor.execute("""
            UPDATE experiments
            SET status = 'completed', duration_seconds = ?
            WHERE id = ?
        """, (duration_seconds, experiment_id))

        # Store aggregated metrics
        cursor.execute("""
            INSERT INTO metrics (
                experiment_id, avg_energy, avg_enhanced_energy,
                avg_coherence, pattern_collapse_rate, on_topic_rate,
                epistemic_humility_rate, final_trust_score, total_turns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            metrics.get('avg_energy'),
            metrics.get('avg_enhanced_energy'),
            metrics.get('avg_coherence'),
            metrics.get('pattern_collapse_rate'),
            metrics.get('on_topic_rate'),
            metrics.get('epistemic_humility_rate'),
            metrics.get('final_trust_score'),
            metrics.get('total_turns')
        ))

        self.conn.commit()

    def create_comparison(
        self,
        training_size: int,
        bare_exp_id: int,
        scaffolded_exp_id: int,
        analysis: Dict[str, Any]
    ):
        """Create comparison record"""

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO comparisons (
                training_size, bare_experiment_id, scaffolded_experiment_id,
                bare_better, scaffolded_better, threshold_crossed, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            training_size, bare_exp_id, scaffolded_exp_id,
            analysis.get('bare_better'),
            analysis.get('scaffolded_better'),
            analysis.get('threshold_crossed'),
            json.dumps(analysis.get('notes', {}))
        ))

        self.conn.commit()

    def get_experiments_by_size(self, training_size: int) -> pd.DataFrame:
        """Get all experiments for a training size"""

        query = """
            SELECT e.*, m.*
            FROM experiments e
            LEFT JOIN metrics m ON e.id = m.experiment_id
            WHERE e.training_size = ?
            ORDER BY e.timestamp
        """

        return pd.read_sql_query(query, self.conn, params=(training_size,))

    def get_threshold_analysis(self) -> pd.DataFrame:
        """Get threshold crossing analysis"""

        query = """
            SELECT
                c.training_size,
                c.bare_better,
                c.scaffolded_better,
                c.threshold_crossed,
                e1.id as bare_exp_id,
                e2.id as scaffolded_exp_id,
                m1.avg_enhanced_energy as bare_energy,
                m2.avg_enhanced_energy as scaffolded_energy,
                m1.pattern_collapse_rate as bare_collapse,
                m2.pattern_collapse_rate as scaffolded_collapse
            FROM comparisons c
            JOIN experiments e1 ON c.bare_experiment_id = e1.id
            JOIN experiments e2 ON c.scaffolded_experiment_id = e2.id
            LEFT JOIN metrics m1 ON e1.id = m1.experiment_id
            LEFT JOIN metrics m2 ON e2.id = m2.experiment_id
            ORDER BY c.training_size
        """

        return pd.read_sql_query(query, self.conn)

    def get_all_results(self, experiment_id: int) -> pd.DataFrame:
        """Get all turn results for an experiment"""

        query = """
            SELECT * FROM results
            WHERE experiment_id = ?
            ORDER BY turn_number
        """

        return pd.read_sql_query(query, self.conn, params=(experiment_id,))

    def export_summary(self, output_path: str = "./exploration/research_summary.json"):
        """Export complete research summary"""

        summary = {
            'generated_at': datetime.now().isoformat(),
            'experiments': {},
            'threshold_analysis': {},
            'key_findings': []
        }

        # Get all experiments
        query = "SELECT * FROM experiments WHERE status = 'completed'"
        experiments = pd.read_sql_query(query, self.conn)

        for _, exp in experiments.iterrows():
            exp_id = exp['id']

            # Get metrics
            metrics_query = "SELECT * FROM metrics WHERE experiment_id = ?"
            metrics = pd.read_sql_query(metrics_query, self.conn, params=(exp_id,))

            # Get results
            results_query = "SELECT * FROM results WHERE experiment_id = ?"
            results = pd.read_sql_query(results_query, self.conn, params=(exp_id,))

            summary['experiments'][exp_id] = {
                'info': exp.to_dict(),
                'metrics': metrics.to_dict('records')[0] if not metrics.empty else {},
                'results': results.to_dict('records')
            }

        # Get threshold analysis
        threshold_df = self.get_threshold_analysis()
        summary['threshold_analysis'] = threshold_df.to_dict('records')

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Test database
    print("Research Database Test")
    print("=" * 80)

    db = ResearchDB()

    # Create test experiment
    exp_id = db.create_experiment(
        experiment_type="threshold_test",
        model_variant="phase1",
        training_size=25,
        scaffolding_type="bare",
        parameters={
            'temperature': 0.7,
            'max_tokens': 200
        },
        notes="Test experiment"
    )

    print(f"Created experiment: {exp_id}")

    # Add test result
    db.add_result(
        experiment_id=exp_id,
        turn_number=1,
        prompt="Test question?",
        response="Test response that is coherent and on-topic.",
        enhanced_energy=0.15,
        coherence_score=0.85,
        pattern_collapse=False,
        on_topic=True,
        epistemic_humility=True
    )

    print("Added result")

    # Complete experiment
    db.complete_experiment(
        experiment_id=exp_id,
        duration_seconds=120.5,
        metrics={
            'avg_enhanced_energy': 0.15,
            'avg_coherence': 0.85,
            'pattern_collapse_rate': 0.0,
            'on_topic_rate': 1.0,
            'epistemic_humility_rate': 1.0,
            'total_turns': 1
        }
    )

    print("Completed experiment")

    # Query
    df = db.get_experiments_by_size(25)
    print(f"\nExperiments with 25 examples:")
    print(df[['id', 'experiment_type', 'scaffolding_type', 'status']])

    # Export
    summary = db.export_summary()
    print(f"\nExported summary: {len(summary['experiments'])} experiments")

    db.close()
    print("\n✓ Database test complete!")
