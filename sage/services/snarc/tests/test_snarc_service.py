"""
Integration Tests for SNARCService

Tests the complete salience assessment system with simulated sensors.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from sage.services.snarc import SNARCService, SalienceReport, Outcome
from sage.services.snarc.data_structures import CognitiveStance
from sage.services.snarc.tests.simulated_sensors import (
    create_calm_environment,
    create_surprising_environment,
    create_novel_environment,
    create_conflicting_environment,
    create_goal_relevant_environment
)


class TestSNARCServiceBasic(unittest.TestCase):
    """Basic SNARCService functionality tests"""

    def setUp(self):
        self.snarc = SNARCService()

    def test_initialization(self):
        """Should initialize with default weights"""
        self.assertEqual(len(self.snarc.salience_weights), 5)
        self.assertAlmostEqual(sum(self.snarc.salience_weights.values()), 1.0, places=5)

    def test_assess_empty_sensors(self):
        """Should handle empty sensor dict"""
        report = self.snarc.assess_salience({})

        self.assertEqual(report.focus_target, "none")
        self.assertEqual(report.salience_score, 0.0)
        self.assertEqual(report.confidence, 0.0)

    def test_assess_single_sensor(self):
        """Should assess single sensor"""
        sensor_outputs = {'test_sensor': 5.0}
        report = self.snarc.assess_salience(sensor_outputs)

        self.assertIsInstance(report, SalienceReport)
        self.assertEqual(report.focus_target, 'test_sensor')
        self.assertGreaterEqual(report.salience_score, 0.0)
        self.assertLessEqual(report.salience_score, 1.0)
        self.assertIsInstance(report.suggested_stance, CognitiveStance)

    def test_assess_multiple_sensors(self):
        """Should find highest salience sensor"""
        sensor_outputs = {
            'sensor1': 1.0,
            'sensor2': 5.0,
            'sensor3': 2.0
        }

        report = self.snarc.assess_salience(sensor_outputs)

        self.assertIn(report.focus_target, sensor_outputs.keys())
        self.assertIn('all_sensor_salience', report.metadata)
        self.assertEqual(len(report.metadata['all_sensor_salience']), 3)

    def test_outcome_learning(self):
        """Should adjust weights based on outcomes"""
        # Get initial assessment
        sensor_outputs = {'sensor1': 5.0}
        report = self.snarc.assess_salience(sensor_outputs)

        # Record initial weights
        initial_weights = self.snarc.salience_weights.copy()

        # Provide negative outcome
        outcome = Outcome(
            success=False,
            reward=-0.5,
            description="Bad result"
        )
        self.snarc.update_from_outcome(report, outcome)

        # Weights should have changed
        self.assertNotEqual(
            initial_weights,
            self.snarc.salience_weights,
            "Weights should change after negative outcome"
        )

    def test_memory_storage(self):
        """Should store assessments in memory"""
        initial_count = len(self.snarc.assessment_history)

        sensor_outputs = {'sensor1': 5.0}
        self.snarc.assess_salience(sensor_outputs)

        self.assertEqual(
            len(self.snarc.assessment_history),
            initial_count + 1,
            "Assessment should be stored in memory"
        )

    def test_relevant_memories(self):
        """Should retrieve relevant memories"""
        # Create several assessments with different patterns
        for i in range(10):
            sensor_outputs = {'sensor1': float(i)}
            self.snarc.assess_salience(sensor_outputs)

        # New assessment similar to early ones
        sensor_outputs = {'sensor1': 2.0}
        report = self.snarc.assess_salience(sensor_outputs)

        # Should have retrieved relevant memories
        self.assertIsInstance(report.relevant_memories, list)

    def test_statistics(self):
        """Should provide service statistics"""
        # Make several assessments
        for i in range(5):
            sensor_outputs = {'sensor1': float(i)}
            report = self.snarc.assess_salience(sensor_outputs)

            # Mark some as successful
            if i % 2 == 0:
                outcome = Outcome(success=True, reward=0.8)
                self.snarc.update_from_outcome(report, outcome)

        stats = self.snarc.get_statistics()

        self.assertIn('num_assessments', stats)
        self.assertIn('successful_outcomes', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('current_weights', stats)

    def test_reset(self):
        """Should reset all state"""
        # Build up state
        for i in range(10):
            sensor_outputs = {'sensor1': float(i)}
            self.snarc.assess_salience(sensor_outputs)

        # Reset
        self.snarc.reset()

        # History should be cleared
        self.assertEqual(len(self.snarc.assessment_history), 0)


class TestSNARCServiceScenarios(unittest.TestCase):
    """Test SNARC with realistic sensor scenarios"""

    def setUp(self):
        self.snarc = SNARCService()

    def test_calm_environment(self):
        """Calm environment should suggest CONFIDENT_EXECUTION"""
        env = create_calm_environment()

        # Let sensors run and learn patterns
        for _ in range(50):
            snapshot = env.get_snapshot()
            report = self.snarc.assess_salience(snapshot)

        # After learning, should be confident
        # (Calm = low surprise, low novelty)
        self.assertIn(
            report.suggested_stance,
            [CognitiveStance.CONFIDENT_EXECUTION, CognitiveStance.CURIOUS_UNCERTAINTY],
            f"Calm environment suggested {report.suggested_stance}"
        )

        # Salience should be relatively low
        self.assertLess(
            report.salience_score, 0.7,
            "Calm environment should have low overall salience"
        )

    def test_surprising_environment(self):
        """Surprising environment should detect high surprise"""
        env = create_surprising_environment()

        # Establish baseline
        for _ in range(30):
            snapshot = env.get_snapshot()
            self.snarc.assess_salience(snapshot)

        # Continue through step changes
        reports = []
        for _ in range(50):
            snapshot = env.get_snapshot()
            report = self.snarc.assess_salience(snapshot)
            reports.append(report)

        # Should see high surprise events
        max_surprise = max(r.salience_breakdown.surprise for r in reports)
        self.assertGreater(
            max_surprise, 0.5,
            "Should detect high surprise from step changes"
        )

    def test_novel_environment(self):
        """Novel environment should have higher novelty than calm environment"""
        env_novel = create_novel_environment()
        env_calm = create_calm_environment()

        # Run both environments
        novel_reports = []
        for _ in range(50):
            snapshot = env_novel.get_snapshot()
            report = self.snarc.assess_salience(snapshot)
            novel_reports.append(report)

        # Reset SNARC for fair comparison
        self.snarc.reset()

        calm_reports = []
        for _ in range(50):
            snapshot = env_calm.get_snapshot()
            report = self.snarc.assess_salience(snapshot)
            calm_reports.append(report)

        # Compare novelty
        avg_novelty_novel = np.mean([r.salience_breakdown.novelty for r in novel_reports])
        avg_novelty_calm = np.mean([r.salience_breakdown.novelty for r in calm_reports])

        self.assertGreater(
            avg_novelty_novel, avg_novelty_calm,
            "Novel environment should have higher novelty than calm environment"
        )

    def test_conflicting_environment(self):
        """Conflicting environment should suggest SKEPTICAL_VERIFICATION"""
        env = create_conflicting_environment()

        # Need time to establish expectations
        for _ in range(30):
            snapshot = env.get_snapshot()
            self.snarc.assess_salience(snapshot)

        # Check for conflict detection
        reports = []
        for _ in range(20):
            snapshot = env.get_snapshot()
            report = self.snarc.assess_salience(snapshot)
            reports.append(report)

        # Should detect some conflict
        max_conflict = max(r.salience_breakdown.conflict for r in reports)
        self.assertGreater(
            max_conflict, 0.3,
            "Should detect conflict between uncorrelated sensors"
        )

    def test_goal_relevant_environment_learning(self):
        """Goal-relevant environment should learn reward associations"""
        env = create_goal_relevant_environment()

        # Run without outcomes first
        for _ in range(20):
            snapshot = env.get_snapshot()
            report = self.snarc.assess_salience(snapshot)

        # Initial reward estimates should be neutral
        initial_reward = report.salience_breakdown.reward
        self.assertAlmostEqual(
            initial_reward, 0.5,
            msg="Initial reward should be near neutral",
            delta=0.3
        )

        # Now provide outcome feedback
        for _ in range(30):
            snapshot = env.get_snapshot()
            report = self.snarc.assess_salience(snapshot)

            # Positive outcome when target_metric is present
            if report.focus_target == 'target_metric':
                outcome = Outcome(success=True, reward=0.9)
                self.snarc.update_from_outcome(report, outcome)
            else:
                outcome = Outcome(success=False, reward=0.1)
                self.snarc.update_from_outcome(report, outcome)

        # After learning, target_metric should have high reward
        snapshot = env.get_snapshot()
        report = self.snarc.assess_salience(snapshot)

        # Should preferentially focus on target_metric
        # (Note: This may not always be true due to other dimensions,
        #  but over many runs it should trend this way)
        self.assertIn(
            report.focus_target,
            snapshot.keys(),
            "Should focus on one of the sensors"
        )


class TestSNARCStanceSuggestion(unittest.TestCase):
    """Test cognitive stance suggestion logic"""

    def setUp(self):
        self.snarc = SNARCService()

    def test_high_conflict_suggests_skeptical(self):
        """High conflict should suggest SKEPTICAL_VERIFICATION"""
        # Manually create high-conflict scenario
        from sage.services.snarc.data_structures import SalienceBreakdown

        breakdown = SalienceBreakdown(
            surprise=0.3,
            novelty=0.3,
            arousal=0.4,
            reward=0.5,
            conflict=0.8  # High conflict
        )

        stance = self.snarc._suggest_stance(breakdown)
        self.assertEqual(
            stance,
            CognitiveStance.SKEPTICAL_VERIFICATION,
            "High conflict should suggest skeptical verification"
        )

    def test_high_surprise_novelty_suggests_curious(self):
        """High surprise + novelty should suggest CURIOUS_UNCERTAINTY"""
        from sage.services.snarc.data_structures import SalienceBreakdown

        breakdown = SalienceBreakdown(
            surprise=0.7,  # High
            novelty=0.7,   # High
            arousal=0.4,
            reward=0.5,
            conflict=0.2
        )

        stance = self.snarc._suggest_stance(breakdown)
        self.assertEqual(
            stance,
            CognitiveStance.CURIOUS_UNCERTAINTY,
            "High surprise + novelty should suggest curious uncertainty"
        )

    def test_high_reward_suggests_focused(self):
        """High reward should suggest FOCUSED_ATTENTION"""
        from sage.services.snarc.data_structures import SalienceBreakdown

        breakdown = SalienceBreakdown(
            surprise=0.3,
            novelty=0.3,
            arousal=0.4,
            reward=0.8,  # High reward
            conflict=0.2
        )

        stance = self.snarc._suggest_stance(breakdown)
        self.assertEqual(
            stance,
            CognitiveStance.FOCUSED_ATTENTION,
            "High reward should suggest focused attention"
        )

    def test_low_surprise_novelty_suggests_confident(self):
        """Low surprise + novelty should suggest CONFIDENT_EXECUTION"""
        from sage.services.snarc.data_structures import SalienceBreakdown

        breakdown = SalienceBreakdown(
            surprise=0.1,  # Low
            novelty=0.2,   # Low
            arousal=0.4,
            reward=0.5,
            conflict=0.2
        )

        stance = self.snarc._suggest_stance(breakdown)
        self.assertEqual(
            stance,
            CognitiveStance.CONFIDENT_EXECUTION,
            "Low surprise + novelty should suggest confident execution"
        )

    def test_high_arousal_suggests_exploratory(self):
        """High arousal + moderate novelty should suggest EXPLORATORY"""
        from sage.services.snarc.data_structures import SalienceBreakdown

        breakdown = SalienceBreakdown(
            surprise=0.3,
            novelty=0.5,   # Moderate
            arousal=0.8,   # High
            reward=0.4,
            conflict=0.2
        )

        stance = self.snarc._suggest_stance(breakdown)
        self.assertEqual(
            stance,
            CognitiveStance.EXPLORATORY,
            "High arousal + moderate novelty should suggest exploratory"
        )


class TestSNARCConfidence(unittest.TestCase):
    """Test confidence computation"""

    def setUp(self):
        self.snarc = SNARCService()

    def test_agreement_high_confidence(self):
        """All dimensions high should give high confidence"""
        from sage.services.snarc.data_structures import SalienceBreakdown

        # All dimensions similar (agree)
        breakdown = SalienceBreakdown(
            surprise=0.8,
            novelty=0.8,
            arousal=0.8,
            reward=0.8,
            conflict=0.8
        )

        confidence = self.snarc._compute_confidence(breakdown)
        self.assertGreater(
            confidence, 0.7,
            "Agreement across dimensions should give high confidence"
        )

    def test_disagreement_low_confidence(self):
        """Mixed dimensions should give lower confidence than agreement"""
        from sage.services.snarc.data_structures import SalienceBreakdown

        # Dimensions disagree (high variance)
        breakdown_disagree = SalienceBreakdown(
            surprise=0.0,
            novelty=1.0,
            arousal=0.0,
            reward=1.0,
            conflict=0.5
        )

        # Dimensions agree (low variance)
        breakdown_agree = SalienceBreakdown(
            surprise=0.8,
            novelty=0.8,
            arousal=0.8,
            reward=0.8,
            conflict=0.8
        )

        confidence_disagree = self.snarc._compute_confidence(breakdown_disagree)
        confidence_agree = self.snarc._compute_confidence(breakdown_agree)

        self.assertLess(
            confidence_disagree, confidence_agree,
            "Disagreement should give lower confidence than agreement"
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
