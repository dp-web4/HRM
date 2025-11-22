"""
Test Suite for Epistemic Memory Bridge

Tests SAGE ↔ Epistemic integration:
- SNARC score mapping
- Observation storage
- Session recording
- Context queries
- Fallback mode

Run:
    cd /home/dp/ai-workspace/HRM
    python3 sage/integration/test_epistemic_bridge.py

Author: Thor
Date: 2025-11-22
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from integration.epistemic_memory import (
    EpistemicMemoryBridge,
    SNARCScores,
    Observation,
    LearningSession,
    EPISTEMIC_AVAILABLE
)


class TestSNARCScores(unittest.TestCase):
    """Test SNARC score calculations and mappings"""

    def test_composite_score(self):
        """Test composite salience calculation"""
        scores = SNARCScores(
            surprise=0.8,
            novelty=0.7,
            arousal=0.6,
            reward=0.9,
            conflict=0.3
        )

        composite = scores.composite_score()

        # Weighted: 0.3*0.8 + 0.25*0.7 + 0.2*0.6 + 0.15*0.9 + 0.1*0.3
        # = 0.24 + 0.175 + 0.12 + 0.135 + 0.03 = 0.70
        self.assertAlmostEqual(composite, 0.70, places=2)

    def test_epistemic_score_mapping(self):
        """Test SNARC → epistemic dimension mapping"""
        scores = SNARCScores(
            surprise=0.8,
            novelty=0.7,
            arousal=0.6,
            reward=0.9,
            conflict=0.2
        )

        epistemic = scores.to_epistemic_scores()

        # Verify swap: epistemic 'surprise' = SNARC 'novelty'
        self.assertEqual(epistemic['surprise'], 0.7)
        # Verify swap: epistemic 'novelty' = SNARC 'surprise'
        self.assertEqual(epistemic['novelty'], 0.8)
        # Direct mappings
        self.assertEqual(epistemic['arousal'], 0.6)
        self.assertEqual(epistemic['confidence'], 0.9)
        # Conflict → validation status
        self.assertEqual(epistemic['validation_status'], 'proven')

    def test_conflict_to_validation_mapping(self):
        """Test conflict score maps to validation statuses"""
        # Low conflict = proven
        scores1 = SNARCScores(0.5, 0.5, 0.5, 0.5, 0.2)
        self.assertEqual(scores1._conflict_to_validation(), 'proven')

        # Medium conflict = tested
        scores2 = SNARCScores(0.5, 0.5, 0.5, 0.5, 0.5)
        self.assertEqual(scores2._conflict_to_validation(), 'tested')

        # High conflict = speculative
        scores3 = SNARCScores(0.5, 0.5, 0.5, 0.5, 0.8)
        self.assertEqual(scores3._conflict_to_validation(), 'speculative')


class TestObservation(unittest.TestCase):
    """Test observation data structures"""

    def test_observation_creation(self):
        """Test creating observation with SNARC scores"""
        snarc = SNARCScores(0.8, 0.7, 0.6, 0.9, 0.3)
        obs = Observation(
            description="Novel pattern detected in attention convergence",
            modality="multimodal",
            snarc_scores=snarc,
            timestamp=datetime.now(timezone.utc),
            context={'plugin': 'vision', 'iteration': 5}
        )

        self.assertEqual(obs.modality, "multimodal")
        self.assertIsNotNone(obs.summary())
        self.assertIn('plugin', obs.to_dict()['context'])

    def test_observation_summary_truncation(self):
        """Test long descriptions get truncated"""
        long_desc = "x" * 300
        snarc = SNARCScores(0.8, 0.7, 0.6, 0.9, 0.3)
        obs = Observation(long_desc, "language", snarc, datetime.now(timezone.utc))

        summary = obs.summary()
        self.assertLessEqual(len(summary), 200)


class TestLearningSession(unittest.TestCase):
    """Test learning session data structures"""

    def test_session_creation(self):
        """Test creating learning session"""
        session = LearningSession(
            session_id="test-session-001",
            started=datetime.now(timezone.utc),
            ended=datetime.now(timezone.utc),
            iterations=10,
            plugins_used=['vision', 'language', 'memory'],
            high_salience_count=3,
            convergence_failures=1,
            trust_updates=2,
            discoveries_witnessed=['disc-001', 'disc-002'],
            quality_score=0.85
        )

        self.assertEqual(session.iterations, 10)
        self.assertEqual(len(session.plugins_used), 3)
        self.assertAlmostEqual(session.quality_score, 0.85)

    def test_session_serialization(self):
        """Test session converts to dict properly"""
        session = LearningSession(
            session_id="test-002",
            started=datetime.now(timezone.utc),
            ended=datetime.now(timezone.utc),
            iterations=5,
            plugins_used=['vision'],
            high_salience_count=1,
            convergence_failures=0,
            trust_updates=1,
            discoveries_witnessed=[],
            quality_score=0.9
        )

        data = session.to_dict()
        self.assertIn('session_id', data)
        self.assertIn('quality', data)
        self.assertEqual(data['iterations'], 5)


class TestEpistemicMemoryBridge(unittest.TestCase):
    """Test epistemic memory bridge functionality"""

    def setUp(self):
        """Set up test bridge"""
        self.bridge = EpistemicMemoryBridge(
            machine='thor',
            project='sage-test',
            salience_threshold=0.7,
            enable_witnessing=False  # Disable for tests
        )

    def test_bridge_initialization(self):
        """Test bridge initializes correctly"""
        self.assertEqual(self.bridge.machine, 'thor')
        self.assertEqual(self.bridge.project, 'sage-test')
        self.assertEqual(self.bridge.salience_threshold, 0.7)
        self.assertFalse(self.bridge.enable_witnessing)

    def test_salience_threshold_check(self):
        """Test high/low salience detection"""
        high = SNARCScores(0.8, 0.8, 0.7, 0.9, 0.5)
        low = SNARCScores(0.3, 0.3, 0.2, 0.4, 0.1)

        self.assertTrue(self.bridge.is_high_salience(high))
        self.assertFalse(self.bridge.is_high_salience(low))

    def test_store_observation_high_salience(self):
        """Test storing high-salience observation"""
        snarc = SNARCScores(0.8, 0.8, 0.7, 0.9, 0.3)
        obs = Observation(
            description="Test high-salience observation",
            modality="vision",
            snarc_scores=snarc,
            timestamp=datetime.now(timezone.utc)
        )

        obs_id = self.bridge.store_observation(obs)
        self.assertIsNotNone(obs_id)

        # If epistemic available, should be real ID
        # If not, should be local cache ID
        if EPISTEMIC_AVAILABLE:
            self.assertTrue(len(obs_id) > 10)  # Real knowledge ID
        else:
            self.assertTrue(obs_id.startswith('local-obs-'))

    def test_store_observation_low_salience(self):
        """Test low-salience observation is NOT stored"""
        snarc = SNARCScores(0.3, 0.3, 0.2, 0.4, 0.1)
        obs = Observation(
            description="Test low-salience observation",
            modality="vision",
            snarc_scores=snarc,
            timestamp=datetime.now(timezone.utc)
        )

        obs_id = self.bridge.store_observation(obs)
        self.assertIsNone(obs_id)  # Should not be stored

    def test_store_learning_session(self):
        """Test storing learning session"""
        session = LearningSession(
            session_id="test-session-bridge-001",
            started=datetime.now(timezone.utc),
            ended=datetime.now(timezone.utc),
            iterations=10,
            plugins_used=['vision', 'language'],
            high_salience_count=3,
            convergence_failures=1,
            trust_updates=2,
            discoveries_witnessed=[],
            quality_score=0.85
        )

        session_id = self.bridge.store_learning_session(session)
        self.assertIsNotNone(session_id)

    def test_query_relevant_context(self):
        """Test context query (may return empty if epistemic unavailable)"""
        context = self.bridge.query_relevant_context(
            "attention convergence patterns",
            limit=5
        )

        # Should return structure even if empty
        self.assertIn('similar_episodes', context)
        self.assertIn('relevant_skills', context)
        self.assertIn('known_failures', context)
        self.assertIn('cross_refs', context)

    def test_get_recent_episodes(self):
        """Test getting recent episodes"""
        episodes = self.bridge.get_recent_episodes(limit=10)

        # Should return list (may be empty)
        self.assertIsInstance(episodes, list)

    def test_integration_status_summary(self):
        """Test getting integration status"""
        status = self.bridge.summarize_integration_status()

        self.assertIn('status', status)
        self.assertIn('machine', status)
        self.assertIn('project', status)
        self.assertEqual(status['machine'], 'thor')
        self.assertEqual(status['project'], 'sage-test')

    def test_fallback_mode_storage(self):
        """Test local cache works in fallback mode"""
        # Even if epistemic unavailable, should store locally
        snarc = SNARCScores(0.8, 0.8, 0.7, 0.9, 0.3)
        obs = Observation(
            description="Fallback test observation",
            modality="language",
            snarc_scores=snarc,
            timestamp=datetime.now(timezone.utc)
        )

        obs_id = self.bridge.store_observation(obs)
        self.assertIsNotNone(obs_id)

        # Check local cache has the observation
        if not EPISTEMIC_AVAILABLE:
            self.assertGreater(len(self.bridge.local_cache['observations']), 0)


def run_tests():
    """Run all tests"""
    print("="*70)
    print("SAGE Epistemic Bridge Test Suite")
    print("="*70)
    print(f"Epistemic tools available: {EPISTEMIC_AVAILABLE}")
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSNARCScores))
    suite.addTests(loader.loadTestsFromTestCase(TestObservation))
    suite.addTests(loader.loadTestsFromTestCase(TestLearningSession))
    suite.addTests(loader.loadTestsFromTestCase(TestEpistemicMemoryBridge))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("="*70)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
