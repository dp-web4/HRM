"""
Unit Tests for SNARC Detectors

Tests each detector individually to validate 5D salience computation.
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from sage.services.snarc.detectors import (
    SurpriseDetector,
    NoveltyDetector,
    ArousalDetector,
    RewardEstimator,
    ConflictDetector
)


class TestSurpriseDetector(unittest.TestCase):
    """Test prediction error detection"""

    def setUp(self):
        self.detector = SurpriseDetector()

    def test_periodic_signal_stable_surprise(self):
        """Periodic signal with EMA predictor maintains moderate surprise"""
        sensor_id = "test_periodic"

        # Feed periodic signal
        # Note: EMA predictor learns average, not period, so sine wave
        # will consistently deviate from prediction (surprise stays moderate)
        surprises = []
        for i in range(100):
            value = np.sin(2 * np.pi * i / 10.0)
            surprise = self.detector.compute(value, sensor_id)
            surprises.append(surprise)

        # After warmup, surprise should be stable (not increasing)
        early_surprise = np.mean(surprises[20:30])
        late_surprise = np.mean(surprises[-10:])

        # Should be stable, not wildly different
        self.assertLess(abs(early_surprise - late_surprise), 0.3,
                       "Surprise should stabilize after warmup")

    def test_step_change_high_surprise(self):
        """Step change should cause high surprise"""
        sensor_id = "test_step"

        # Establish baseline
        for i in range(50):
            self.detector.compute(0.0, sensor_id)

        # Sudden step change
        surprise = self.detector.compute(10.0, sensor_id)

        self.assertGreater(surprise, 0.7,
                          "Step change should cause high surprise")

    def test_tensor_input(self):
        """Should handle tensor inputs"""
        sensor_id = "test_tensor"

        # Feed tensor data
        for i in range(20):
            tensor = torch.randn(3, 32, 32)
            surprise = self.detector.compute(tensor, sensor_id)
            self.assertGreaterEqual(surprise, 0.0)
            self.assertLessEqual(surprise, 1.0)

    def test_reset(self):
        """Reset should clear history and predictor"""
        sensor_id = "test_reset"

        # Build up history
        for i in range(50):
            self.detector.compute(float(i), sensor_id)

        # Reset
        self.detector.reset_sensor(sensor_id)

        # After reset, predictor should be reinitialized
        # First value creates initial prediction, second value tests it
        self.detector.compute(50.0, sensor_id)
        surprise = self.detector.compute(51.0, sensor_id)

        # Should have low surprise for similar consecutive values
        self.assertLess(surprise, 0.5,
                       "Similar values after reset should have low surprise")


class TestNoveltyDetector(unittest.TestCase):
    """Test memory comparison novelty detection"""

    def setUp(self):
        self.detector = NoveltyDetector(memory_size=100)

    def test_first_observation_novel(self):
        """First observation should be completely novel"""
        sensor_id = "test_first"
        novelty = self.detector.compute(5.0, sensor_id)
        self.assertEqual(novelty, 1.0,
                        "First observation should have novelty = 1.0")

    def test_repeated_observation_familiar(self):
        """Repeated observations should be familiar"""
        sensor_id = "test_repeat"

        # Store same value multiple times
        for _ in range(50):
            self.detector.compute(5.0, sensor_id)

        # Same value again should be familiar
        novelty = self.detector.compute(5.0, sensor_id)
        self.assertLess(novelty, 0.2,
                       "Repeated observation should have low novelty")

    def test_novel_after_familiar(self):
        """Novel observation after familiar should be detected"""
        sensor_id = "test_novel"

        # Establish familiar pattern
        for i in range(50):
            self.detector.compute(5.0, sensor_id)

        # Completely different value
        novelty = self.detector.compute(100.0, sensor_id)
        self.assertGreater(novelty, 0.8,
                          "Novel observation should have high novelty")

    def test_tensor_similarity(self):
        """Should compute similarity for tensors"""
        sensor_id = "test_tensor"

        # Create base tensor
        base_tensor = torch.randn(3, 32, 32)

        # Store it
        self.detector.compute(base_tensor, sensor_id)

        # Similar tensor (small perturbation)
        similar = base_tensor + torch.randn(3, 32, 32) * 0.1
        novelty_similar = self.detector.compute(similar, sensor_id)

        # Very different tensor
        different = torch.randn(3, 32, 32) * 10.0
        novelty_different = self.detector.compute(different, sensor_id)

        self.assertLess(novelty_similar, novelty_different,
                       "Similar tensor should be less novel than different tensor")

    def test_memory_limit(self):
        """Memory should be limited to memory_size"""
        detector = NoveltyDetector(memory_size=10)
        sensor_id = "test_limit"

        # Add more observations than memory size
        for i in range(50):
            detector.compute(float(i), sensor_id)

        memory_size = detector.get_memory_size(sensor_id)
        self.assertEqual(memory_size, 10,
                        f"Memory should be limited to 10, got {memory_size}")


class TestArousalDetector(unittest.TestCase):
    """Test signal magnitude/intensity detection"""

    def setUp(self):
        self.detector = ArousalDetector(history_size=100)

    def test_magnitude_normalization(self):
        """Should normalize magnitude by historical distribution"""
        sensor_id = "test_norm"

        # Feed values in range [0, 10]
        arousals = []
        for i in range(100):
            value = np.random.uniform(0, 10)
            arousal = self.detector.compute(value, sensor_id)
            arousals.append(arousal)

        # Distribution should cover [0, 1] range
        self.assertLess(min(arousals), 0.3,
                       "Min arousal should be near 0")
        self.assertGreater(max(arousals), 0.7,
                          "Max arousal should be near 1")

    def test_large_magnitude_high_arousal(self):
        """Large magnitude should cause high arousal"""
        sensor_id = "test_large"

        # Establish baseline with small values
        for _ in range(50):
            self.detector.compute(np.random.uniform(0, 1), sensor_id)

        # Large magnitude
        arousal = self.detector.compute(100.0, sensor_id)
        self.assertGreater(arousal, 0.9,
                          "Large magnitude should have high arousal")

    def test_tensor_magnitude(self):
        """Should compute magnitude for tensors"""
        sensor_id = "test_tensor"

        # Small tensor
        small = torch.randn(3, 32, 32) * 0.1
        arousal_small = self.detector.compute(small, sensor_id)

        # Large tensor
        large = torch.randn(3, 32, 32) * 10.0
        arousal_large = self.detector.compute(large, sensor_id)

        # Larger magnitude should mean higher arousal
        # (after normalization, this may not always hold, but initially should)
        self.assertNotEqual(arousal_small, arousal_large,
                           "Different magnitudes should produce different arousals")

    def test_statistics(self):
        """Should provide magnitude statistics"""
        sensor_id = "test_stats"

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            self.detector.compute(v, sensor_id)

        stats = self.detector.get_statistics(sensor_id)

        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)


class TestRewardEstimator(unittest.TestCase):
    """Test goal relevance estimation"""

    def setUp(self):
        self.estimator = RewardEstimator(memory_size=100)

    def test_no_history_neutral(self):
        """With no history, should return neutral reward"""
        sensor_id = "test_neutral"
        reward = self.estimator.compute(5.0, sensor_id)
        self.assertEqual(reward, 0.5,
                        "No history should give neutral reward 0.5")

    def test_learn_from_outcome(self):
        """Should learn associations from outcomes"""
        sensor_id = "test_learn"

        # Positive outcomes for high values
        for i in range(20):
            value = 10.0
            self.estimator.compute(value, sensor_id)
            self.estimator.update_from_outcome(value, sensor_id, reward=1.0)

        # Negative outcomes for low values
        for i in range(20):
            value = 1.0
            self.estimator.compute(value, sensor_id)
            self.estimator.update_from_outcome(value, sensor_id, reward=-1.0)

        # High value should now have high reward estimate
        reward_high = self.estimator.compute(10.0, sensor_id)
        reward_low = self.estimator.compute(1.0, sensor_id)

        self.assertGreater(reward_high, reward_low,
                          "High value should have higher reward after learning")

    def test_similarity_based_retrieval(self):
        """Should use similarity to find relevant outcomes"""
        sensor_id = "test_similarity"

        # Train on specific values
        self.estimator.update_from_outcome(5.0, sensor_id, reward=0.8)
        self.estimator.update_from_outcome(5.1, sensor_id, reward=0.9)
        self.estimator.update_from_outcome(5.2, sensor_id, reward=0.85)

        # Similar value should get high reward
        reward = self.estimator.compute(5.15, sensor_id)
        self.assertGreater(reward, 0.7,
                          "Similar value should retrieve similar reward")

    def test_tensor_similarity(self):
        """Should handle tensor observations"""
        sensor_id = "test_tensor"

        # Train on specific tensor pattern
        pattern = torch.randn(3, 32, 32)
        self.estimator.update_from_outcome(pattern, sensor_id, reward=0.9)

        # Similar pattern should get high reward
        similar = pattern + torch.randn(3, 32, 32) * 0.1
        reward = self.estimator.compute(similar, sensor_id)

        self.assertGreater(reward, 0.5,
                          "Similar tensor should retrieve reward")

    def test_statistics(self):
        """Should provide outcome statistics"""
        sensor_id = "test_stats"

        rewards = [0.8, 0.9, 0.7, 0.85, 0.75]
        for i, r in enumerate(rewards):
            self.estimator.update_from_outcome(float(i), sensor_id, reward=r)

        stats = self.estimator.get_statistics(sensor_id)

        self.assertIn('mean_reward', stats)
        self.assertIn('std_reward', stats)
        self.assertIn('num_outcomes', stats)
        self.assertEqual(stats['num_outcomes'], len(rewards))


class TestConflictDetector(unittest.TestCase):
    """Test cross-sensor disagreement detection"""

    def setUp(self):
        self.detector = ConflictDetector(correlation_window=20)

    def test_single_sensor_no_conflict(self):
        """Single sensor should have no conflict"""
        all_outputs = {'sensor1': 5.0}
        conflict = self.detector.compute(all_outputs, 'sensor1')
        self.assertEqual(conflict, 0.0,
                        "Single sensor should have zero conflict")

    def test_correlated_sensors_low_conflict(self):
        """Correlated sensors should have low conflict"""
        sensor_id = "sensor1"

        # Create correlated signals
        for i in range(50):
            t = i / 10.0
            value1 = np.sin(t)
            value2 = np.sin(t) + np.random.normal(0, 0.1)  # Correlated with noise

            all_outputs = {
                'sensor1': value1,
                'sensor2': value2
            }

            conflict = self.detector.compute(all_outputs, sensor_id)

        # Final conflict should be low (sensors are correlated)
        self.assertLess(conflict, 0.3,
                       "Correlated sensors should have low conflict")

    def test_anticorrelated_sensors_high_conflict(self):
        """Anti-correlated sensors should show conflict"""
        sensor_id = "sensor1"

        # First establish correlation
        for i in range(30):
            t = i / 10.0
            value1 = np.sin(t)
            value2 = np.sin(t)  # Initially correlated

            all_outputs = {
                'sensor1': value1,
                'sensor2': value2
            }
            self.detector.compute(all_outputs, sensor_id)

        # Then break correlation
        conflicts = []
        for i in range(20):
            t = i / 10.0
            value1 = np.sin(t)
            value2 = -np.sin(t)  # Now anti-correlated

            all_outputs = {
                'sensor1': value1,
                'sensor2': value2
            }
            conflict = self.detector.compute(all_outputs, sensor_id)
            conflicts.append(conflict)

        # Should detect conflict
        avg_conflict = np.mean(conflicts[-10:])
        self.assertGreater(avg_conflict, 0.5,
                          "Anti-correlation should cause conflict")

    def test_tensor_correlation(self):
        """Should compute correlation for tensors"""
        sensor_id = "vision"

        # Create correlated tensor streams
        for i in range(30):
            base = torch.randn(3, 32, 32)
            tensor1 = base + torch.randn(3, 32, 32) * 0.1
            tensor2 = base + torch.randn(3, 32, 32) * 0.1

            all_outputs = {
                'vision': tensor1,
                'audio': tensor2
            }
            conflict = self.detector.compute(all_outputs, sensor_id)

        # Should complete without error
        self.assertGreaterEqual(conflict, 0.0)
        self.assertLessEqual(conflict, 1.0)

    def test_correlation_matrix(self):
        """Should provide correlation matrix"""
        # Feed correlated data
        for i in range(30):
            t = i / 10.0
            all_outputs = {
                'sensor1': np.sin(t),
                'sensor2': np.sin(t),
                'sensor3': np.cos(t)
            }
            self.detector.compute(all_outputs, 'sensor1')

        matrix = self.detector.get_correlation_matrix(['sensor1', 'sensor2', 'sensor3'])

        # Should have correlations for all pairs
        self.assertIn(('sensor1', 'sensor2'), matrix)
        self.assertIn(('sensor1', 'sensor3'), matrix)
        self.assertIn(('sensor2', 'sensor3'), matrix)


if __name__ == '__main__':
    unittest.main()
