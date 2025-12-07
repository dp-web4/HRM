"""
Test Suite for Threshold-Based Attention Decisions
===================================================

Validates compression-action-threshold pattern implementation:
- Layer 3: Metabolic-state-dependent threshold computation
- Layer 4: Binary attention decision logic

Author: Claude (Sonnet 4.5)
Date: 2025-12-07
"""

import unittest
from threshold_decision import (
    MetabolicState,
    get_attention_threshold,
    make_attention_decision,
    compute_threshold_grid,
    demonstrate_mrh_dependent_threshold
)


class TestThresholdComputation(unittest.TestCase):
    """Test get_attention_threshold() function"""

    def test_base_thresholds(self):
        """Test base thresholds for each metabolic state (nominal conditions)"""
        atp = 1.0  # Full ATP
        crit = 0.0  # No criticality

        # Should get base values (no modulation)
        self.assertAlmostEqual(
            get_attention_threshold(MetabolicState.WAKE, atp, crit),
            0.5, places=2
        )
        self.assertAlmostEqual(
            get_attention_threshold(MetabolicState.FOCUS, atp, crit),
            0.3, places=2
        )
        self.assertAlmostEqual(
            get_attention_threshold(MetabolicState.REST, atp, crit),
            0.8, places=2
        )
        self.assertAlmostEqual(
            get_attention_threshold(MetabolicState.DREAM, atp, crit),
            0.1, places=2
        )
        self.assertAlmostEqual(
            get_attention_threshold(MetabolicState.CRISIS, atp, crit),
            0.9, places=2
        )

    def test_atp_modulation_raises_threshold(self):
        """Low ATP should raise threshold (conserve energy)"""
        base_threshold = get_attention_threshold(
            MetabolicState.WAKE, atp_remaining=1.0, task_criticality=0.0
        )
        low_atp_threshold = get_attention_threshold(
            MetabolicState.WAKE, atp_remaining=0.0, task_criticality=0.0
        )

        # Low ATP → higher threshold (more selective)
        self.assertGreater(low_atp_threshold, base_threshold)
        self.assertAlmostEqual(low_atp_threshold - base_threshold, 0.2, places=2)

    def test_criticality_modulation_lowers_threshold(self):
        """High criticality should lower threshold (don't miss important signals)"""
        base_threshold = get_attention_threshold(
            MetabolicState.WAKE, atp_remaining=1.0, task_criticality=0.0
        )
        high_crit_threshold = get_attention_threshold(
            MetabolicState.WAKE, atp_remaining=1.0, task_criticality=1.0
        )

        # High criticality → lower threshold (less selective)
        self.assertLess(high_crit_threshold, base_threshold)
        self.assertAlmostEqual(base_threshold - high_crit_threshold, 0.1, places=2)

    def test_threshold_clamping(self):
        """Thresholds should be clamped to [0, 1]"""
        # Try to push below 0
        threshold_min = get_attention_threshold(
            MetabolicState.DREAM,  # Base 0.1
            atp_remaining=1.0,      # -0.0 modulation
            task_criticality=1.0    # -0.1 modulation
        )
        self.assertGreaterEqual(threshold_min, 0.0)
        self.assertEqual(threshold_min, 0.0)  # 0.1 - 0.1 = 0.0

        # Try to push above 1
        threshold_max = get_attention_threshold(
            MetabolicState.CRISIS,  # Base 0.9
            atp_remaining=0.0,      # +0.2 modulation
            task_criticality=0.0    # -0.0 modulation
        )
        self.assertLessEqual(threshold_max, 1.0)
        # 0.9 + 0.2 = 1.1, clamped to 1.0

    def test_expected_thresholds_from_design_doc(self):
        """Validate examples from ATTENTION_COMPRESSION_DESIGN.md"""
        # From design doc table:
        test_cases = [
            (MetabolicState.WAKE, 0.8, 0.5, 0.49),
            (MetabolicState.FOCUS, 0.6, 0.7, 0.31),
            (MetabolicState.REST, 0.9, 0.1, 0.81),
            (MetabolicState.CRISIS, 0.3, 0.9, 0.95),
            (MetabolicState.DREAM, 0.7, 0.0, 0.16),
        ]

        for state, atp, crit, expected in test_cases:
            threshold = get_attention_threshold(state, atp, crit)
            self.assertAlmostEqual(threshold, expected, places=2,
                msg=f"{state.value}: expected {expected}, got {threshold}")


class TestAttentionDecision(unittest.TestCase):
    """Test make_attention_decision() function"""

    def test_attend_when_above_threshold_and_sufficient_atp(self):
        """Should ATTEND when salience > threshold and ATP sufficient"""
        decision = make_attention_decision(
            salience=0.7,
            threshold=0.5,
            plugin_name="test",
            atp_cost=10.0,
            atp_budget=50.0
        )

        self.assertTrue(decision.should_attend)
        self.assertIn("Salience 0.70 > threshold 0.50", decision.reason)
        self.assertEqual(decision.salience, 0.7)
        self.assertEqual(decision.threshold, 0.5)

    def test_ignore_when_below_threshold(self):
        """Should IGNORE when salience <= threshold (even if ATP sufficient)"""
        decision = make_attention_decision(
            salience=0.4,
            threshold=0.5,
            plugin_name="test",
            atp_cost=10.0,
            atp_budget=50.0
        )

        self.assertFalse(decision.should_attend)
        self.assertIn("below threshold", decision.reason)

    def test_ignore_when_insufficient_atp(self):
        """Should IGNORE when ATP insufficient (even if salience high)"""
        decision = make_attention_decision(
            salience=0.9,
            threshold=0.5,
            plugin_name="test",
            atp_cost=60.0,
            atp_budget=50.0
        )

        self.assertFalse(decision.should_attend)
        self.assertIn("Insufficient ATP", decision.reason)

    def test_edge_case_salience_equals_threshold(self):
        """Edge case: salience == threshold should IGNORE"""
        decision = make_attention_decision(
            salience=0.5,
            threshold=0.5,
            plugin_name="test",
            atp_cost=10.0,
            atp_budget=50.0
        )

        self.assertFalse(decision.should_attend)

    def test_edge_case_atp_equals_cost(self):
        """Edge case: ATP exactly equals cost should ATTEND"""
        decision = make_attention_decision(
            salience=0.7,
            threshold=0.5,
            plugin_name="test",
            atp_cost=50.0,
            atp_budget=50.0
        )

        self.assertTrue(decision.should_attend)

    def test_decision_to_dict(self):
        """Test AttentionDecision serialization"""
        decision = make_attention_decision(
            salience=0.7,
            threshold=0.5,
            plugin_name="test",
            atp_cost=10.0,
            atp_budget=50.0
        )

        d = decision.to_dict()

        self.assertIsInstance(d, dict)
        self.assertIn('should_attend', d)
        self.assertIn('reason', d)
        self.assertIn('salience', d)
        self.assertIn('threshold', d)
        self.assertTrue(d['should_attend'])


class TestMRHDependentThreshold(unittest.TestCase):
    """Test context-dependent threshold behavior"""

    def test_same_salience_different_decisions(self):
        """Same salience should trigger different decisions in different states"""
        salience = 0.6
        atp = 0.8
        crit = 0.5

        # WAKE: should ATTEND (threshold ~0.49)
        threshold_wake = get_attention_threshold(MetabolicState.WAKE, atp, crit)
        decision_wake = make_attention_decision(salience, threshold_wake, "test", 10, 50)
        self.assertTrue(decision_wake.should_attend)

        # REST: should IGNORE (threshold ~0.81)
        threshold_rest = get_attention_threshold(MetabolicState.REST, atp, crit)
        decision_rest = make_attention_decision(salience, threshold_rest, "test", 10, 50)
        self.assertFalse(decision_rest.should_attend)

        # This is the MRH-dependent threshold!
        # Same observation (salience 0.6) has different meaning in different contexts

    def test_demonstrate_function(self):
        """Test demonstrate_mrh_dependent_threshold() function"""
        examples = demonstrate_mrh_dependent_threshold()

        self.assertEqual(len(examples), 5)  # 5 metabolic states

        # Check that decisions vary
        decisions = [ex['decision'] for ex in examples]
        self.assertIn('ATTEND', decisions)
        self.assertIn('IGNORE', decisions)


class TestThresholdGrid(unittest.TestCase):
    """Test compute_threshold_grid() utility"""

    def test_grid_computation(self):
        """Test threshold grid computation"""
        grid = compute_threshold_grid(
            states=[MetabolicState.WAKE, MetabolicState.FOCUS],
            atp_range=(0.0, 1.0),
            criticality_range=(0.0, 1.0),
            steps=3
        )

        # Should have entries for both states
        self.assertIn('wake', grid)
        self.assertIn('focus', grid)

        # Each state should have 3x3 = 9 samples
        self.assertEqual(len(grid['wake']), 9)
        self.assertEqual(len(grid['focus']), 9)

        # Check structure
        sample = grid['wake'][0]
        self.assertIn('atp', sample)
        self.assertIn('criticality', sample)
        self.assertIn('threshold', sample)

        # Thresholds should be in [0, 1]
        for state_grid in grid.values():
            for sample in state_grid:
                self.assertGreaterEqual(sample['threshold'], 0.0)
                self.assertLessEqual(sample['threshold'], 1.0)


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios"""

    def test_normal_operation_wake_state(self):
        """Normal operation in WAKE state"""
        # Moderate salience, good ATP, moderate criticality
        salience = 0.6
        atp_remaining = 0.8
        criticality = 0.5

        threshold = get_attention_threshold(
            MetabolicState.WAKE, atp_remaining, criticality
        )
        decision = make_attention_decision(
            salience, threshold, "vision", atp_cost=10.0, atp_budget=80.0
        )

        # Should attend (salience 0.6 > threshold ~0.49)
        self.assertTrue(decision.should_attend)

    def test_crisis_mode_selective_attention(self):
        """Crisis mode with low ATP - very selective"""
        # Same salience as above, but crisis mode + low ATP
        salience = 0.6
        atp_remaining = 0.2
        criticality = 0.9

        threshold = get_attention_threshold(
            MetabolicState.CRISIS, atp_remaining, criticality
        )
        decision = make_attention_decision(
            salience, threshold, "vision", atp_cost=10.0, atp_budget=20.0
        )

        # Should ignore (salience 0.6 < threshold ~0.99)
        self.assertFalse(decision.should_attend)
        self.assertIn("below threshold", decision.reason)

    def test_dream_mode_exploration(self):
        """Dream mode with high ATP - explore widely"""
        # Even low salience should be attended in dream mode
        salience = 0.3
        atp_remaining = 0.9
        criticality = 0.0

        threshold = get_attention_threshold(
            MetabolicState.DREAM, atp_remaining, criticality
        )
        decision = make_attention_decision(
            salience, threshold, "exploration", atp_cost=5.0, atp_budget=90.0
        )

        # Should attend (salience 0.3 > threshold ~0.11)
        self.assertTrue(decision.should_attend)

    def test_focus_mode_detailed_attention(self):
        """Focus mode attends to fine details"""
        # Medium-low salience that would be ignored in WAKE
        salience = 0.4
        atp_remaining = 0.7
        criticality = 0.6

        threshold = get_attention_threshold(
            MetabolicState.FOCUS, atp_remaining, criticality
        )
        decision = make_attention_decision(
            salience, threshold, "detail", atp_cost=15.0, atp_budget=70.0
        )

        # Should attend (salience 0.4 > threshold ~0.27)
        self.assertTrue(decision.should_attend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
