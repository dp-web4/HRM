"""
Tests for Complete ATP Framework Integration

Tests the integration of:
1. Multi-modal ATP pricing
2. MRH-aware attention allocation
3. Metabolic state transitions
4. Resource decision making

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-27
Session: Complete ATP Framework Integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
from sage.core.multimodal_atp_pricing import MultiModalATPPricer
from sage.core.mrh_aware_attention import MRHAwareAttentionManager
from sage.core.mrh_profile import (
    PROFILE_REFLEXIVE,
    PROFILE_FOCUSED,
    PROFILE_LEARNING,
    PROFILE_CONSOLIDATION,
    PROFILE_CRISIS_COORDINATION
)
from sage.core.attention_manager import MetabolicState


class TestATPFrameworkIntegration(unittest.TestCase):
    """Test suite for complete ATP framework integration"""

    def setUp(self):
        """Initialize components for each test"""
        self.pricer = MultiModalATPPricer()
        self.manager = MRHAwareAttentionManager(total_atp=100.0)

    def test_scenario_1_quick_factual_query(self):
        """
        Test Scenario 1: Quick factual query with state transition

        Expected: WAKE insufficient → transition to FOCUS → execute
        """
        # Initial state
        self.manager.current_state = MetabolicState.WAKE
        horizon = PROFILE_REFLEXIVE

        # Calculate cost
        cost = self.pricer.calculate_cost(
            task_type='llm_inference',
            complexity='low',
            latency=5.0,
            quality=0.95
        )

        # Check initial budget insufficient
        initial_budget = self.manager.get_total_allocated_atp(horizon)
        self.assertGreater(cost, initial_budget,
                          "Quick query should exceed WAKE budget")

        # Transition to FOCUS
        self.manager.current_state = MetabolicState.FOCUS
        focus_budget = self.manager.get_total_allocated_atp(horizon)

        # Check FOCUS budget sufficient
        self.assertLess(cost, focus_budget,
                       "Quick query should be affordable in FOCUS")

        # Verify specific values match design doc
        self.assertAlmostEqual(cost, 24.5, delta=1.0,
                              msg="Cost should be ~24.5 ATP")
        self.assertAlmostEqual(initial_budget, 6.8, delta=1.0,
                              msg="WAKE+REFLEXIVE budget should be ~6.8 ATP")
        self.assertAlmostEqual(focus_budget, 68.0, delta=5.0,
                              msg="FOCUS+REFLEXIVE budget should be ~68.0 ATP")

    def test_scenario_2_complex_reasoning(self):
        """
        Test Scenario 2: Complex reasoning at budget boundary

        Expected: Slightly over budget but within tolerance
        """
        # State: FOCUS
        self.manager.current_state = MetabolicState.FOCUS
        horizon = PROFILE_FOCUSED

        # Calculate cost
        cost = self.pricer.calculate_cost(
            task_type='llm_inference',
            complexity='high',
            latency=30.0,
            quality=0.85
        )

        # Get budget
        budget = self.manager.get_total_allocated_atp(horizon)

        # Check slightly over budget but within 15% tolerance
        self.assertGreater(cost, budget, "Should be slightly over budget")
        tolerance = budget * 1.15
        self.assertLess(cost, tolerance, "Should be within 15% tolerance")

        # Verify specific values
        self.assertAlmostEqual(cost, 88.5, delta=1.0,
                              msg="Cost should be ~88.5 ATP")
        self.assertAlmostEqual(budget, 80.0, delta=5.0,
                              msg="FOCUS+FOCUSED budget should be ~80.0 ATP")

    def test_scenario_3_cross_session_learning(self):
        """
        Test Scenario 3: Cross-session learning deferred

        Expected: Vastly over budget → defer to background
        """
        # State: DREAM
        self.manager.current_state = MetabolicState.DREAM
        horizon = PROFILE_LEARNING

        # Calculate cost
        cost = self.pricer.calculate_cost(
            task_type='consolidation',
            complexity='high',
            latency=10.0,  # minutes
            quality=0.90
        )

        # Get budget
        budget = self.manager.get_total_allocated_atp(horizon)

        # Check vastly over budget (40× or more)
        over_budget_ratio = cost / budget
        self.assertGreater(over_budget_ratio, 30.0,
                          "Should be vastly over budget (30×+)")

        # Verify specific values
        self.assertAlmostEqual(cost, 1145.0, delta=50.0,
                              msg="Cost should be ~1,145 ATP")
        self.assertAlmostEqual(budget, 27.8, delta=3.0,
                              msg="DREAM+LEARNING budget should be ~27.8 ATP")

    def test_scenario_4_emergency_coordination(self):
        """
        Test Scenario 4: Emergency coordination with CRISIS override

        Expected: Over budget but CRISIS can exceed 100% (adrenaline)
        """
        # State: CRISIS
        self.manager.current_state = MetabolicState.CRISIS
        horizon = PROFILE_CRISIS_COORDINATION

        # Calculate cost
        cost = self.pricer.calculate_cost(
            task_type='coordination',
            complexity='critical',
            latency=60.0,
            quality=0.95
        )

        # Get budget (CRISIS can exceed total_atp)
        budget = self.manager.get_total_allocated_atp(horizon)

        # Check budget exceeds 100 ATP (adrenaline override)
        self.assertGreater(budget, 100.0,
                          "CRISIS budget should exceed total ATP pool")

        # Still over budget but crisis executes anyway
        self.assertGreater(cost, budget,
                          "Emergency task should still exceed CRISIS budget")

        # Verify specific values
        self.assertAlmostEqual(cost, 1139.0, delta=50.0,
                              msg="Cost should be ~1,139 ATP")
        self.assertAlmostEqual(budget, 134.0, delta=10.0,
                              msg="CRISIS+CRISIS_COORD budget should be ~134.0 ATP")

    def test_multi_modal_pricing_consistency(self):
        """Test multi-modal pricing is consistent with standalone tests"""
        # Test vision pricing
        vision_cost = self.pricer.calculate_cost(
            task_type='vision',
            complexity='medium',
            latency=0.05,  # 50ms
            quality=0.85
        )
        self.assertLess(vision_cost, 100.0,
                       "Vision tasks should be affordable")

        # Test LLM pricing
        llm_cost = self.pricer.calculate_cost(
            task_type='llm_inference',
            complexity='medium',
            latency=20.0,  # 20s
            quality=0.85
        )
        self.assertLess(llm_cost, 100.0,
                       "LLM tasks should be economically viable")

        # Vision should be cheaper than LLM for similar quality
        vision_cost_comparable = self.pricer.calculate_cost(
            task_type='vision',
            complexity='medium',
            latency=0.02,  # 20ms
            quality=0.85
        )
        self.assertLess(vision_cost_comparable, llm_cost,
                       "Vision should be cheaper than LLM per result")

    def test_mrh_aware_budget_scaling(self):
        """Test MRH-aware budget scaling across horizons"""
        self.manager.current_state = MetabolicState.FOCUS

        # Get budgets for different horizons
        reflexive_budget = self.manager.get_total_allocated_atp(PROFILE_REFLEXIVE)
        focused_budget = self.manager.get_total_allocated_atp(PROFILE_FOCUSED)
        learning_budget = self.manager.get_total_allocated_atp(PROFILE_LEARNING)

        # Reflexive should be lowest (0.85× scaling)
        self.assertLess(reflexive_budget, focused_budget,
                       "Reflexive budget should be less than focused")

        # Learning should be highest (but capped at total_atp)
        self.assertGreater(learning_budget, focused_budget,
                          "Learning budget should be more than focused")

        # Verify scaling factors are applied
        self.assertAlmostEqual(reflexive_budget / focused_budget, 0.85, delta=0.05,
                              msg="Reflexive should be ~0.85× focused")

        # Learning is capped at 100 ATP, so ratio is 1.25 instead of theoretical 1.39
        # (80 * 1.39 = 111.2, but capped at 100)
        self.assertAlmostEqual(learning_budget / focused_budget, 1.25, delta=0.05,
                              msg="Learning should be ~1.25× focused (capped at total_atp)")

    def test_metabolic_state_transitions(self):
        """Test budget changes with metabolic state transitions"""
        horizon = PROFILE_FOCUSED

        # Test each state
        states_and_budgets = [
            (MetabolicState.WAKE, 8.0),    # 8% of 100
            (MetabolicState.FOCUS, 80.0),  # 80% of 100
            (MetabolicState.REST, 40.0),   # 40% of 100
            (MetabolicState.DREAM, 20.0),  # 20% of 100
            (MetabolicState.CRISIS, 95.0)  # 95% of 100
        ]

        for state, expected_budget in states_and_budgets:
            self.manager.current_state = state
            budget = self.manager.get_total_allocated_atp(horizon)
            self.assertAlmostEqual(budget, expected_budget, delta=5.0,
                                  msg=f"{state.value} should have ~{expected_budget} ATP")

    def test_crisis_can_exceed_total_atp(self):
        """Test CRISIS state can exceed total ATP pool (adrenaline override)"""
        self.manager.current_state = MetabolicState.CRISIS
        horizon = PROFILE_CRISIS_COORDINATION

        budget = self.manager.get_total_allocated_atp(horizon)

        # CRISIS budget should exceed total ATP
        self.assertGreater(budget, self.manager.total_atp,
                          "CRISIS should be able to exceed total ATP pool")

        # Other states should not
        self.manager.current_state = MetabolicState.FOCUS
        focus_budget = self.manager.get_total_allocated_atp(horizon)
        self.assertLessEqual(focus_budget, self.manager.total_atp,
                           "FOCUS should not exceed total ATP pool")

    def test_resource_decision_logic(self):
        """Test resource decision logic across scenarios"""
        test_cases = [
            # (state, horizon, task_type, complexity, latency, quality, should_execute)
            (MetabolicState.FOCUS, PROFILE_FOCUSED, 'llm_inference', 'low', 5.0, 0.95, True),
            (MetabolicState.WAKE, PROFILE_REFLEXIVE, 'llm_inference', 'low', 5.0, 0.95, False),
            (MetabolicState.DREAM, PROFILE_LEARNING, 'consolidation', 'high', 10.0, 0.90, False),
        ]

        for state, horizon, task_type, complexity, latency, quality, should_execute in test_cases:
            self.manager.current_state = state
            cost = self.pricer.calculate_cost(task_type, complexity, latency, quality)
            budget = self.manager.get_total_allocated_atp(horizon)

            if should_execute:
                self.assertLessEqual(cost, budget,
                                    f"{state.value} + {horizon} should afford {task_type}")
            else:
                self.assertGreater(cost, budget,
                                  f"{state.value} + {horizon} should not afford {task_type}")

    def test_biological_validation(self):
        """Test ATP allocations match biological timescales"""
        # Amygdala (startle): milliseconds
        self.manager.current_state = MetabolicState.WAKE
        amygdala_budget = self.manager.get_total_allocated_atp(PROFILE_REFLEXIVE)
        self.assertAlmostEqual(amygdala_budget, 6.8, delta=1.0,
                              msg="Amygdala-like responses should use ~6.8 ATP")

        # PFC (reasoning): seconds-minutes
        self.manager.current_state = MetabolicState.FOCUS
        pfc_budget = self.manager.get_total_allocated_atp(PROFILE_FOCUSED)
        self.assertAlmostEqual(pfc_budget, 80.0, delta=5.0,
                              msg="PFC-like reasoning should use ~80.0 ATP")

        # Hippocampus (learning): hours-days
        self.manager.current_state = MetabolicState.DREAM
        hippocampus_budget = self.manager.get_total_allocated_atp(PROFILE_LEARNING)
        self.assertAlmostEqual(hippocampus_budget, 27.8, delta=3.0,
                              msg="Hippocampus-like learning should use ~27.8 ATP")

        # Adrenaline (emergency): override
        self.manager.current_state = MetabolicState.CRISIS
        adrenaline_budget = self.manager.get_total_allocated_atp(PROFILE_CRISIS_COORDINATION)
        self.assertAlmostEqual(adrenaline_budget, 134.0, delta=10.0,
                              msg="Adrenaline-like crisis should use ~134.0 ATP")
        self.assertGreater(adrenaline_budget, 100.0,
                          "Adrenaline should exceed normal ATP pool")


def run_tests():
    """Run all tests and display results"""
    print("\n" + "=" * 80)
    print("  ATP Framework Integration Test Suite")
    print("  Thor SAGE Session - November 27, 2025")
    print("=" * 80)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestATPFrameworkIntegration)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"  ✓ ALL TESTS PASSED ({result.testsRun} tests)")
        print("\n  Complete ATP Framework Integration: VALIDATED")
        print("  - Multi-modal pricing working ✓")
        print("  - MRH-aware budgets working ✓")
        print("  - State transitions working ✓")
        print("  - Biological validation passed ✓")
    else:
        print(f"  ✗ SOME TESTS FAILED")
        print(f"    Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"    Failed: {len(result.failures)}")
        print(f"    Errors: {len(result.errors)}")
    print("=" * 80)
    print()

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
