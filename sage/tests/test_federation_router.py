"""
Tests for SAGE Federation Router

Tests Phase 1 implementation:
- Delegation decision logic
- Platform capability matching
- Horizon validation
- Reputation tracking

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-28
Session: Autonomous SAGE Research - Federation Readiness
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import time

from sage.federation import (
    FederationRouter,
    FederationTask,
    ExecutionProof,
    QualityRequirements,
    create_thor_identity,
    create_sprout_identity,
    StakeStatus
)
from sage.core.mrh_profile import (
    PROFILE_REFLEXIVE,
    PROFILE_FOCUSED,
    PROFILE_LEARNING,
    PROFILE_CONSOLIDATION
)
from sage.core.attention_manager import MetabolicState


class TestFederationRouter(unittest.TestCase):
    """Test suite for federation router"""

    def setUp(self):
        """Initialize router and platforms for each test"""
        self.thor = create_thor_identity()
        self.sprout = create_sprout_identity()
        self.router = FederationRouter(self.thor)
        self.router.register_platform(self.sprout)

    def test_should_not_delegate_when_local_budget_sufficient(self):
        """Test: Don't delegate if local ATP budget is sufficient"""
        task = FederationTask(
            task_id="task_1",
            task_type="llm_inference",
            task_data={'query': "What is ATP?"},
            estimated_cost=54.0,
            task_horizon=PROFILE_REFLEXIVE,
            complexity="low",
            delegating_platform=self.thor.lct_id,
            delegating_state=MetabolicState.FOCUS,
            quality_requirements=QualityRequirements(),
            max_latency=20.0,
            deadline=time.time() + 20.0
        )

        should_delegate, reason = self.router.should_delegate(task, local_budget=75.0)

        self.assertFalse(should_delegate)
        self.assertEqual(reason, "sufficient_local_atp")

    def test_should_delegate_when_local_budget_insufficient(self):
        """Test: Delegate if local ATP budget insufficient but platforms available"""
        # Add more platforms for witness diversity
        # (In Phase 1, we need ≥3 total platforms for witnesses)
        platform3 = create_thor_identity()  # Create another Thor-like platform
        platform3.lct_id = "platform3_lct"
        platform3.platform_name = "Platform3"
        self.router.register_platform(platform3)

        platform4 = create_thor_identity()
        platform4.lct_id = "platform4_lct"
        platform4.platform_name = "Platform4"
        self.router.register_platform(platform4)

        # Create task Thor can delegate (REFLEXIVE, simple)
        task = FederationTask(
            task_id="task_2",
            task_type="llm_inference",
            task_data={'query': "Simple query"},
            estimated_cost=90.0,
            task_horizon=PROFILE_REFLEXIVE,  # Simple horizon platforms can handle
            complexity="medium",
            delegating_platform=self.thor.lct_id,
            delegating_state=MetabolicState.WAKE,
            quality_requirements=QualityRequirements(),
            max_latency=30.0,
            deadline=time.time() + 30.0
        )

        should_delegate, reason = self.router.should_delegate(task, local_budget=30.0)

        # Should delegate (insufficient local ATP, capable platforms available, witnesses available)
        self.assertTrue(should_delegate, f"Should delegate but got: {reason}")
        self.assertIn("federation_routing", reason)

    def test_horizon_capability_matching(self):
        """Test: Platform horizon capability correctly evaluated"""
        # Thor can handle large horizons
        thor_can_learning = self.router._can_handle_horizon(self.thor, PROFILE_LEARNING)
        self.assertTrue(thor_can_learning, "Thor should handle LEARNING horizon")

        # Sprout cannot (max is LOCAL/SESSION/AGENT_SCALE)
        sprout_can_learning = self.router._can_handle_horizon(self.sprout, PROFILE_LEARNING)
        self.assertFalse(sprout_can_learning, "Sprout should NOT handle LEARNING horizon")

        # Sprout can handle smaller horizons
        sprout_can_reflexive = self.router._can_handle_horizon(self.sprout, PROFILE_REFLEXIVE)
        self.assertTrue(sprout_can_reflexive, "Sprout should handle REFLEXIVE horizon")

    def test_find_capable_platforms_filters_correctly(self):
        """Test: find_capable_platforms filters by all criteria"""
        # Task that Sprout cannot handle (LEARNING horizon)
        task = FederationTask(
            task_id="task_3",
            task_type="consolidation",
            task_data={'sessions': list(range(20))},
            estimated_cost=1200.0,
            task_horizon=PROFILE_LEARNING,
            complexity="high",
            delegating_platform=self.thor.lct_id,
            delegating_state=MetabolicState.DREAM,
            quality_requirements=QualityRequirements(),
            max_latency=600.0,
            deadline=time.time() + 600.0
        )

        candidates = self.router.find_capable_platforms(task)

        # Sprout should be filtered out (can't handle LEARNING horizon)
        self.assertEqual(len(candidates), 0, "No platforms should be capable")

    def test_slashed_platform_excluded(self):
        """Test: Platforms with slashed stakes cannot receive tasks"""
        # Slash Sprout's stake
        self.sprout.stake.slash("test_malicious_behavior", slash_percentage=1.0)

        # Task Sprout could otherwise handle
        task = FederationTask(
            task_id="task_4",
            task_type="llm_inference",
            task_data={'query': "Simple query"},
            estimated_cost=50.0,
            task_horizon=PROFILE_REFLEXIVE,
            complexity="low",
            delegating_platform=self.thor.lct_id,
            delegating_state=MetabolicState.FOCUS,
            quality_requirements=QualityRequirements(),
            max_latency=20.0,
            deadline=time.time() + 20.0
        )

        candidates = self.router.find_capable_platforms(task)

        # Sprout should be excluded (slashed)
        self.assertEqual(len(candidates), 0, "Slashed platform should be excluded")
        self.assertEqual(self.sprout.stake.status, StakeStatus.SLASHED)

    def test_execution_proof_validation(self):
        """Test: Execution proof validation checks quality and latency"""
        task = FederationTask(
            task_id="task_5",
            task_type="llm_inference",
            task_data={'query': "Test query"},
            estimated_cost=50.0,
            task_horizon=PROFILE_REFLEXIVE,
            complexity="low",
            delegating_platform=self.thor.lct_id,
            delegating_state=MetabolicState.FOCUS,
            quality_requirements=QualityRequirements(min_quality=0.7, min_convergence=0.6),
            max_latency=20.0,
            deadline=time.time() + 20.0
        )

        # Good proof
        good_proof = ExecutionProof(
            task_id=task.task_id,
            executing_platform=self.sprout.lct_id,
            result_data={'response': "Good response"},
            actual_latency=15.0,
            actual_cost=50.0,
            irp_iterations=3,
            final_energy=0.5,
            convergence_quality=0.8,
            quality_score=0.85
        )

        valid, reason = self.router.validate_execution_proof(good_proof, task)
        self.assertTrue(valid, f"Good proof should be valid: {reason}")

        # Bad proof (quality too low)
        bad_proof = ExecutionProof(
            task_id=task.task_id,
            executing_platform=self.sprout.lct_id,
            result_data={'response': "Poor response"},
            actual_latency=15.0,
            actual_cost=50.0,
            irp_iterations=3,
            final_energy=0.8,
            convergence_quality=0.4,
            quality_score=0.5
        )

        valid, reason = self.router.validate_execution_proof(bad_proof, task)
        self.assertFalse(valid, "Low quality proof should be rejected")
        self.assertIn("quality", reason.lower())

    def test_reputation_update(self):
        """Test: Platform reputation updates based on execution quality"""
        initial_reputation = self.sprout.reputation_score

        # Record high-quality execution
        self.router.update_platform_reputation(self.sprout.lct_id, execution_quality=0.95)

        # Reputation should increase
        self.assertGreater(self.sprout.reputation_score, initial_reputation)

        # Record low-quality execution
        self.router.update_platform_reputation(self.sprout.lct_id, execution_quality=0.3)

        # Reputation should decrease
        self.assertLess(self.sprout.reputation_score, 0.75)  # Started at 0.75

    def test_federation_stats(self):
        """Test: Federation statistics tracking"""
        stats = self.router.get_federation_stats()

        self.assertEqual(stats['local_platform'], 'Thor')
        self.assertEqual(stats['known_platforms'], 1)
        self.assertIn('Sprout', [p['name'] for p in stats['platforms'].values()])


def run_tests():
    """Run all tests and display results"""
    print("\n" + "=" * 80)
    print("  Federation Router Test Suite")
    print("  Thor SAGE Session - November 28, 2025")
    print("=" * 80)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFederationRouter)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"  ✓ ALL TESTS PASSED ({result.testsRun} tests)")
        print("\n  Federation Router Phase 1: VALIDATED")
        print("  - Delegation decision logic working ✓")
        print("  - Capability matching working ✓")
        print("  - Horizon validation working ✓")
        print("  - Reputation tracking working ✓")
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
