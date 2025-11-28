"""
Tests for Federation Challenge System

Validates quality challenge, timeout, progressive penalties,
and integration with federation reputation system.

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-28
Session: Autonomous SAGE Research - ATP-Security Integration
"""

import unittest
import time
from sage.federation.federation_challenge_system import (
    FederationChallengeSystem,
    QualityChallenge,
    EvasionRecord,
    ChallengeStatus,
    EvasionPenaltyLevel
)
from sage.federation.federation_types import (
    FederationIdentity,
    ExecutionProof,
    HardwareSpec,
    create_thor_identity,
    create_sprout_identity
)
from sage.core.mrh_profile import MRHProfile


class TestFederationChallengeSystem(unittest.TestCase):
    """Test suite for federation challenge system"""

    def setUp(self):
        """Setup test fixtures"""
        self.challenge_system = FederationChallengeSystem(
            default_timeout=100.0,  # 100s for testing (instead of 24h)
            re_challenge_cooldown=300.0  # 5min for testing (instead of 7 days)
        )

        # Create test platforms
        self.thor = create_thor_identity()
        self.sprout = create_sprout_identity()

        # Create test execution proof
        self.test_proof = ExecutionProof(
            task_id="test_task_001",
            executing_platform="sprout_lct",
            result_data={"response": "Test response"},
            actual_latency=15.2,
            actual_cost=54.0,
            irp_iterations=3,
            final_energy=0.064,
            convergence_quality=0.85,
            quality_score=0.88  # Claimed quality
        )

    def test_issue_challenge_success(self):
        """Test successful challenge issuance"""
        success, reason, challenge = self.challenge_system.issue_challenge(
            platform_lct_id="sprout_lct",
            challenger_lct_id="thor_lct",
            execution_proof=self.test_proof
        )

        self.assertTrue(success)
        self.assertEqual(reason, "challenge_issued")
        self.assertIsNotNone(challenge)
        self.assertEqual(challenge.platform_lct_id, "sprout_lct")
        self.assertEqual(challenge.challenger_lct_id, "thor_lct")
        self.assertEqual(challenge.claimed_quality, 0.88)
        self.assertEqual(challenge.status, ChallengeStatus.PENDING)

    def test_challenge_cooldown_prevents_spam(self):
        """Test that cooldown period prevents challenge spam"""
        # Issue first challenge
        success1, _, _ = self.challenge_system.issue_challenge(
            platform_lct_id="sprout_lct",
            challenger_lct_id="thor_lct",
            execution_proof=self.test_proof
        )
        self.assertTrue(success1)

        # Try immediate second challenge (should fail - cooldown)
        success2, reason, _ = self.challenge_system.issue_challenge(
            platform_lct_id="sprout_lct",
            challenger_lct_id="thor_lct",
            execution_proof=self.test_proof
        )
        self.assertFalse(success2)
        self.assertIn("cooldown", reason)

    def test_respond_to_challenge_within_timeout(self):
        """Test platform responding to challenge within timeout"""
        # Issue challenge
        _, _, challenge = self.challenge_system.issue_challenge(
            platform_lct_id="sprout_lct",
            challenger_lct_id="thor_lct",
            execution_proof=self.test_proof
        )

        # Respond within timeout
        evidence = {
            "re_execution": True,
            "measured_quality": 0.87,
            "irp_iterations": 3
        }
        success, reason = self.challenge_system.respond_to_challenge(
            challenge_id=challenge.challenge_id,
            evidence=evidence,
            current_time=challenge.issue_timestamp + 50.0  # 50s later (< 100s timeout)
        )

        self.assertTrue(success)
        self.assertEqual(reason, "response_accepted")
        self.assertEqual(challenge.status, ChallengeStatus.RESPONDED)
        self.assertIsNotNone(challenge.response_evidence)

    def test_respond_after_timeout_fails(self):
        """Test that responding after timeout fails"""
        # Issue challenge
        _, _, challenge = self.challenge_system.issue_challenge(
            platform_lct_id="sprout_lct",
            challenger_lct_id="thor_lct",
            execution_proof=self.test_proof
        )

        # Try to respond AFTER timeout
        evidence = {"measured_quality": 0.87}
        success, reason = self.challenge_system.respond_to_challenge(
            challenge_id=challenge.challenge_id,
            evidence=evidence,
            current_time=challenge.issue_timestamp + 150.0  # 150s later (> 100s timeout)
        )

        self.assertFalse(success)
        self.assertEqual(reason, "timeout_expired")
        # Status should still be PENDING (not RESPONDED)
        self.assertEqual(challenge.status, ChallengeStatus.PENDING)

    def test_check_timeouts_marks_evaded(self):
        """Test that timeout check marks challenges as evaded"""
        # Issue challenge
        _, _, challenge = self.challenge_system.issue_challenge(
            platform_lct_id="sprout_lct",
            challenger_lct_id="thor_lct",
            execution_proof=self.test_proof
        )

        # Check timeouts after timeout period
        evaded_challenges = self.challenge_system.check_timeouts(
            current_time=challenge.issue_timestamp + 150.0
        )

        self.assertEqual(len(evaded_challenges), 1)
        self.assertEqual(evaded_challenges[0].challenge_id, challenge.challenge_id)
        self.assertEqual(challenge.status, ChallengeStatus.EVADED)

    def test_progressive_penalties_escalate(self):
        """Test that progressive penalties escalate with strikes"""
        # Create record with different strike counts
        record = EvasionRecord(platform_lct_id="test_platform")

        # Test penalty escalation
        record.strike_count = 0
        self.assertEqual(record.get_penalty_level(), EvasionPenaltyLevel.NONE)

        record.strike_count = 1
        self.assertEqual(record.get_penalty_level(), EvasionPenaltyLevel.WARNING)

        record.strike_count = 2
        self.assertEqual(record.get_penalty_level(), EvasionPenaltyLevel.MODERATE)

        record.strike_count = 3
        self.assertEqual(record.get_penalty_level(), EvasionPenaltyLevel.SEVERE)

        record.strike_count = 4
        self.assertEqual(record.get_penalty_level(), EvasionPenaltyLevel.PERMANENT)

    def test_reputation_decay_applied_correctly(self):
        """Test that reputation decay is applied correctly"""
        # Create platform with high reputation
        platform = create_sprout_identity()
        platform.reputation_score = 0.95

        # Simulate 1 evasion (WARNING level, 5% decay)
        record = self.challenge_system.get_evasion_record(platform.lct_id)
        record.strike_count = 1

        new_rep = self.challenge_system.apply_evasion_penalty(platform)
        expected = 0.95 * (1.0 - 0.05)  # 5% decay
        self.assertAlmostEqual(new_rep, expected, places=4)
        self.assertAlmostEqual(new_rep, 0.9025, places=4)

    def test_multiple_strikes_compound_reputation_loss(self):
        """Test that multiple evasions lead to compounding reputation loss"""
        platform = create_sprout_identity()
        platform.reputation_score = 0.95

        # Strike 1: WARNING (5% decay)
        record = self.challenge_system.get_evasion_record(platform.lct_id)
        record.strike_count = 1
        self.challenge_system.apply_evasion_penalty(platform)
        rep_after_1 = platform.reputation_score
        self.assertAlmostEqual(rep_after_1, 0.9025, places=4)  # 0.95 * 0.95

        # Strike 2: MODERATE (15% decay)
        record.strike_count = 2
        self.challenge_system.apply_evasion_penalty(platform)
        rep_after_2 = platform.reputation_score
        expected = 0.9025 * (1.0 - 0.15)  # 15% decay from previous
        self.assertAlmostEqual(rep_after_2, expected, places=4)
        self.assertAlmostEqual(rep_after_2, 0.7671, places=4)

        # Strike 3: SEVERE (30% decay)
        record.strike_count = 3
        self.challenge_system.apply_evasion_penalty(platform)
        rep_after_3 = platform.reputation_score
        expected = 0.7671 * (1.0 - 0.30)  # 30% decay
        self.assertAlmostEqual(rep_after_3, expected, places=4)
        self.assertAlmostEqual(rep_after_3, 0.5370, places=4)

        # Strike 4: PERMANENT (50% decay)
        record.strike_count = 4
        self.challenge_system.apply_evasion_penalty(platform)
        rep_after_4 = platform.reputation_score
        expected = 0.5370 * (1.0 - 0.50)  # 50% decay
        self.assertAlmostEqual(rep_after_4, expected, places=4)
        self.assertAlmostEqual(rep_after_4, 0.2685, places=4)

    def test_verified_response_updates_quality_average(self):
        """Test that verified responses update average quality"""
        # Issue and respond to challenge
        _, _, challenge = self.challenge_system.issue_challenge(
            platform_lct_id="sprout_lct",
            challenger_lct_id="thor_lct",
            execution_proof=self.test_proof
        )

        evidence = {"measured_quality": 0.85}
        self.challenge_system.respond_to_challenge(
            challenge_id=challenge.challenge_id,
            evidence=evidence
        )

        # Verify response
        self.challenge_system.verify_challenge_response(
            challenge_id=challenge.challenge_id,
            verified_quality=0.85,
            is_valid=True
        )

        # Check evasion record quality tracking
        record = self.challenge_system.get_evasion_record("sprout_lct")
        self.assertGreater(record.total_verifications, 0)
        # Quality should move toward verified quality (0.85) from initial (0.5)
        self.assertGreater(record.average_verified_quality, 0.5)
        self.assertLess(record.average_verified_quality, 0.85)

    def test_platform_challenge_stats(self):
        """Test platform challenge statistics tracking"""
        # Temporarily reduce cooldown for testing
        original_cooldown = self.challenge_system.re_challenge_cooldown
        self.challenge_system.re_challenge_cooldown = 1.0  # 1 second

        # Issue and respond to multiple challenges
        for i in range(3):
            proof = ExecutionProof(
                task_id=f"task_{i}",
                executing_platform="sprout_lct",
                result_data={},
                actual_latency=15.0,
                actual_cost=50.0,
                irp_iterations=3,
                final_energy=0.064,
                convergence_quality=0.85,
                quality_score=0.88
            )

            # Wait to avoid cooldown between challenges
            if i > 0:
                time.sleep(1.1)  # Wait slightly more than cooldown

            success, _, challenge = self.challenge_system.issue_challenge(
                platform_lct_id="sprout_lct",
                challenger_lct_id="thor_lct",
                execution_proof=proof
            )

            self.assertTrue(success, f"Challenge {i} should succeed")

            if i < 2:
                # Respond to first 2 challenges
                current_time = time.time()
                self.challenge_system.respond_to_challenge(
                    challenge_id=challenge.challenge_id,
                    evidence={"quality": 0.85},
                    current_time=current_time
                )
                self.challenge_system.verify_challenge_response(
                    challenge_id=challenge.challenge_id,
                    verified_quality=0.85,
                    is_valid=True
                )
            else:
                # Let 3rd challenge timeout (evade)
                timeout_time = challenge.timeout_timestamp + 10.0
                self.challenge_system.check_timeouts(timeout_time)

        # Restore original cooldown
        self.challenge_system.re_challenge_cooldown = original_cooldown

        # Check stats
        stats = self.challenge_system.get_platform_challenge_stats("sprout_lct")
        self.assertEqual(stats["total_challenges"], 3)
        self.assertEqual(stats["responded"], 2)
        self.assertEqual(stats["evaded"], 1)
        self.assertAlmostEqual(stats["response_rate"], 2/3, places=2)
        self.assertAlmostEqual(stats["evasion_rate"], 1/3, places=2)
        self.assertEqual(stats["strikes"], 1)
        self.assertEqual(stats["penalty_level"], "WARNING")

    def test_system_stats_aggregation(self):
        """Test overall system statistics"""
        # Issue several challenges (each to different platform, no cooldown issues)
        for i in range(5):
            proof = ExecutionProof(
                task_id=f"task_{i}",
                executing_platform=f"platform_{i}",
                result_data={},
                actual_latency=15.0,
                actual_cost=50.0,
                irp_iterations=3,
                final_energy=0.064,
                convergence_quality=0.85,
                quality_score=0.88
            )

            success, _, challenge = self.challenge_system.issue_challenge(
                platform_lct_id=f"platform_{i}",
                challenger_lct_id="challenger",
                execution_proof=proof
            )

            self.assertTrue(success, f"Challenge {i} should succeed")

            if i % 2 == 0:
                # Respond to even-numbered challenges (0, 2, 4)
                current_time = time.time()
                self.challenge_system.respond_to_challenge(
                    challenge_id=challenge.challenge_id,
                    evidence={"quality": 0.85},
                    current_time=current_time
                )

        # Check system stats
        stats = self.challenge_system.get_system_stats()
        self.assertEqual(stats["total_challenges"], 5)
        self.assertEqual(stats["total_responses"], 3)  # 3 responses (even: 0, 2, 4)
        self.assertEqual(stats["platforms_tracked"], 5)


if __name__ == '__main__':
    unittest.main()
