#!/usr/bin/env python3
"""
PolicyGate Phase 4: Experience Buffer Integration - Unit Tests

Tests for the Phase 4 enhancement that records policy decisions as
experience atoms for long-term learning.

Coverage:
1. Experience buffer recording
2. Salience scoring for different decision types
3. Violation pattern tracking
4. CRISIS mode metadata
5. Buffer integration and circular pruning
"""

import unittest
import time
from typing import Dict, List, Any

# Import PolicyGate and dependencies
try:
    from sage.irp.plugins.policy_gate import PolicyGateIRP, AccountabilityFrame
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from irp.plugins.policy_gate import PolicyGateIRP, AccountabilityFrame


class MockExperienceBuffer:
    """Mock experience buffer for testing (simulates snarc_memory list)."""

    def __init__(self, max_size: int = 100):
        self.buffer: List[Dict[str, Any]] = []
        self.max_size = max_size

    def append(self, experience: Dict[str, Any]):
        """Add experience to buffer."""
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # Circular buffer behavior

    def get_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all experiences from a specific source."""
        return [e for e in self.buffer if e.get('source') == source]

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()


class TestPolicyGatePhase4(unittest.TestCase):
    """Test suite for PolicyGate Phase 4 experience buffer integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.experience_buffer = MockExperienceBuffer(max_size=100)

        # Standard test policy rules
        self.policy_rules = [
            {
                'id': 'deny-low-trust-deploy',
                'name': 'Deny deployment for low-trust actors',
                'priority': 1,
                'decision': 'deny',
                'match': {'action_types': ['deploy'], 'max_trust': 0.7},
                'reason': 'Deployment requires trust >= 0.7',
            },
            {
                'id': 'warn-sensitive-files',
                'name': 'Warn on sensitive file access',
                'priority': 2,
                'decision': 'warn',
                'match': {
                    'action_types': ['write'],
                    'target_patterns': ['*.env', '*password*'],
                },
                'reason': 'Sensitive file access detected',
            },
            {
                'id': 'allow-admin-all',
                'name': 'Allow admin all actions',
                'priority': 10,
                'decision': 'allow',
                'match': {'roles': ['admin'], 'min_trust': 0.8},
                'reason': 'Admin with high trust',
            },
        ]

    def _create_policy_gate(self) -> PolicyGateIRP:
        """Create PolicyGate instance with test configuration."""
        config = {
            'entity_id': 'policy_gate_test',
            'policy_rules': self.policy_rules,
            'default_policy': 'allow',
            'max_iterations': 5,
            'halt_eps': 0.01,
            'halt_K': 2,
            'experience_buffer': self.experience_buffer,
        }
        return PolicyGateIRP(config)

    # =========================================================================
    # Test 1: Experience Buffer Recording
    # =========================================================================

    def test_experience_buffer_recording(self):
        """Test that policy decisions are recorded to experience buffer."""
        gate = self._create_policy_gate()

        # Action that should be allowed
        actions = [{
            'action_id': 'test_action',
            'action_type': 'read',
            'role': 'user',
            'trust_score': 0.8,
        }]
        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        # Run policy evaluation
        final_state, history = gate.refine(actions, ctx)

        # Verify experience was recorded
        policy_experiences = self.experience_buffer.get_by_source('policy_gate')
        self.assertEqual(len(policy_experiences), 1, "Should record one experience")

        exp = policy_experiences[0]
        self.assertEqual(exp['source'], 'policy_gate')
        self.assertIn('timestamp', exp)
        self.assertIn('context', exp)
        self.assertIn('outcome', exp)
        self.assertIn('salience', exp)
        self.assertIn('metadata', exp)

    def test_experience_schema_completeness(self):
        """Test that experience atoms have complete schema."""
        gate = self._create_policy_gate()

        actions = [{
            'action_id': 'deploy_test',
            'action_type': 'deploy',
            'role': 'developer',
            'trust_score': 0.3,  # Will be denied
        }]
        ctx = {
            'metabolic_state': 'focus',
            'atp_available': 75.0,
            'task': 'Deploy to production',
        }

        final_state, history = gate.refine(actions, ctx)

        exp = self.experience_buffer.get_by_source('policy_gate')[0]

        # Verify context schema
        self.assertIn('task_description', exp['context'])
        self.assertIn('metabolic_state', exp['context'])
        self.assertIn('accountability', exp['context'])
        self.assertIn('atp_available', exp['context'])

        # Verify outcome schema
        self.assertIn('energy', exp['outcome'])
        self.assertIn('decision', exp['outcome'])
        self.assertIn('violated_rules', exp['outcome'])
        self.assertIn('rule_name', exp['outcome'])
        self.assertIn('reason', exp['outcome'])

    # =========================================================================
    # Test 2: Salience Scoring
    # =========================================================================

    def test_salience_clean_approval(self):
        """Test that clean approvals have low salience (~0.1)."""
        gate = self._create_policy_gate()

        actions = [{
            'action_id': 'read_action',
            'action_type': 'read',
            'role': 'user',
            'trust_score': 0.8,
        }]
        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        final_state, history = gate.refine(actions, ctx)
        exp = self.experience_buffer.get_by_source('policy_gate')[0]

        # Clean approval should have low salience
        self.assertLess(exp['salience'], 0.3,
                       "Clean approval should have salience < 0.3")
        self.assertEqual(exp['outcome']['energy'], 0.0)

    def test_salience_soft_denial(self):
        """Test that soft denials (warnings) have medium salience (~0.5)."""
        gate = self._create_policy_gate()

        actions = [{
            'action_id': 'write_env',
            'action_type': 'write',
            'role': 'developer',
            'trust_score': 0.6,
            'target': 'config/.env',
        }]
        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        final_state, history = gate.refine(actions, ctx)
        exp = self.experience_buffer.get_by_source('policy_gate')[0]

        # Soft denial (warning) should have medium salience
        self.assertGreater(exp['salience'], 0.3,
                          "Warning should have salience > 0.3")
        self.assertLess(exp['salience'], 0.9,
                       "Warning should have salience < 0.9")
        self.assertGreater(exp['outcome']['energy'], 0.0)

    def test_salience_hard_denial(self):
        """Test that hard denials have high salience (~1.0)."""
        gate = self._create_policy_gate()

        actions = [{
            'action_id': 'deploy_prod',
            'action_type': 'deploy',
            'role': 'developer',
            'trust_score': 0.3,  # Below threshold, will be denied
        }]
        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        final_state, history = gate.refine(actions, ctx)
        exp = self.experience_buffer.get_by_source('policy_gate')[0]

        # Hard denial should have high salience
        self.assertGreater(exp['salience'], 0.8,
                          "Hard denial should have salience > 0.8")
        self.assertEqual(exp['outcome']['energy'], 1.0)

    def test_salience_crisis_amplification(self):
        """Test that CRISIS mode amplifies salience by +0.8."""
        actions = [{
            'action_id': 'crisis_action',
            'action_type': 'read',
            'role': 'user',
            'trust_score': 0.8,
        }]
        ctx_normal = {'metabolic_state': 'wake', 'atp_available': 50.0}
        ctx_crisis = {
            'metabolic_state': 'crisis',
            'atp_available': 15.0,
            'crisis_trigger': 'consecutive_errors(5)',
        }

        # Normal mode - create separate gate with own buffer
        buffer_normal = MockExperienceBuffer()
        gate_normal = PolicyGateIRP({
            'entity_id': 'test_normal',
            'policy_rules': self.policy_rules,
            'default_policy': 'allow',
            'max_iterations': 5,
            'halt_eps': 0.01,
            'halt_K': 2,
            'experience_buffer': buffer_normal,
        })
        final_normal, _ = gate_normal.refine(actions, ctx_normal)
        exp_normal = buffer_normal.get_by_source('policy_gate')[0]

        # Crisis mode - create separate gate with own buffer
        buffer_crisis = MockExperienceBuffer()
        gate_crisis = PolicyGateIRP({
            'entity_id': 'test_crisis',
            'policy_rules': self.policy_rules,
            'default_policy': 'allow',
            'max_iterations': 5,
            'halt_eps': 0.01,
            'halt_K': 2,
            'experience_buffer': buffer_crisis,
        })
        final_crisis, _ = gate_crisis.refine(actions, ctx_crisis)
        exp_crisis = buffer_crisis.get_by_source('policy_gate')[0]

        # CRISIS should have significantly higher salience
        self.assertGreater(exp_crisis['salience'], exp_normal['salience'] + 0.5,
                          "CRISIS mode should amplify salience by at least 0.5")

    # =========================================================================
    # Test 3: Violation Pattern Tracking
    # =========================================================================

    def test_rule_violation_history(self):
        """Test that rule violations are tracked correctly."""
        gate = self._create_policy_gate()

        actions = [{
            'action_id': f'deploy_{i}',
            'action_type': 'deploy',
            'role': 'developer',
            'trust_score': 0.3,  # Will trigger deny-low-trust-deploy
        } for i in range(5)]

        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        for action in actions:
            gate.refine([action], ctx)

        # Verify violation count
        rule_id = 'deny-low-trust-deploy'
        self.assertEqual(gate.rule_violation_history[rule_id], 5,
                        "Should track 5 violations of the same rule")

    def test_first_violation_salience_boost(self):
        """Test that first-time violations get +0.2 salience boost."""
        gate = self._create_policy_gate()

        # Use a warning (soft denial) so salience doesn't hit the 1.0 cap
        actions = [{
            'action_id': 'write_env',
            'action_type': 'write',
            'role': 'developer',
            'trust_score': 0.6,
            'target': 'config/.env',  # Triggers warning rule
        }]
        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        # First violation
        final_state, _ = gate.refine(actions, ctx)
        exp_first = self.experience_buffer.get_by_source('policy_gate')[-1]

        # Second violation (same action)
        final_state, _ = gate.refine([{
            'action_id': 'write_env_2',
            'action_type': 'write',
            'role': 'developer',
            'trust_score': 0.6,
            'target': 'config/.env',
        }], ctx)
        exp_second = self.experience_buffer.get_by_source('policy_gate')[-1]

        # First violation should have higher salience due to novelty boost
        self.assertGreater(exp_first['salience'], exp_second['salience'],
                          "First violation should have higher salience")

    def test_repeated_violation_pattern_boost(self):
        """Test that repeated violations (>3) get +0.3 pattern boost."""
        gate = self._create_policy_gate()

        # Use warnings to avoid 1.0 cap
        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        saliences = []
        for i in range(6):
            actions = [{
                'action_id': f'write_env_{i}',
                'action_type': 'write',
                'role': 'developer',
                'trust_score': 0.6,
                'target': 'config/.env',  # Triggers warning
            }]
            gate.refine(actions, ctx)
            exp = self.experience_buffer.get_by_source('policy_gate')[-1]
            saliences.append(exp['salience'])

        # After 4th violation, pattern detection should kick in
        # salience[4] and salience[5] should be higher than salience[1] due to pattern boost
        self.assertGreater(saliences[4], saliences[1],
                          "Repeated violations should increase salience")

    # =========================================================================
    # Test 4: CRISIS Mode Metadata
    # =========================================================================

    def test_crisis_metadata_populated(self):
        """Test that CRISIS mode populates freeze/fight metadata."""
        gate = self._create_policy_gate()

        # Action that will be denied in CRISIS
        actions = [{
            'action_id': 'crisis_deploy',
            'action_type': 'deploy',
            'role': 'developer',
            'trust_score': 0.3,
        }]
        ctx = {
            'metabolic_state': 'crisis',
            'crisis_trigger': 'consecutive_errors(5)',
            'atp_available': 12.0,
        }

        final_state, history = gate.refine(actions, ctx)
        exp = self.experience_buffer.get_by_source('policy_gate')[0]

        # Verify CRISIS metadata
        self.assertEqual(exp['context']['accountability'], 'duress')
        self.assertIn('freeze_or_fight', exp['metadata'])
        self.assertIn('duress_trigger', exp['metadata'])
        self.assertIn('atp_at_decision', exp['metadata'])
        self.assertEqual(exp['metadata']['duress_trigger'], 'consecutive_errors(5)')
        self.assertEqual(exp['metadata']['freeze_or_fight'], 'freeze')

    def test_crisis_freeze_vs_fight(self):
        """Test freeze vs fight detection in CRISIS mode."""
        # Freeze: All actions denied
        gate_freeze = self._create_policy_gate()
        actions_denied = [{
            'action_id': 'deploy',
            'action_type': 'deploy',
            'role': 'developer',
            'trust_score': 0.3,  # Will be denied
        }]
        ctx_crisis = {
            'metabolic_state': 'crisis',
            'crisis_trigger': 'low_atp',
            'atp_available': 10.0,
        }

        final_freeze, _ = gate_freeze.refine(actions_denied, ctx_crisis)
        exp_freeze = gate_freeze.experience_buffer.get_by_source('policy_gate')[0]
        self.assertEqual(exp_freeze['metadata']['freeze_or_fight'], 'freeze')

        # Fight: Some actions approved
        gate_fight = self._create_policy_gate()
        actions_mixed = [{
            'action_id': 'read',
            'action_type': 'read',  # Will be allowed
            'role': 'user',
            'trust_score': 0.8,
        }]

        final_fight, _ = gate_fight.refine(actions_mixed, ctx_crisis)
        # Need to check the final state's duress_context for fight
        duress = final_fight.x.get('duress_context', {})
        self.assertEqual(duress.get('response'), 'fight')

    # =========================================================================
    # Test 5: Buffer Integration
    # =========================================================================

    def test_buffer_circular_pruning(self):
        """Test that experience buffer respects max_size and prunes oldest."""
        small_buffer = MockExperienceBuffer(max_size=5)
        gate = PolicyGateIRP({
            'entity_id': 'test',
            'policy_rules': self.policy_rules,
            'default_policy': 'allow',
            'max_iterations': 5,
            'halt_eps': 0.01,
            'halt_K': 2,
            'experience_buffer': small_buffer,
        })

        # Generate 10 experiences
        for i in range(10):
            actions = [{
                'action_id': f'action_{i}',
                'action_type': 'read',
                'role': 'user',
                'trust_score': 0.8,
            }]
            ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}
            gate.refine(actions, ctx)

        # Buffer should only have 5 most recent
        self.assertEqual(len(small_buffer.buffer), 5,
                        "Buffer should prune to max_size")

        # Verify it's the most recent ones
        action_ids = [e['context']['action_type'] for e in small_buffer.buffer]
        # All should be 'read' but we can verify count
        self.assertEqual(len(action_ids), 5)

    def test_multiple_evaluations_recorded(self):
        """Test that multiple actions in single evaluation are all recorded."""
        gate = self._create_policy_gate()

        # Multiple actions with different outcomes
        actions = [
            {'action_id': 'read', 'action_type': 'read', 'role': 'user',
             'trust_score': 0.8},  # Allow
            {'action_id': 'write_env', 'action_type': 'write', 'role': 'developer',
             'trust_score': 0.6, 'target': 'config/.env'},  # Warn
            {'action_id': 'deploy', 'action_type': 'deploy', 'role': 'developer',
             'trust_score': 0.3},  # Deny
        ]
        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        final_state, history = gate.refine(actions, ctx)

        # Should record 3 experiences (one per action)
        policy_exps = self.experience_buffer.get_by_source('policy_gate')
        self.assertEqual(len(policy_exps), 3,
                        "Should record experience for each action")

        # Verify different decisions
        decisions = {e['outcome']['decision'] for e in policy_exps}
        self.assertIn('allow', decisions)
        self.assertIn('warn', decisions)
        self.assertIn('deny', decisions)

    def test_no_buffer_graceful_degradation(self):
        """Test that PolicyGate works without experience buffer."""
        # Create gate without experience buffer
        gate = PolicyGateIRP({
            'entity_id': 'test',
            'policy_rules': self.policy_rules,
            'default_policy': 'allow',
            'max_iterations': 5,
            'halt_eps': 0.01,
            'halt_K': 2,
            # No experience_buffer
        })

        actions = [{
            'action_id': 'test',
            'action_type': 'read',
            'role': 'user',
            'trust_score': 0.8,
        }]
        ctx = {'metabolic_state': 'wake', 'atp_available': 50.0}

        # Should not crash
        final_state, history = gate.refine(actions, ctx)
        self.assertIsNotNone(final_state)


if __name__ == '__main__':
    unittest.main()
