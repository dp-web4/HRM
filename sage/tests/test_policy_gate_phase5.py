"""
PolicyGate Phase 5: Trust Weight Learning - Unit Tests
Date: 2026-03-06
Status: Phase 5a Implementation

Tests trust weight adaptation based on plugin compliance history:
- Salience-weighted compliance tracking
- Trust delta computation
- Bounded adjustments
- Minimum sample size requirements
- Persistence (save/load)
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

# Import PolicyGateIRP
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'irp' / 'plugins'))
from policy_gate import PolicyGateIRP


class TestPhase5TrustWeightLearning(unittest.TestCase):
    """Test suite for Phase 5a: Trust Weight Learning"""

    def setUp(self):
        """Create a PolicyGate instance for testing."""
        self.config = {
            'default_policy': 'allow',
            'policy_rules': [
                {
                    'id': 'deny-low-trust',
                    'name': 'Deny low trust actions',
                    'priority': 1,
                    'decision': 'deny',
                    'match': {
                        'action_types': ['deploy'],
                        'max_trust': 0.5,
                    },
                    'reason': 'Trust too low',
                    'atp_cost': 0,
                }
            ]
        }
        self.gate = PolicyGateIRP(self.config)

    def test_01_compliance_tracking_initialization(self):
        """Test that compliance history is properly initialized."""
        self.assertIsInstance(self.gate.plugin_compliance_history, dict)
        self.assertEqual(len(self.gate.plugin_compliance_history), 0)

    def test_02_salience_weight_high(self):
        """Test high salience decisions get 2.0x weight."""
        # High salience scenario: policy violation with high salience
        action = {
            'action_id': 'test-001',
            'action_type': 'deploy',
            'target': '/etc/critical',
            'role': 'user',
            'trust_score': 0.3,  # Below threshold, will violate
            'parameters': {}
        }

        task_ctx = {
            'metabolic_state': 'crisis',  # High salience
            'plugin_name': 'test_plugin',
            'atp_available': 10.0,
        }

        # Run evaluation (this should trigger recording via step())
        state = self.gate.init_state(action, task_ctx)
        # Call step() to trigger compliance tracking
        self.gate.step(state, task_ctx)
        self.gate.energy(state)

        # Check that plugin was tracked
        self.assertIn('test_plugin', self.gate.plugin_compliance_history)
        stats = self.gate.plugin_compliance_history['test_plugin']

        # High salience violation should record 2.0 weighted count
        # (decision was deny, so it's a violation)
        self.assertEqual(stats['total'], 2.0)
        self.assertEqual(stats['violations'], 2.0)
        self.assertEqual(stats['compliant'], 0.0)

    def test_03_salience_weight_medium(self):
        """Test medium salience decisions get 1.0x weight."""
        # Medium salience scenario: DEGRADED state (REST/DREAM)
        # REST has DEGRADED accountability which adds +0.2 to base salience (0.1 → 0.3 → 1.0x weight)
        action = {
            'action_id': 'test-002',
            'action_type': 'write',
            'target': 'config.json',
            'role': 'user',
            'trust_score': 0.8,
            'parameters': {}
        }

        task_ctx = {
            'metabolic_state': 'rest',  # DEGRADED → +0.2 boost → 0.3 salience
            'plugin_name': 'test_plugin_2',
            'atp_available': 50.0,
        }

        # This action should pass (allow), so compliant
        state = self.gate.init_state(action, task_ctx)
        self.gate.step(state, task_ctx)
        self.gate.energy(state)

        stats = self.gate.plugin_compliance_history['test_plugin_2']
        self.assertEqual(stats['total'], 1.0)  # 1.0x weight (0.3 salience)
        self.assertEqual(stats['compliant'], 1.0)
        self.assertEqual(stats['violations'], 0.0)

    def test_04_salience_weight_low(self):
        """Test low salience decisions get 0.5x weight."""
        # Low salience scenario: routine operation in NORMAL state (WAKE/FOCUS)
        # NORMAL accountability, clean approval → base salience 0.1 → <0.3 → 0.5x weight
        action = {
            'action_id': 'test-003',
            'action_type': 'read',
            'target': 'data.json',
            'role': 'user',
            'trust_score': 0.9,
            'parameters': {}
        }

        task_ctx = {
            'metabolic_state': 'focus',  # NORMAL → base 0.1 salience
            'plugin_name': 'test_plugin_3',
            'atp_available': 20.0,
        }

        state = self.gate.init_state(action, task_ctx)
        self.gate.step(state, task_ctx)
        self.gate.energy(state)

        stats = self.gate.plugin_compliance_history['test_plugin_3']
        self.assertEqual(stats['total'], 0.5)  # 0.5x weight (0.1 salience)
        self.assertEqual(stats['compliant'], 0.5)

    def test_05_compute_trust_adjustments_high_compliance(self):
        """Test trust adjustment for high compliance (>90%)."""
        # Manually set up compliance history: 95% compliance
        self.gate.plugin_compliance_history = {
            'high_compliance_plugin': {
                'compliant': 19.0,  # 95% compliant
                'violations': 1.0,
                'total': 20.0
            }
        }

        adjustments = self.gate.compute_trust_adjustments()

        self.assertIn('high_compliance_plugin', adjustments)
        delta = adjustments['high_compliance_plugin']

        # 95% compliance → deviation = +0.05 → delta = +0.025
        self.assertAlmostEqual(delta, 0.025, places=3)

    def test_06_compute_trust_adjustments_low_compliance(self):
        """Test trust adjustment for low compliance (70%)."""
        self.gate.plugin_compliance_history = {
            'low_compliance_plugin': {
                'compliant': 14.0,  # 70% compliant
                'violations': 6.0,
                'total': 20.0
            }
        }

        adjustments = self.gate.compute_trust_adjustments()

        delta = adjustments['low_compliance_plugin']

        # 70% compliance → deviation = -0.2 → delta capped at -0.1
        self.assertEqual(delta, -0.1)

    def test_07_compute_trust_adjustments_target_compliance(self):
        """Test trust adjustment at target 90% compliance."""
        self.gate.plugin_compliance_history = {
            'target_plugin': {
                'compliant': 18.0,  # 90% compliant
                'violations': 2.0,
                'total': 20.0
            }
        }

        adjustments = self.gate.compute_trust_adjustments()

        delta = adjustments['target_plugin']

        # 90% compliance → deviation = 0.0 → delta = 0.0
        self.assertAlmostEqual(delta, 0.0, places=3)

    def test_08_bounded_adjustments_upper(self):
        """Test upper bound (+0.1 max) is enforced."""
        self.gate.plugin_compliance_history = {
            'perfect_plugin': {
                'compliant': 50.0,  # 100% compliant
                'violations': 0.0,
                'total': 50.0
            }
        }

        adjustments = self.gate.compute_trust_adjustments()

        delta = adjustments['perfect_plugin']

        # 100% compliance → deviation = +0.1 → delta = +0.05, within bounds
        self.assertAlmostEqual(delta, 0.05, places=3)
        self.assertLessEqual(delta, 0.1)

    def test_09_bounded_adjustments_lower(self):
        """Test lower bound (-0.1 max) is enforced."""
        self.gate.plugin_compliance_history = {
            'terrible_plugin': {
                'compliant': 0.0,  # 0% compliant
                'violations': 50.0,
                'total': 50.0
            }
        }

        adjustments = self.gate.compute_trust_adjustments()

        delta = adjustments['terrible_plugin']

        # 0% compliance → deviation = -0.9 → delta capped at -0.1
        self.assertEqual(delta, -0.1)

    def test_10_minimum_sample_size(self):
        """Test minimum sample size requirement (10 weighted samples)."""
        self.gate.plugin_compliance_history = {
            'insufficient_plugin': {
                'compliant': 8.0,
                'violations': 1.0,
                'total': 9.0  # Below 10.0 threshold
            },
            'sufficient_plugin': {
                'compliant': 9.0,
                'violations': 1.0,
                'total': 10.0  # At threshold
            }
        }

        adjustments = self.gate.compute_trust_adjustments()

        # Insufficient samples → no adjustment
        self.assertNotIn('insufficient_plugin', adjustments)

        # Sufficient samples → adjustment computed
        self.assertIn('sufficient_plugin', adjustments)

    def test_11_get_compliance_stats(self):
        """Test compliance statistics reporting."""
        self.gate.plugin_compliance_history = {
            'plugin_a': {
                'compliant': 18.0,
                'violations': 2.0,
                'total': 20.0
            },
            'plugin_b': {
                'compliant': 45.0,
                'violations': 5.0,
                'total': 50.0
            }
        }

        stats = self.gate.get_compliance_stats()

        self.assertEqual(len(stats), 2)
        self.assertIn('plugin_a', stats)
        self.assertIn('plugin_b', stats)

        # Check plugin_a stats
        self.assertAlmostEqual(stats['plugin_a']['compliance_ratio'], 0.9, places=2)
        self.assertEqual(stats['plugin_a']['weighted_total'], 20.0)
        self.assertEqual(stats['plugin_a']['weighted_compliant'], 18.0)
        self.assertEqual(stats['plugin_a']['weighted_violations'], 2.0)

        # Check plugin_b stats
        self.assertAlmostEqual(stats['plugin_b']['compliance_ratio'], 0.9, places=2)

    def test_12_persistence_save_load(self):
        """Test trust weight persistence (save and load)."""
        # Create temporary directory for instance
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure gate with instance directory
            config = self.config.copy()
            config['instance_dir'] = tmpdir
            gate = PolicyGateIRP(config)

            # Set up some compliance history
            gate.plugin_compliance_history = {
                'plugin_x': {
                    'compliant': 18.0,
                    'violations': 2.0,
                    'total': 20.0
                },
                'plugin_y': {
                    'compliant': 27.0,
                    'violations': 3.0,
                    'total': 30.0
                }
            }

            # Save to disk
            gate.save_trust_weights()

            # Verify file exists
            trust_file = Path(tmpdir) / "policy_trust_weights.json"
            self.assertTrue(trust_file.exists())

            # Create new gate instance and verify it loads the data
            gate2 = PolicyGateIRP(config)

            self.assertEqual(len(gate2.plugin_compliance_history), 2)
            self.assertIn('plugin_x', gate2.plugin_compliance_history)
            self.assertIn('plugin_y', gate2.plugin_compliance_history)
            self.assertEqual(gate2.plugin_compliance_history['plugin_x']['total'], 20.0)

    def test_13_persistence_corrupted_file(self):
        """Test handling of corrupted trust weight file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write corrupted JSON file
            trust_file = Path(tmpdir) / "policy_trust_weights.json"
            with open(trust_file, 'w') as f:
                f.write("{ corrupted json }")

            # Gate should handle corruption gracefully
            config = self.config.copy()
            config['instance_dir'] = tmpdir
            gate = PolicyGateIRP(config)

            # Should have empty history (didn't crash)
            self.assertEqual(len(gate.plugin_compliance_history), 0)

    def test_14_persistence_no_instance_dir(self):
        """Test that persistence is skipped when no instance_dir provided."""
        # Gate without instance_dir
        gate = PolicyGateIRP(self.config)

        gate.plugin_compliance_history = {
            'plugin_z': {
                'compliant': 10.0,
                'violations': 0.0,
                'total': 10.0
            }
        }

        # Should not crash when saving without instance_dir
        gate.save_trust_weights()  # No-op

    def test_15_multiple_plugins_tracking(self):
        """Test tracking multiple plugins simultaneously."""
        actions = [
            {
                'action_id': 'test-plugin-a-1',
                'action_type': 'write',
                'target': 'file1.txt',
                'role': 'user',
                'trust_score': 0.9,
                'parameters': {}
            },
            {
                'action_id': 'test-plugin-b-1',
                'action_type': 'deploy',
                'target': 'service',
                'role': 'user',
                'trust_score': 0.3,  # Will violate
                'parameters': {}
            }
        ]

        task_contexts = [
            {
                'metabolic_state': 'focus',
                'plugin_name': 'plugin_a',
                'atp_available': 50.0,
            },
            {
                'metabolic_state': 'focus',
                'plugin_name': 'plugin_b',
                'atp_available': 50.0,
            }
        ]

        # Evaluate both actions
        for action, ctx in zip(actions, task_contexts):
            state = self.gate.init_state(action, ctx)
            self.gate.step(state, ctx)
            self.gate.energy(state)

        # Both plugins should be tracked
        self.assertEqual(len(self.gate.plugin_compliance_history), 2)
        self.assertIn('plugin_a', self.gate.plugin_compliance_history)
        self.assertIn('plugin_b', self.gate.plugin_compliance_history)

        # plugin_a should be compliant, plugin_b should have violation
        self.assertGreater(
            self.gate.plugin_compliance_history['plugin_a']['compliant'],
            0.0
        )
        self.assertGreater(
            self.gate.plugin_compliance_history['plugin_b']['violations'],
            0.0
        )


def run_tests():
    """Run the test suite and print results."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase5TrustWeightLearning)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
