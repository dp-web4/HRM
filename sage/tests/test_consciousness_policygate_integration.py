"""
Test PolicyGate Phase 5a integration with SAGE consciousness loop.

Tests the _update_policygate_trust_weights() method logic using
a minimal mock of the consciousness object.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sage.irp.plugins.policy_gate import PolicyGateIRP


# Copy of the _update_policygate_trust_weights() method for isolated testing
def update_policygate_trust_weights(consciousness_mock):
    """
    Test implementation of _update_policygate_trust_weights().
    This is the same logic as in SAGEConsciousness.
    """
    if not consciousness_mock.policy_gate_enabled or not consciousness_mock.policy_gate:
        return

    try:
        # Get trust adjustments from PolicyGate compliance analysis
        adjustments = consciousness_mock.policy_gate.compute_trust_adjustments()

        if not adjustments:
            return  # No plugins have enough samples yet

        # Get compliance statistics for logging
        stats = consciousness_mock.policy_gate.get_compliance_stats()

        # Apply adjustments with exponential moving average
        alpha = 0.1  # Same as IRP trust update rate
        trust_min = 0.3
        trust_max = 1.0

        for plugin_name, delta in adjustments.items():
            current_trust = consciousness_mock.plugin_trust_weights.get(plugin_name, 1.0)

            # Apply delta with EMA smoothing
            new_trust = current_trust + (alpha * delta)

            # Enforce bounds
            new_trust = max(trust_min, min(trust_max, new_trust))

            # Update
            consciousness_mock.plugin_trust_weights[plugin_name] = new_trust

        # Save trust weights to disk for persistence
        consciousness_mock.policy_gate.save_trust_weights()

    except Exception as e:
        # Don't crash consciousness loop on trust update errors
        print(f"[PolicyGate Learning] Warning: Trust update failed: {e}")


class TestConsciousnessPolicyGateIntegration(unittest.TestCase):
    """Test PolicyGate Phase 5a integration logic."""

    def setUp(self):
        """Create minimal test fixtures."""
        self.instance_dir = tempfile.mkdtemp()

    def test_01_no_samples_no_crash(self):
        """Test that update doesn't crash with no samples."""
        mock_consciousness = Mock()
        mock_consciousness.policy_gate_enabled = True
        mock_consciousness.plugin_trust_weights = {}

        gate_config = {
            'entity_id': 'policy_gate',
            'policy_rules': [],
            'default_policy': 'allow',
            'instance_dir': self.instance_dir,
        }
        mock_consciousness.policy_gate = PolicyGateIRP(gate_config)

        # Should not crash
        update_policygate_trust_weights(mock_consciousness)

        # No changes
        self.assertEqual(mock_consciousness.plugin_trust_weights, {})

    def test_02_target_compliance_no_change(self):
        """Test that 90% compliance (target) doesn't change trust."""
        mock_consciousness = Mock()
        mock_consciousness.policy_gate_enabled = True
        mock_consciousness.plugin_trust_weights = {'test_plugin': 0.8}

        gate_config = {
            'entity_id': 'policy_gate',
            'policy_rules': [],
            'default_policy': 'allow',
            'instance_dir': self.instance_dir,
        }
        mock_consciousness.policy_gate = PolicyGateIRP(gate_config)

        # Set compliance to 90% (target)
        mock_consciousness.policy_gate.plugin_compliance_history = {
            'test_plugin': {
                'compliant': 18.0,
                'violations': 2.0,
                'total': 20.0
            }
        }

        update_policygate_trust_weights(mock_consciousness)

        # Trust should be ~unchanged (delta = 0)
        self.assertAlmostEqual(mock_consciousness.plugin_trust_weights['test_plugin'], 0.8, places=2)

    def test_03_high_compliance_increases_trust(self):
        """Test that 95% compliance increases trust."""
        mock_consciousness = Mock()
        mock_consciousness.policy_gate_enabled = True
        mock_consciousness.plugin_trust_weights = {'good_plugin': 0.8}

        gate_config = {
            'entity_id': 'policy_gate',
            'policy_rules': [],
            'default_policy': 'allow',
            'instance_dir': self.instance_dir,
        }
        mock_consciousness.policy_gate = PolicyGateIRP(gate_config)

        # Set compliance to 95%
        mock_consciousness.policy_gate.plugin_compliance_history = {
            'good_plugin': {
                'compliant': 19.0,
                'violations': 1.0,
                'total': 20.0
            }
        }

        update_policygate_trust_weights(mock_consciousness)

        # Trust should increase
        self.assertGreater(mock_consciousness.plugin_trust_weights['good_plugin'], 0.8)

    def test_04_low_compliance_decreases_trust(self):
        """Test that 70% compliance decreases trust."""
        mock_consciousness = Mock()
        mock_consciousness.policy_gate_enabled = True
        mock_consciousness.plugin_trust_weights = {'bad_plugin': 0.8}

        gate_config = {
            'entity_id': 'policy_gate',
            'policy_rules': [],
            'default_policy': 'allow',
            'instance_dir': self.instance_dir,
        }
        mock_consciousness.policy_gate = PolicyGateIRP(gate_config)

        # Set compliance to 70%
        mock_consciousness.policy_gate.plugin_compliance_history = {
            'bad_plugin': {
                'compliant': 14.0,
                'violations': 6.0,
                'total': 20.0
            }
        }

        update_policygate_trust_weights(mock_consciousness)

        # Trust should decrease
        self.assertLess(mock_consciousness.plugin_trust_weights['bad_plugin'], 0.8)

    def test_05_trust_bounded_lower(self):
        """Test that trust doesn't go below 0.3."""
        mock_consciousness = Mock()
        mock_consciousness.policy_gate_enabled = True
        mock_consciousness.plugin_trust_weights = {'terrible_plugin': 0.35}

        gate_config = {
            'entity_id': 'policy_gate',
            'policy_rules': [],
            'default_policy': 'allow',
            'instance_dir': self.instance_dir,
        }
        mock_consciousness.policy_gate = PolicyGateIRP(gate_config)

        # Set compliance to 0%
        mock_consciousness.policy_gate.plugin_compliance_history = {
            'terrible_plugin': {
                'compliant': 0.0,
                'violations': 50.0,
                'total': 50.0
            }
        }

        update_policygate_trust_weights(mock_consciousness)

        # Trust should be >= 0.3
        self.assertGreaterEqual(mock_consciousness.plugin_trust_weights['terrible_plugin'], 0.3)

    def test_06_trust_bounded_upper(self):
        """Test that trust doesn't go above 1.0."""
        mock_consciousness = Mock()
        mock_consciousness.policy_gate_enabled = True
        mock_consciousness.plugin_trust_weights = {'perfect_plugin': 0.95}

        gate_config = {
            'entity_id': 'policy_gate',
            'policy_rules': [],
            'default_policy': 'allow',
            'instance_dir': self.instance_dir,
        }
        mock_consciousness.policy_gate = PolicyGateIRP(gate_config)

        # Set compliance to 100%
        mock_consciousness.policy_gate.plugin_compliance_history = {
            'perfect_plugin': {
                'compliant': 50.0,
                'violations': 0.0,
                'total': 50.0
            }
        }

        update_policygate_trust_weights(mock_consciousness)

        # Trust should be <= 1.0
        self.assertLessEqual(mock_consciousness.plugin_trust_weights['perfect_plugin'], 1.0)

    def test_07_persistence_works(self):
        """Test that trust weights are saved to disk."""
        mock_consciousness = Mock()
        mock_consciousness.policy_gate_enabled = True
        mock_consciousness.plugin_trust_weights = {'test_plugin': 0.8}

        gate_config = {
            'entity_id': 'policy_gate',
            'policy_rules': [],
            'default_policy': 'allow',
            'instance_dir': self.instance_dir,
        }
        mock_consciousness.policy_gate = PolicyGateIRP(gate_config)

        # Set compliance
        mock_consciousness.policy_gate.plugin_compliance_history = {
            'test_plugin': {
                'compliant': 18.0,
                'violations': 2.0,
                'total': 20.0
            }
        }

        update_policygate_trust_weights(mock_consciousness)

        # Verify file was created
        trust_file = Path(self.instance_dir) / "policy_trust_weights.json"
        self.assertTrue(trust_file.exists())

        # Verify content
        with open(trust_file) as f:
            saved_data = json.load(f)
        self.assertIn('test_plugin', saved_data)


def run_tests():
    """Run the test suite and print results."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConsciousnessPolicyGateIntegration)
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
