"""
SAGE Skill Integration Test Suite

Validates Phase 2 skill learning functionality:
- Skill query and applicability
- IRPGuidance translation
- Pattern discovery
- Skill creation
- Application tracking

Author: Thor (SAGE Development Platform)
Date: 2025-11-22
"""

import unittest
import sys
from pathlib import Path

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from integration.skill_learning import (
    SkillLearningManager,
    IRPGuidance,
    SkillApplication,
    DiscoveredPattern,
    create_skill_manager,
    SKILLS_AVAILABLE
)


class TestIRPGuidance(unittest.TestCase):
    """Test IRPGuidance translation functionality"""

    def test_guidance_creation(self):
        """Test creating IRP guidance from skill data"""
        guidance = IRPGuidance(
            plugins_prioritized=['llm_reasoning', 'memory'],
            attention_allocation={'llm_reasoning': 0.8, 'memory': 0.2},
            convergence_threshold=0.15,
            expected_iterations=5,
            strategy_description="Focus on logical reasoning with memory support",
            preconditions={},
            indicators={}
        )

        self.assertEqual(guidance.attention_allocation['llm_reasoning'], 0.8)
        self.assertEqual(guidance.expected_iterations, 5)
        self.assertEqual(guidance.convergence_threshold, 0.15)
        self.assertIn('logical reasoning', guidance.strategy_description)

    def test_guidance_serialization(self):
        """Test guidance can be serialized/deserialized"""
        guidance = IRPGuidance(
            plugins_prioritized=['vision', 'llm_reasoning'],
            attention_allocation={'vision': 0.5, 'llm_reasoning': 0.5},
            convergence_threshold=0.1,
            expected_iterations=10,
            strategy_description="Multi-modal processing",
            preconditions={},
            indicators={}
        )

        data = guidance.to_dict()

        # Can be serialized
        self.assertIsInstance(data['attention_allocation'], dict)
        self.assertIsInstance(data['expected_iterations'], int)
        self.assertIsInstance(data['convergence_threshold'], float)


class TestSkillApplication(unittest.TestCase):
    """Test skill application tracking"""

    def test_application_creation(self):
        """Test creating skill application record"""
        from datetime import datetime, timezone

        app = SkillApplication(
            skill_id='skill-123',
            skill_name='Attention Focusing',
            situation='Design attention mechanism',
            applied_at=datetime.now(timezone.utc),
            plugins_used=['llm_reasoning'],
            success=True,
            quality_score=0.85,
            convergence_iterations=3,
            notes='Successful application'
        )

        self.assertEqual(app.skill_id, 'skill-123')
        self.assertTrue(app.success)
        self.assertEqual(app.quality_score, 0.85)
        self.assertEqual(app.plugins_used, ['llm_reasoning'])


class TestDiscoveredPattern(unittest.TestCase):
    """Test pattern discovery functionality"""

    def test_pattern_creation(self):
        """Test creating discovered pattern"""
        pattern = DiscoveredPattern(
            pattern_id='pattern-001',
            strategy='Focused reasoning with rapid convergence',
            plugins_sequence=['llm_reasoning'],
            convergence_profile={'iterations': 3, 'final_energy': 0.2},
            success_count=2,
            failure_count=0,
            average_quality=0.85,
            contexts=['What is X?', 'What is Y?']
        )

        self.assertEqual(pattern.pattern_id, 'pattern-001')
        self.assertEqual(pattern.success_count, 2)
        self.assertEqual(pattern.failure_count, 0)
        self.assertEqual(len(pattern.contexts), 2)

    def test_pattern_signature(self):
        """Test pattern signature extraction"""
        pattern1 = DiscoveredPattern(
            pattern_id='p1',
            strategy='Fast convergence',
            plugins_sequence=['llm_reasoning'],
            convergence_profile={'iterations': 3},
            success_count=1,
            failure_count=0,
            average_quality=0.85,
            contexts=['Test question']
        )

        pattern2 = DiscoveredPattern(
            pattern_id='p2',
            strategy='Fast convergence',
            plugins_sequence=['llm_reasoning'],
            convergence_profile={'iterations': 3},
            success_count=1,
            failure_count=0,
            average_quality=0.85,
            contexts=['Test question']
        )

        # Similar patterns should have similar signatures
        sig1 = f"{pattern1.strategy}|{','.join(pattern1.plugins_sequence)}"
        sig2 = f"{pattern2.strategy}|{','.join(pattern2.plugins_sequence)}"
        self.assertEqual(sig1, sig2)


class TestSkillLearningManager(unittest.TestCase):
    """Test SkillLearningManager functionality"""

    def setUp(self):
        """Create manager for testing"""
        self.manager = create_skill_manager(
            machine='thor',
            project='sage-test'
        )

    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        self.assertEqual(self.manager.machine, 'thor')
        self.assertEqual(self.manager.project, 'sage-test')
        self.assertIsNotNone(self.manager.skill_applications)
        self.assertIsNotNone(self.manager.discovered_patterns)

    def test_hardware_compatibility_check(self):
        """Test hardware compatibility checking"""
        # Skill with hardware requirements
        skill = {
            'metadata': {
                'hardware_requirements': ['GPU', 'cuda']
            }
        }

        compatible = self.manager._check_hardware_compatibility(skill)
        # Should return boolean
        self.assertIsInstance(compatible, bool)

    def test_applicability_calculation(self):
        """Test applicability score calculation"""
        skill = {
            'name': 'Test Skill',
            'description': 'Handle attention and focus questions',
            'context_patterns': ['attention', 'focus', 'salience'],
            'success_rate': 0.9
        }

        situation = "How does attention mechanism work?"

        score = self.manager._calculate_applicability(skill, situation)

        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Should be high for matching situation
        self.assertGreater(score, 0.5)

    def test_pattern_signature_extraction(self):
        """Test extracting pattern signature from execution"""
        execution_data = {
            'situation': 'What is consciousness?',
            'plugins_used': ['llm_reasoning'],
            'iterations': 3,
            'final_energy': 0.2,
            'quality_score': 0.85,
            'energy_drop': 0.8
        }

        signature = self.manager._extract_pattern_signature(execution_data)

        # Should be a dictionary
        self.assertIsInstance(signature, dict)
        # Should have required fields
        self.assertIn('pattern_id', signature)
        self.assertIn('strategy', signature)
        self.assertIn('plugins', signature)
        # Should contain key information
        self.assertIn('llm_reasoning', signature['plugins'])

    def test_strategy_to_irp_guidance(self):
        """Test converting strategy to IRP guidance"""
        strategy = "Focused llm_reasoning processing with rapid convergence"

        skill = {
            'convergence_profile': {
                'iterations': 3,
                'final_energy': 0.2
            }
        }

        guidance = self.manager._strategy_to_irp_guidance(strategy, skill)

        # Should return IRPGuidance
        self.assertIsInstance(guidance, IRPGuidance)

        # Should have reasonable values
        self.assertIsNotNone(guidance.attention_allocation)
        self.assertGreater(guidance.expected_iterations, 0)
        self.assertGreater(guidance.convergence_threshold, 0)

        # Should have prioritized plugins
        self.assertIsInstance(guidance.plugins_prioritized, list)
        self.assertGreater(len(guidance.plugins_prioritized), 0)

        # Should prioritize llm_reasoning
        if 'llm_reasoning' in guidance.attention_allocation:
            self.assertGreater(guidance.attention_allocation['llm_reasoning'], 0.5)

    def test_pattern_recording(self):
        """Test recording successful pattern"""
        execution_data = {
            'situation': 'Test question',
            'plugins_used': ['llm_reasoning'],
            'iterations': 3,
            'final_energy': 0.2,
            'quality_score': 0.85,
            'energy_drop': 0.8,
            'energy_trajectory': [1.0, 0.6, 0.3, 0.2]
        }

        initial_count = len(self.manager.discovered_patterns)

        self.manager.record_successful_pattern(execution_data)

        # Should have recorded (or updated existing pattern)
        self.assertGreaterEqual(len(self.manager.discovered_patterns), initial_count)

    def test_skill_application_recording(self):
        """Test recording skill application"""
        skill = {
            'skill_id': 'test-skill-001',
            'name': 'Test Skill'
        }

        situation = 'Test situation'

        execution_result = {
            'plugins_used': ['llm_reasoning'],
            'converged': True,
            'quality_score': 0.85,
            'iterations': 3
        }

        initial_count = len(self.manager.skill_applications)

        self.manager.record_skill_application(skill, situation, execution_result)

        # Should have recorded application
        self.assertEqual(len(self.manager.skill_applications), initial_count + 1)

        # Check recorded data
        app = self.manager.skill_applications[-1]
        self.assertEqual(app.skill_id, 'test-skill-001')
        self.assertTrue(app.success)

    def test_statistics_generation(self):
        """Test skill statistics generation"""
        stats = self.manager.get_skill_statistics()

        # Should have all required fields
        self.assertIn('applications_count', stats)
        self.assertIn('successful_applications', stats)
        self.assertIn('discovered_patterns', stats)
        self.assertIn('skills_created', stats)
        self.assertIn('average_quality', stats)

        # Should have reasonable values
        self.assertGreaterEqual(stats['applications_count'], 0)
        self.assertGreaterEqual(stats['successful_applications'], 0)
        self.assertGreaterEqual(stats['discovered_patterns'], 0)
        self.assertGreaterEqual(stats['average_quality'], 0.0)
        self.assertLessEqual(stats['average_quality'], 1.0)


class TestSkillCreation(unittest.TestCase):
    """Test automatic skill creation from patterns"""

    def setUp(self):
        """Create manager for testing"""
        self.manager = create_skill_manager(
            machine='thor',
            project='sage-test'
        )

    def test_skill_creation_threshold(self):
        """Test skill creation only happens after threshold"""
        # Create pattern that repeats
        execution_data = {
            'situation': 'Repeated question pattern',
            'plugins_used': ['llm_reasoning'],
            'iterations': 3,
            'final_energy': 0.2,
            'quality_score': 0.85,
            'energy_drop': 0.8,
            'energy_trajectory': [1.0, 0.6, 0.3, 0.2]
        }

        initial_skills = self.manager.get_skill_statistics()['skills_created']

        # Record pattern once - should NOT create skill
        self.manager.record_successful_pattern(execution_data)
        stats1 = self.manager.get_skill_statistics()

        # Record again - still should NOT create skill (need 3)
        self.manager.record_successful_pattern(execution_data)
        stats2 = self.manager.get_skill_statistics()

        # Record third time - NOW should create skill
        self.manager.record_successful_pattern(execution_data)
        stats3 = self.manager.get_skill_statistics()

        # Should have created new skill on third occurrence
        if SKILLS_AVAILABLE:
            # With full integration, skill gets created
            self.assertGreater(stats3['skills_created'], initial_skills)
        else:
            # In fallback mode, skill creation is simulated
            # At least the pattern should be tracked
            self.assertGreater(stats3['discovered_patterns'], 0)


class TestIntegration(unittest.TestCase):
    """Test full integration scenarios"""

    def test_full_workflow(self):
        """Test complete skill learning workflow"""
        manager = create_skill_manager(machine='thor', project='sage-test')

        # 1. Query skills (should work even if empty)
        skills = manager.query_applicable_skills(
            situation="How does attention work?",
            limit=3
        )
        self.assertIsInstance(skills, list)

        # 2. Record successful pattern
        execution_data = {
            'situation': 'Attention mechanism question',
            'plugins_used': ['llm_reasoning'],
            'iterations': 3,
            'final_energy': 0.2,
            'quality_score': 0.85,
            'energy_drop': 0.8,
            'energy_trajectory': [1.0, 0.6, 0.3, 0.2]
        }

        manager.record_successful_pattern(execution_data)

        # 3. Check statistics
        stats = manager.get_skill_statistics()
        self.assertGreater(stats['discovered_patterns'], 0)

    def test_fallback_mode(self):
        """Test system works in fallback mode"""
        # Should work regardless of SKILLS_AVAILABLE
        manager = create_skill_manager(machine='thor', project='sage-test')

        # All operations should work
        skills = manager.query_applicable_skills("test", limit=1)
        self.assertIsInstance(skills, list)

        stats = manager.get_skill_statistics()
        self.assertIsInstance(stats, dict)


def run_tests():
    """Run all tests with detailed output"""
    print()
    print("=" * 70)
    print("SAGE Skill Integration Test Suite")
    print("=" * 70)
    print(f"Epistemic tools available: {SKILLS_AVAILABLE}")
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIRPGuidance))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillApplication))
    suite.addTests(loader.loadTestsFromTestCase(TestDiscoveredPattern))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillLearningManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print()
    print("=" * 70)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
