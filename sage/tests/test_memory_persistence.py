#!/usr/bin/env python3
"""
Cross-Session Memory Persistence Tests

Tests save/load functionality for consolidated memories, enabling
SAGE to persist learnings across sessions.

Session Context: Autonomous research session (Dec 16, 2025)
Related: Session 50 (DREAM consolidation), Session 51 (Transfer learning)
Gap Identified: Memories created but not persisted across restarts

Author: Thor-SAGE-Researcher (Autonomous)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
import tempfile
import shutil
import json
from sage.core.dream_consolidation import (
    DREAMConsolidator,
    ConsolidatedMemory,
    MemoryPattern,
    QualityLearning,
    CreativeAssociation
)


class TestMemoryPersistence(unittest.TestCase):
    """Test suite for memory persistence functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.consolidator = DREAMConsolidator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_sample_memory(self, session_id: int = 1) -> ConsolidatedMemory:
        """Create a sample consolidated memory for testing"""
        patterns = [
            MemoryPattern(
                pattern_type="QUALITY",
                description="High quality correlates with specific technical terms",
                strength=0.85,
                examples=[1, 3, 5, 7],
                frequency=4,
                created_at=1234567890.0
            ),
            MemoryPattern(
                pattern_type="EPISTEMIC",
                description="High confidence when citing specific sources",
                strength=0.78,
                examples=[2, 4, 6],
                frequency=3,
                created_at=1234567891.0
            )
        ]

        learnings = [
            QualityLearning(
                characteristic="specific_terms",
                positive_correlation=True,
                confidence=0.92,
                sample_size=20,
                average_quality_with=0.85,
                average_quality_without=0.45
            ),
            QualityLearning(
                characteristic="hedging_language",
                positive_correlation=False,
                confidence=0.88,
                sample_size=18,
                average_quality_with=0.40,
                average_quality_without=0.80
            )
        ]

        associations = [
            CreativeAssociation(
                concept_a="consciousness",
                concept_b="energy_management",
                association_type="causal",
                strength=0.75,
                supporting_cycles=[5, 12, 18],
                insight="Consciousness quality degrades with low ATP"
            ),
            CreativeAssociation(
                concept_a="pattern_retrieval",
                concept_b="quality_improvement",
                association_type="functional",
                strength=0.82,
                supporting_cycles=[15, 20],
                insight="Retrieved patterns guide better responses"
            )
        ]

        insights = [
            "Transfer learning requires DREAM consolidation",
            "Quality improvement needs variable input quality",
            "Circadian timing affects consolidation effectiveness"
        ]

        return ConsolidatedMemory(
            dream_session_id=session_id,
            timestamp=1234567900.0,
            cycles_processed=25,
            patterns=patterns,
            quality_learnings=learnings,
            creative_associations=associations,
            epistemic_insights=insights,
            consolidation_time=2.5
        )

    def test_memory_pattern_serialization(self):
        """Test MemoryPattern to_dict/from_dict round-trip"""
        original = MemoryPattern(
            pattern_type="QUALITY",
            description="Test pattern",
            strength=0.75,
            examples=[1, 2, 3],
            frequency=3,
            created_at=1234567890.0
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = MemoryPattern.from_dict(data)

        # Verify all fields match
        self.assertEqual(restored.pattern_type, original.pattern_type)
        self.assertEqual(restored.description, original.description)
        self.assertAlmostEqual(restored.strength, original.strength)
        self.assertEqual(restored.examples, original.examples)
        self.assertEqual(restored.frequency, original.frequency)
        self.assertAlmostEqual(restored.created_at, original.created_at)

    def test_quality_learning_serialization(self):
        """Test QualityLearning to_dict/from_dict round-trip"""
        original = QualityLearning(
            characteristic="specific_terms",
            positive_correlation=True,
            confidence=0.85,
            sample_size=20,
            average_quality_with=0.80,
            average_quality_without=0.45
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = QualityLearning.from_dict(data)

        # Verify all fields match
        self.assertEqual(restored.characteristic, original.characteristic)
        self.assertEqual(restored.positive_correlation, original.positive_correlation)
        self.assertAlmostEqual(restored.confidence, original.confidence)
        self.assertEqual(restored.sample_size, original.sample_size)
        self.assertAlmostEqual(restored.average_quality_with, original.average_quality_with)
        self.assertAlmostEqual(restored.average_quality_without, original.average_quality_without)

    def test_creative_association_serialization(self):
        """Test CreativeAssociation to_dict/from_dict round-trip"""
        original = CreativeAssociation(
            concept_a="consciousness",
            concept_b="energy",
            association_type="causal",
            strength=0.75,
            supporting_cycles=[1, 2, 3],
            insight="Energy affects consciousness quality"
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = CreativeAssociation.from_dict(data)

        # Verify all fields match
        self.assertEqual(restored.concept_a, original.concept_a)
        self.assertEqual(restored.concept_b, original.concept_b)
        self.assertEqual(restored.association_type, original.association_type)
        self.assertAlmostEqual(restored.strength, original.strength)
        self.assertEqual(restored.supporting_cycles, original.supporting_cycles)
        self.assertEqual(restored.insight, original.insight)

    def test_creative_association_no_insight(self):
        """Test CreativeAssociation serialization without insight field"""
        original = CreativeAssociation(
            concept_a="test_a",
            concept_b="test_b",
            association_type="analogical",
            strength=0.65,
            supporting_cycles=[5],
            insight=None
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = CreativeAssociation.from_dict(data)

        # Verify insight is None
        self.assertIsNone(restored.insight)

    def test_consolidated_memory_serialization(self):
        """Test ConsolidatedMemory to_dict/from_dict round-trip"""
        original = self.create_sample_memory(session_id=1)

        # Serialize and deserialize
        data = original.to_dict()
        restored = ConsolidatedMemory.from_dict(data)

        # Verify top-level fields
        self.assertEqual(restored.dream_session_id, original.dream_session_id)
        self.assertAlmostEqual(restored.timestamp, original.timestamp)
        self.assertEqual(restored.cycles_processed, original.cycles_processed)
        self.assertAlmostEqual(restored.consolidation_time, original.consolidation_time)

        # Verify patterns
        self.assertEqual(len(restored.patterns), len(original.patterns))
        for i, pattern in enumerate(restored.patterns):
            self.assertEqual(pattern.pattern_type, original.patterns[i].pattern_type)
            self.assertEqual(pattern.description, original.patterns[i].description)
            self.assertAlmostEqual(pattern.strength, original.patterns[i].strength)

        # Verify learnings
        self.assertEqual(len(restored.quality_learnings), len(original.quality_learnings))
        for i, learning in enumerate(restored.quality_learnings):
            self.assertEqual(learning.characteristic, original.quality_learnings[i].characteristic)
            self.assertEqual(learning.positive_correlation, original.quality_learnings[i].positive_correlation)

        # Verify associations
        self.assertEqual(len(restored.creative_associations), len(original.creative_associations))
        for i, assoc in enumerate(restored.creative_associations):
            self.assertEqual(assoc.concept_a, original.creative_associations[i].concept_a)
            self.assertEqual(assoc.concept_b, original.creative_associations[i].concept_b)

        # Verify insights
        self.assertEqual(restored.epistemic_insights, original.epistemic_insights)

    def test_export_import_single_memory(self):
        """Test export and import of single consolidated memory"""
        memory = self.create_sample_memory(session_id=1)
        filepath = os.path.join(self.temp_dir, "test_memory.json")

        # Export
        self.consolidator.export_consolidated_memory(memory, filepath)

        # Verify file exists
        self.assertTrue(os.path.exists(filepath))

        # Import
        restored = self.consolidator.import_consolidated_memory(filepath)

        # Verify content matches
        self.assertEqual(restored.dream_session_id, memory.dream_session_id)
        self.assertEqual(len(restored.patterns), len(memory.patterns))
        self.assertEqual(len(restored.quality_learnings), len(memory.quality_learnings))
        self.assertEqual(len(restored.creative_associations), len(memory.creative_associations))
        self.assertEqual(restored.epistemic_insights, memory.epistemic_insights)

    def test_save_all_memories(self):
        """Test batch save of all consolidated memories"""
        # Create multiple memories
        memories = [
            self.create_sample_memory(session_id=1),
            self.create_sample_memory(session_id=2),
            self.create_sample_memory(session_id=3)
        ]

        # Add to consolidator
        self.consolidator.consolidated_memories = memories
        self.consolidator.dream_session_count = 3

        # Save all
        self.consolidator.save_all_memories(self.temp_dir)

        # Verify files exist
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "memory_001.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "memory_002.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "memory_003.json")))

    def test_load_all_memories(self):
        """Test batch load of all consolidated memories"""
        # Create and save multiple memories
        original_memories = [
            self.create_sample_memory(session_id=1),
            self.create_sample_memory(session_id=2),
            self.create_sample_memory(session_id=3)
        ]

        for memory in original_memories:
            filename = f"memory_{memory.dream_session_id:03d}.json"
            filepath = os.path.join(self.temp_dir, filename)
            self.consolidator.export_consolidated_memory(memory, filepath)

        # Create new consolidator and load
        new_consolidator = DREAMConsolidator()
        loaded_count = new_consolidator.load_all_memories(self.temp_dir)

        # Verify count
        self.assertEqual(loaded_count, 3)
        self.assertEqual(len(new_consolidator.consolidated_memories), 3)

        # Verify session count updated
        self.assertEqual(new_consolidator.dream_session_count, 3)

        # Verify content
        for i, memory in enumerate(new_consolidator.consolidated_memories):
            self.assertEqual(memory.dream_session_id, i + 1)
            self.assertEqual(len(memory.patterns), 2)
            self.assertEqual(len(memory.quality_learnings), 2)

    def test_load_nonexistent_directory(self):
        """Test loading from nonexistent directory raises error"""
        consolidator = DREAMConsolidator()
        fake_dir = "/nonexistent/directory/path"

        with self.assertRaises(FileNotFoundError):
            consolidator.load_all_memories(fake_dir)

    def test_load_corrupted_file_warning(self):
        """Test loading corrupted JSON file prints warning and continues"""
        # Create valid memory file
        memory = self.create_sample_memory(session_id=1)
        self.consolidator.export_consolidated_memory(
            memory,
            os.path.join(self.temp_dir, "memory_001.json")
        )

        # Create corrupted file
        corrupted_path = os.path.join(self.temp_dir, "memory_002.json")
        with open(corrupted_path, 'w') as f:
            f.write("{ invalid json content")

        # Load should succeed but only load 1 memory
        new_consolidator = DREAMConsolidator()
        loaded_count = new_consolidator.load_all_memories(self.temp_dir)

        self.assertEqual(loaded_count, 1)

    def test_cross_session_persistence_workflow(self):
        """Test complete workflow: create → save → new session → load → use"""
        # SESSION 1: Create and consolidate some memories
        session1_consolidator = DREAMConsolidator()
        memory1 = self.create_sample_memory(session_id=1)
        memory2 = self.create_sample_memory(session_id=2)

        session1_consolidator.consolidated_memories = [memory1, memory2]
        session1_consolidator.dream_session_count = 2

        # Save at end of session
        session1_consolidator.save_all_memories(self.temp_dir)

        # SESSION 2: New consolidator (simulates restart)
        session2_consolidator = DREAMConsolidator()

        # Load previous memories
        loaded_count = session2_consolidator.load_all_memories(self.temp_dir)
        self.assertEqual(loaded_count, 2)

        # Verify can access loaded memories
        memories = session2_consolidator.get_consolidated_memories()
        self.assertEqual(len(memories), 2)

        # Create new memory (session continues)
        memory3 = self.create_sample_memory(session_id=3)
        session2_consolidator.consolidated_memories.append(memory3)
        session2_consolidator.dream_session_count = 3

        # Save again
        session2_consolidator.save_all_memories(self.temp_dir)

        # SESSION 3: Verify all 3 memories persist
        session3_consolidator = DREAMConsolidator()
        loaded_count = session3_consolidator.load_all_memories(self.temp_dir)
        self.assertEqual(loaded_count, 3)

    def test_json_format_human_readable(self):
        """Test exported JSON is human-readable (formatted)"""
        memory = self.create_sample_memory(session_id=1)
        filepath = os.path.join(self.temp_dir, "readable_test.json")

        self.consolidator.export_consolidated_memory(memory, filepath)

        # Read raw file content
        with open(filepath, 'r') as f:
            content = f.read()

        # Verify it's formatted (has newlines and indentation)
        self.assertIn('\n', content)
        self.assertIn('  ', content)  # Indentation

        # Verify it's valid JSON
        data = json.loads(content)
        self.assertIn('patterns', data)
        self.assertIn('quality_learnings', data)


def run_tests():
    """Run all tests with detailed output"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMemoryPersistence)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 80)
    print("SAGE Cross-Session Memory Persistence Tests")
    print("Session: Autonomous research (Dec 16, 2025)")
    print("=" * 80)
    print()

    success = run_tests()

    print()
    print("=" * 80)
    if success:
        print("✅ ALL TESTS PASSED - Memory persistence working!")
        print()
        print("Next Steps:")
        print("1. Integrate with unified_consciousness.py")
        print("2. Add persistence directory configuration")
        print("3. Test with real DREAM consolidation data")
        print("4. Document cross-session memory usage pattern")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    print("=" * 80)

    sys.exit(0 if success else 1)
