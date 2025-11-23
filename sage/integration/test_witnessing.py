"""
Witness Manager Test Suite

Tests blockchain witnessing functionality:
- WitnessEvent creation and hashing
- MerkleBatch construction
- Individual event witnessing
- Batch witnessing with Merkle trees
- Integration with SAGE components

Author: Thor (SAGE Development Platform)
Date: 2025-11-22
Status: Phase 3 Testing
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from integration.witness_manager import (
    WitnessEvent,
    MerkleBatch,
    WitnessManager,
    create_witness_manager,
    BLOCKCHAIN_AVAILABLE
)


class TestWitnessEvent(unittest.TestCase):
    """Test WitnessEvent functionality"""

    def test_event_creation(self):
        """Test creating witness event"""
        event = WitnessEvent(
            event_type='discovery',
            entity='thor',
            timestamp=datetime.now(timezone.utc),
            data={'test': 'data'},
            quality_score=0.85
        )

        self.assertEqual(event.event_type, 'discovery')
        self.assertEqual(event.entity, 'thor')
        self.assertEqual(event.quality_score, 0.85)
        self.assertIsNotNone(event.hash)

    def test_event_hashing(self):
        """Test event hash calculation"""
        event1 = WitnessEvent(
            event_type='discovery',
            entity='thor',
            timestamp=datetime(2025, 11, 22, 12, 0, 0, tzinfo=timezone.utc),
            data={'test': 'data'},
            quality_score=0.85
        )

        event2 = WitnessEvent(
            event_type='discovery',
            entity='thor',
            timestamp=datetime(2025, 11, 22, 12, 0, 0, tzinfo=timezone.utc),
            data={'test': 'data'},
            quality_score=0.85
        )

        # Same data should produce same hash
        self.assertEqual(event1.hash, event2.hash)

    def test_event_hash_changes(self):
        """Test that different events produce different hashes"""
        base_event = WitnessEvent(
            event_type='discovery',
            entity='thor',
            timestamp=datetime.now(timezone.utc),
            data={'test': 'data'},
            quality_score=0.85
        )

        different_event = WitnessEvent(
            event_type='skill',  # Different type
            entity='thor',
            timestamp=base_event.timestamp,
            data={'test': 'data'},
            quality_score=0.85
        )

        self.assertNotEqual(base_event.hash, different_event.hash)


class TestMerkleBatch(unittest.TestCase):
    """Test Merkle batch functionality"""

    def test_batch_creation(self):
        """Test creating Merkle batch"""
        batch = MerkleBatch(batch_id='test-batch')

        self.assertEqual(batch.batch_id, 'test-batch')
        self.assertEqual(len(batch.events), 0)
        self.assertIsNone(batch.merkle_root)

    def test_add_events(self):
        """Test adding events to batch"""
        batch = MerkleBatch(batch_id='test-batch')

        event = WitnessEvent(
            event_type='discovery',
            entity='thor',
            timestamp=datetime.now(timezone.utc),
            data={'test': 'data'},
            quality_score=0.85
        )

        batch.add_event(event)
        self.assertEqual(len(batch.events), 1)

    def test_merkle_tree_single_event(self):
        """Test Merkle tree with single event"""
        batch = MerkleBatch(batch_id='test-batch')

        event = WitnessEvent(
            event_type='discovery',
            entity='thor',
            timestamp=datetime.now(timezone.utc),
            data={'test': 'data'},
            quality_score=0.85
        )

        batch.add_event(event)
        root = batch.build_merkle_tree()

        self.assertIsNotNone(root)
        self.assertEqual(batch.merkle_root, root)

    def test_merkle_tree_multiple_events(self):
        """Test Merkle tree with multiple events"""
        batch = MerkleBatch(batch_id='test-batch')

        for i in range(5):
            event = WitnessEvent(
                event_type='discovery',
                entity='thor',
                timestamp=datetime.now(timezone.utc),
                data={'index': i},
                quality_score=0.85
            )
            batch.add_event(event)

        root = batch.build_merkle_tree()

        self.assertIsNotNone(root)
        self.assertIsInstance(root, str)
        # Merkle root should be hex string
        self.assertTrue(all(c in '0123456789abcdef' for c in root))

    def test_merkle_tree_deterministic(self):
        """Test that same events produce same Merkle root"""
        batch1 = MerkleBatch(batch_id='batch1')
        batch2 = MerkleBatch(batch_id='batch2')

        # Add same events to both batches
        for i in range(3):
            timestamp = datetime(2025, 11, 22, 12, 0, i, tzinfo=timezone.utc)

            event1 = WitnessEvent(
                event_type='discovery',
                entity='thor',
                timestamp=timestamp,
                data={'index': i},
                quality_score=0.85
            )

            event2 = WitnessEvent(
                event_type='discovery',
                entity='thor',
                timestamp=timestamp,
                data={'index': i},
                quality_score=0.85
            )

            batch1.add_event(event1)
            batch2.add_event(event2)

        root1 = batch1.build_merkle_tree()
        root2 = batch2.build_merkle_tree()

        # Same events should produce same root
        self.assertEqual(root1, root2)


class TestWitnessManager(unittest.TestCase):
    """Test WitnessManager functionality"""

    def setUp(self):
        """Create manager for testing"""
        self.manager = create_witness_manager(
            machine='thor',
            project='sage-test',
            enable_witnessing=False  # Disable actual blockchain witnessing for tests
        )

    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        self.assertEqual(self.manager.machine, 'thor')
        self.assertEqual(self.manager.project, 'sage-test')
        self.assertIsNotNone(self.manager.current_batch)

    def test_witness_discovery(self):
        """Test witnessing discovery"""
        # In test mode (witnessing disabled), should return None
        result = self.manager.witness_discovery(
            knowledge_id='test-001',
            title='Test Discovery',
            domain='test/domain',
            quality=0.85,
            tags=['test']
        )

        # Should return None (not witnessed) but not fail
        self.assertIsNone(result)

    def test_witness_episode(self):
        """Test witnessing episode"""
        result = self.manager.witness_episode(
            episode_id='episode-001',
            quality=0.75,
            discoveries=3,
            failures=1,
            shifts=2
        )

        # In test mode, should return None
        self.assertIsNone(result)

    def test_witness_skill(self):
        """Test witnessing skill"""
        result = self.manager.witness_skill(
            skill_id='skill-001',
            skill_name='Test Skill',
            category='testing',
            quality=0.90,
            success_rate=1.0
        )

        # In test mode, should return None
        self.assertIsNone(result)

    def test_batch_witnessing(self):
        """Test batch witnessing with patterns"""
        # Enable batching and witnessing (but won't actually write to blockchain in test)
        manager = create_witness_manager(
            machine='thor',
            project='sage-test',
            enable_batching=True,
            enable_witnessing=True  # Enable to allow batch adds
        )

        # Add multiple pattern witnesses (batched)
        for i in range(5):
            manager.witness_pattern(
                pattern_id=f'pattern-{i}',
                strategy='Test strategy',
                success_count=3,
                quality=0.85
            )

        # Should have added to batch
        self.assertGreater(len(manager.current_batch.events), 0)

    def test_batch_auto_flush(self):
        """Test batch auto-flushes at size limit"""
        manager = create_witness_manager(
            machine='thor',
            project='sage-test',
            enable_batching=True,
            batch_size=3,
            enable_witnessing=True  # Enable to allow batch adds
        )

        # Add events up to limit
        for i in range(4):
            manager.witness_pattern(
                pattern_id=f'pattern-{i}',
                strategy='Test strategy',
                success_count=3,
                quality=0.85
            )

        # Should have auto-flushed and created new batch
        # New batch should have 1 event (4th event)
        self.assertEqual(len(manager.current_batch.events), 1)

    def test_manual_flush(self):
        """Test manual batch flush"""
        manager = create_witness_manager(
            machine='thor',
            project='sage-test',
            enable_batching=True,
            enable_witnessing=False
        )

        # Add some events
        for i in range(3):
            manager.witness_pattern(
                pattern_id=f'pattern-{i}',
                strategy='Test strategy',
                success_count=3,
                quality=0.85
            )

        # Flush manually
        manager.flush_batch()

        # New batch should be empty
        self.assertEqual(len(manager.current_batch.events), 0)

    def test_statistics(self):
        """Test statistics tracking"""
        stats = self.manager.get_statistics()

        self.assertIn('individual_witnesses', stats)
        self.assertIn('batch_witnesses', stats)
        self.assertIn('total_events_witnessed', stats)
        self.assertIn('current_batch_size', stats)
        self.assertIn('blockchain_available', stats)


class TestIntegration(unittest.TestCase):
    """Test integration with SAGE components"""

    def test_fallback_mode(self):
        """Test system works in fallback mode"""
        manager = create_witness_manager(
            machine='thor',
            project='sage-test',
            enable_witnessing=False
        )

        # All operations should work without errors
        manager.witness_discovery('test', 'Test', 'domain', 0.85)
        manager.witness_episode('ep-001', 0.75, 1, 0, 0)
        manager.witness_skill('skill-001', 'Test', 'cat', 0.9, 1.0)
        manager.flush_batch()

        stats = manager.get_statistics()
        self.assertIsInstance(stats, dict)

    def test_batching_disabled(self):
        """Test with batching disabled"""
        manager = create_witness_manager(
            machine='thor',
            project='sage-test',
            enable_batching=False,
            enable_witnessing=False
        )

        # Batch should be None
        self.assertIsNone(manager.current_batch)

        # Pattern witnessing should do nothing
        result = manager.witness_pattern('p1', 'strategy', 3, 0.85)
        self.assertIsNone(result)


def run_tests():
    """Run all tests with detailed output"""
    print()
    print("=" * 70)
    print("SAGE Witness Manager Test Suite")
    print("=" * 70)
    print(f"Blockchain available: {BLOCKCHAIN_AVAILABLE}")
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWitnessEvent))
    suite.addTests(loader.loadTestsFromTestCase(TestMerkleBatch))
    suite.addTests(loader.loadTestsFromTestCase(TestWitnessManager))
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
