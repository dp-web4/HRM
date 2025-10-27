#!/usr/bin/env python3
"""
Comprehensive tests for Universal Chunking Base Framework

Tests:
1. ChunkTrustMetrics - Trust computation and validation
2. ChunkSalienceMetrics - SNARC salience computation
3. UniversalChunk - Chunk creation and priority computation
4. UniversalChunker - Abstract base class (via mock implementation)
5. Utility functions - ATP allocation, eviction, temporal grouping
"""

import unittest
import time
from typing import Tuple, Any

import sys
sys.path.insert(0, '/home/dp/ai-workspace/HRM/sage')

from cognitive.universal_chunking import (
    ChunkTrustMetrics,
    ChunkSalienceMetrics,
    UniversalChunk,
    UniversalChunker,
    allocate_attention,
    evict_lowest_priority_chunk,
    group_by_time
)


class TestChunkTrustMetrics(unittest.TestCase):
    """Test trust metrics computation"""

    def test_trust_computation_balanced(self):
        """Test trust with balanced scores"""
        trust = ChunkTrustMetrics(
            confidence=0.8,
            consistency=0.7,
            completeness=0.9,
            fidelity=0.6
        )

        overall = trust.compute_overall_trust()

        # Weighted: 0.8*0.35 + 0.7*0.25 + 0.9*0.25 + 0.6*0.15
        expected = 0.28 + 0.175 + 0.225 + 0.09
        self.assertAlmostEqual(overall, expected, places=5)

    def test_trust_computation_perfect(self):
        """Test trust with perfect scores"""
        trust = ChunkTrustMetrics(
            confidence=1.0,
            consistency=1.0,
            completeness=1.0,
            fidelity=1.0
        )

        overall = trust.compute_overall_trust()
        self.assertAlmostEqual(overall, 1.0, places=5)

    def test_trust_computation_poor(self):
        """Test trust with poor scores"""
        trust = ChunkTrustMetrics(
            confidence=0.2,
            consistency=0.1,
            completeness=0.3,
            fidelity=0.0
        )

        overall = trust.compute_overall_trust()

        # Should be low (weighted average of low scores)
        self.assertLess(overall, 0.3)
        self.assertGreater(overall, 0.0)

    def test_trust_to_dict(self):
        """Test trust export to dictionary"""
        trust = ChunkTrustMetrics(
            confidence=0.8,
            consistency=0.7,
            completeness=0.9,
            fidelity=0.6
        )

        trust_dict = trust.to_dict()

        self.assertIn('confidence', trust_dict)
        self.assertIn('consistency', trust_dict)
        self.assertIn('completeness', trust_dict)
        self.assertIn('fidelity', trust_dict)
        self.assertIn('overall', trust_dict)
        self.assertAlmostEqual(trust_dict['overall'], trust.compute_overall_trust())


class TestChunkSalienceMetrics(unittest.TestCase):
    """Test SNARC salience computation"""

    def test_salience_computation_balanced(self):
        """Test salience with balanced SNARC scores"""
        salience = ChunkSalienceMetrics(
            surprise=0.6,
            novelty=0.5,
            arousal=0.7,
            reward=0.4,
            conflict=0.3,
            prosodic=0.8  # Major boundary
        )

        overall = salience.compute_overall_salience()

        # Base SNARC: (0.6+0.5+0.7+0.4+0.3)/5 = 0.5
        # Modulated: 0.5 * (0.7 + 0.8*0.3) = 0.5 * 0.94 = 0.47
        expected = 0.5 * (0.7 + 0.8 * 0.3)
        self.assertAlmostEqual(overall, expected, places=5)

    def test_salience_prosodic_amplification(self):
        """Test prosodic salience amplifies base SNARC"""
        # Same base SNARC, different prosodic
        base_snarc = {'surprise': 0.6, 'novelty': 0.6, 'arousal': 0.6, 'reward': 0.6, 'conflict': 0.6}

        # Major boundary (high prosodic)
        sal_major = ChunkSalienceMetrics(**base_snarc, prosodic=0.9)
        overall_major = sal_major.compute_overall_salience()

        # Minor boundary (low prosodic)
        sal_minor = ChunkSalienceMetrics(**base_snarc, prosodic=0.3)
        overall_minor = sal_minor.compute_overall_salience()

        # Major boundary should have higher overall salience
        self.assertGreater(overall_major, overall_minor)

    def test_salience_computation_high(self):
        """Test salience with high SNARC scores"""
        salience = ChunkSalienceMetrics(
            surprise=0.9,
            novelty=0.8,
            arousal=0.9,
            reward=0.8,
            conflict=0.7,
            prosodic=0.9
        )

        overall = salience.compute_overall_salience()

        # Should be high (base ~0.82, modulated by 0.97)
        self.assertGreater(overall, 0.7)

    def test_salience_to_dict(self):
        """Test salience export to dictionary"""
        salience = ChunkSalienceMetrics(
            surprise=0.6,
            novelty=0.5,
            arousal=0.7,
            reward=0.4,
            conflict=0.3,
            prosodic=0.8
        )

        sal_dict = salience.to_dict()

        self.assertIn('surprise', sal_dict)
        self.assertIn('novelty', sal_dict)
        self.assertIn('arousal', sal_dict)
        self.assertIn('reward', sal_dict)
        self.assertIn('conflict', sal_dict)
        self.assertIn('prosodic', sal_dict)
        self.assertIn('overall', sal_dict)
        self.assertAlmostEqual(sal_dict['overall'], salience.compute_overall_salience())


class TestUniversalChunk(unittest.TestCase):
    """Test universal chunk creation and methods"""

    def setUp(self):
        """Create test trust and salience metrics"""
        self.trust = ChunkTrustMetrics(
            confidence=0.8,
            consistency=0.7,
            completeness=0.9,
            fidelity=0.6
        )

        self.salience = ChunkSalienceMetrics(
            surprise=0.6,
            novelty=0.5,
            arousal=0.7,
            reward=0.4,
            conflict=0.3,
            prosodic=0.8
        )

    def test_chunk_creation(self):
        """Test creating a universal chunk"""
        chunk = UniversalChunk(
            content="test content",
            modality="audio",
            timestamp=time.time(),
            duration=2.5,
            boundary_type="major",
            chunk_size=12,
            continuation=True,
            trust_score=0.75,
            trust_breakdown=self.trust,
            salience_score=0.47,
            salience_breakdown=self.salience
        )

        self.assertEqual(chunk.modality, "audio")
        self.assertEqual(chunk.boundary_type, "major")
        self.assertEqual(chunk.chunk_size, 12)
        self.assertTrue(chunk.continuation)

    def test_chunk_priority(self):
        """Test priority computation (trust × salience)"""
        chunk = UniversalChunk(
            content="test",
            modality="vision",
            timestamp=time.time(),
            duration=1.5,
            boundary_type="minor",
            chunk_size=5,
            continuation=False,
            trust_score=0.8,
            salience_score=0.6,
            trust_breakdown=self.trust,
            salience_breakdown=self.salience
        )

        priority = chunk.get_priority()
        expected = 0.8 * 0.6
        self.assertAlmostEqual(priority, expected, places=5)

    def test_chunk_needs_verification(self):
        """Test low-trust chunk detection"""
        # High trust - no verification needed
        chunk_high = UniversalChunk(
            content="test",
            modality="audio",
            timestamp=time.time(),
            duration=2.0,
            boundary_type="major",
            chunk_size=10,
            continuation=True,
            trust_score=0.8,
            salience_score=0.5,
            trust_breakdown=self.trust,
            salience_breakdown=self.salience
        )
        self.assertFalse(chunk_high.needs_verification())

        # Low trust - needs verification
        chunk_low = UniversalChunk(
            content="test",
            modality="audio",
            timestamp=time.time(),
            duration=2.0,
            boundary_type="forced",
            chunk_size=10,
            continuation=True,
            trust_score=0.3,
            salience_score=0.5,
            trust_breakdown=self.trust,
            salience_breakdown=self.salience
        )
        self.assertTrue(chunk_low.needs_verification())

    def test_chunk_high_salience(self):
        """Test high-salience chunk detection"""
        # Low salience
        chunk_low = UniversalChunk(
            content="test",
            modality="memory",
            timestamp=time.time(),
            duration=1.0,
            boundary_type="minor",
            chunk_size=5,
            continuation=True,
            trust_score=0.7,
            salience_score=0.4,
            trust_breakdown=self.trust,
            salience_breakdown=self.salience
        )
        self.assertFalse(chunk_low.is_high_salience())

        # High salience
        chunk_high = UniversalChunk(
            content="test",
            modality="memory",
            timestamp=time.time(),
            duration=1.0,
            boundary_type="major",
            chunk_size=5,
            continuation=True,
            trust_score=0.7,
            salience_score=0.85,
            trust_breakdown=self.trust,
            salience_breakdown=self.salience
        )
        self.assertTrue(chunk_high.is_high_salience())

    def test_chunk_to_dict(self):
        """Test chunk export to dictionary"""
        chunk = UniversalChunk(
            content="test content",
            modality="language",
            timestamp=time.time(),
            duration=2.5,
            boundary_type="major",
            chunk_size=12,
            continuation=True,
            trust_score=0.75,
            trust_breakdown=self.trust,
            salience_score=0.47,
            salience_breakdown=self.salience,
            metadata={'foo': 'bar'}
        )

        chunk_dict = chunk.to_dict()

        self.assertIn('modality', chunk_dict)
        self.assertIn('boundary_type', chunk_dict)
        self.assertIn('trust_score', chunk_dict)
        self.assertIn('salience_score', chunk_dict)
        self.assertIn('priority', chunk_dict)
        self.assertIn('needs_verification', chunk_dict)
        self.assertIn('is_high_salience', chunk_dict)
        self.assertIn('metadata', chunk_dict)
        self.assertEqual(chunk_dict['metadata']['foo'], 'bar')


class MockChunker(UniversalChunker):
    """Mock chunker for testing abstract base class"""

    def detect_boundary(self, buffer: Any, new_item: Any) -> Tuple[bool, str]:
        """Mock boundary detection - boundary every 5 items"""
        if isinstance(buffer, list) and len(buffer) >= 5:
            return (True, "major")
        return (False, None)

    def compute_trust(self, chunk_content: Any) -> ChunkTrustMetrics:
        """Mock trust computation"""
        return ChunkTrustMetrics(
            confidence=0.8,
            consistency=0.7,
            completeness=0.9,
            fidelity=0.6
        )

    def compute_salience(self, chunk_content: Any) -> ChunkSalienceMetrics:
        """Mock salience computation"""
        return ChunkSalienceMetrics(
            surprise=0.6,
            novelty=0.5,
            arousal=0.7,
            reward=0.4,
            conflict=0.3,
            prosodic=0.8
        )

    def extract_prosody(self, chunk_content: Any) -> Any:
        """Mock prosody extraction"""
        return {'mock_prosody': True}


class TestUniversalChunker(unittest.TestCase):
    """Test universal chunker base class via mock"""

    def setUp(self):
        """Create mock chunker"""
        self.chunker = MockChunker(
            modality="mock",
            min_chunk_size=3,
            target_chunk_size=5,
            max_chunk_size=10,
            chunk_duration=(1.0, 4.0)
        )

    def test_chunker_initialization(self):
        """Test chunker initialization"""
        self.assertEqual(self.chunker.modality, "mock")
        self.assertEqual(self.chunker.min_chunk_size, 3)
        self.assertEqual(self.chunker.target_chunk_size, 5)
        self.assertEqual(self.chunker.max_chunk_size, 10)
        self.assertEqual(self.chunker.min_duration, 1.0)
        self.assertEqual(self.chunker.max_duration, 4.0)

    def test_detect_boundary(self):
        """Test boundary detection"""
        buffer = [1, 2, 3]
        is_boundary, boundary_type = self.chunker.detect_boundary(buffer, 4)
        self.assertFalse(is_boundary)

        buffer = [1, 2, 3, 4, 5]
        is_boundary, boundary_type = self.chunker.detect_boundary(buffer, 6)
        self.assertTrue(is_boundary)
        self.assertEqual(boundary_type, "major")

    def test_create_chunk(self):
        """Test chunk creation through chunker"""
        content = [1, 2, 3, 4, 5]
        chunk = self.chunker.create_chunk(
            content=content,
            boundary_type="major",
            chunk_size=5,
            duration=2.0,
            continuation=True
        )

        self.assertEqual(chunk.modality, "mock")
        self.assertEqual(chunk.boundary_type, "major")
        self.assertEqual(chunk.chunk_size, 5)
        self.assertAlmostEqual(chunk.duration, 2.0)
        self.assertTrue(chunk.continuation)

        # Trust and salience should be computed
        self.assertIsNotNone(chunk.trust_breakdown)
        self.assertIsNotNone(chunk.salience_breakdown)
        self.assertGreater(chunk.trust_score, 0.0)
        self.assertGreater(chunk.salience_score, 0.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for cross-modal coordination"""

    def setUp(self):
        """Create test chunks"""
        trust = ChunkTrustMetrics(0.8, 0.7, 0.9, 0.6)
        salience = ChunkSalienceMetrics(0.6, 0.5, 0.7, 0.4, 0.3, 0.8)

        self.chunks = [
            UniversalChunk(
                content="high priority",
                modality="audio",
                timestamp=100.0,
                duration=2.0,
                boundary_type="major",
                chunk_size=10,
                continuation=True,
                trust_score=0.9,
                salience_score=0.8,
                trust_breakdown=trust,
                salience_breakdown=salience
            ),
            UniversalChunk(
                content="medium priority",
                modality="vision",
                timestamp=100.5,
                duration=1.5,
                boundary_type="minor",
                chunk_size=5,
                continuation=True,
                trust_score=0.7,
                salience_score=0.6,
                trust_breakdown=trust,
                salience_breakdown=salience
            ),
            UniversalChunk(
                content="low priority",
                modality="memory",
                timestamp=102.0,
                duration=1.0,
                boundary_type="micro",
                chunk_size=3,
                continuation=False,
                trust_score=0.4,
                salience_score=0.3,
                trust_breakdown=trust,
                salience_breakdown=salience
            )
        ]

    def test_allocate_attention(self):
        """Test ATP allocation based on priority"""
        total_atp = 100.0
        allocations = allocate_attention(self.chunks, total_atp)

        # Should allocate to all chunks
        self.assertEqual(len(allocations), 3)

        # Extract ATP values (same order as input chunks)
        atp_values = [atp for chunk, atp in allocations]

        # Total should sum to total_atp
        total_allocated = sum(atp_values)
        self.assertAlmostEqual(total_allocated, total_atp, places=2)

        # High priority chunk (index 0) should get most ATP
        # Medium priority (index 1) should get middle ATP
        # Low priority (index 2) should get least ATP
        self.assertGreater(atp_values[0], atp_values[1])
        self.assertGreater(atp_values[1], atp_values[2])

    def test_allocate_attention_empty(self):
        """Test ATP allocation with empty chunk list"""
        allocations = allocate_attention([], 100.0)
        self.assertEqual(len(allocations), 0)

    def test_evict_lowest_priority(self):
        """Test eviction of lowest priority chunk"""
        evicted = evict_lowest_priority_chunk(self.chunks)

        # Should evict the low priority chunk
        self.assertEqual(evicted.content, "low priority")
        self.assertEqual(evicted.modality, "memory")

    def test_evict_empty_buffer(self):
        """Test eviction with empty buffer"""
        evicted = evict_lowest_priority_chunk([])
        self.assertIsNone(evicted)

    def test_group_by_time(self):
        """Test temporal grouping of chunks"""
        # Chunks at t=100.0, t=100.5, t=102.0
        # Window of 1.0 second should group first two together
        groups = group_by_time(self.chunks, temporal_window=1.0)

        # Should have 2 groups (100.0-101.0 and 102.0-103.0)
        self.assertEqual(len(groups), 2)

        # First group should have 2 chunks
        first_group = groups[100.0]
        self.assertEqual(len(first_group), 2)
        self.assertEqual(first_group[0].modality, "audio")
        self.assertEqual(first_group[1].modality, "vision")

        # Second group should have 1 chunk
        second_group = groups[102.0]
        self.assertEqual(len(second_group), 1)
        self.assertEqual(second_group[0].modality, "memory")

    def test_group_by_time_small_window(self):
        """Test temporal grouping with very small window"""
        # Window of 0.1 seconds should create 3 separate groups
        groups = group_by_time(self.chunks, temporal_window=0.1)

        self.assertEqual(len(groups), 3)


if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)
