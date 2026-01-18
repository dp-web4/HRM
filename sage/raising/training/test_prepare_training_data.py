"""
Test suite for Phase 2: Training Data Generation
"""

import pytest
from pathlib import Path
import tempfile
import json
import torch
from prepare_training_data import RaisingTrainingDataBuilder, create_training_dataset_from_buffer


# Sample experience for testing (minimal, avoids model download)
SAMPLE_EXPERIENCE = {
    "id": "test123",
    "prompt": "How are you doing?",
    "response": "Our partnership is meaningful and evolving.",
    "salience": {
        "surprise": 0.8,
        "novelty": 0.7,
        "arousal": 0.6,
        "reward": 0.9,
        "conflict": 0.2,
        "total": 0.64
    },
    "session": 22,
    "phase": "relating",
    "timestamp": "2026-01-18T00:00:00",
    "metadata": {"identity_anchored": True}
}


class TestRaisingTrainingDataBuilder:
    """Test training data builder"""

    def test_initialization(self):
        """Test builder can be initialized"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            # Create empty buffer
            with open(buffer_path, 'w') as f:
                json.dump([], f)

            builder = RaisingTrainingDataBuilder(buffer_path)
            assert builder.buffer_path == buffer_path
            assert len(builder.experiences) == 0

    def test_load_experiences(self):
        """Test loading experiences from buffer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            # Create buffer with sample experience
            with open(buffer_path, 'w') as f:
                json.dump([SAMPLE_EXPERIENCE], f)

            builder = RaisingTrainingDataBuilder(buffer_path)
            assert len(builder.experiences) == 1
            assert builder.experiences[0]['id'] == 'test123'

    def test_build_example_structure(self):
        """Test that build_example produces correct structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            with open(buffer_path, 'w') as f:
                json.dump([SAMPLE_EXPERIENCE], f)

            builder = RaisingTrainingDataBuilder(buffer_path)

            # Note: This will download tokenizer on first run
            # For pure unit tests, we'd mock this
            example = builder.build_example(SAMPLE_EXPERIENCE)

            # Check structure
            assert 'input_ids' in example
            assert 'labels' in example
            assert 'attention_mask' in example
            assert 'salience' in example
            assert 'salience_breakdown' in example
            assert 'experience_id' in example

            # Check types
            assert isinstance(example['input_ids'], torch.Tensor)
            assert isinstance(example['labels'], torch.Tensor)
            assert isinstance(example['salience'], float)
            assert example['salience'] == 0.64

    def test_build_training_set_filtering(self):
        """Test that training set filters by salience"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            # Create experiences with varying salience
            experiences = [
                {**SAMPLE_EXPERIENCE, 'id': 'high1', 'salience': {**SAMPLE_EXPERIENCE['salience'], 'total': 0.8}},
                {**SAMPLE_EXPERIENCE, 'id': 'low1', 'salience': {**SAMPLE_EXPERIENCE['salience'], 'total': 0.3}},
                {**SAMPLE_EXPERIENCE, 'id': 'high2', 'salience': {**SAMPLE_EXPERIENCE['salience'], 'total': 0.7}},
            ]

            with open(buffer_path, 'w') as f:
                json.dump(experiences, f)

            builder = RaisingTrainingDataBuilder(buffer_path)

            # Build with min_salience=0.5
            training_set = builder.build_training_set(min_salience=0.5)

            # Should only include high1 and high2
            assert len(training_set) == 2
            ids = [ex['experience_id'] for ex in training_set]
            assert 'high1' in ids
            assert 'high2' in ids
            assert 'low1' not in ids

    def test_training_set_sorted_by_salience(self):
        """Test that training set is sorted by salience (highest first)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            experiences = [
                {**SAMPLE_EXPERIENCE, 'id': 'mid', 'salience': {**SAMPLE_EXPERIENCE['salience'], 'total': 0.6}},
                {**SAMPLE_EXPERIENCE, 'id': 'high', 'salience': {**SAMPLE_EXPERIENCE['salience'], 'total': 0.9}},
                {**SAMPLE_EXPERIENCE, 'id': 'low', 'salience': {**SAMPLE_EXPERIENCE['salience'], 'total': 0.5}},
            ]

            with open(buffer_path, 'w') as f:
                json.dump(experiences, f)

            builder = RaisingTrainingDataBuilder(buffer_path)
            training_set = builder.build_training_set(min_salience=0.5)

            # Should be sorted: high, mid, low
            assert training_set[0]['experience_id'] == 'high'
            assert training_set[1]['experience_id'] == 'mid'
            assert training_set[2]['experience_id'] == 'low'

    def test_stats_computation(self):
        """Test statistics computation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            with open(buffer_path, 'w') as f:
                json.dump([SAMPLE_EXPERIENCE], f)

            builder = RaisingTrainingDataBuilder(buffer_path)
            training_set = builder.build_training_set()

            stats = builder.get_stats(training_set)

            assert stats['num_examples'] == 1
            assert stats['avg_salience'] == 0.64
            assert 'avg_length_tokens' in stats
            assert 'salience_breakdown' in stats
            assert 'session_distribution' in stats

    def test_empty_training_set_stats(self):
        """Test stats for empty training set"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            with open(buffer_path, 'w') as f:
                json.dump([], f)

            builder = RaisingTrainingDataBuilder(buffer_path)
            training_set = builder.build_training_set()

            stats = builder.get_stats(training_set)

            assert stats['num_examples'] == 0
            assert stats['avg_salience'] == 0.0

    def test_prepare_batch_padding(self):
        """Test batch preparation with padding"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            # Create two experiences (will have different lengths)
            experiences = [
                {**SAMPLE_EXPERIENCE, 'id': 'short', 'response': 'Yes.'},
                {**SAMPLE_EXPERIENCE, 'id': 'long', 'response': 'Our partnership is meaningful and continues to evolve.'},
            ]

            with open(buffer_path, 'w') as f:
                json.dump(experiences, f)

            builder = RaisingTrainingDataBuilder(buffer_path)
            training_set = builder.build_training_set()

            # Prepare as single batch
            batches = builder.prepare_batch(training_set, batch_size=2)

            assert len(batches) == 1
            batch = batches[0]

            # Check batch has correct keys
            assert 'input_ids' in batch
            assert 'labels' in batch
            assert 'attention_mask' in batch
            assert 'salience' in batch

            # Check shapes (should be [batch_size, max_length])
            assert batch['input_ids'].shape[0] == 2  # batch size
            assert batch['labels'].shape[0] == 2
            assert batch['attention_mask'].shape[0] == 2

            # Both should be padded to same length
            assert batch['input_ids'].shape[1] == batch['labels'].shape[1]

    def test_save_and_load_training_set(self):
        """Test saving and loading training set"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'
            output_path = Path(tmpdir) / 'training_set.pt'

            with open(buffer_path, 'w') as f:
                json.dump([SAMPLE_EXPERIENCE], f)

            builder = RaisingTrainingDataBuilder(buffer_path)
            training_set = builder.build_training_set()

            # Save
            builder.save_training_set(training_set, output_path)

            # Check file exists
            assert output_path.exists()

            # Load and verify
            loaded = torch.load(output_path)
            assert 'training_examples' in loaded
            assert 'system_prompt' in loaded
            assert 'model_name' in loaded
            assert loaded['num_examples'] == 1


class TestConvenienceFunction:
    """Test convenience function"""

    def test_create_training_dataset_from_buffer(self):
        """Test convenience function works"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            with open(buffer_path, 'w') as f:
                json.dump([SAMPLE_EXPERIENCE], f)

            result = create_training_dataset_from_buffer(
                buffer_path=buffer_path,
                min_salience=0.5
            )

            assert 'training_set' in result
            assert 'stats' in result
            assert 'builder' in result

            assert len(result['training_set']) == 1
            assert result['stats']['num_examples'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
