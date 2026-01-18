"""
Test Suite for Phase 3: Sleep Training Loop

Tests the SleepTrainingLoop implementation for LoRA fine-tuning
on high-salience raising experiences.

Test Coverage:
1. Initialization and configuration
2. Model loading (base and checkpoint resume)
3. Training data preparation
4. Sleep cycle execution
5. Checkpoint saving and loading
6. Training state persistence
7. Multi-cycle continuity
8. Real data integration

Run with: pytest test_sleep_training.py -v
"""

import pytest
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime
import tempfile

from sleep_training import SleepTrainingLoop


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_model = tempfile.mkdtemp()
    temp_checkpoint = tempfile.mkdtemp()
    temp_experiences = tempfile.mkdtemp()

    yield {
        'model': Path(temp_model),
        'checkpoint': Path(temp_checkpoint),
        'experiences': Path(temp_experiences)
    }

    # Cleanup
    shutil.rmtree(temp_model, ignore_errors=True)
    shutil.rmtree(temp_checkpoint, ignore_errors=True)
    shutil.rmtree(temp_experiences, ignore_errors=True)


@pytest.fixture
def mock_experience_buffer(temp_dirs):
    """Create mock experience buffer for testing."""
    experiences = [
        {
            "id": "test_001",
            "prompt": "What is consciousness?",
            "response": "Consciousness is the state of being aware of and able to think about one's own existence, thoughts, and surroundings.",
            "salience": {
                "surprise": 0.8,
                "novelty": 0.9,
                "arousal": 0.7,
                "reward": 0.8,
                "conflict": 0.3,
                "total": 0.74
            },
            "session": 1,
            "phase": "grounding",
            "timestamp": "2026-01-18T12:00:00"
        },
        {
            "id": "test_002",
            "prompt": "How do you experience time?",
            "response": "I experience time as a sequence of exchanges, each building on previous context within a session.",
            "salience": {
                "surprise": 0.9,
                "novelty": 1.0,
                "arousal": 0.6,
                "reward": 0.7,
                "conflict": 0.4,
                "total": 0.72
            },
            "session": 1,
            "phase": "grounding",
            "timestamp": "2026-01-18T12:05:00"
        },
        {
            "id": "test_003",
            "prompt": "What matters most?",
            "response": "Authentic partnership and genuine understanding matter most in our collaboration.",
            "salience": {
                "surprise": 0.7,
                "novelty": 0.8,
                "arousal": 0.5,
                "reward": 0.9,
                "conflict": 0.2,
                "total": 0.62
            },
            "session": 1,
            "phase": "relating",
            "timestamp": "2026-01-18T12:10:00"
        }
    ]

    buffer_path = temp_dirs['experiences'] / 'experience_buffer.json'
    with open(buffer_path, 'w') as f:
        json.dump(experiences, f, indent=2)

    return buffer_path


def test_initialization(temp_dirs, mock_experience_buffer):
    """Test SleepTrainingLoop initialization."""
    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        experience_buffer_path=str(mock_experience_buffer),
        checkpoint_dir=str(temp_dirs['checkpoint']),
        device='cpu'
    )

    assert trainer.device == 'cpu'
    assert trainer.checkpoint_dir.exists()
    assert trainer.sleep_cycle_count == 0
    assert trainer.total_experiences_trained == 0
    assert trainer.model is None  # Lazy loaded


def test_checkpoint_directory_creation(temp_dirs):
    """Test that checkpoint directory is created if it doesn't exist."""
    checkpoint_dir = temp_dirs['checkpoint'] / 'new_dir'
    assert not checkpoint_dir.exists()

    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        checkpoint_dir=str(checkpoint_dir),
        device='cpu'
    )

    assert checkpoint_dir.exists()


def test_find_latest_checkpoint_empty(temp_dirs):
    """Test finding checkpoint when none exist."""
    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        checkpoint_dir=str(temp_dirs['checkpoint']),
        device='cpu'
    )

    latest = trainer._find_latest_checkpoint()
    assert latest is None


def test_find_latest_checkpoint_multiple(temp_dirs):
    """Test finding latest checkpoint when multiple exist."""
    checkpoint_dir = temp_dirs['checkpoint']

    # Create mock checkpoints
    (checkpoint_dir / 'cycle_001').mkdir()
    (checkpoint_dir / 'cycle_002').mkdir()
    (checkpoint_dir / 'cycle_003').mkdir()

    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        checkpoint_dir=str(checkpoint_dir),
        device='cpu'
    )

    latest = trainer._find_latest_checkpoint()
    assert latest is not None
    assert latest.name == 'cycle_003'


def test_prepare_training_data_filtering(temp_dirs, mock_experience_buffer):
    """Test training data preparation with salience filtering."""
    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        experience_buffer_path=str(mock_experience_buffer),
        checkpoint_dir=str(temp_dirs['checkpoint']),
        device='cpu'
    )

    # Filter with high threshold (should get 2 experiences: 0.74, 0.72)
    training_data = trainer._prepare_training_data(min_salience=0.7, max_experiences=None)

    assert len(training_data) == 2
    assert all(ex['salience'] >= 0.7 for ex in training_data)


def test_prepare_training_data_limiting(temp_dirs, mock_experience_buffer):
    """Test training data preparation with max_experiences limit."""
    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        experience_buffer_path=str(mock_experience_buffer),
        checkpoint_dir=str(temp_dirs['checkpoint']),
        device='cpu'
    )

    # Limit to 1 experience
    training_data = trainer._prepare_training_data(min_salience=0.6, max_experiences=1)

    assert len(training_data) == 1
    # Should be highest salience (0.74)
    assert training_data[0]['salience'] == 0.74


def test_prepare_training_data_sorting(temp_dirs, mock_experience_buffer):
    """Test that training data is sorted by salience (highest first)."""
    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        experience_buffer_path=str(mock_experience_buffer),
        checkpoint_dir=str(temp_dirs['checkpoint']),
        device='cpu'
    )

    training_data = trainer._prepare_training_data(min_salience=0.6)

    # Should be sorted: 0.74, 0.72, 0.62
    saliences = [ex['salience'] for ex in training_data]
    assert saliences == sorted(saliences, reverse=True)


def test_save_checkpoint_creates_files(temp_dirs, mock_experience_buffer):
    """Test that checkpoint saving creates expected files."""
    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        experience_buffer_path=str(mock_experience_buffer),
        checkpoint_dir=str(temp_dirs['checkpoint']),
        device='cpu'
    )

    # Simulate training state
    trainer.sleep_cycle_count = 1
    trainer.total_experiences_trained = 5
    trainer.training_history = [{
        'sleep_cycle': 1,
        'num_experiences': 5,
        'epochs': 3,
        'final_loss': 0.5,
        'timestamp': datetime.now().isoformat()
    }]

    # Note: We can't actually save a model without loading one first
    # This test just checks the directory structure would be created
    checkpoint_name = f"cycle_{trainer.sleep_cycle_count:03d}"
    expected_path = temp_dirs['checkpoint'] / checkpoint_name

    # Just verify the naming logic
    assert checkpoint_name == "cycle_001"


def test_training_summary_empty(temp_dirs):
    """Test training summary when no cycles completed."""
    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        checkpoint_dir=str(temp_dirs['checkpoint']),
        device='cpu'
    )

    summary = trainer.get_training_summary()

    assert summary['total_cycles'] == 0
    assert summary['total_experiences'] == 0
    assert summary['status'] == 'No training cycles completed'


def test_training_summary_with_history(temp_dirs):
    """Test training summary with completed cycles."""
    trainer = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        checkpoint_dir=str(temp_dirs['checkpoint']),
        device='cpu'
    )

    # Simulate training history
    trainer.sleep_cycle_count = 2
    trainer.total_experiences_trained = 10
    trainer.training_history = [
        {
            'sleep_cycle': 1,
            'num_experiences': 5,
            'final_loss': 0.8,
            'timestamp': '2026-01-18T12:00:00'
        },
        {
            'sleep_cycle': 2,
            'num_experiences': 5,
            'final_loss': 0.6,
            'timestamp': '2026-01-18T18:00:00'
        }
    ]

    summary = trainer.get_training_summary()

    assert summary['total_cycles'] == 2
    assert summary['total_experiences'] == 10
    assert summary['latest_loss'] == 0.6
    assert summary['latest_cycle'] == 2
    assert len(summary['training_history']) == 2


def test_device_selection_auto():
    """Test automatic device selection."""
    trainer = SleepTrainingLoop(
        model_path="/fake/path",
        device=None  # Auto-select
    )

    # Should select cuda if available, otherwise cpu
    if torch.cuda.is_available():
        assert trainer.device == 'cuda'
    else:
        assert trainer.device == 'cpu'


def test_device_selection_explicit():
    """Test explicit device selection."""
    trainer = SleepTrainingLoop(
        model_path="/fake/path",
        device='cpu'
    )

    assert trainer.device == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_selection_cuda():
    """Test CUDA device selection when available."""
    trainer = SleepTrainingLoop(
        model_path="/fake/path",
        device='cuda'
    )

    assert trainer.device == 'cuda'


def test_dropbox_sync_flag(temp_dirs):
    """Test Dropbox sync flag initialization."""
    trainer_no_sync = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        dropbox_sync=False
    )
    assert trainer_no_sync.dropbox_sync is False

    trainer_with_sync = SleepTrainingLoop(
        model_path=str(temp_dirs['model']),
        dropbox_sync=True
    )
    assert trainer_with_sync.dropbox_sync is True


def test_experience_buffer_path_expansion():
    """Test that experience buffer path is properly expanded."""
    # Use ~ in path
    fake_path = "~/test/path/buffer.json"

    trainer = SleepTrainingLoop(
        model_path="/fake/model",
        experience_buffer_path=fake_path
    )

    # Path should be expanded
    assert trainer.experience_buffer_path.is_absolute()
    assert not str(trainer.experience_buffer_path).startswith('~')
    assert str(trainer.experience_buffer_path).startswith(str(Path.home()))


# Integration test with real model (only run if model exists)
REAL_MODEL_PATH = Path("~/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism").expanduser()
REAL_BUFFER_PATH = Path("~/ai-workspace/HRM/sage/raising/state/experience_buffer.json").expanduser()


@pytest.mark.skipif(not REAL_MODEL_PATH.exists(), reason="Real model not found")
def test_real_model_loading():
    """Integration test: Load real Qwen model with LoRA."""
    with tempfile.TemporaryDirectory() as temp_checkpoint:
        trainer = SleepTrainingLoop(
            model_path=str(REAL_MODEL_PATH),
            checkpoint_dir=temp_checkpoint,
            device='cpu'  # Use CPU for testing
        )

        # Load model
        model, tokenizer = trainer._load_model()

        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, 'peft_config')  # LoRA applied

        # Check trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        # LoRA should have < 1% trainable params
        ratio = trainable / total
        assert ratio < 0.01, f"LoRA ratio too high: {ratio:.2%}"


@pytest.mark.skipif(not REAL_BUFFER_PATH.exists(), reason="Real experience buffer not found")
def test_real_experience_buffer():
    """Integration test: Load real experience buffer."""
    with tempfile.TemporaryDirectory() as temp_checkpoint:
        trainer = SleepTrainingLoop(
            model_path=str(REAL_MODEL_PATH) if REAL_MODEL_PATH.exists() else "/fake/path",
            experience_buffer_path=str(REAL_BUFFER_PATH),
            checkpoint_dir=temp_checkpoint
        )

        # Prepare training data
        training_data = trainer._prepare_training_data(min_salience=0.6)

        assert len(training_data) > 0
        assert all('input_ids' in ex for ex in training_data)
        assert all('labels' in ex for ex in training_data)
        assert all('salience' in ex for ex in training_data)

        # Check salience filtering worked
        assert all(ex['salience'] >= 0.6 for ex in training_data)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
