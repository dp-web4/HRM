"""
SAGE Sleep Cycle Manager

Orchestrates sleep cycles where SAGE:
1. Reviews accumulated experiences (SNARC-scored conversation history)
2. Extracts training data with augmentation (variations for robustness)
3. Fine-tunes models during "sleep" (both slow-path Qwen and fast-path tiny model)
4. Consolidates patterns into long-term memory
5. Prunes low-value experiences to free context window

Initial implementation uses manual orchestration (human + Claude).
Future: SAGE will autonomously decide when to sleep and what to learn.

Sleep cycle triggers:
- Manual: User/Claude initiates sleep
- Automatic (future): High buffer utilization (>80%)
- Automatic (future): Low-quality fast-path performance
- Automatic (future): Scheduled (e.g., daily at 3am)
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SleepCycleConfig:
    """Configuration for a sleep cycle"""
    # Cycle identification
    cycle_id: str
    cycle_type: str  # "manual", "scheduled", "triggered"
    timestamp: float

    # Training parameters
    deep_reasoning_epochs: int = 3
    fast_path_epochs: int = 10
    learning_rate: float = 1e-5

    # Data augmentation
    augmentation_enabled: bool = True
    augmentation_strategies: List[str] = None

    # Memory management
    prune_low_salience: bool = True
    salience_threshold: float = 0.3

    # Quality thresholds
    min_examples_deep: int = 10
    min_examples_fast: int = 50

    def __post_init__(self):
        if self.augmentation_strategies is None:
            self.augmentation_strategies = ['paraphrase', 'context_shift']


@dataclass
class SleepCycleResults:
    """Results from a completed sleep cycle"""
    cycle_id: str
    start_time: float
    end_time: float
    duration_seconds: float

    # Training data stats
    examples_extracted: Dict[str, int]
    examples_augmented: int

    # Training results
    deep_model_trained: bool = False
    deep_model_loss: Optional[float] = None
    fast_model_trained: bool = False
    fast_model_loss: Optional[float] = None

    # Memory operations
    patterns_consolidated: int = 0
    experiences_pruned: int = 0
    buffer_freed_percent: float = 0.0

    # Quality metrics
    notes: str = ""


class SleepCycleManager:
    """
    Manages SAGE sleep cycles for memory consolidation and model fine-tuning.

    Sleep cycles are periods where SAGE:
    - Stops active conversation
    - Reviews accumulated experiences
    - Extracts and augments training data
    - Fine-tunes models (slow-path Qwen, fast-path tiny model)
    - Consolidates high-salience memories
    - Prunes low-value experiences

    Initially: Manual orchestration by human + Claude
    Future: Autonomous sleep cycle management by SAGE
    """

    def __init__(self, storage_dir: str = "sleep_cycles"):
        """
        Initialize sleep cycle manager.

        Args:
            storage_dir: Directory to store sleep cycle data and results
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Subdirectories
        self.training_data_dir = self.storage_dir / "training_data"
        self.models_dir = self.storage_dir / "models"
        self.logs_dir = self.storage_dir / "logs"

        for d in [self.training_data_dir, self.models_dir, self.logs_dir]:
            d.mkdir(exist_ok=True)

        # Track sleep cycles
        self.cycle_history: List[SleepCycleResults] = []
        self.current_cycle: Optional[str] = None

    def initiate_sleep_cycle(self,
                            memory_manager,
                            config: Optional[SleepCycleConfig] = None) -> str:
        """
        Initiate a new sleep cycle.

        Args:
            memory_manager: SNARCMemoryManager with conversation history
            config: Sleep cycle configuration (uses defaults if None)

        Returns:
            Cycle ID for tracking
        """
        if config is None:
            config = SleepCycleConfig(
                cycle_id=self._generate_cycle_id(),
                cycle_type="manual",
                timestamp=time.time()
            )

        self.current_cycle = config.cycle_id

        # Log cycle initiation
        log_entry = {
            'event': 'sleep_cycle_initiated',
            'cycle_id': config.cycle_id,
            'timestamp': time.time(),
            'config': asdict(config),
            'memory_stats': memory_manager.get_stats()
        }
        self._log_event(log_entry)

        return config.cycle_id

    def extract_training_data(self,
                             memory_manager,
                             extractor,
                             config: SleepCycleConfig) -> Dict[str, int]:
        """
        Extract training data from memory with augmentation.

        Args:
            memory_manager: SNARCMemoryManager instance
            extractor: TrainingDataExtractor instance
            config: Sleep cycle configuration

        Returns:
            Statistics on extracted examples
        """
        # Extract base examples
        extractor.extract_from_memory(memory_manager)
        base_stats = extractor.get_statistics()

        # Augment examples if enabled
        if config.augmentation_enabled:
            # Augment deep reasoning examples (critical for sleep cycle learning)
            extractor.deep_reasoning_examples = extractor.augment_examples(
                extractor.deep_reasoning_examples,
                config.augmentation_strategies
            )

            # Augment fast ack examples (for robust fast-path model)
            extractor.fast_ack_examples = extractor.augment_examples(
                extractor.fast_ack_examples,
                config.augmentation_strategies
            )

        # Save training data
        cycle_data_dir = self.training_data_dir / config.cycle_id
        cycle_data_dir.mkdir(exist_ok=True)
        extractor.output_dir = cycle_data_dir

        saved_files = extractor.save_training_data(format="jsonl")

        # Log extraction
        log_entry = {
            'event': 'training_data_extracted',
            'cycle_id': config.cycle_id,
            'timestamp': time.time(),
            'statistics': extractor.get_statistics(),
            'files_saved': {k: str(v) for k, v in saved_files.items()}
        }
        self._log_event(log_entry)

        return extractor.get_statistics()

    def fine_tune_models(self,
                        config: SleepCycleConfig,
                        training_data_stats: Dict[str, int]) -> Dict[str, bool]:
        """
        Fine-tune models during sleep cycle.

        This is a placeholder for the actual fine-tuning logic.
        In production, this will:
        1. Load Qwen 2.5-0.5B model
        2. Fine-tune on deep_reasoning.jsonl
        3. Load/create tiny fast-path model (~10M params)
        4. Fine-tune on fast_ack.jsonl + handoff.jsonl
        5. Validate both models
        6. Save checkpoints

        Args:
            config: Sleep cycle configuration
            training_data_stats: Statistics from training data extraction

        Returns:
            Dictionary indicating which models were trained
        """
        results = {
            'deep_model_trained': False,
            'fast_model_trained': False,
            'deep_model_ready': False,
            'fast_model_ready': False
        }

        # Check if we have enough data
        if training_data_stats['deep_reasoning'] < config.min_examples_deep:
            log_entry = {
                'event': 'insufficient_training_data',
                'cycle_id': config.cycle_id,
                'timestamp': time.time(),
                'model': 'deep_reasoning',
                'required': config.min_examples_deep,
                'available': training_data_stats['deep_reasoning']
            }
            self._log_event(log_entry)
        else:
            results['deep_model_ready'] = True

        if training_data_stats['fast_ack'] < config.min_examples_fast:
            log_entry = {
                'event': 'insufficient_training_data',
                'cycle_id': config.cycle_id,
                'timestamp': time.time(),
                'model': 'fast_path',
                'required': config.min_examples_fast,
                'available': training_data_stats['fast_ack']
            }
            self._log_event(log_entry)
        else:
            results['fast_model_ready'] = True

        # TODO: Implement actual fine-tuning
        # For now, just log what would happen
        log_entry = {
            'event': 'fine_tuning_placeholder',
            'cycle_id': config.cycle_id,
            'timestamp': time.time(),
            'note': 'Fine-tuning not yet implemented. Would train:',
            'deep_model_epochs': config.deep_reasoning_epochs if results['deep_model_ready'] else 0,
            'fast_model_epochs': config.fast_path_epochs if results['fast_model_ready'] else 0,
            'learning_rate': config.learning_rate,
            'training_data': training_data_stats
        }
        self._log_event(log_entry)

        return results

    def consolidate_memory(self,
                          memory_manager,
                          config: SleepCycleConfig) -> Dict[str, int]:
        """
        Consolidate memory during sleep cycle.

        Operations:
        1. Prune low-salience experiences from buffer
        2. Ensure high-salience experiences in long-term storage
        3. Free up context window space

        Args:
            memory_manager: SNARCMemoryManager instance
            config: Sleep cycle configuration

        Returns:
            Statistics on memory operations
        """
        stats = {
            'pruned': 0,
            'consolidated': 0,
            'buffer_before': len(memory_manager.conversation_buffer),
            'buffer_after': 0
        }

        if config.prune_low_salience:
            # Prune low-salience turns
            before_count = len(memory_manager.conversation_buffer)

            memory_manager.conversation_buffer = [
                turn for turn in memory_manager.conversation_buffer
                if turn.salience_score >= config.salience_threshold
            ]

            after_count = len(memory_manager.conversation_buffer)
            stats['pruned'] = before_count - after_count
            stats['buffer_after'] = after_count

        # Re-extract high-salience to long-term (consolidation)
        high_salience_turns = [
            turn for turn in memory_manager.conversation_buffer
            if turn.salience_score >= 0.7
        ]

        existing_timestamps = {turn.timestamp for turn in memory_manager.long_term_memory}
        for turn in high_salience_turns:
            if turn.timestamp not in existing_timestamps:
                memory_manager.long_term_memory.append(turn)
                stats['consolidated'] += 1

        # Log consolidation
        log_entry = {
            'event': 'memory_consolidated',
            'cycle_id': config.cycle_id,
            'timestamp': time.time(),
            'statistics': stats,
            'buffer_freed_percent': (stats['pruned'] / stats['buffer_before']) * 100 if stats['buffer_before'] > 0 else 0
        }
        self._log_event(log_entry)

        return stats

    def complete_sleep_cycle(self,
                           cycle_id: str,
                           results: SleepCycleResults) -> None:
        """
        Mark sleep cycle as complete and save results.

        Args:
            cycle_id: Cycle ID
            results: Sleep cycle results
        """
        results.end_time = time.time()
        results.duration_seconds = results.end_time - results.start_time

        # Save results
        results_file = self.logs_dir / f"{cycle_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(results), f, indent=2)

        self.cycle_history.append(results)
        self.current_cycle = None

        # Log completion
        log_entry = {
            'event': 'sleep_cycle_completed',
            'cycle_id': cycle_id,
            'timestamp': time.time(),
            'duration_seconds': results.duration_seconds,
            'summary': asdict(results)
        }
        self._log_event(log_entry)

    def _generate_cycle_id(self) -> str:
        """Generate unique cycle ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cycle_{timestamp}"

    def _log_event(self, event: Dict) -> None:
        """Log sleep cycle event"""
        log_file = self.logs_dir / "sleep_cycle.log"
        with open(log_file, 'a') as f:
            json.dump(event, f)
            f.write('\n')
