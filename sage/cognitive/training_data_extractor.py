"""
SAGE Training Data Extraction System

Extracts training data from conversation history for:
1. Sleep cycle fine-tuning (full Qwen model refinement)
2. Fast-path model training (10M param model for performatory acknowledgments)
3. Augmented experience generation (variations for robust learning)

The system captures:
- Successful slow-path interactions (deep reasoning examples)
- Fast-path worthy exchanges (context-adjacent acknowledgments)
- SNARC-scored experiences (salience-weighted training data)
- Handoff patterns (when to defer to slow path)
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib


@dataclass
class TrainingExample:
    """Single training example with metadata"""
    input_text: str
    output_text: str
    context_history: List[Tuple[str, str]]  # (speaker, text) tuples

    # Metadata
    timestamp: float
    path_used: str  # "fast" or "slow"
    salience_score: float
    latency_ms: float

    # Classification
    example_type: str  # "deep_reasoning", "fast_ack", "handoff", "general"
    confidence: float  # How confident was the response?

    # Quality indicators
    user_feedback: Optional[str] = None  # "positive", "negative", "neutral", None
    learned_pattern: bool = False  # Was this example used to learn a pattern?

    # Augmentation tracking
    is_augmented: bool = False
    augmentation_type: Optional[str] = None
    original_hash: Optional[str] = None

    def get_hash(self) -> str:
        """Generate unique hash for deduplication"""
        content = f"{self.input_text}|{self.output_text}"
        return hashlib.md5(content.encode()).hexdigest()


class TrainingDataExtractor:
    """
    Extracts and organizes training data from SAGE conversation history.

    Separates data into:
    1. Deep reasoning examples (for sleep cycle fine-tuning)
    2. Fast acknowledgment examples (for tiny fast-path model)
    3. Handoff examples (learning when to defer to slow path)
    """

    def __init__(self, output_dir: str = "training_data"):
        """
        Initialize training data extractor.

        Args:
            output_dir: Directory to store extracted training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Separate storage for different training purposes
        self.deep_reasoning_examples: List[TrainingExample] = []
        self.fast_ack_examples: List[TrainingExample] = []
        self.handoff_examples: List[TrainingExample] = []
        self.general_examples: List[TrainingExample] = []

        # Statistics
        self.stats = defaultdict(int)

    def extract_from_memory(self, memory_manager) -> None:
        """
        Extract training examples from SNARC memory manager.

        Args:
            memory_manager: SNARCMemoryManager instance with conversation history
        """
        # Extract from conversation buffer (recent experiences)
        for turn_idx in range(0, len(memory_manager.conversation_buffer) - 1, 2):
            # Get user input and assistant response
            if turn_idx + 1 >= len(memory_manager.conversation_buffer):
                break

            user_turn = memory_manager.conversation_buffer[turn_idx]
            assistant_turn = memory_manager.conversation_buffer[turn_idx + 1]

            if user_turn.speaker != "User" or assistant_turn.speaker != "Assistant":
                continue

            # Get context history (last 10 turns before this exchange)
            context_start = max(0, turn_idx - 10)
            context_history = [
                (t.speaker, t.text)
                for t in memory_manager.conversation_buffer[context_start:turn_idx]
            ]

            # Create training example
            example = TrainingExample(
                input_text=user_turn.text,
                output_text=assistant_turn.text,
                context_history=context_history,
                timestamp=assistant_turn.timestamp,
                path_used=assistant_turn.metadata.get('path', 'unknown'),
                salience_score=assistant_turn.salience_score,
                latency_ms=assistant_turn.metadata.get('llm_latency', 0) * 1000,
                example_type=self._classify_example(user_turn, assistant_turn),
                confidence=assistant_turn.metadata.get('confidence', 0.5),
                learned_pattern=assistant_turn.metadata.get('learned', False)
            )

            # Categorize and store
            self._categorize_example(example)
            self.stats['total_extracted'] += 1

        # Extract from long-term memory (high-salience experiences)
        for turn_idx in range(0, len(memory_manager.long_term_memory) - 1, 2):
            if turn_idx + 1 >= len(memory_manager.long_term_memory):
                break

            user_turn = memory_manager.long_term_memory[turn_idx]
            assistant_turn = memory_manager.long_term_memory[turn_idx + 1]

            if user_turn.speaker != "User" or assistant_turn.speaker != "Assistant":
                continue

            # Long-term memories get special treatment (already high-salience)
            example = TrainingExample(
                input_text=user_turn.text,
                output_text=assistant_turn.text,
                context_history=[],  # Context not preserved in long-term
                timestamp=assistant_turn.timestamp,
                path_used=assistant_turn.metadata.get('path', 'slow'),
                salience_score=assistant_turn.salience_score,
                latency_ms=assistant_turn.metadata.get('llm_latency', 0) * 1000,
                example_type="deep_reasoning",  # Long-term = deep reasoning
                confidence=1.0,  # High-salience = high confidence
                learned_pattern=False
            )

            self._categorize_example(example)
            self.stats['longterm_extracted'] += 1

    def _classify_example(self, user_turn, assistant_turn) -> str:
        """
        Classify training example by type.

        Types:
        - deep_reasoning: Complex questions requiring thought
        - fast_ack: Simple acknowledgments, greetings, confirmations
        - handoff: Cases where fast path should defer to slow path
        - general: Everything else
        """
        user_text = user_turn.text.lower()
        assistant_text = assistant_turn.text.lower()
        path = assistant_turn.metadata.get('path', 'unknown')

        # Fast acknowledgments (short, high confidence, fast path)
        if path == 'fast' and len(assistant_text) < 50:
            # Check if it's a performatory acknowledgment
            ack_keywords = [
                'thanks', 'thank you', 'ok', 'okay', 'got it', 'sure',
                'yes', 'no', 'hello', 'hi', 'bye', 'see you'
            ]
            if any(kw in assistant_text for kw in ack_keywords):
                return "fast_ack"

        # Handoff indicators (should have gone to slow path but didn't)
        handoff_indicators = [
            'hmm', 'let me think', 'interesting question',
            'that\'s complex', 'need to consider'
        ]
        if path == 'fast' and any(ind in assistant_text for ind in handoff_indicators):
            return "handoff"

        # Deep reasoning (slow path, high salience, complex)
        if path == 'slow' and assistant_turn.salience_score > 0.5:
            return "deep_reasoning"

        return "general"

    def _categorize_example(self, example: TrainingExample) -> None:
        """Categorize example into appropriate storage"""
        if example.example_type == "deep_reasoning":
            self.deep_reasoning_examples.append(example)
            self.stats['deep_reasoning'] += 1
        elif example.example_type == "fast_ack":
            self.fast_ack_examples.append(example)
            self.stats['fast_ack'] += 1
        elif example.example_type == "handoff":
            self.handoff_examples.append(example)
            self.stats['handoff'] += 1
        else:
            self.general_examples.append(example)
            self.stats['general'] += 1

    def augment_examples(self, examples: List[TrainingExample],
                        augmentation_strategies: List[str]) -> List[TrainingExample]:
        """
        Generate augmented variations of training examples.

        Augmentation strategies (inspired by HRM's approach):
        - paraphrase: Rephrase input/output while preserving meaning
        - context_shift: Add/remove context turns
        - noise_injection: Add minor variations (typos, informal language)
        - formality_shift: Adjust formality level

        Args:
            examples: Original training examples
            augmentation_strategies: List of strategies to apply

        Returns:
            List of augmented examples (includes originals)
        """
        augmented = []

        for example in examples:
            # Always include original
            augmented.append(example)

            # Generate variations
            if 'paraphrase' in augmentation_strategies:
                # Simplified paraphrase (would use LLM in production)
                aug_example = TrainingExample(
                    input_text=self._simple_paraphrase(example.input_text),
                    output_text=example.output_text,
                    context_history=example.context_history,
                    timestamp=time.time(),
                    path_used=example.path_used,
                    salience_score=example.salience_score,
                    latency_ms=example.latency_ms,
                    example_type=example.example_type,
                    confidence=example.confidence * 0.9,  # Slightly lower confidence
                    is_augmented=True,
                    augmentation_type='paraphrase',
                    original_hash=example.get_hash()
                )
                augmented.append(aug_example)
                self.stats['augmented'] += 1

            if 'context_shift' in augmentation_strategies and len(example.context_history) > 2:
                # Remove half the context
                reduced_context = example.context_history[len(example.context_history)//2:]
                aug_example = TrainingExample(
                    input_text=example.input_text,
                    output_text=example.output_text,
                    context_history=reduced_context,
                    timestamp=time.time(),
                    path_used=example.path_used,
                    salience_score=example.salience_score * 0.8,
                    latency_ms=example.latency_ms,
                    example_type=example.example_type,
                    confidence=example.confidence * 0.85,
                    is_augmented=True,
                    augmentation_type='context_shift',
                    original_hash=example.get_hash()
                )
                augmented.append(aug_example)
                self.stats['augmented'] += 1

        return augmented

    def _simple_paraphrase(self, text: str) -> str:
        """
        Simple paraphrasing (placeholder for LLM-based paraphrasing).

        In production, this would use the slow-path LLM to generate
        semantic-preserving variations.
        """
        # Basic substitutions as placeholder
        substitutions = {
            "what is": "what's",
            "how do": "how does",
            "can you": "could you",
            "tell me": "explain",
        }

        paraphrased = text
        for old, new in substitutions.items():
            paraphrased = paraphrased.replace(old, new)

        return paraphrased

    def save_training_data(self, format: str = "jsonl") -> Dict[str, Path]:
        """
        Save training data to disk in specified format.

        Args:
            format: "jsonl", "csv", or "huggingface"

        Returns:
            Dictionary mapping dataset type to file path
        """
        saved_files = {}

        if format == "jsonl":
            # Save deep reasoning examples (for sleep cycle fine-tuning)
            deep_file = self.output_dir / "deep_reasoning.jsonl"
            self._save_jsonl(self.deep_reasoning_examples, deep_file)
            saved_files['deep_reasoning'] = deep_file

            # Save fast acknowledgment examples (for tiny model training)
            fast_file = self.output_dir / "fast_ack.jsonl"
            self._save_jsonl(self.fast_ack_examples, fast_file)
            saved_files['fast_ack'] = fast_file

            # Save handoff examples (learning when to defer)
            handoff_file = self.output_dir / "handoff.jsonl"
            self._save_jsonl(self.handoff_examples, handoff_file)
            saved_files['handoff'] = handoff_file

            # Save general examples
            general_file = self.output_dir / "general.jsonl"
            self._save_jsonl(self.general_examples, general_file)
            saved_files['general'] = general_file

        # Save statistics
        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(dict(self.stats), f, indent=2)
        saved_files['stats'] = stats_file

        return saved_files

    def _save_jsonl(self, examples: List[TrainingExample], filepath: Path) -> None:
        """Save examples in JSONL format (one JSON per line)"""
        with open(filepath, 'w') as f:
            for example in examples:
                json.dump(asdict(example), f)
                f.write('\n')

    def get_statistics(self) -> Dict:
        """Get extraction statistics"""
        return {
            'total_extracted': self.stats['total_extracted'],
            'from_longterm': self.stats['longterm_extracted'],
            'deep_reasoning': len(self.deep_reasoning_examples),
            'fast_ack': len(self.fast_ack_examples),
            'handoff': len(self.handoff_examples),
            'general': len(self.general_examples),
            'augmented': self.stats['augmented'],
        }
