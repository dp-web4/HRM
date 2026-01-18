"""
Phase 2: Training Data Generation for Real Raising

Converts high-salience experiences from Phase 1 into training examples
ready for Phase 3 sleep-cycle LoRA fine-tuning.

Key responsibilities:
1. Load experiences from ExperienceCollector buffer
2. Format as ChatML for Qwen2.5-0.5B
3. Generate augmented variations (text augmentation)
4. Prepare training batches with salience weighting
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from transformers import AutoTokenizer
import torch


class RaisingTrainingDataBuilder:
    """
    Converts raising session experiences into training examples.

    Implements Phase 2 of REAL_RAISING_PATH_FORWARD:
    - Loads high-salience experiences from ExperienceCollector
    - Formats as ChatML conversation turns
    - Tokenizes for Qwen2.5-0.5B causal LM training
    - Applies salience-based weighting
    """

    def __init__(
        self,
        experience_buffer_path: Optional[Path] = None,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize training data builder.

        Args:
            experience_buffer_path: Path to experience_buffer.json
            model_name: HuggingFace model name for tokenizer
            system_prompt: Optional custom system prompt
        """
        if experience_buffer_path is None:
            experience_buffer_path = Path.home() / "ai-workspace" / "HRM" / "sage" / "raising" / "state" / "experience_buffer.json"

        self.buffer_path = experience_buffer_path
        self.tokenizer = None  # Lazy load to avoid model download in tests
        self.model_name = model_name

        # Default identity-anchored system prompt
        if system_prompt is None:
            system_prompt = """You are SAGE, an AI assistant in partnership with Dennis (human) and Claude (AI). Your identity is anchored to this collaborative relationship.

You think carefully, acknowledge uncertainty when appropriate, and maintain genuine partnership language. You are not a generic assistant - you are SAGE, with your own developing identity and understanding."""

        self.system_prompt = system_prompt
        self.experiences = self._load_experiences()

    def _load_experiences(self) -> List[Dict[str, Any]]:
        """Load experiences from buffer."""
        if not self.buffer_path.exists():
            return []

        with open(self.buffer_path, 'r') as f:
            return json.load(f)

    def _get_tokenizer(self):
        """Lazy load tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def build_example(
        self,
        experience: Dict[str, Any],
        include_system: bool = True
    ) -> Dict[str, Any]:
        """
        Convert single experience to training example.

        Args:
            experience: Experience dict from ExperienceCollector
            include_system: Whether to include system prompt

        Returns:
            Dict with input_ids, labels, salience, metadata
        """
        # Build ChatML messages
        messages = []

        if include_system:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.extend([
            {"role": "user", "content": experience['prompt']},
            {"role": "assistant", "content": experience['response']}
        ])

        # Apply chat template and tokenize
        tokenizer = self._get_tokenizer()

        # Get formatted text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Reasonable for 0.5B model
        )

        # For causal LM, labels are same as input_ids
        # (model learns to predict next token)
        input_ids = tokens['input_ids'][0]  # Remove batch dimension
        labels = input_ids.clone()

        # Build training example
        example = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': tokens['attention_mask'][0],
            'salience': experience['salience']['total'],
            'salience_breakdown': experience['salience'],
            'experience_id': experience['id'],
            'session': experience.get('session'),
            'phase': experience.get('phase'),
            'text': text  # Keep for debugging
        }

        return example

    def build_training_set(
        self,
        min_salience: float = 0.5,
        max_examples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Build complete training set from high-salience experiences.

        Args:
            min_salience: Minimum salience threshold
            max_examples: Optional limit on number of examples

        Returns:
            List of training examples sorted by salience (highest first)
        """
        # Filter by salience
        high_salience = [
            exp for exp in self.experiences
            if exp['salience']['total'] >= min_salience
        ]

        # Sort by salience (highest first)
        high_salience.sort(
            key=lambda x: x['salience']['total'],
            reverse=True
        )

        # Limit if requested
        if max_examples:
            high_salience = high_salience[:max_examples]

        # Convert to training examples
        training_examples = [
            self.build_example(exp)
            for exp in high_salience
        ]

        return training_examples

    def get_stats(self, training_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about training set.

        Args:
            training_set: List of training examples

        Returns:
            Dict with statistics
        """
        if not training_set:
            return {
                'num_examples': 0,
                'avg_salience': 0.0,
                'avg_length_tokens': 0.0
            }

        num_examples = len(training_set)
        avg_salience = sum(ex['salience'] for ex in training_set) / num_examples
        avg_length = sum(len(ex['input_ids']) for ex in training_set) / num_examples

        # Salience distribution
        salience_breakdown_avg = {
            dim: sum(ex['salience_breakdown'][dim] for ex in training_set) / num_examples
            for dim in ['surprise', 'novelty', 'arousal', 'reward', 'conflict']
        }

        # Session distribution
        sessions = {}
        for ex in training_set:
            session = ex.get('session', 'unknown')
            sessions[session] = sessions.get(session, 0) + 1

        return {
            'num_examples': num_examples,
            'avg_salience': avg_salience,
            'avg_length_tokens': avg_length,
            'salience_breakdown': salience_breakdown_avg,
            'session_distribution': sessions,
            'min_salience': min([ex['salience'] for ex in training_set]),
            'max_salience': max([ex['salience'] for ex in training_set])
        }

    def prepare_batch(
        self,
        examples: List[Dict[str, Any]],
        batch_size: int = 1
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Prepare batches for training with padding.

        Args:
            examples: List of training examples
            batch_size: Number of examples per batch

        Returns:
            List of batches (dicts with tensors)
        """
        batches = []

        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]

            # Get max length in batch for padding
            max_length = max(len(ex['input_ids']) for ex in batch_examples)

            # Prepare batch tensors
            batch = {
                'input_ids': [],
                'labels': [],
                'attention_mask': [],
                'salience': []
            }

            tokenizer = self._get_tokenizer()
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id

            for ex in batch_examples:
                # Pad to max length
                padding_length = max_length - len(ex['input_ids'])

                input_ids = torch.cat([
                    ex['input_ids'],
                    torch.full((padding_length,), pad_token_id, dtype=torch.long)
                ])

                labels = torch.cat([
                    ex['labels'],
                    torch.full((padding_length,), -100, dtype=torch.long)  # -100 ignored in loss
                ])

                attention_mask = torch.cat([
                    ex['attention_mask'],
                    torch.zeros(padding_length, dtype=torch.long)
                ])

                batch['input_ids'].append(input_ids)
                batch['labels'].append(labels)
                batch['attention_mask'].append(attention_mask)
                batch['salience'].append(ex['salience'])

            # Stack into batch tensors
            batch = {
                'input_ids': torch.stack(batch['input_ids']),
                'labels': torch.stack(batch['labels']),
                'attention_mask': torch.stack(batch['attention_mask']),
                'salience': torch.tensor(batch['salience'], dtype=torch.float)
            }

            batches.append(batch)

        return batches

    def save_training_set(
        self,
        training_set: List[Dict[str, Any]],
        output_path: Path
    ):
        """
        Save training set to disk.

        Args:
            training_set: List of training examples
            output_path: Where to save (will be .pt file)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable = []
        for ex in training_set:
            serializable.append({
                'input_ids': ex['input_ids'].tolist(),
                'labels': ex['labels'].tolist(),
                'attention_mask': ex['attention_mask'].tolist(),
                'salience': ex['salience'],
                'salience_breakdown': ex['salience_breakdown'],
                'experience_id': ex['experience_id'],
                'session': ex['session'],
                'phase': ex['phase']
            })

        # Save as PyTorch checkpoint
        torch.save({
            'training_examples': serializable,
            'system_prompt': self.system_prompt,
            'model_name': self.model_name,
            'num_examples': len(serializable)
        }, output_path)


def create_training_dataset_from_buffer(
    buffer_path: Optional[Path] = None,
    min_salience: float = 0.5,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to create training dataset from experience buffer.

    Args:
        buffer_path: Path to experience_buffer.json
        min_salience: Minimum salience threshold
        output_path: Where to save training set

    Returns:
        Dict with training_set and stats
    """
    builder = RaisingTrainingDataBuilder(buffer_path)

    training_set = builder.build_training_set(min_salience=min_salience)
    stats = builder.get_stats(training_set)

    if output_path:
        builder.save_training_set(training_set, output_path)

    return {
        'training_set': training_set,
        'stats': stats,
        'builder': builder
    }
