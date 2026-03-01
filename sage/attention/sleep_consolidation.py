#!/usr/bin/env python3
"""
Sleep Consolidation Bridge for Attention Kernel

Bridges the attention kernel's experience buffer to the raising pipeline's
sleep training loop. Converts high-salience experiences from kernel format
to training format and invokes LoRA consolidation.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add raising path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'raising'))

try:
    from sage.raising.training.sleep_training import SleepTrainingLoop
    SLEEP_TRAINING_AVAILABLE = True
except ImportError:
    SLEEP_TRAINING_AVAILABLE = False

try:
    from sage.instances.sleep_capability import SleepCapability
    _cap = SleepCapability.detect()
    if not _cap.sleep_lora:
        # Only warn if JSONL/remote are the fallback — not an error
        print(f"[SleepConsolidation] LoRA not available (mode: {_cap.best_mode})")
except Exception:
    if not SLEEP_TRAINING_AVAILABLE:
        print("[SleepConsolidation] Warning: Sleep training not available")


class ExperienceToTrainingConverter:
    """
    Converts attention kernel experiences to raising training format.

    Attention kernel experiences:
        {
            'ts': float,
            'source': str,
            'context': dict,
            'outcome': dict,
            'salience': float
        }

    Raising training format (expected by RaisingTrainingDataBuilder):
        {
            'text': str,              # User input
            'response': str,          # SAGE response
            'salience': float,        # SNARC score
            'timestamp': float,
            'metadata': dict
        }
    """

    def __init__(self):
        self.conversion_count = 0

    def convert_batch(
        self,
        experiences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert batch of kernel experiences to training format.

        Args:
            experiences: List of experience atoms from kernel buffer

        Returns:
            List of training examples in raising format
        """
        training_examples = []

        for exp in experiences:
            try:
                example = self._convert_single(exp)
                if example:
                    training_examples.append(example)
            except Exception as e:
                print(f"[Converter] Failed to convert experience: {e}")
                continue

        self.conversion_count += len(training_examples)
        return training_examples

    def _convert_single(self, exp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert single experience to training format"""
        source = exp.get('source', 'unknown')
        context = exp.get('context', {})
        outcome = exp.get('outcome', {})
        salience = exp.get('salience', 0.0)
        timestamp = exp.get('ts', time.time())

        # Build conversational representation from experience
        # Different sources create different conversation patterns

        if source == 'focus':
            # FOCUS experiences: context gathering
            text = self._build_focus_text(context)
            response = self._build_focus_response(outcome)

        elif source == 'think':
            # THINK experiences: LLM reasoning
            text = context.get('prompt', 'What should I focus on?')
            response = outcome.get('text', outcome.get('response', ''))

        elif source == 'act':
            # ACT experiences: action execution
            text = self._build_action_text(context)
            response = self._build_action_response(outcome)

        else:
            # Generic conversion
            text = str(context)[:200]
            response = str(outcome)[:200]

        # Ensure we have at least minimal text/response
        if not text:
            text = f"Kernel event from {source}"
        if not response:
            response = "Processing complete."

        return {
            'text': text,
            'response': response,
            'salience': salience,
            'timestamp': timestamp,
            'metadata': {
                'source': source,
                'original_context': context,
                'original_outcome': outcome
            }
        }

    def _build_focus_text(self, context: Dict[str, Any]) -> str:
        """Build prompt from FOCUS context"""
        goal = context.get('goal', 'observe')
        tick = context.get('tick', 0)
        return f"In this moment (tick {tick}), my goal is to {goal}. What should I focus on?"

    def _build_focus_response(self, outcome: Dict[str, Any]) -> str:
        """Build response from FOCUS outcome"""
        if 'error' in outcome:
            return f"I encountered an issue: {outcome['error']}"

        status = outcome.get('status', None)
        confidence = outcome.get('confidence', None)
        disagreement = outcome.get('disagreement', None)

        if status == 'no_plugins':
            return "No cognitive plugins are available to process this situation."

        # Build response from available fields
        parts = []
        if confidence is not None:
            if confidence > 0.7:
                parts.append(f"I have high confidence in my assessment (confidence={confidence:.2f}).")
            else:
                parts.append(f"My assessment shows moderate confidence (confidence={confidence:.2f}).")

        if disagreement is not None and disagreement > 0.5:
            parts.append(f"My cognitive modules disagree (disagreement={disagreement:.2f}). I need deeper reasoning.")

        # Fallback if no fields available
        if not parts:
            # Return summary of outcome as-is
            outcome_str = ', '.join(f"{k}={v}" for k, v in outcome.items() if k not in ['raw'])
            return f"Assessment result: {outcome_str}" if outcome_str else "Assessment complete."

        return ' '.join(parts)

    def _build_action_text(self, context: Dict[str, Any]) -> str:
        """Build prompt from ACT context"""
        action_type = context.get('type', 'unknown')
        target = context.get('target', 'environment')
        return f"I am taking action: {action_type} on {target}."

    def _build_action_response(self, outcome: Dict[str, Any]) -> str:
        """Build response from ACT outcome"""
        status = outcome.get('status', 'unknown')
        result = outcome.get('result', 'no result')
        return f"Action completed with status: {status}. Result: {result}"


class SleepConsolidationBridge:
    """
    Bridge between attention kernel SLEEP state and raising sleep training.

    Manages the complete sleep cycle:
    1. Extract high-salience experiences from kernel buffer
    2. Convert to training format
    3. Invoke SleepTrainingLoop for LoRA consolidation
    4. Update kernel with results
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}

        # Model and checkpoint paths
        self.model_path = model_path or config.get(
            'model_path',
            '~/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1'
        )
        self.checkpoint_dir = checkpoint_dir or config.get(
            'checkpoint_dir',
            'logs/attention/sleep_checkpoints'
        )

        # Sleep training configuration
        self.sleep_config = {
            'min_salience': config.get('min_salience', 0.6),
            'max_experiences': config.get('max_experiences', 20),
            'epochs': config.get('epochs', 3),
            'learning_rate': config.get('learning_rate', 1e-5),
            'device': config.get('device', 'cuda')
        }

        # Components
        self.converter = ExperienceToTrainingConverter()
        self.sleep_trainer = None
        self.enabled = SLEEP_TRAINING_AVAILABLE and config.get('enabled', True)

        # Statistics
        self.sleep_cycles_completed = 0
        self.total_experiences_consolidated = 0
        self.sleep_history = []

        print(f"[SleepConsolidationBridge] Initialized")
        print(f"  Enabled: {self.enabled}")
        print(f"  Model: {self.model_path}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Min salience: {self.sleep_config['min_salience']}")

    async def consolidate(
        self,
        experience_buffer: Any,
        kernel_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute sleep consolidation cycle.

        Args:
            experience_buffer: Kernel's ExperienceBuffer with high-salience atoms
            kernel_stats: Optional kernel statistics for context

        Returns:
            Dictionary with consolidation results
        """
        if not self.enabled:
            print("[SleepConsolidationBridge] Consolidation disabled")
            return {
                'status': 'disabled',
                'message': 'Sleep consolidation not available or disabled'
            }

        print(f"\n{'='*60}")
        print(f"SLEEP CONSOLIDATION - STARTING")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Step 1: Extract high-salience experiences
        min_salience = self.sleep_config['min_salience']
        max_experiences = self.sleep_config['max_experiences']

        top_experiences = experience_buffer.get_top_k(
            min(max_experiences, experience_buffer.size)
        )

        # Filter by salience threshold
        high_salience = [
            exp for exp in top_experiences
            if exp.get('salience', 0.0) >= min_salience
        ]

        print(f"Buffer size: {experience_buffer.size}")
        print(f"High-salience experiences (>={min_salience}): {len(high_salience)}")

        if len(high_salience) == 0:
            print("No experiences meet salience threshold - skipping consolidation")
            return {
                'status': 'skipped',
                'reason': 'no_high_salience_experiences',
                'buffer_size': experience_buffer.size,
                'min_salience': min_salience
            }

        # Step 2: Convert to training format
        print("Converting experiences to training format...")
        training_data = self.converter.convert_batch(high_salience)

        if len(training_data) == 0:
            print("No experiences could be converted - skipping consolidation")
            return {
                'status': 'skipped',
                'reason': 'conversion_failed',
                'high_salience_count': len(high_salience)
            }

        print(f"Converted {len(training_data)} experiences to training format")

        # Step 3: Save training data to temporary buffer
        temp_buffer_path = Path(self.checkpoint_dir) / 'temp_experience_buffer.json'
        temp_buffer_path.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_buffer_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"Training data saved to: {temp_buffer_path}")

        # Step 4: Initialize sleep trainer (lazy)
        if self.sleep_trainer is None:
            self.sleep_trainer = SleepTrainingLoop(
                model_path=self.model_path,
                experience_buffer_path=str(temp_buffer_path),
                checkpoint_dir=self.checkpoint_dir,
                device=self.sleep_config['device']
            )
        else:
            # Update buffer path for existing trainer
            self.sleep_trainer.experience_buffer_path = temp_buffer_path
            self.sleep_trainer.data_builder = None  # Reset data builder

        # Step 5: Run sleep training cycle
        print("\nInvoking sleep training loop...")
        try:
            results = self.sleep_trainer.run_sleep_cycle(
                min_salience=0.0,  # Already filtered
                max_experiences=None,  # Use all converted
                epochs=self.sleep_config['epochs'],
                learning_rate=self.sleep_config['learning_rate'],
                save_checkpoint=True
            )

            # Update statistics
            self.sleep_cycles_completed += 1
            self.total_experiences_consolidated += results['num_experiences']
            self.sleep_history.append(results)

            # Add consolidation metadata
            results['consolidation_time'] = time.time() - start_time
            results['buffer_size_before'] = experience_buffer.size
            results['high_salience_extracted'] = len(high_salience)
            results['training_examples_created'] = len(training_data)

            print(f"\n{'='*60}")
            print(f"SLEEP CONSOLIDATION - COMPLETE")
            print(f"Cycle: {self.sleep_cycles_completed}")
            print(f"Experiences consolidated: {results['num_experiences']}")
            print(f"Final loss: {results.get('final_loss', 'N/A')}")
            print(f"Time: {results['consolidation_time']:.2f}s")
            print(f"{'='*60}\n")

            return results

        except Exception as e:
            print(f"[SleepConsolidationBridge] Error during consolidation: {e}")
            import traceback
            traceback.print_exc()

            return {
                'status': 'error',
                'error': str(e),
                'high_salience_count': len(high_salience),
                'training_data_count': len(training_data)
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get consolidation statistics"""
        return {
            'enabled': self.enabled,
            'sleep_cycles_completed': self.sleep_cycles_completed,
            'total_experiences_consolidated': self.total_experiences_consolidated,
            'sleep_history': self.sleep_history,
            'latest_cycle': self.sleep_history[-1] if self.sleep_history else None,
            'model_path': self.model_path,
            'checkpoint_dir': self.checkpoint_dir,
            'config': self.sleep_config
        }


if __name__ == '__main__':
    # Test conversion
    print("Testing ExperienceToTrainingConverter...")

    converter = ExperienceToTrainingConverter()

    # Sample experiences
    test_experiences = [
        {
            'ts': time.time(),
            'source': 'focus',
            'context': {'goal': 'explore', 'tick': 42},
            'outcome': {'confidence': 0.8, 'disagreement': 0.1},
            'salience': 0.75
        },
        {
            'ts': time.time(),
            'source': 'think',
            'context': {'prompt': 'What should I focus on next?'},
            'outcome': {'text': 'I should prioritize high-uncertainty observations.'},
            'salience': 0.85
        },
        {
            'ts': time.time(),
            'source': 'act',
            'context': {'type': 'observe', 'target': 'environment'},
            'outcome': {'status': 'success', 'result': 'observation captured'},
            'salience': 0.65
        }
    ]

    training_data = converter.convert_batch(test_experiences)

    print(f"\nConverted {len(training_data)} experiences:")
    for i, example in enumerate(training_data):
        print(f"\nExample {i+1}:")
        print(f"  Text: {example['text'][:80]}...")
        print(f"  Response: {example['response'][:80]}...")
        print(f"  Salience: {example['salience']:.3f}")

    print("\n✓ Conversion test complete")
