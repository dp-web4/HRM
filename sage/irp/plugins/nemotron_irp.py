#!/usr/bin/env python3
"""
Nemotron IRP Plugin - Language reasoning for SAGE

Integrates NVIDIA Nemotron-H-4B-Instruct-128K as an IRP plugin for strategic
language reasoning within the SAGE orchestration framework.

Model: nvidia/Nemotron-H-4B-Instruct-128K
- 4B parameters (hybrid Mamba-Transformer)
- 128K context window
- NeurIPS 2025 accepted
- Publicly available

This follows the same pattern as qwen_7b_irp.py but adapted for Nemotron's
characteristics and optimizations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional, List
import time
from pathlib import Path

# Import IRP base classes
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import IRPPlugin, IRPState


class NemotronIRPPlugin(IRPPlugin):
    """
    Nemotron as an Iterative Refinement Protocol plugin.

    This plugin wraps Nemotron for integration into SAGE's consciousness
    orchestration framework. It treats language generation as progressive
    refinement toward semantic convergence.

    Key Features:
    - 4B parameters (7.5x smaller than Q3-Omni)
    - 128K context window
    - Jetson-optimized (tested on Orin platform)
    - Progressive token generation (step = +1 token)
    - Energy-based convergence detection
    - ATP budget awareness

    Integration Roles:
    1. Language IRP Plugin (primary)
    2. Semantic importance scorer
    3. Strategic decision reasoner
    4. Q&A interface over observations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Nemotron IRP plugin.

        Args:
            config: Configuration dictionary with optional keys:
                - model_path: Path to model (default: model-zoo/sage/language-models/nemotron-4b-instruct)
                - device: Device to use (default: 'cuda' if available)
                - dtype: Data type (default: torch.bfloat16)
                - max_length: Maximum generation length (default: 2048)
                - halt_threshold: Energy convergence threshold (default: 0.01)
                - lazy_load: Don't load model until first use (default: True)
        """
        super().__init__(config or {})

        # Configuration
        self.model_path = self.config.get(
            'model_path',
            'model-zoo/sage/language-models/nemotron-h-4b-instruct-128k'
        )
        self.device = self.config.get(
            'device',
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.dtype = self.config.get('dtype', torch.bfloat16)
        self.max_length = self.config.get('max_length', 2048)
        self.halt_threshold = self.config.get('halt_eps', 0.01)
        self.lazy_load = self.config.get('lazy_load', True)

        # Model components (loaded lazily by default)
        self.tokenizer = None
        self.model = None
        self._loaded = False

        # Metrics
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        self.generation_count = 0

        if not self.lazy_load:
            self._ensure_loaded()

    def _ensure_loaded(self):
        """Load model and tokenizer if not already loaded."""
        if self._loaded:
            return

        print(f"Loading Nemotron from {self.model_path}...")
        start = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )

        self.model.eval()  # Inference mode

        load_time = time.time() - start
        print(f"✅ Nemotron loaded in {load_time:.1f}s")

        # Memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 GPU Memory: {allocated:.2f} GB allocated")

        self._loaded = True

    def init_state(self, x0: Any, task_ctx: Optional[Dict] = None) -> IRPState:
        """
        Initialize refinement state from input.

        Args:
            x0: Initial input (prompt string or conversation history)
            task_ctx: Optional task context with:
                - conversation_history: List of previous messages
                - system_prompt: System-level instructions
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter

        Returns:
            IRPState with tokenized input and metadata
        """
        self._ensure_loaded()

        task_ctx = task_ctx or {}

        # Format prompt (handle both string and conversation format)
        if isinstance(x0, str):
            prompt = x0
        elif isinstance(x0, list):
            # Conversation format: [{"role": "user", "content": "..."}]
            prompt = self.tokenizer.apply_chat_template(
                x0,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            raise ValueError(f"Unsupported input type: {type(x0)}")

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Create state
        state = IRPState(
            x=inputs['input_ids'],
            step_idx=0,
            metadata={
                'prompt': prompt,
                'input_length': inputs['input_ids'].shape[1],
                'temperature': task_ctx.get('temperature', 0.7),
                'top_p': task_ctx.get('top_p', 0.95),
                'attention_mask': inputs.get('attention_mask'),
            }
        )

        return state

    def step(self, state: IRPState, noise_schedule: Optional[Any] = None) -> IRPState:
        """
        Execute one refinement step (generate one token).

        Args:
            state: Current refinement state
            noise_schedule: Optional noise schedule (unused for autoregressive generation)

        Returns:
            Updated state with new token
        """
        self._ensure_loaded()

        start = time.time()

        # Generate one token
        with torch.no_grad():
            outputs = self.model.generate(
                state.x,
                max_new_tokens=1,
                temperature=state.metadata['temperature'],
                top_p=state.metadata['top_p'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=state.metadata.get('attention_mask'),
            )

        # Update state
        state.x = outputs
        state.step_idx += 1

        # Update attention mask if present
        if state.metadata.get('attention_mask') is not None:
            new_mask = torch.ones(
                (outputs.shape[0], outputs.shape[1]),
                dtype=torch.long,
                device=self.device
            )
            state.metadata['attention_mask'] = new_mask

        # Track metrics
        self.total_tokens_generated += 1
        self.total_inference_time += (time.time() - start)

        return state

    def energy(self, state: IRPState) -> float:
        """
        Compute energy of current state.

        Energy measures how far we are from semantic convergence.
        Lower energy = more refined output.

        For language generation, we use:
        - Token count remaining (simple heuristic)
        - Log probability (if available)
        - Semantic stability (future enhancement)

        Args:
            state: Current state

        Returns:
            Energy value (lower is better)
        """
        # Simple energy: tokens remaining until max length
        current_length = state.x.shape[1]
        input_length = state.metadata['input_length']
        tokens_generated = current_length - input_length

        # Energy decreases as we generate more tokens
        # But increases if we approach max length (penalty for verbosity)
        if tokens_generated < 50:
            # Early tokens have high energy (not refined yet)
            energy = 1.0 - (tokens_generated / 50.0)
        else:
            # After 50 tokens, energy is low but increases toward max
            energy = 0.1 + (tokens_generated / self.max_length) * 0.5

        return energy

    def halt(self, history: List[float]) -> bool:
        """
        Detect convergence (stopping criterion).

        Args:
            history: List of energy values from previous steps

        Returns:
            True if converged (should stop), False otherwise
        """
        if len(history) < 2:
            return False

        # Check for energy convergence
        recent_energy = history[-1]
        prev_energy = history[-2]
        energy_change = abs(recent_energy - prev_energy)

        if energy_change < self.halt_threshold:
            return True

        # Check for EOS token
        if self._loaded and self.tokenizer:
            last_token = self.current_state.x[0, -1].item() if hasattr(self, 'current_state') else None
            if last_token == self.tokenizer.eos_token_id:
                return True

        # Check max length
        if hasattr(self, 'current_state'):
            current_length = self.current_state.x.shape[1]
            if current_length >= self.max_length:
                return True

        return False

    def get_output(self, state: IRPState) -> str:
        """
        Extract generated text from state.

        Args:
            state: Final state

        Returns:
            Generated text (decoded)
        """
        self._ensure_loaded()

        # Decode only the generated tokens (skip input)
        input_length = state.metadata['input_length']
        generated_ids = state.x[:, input_length:]

        text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return text

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.

        Returns:
            Dictionary with performance stats
        """
        avg_time = (
            self.total_inference_time / self.total_tokens_generated
            if self.total_tokens_generated > 0
            else 0.0
        )

        tokens_per_sec = (
            self.total_tokens_generated / self.total_inference_time
            if self.total_inference_time > 0
            else 0.0
        )

        return {
            'total_tokens_generated': self.total_tokens_generated,
            'total_inference_time': self.total_inference_time,
            'generation_count': self.generation_count,
            'avg_time_per_token': avg_time,
            'tokens_per_second': tokens_per_sec,
            'model_loaded': self._loaded,
        }

    def unload(self):
        """Unload model from memory (for resource management)."""
        if self._loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Nemotron unloaded from memory")


def test_nemotron_irp():
    """Test Nemotron IRP plugin."""
    print("=" * 80)
    print("Testing Nemotron IRP Plugin")
    print("=" * 80)
    print()

    # Initialize plugin
    config = {
        'lazy_load': False,  # Load immediately for testing
        'max_length': 100,
    }

    plugin = NemotronIRPPlugin(config)

    # Test case 1: Simple prompt
    print("Test 1: Simple prompt")
    print("-" * 80)

    prompt = "What is consciousness?"
    state = plugin.init_state(prompt)

    print(f"Initial state: {state.x.shape[1]} tokens")
    print(f"Initial energy: {plugin.energy(state):.4f}")
    print()

    # Generate a few tokens
    history = []
    for i in range(10):
        state = plugin.step(state)
        e = plugin.energy(state)
        history.append(e)

        if plugin.halt(history):
            print(f"Converged after {i+1} steps")
            break

        if i % 5 == 0:
            partial_output = plugin.get_output(state)
            print(f"Step {i}: {partial_output[:100]}...")

    # Final output
    output = plugin.get_output(state)
    print()
    print("Generated text:")
    print(output)
    print()

    # Metrics
    metrics = plugin.get_metrics()
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 80)
    print("✅ Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_nemotron_irp()
