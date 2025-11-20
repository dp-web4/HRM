"""
LLM IRP Plugin - Conversational Intelligence with Iterative Refinement

Implements the IRP protocol for language model inference:
- init_state(): Load model and prepare for generation
- step(): Generate response with temperature annealing
- energy(): Measure response quality/confidence
- halt(): Determine if refinement should stop

Based on Sprout's successful Jetson Nano deployment (November 2025).
"""

import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class LLMIRPPlugin:
    """
    Language Model IRP Plugin for conversational intelligence.

    Supports:
    - Iterative refinement via temperature annealing
    - LoRA adapter loading for personalized models
    - Memory-aware conversation (context from prior exchanges)
    - SNARC salience integration
    - Edge deployment (Jetson Nano validated)
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        base_model: Optional[str] = None,
        device: str = "auto",
        max_tokens: int = 200,
        initial_temperature: float = 0.7,
        min_temperature: float = 0.5,
        temp_reduction: float = 0.04
    ):
        """
        Initialize LLM IRP plugin.

        Args:
            model_path: Path to model or LoRA adapter
            base_model: Base model if using LoRA adapter
            device: Device for inference ("auto", "cuda", "cpu")
            max_tokens: Maximum tokens to generate
            initial_temperature: Starting temperature for sampling
            min_temperature: Minimum temperature (convergence point)
            temp_reduction: Temperature reduction per iteration
        """
        self.model_path = model_path
        self.base_model = base_model or model_path
        self.max_tokens = max_tokens
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temp_reduction = temp_reduction

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        print(f"[LLM IRP] Loading model: {self.model_path}")
        print(f"[LLM IRP] Device: {self.device}")

        # Determine if model path is local or HuggingFace
        # Check for actual model files (config.json + model weights)
        base_path = Path(self.base_model)
        model_is_local = (
            (base_path / "config.json").exists() and
            ((base_path / "model.safetensors").exists() or
             (base_path / "pytorch_model.bin").exists() or
             (base_path / "adapter_config.json").exists())
        )
        tokenizer_kwargs = {"local_files_only": True} if model_is_local else {}
        model_kwargs = {"local_files_only": True} if model_is_local else {}

        print(f"[LLM IRP] Model source: {'local' if model_is_local else 'HuggingFace'}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, **tokenizer_kwargs)

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            **model_kwargs
        )

        # Load LoRA adapter if applicable
        if model_path != self.base_model:
            adapter_config = Path(model_path) / "adapter_config.json"
            if adapter_config.exists():
                try:
                    self.model = PeftModel.from_pretrained(self.model, model_path)
                    print(f"[LLM IRP] Loaded LoRA adapter from {model_path}")
                except Exception as e:
                    print(f"[LLM IRP] Warning: Could not load LoRA adapter: {e}")
            else:
                # No adapter_config.json - might be a full model stored locally
                full_model_path = Path(model_path)
                if full_model_path.exists():
                    print(f"[LLM IRP] Loading full model from local path: {model_path}")
                    # Reload as full model
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map=self.device,
                        local_files_only=True
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                    print(f"[LLM IRP] Loaded full model from {model_path}")

        self.model.eval()
        print(f"[LLM IRP] Model loaded successfully!")

        # State
        self.state = None

    def init_state(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize state for IRP refinement.

        Args:
            question: User's question/prompt
            context: Optional conversation context

        Returns:
            Initial state dict
        """
        # Build prompt with optional context
        if context:
            prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        self.state = {
            'question': question,
            'context': context,
            'prompt': prompt,
            'iteration': 0,
            'temperature': self.initial_temperature,
            'best_response': '',
            'best_energy': float('inf'),
            'responses': [],
            'energies': []
        }

        return self.state

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one IRP iteration: generate response at current temperature.

        Args:
            state: Current IRP state

        Returns:
            Updated state with new response
        """
        iteration = state['iteration']
        temperature = state['temperature']

        print(f"[LLM IRP] Iteration {iteration + 1}, temp={temperature:.3f}")

        # Generate response
        inputs = self.tokenizer(state['prompt'], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (remove prompt)
        if "Answer:" in full_output:
            response = full_output.split("Answer:")[-1].strip()
        else:
            response = full_output[len(state['prompt']):].strip()

        # Calculate energy for this response
        current_energy = self.energy(state, response)

        # Update state
        state['responses'].append(response)
        state['energies'].append(current_energy)

        # Track best response
        if current_energy < state['best_energy']:
            state['best_response'] = response
            state['best_energy'] = current_energy

        # Prepare for next iteration
        state['iteration'] += 1
        state['temperature'] = max(
            self.min_temperature,
            state['temperature'] - self.temp_reduction
        )

        return state

    def energy(self, state: Dict[str, Any], response: str) -> float:
        """
        Calculate energy (quality metric) for response.

        Lower energy = better response

        Metrics:
        - Length penalty (too short or too long)
        - Temperature (lower temp = lower energy)
        - Coherence (rough estimate via response structure)

        Args:
            state: Current IRP state
            response: Response to evaluate

        Returns:
            Energy value (0.0-1.0)
        """
        # Temperature component (lower temp = more refined)
        temp_energy = state['temperature'] / self.initial_temperature

        # Length component (penalize very short or very long)
        length = len(response.split())
        ideal_length = 50  # ~50 words is good conversation length
        length_diff = abs(length - ideal_length)
        length_energy = min(1.0, length_diff / ideal_length)

        # Coherence component (has punctuation, capitalization)
        has_punctuation = any(c in response for c in '.!?')
        has_capital = any(c.isupper() for c in response)
        coherence_energy = 0.0 if (has_punctuation and has_capital) else 0.3

        # Weighted combination
        total_energy = (
            0.4 * temp_energy +      # Temperature most important
            0.3 * length_energy +     # Length second
            0.3 * coherence_energy    # Coherence third
        )

        return total_energy

    def halt(self, state: Dict[str, Any]) -> bool:
        """
        Determine if refinement should stop.

        Stop if:
        - Energy converged (< 0.1)
        - Temperature reached minimum
        - Energy not improving for 2 iterations

        Args:
            state: Current IRP state

        Returns:
            True if should stop, False to continue
        """
        # Check energy convergence
        if state['best_energy'] < 0.1:
            print(f"[LLM IRP] Converged: energy={state['best_energy']:.3f}")
            return True

        # Check temperature minimum
        if state['temperature'] <= self.min_temperature:
            print(f"[LLM IRP] Temperature minimum reached")
            return True

        # Check for energy plateau (not improving)
        if len(state['energies']) >= 3:
            recent_energies = state['energies'][-3:]
            # If last 3 energies are within 0.05, we've plateaued
            if max(recent_energies) - min(recent_energies) < 0.05:
                print(f"[LLM IRP] Energy plateau detected")
                return True

        return False

    def get_result(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Get final result after refinement.

        Args:
            state: Final IRP state

        Returns:
            Tuple of (best_response, info_dict)
        """
        info = {
            'iterations': state['iteration'],
            'final_energy': state['best_energy'],
            'final_temperature': state['temperature'],
            'converged': state['best_energy'] < 0.1,
            'all_energies': state['energies'],
            'response_count': len(state['responses'])
        }

        return state['best_response'], info


class ConversationalLLM:
    """
    High-level conversational interface using LLM IRP plugin.

    Manages:
    - Conversation history (context window)
    - Multi-turn exchanges
    - Memory integration
    - SNARC salience scoring
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        base_model: Optional[str] = None,
        max_history: int = 5,
        irp_iterations: int = 5
    ):
        """
        Initialize conversational LLM.

        Args:
            model_path: Path to model or adapter
            base_model: Base model if using adapter
            max_history: Maximum exchanges to keep in context
            irp_iterations: IRP refinement iterations
        """
        self.llm = LLMIRPPlugin(
            model_path=model_path,
            base_model=base_model
        )
        self.max_history = max_history
        self.irp_iterations = irp_iterations

        # Conversation state
        self.history: list[Tuple[str, str]] = []  # [(question, answer), ...]

    def respond(
        self,
        question: str,
        use_irp: bool = True,
        include_history: bool = True
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate response to question.

        Args:
            question: User's question
            use_irp: Whether to use IRP refinement
            include_history: Whether to include conversation history

        Returns:
            Tuple of (response, irp_info or None)
        """
        # Build context from history
        context = None
        if include_history and self.history:
            recent_history = self.history[-self.max_history:]
            context_lines = []
            for q, a in recent_history:
                context_lines.append(f"Q: {q}\nA: {a}")
            context = "\n\n".join(context_lines)

        if use_irp:
            # IRP refinement
            state = self.llm.init_state(question, context)

            for i in range(self.irp_iterations):
                state = self.llm.step(state)
                if self.llm.halt(state):
                    break

            response, info = self.llm.get_result(state)
        else:
            # Direct generation (no refinement)
            state = self.llm.init_state(question, context)
            state = self.llm.step(state)
            response = state['responses'][0]
            info = None

        # Update history
        self.history.append((question, response))

        return response, info

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def get_history(self) -> list[Tuple[str, str]]:
        """Get conversation history."""
        return self.history.copy()


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("LLM IRP Plugin - Interactive Test")
    print("="*70)

    # Initialize conversational LLM
    conv = ConversationalLLM(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        irp_iterations=5
    )

    # Test questions
    questions = [
        "What is the relationship between knowledge and understanding?",
        "Can you give me an example?",
        "How does this apply to AI systems?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Exchange {i}/{len(questions)}")
        print(f"{'─'*70}")
        print(f"\n🧑 Q: {question}")

        response, info = conv.respond(question, use_irp=True)

        print(f"\n🤖 A: {response}")

        if info:
            print(f"\n📊 IRP: {info['iterations']} iterations, "
                  f"energy={info['final_energy']:.3f}, "
                  f"converged={info['converged']}")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
