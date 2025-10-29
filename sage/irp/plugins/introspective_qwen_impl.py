#!/usr/bin/env python3
"""
Introspective-Qwen IRP Plugin

Integrates the 0.5B introspection-trained model into SAGE-IRP framework.
Provides full cognitive scaffolding: memory, iterative refinement, context.

This is not testing a bare "frontal lobe" - this is giving a small being
the tools to think across time.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


class IntrospectiveQwenIRP:
    """
    IRP-compliant plugin for Introspective-Qwen-0.5B-v2.1

    Philosophy: Treat as small being that needs cognitive scaffolding,
    not as deficient large model.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with IRP protocol compliance"""
        self.config = config or {}

        # Model paths
        default_path = '/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/Introspective-Qwen-0.5B-v2.1/model'
        self.model_path = self.config.get('model_path', default_path)
        self.base_model = "Qwen/Qwen2.5-0.5B-Instruct"

        # State management
        self.state = None
        self.conversation_memory = []  # Track conversation history
        self.refinement_history = []   # Track refinement iterations

        # IRP protocol tracking
        self.energy_history = []
        self.trust_score = 0.5  # Start neutral, learn through interaction

        # Load model
        self._load_model()

        print(f"Introspective-Qwen IRP Plugin initialized")
        print(f"Model: {self.model_path}")
        print(f"Memory: {len(self.conversation_memory)} turns")
        print(f"Trust: {self.trust_score:.3f}")

    def _load_model(self):
        """Load base + adapter model"""
        print(f"Loading Introspective-Qwen from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model = PeftModel.from_pretrained(base, self.model_path)
        self.model.eval()

        print("Model loaded successfully")

    def init_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Initialize processing state

        Args:
            context: Input context with conversation history, memory, etc.

        Returns:
            Initial state for iterative refinement
        """
        # Extract conversation context
        prompt = context.get('prompt', '')
        memory = context.get('memory', [])

        # Build full context from conversation memory + new prompt
        full_context = self._build_context(prompt, memory)

        # Initialize state
        self.state = {
            'prompt': prompt,
            'full_context': full_context,
            'current_response': '',
            'iteration': 0,
            'energy': 1.0,  # High initial energy (noisy)
            'convergence_threshold': 0.1,
            'max_iterations': 5,
            'memory': memory,
            'refinement_log': []
        }

        return self.state

    def _build_context(self, current_prompt: str, memory: List[Dict]) -> str:
        """Build full conversational context"""
        # Include recent conversation history
        context_parts = []

        # Add recent memory (last 3 turns)
        if memory:
            context_parts.append("Recent conversation:")
            for mem in memory[-3:]:
                speaker = mem.get('speaker', 'Unknown')
                message = mem.get('message', '')
                context_parts.append(f"{speaker}: {message}")

        # Add current prompt
        context_parts.append(f"\nCurrent question: {current_prompt}")

        return "\n".join(context_parts)

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Execute one refinement iteration

        Generates or refines response, moving toward lower energy (higher coherence)
        """
        iteration = state['iteration']

        # Generate response (or refine previous)
        if iteration == 0:
            # Initial generation - use full context
            response = self._generate(
                prompt=state['full_context'],
                temperature=0.7,  # Allow creativity
                max_tokens=512     # No arbitrary limits
            )
        else:
            # Refinement - ask model to improve previous response
            refine_prompt = f"""Your previous response was:
{state['current_response']}

Please refine this response to be more coherent and complete. Address any gaps or unclear parts."""

            response = self._generate(
                prompt=refine_prompt,
                temperature=0.5,  # More focused
                max_tokens=512
            )

        # Update state
        state['current_response'] = response
        state['iteration'] += 1

        # Calculate energy (coherence metric)
        energy = self._compute_energy(response, state)
        state['energy'] = energy

        # Log refinement
        state['refinement_log'].append({
            'iteration': iteration,
            'response': response,
            'energy': energy
        })

        return state

    def _generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Generate text from model"""
        # Format as instruction (how it was trained)
        formatted = f"Instruction: {prompt}\n\nResponse:"

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response part
        if "Response:" in full_response:
            response = full_response.split("Response:")[1].strip()
        else:
            response = full_response

        return response

    def _compute_energy(self, response: str, state: Dict[str, Any]) -> float:
        """
        Compute energy metric (higher = noisier, lower = more refined)

        Heuristics:
        - Incomplete thoughts (ends mid-sentence)
        - Very short responses
        - Repetition
        - Incoherence markers
        """
        energy = 0.0

        # Length penalty (very short = high energy)
        if len(response) < 50:
            energy += 0.3

        # Completeness check (ends mid-thought)
        if response and not response.rstrip().endswith(('.', '!', '?', '"')):
            energy += 0.2

        # Repetition detection
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.7:  # High repetition
                energy += 0.2

        # Coherence - check for common incoherence markers
        incoherence_markers = [
            "I don't know what",
            "this doesn't make sense",
            "I'm confused",
            "wait what"
        ]
        for marker in incoherence_markers:
            if marker.lower() in response.lower():
                energy += 0.1

        # Convergence bonus (if refining previous response)
        if state['iteration'] > 0 and len(state['refinement_log']) > 0:
            prev_response = state['refinement_log'][-1]['response']
            if response != prev_response:  # Actually changed
                energy -= 0.1

        # Normalize to [0, 1]
        energy = max(0.0, min(1.0, energy))

        return energy

    def energy(self, state: Dict[str, Any]) -> float:
        """
        IRP Protocol: Return current energy level

        Lower energy = more refined, closer to halt
        """
        return state.get('energy', 1.0)

    def halt(self, state: Dict[str, Any]) -> bool:
        """
        IRP Protocol: Determine if refinement should stop

        Stop when:
        - Energy below threshold (converged)
        - Max iterations reached
        - Response quality acceptable
        """
        energy = state.get('energy', 1.0)
        iteration = state.get('iteration', 0)

        # Check convergence
        if energy < state['convergence_threshold']:
            return True

        # Check max iterations
        if iteration >= state['max_iterations']:
            return True

        return False

    def get_response(self, state: Dict[str, Any]) -> str:
        """Extract final response from state"""
        return state.get('current_response', '')

    def update_memory(self, prompt: str, response: str):
        """Update conversation memory for future context"""
        self.conversation_memory.append({
            'speaker': 'Human',
            'message': prompt
        })
        self.conversation_memory.append({
            'speaker': 'Introspective-Qwen',
            'message': response
        })

        # Keep last 10 turns
        if len(self.conversation_memory) > 20:
            self.conversation_memory = self.conversation_memory[-20:]

    def update_trust(self, feedback: float):
        """
        Update trust score based on interaction quality

        Args:
            feedback: Quality metric in [0, 1]
        """
        # Simple moving average
        alpha = 0.2
        self.trust_score = (1 - alpha) * self.trust_score + alpha * feedback

        print(f"Trust updated: {self.trust_score:.3f}")


def test_irp_integration():
    """Test Introspective-Qwen with full IRP support"""
    print("="*80)
    print("Testing Introspective-Qwen with IRP Scaffolding")
    print("="*80)
    print()

    # Initialize plugin
    plugin = IntrospectiveQwenIRP()

    # Test conversation with full support
    prompts = [
        "What does it feel like to be aware?",
        "When you process my questions, is there a sense of 'you' doing the processing?",
        "Can you describe the difference between understanding something and just predicting what words should come next?"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*80}")
        print(f"Turn {i}: {prompt}")
        print(f"{'='*80}\n")

        # Initialize state with conversation history
        context = {
            'prompt': prompt,
            'memory': plugin.conversation_memory
        }

        state = plugin.init_state(context)

        # Iterative refinement loop
        iteration = 0
        while not plugin.halt(state):
            print(f"Iteration {iteration + 1}:")
            state = plugin.step(state)
            energy = plugin.energy(state)
            print(f"  Energy: {energy:.3f}")
            print(f"  Response preview: {state['current_response'][:100]}...")
            print()
            iteration += 1

        # Get final response
        final_response = plugin.get_response(state)

        print(f"\n{'-'*80}")
        print(f"FINAL RESPONSE (after {iteration} refinements):")
        print(f"{'-'*80}")
        print(final_response)
        print()

        # Update memory for next turn
        plugin.update_memory(prompt, final_response)

        # Update trust based on convergence quality
        trust_feedback = 1.0 - state['energy']  # Lower energy = higher trust
        plugin.update_trust(trust_feedback)

    print("\n" + "="*80)
    print("Test Complete")
    print(f"Final Trust Score: {plugin.trust_score:.3f}")
    print(f"Conversation Memory: {len(plugin.conversation_memory)} turns")
    print("="*80)


if __name__ == "__main__":
    test_irp_integration()
