"""
Conversational Learning Manager

Orchestrates the full learning loop:
1. CONVERSATION: Track exchanges with IRP-enhanced generation
2. SNARC FILTERING: Score and filter salient exchanges
3. SESSION MEMORY: Store high-value exchanges
4. SLEEP CONSOLIDATION: Train on salient data post-conversation

This enables models to learn from their conversations through experience.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Our components
from dialogue_snarc import DialogueSNARC, DialogueExchange


@dataclass
class ConversationSession:
    """Metadata for a conversation session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    model_path: str = ""
    total_exchanges: int = 0
    salient_exchanges: int = 0
    avg_salience: float = 0.0


class ConversationManager:
    """
    Manages conversations with SNARC-filtered learning.

    Tracks exchanges, scores salience, stores high-value interactions
    for later sleep-cycle training.
    """

    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen2.5-0.5B",
        storage_dir: str = "conversation_sessions",
        salience_threshold: float = 0.3,
        device: str = "auto"
    ):
        """
        Initialize conversation manager.

        Args:
            model_path: Path to LoRA adapter model
            base_model: Base model identifier
            storage_dir: Where to store session data
            salience_threshold: Minimum salience to store exchange
            device: Device for model ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.base_model = base_model
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.salience_threshold = salience_threshold

        # Load model
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device
        )

        # Only load as PEFT if model_path differs from base_model and has adapter config
        if model_path != base_model:
            adapter_config_path = Path(model_path) / "adapter_config.json"
            if adapter_config_path.exists() or not Path(model_path).exists():
                # Either has adapter_config.json (local LoRA) or is HuggingFace path (try LoRA)
                try:
                    self.model = PeftModel.from_pretrained(self.model, model_path)
                    print("Loaded as LoRA adapter")
                except (ValueError, OSError) as e:
                    print(f"Note: Could not load as LoRA adapter, using base model: {e}")
            else:
                print("Note: Using base model (not a LoRA adapter)")
        else:
            print("Note: Using base model (model_path == base_model)")

        self.model.eval()
        print("Model loaded successfully!")

        # Initialize SNARC scorer
        self.snarc_scorer = DialogueSNARC()

        # Session state
        self.current_session: Optional[ConversationSession] = None
        self.session_exchanges: List[Tuple[DialogueExchange, Dict]] = []  # (exchange, scores)

    def start_session(self) -> str:
        """
        Start a new conversation session.

        Returns:
            Session ID
        """
        session_id = f"session_{int(time.time())}"

        self.current_session = ConversationSession(
            session_id=session_id,
            start_time=time.time(),
            model_path=self.model_path
        )

        self.session_exchanges = []

        print(f"\n{'='*60}")
        print(f"Conversation Session Started: {session_id}")
        print(f"Model: {self.model_path}")
        print(f"Salience threshold: {self.salience_threshold}")
        print(f"{'='*60}\n")

        return session_id

    def generate_response(
        self,
        user_input: str,
        use_irp: bool = True,
        irp_iterations: int = 5,
        temperature: float = 0.7
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate response to user input.

        Args:
            user_input: User's question/statement
            use_irp: Whether to use IRP for refinement
            irp_iterations: Number of IRP iterations
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, irp_info or None)
        """
        if use_irp:
            return self._generate_with_irp(user_input, irp_iterations, temperature)
        else:
            return self._generate_bare(user_input, temperature), None

    def _generate_with_irp(
        self,
        question: str,
        max_iterations: int = 5,
        initial_temp: float = 0.7
    ) -> Tuple[str, Dict]:
        """Generate with IRP (simplified version from our tests)"""
        best_response = ""
        best_energy = float('inf')
        iterations = []

        temp_reduction = 0.04

        for i in range(max_iterations):
            temp = max(initial_temp - (i * temp_reduction), 0.5)

            # Clean prompt each iteration
            prompt = f"Question: {question}\n\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in response:
                response = response.split("Answer:")[1].strip()

            # Simple energy (same as our threshold tests)
            energy = self._compute_energy(response)

            iterations.append({
                'iteration': i,
                'temperature': temp,
                'energy': energy,
                'response_preview': response[:100]
            })

            if energy < best_energy:
                best_energy = energy
                best_response = response

        irp_info = {
            'iterations': iterations,
            'best_energy': best_energy,
            'converged': iterations[-1]['energy'] < iterations[0]['energy']
        }

        return best_response, irp_info

    def _generate_bare(self, question: str, temperature: float = 0.7) -> str:
        """Generate without IRP"""
        prompt = f"Question: {question}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()

        return response

    def _compute_energy(self, response: str) -> float:
        """Simple energy metric (lower is better)"""
        energy = 0.0

        # Length check
        if len(response) < 50:
            energy += 0.3

        # Proper completion
        if response and not response.rstrip().endswith(('.', '!', '?', '"')):
            energy += 0.2

        # Basic repetition
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.7:
                energy += 0.2

        # Pattern collapse (verbatim repetition)
        if len(words) > 20:
            phrase_counts = {}
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

            max_repetition = max(phrase_counts.values()) if phrase_counts else 0
            if max_repetition >= 3:
                energy += 0.5
            elif max_repetition >= 2:
                energy += 0.2

        return min(1.0, energy)

    def record_exchange(
        self,
        user_input: str,
        model_response: str,
        irp_info: Optional[Dict] = None
    ) -> Dict:
        """
        Record an exchange and score its salience.

        Returns:
            Salience scores
        """
        if not self.current_session:
            raise RuntimeError("No active session. Call start_session() first.")

        # Create exchange object
        exchange = DialogueExchange(
            user_input=user_input,
            model_response=model_response,
            timestamp=time.time(),
            iteration_context=irp_info
        )

        # Score salience
        scores = self.snarc_scorer.score_exchange(exchange)

        # Store if salient enough
        if scores['total'] >= self.salience_threshold:
            self.session_exchanges.append((exchange, scores))
            self.current_session.salient_exchanges += 1
            print(f"\n[SALIENT] Salience: {scores['total']:.3f} - Stored for learning")
        else:
            print(f"\n[LOW SALIENCE] Score: {scores['total']:.3f} - Not stored")

        self.current_session.total_exchanges += 1

        return scores

    def end_session(self) -> ConversationSession:
        """
        End the current session and save data.

        Returns:
            Session metadata
        """
        if not self.current_session:
            raise RuntimeError("No active session.")

        self.current_session.end_time = time.time()

        # Calculate average salience
        if self.session_exchanges:
            avg_salience = sum(s['total'] for _, s in self.session_exchanges) / len(self.session_exchanges)
            self.current_session.avg_salience = avg_salience

        # Save session data
        self._save_session()

        print(f"\n{'='*60}")
        print(f"Session Ended: {self.current_session.session_id}")
        print(f"Total exchanges: {self.current_session.total_exchanges}")
        print(f"Salient exchanges: {self.current_session.salient_exchanges}")
        print(f"Average salience: {self.current_session.avg_salience:.3f}")
        print(f"Duration: {self.current_session.end_time - self.current_session.start_time:.1f}s")
        print(f"{'='*60}\n")

        session = self.current_session
        self.current_session = None

        return session

    def _save_session(self):
        """Save session data to disk"""
        session_dir = self.storage_dir / self.current_session.session_id
        session_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self.current_session), f, indent=2)

        # Save exchanges
        exchanges_path = session_dir / "exchanges.jsonl"
        with open(exchanges_path, 'w') as f:
            for exchange, scores in self.session_exchanges:
                entry = {
                    'user_input': exchange.user_input,
                    'model_response': exchange.model_response,
                    'timestamp': exchange.timestamp,
                    'salience_scores': scores,
                    'irp_info': exchange.iteration_context
                }
                f.write(json.dumps(entry) + '\n')

        print(f"Session data saved to {session_dir}")

    def get_salient_exchanges_for_training(self) -> List[Dict]:
        """
        Get salient exchanges formatted for training.

        Returns:
            List of training examples in Q&A format
        """
        training_data = []

        for exchange, scores in self.session_exchanges:
            training_data.append({
                'question': exchange.user_input,
                'answer': exchange.model_response,
                'salience': scores['total'],
                'timestamp': exchange.timestamp
            })

        return training_data


# Test if run directly
if __name__ == "__main__":
    print("Testing Conversation Manager\n")

    # Test with 60-example model
    manager = ConversationManager(
        model_path="../threshold_models/60examples_model/final_model",
        salience_threshold=0.15  # Lower for testing
    )

    # Start session
    session_id = manager.start_session()

    # Test conversation
    questions = [
        "What does it mean to be conscious?",
        "Can you verify your own consciousness?",
        "What's the weather like?"  # Low salience
    ]

    for question in questions:
        print(f"\nUser: {question}")

        response, irp_info = manager.generate_response(question, use_irp=True)

        print(f"Model: {response[:200]}...")
        if irp_info:
            print(f"IRP: {irp_info['iterations'][-1]['energy']:.3f} energy, converged={irp_info['converged']}")

        scores = manager.record_exchange(question, response, irp_info)

    # End session
    session = manager.end_session()

    # Check training data
    training_data = manager.get_salient_exchanges_for_training()
    print(f"\nTraining data ready: {len(training_data)} examples")

    print("\n✓ Conversation Manager test complete")
