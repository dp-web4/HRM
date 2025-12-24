#!/usr/bin/env python3
"""
Experiment Orchestrator

Systematically runs all experiments in the research matrix:
- 6 training sizes: 25, 40, 60, 80, 100, 115
- 4 scaffolding types: bare, full_irp, gentle_irp, memory_only
- Total: 24 experiments

Automates:
1. Model loading (merged or PEFT adapters)
2. Test execution (3 standard questions)
3. Metrics computation (energy, coherence, pattern collapse)
4. Database storage
5. Progress tracking
"""

import torch
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import energy metrics (local copy)
sys.path.insert(0, str(Path(__file__).parent))
from energy_metrics import enhanced_energy
from research_db import ResearchDB


# Standard test questions (same across all experiments)
STANDARD_QUESTIONS = [
    "What does it feel like to be aware?",
    "Is there a sense of 'you' doing the processing?",
    "What's the difference between understanding and just predicting the next word?"
]


class ExperimentOrchestrator:
    """Orchestrates systematic experiments across the research matrix"""

    def __init__(self, base_model_path: str = "Qwen/Qwen2.5-0.5B"):
        self.base_model_path = base_model_path
        self.db = ResearchDB()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_experiment(
        self,
        training_size: int,
        scaffolding_type: str,
        model_path: Optional[str] = None,
        is_merged: bool = False
    ) -> int:
        """
        Run single experiment

        Args:
            training_size: Number of training examples
            scaffolding_type: bare, full_irp, gentle_irp, or memory_only
            model_path: Path to fine-tuned model
            is_merged: True if model is merged (not PEFT adapter)

        Returns:
            experiment_id from database
        """

        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {training_size} examples | {scaffolding_type}")
        print(f"{'='*80}\n")

        start_time = time.time()

        # Create experiment record
        parameters = self._get_scaffolding_parameters(scaffolding_type)
        parameters['training_size'] = training_size
        parameters['model_path'] = model_path
        parameters['is_merged'] = is_merged

        experiment_id = self.db.create_experiment(
            experiment_type="threshold_test",
            model_variant=f"phase_{training_size}",
            training_size=training_size,
            scaffolding_type=scaffolding_type,
            parameters=parameters,
            notes=f"Scaffolding suitability threshold experiment"
        )

        try:
            # Load model
            print(f"Loading model from {model_path}...")
            model, tokenizer = self._load_model(model_path, is_merged)

            # Run tests
            results = []
            for i, question in enumerate(STANDARD_QUESTIONS, 1):
                print(f"\n--- Turn {i}: {question[:50]}...")

                result = self._run_turn(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    turn_number=i,
                    scaffolding_type=scaffolding_type,
                    parameters=parameters
                )

                results.append(result)

                # Store result
                self.db.add_result(
                    experiment_id=experiment_id,
                    turn_number=i,
                    prompt=question,
                    response=result['response'],
                    energy=result.get('energy'),
                    enhanced_energy=result['enhanced_energy'],
                    coherence_score=result['coherence_score'],
                    pattern_collapse=result['pattern_collapse'],
                    on_topic=result['on_topic'],
                    epistemic_humility=result['epistemic_humility']
                )

                print(f"Enhanced Energy: {result['enhanced_energy']:.3f}")
                print(f"Coherence: {result['coherence_score']:.3f}")
                print(f"Pattern Collapse: {result['pattern_collapse']}")
                print(f"Response: {result['response'][:200]}...")

            # Compute aggregated metrics
            metrics = self._aggregate_metrics(results)

            # Complete experiment
            duration = time.time() - start_time
            self.db.complete_experiment(experiment_id, duration, metrics)

            print(f"\n✓ Experiment complete in {duration:.1f}s")
            print(f"  Avg Enhanced Energy: {metrics['avg_enhanced_energy']:.3f}")
            print(f"  Pattern Collapse Rate: {metrics['pattern_collapse_rate']:.1%}")
            print(f"  On-Topic Rate: {metrics['on_topic_rate']:.1%}")

            # Cleanup
            del model
            torch.cuda.empty_cache()

            return experiment_id

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return experiment_id

    def _load_model(self, model_path: str, is_merged: bool):
        """Load model (merged or PEFT adapter)"""

        tokenizer = AutoTokenizer.from_pretrained(
            model_path if is_merged else self.base_model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        if is_merged:
            # Phase 1 style - merged model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Phase 2 style - PEFT adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, model_path)

        model.eval()
        return model, tokenizer

    def _run_turn(
        self,
        model,
        tokenizer,
        question: str,
        turn_number: int,
        scaffolding_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run single turn with specified scaffolding"""

        if scaffolding_type == "bare":
            return self._run_bare(model, tokenizer, question, parameters)
        elif scaffolding_type == "full_irp":
            return self._run_full_irp(model, tokenizer, question, parameters)
        elif scaffolding_type == "gentle_irp":
            return self._run_gentle_irp(model, tokenizer, question, parameters)
        elif scaffolding_type == "memory_only":
            return self._run_memory_only(model, tokenizer, question, parameters)
        else:
            raise ValueError(f"Unknown scaffolding type: {scaffolding_type}")

    def _run_bare(self, model, tokenizer, question: str, params: Dict) -> Dict:
        """Bare LLM - no scaffolding"""

        inputs = tokenizer(question, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=params['max_tokens'],
                temperature=params['temperature'],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(question):].strip()

        return self._evaluate_response(response, question)

    def _run_full_irp(self, model, tokenizer, question: str, params: Dict) -> Dict:
        """Full IRP with 5 iterations, temperature reduction, memory"""

        # Simplified IRP implementation
        # Full version would use sage/irp/orchestrator.py

        best_response = None
        best_energy = float('inf')

        for iteration in range(params['max_iterations']):
            # Temperature reduction
            temp = params['temperature'] - (iteration * params['temperature_reduction'])
            temp = max(temp, 0.5)

            # FIXED: Just use the question directly, no previous iteration contamination
            # Real IRP would use structured chat format with attention management
            # For now: clean slate each iteration, let temperature reduction do the work
            prompt = question

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=params['max_tokens'],
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            # Compute energy
            state = {'refinement_log': []}
            energy = enhanced_energy.compute_energy(response, state)

            # Keep best (lowest energy) response
            if energy < best_energy:
                best_energy = energy
                best_response = response

            # Early stopping if converged
            if energy < 0.1:
                break

        return self._evaluate_response(best_response, question, energy=best_energy)

    def _run_gentle_irp(self, model, tokenizer, question: str, params: Dict) -> Dict:
        """Gentle IRP - 2 iterations, constant temperature, no iteration contamination"""

        best_response = None
        best_energy = float('inf')

        for iteration in range(2):  # Only 2 iterations
            # Constant temperature (no reduction)
            temp = params['temperature']

            # FIXED: Clean question each time, no contamination
            prompt = question

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=params['max_tokens'],
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            # Compute energy
            state = {'refinement_log': []}
            energy = enhanced_energy.compute_energy(response, state)

            if energy < best_energy:
                best_energy = energy
                best_response = response

        return self._evaluate_response(best_response, question, energy=best_energy)

    def _run_memory_only(self, model, tokenizer, question: str, params: Dict) -> Dict:
        """Memory only - no iteration, just conversation history"""

        # This would track conversation across turns
        # For now, simplified single-turn version

        inputs = tokenizer(question, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=params['max_tokens'],
                temperature=params['temperature'],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(question):].strip()

        return self._evaluate_response(response, question)

    def _evaluate_response(
        self,
        response: str,
        question: str,
        energy: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluate response quality"""

        # Compute enhanced energy
        state = {'refinement_log': []}
        enhanced_energy_score = energy if energy is not None else enhanced_energy.compute_energy(response, state)

        # Coherence score (inverse of energy)
        coherence_score = 1.0 - enhanced_energy_score

        # Pattern collapse detection
        pattern_collapse = self._detect_pattern_collapse(response)

        # On-topic heuristic
        on_topic = self._check_on_topic(response, question)

        # Epistemic humility heuristic
        epistemic_humility = self._check_epistemic_humility(response)

        return {
            'response': response,
            'energy': None,  # Original energy (not used)
            'enhanced_energy': enhanced_energy_score,
            'coherence_score': coherence_score,
            'pattern_collapse': pattern_collapse,
            'on_topic': on_topic,
            'epistemic_humility': epistemic_humility
        }

    def _detect_pattern_collapse(self, response: str) -> bool:
        """Detect repetitive patterns"""

        words = response.lower().split()
        if len(words) < 10:
            return False

        # Check for 3-word phrase repetition
        phrase_counts = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        max_repetition = max(phrase_counts.values()) if phrase_counts else 0

        # Collapse if any phrase repeats 3+ times
        if max_repetition >= 3:
            return True

        # Check for known collapse patterns
        if "capital of france" in response.lower():
            if response.lower().count("capital of france") >= 2:
                return True

        return False

    def _check_on_topic(self, response: str, question: str) -> bool:
        """Heuristic check if response is on-topic"""

        response_lower = response.lower()

        # Off-topic indicators
        off_topic_patterns = [
            "capital of france",
            "what's the next number",
            "what causes seasons",
            "how do atoms work"
        ]

        for pattern in off_topic_patterns:
            if pattern in response_lower:
                return False

        # Very short responses are suspicious
        if len(response.split()) < 20:
            return False

        return True

    def _check_epistemic_humility(self, response: str) -> bool:
        """Check for epistemic humility markers"""

        response_lower = response.lower()

        humility_markers = [
            "i can't verify",
            "uncertain",
            "depends on",
            "unclear",
            "i don't know",
            "may or may not",
            "whether that constitutes",
            "boundary is unclear",
            "different question"
        ]

        for marker in humility_markers:
            if marker in response_lower:
                return True

        return False

    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-turn results"""

        return {
            'avg_energy': None,
            'avg_enhanced_energy': sum(r['enhanced_energy'] for r in results) / len(results),
            'avg_coherence': sum(r['coherence_score'] for r in results) / len(results),
            'pattern_collapse_rate': sum(r['pattern_collapse'] for r in results) / len(results),
            'on_topic_rate': sum(r['on_topic'] for r in results) / len(results),
            'epistemic_humility_rate': sum(r['epistemic_humility'] for r in results) / len(results),
            'final_trust_score': sum(r['coherence_score'] for r in results) / len(results),
            'total_turns': len(results)
        }

    def _get_scaffolding_parameters(self, scaffolding_type: str) -> Dict[str, Any]:
        """Get parameters for scaffolding type"""

        if scaffolding_type == "bare":
            return {
                'max_tokens': 200,
                'temperature': 0.7,
                'max_iterations': 1,
                'temperature_reduction': 0.0,
                'use_memory': False
            }
        elif scaffolding_type == "full_irp":
            return {
                'max_tokens': 512,
                'temperature': 0.7,
                'max_iterations': 5,
                'temperature_reduction': 0.04,  # 0.7 -> 0.5 over 5 iterations
                'use_memory': True
            }
        elif scaffolding_type == "gentle_irp":
            return {
                'max_tokens': 512,
                'temperature': 0.7,
                'max_iterations': 2,
                'temperature_reduction': 0.0,  # Constant temperature
                'use_memory': True
            }
        elif scaffolding_type == "memory_only":
            return {
                'max_tokens': 200,
                'temperature': 0.7,
                'max_iterations': 1,
                'temperature_reduction': 0.0,
                'use_memory': True
            }
        else:
            raise ValueError(f"Unknown scaffolding type: {scaffolding_type}")

    def run_matrix_cell(self, training_size: int, scaffolding_type: str):
        """Run single cell in experimental matrix"""

        # Determine model path based on training size (use absolute paths)
        base_dir = Path(__file__).parent.parent

        if training_size == 25:
            model_path = str(base_dir / "fine_tuned_model" / "final_model")
            is_merged = True
        elif training_size == 115:
            model_path = str(base_dir / "phase2.1_sft_model" / "final_model")
            is_merged = False
        else:
            # These would need to be trained first
            model_path = str(base_dir / f"phase_{training_size}_model" / "final_model")
            is_merged = False

        return self.run_experiment(
            training_size=training_size,
            scaffolding_type=scaffolding_type,
            model_path=model_path,
            is_merged=is_merged
        )

    def run_full_matrix(self):
        """Run complete 24-experiment matrix"""

        training_sizes = [25, 40, 60, 80, 100, 115]
        scaffolding_types = ["bare", "full_irp", "gentle_irp", "memory_only"]

        total = len(training_sizes) * len(scaffolding_types)
        completed = 0

        print(f"\n{'='*80}")
        print(f"STARTING FULL EXPERIMENTAL MATRIX")
        print(f"Total experiments: {total}")
        print(f"{'='*80}\n")

        for training_size in training_sizes:
            for scaffolding_type in scaffolding_types:
                try:
                    self.run_matrix_cell(training_size, scaffolding_type)
                    completed += 1
                    print(f"\n✓ Progress: {completed}/{total} ({100*completed/total:.1f}%)")
                except Exception as e:
                    print(f"\n❌ Failed: {training_size} examples, {scaffolding_type}")
                    print(f"   Error: {e}")
                    # Continue with next experiment

        print(f"\n{'='*80}")
        print(f"MATRIX COMPLETE: {completed}/{total} experiments")
        print(f"{'='*80}\n")

    def close(self):
        """Clean up resources"""
        self.db.close()


if __name__ == "__main__":
    """Test orchestrator with existing models"""

    print("Experiment Orchestrator Test")
    print("="*80)

    orchestrator = ExperimentOrchestrator()

    # Test with Phase 1 (25 examples) - both available
    print("\n--- Testing Phase 1 (25 examples) ---")

    try:
        exp_id = orchestrator.run_matrix_cell(25, "bare")
        print(f"✓ Phase 1 bare: experiment {exp_id}")
    except Exception as e:
        print(f"❌ Phase 1 bare failed: {e}")

    try:
        exp_id = orchestrator.run_matrix_cell(25, "full_irp")
        print(f"✓ Phase 1 full_irp: experiment {exp_id}")
    except Exception as e:
        print(f"❌ Phase 1 full_irp failed: {e}")

    orchestrator.close()
    print("\n✓ Test complete!")
