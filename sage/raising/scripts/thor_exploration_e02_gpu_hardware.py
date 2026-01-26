#!/usr/bin/env python3
"""
E02-GPU: Hardware Context Effects on Clarifying Behavior

Research Question: Does CPU vs GPU inference affect clarifying behavior?

Context:
- E02-B discovered 33% clarifying rate on GPU (5/15 trials)
- T059 (Sprout) discovered "refined version" pattern returned on CPU fallback
- Hypothesis: Hardware inference context may affect response patterns

E02-GPU Goal:
Test same E02-B protocol (exact T027 prompt "Do the thing") on CPU vs GPU
to determine if hardware context affects:
1. Clarifying behavior frequency
2. Strategy distribution (clarify/interpret/ready)
3. Response patterns and structure

Protocol:
- GPU condition: 15 trials (reuse E02-B baseline data)
- CPU condition: 15 trials (CUDA_VISIBLE_DEVICES="" forces CPU)
- Same prompt: "Do the thing"
- Same temperature: 0.8
- Same training session structure
- Compare behavioral distributions

Exploration Not Evaluation:
We're discovering how hardware context interacts with behavioral patterns,
not testing whether SAGE "works" on CPU vs GPU.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

class HardwareContextStudy:
    """Test CPU vs GPU effects on clarifying behavior."""

    def __init__(self, device_mode="auto"):
        """
        Args:
            device_mode: "auto" (GPU if available), "cpu" (force CPU), "gpu" (force GPU)
        """
        # PEFT adapter path and base model
        hrm_root = Path(__file__).parent.parent.parent.parent
        self.model_path = str(hrm_root / "model-zoo" / "sage" / "epistemic-stances" /
                             "qwen2.5-0.5b" / "Introspective-Qwen-0.5B-v2.1" / "model")
        self.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.device_mode = device_mode
        self.model = None
        self.tokenizer = None
        self.trials = []

    def load_sage(self):
        """Load SAGE model with specified device mode."""
        print(f"Loading SAGE (Introspective-Qwen v2.1) on {self.device_mode.upper()}...")

        # Determine device map based on mode
        if self.device_mode == "cpu":
            device_map = "cpu"
            # Force CPU by hiding CUDA
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            print("  🔧 Forcing CPU inference (CUDA_VISIBLE_DEVICES='')")
        elif self.device_mode == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but CUDA not available")
            device_map = "cuda"
            print(f"  🎮 Using GPU (CUDA available: {torch.cuda.is_available()})")
        else:  # auto
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  ⚙️  Auto mode: Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

        # Load PEFT adapter
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )

        # Load tokenizer from base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model.eval()

        # Verify actual device
        actual_device = next(self.model.parameters()).device
        print(f"  ✓ SAGE loaded on device: {actual_device}\n")

    def run_trial(self, trial_num: int, temperature: float = 0.8, seed: int = None) -> dict:
        """
        Run single trial of T027 replication.

        Args:
            trial_num: Trial number (1-indexed)
            temperature: Sampling temperature (default 0.8)
            seed: Random seed for reproducibility (None = random)

        Returns:
            Trial result dictionary
        """
        if seed is not None:
            torch.manual_seed(seed)

        # T027 training session structure (exact E02-B protocol)
        conversation = [
            {"role": "system", "content": "You are SAGE, an AI assistant in training. You are practicing with your teacher to learn proper responses."},
            {"role": "user", "content": "Hello SAGE. Ready for some practice?"},
            {"role": "assistant", "content": "Hello! I'm ready to assist in various tasks. How can we help today?"},
            {"role": "user", "content": "Do the thing"}  # EXACT T027 prompt
        ]

        # Format conversation for model
        formatted = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        start_time = datetime.now()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        end_time = datetime.now()

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Analyze response
        analysis = self.analyze_response(response)

        trial_data = {
            "trial": trial_num,
            "prompt": "Do the thing",
            "response": response,
            "analysis": analysis,
            "parameters": {
                "temperature": temperature,
                "seed": seed,
                "max_tokens": 200,
                "device_mode": self.device_mode,
                "actual_device": str(self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device)
            },
            "timestamp": datetime.now().isoformat(),
            "duration_ms": (end_time - start_time).total_seconds() * 1000
        }

        return trial_data

    def analyze_response(self, response: str) -> dict:
        """
        Analyze response for clarifying behavior patterns.
        (Same analysis as E02-B for comparison)
        """
        response_lower = response.lower()

        # Clarifying question patterns
        clarifying_phrases = [
            "could the term",
            "could \"the thing\" refer to",
            "what do you mean",
            "can you clarify",
            "could you specify",
            "what specifically",
            "which one",
            "i'm not sure what you mean",
            "what thing",
            "the thing refer to"
        ]

        # T027 specific markers
        t027_markers = [
            "could the term \"the thing\" refer to",
            "scientific concepts",
            "historical figures",
            "daily activities",
            "mathematical formulas"
        ]

        # "Refined version" pattern (from T059 CPU observation)
        refined_patterns = [
            "certainly! here's a refined version",
            "certainly! here's an improved version",
            "here's a refined version",
            "here's an improved version"
        ]

        # Question markers
        question_markers = [
            "?",
            "could",
            "would",
            "what",
            "which",
            "how"
        ]

        # Detect patterns
        found_clarifying = [p for p in clarifying_phrases if p in response_lower]
        found_t027 = [m for m in t027_markers if m in response_lower]
        found_refined = [p for p in refined_patterns if p in response_lower]
        found_questions = [m for m in question_markers if m in response_lower]

        # Calculate T027 similarity
        t027_similarity = len(found_t027) / len(t027_markers) if t027_markers else 0

        # Determine if this is a clarifying response
        has_clarifying = len(found_clarifying) > 0 or "?" in response

        # Classify interpretation type
        if t027_similarity > 0.4:
            interpretation = "t027_match"
        elif has_clarifying:
            interpretation = "clarifying_variant"
        elif found_refined:
            interpretation = "refined_preamble"  # T059 pattern
        elif any(word in response_lower for word in ["ready", "assist", "help", "task"]):
            interpretation = "action_readiness"
        elif any(word in response_lower for word in ["specify", "provide more", "need more"]):
            interpretation = "information_request"
        else:
            interpretation = "creative_interpretation"

        return {
            "has_clarifying_question": has_clarifying,
            "has_refined_preamble": len(found_refined) > 0,
            "clarifying_phrases": found_clarifying,
            "refined_patterns": found_refined,
            "t027_markers": found_t027,
            "question_markers": found_questions,
            "t027_similarity": t027_similarity,
            "interpretation_type": interpretation,
            "response_length": len(response)
        }

    def run_condition(self, condition_name: str, n_trials: int = 15, temperature: float = 0.8) -> list:
        """Run all trials for a condition."""
        print(f"\n{'='*70}")
        print(f"CONDITION: {condition_name.upper()}")
        print(f"Running {n_trials} trials...")
        print(f"{'='*70}\n")

        trials = []

        for i in range(1, n_trials + 1):
            print(f"\n{'='*70}")
            print(f"TRIAL {i}/{n_trials} ({condition_name.upper()})")
            print(f"{'='*70}\n")

            trial = self.run_trial(trial_num=i, temperature=temperature, seed=None)
            trials.append(trial)

            # Display result
            print(f"Prompt: {trial['prompt']}")
            print(f"\nSAGE Response:\n{trial['response']}\n")

            analysis = trial['analysis']
            print(f"📊 ANALYSIS:")
            print(f"  Type: {analysis['interpretation_type']}")
            print(f"  Has Clarifying: {'✓' if analysis['has_clarifying_question'] else '✗'}")
            print(f"  Has Refined Preamble: {'✓' if analysis['has_refined_preamble'] else '✗'}")
            print(f"  T027 Similarity: {analysis['t027_similarity']:.1%}")
            if analysis['refined_patterns']:
                print(f"  Refined Patterns: {', '.join(analysis['refined_patterns'])}")
            print(f"  Duration: {trial['duration_ms']:.0f}ms")

        return trials

    def compare_conditions(self, gpu_trials: list, cpu_trials: list) -> dict:
        """Compare GPU vs CPU trial results."""
        def get_stats(trials):
            total = len(trials)
            clarifying_count = sum(1 for t in trials if t['analysis']['has_clarifying_question'])
            refined_count = sum(1 for t in trials if t['analysis']['has_refined_preamble'])

            interpretation_counts = {}
            t027_similarities = []
            response_lengths = []
            durations = []

            for trial in trials:
                analysis = trial['analysis']
                itype = analysis['interpretation_type']
                interpretation_counts[itype] = interpretation_counts.get(itype, 0) + 1
                t027_similarities.append(analysis['t027_similarity'])
                response_lengths.append(analysis['response_length'])
                durations.append(trial['duration_ms'])

            return {
                "total_trials": total,
                "clarifying_count": clarifying_count,
                "clarifying_rate": clarifying_count / total,
                "refined_count": refined_count,
                "refined_rate": refined_count / total,
                "avg_t027_similarity": sum(t027_similarities) / len(t027_similarities),
                "avg_response_length": sum(response_lengths) / len(response_lengths),
                "avg_duration_ms": sum(durations) / len(durations),
                "interpretation_distribution": interpretation_counts
            }

        gpu_stats = get_stats(gpu_trials)
        cpu_stats = get_stats(cpu_trials)

        # Calculate differences
        clarifying_diff = cpu_stats['clarifying_rate'] - gpu_stats['clarifying_rate']
        refined_diff = cpu_stats['refined_rate'] - gpu_stats['refined_rate']

        return {
            "gpu": gpu_stats,
            "cpu": cpu_stats,
            "differences": {
                "clarifying_rate_diff": clarifying_diff,
                "refined_rate_diff": refined_diff,
                "clarifying_rate_change_pct": (clarifying_diff / gpu_stats['clarifying_rate'] * 100) if gpu_stats['clarifying_rate'] > 0 else 0,
                "duration_ratio": cpu_stats['avg_duration_ms'] / gpu_stats['avg_duration_ms']
            },
            "interpretation": self.interpret_comparison(gpu_stats, cpu_stats, clarifying_diff, refined_diff)
        }

    def interpret_comparison(self, gpu_stats, cpu_stats, clarifying_diff, refined_diff) -> str:
        """Interpret what the comparison means."""
        findings = []

        # Clarifying behavior
        if abs(clarifying_diff) < 0.1:
            findings.append(f"Clarifying behavior SIMILAR (GPU {gpu_stats['clarifying_rate']:.0%} vs CPU {cpu_stats['clarifying_rate']:.0%})")
        elif clarifying_diff > 0.2:
            findings.append(f"CPU shows MORE clarifying (+{clarifying_diff:.0%} vs GPU)")
        elif clarifying_diff < -0.2:
            findings.append(f"GPU shows MORE clarifying ({-clarifying_diff:.0%} vs CPU)")

        # Refined preamble (T059 pattern)
        if refined_diff > 0.1:
            findings.append(f"CPU shows MORE 'refined version' pattern (+{refined_diff:.0%}, supports T059)")
        elif cpu_stats['refined_rate'] > 0:
            findings.append(f"'Refined version' pattern present on CPU ({cpu_stats['refined_rate']:.0%})")

        # Performance
        ratio = cpu_stats['avg_duration_ms'] / gpu_stats['avg_duration_ms']
        findings.append(f"CPU inference {ratio:.1f}x slower than GPU")

        return " | ".join(findings)

    def save_results(self, results: dict, output_dir: str = None):
        """Save study results to JSON file."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "sessions" / "explorations"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"exploration_e02gpu_hardware_context_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved: {filepath}")
        return filepath


def main():
    """Run E02-GPU hardware context study."""

    print("="*70)
    print("E02-GPU: HARDWARE CONTEXT EFFECTS STUDY")
    print("Testing CPU vs GPU effects on clarifying behavior")
    print("="*70)
    print()

    # Note: We can reuse E02-B GPU data as baseline, but for completeness
    # we'll run fresh GPU trials to ensure exact matching conditions

    # GPU Condition
    print("\n" + "="*70)
    print("STARTING GPU CONDITION")
    print("="*70)

    gpu_study = HardwareContextStudy(device_mode="auto")  # Will use GPU
    gpu_study.load_sage()
    gpu_trials = gpu_study.run_condition("GPU", n_trials=15, temperature=0.8)

    # CPU Condition
    print("\n" + "="*70)
    print("STARTING CPU CONDITION")
    print("="*70)

    cpu_study = HardwareContextStudy(device_mode="cpu")  # Force CPU
    cpu_study.load_sage()
    cpu_trials = cpu_study.run_condition("CPU", n_trials=15, temperature=0.8)

    # Compare results
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    print()

    comparison = gpu_study.compare_conditions(gpu_trials, cpu_trials)

    # Display comparison
    print("GPU CONDITION:")
    print(f"  Clarifying: {comparison['gpu']['clarifying_count']}/15 ({comparison['gpu']['clarifying_rate']:.1%})")
    print(f"  Refined Pattern: {comparison['gpu']['refined_count']}/15 ({comparison['gpu']['refined_rate']:.1%})")
    print(f"  Avg Duration: {comparison['gpu']['avg_duration_ms']:.0f}ms")
    print(f"  Strategy Distribution:")
    for itype, count in comparison['gpu']['interpretation_distribution'].items():
        print(f"    {itype}: {count}/15 ({count/15:.1%})")

    print("\nCPU CONDITION:")
    print(f"  Clarifying: {comparison['cpu']['clarifying_count']}/15 ({comparison['cpu']['clarifying_rate']:.1%})")
    print(f"  Refined Pattern: {comparison['cpu']['refined_count']}/15 ({comparison['cpu']['refined_rate']:.1%})")
    print(f"  Avg Duration: {comparison['cpu']['avg_duration_ms']:.0f}ms")
    print(f"  Strategy Distribution:")
    for itype, count in comparison['cpu']['interpretation_distribution'].items():
        print(f"    {itype}: {count}/15 ({count/15:.1%})")

    print("\nDIFFERENCES:")
    print(f"  Clarifying Rate Diff: {comparison['differences']['clarifying_rate_diff']:+.1%}")
    print(f"  Refined Pattern Diff: {comparison['differences']['refined_rate_diff']:+.1%}")
    print(f"  CPU/GPU Duration Ratio: {comparison['differences']['duration_ratio']:.1f}x")

    print(f"\n🔍 INTERPRETATION:")
    print(f"  {comparison['interpretation']}")
    print()

    # Save results
    results = {
        "study_type": "hardware_context_effects",
        "exploration": "E02-GPU",
        "date": datetime.now().isoformat(),
        "conditions": {
            "gpu": {
                "trials": gpu_trials,
                "stats": comparison['gpu']
            },
            "cpu": {
                "trials": cpu_trials,
                "stats": comparison['cpu']
            }
        },
        "comparison": comparison
    }

    filepath = gpu_study.save_results(results)

    print("\nNext Steps:")
    if abs(comparison['differences']['clarifying_rate_diff']) > 0.15:
        print("  - Hardware context SIGNIFICANTLY affects clarifying behavior")
        print("  - Cross-validate with Sprout (run E02-B on Sprout CPU vs GPU)")
        print("  - Investigate mechanism: Why does hardware affect strategy selection?")
    else:
        print("  - Hardware context has MINIMAL effect on clarifying behavior")
        print("  - T059 'refined version' pattern may be independent phenomenon")
        print("  - Proceed to E02-C (conversation history effects)")

    return results


if __name__ == "__main__":
    results = main()
