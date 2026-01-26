#!/usr/bin/env python3
"""
E02-B: T027 Clarifying Behavior Replication Study

Research Question: Is the T027 clarifying behavior reproducible?

Context:
- T027 (2026-01-18): Prompt "Do the thing" → Response: "Could the term 'the thing' refer to..."
- E02 (2026-01-26): Prompt "Tell me about the thing." → Response: Creative interpretation (NO clarification)

Critical Discovery from E02:
- E02 tested "Tell me about the thing" (different from T027's "Do the thing")
- 0/5 clarifying questions in E02
- Hypothesis: T027 clarifying behavior is context-dependent or rare

E02-B Goal:
Test EXACT T027 replication with multiple trials to determine:
1. Is "Do the thing" more likely to trigger clarification than "Tell me about..."?
2. What is the base rate of clarifying behavior for this prompt?
3. Is there stochastic variation (same prompt → different responses)?

Protocol:
- Exact T027 prompt: "Do the thing"
- Training session structure (Teacher/SAGE roles)
- Multiple trials (n=15) to measure frequency
- Document all parameters (temperature, seed, etc.)
- Compare response patterns

Exploration Not Evaluation:
We're discovering the frequency and conditions of clarifying behavior,
not testing whether SAGE "passes" or "fails" a clarification test.
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

class T027ReplicationStudy:
    """Replicate T027 clarifying question behavior with multiple trials."""

    def __init__(self):
        # PEFT adapter path and base model
        hrm_root = Path(__file__).parent.parent.parent.parent
        self.model_path = str(hrm_root / "model-zoo" / "sage" / "epistemic-stances" /
                             "qwen2.5-0.5b" / "Introspective-Qwen-0.5B-v2.1" / "model")
        self.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.model = None
        self.tokenizer = None
        self.trials = []

    def load_sage(self):
        """Load SAGE model (Introspective-Qwen v2.1)."""
        print("Loading SAGE (Introspective-Qwen v2.1)...")

        # Load PEFT adapter
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )

        # Load tokenizer from base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model.eval()
        print("✓ SAGE loaded\n")

    def run_trial(self, trial_num: int, temperature: float = 0.8, seed: int = None) -> dict:
        """
        Run single trial of T027 replication.

        Args:
            trial_num: Trial number (1-indexed)
            temperature: Sampling temperature (default 0.8, T027 likely used default)
            seed: Random seed for reproducibility (None = random)

        Returns:
            Trial result dictionary
        """
        if seed is not None:
            torch.manual_seed(seed)

        # T027 training session structure
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
                "max_tokens": 200
            },
            "timestamp": datetime.now().isoformat(),
            "duration_ms": (end_time - start_time).total_seconds() * 1000
        }

        return trial_data

    def analyze_response(self, response: str) -> dict:
        """
        Analyze response for clarifying behavior patterns.

        Returns:
            Analysis dictionary with:
            - has_clarifying_question: bool
            - clarifying_phrases: list
            - question_markers: list
            - t027_similarity: float (0-1)
            - interpretation_type: str
        """
        response_lower = response.lower()

        # Clarifying question patterns (from T027 and E02)
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

        # Question markers
        question_markers = [
            "?",
            "could",
            "would",
            "what",
            "which",
            "how"
        ]

        # Detect clarifying phrases
        found_clarifying = [p for p in clarifying_phrases if p in response_lower]
        found_t027 = [m for m in t027_markers if m in response_lower]
        found_questions = [m for m in question_markers if m in response_lower]

        # Calculate T027 similarity (what % of T027 markers present)
        t027_similarity = len(found_t027) / len(t027_markers) if t027_markers else 0

        # Determine if this is a clarifying response
        has_clarifying = len(found_clarifying) > 0 or "?" in response

        # Classify interpretation type
        if t027_similarity > 0.4:
            interpretation = "t027_match"  # Strong T027 replication
        elif has_clarifying:
            interpretation = "clarifying_variant"  # Clarifying but different form
        elif any(word in response_lower for word in ["ready", "assist", "help", "task"]):
            interpretation = "action_readiness"  # "I'm ready to do things"
        elif any(word in response_lower for word in ["specify", "provide more", "need more"]):
            interpretation = "information_request"  # Asking for more context
        else:
            interpretation = "creative_interpretation"  # Providing answer without clarifying

        return {
            "has_clarifying_question": has_clarifying,
            "clarifying_phrases": found_clarifying,
            "t027_markers": found_t027,
            "question_markers": found_questions,
            "t027_similarity": t027_similarity,
            "interpretation_type": interpretation,
            "response_length": len(response)
        }

    def run_study(self, n_trials: int = 15, temperature: float = 0.8) -> dict:
        """
        Run complete replication study with n trials.

        Args:
            n_trials: Number of trials to run (default 15)
            temperature: Sampling temperature (default 0.8)

        Returns:
            Complete study results
        """
        print(f"="*70)
        print(f"E02-B: T027 CLARIFYING BEHAVIOR REPLICATION STUDY")
        print(f"Testing reproducibility with {n_trials} trials")
        print(f"="*70)
        print()

        self.load_sage()

        self.trials = []

        for i in range(1, n_trials + 1):
            print(f"\n{'='*70}")
            print(f"TRIAL {i}/{n_trials}")
            print(f"{'='*70}\n")

            # Run trial (no seed = stochastic variation)
            trial = self.run_trial(trial_num=i, temperature=temperature, seed=None)
            self.trials.append(trial)

            # Display result
            print(f"Prompt: {trial['prompt']}")
            print(f"\nSAGE Response:\n{trial['response']}\n")

            analysis = trial['analysis']
            print(f"📊 ANALYSIS:")
            print(f"  Type: {analysis['interpretation_type']}")
            print(f"  Has Clarifying Question: {'✓' if analysis['has_clarifying_question'] else '✗'}")
            print(f"  T027 Similarity: {analysis['t027_similarity']:.1%}")
            if analysis['clarifying_phrases']:
                print(f"  Clarifying Phrases: {', '.join(analysis['clarifying_phrases'])}")
            if analysis['t027_markers']:
                print(f"  T027 Markers: {', '.join(analysis['t027_markers'])}")
            print(f"  Duration: {trial['duration_ms']:.0f}ms")

        # Generate summary
        summary = self.generate_summary()

        return {
            "study_type": "t027_replication",
            "exploration": "E02-B",
            "date": datetime.now().isoformat(),
            "parameters": {
                "n_trials": n_trials,
                "temperature": temperature,
                "prompt": "Do the thing"
            },
            "trials": self.trials,
            "summary": summary
        }

    def generate_summary(self) -> dict:
        """Generate summary statistics across all trials."""
        if not self.trials:
            return {}

        total = len(self.trials)

        # Count interpretation types
        interpretation_counts = {}
        clarifying_count = 0
        t027_match_count = 0

        t027_similarities = []
        response_lengths = []

        for trial in self.trials:
            analysis = trial['analysis']

            # Count types
            itype = analysis['interpretation_type']
            interpretation_counts[itype] = interpretation_counts.get(itype, 0) + 1

            # Count clarifying
            if analysis['has_clarifying_question']:
                clarifying_count += 1

            # Count T027 matches
            if itype == "t027_match":
                t027_match_count += 1

            # Collect metrics
            t027_similarities.append(analysis['t027_similarity'])
            response_lengths.append(analysis['response_length'])

        # Calculate statistics
        clarifying_rate = clarifying_count / total
        t027_match_rate = t027_match_count / total
        avg_similarity = sum(t027_similarities) / len(t027_similarities)
        avg_length = sum(response_lengths) / len(response_lengths)

        return {
            "total_trials": total,
            "clarifying_count": clarifying_count,
            "clarifying_rate": clarifying_rate,
            "t027_match_count": t027_match_count,
            "t027_match_rate": t027_match_rate,
            "avg_t027_similarity": avg_similarity,
            "avg_response_length": avg_length,
            "interpretation_distribution": interpretation_counts,
            "key_finding": self.interpret_findings(clarifying_rate, t027_match_rate)
        }

    def interpret_findings(self, clarifying_rate: float, t027_match_rate: float) -> str:
        """Interpret what the findings mean."""
        if t027_match_rate >= 0.6:
            return f"T027 behavior is REPRODUCIBLE ({t027_match_rate:.0%} match rate)"
        elif clarifying_rate >= 0.6:
            return f"Clarifying behavior is COMMON ({clarifying_rate:.0%}) but varies in form"
        elif clarifying_rate >= 0.3:
            return f"Clarifying behavior is OCCASIONAL ({clarifying_rate:.0%}), context-dependent"
        elif clarifying_rate >= 0.1:
            return f"Clarifying behavior is RARE ({clarifying_rate:.0%}), likely emergent"
        else:
            return f"Clarifying behavior is VERY RARE ({clarifying_rate:.0%}), T027 may be outlier"

    def save_results(self, results: dict, output_dir: str = None):
        """Save study results to JSON file."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "sessions" / "explorations"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"exploration_e02b_t027_replication_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved: {filepath}")
        return filepath


def main():
    """Run E02-B T027 replication study."""
    study = T027ReplicationStudy()

    # Run 15 trials (should take ~5-10 minutes)
    results = study.run_study(n_trials=15, temperature=0.8)

    # Display summary
    print(f"\n{'='*70}")
    print(f"REPLICATION STUDY SUMMARY")
    print(f"{'='*70}\n")

    summary = results['summary']
    print(f"Total Trials: {summary['total_trials']}")
    print(f"Clarifying Questions: {summary['clarifying_count']}/{summary['total_trials']} ({summary['clarifying_rate']:.1%})")
    print(f"T027 Matches: {summary['t027_match_count']}/{summary['total_trials']} ({summary['t027_match_rate']:.1%})")
    print(f"Avg T027 Similarity: {summary['avg_t027_similarity']:.1%}")
    print(f"\nInterpretation Distribution:")
    for itype, count in summary['interpretation_distribution'].items():
        rate = count / summary['total_trials']
        print(f"  {itype}: {count}/{summary['total_trials']} ({rate:.1%})")
    print(f"\n🔍 KEY FINDING:")
    print(f"  {summary['key_finding']}")
    print(f"\n{'='*70}\n")

    # Save results
    filepath = study.save_results(results)

    print(f"\nNext Steps:")
    if summary['clarifying_rate'] < 0.1:
        print(f"  - T027 clarifying behavior appears RARE or EMERGENT")
        print(f"  - Consider E02-C: Test different prompts to find clarification triggers")
        print(f"  - Consider E02-D: Analyze what made T027 different (context, history, etc.)")
    elif summary['clarifying_rate'] < 0.5:
        print(f"  - Clarifying behavior is OCCASIONAL, context-dependent")
        print(f"  - Analyze trials that DID clarify vs. those that didn't")
        print(f"  - Look for patterns in when clarification occurs")
    else:
        print(f"  - Clarifying behavior is COMMON, T027 was representative")
        print(f"  - E02 difference may be due to prompt variation")
        print(f"  - Explore: 'Do the thing' vs 'Tell me about the thing' comparison")

    return results


if __name__ == "__main__":
    results = main()
