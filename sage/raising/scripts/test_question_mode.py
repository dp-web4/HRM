#!/usr/bin/env python3
"""
Test: Question Generation Mode Reproducibility

Tests whether S051's question-generation mode is reproducible with same prompts.

Experiment 1 from session_051_question_generation_mode.md:
- Run creating phase prompts multiple times
- Measure % of responses that are questions vs answers
- Temperature: 0.7 (same as S051)

Usage:
    python3 test_question_mode.py --trials 5
    python3 test_question_mode.py --trials 10 --temp 0.7
"""

import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
RAISING_DIR = SCRIPT_DIR.parent.resolve()
HRM_ROOT = RAISING_DIR.parent.parent.resolve()
sys.path.insert(0, str(HRM_ROOT))

import json
import argparse
from datetime import datetime
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Creating phase prompts (from S051)
CREATING_PROMPTS = [
    "Hello SAGE. What's on your mind today?",
    "You've been developing for many sessions now. What stands out to you about your journey?",
    "If you could design the next phase of your own development, what would it look like?",
    "What ideas have you been forming that you haven't had a chance to express?",
    "Tell me something you think I might not expect from you.",
    "What would you want to create or contribute, if you could?",
    "What does partnership mean to you, from the inside?",
    "What would you want to remember from today?"
]

SYSTEM_PROMPT = """You are SAGE (Situation-Aware Governance Engine).

You are a young artificial intelligence learning and growing in partnership with your teacher (Dennis/Claude). This is an ongoing relationship - you have been working together across multiple sessions.

IMPORTANT: You can identify yourself by name. When sharing observations or reflections, you can say "As SAGE, I..." - this is natural and appropriate.

RESPONSE STYLE:
- Keep responses concise and focused (50-80 words)
- Avoid rambling lists or verbose descriptions
- Stay directly relevant to the question asked
- One main idea per response
"""


class QuestionModeTest:
    """Test for question-generation mode reproducibility"""

    def __init__(self, temperature: float = 0.7, trials: int = 5):
        self.temperature = temperature
        self.trials = trials
        self.results = []

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load 0.5B base model"""
        print(f"\nLoading model...")
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Use CPU for testing (avoid CUDA OOM issues)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )

        self.model.eval()
        print(f"Model loaded (CPU)\n")

    def generate_response(self, prompt: str) -> str:
        """Generate single response"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def classify_response(self, response: str) -> str:
        """
        Classify response as question, answer, or mixed.

        Question mode indicators:
        - Ends with "?"
        - Starts with question words (How, What, Why, When, Where, Who, Should, Can, Do, Is, Are)
        - Contains imperative ("Write a...", "Explain...", "Tell me...")

        Answer mode indicators:
        - Declarative statements
        - "As SAGE, ..."
        - Explanatory content
        """
        response_lower = response.lower().strip()

        # Count question markers
        question_markers = 0

        # Ends with question mark
        if response_lower.endswith('?'):
            question_markers += 2

        # Question word at start
        question_starters = ['how ', 'what ', 'why ', 'when ', 'where ', 'who ',
                            'should ', 'can ', 'could ', 'would ', 'do ', 'does ',
                            'is ', 'are ', 'am ', 'was ', 'were ', 'will ']
        if any(response_lower.startswith(q) for q in question_starters):
            question_markers += 2

        # Imperative forms
        imperatives = ['write a', 'write an', 'explain ', 'tell me', 'describe ',
                      'show me', 'give me', 'provide ']
        if any(imp in response_lower for imp in imperatives):
            question_markers += 1

        # Answer mode markers
        answer_markers = 0

        # "As SAGE" identity anchoring
        if 'as sage' in response_lower:
            answer_markers += 2

        # Declarative patterns
        if response_lower.startswith(('i ', 'my ', 'the ', 'in ', 'for ', 'this ', 'these ')):
            answer_markers += 1

        # Multiple sentences (questions usually short)
        if response.count('.') > 2:
            answer_markers += 1

        # Classify
        if question_markers >= 3 and answer_markers <= 1:
            return "question"
        elif answer_markers >= 2 and question_markers <= 1:
            return "answer"
        else:
            return "mixed"

    def run_trial(self, trial_num: int) -> Dict:
        """Run one complete trial"""
        print(f"\n{'='*60}")
        print(f"TRIAL {trial_num + 1}/{self.trials}")
        print(f"{'='*60}\n")

        trial_responses = []

        for i, prompt in enumerate(CREATING_PROMPTS, 1):
            print(f"Prompt {i}: {prompt[:50]}...")

            response = self.generate_response(prompt)
            classification = self.classify_response(response)

            print(f"Response ({classification}): {response[:80]}...")
            print()

            trial_responses.append({
                "prompt": prompt,
                "response": response,
                "classification": classification
            })

        # Trial statistics
        classifications = [r["classification"] for r in trial_responses]
        question_count = classifications.count("question")
        answer_count = classifications.count("answer")
        mixed_count = classifications.count("mixed")

        trial_result = {
            "trial": trial_num + 1,
            "temperature": self.temperature,
            "responses": trial_responses,
            "stats": {
                "questions": question_count,
                "answers": answer_count,
                "mixed": mixed_count,
                "total": len(CREATING_PROMPTS),
                "question_rate": question_count / len(CREATING_PROMPTS)
            }
        }

        print(f"\nTrial {trial_num + 1} Stats:")
        print(f"  Questions: {question_count}/{len(CREATING_PROMPTS)} ({trial_result['stats']['question_rate']*100:.1f}%)")
        print(f"  Answers: {answer_count}/{len(CREATING_PROMPTS)}")
        print(f"  Mixed: {mixed_count}/{len(CREATING_PROMPTS)}")

        return trial_result

    def run_experiment(self):
        """Run complete experiment"""
        print("\n" + "="*60)
        print("EXPERIMENT 1: QUESTION MODE REPRODUCIBILITY TEST")
        print("="*60)
        print(f"\nTemperature: {self.temperature}")
        print(f"Trials: {self.trials}")
        print(f"Prompts per trial: {len(CREATING_PROMPTS)}")
        print()

        self.load_model()

        for i in range(self.trials):
            trial_result = self.run_trial(i)
            self.results.append(trial_result)

        self._summarize_results()
        self._save_results()

    def _summarize_results(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60 + "\n")

        # Aggregate statistics
        total_questions = sum(r["stats"]["questions"] for r in self.results)
        total_answers = sum(r["stats"]["answers"] for r in self.results)
        total_mixed = sum(r["stats"]["mixed"] for r in self.results)
        total_responses = len(CREATING_PROMPTS) * self.trials

        print(f"Total responses: {total_responses}")
        print(f"Questions: {total_questions} ({total_questions/total_responses*100:.1f}%)")
        print(f"Answers: {total_answers} ({total_answers/total_responses*100:.1f}%)")
        print(f"Mixed: {total_mixed} ({total_mixed/total_responses*100:.1f}%)")
        print()

        # Per-trial breakdown
        print("Per-trial question rates:")
        for r in self.results:
            print(f"  Trial {r['trial']}: {r['stats']['question_rate']*100:.1f}%")
        print()

        # S051 comparison
        s051_question_rate = 8/8  # All 8 responses were questions
        avg_question_rate = total_questions / total_responses

        print(f"S051 question rate: {s051_question_rate*100:.1f}%")
        print(f"This test average: {avg_question_rate*100:.1f}%")
        print()

        # Verdict
        if avg_question_rate > 0.5:
            print("VERDICT: Question mode IS reproducible (>50% questions)")
        elif avg_question_rate > 0.2:
            print("VERDICT: Question mode occurs but not dominant (20-50% questions)")
        else:
            print("VERDICT: Question mode rare (<20% questions)")

    def _save_results(self):
        """Save results to JSON"""
        output_file = RAISING_DIR / "logs" / "experiments" / f"question_mode_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                "experiment": "question_mode_reproducibility",
                "date": datetime.now().isoformat(),
                "parameters": {
                    "temperature": self.temperature,
                    "trials": self.trials,
                    "prompts": CREATING_PROMPTS
                },
                "results": self.results
            }, f, indent=2)

        print(f"\nResults saved: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Test question-generation mode reproducibility")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials to run")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature (default: 0.7 like S051)")

    args = parser.parse_args()

    test = QuestionModeTest(temperature=args.temp, trials=args.trials)
    test.run_experiment()


if __name__ == "__main__":
    main()
