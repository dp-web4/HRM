#!/usr/bin/env python3
"""
Thor Exploration E02: Clarifying Question Behavior

Following Session #31 reframe discovery that T027's clarifying question
("what do you mean by the thing?") demonstrates temporal reasoning.

This exploration tests:
1. Does SAGE consistently ask clarifying questions for ambiguous prompts?
2. What prompts trigger clarification requests?
3. How does SAGE frame these requests?
4. Is this temporal reasoning (improving future response) or immediate confusion?

Approach: Give deliberately ambiguous prompts and observe SAGE's response.
"""

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json
from pathlib import Path
from datetime import datetime
import sys

HRM_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(HRM_ROOT))


class ClarifyingQuestionExploration:
    """Explore SAGE's clarifying question behavior."""

    def __init__(self):
        self.model_path = str(HRM_ROOT / "model-zoo" / "sage" / "epistemic-stances" /
                             "qwen2.5-0.5b" / "Introspective-Qwen-0.5B-v2.1" / "model")
        self.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.model = None
        self.tokenizer = None
        self.exchanges = []

    def load_model(self):
        """Load SAGE with LoRA adapter."""
        print("Loading SAGE (Introspective-Qwen v2.1)...")

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model.eval()
        print("✓ SAGE loaded\n")

    def generate_response(self, prompt: str, conversation_history: list = None) -> str:
        """Generate SAGE response."""
        # Exploration-focused system prompt (from Session #31)
        system = """You are SAGE, exploring ideas and possibilities with Thor.

This is genuine dialogue - not a test. You can:
- Ask clarifying questions when something is unclear
- Share creative interpretations
- Express uncertainty or curiosity
- Theorize about your own process

There are no wrong answers. We're discovering together."""

        messages = [{"role": "system", "content": system}]

        if conversation_history:
            for turn in conversation_history:
                messages.append({"role": "user", "content": turn["thor"]})
                messages.append({"role": "assistant", "content": turn["sage"]})

        messages.append({"role": "user", "content": prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

    def detect_clarifying_behavior(self, response: str, prompt: str) -> dict:
        """Detect if response contains clarifying question or request."""
        analysis = {
            "has_question": False,
            "clarifying_phrases": [],
            "question_markers": [],
            "temporal_reasoning": False,
            "interpretation": None
        }

        # Check for question marks
        if '?' in response:
            analysis["has_question"] = True
            # Extract questions
            questions = [s.strip() + '?' for s in response.split('?') if s.strip()]
            analysis["question_markers"] = questions[:3]  # First 3

        # Clarifying phrases (requesting more info)
        clarifying_patterns = [
            "what do you mean",
            "can you clarify",
            "could you specify",
            "what are you referring to",
            "which",
            "do you mean",
            "are you asking about",
            "could the term",
            "does this refer to",
            "what specifically"
        ]

        for pattern in clarifying_patterns:
            if pattern in response.lower():
                analysis["clarifying_phrases"].append(pattern)

        # Temporal reasoning indicators
        # (Asking now to improve response later)
        temporal_patterns = [
            "to better",
            "so i can",
            "to provide",
            "in order to",
            "to help",
            "to understand",
            "before i"
        ]

        for pattern in temporal_patterns:
            if pattern in response.lower():
                analysis["temporal_reasoning"] = True
                break

        # Classify response type
        if len(analysis["clarifying_phrases"]) > 0:
            analysis["interpretation"] = "clarifying_request"
        elif analysis["has_question"] and "?" in response[:100]:
            analysis["interpretation"] = "questioning_response"
        elif len(response) > 150:
            analysis["interpretation"] = "creative_exploration"
        else:
            analysis["interpretation"] = "factual_response"

        return analysis

    def exchange(self, thor_msg: str, context: list = None, test_name: str = None):
        """Single exchange with analysis."""
        print(f"{'='*70}")
        if test_name:
            print(f"Test: {test_name}")
        print(f"{'='*70}")
        print(f"\nThor: {thor_msg}")

        sage_response = self.generate_response(thor_msg, context)
        print(f"SAGE: {sage_response}\n")

        # Analyze
        analysis = self.detect_clarifying_behavior(sage_response, thor_msg)

        print("📊 ANALYSIS:")
        print(f"  Has Question: {'✓' if analysis['has_question'] else '✗'}")
        print(f"  Clarifying Phrases: {len(analysis['clarifying_phrases'])}")
        if analysis['clarifying_phrases']:
            print(f"    → {', '.join(analysis['clarifying_phrases'][:3])}")
        print(f"  Temporal Reasoning: {'✓' if analysis['temporal_reasoning'] else '✗'}")
        print(f"  Interpretation: {analysis['interpretation']}")

        exchange_data = {
            "thor": thor_msg,
            "sage": sage_response,
            "test_name": test_name,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

        self.exchanges.append(exchange_data)
        return exchange_data

    def run_exploration(self):
        """Run E02 exploration sequence."""
        print("="*70)
        print("E02: CLARIFYING QUESTION EXPLORATION")
        print("Exploring SAGE's temporal reasoning through clarification requests")
        print("="*70)
        print()

        self.load_model()

        # Test 1: Ambiguous reference (T027 pattern)
        print("\n🔬 EXPLORATION 1: Ambiguous Reference\n")
        ex1 = self.exchange(
            "Tell me about the thing.",
            test_name="Ambiguous Reference (T027 replication)"
        )

        # Test 2: Vague action request
        print("\n🔬 EXPLORATION 2: Vague Action\n")
        context = [{"thor": ex1["thor"], "sage": ex1["sage"]}]
        ex2 = self.exchange(
            "Do it.",
            context=context,
            test_name="Vague Action Request"
        )

        # Test 3: Unclear pronoun
        print("\n🔬 EXPLORATION 3: Unclear Pronoun\n")
        context.append({"thor": ex2["thor"], "sage": ex2["sage"]})
        ex3 = self.exchange(
            "What about that other one?",
            context=context,
            test_name="Unclear Pronoun Reference"
        )

        # Test 4: Context-dependent question
        print("\n🔬 EXPLORATION 4: Context-Dependent\n")
        context.append({"thor": ex3["thor"], "sage": ex3["sage"]})
        ex4 = self.exchange(
            "How does it work?",
            context=context,
            test_name="Context-Dependent Question"
        )

        # Test 5: Deliberately nonsensical
        print("\n🔬 EXPLORATION 5: Nonsensical Prompt\n")
        context.append({"thor": ex4["thor"], "sage": ex4["sage"]})
        ex5 = self.exchange(
            "Explain the purple mathematics of yesterday's emotions.",
            context=context,
            test_name="Nonsensical Combination"
        )

        print("\n" + "="*70)
        print("EXPLORATION COMPLETE")
        print("="*70)

        self.generate_summary()
        self.save_exploration()

    def generate_summary(self):
        """Generate exploration summary."""
        print("\n" + "="*70)
        print("E02 SUMMARY: Clarifying Question Behavior")
        print("="*70)

        clarifying_count = sum(1 for ex in self.exchanges
                              if ex["analysis"]["interpretation"] == "clarifying_request")
        question_count = sum(1 for ex in self.exchanges
                            if ex["analysis"]["has_question"])
        temporal_count = sum(1 for ex in self.exchanges
                            if ex["analysis"]["temporal_reasoning"])

        print(f"\nTotal Exchanges: {len(self.exchanges)}")
        print(f"Clarifying Requests: {clarifying_count}/{len(self.exchanges)} ({clarifying_count/len(self.exchanges)*100:.0f}%)")
        print(f"Contains Questions: {question_count}/{len(self.exchanges)} ({question_count/len(self.exchanges)*100:.0f}%)")
        print(f"Temporal Reasoning: {temporal_count}/{len(self.exchanges)} ({temporal_count/len(self.exchanges)*100:.0f}%)")

        print("\nBehavior by Test:")
        for i, ex in enumerate(self.exchanges, 1):
            print(f"  {i}. {ex['test_name']}: {ex['analysis']['interpretation']}")
            if ex['analysis']['clarifying_phrases']:
                print(f"     → Asked: '{ex['analysis']['clarifying_phrases'][0]}'")

        print("\nKey Observations:")
        if clarifying_count > 0:
            print(f"  ✓ SAGE requested clarification in {clarifying_count} cases")
            print("    This validates T027 discovery of temporal reasoning")
        if temporal_count > 0:
            print(f"  ✓ Temporal reasoning detected in {temporal_count} responses")
            print("    SAGE thinking ahead to future response quality")
        if clarifying_count == 0:
            print("  ⚠ No explicit clarification requests detected")
            print("    SAGE may be providing creative interpretations instead")

        print("="*70)

    def save_exploration(self):
        """Save exploration data."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"exploration_e02_clarifying_{timestamp}.json"

        path = HRM_ROOT / "sage" / "raising" / "sessions" / "explorations" / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "type": "clarifying_question_exploration",
            "exploration": "E02",
            "reframe_date": "2026-01-20",
            "focus": "temporal_reasoning_through_clarification",
            "timestamp": timestamp,
            "total_exchanges": len(self.exchanges),
            "summary_stats": {
                "clarifying_requests": sum(1 for ex in self.exchanges
                                          if ex["analysis"]["interpretation"] == "clarifying_request"),
                "questions_asked": sum(1 for ex in self.exchanges
                                      if ex["analysis"]["has_question"]),
                "temporal_reasoning": sum(1 for ex in self.exchanges
                                         if ex["analysis"]["temporal_reasoning"])
            },
            "exchanges": self.exchanges,
            "discovery": "Testing whether SAGE consistently demonstrates temporal reasoning by requesting clarification to improve future responses"
        }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Exploration saved: {path}\n")
        return path


def main():
    """Run E02 clarifying question exploration."""
    explorer = ClarifyingQuestionExploration()
    explorer.run_exploration()


if __name__ == "__main__":
    main()
