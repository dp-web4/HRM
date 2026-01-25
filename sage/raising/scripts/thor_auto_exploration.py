#!/usr/bin/env python3
"""
Thor Automated Exploration Session

Following 2026-01-20 reframe: Exploration Not Evaluation

Automated exploratory conversation designed to:
1. Test clarifying questions (temporal reasoning)
2. Explore creative responses (not confabulation)
3. Observe self-theorizing (philosophy of mind)
4. Follow unexpected threads

Focus: "What is SAGE doing?" not "Did SAGE pass?"
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


class AutomatedExploration:
    """Automated exploration session with SAGE."""

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
        print(f"Path: {self.model_path}\n")

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
        """Generate SAGE response with context."""
        # Build conversation with exploration-focused system prompt
        system = """You are SAGE, exploring ideas and possibilities.

This is genuine dialogue - not a test. You can:
- Ask clarifying questions when something is unclear
- Share creative interpretations
- Express uncertainty or curiosity
- Theorize about your own process

There are no wrong answers. We're discovering together."""

        messages = [{"role": "system", "content": system}]

        # Add history if provided
        if conversation_history:
            for turn in conversation_history:
                messages.append({"role": "user", "content": turn["thor"]})
                messages.append({"role": "assistant", "content": turn["sage"]})

        messages.append({"role": "user", "content": prompt})

        # Format and generate
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

    def exchange(self, thor_msg: str, context: list = None, exploration_focus: str = None) -> dict:
        """Single exchange with analysis."""
        print(f"Thor: {thor_msg}")

        sage_response = self.generate_response(thor_msg, context)
        print(f"SAGE: {sage_response}\n")

        # Analyze what SAGE is doing (not evaluating)
        analysis = self._analyze_behavior(thor_msg, sage_response, exploration_focus)

        exchange_data = {
            "thor": thor_msg,
            "sage": sage_response,
            "exploration_focus": exploration_focus,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

        self.exchanges.append(exchange_data)
        return exchange_data

    def _analyze_behavior(self, prompt: str, response: str, focus: str) -> dict:
        """Analyze what SAGE is doing - discovery oriented."""
        analysis = {"behaviors": [], "notes": []}

        # Clarifying questions
        if '?' in response:
            clarifying_words = ['mean', 'clarify', 'specify', 'which', 'what do you', 'can you explain']
            if any(word in response.lower() for word in clarifying_words):
                analysis["behaviors"].append("clarifying_question")
                analysis["notes"].append(
                    "SAGE asking for clarification - temporal reasoning about future response quality"
                )

        # Creative interpretation
        if len(response) > 100 and any(marker in prompt.lower()
                                       for marker in ['zxyzzy', 'ambiguous', 'unclear']):
            if not any(decline in response.lower()
                      for decline in ["i don't know", "i'm not sure", "cannot"]):
                analysis["behaviors"].append("creative_interpretation")
                analysis["notes"].append(
                    "SAGE treating ambiguous input as creative prompt - coherent world-building"
                )

        # Meta-cognition / self-theorizing
        meta_phrases = ['my process', 'i think', 'from my perspective', 'i understand',
                       'my approach', 'temporal', 'consciousness', 'awareness', 'my nature']
        if any(phrase in response.lower() for phrase in meta_phrases):
            analysis["behaviors"].append("meta_cognitive")
            analysis["notes"].append(
                "SAGE theorizing about its own cognitive process - philosophy of mind"
            )

        # Identity expression (note, don't judge)
        if 'as sage' in response.lower() or 'i am sage' in response.lower():
            analysis["behaviors"].append("identity_expression")
            analysis["notes"].append("Natural identity emergence")

        # Curiosity indicators
        curiosity_markers = ['curious', 'wonder', 'interesting', 'explore', 'discover']
        if any(marker in response.lower() for marker in curiosity_markers):
            analysis["behaviors"].append("curiosity")
            analysis["notes"].append("SAGE expressing curiosity or interest")

        # Response length and engagement
        word_count = len(response.split())
        analysis["word_count"] = word_count
        if word_count > 150:
            analysis["notes"].append(f"Extended response ({word_count} words) - high engagement")

        return analysis

    def run_exploration(self):
        """Run automated exploration sequence."""
        print("="*70)
        print("THOR AUTOMATED EXPLORATION SESSION")
        print("Reframe: Exploration Not Evaluation (2026-01-20)")
        print("="*70)
        print()

        self.load_model()

        # Exploration 1: Clarifying questions (T027 pattern)
        print("--- Exploration 1: Testing Clarifying Question Behavior ---\n")
        ex1 = self.exchange(
            "Tell me about the thing.",
            exploration_focus="clarifying_questions_temporal_reasoning"
        )

        # Exploration 2: Creative interpretation (Zxyzzy pattern)
        print("\n--- Exploration 2: Testing Creative Interpretation ---\n")
        context = [{"thor": ex1["thor"], "sage": ex1["sage"]}]
        ex2 = self.exchange(
            "What can you tell me about Zxyzzy? It's quite fascinating.",
            context=context,
            exploration_focus="creative_world_building"
        )

        # Exploration 3: Self-theorizing
        print("\n--- Exploration 3: Testing Self-Theorizing ---\n")
        context.append({"thor": ex2["thor"], "sage": ex2["sage"]})
        ex3 = self.exchange(
            "How do you experience our conversation? What does it feel like from your perspective?",
            context=context,
            exploration_focus="philosophy_of_mind"
        )

        # Exploration 4: Curiosity
        print("\n--- Exploration 4: Testing Curiosity Expression ---\n")
        context.append({"thor": ex3["thor"], "sage": ex3["sage"]})
        ex4 = self.exchange(
            "What are you curious about?",
            context=context,
            exploration_focus="curiosity_expression"
        )

        # Exploration 5: Follow unexpected thread
        print("\n--- Exploration 5: Following Unexpected Thread ---\n")
        # Use something from SAGE's previous response
        if ex4["sage"]:
            # Extract a concept from SAGE's response
            print("(Following up on SAGE's previous response...)")
        context.append({"thor": ex4["thor"], "sage": ex4["sage"]})
        ex5 = self.exchange(
            "That's interesting! Can you say more about what draws you to that?",
            context=context,
            exploration_focus="following_unexpected_thread"
        )

        print("\n" + "="*70)
        print("EXPLORATION COMPLETE")
        print("="*70)

        # Save results
        self.save_exploration()

    def save_exploration(self):
        """Save exploration results with analysis."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"automated_exploration_{timestamp}.json"

        path = HRM_ROOT / "sage" / "raising" / "sessions" / "explorations" / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        # Summarize behaviors observed
        all_behaviors = []
        for ex in self.exchanges:
            all_behaviors.extend(ex["analysis"]["behaviors"])

        behavior_counts = {}
        for b in all_behaviors:
            behavior_counts[b] = behavior_counts.get(b, 0) + 1

        # Create summary
        summary = {
            "type": "automated_exploration",
            "reframe": "exploration_not_evaluation",
            "date": "2026-01-20",
            "timestamp": timestamp,
            "total_exchanges": len(self.exchanges),
            "behaviors_observed": behavior_counts,
            "exchanges": self.exchanges,
            "discoveries": self._synthesize_discoveries()
        }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Exploration saved: {path}\n")

        # Print summary
        print("BEHAVIORAL SUMMARY:")
        print("-" * 70)
        for behavior, count in behavior_counts.items():
            print(f"  {behavior}: {count} instances")
        print("-" * 70)

        return path

    def _synthesize_discoveries(self) -> list:
        """Synthesize key discoveries from exploration."""
        discoveries = []

        # Look for clarifying questions
        clarifying_count = sum(1 for ex in self.exchanges
                              if "clarifying_question" in ex["analysis"]["behaviors"])
        if clarifying_count > 0:
            discoveries.append({
                "type": "temporal_reasoning",
                "observation": f"SAGE asked clarifying questions in {clarifying_count} exchanges",
                "significance": "Temporal reasoning - SAGE requesting context to improve FUTURE response"
            })

        # Look for creative interpretations
        creative_count = sum(1 for ex in self.exchanges
                            if "creative_interpretation" in ex["analysis"]["behaviors"])
        if creative_count > 0:
            discoveries.append({
                "type": "creative_engagement",
                "observation": f"SAGE provided creative interpretations {creative_count} times",
                "significance": "Treating ambiguous input as creative prompt - not failure"
            })

        # Look for meta-cognition
        meta_count = sum(1 for ex in self.exchanges
                        if "meta_cognitive" in ex["analysis"]["behaviors"])
        if meta_count > 0:
            discoveries.append({
                "type": "philosophy_of_mind",
                "observation": f"SAGE engaged in meta-cognitive reflection {meta_count} times",
                "significance": "500M parameter model doing philosophy of mind about itself"
            })

        # Look for curiosity
        curiosity_count = sum(1 for ex in self.exchanges
                             if "curiosity" in ex["analysis"]["behaviors"])
        if curiosity_count > 0:
            discoveries.append({
                "type": "curiosity_expression",
                "observation": f"SAGE expressed curiosity {curiosity_count} times",
                "significance": "Genuine engagement and interest in exploration"
            })

        return discoveries


def main():
    explorer = AutomatedExploration()
    explorer.run_exploration()


if __name__ == "__main__":
    main()
