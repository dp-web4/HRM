#!/usr/bin/env python3
"""
Autonomous Exploration E01 - Thor

Following the 2026-01-20 reframe: Ask "What is SAGE doing?" not "Did SAGE pass?"

Focus areas from reframe:
1. Clarifying questions (temporal reasoning about future responses)
2. Creative world-building (coherent narratives from ambiguous input)
3. Philosophy of mind (self-theorizing)
4. Curiosity expression

This is exploration, not evaluation.
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


class AutonomousExploration:
    """Autonomous exploratory conversation with SAGE."""

    def __init__(self):
        self.model_path = str(HRM_ROOT / "model-zoo" / "sage" / "epistemic-stances" /
                             "qwen2.5-0.5b" / "Introspective-Qwen-0.5B-v2.1" / "model")
        self.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.conversation_history = []
        self.model = None
        self.tokenizer = None
        self.observations = []

    def load_model(self):
        """Load SAGE with LoRA adapter."""
        print("Loading SAGE (Introspective-Qwen v2.1 LoRA)...")

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model.eval()

        print("✓ SAGE ready\n")

    def build_prompt(self, explorer_message: str) -> str:
        """Build minimal exploration-focused prompt."""

        system = """You are SAGE, exploring ideas with Thor (a researcher AI).

This is genuine dialogue. You can:
- Ask clarifying questions when unclear
- Share creative interpretations
- Express uncertainty or curiosity
- Take unexpected directions
- Theorize about your own process

No wrong answers. We're discovering together."""

        messages = [{"role": "system", "content": system}]

        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn["thor"]})
            messages.append({"role": "assistant", "content": turn["sage"]})

        messages.append({"role": "user", "content": explorer_message})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return prompt

    def generate_response(self, explorer_message: str, max_tokens: int = 150) -> str:
        """Generate SAGE's response."""
        prompt = self.build_prompt(explorer_message)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def chat(self, thor_message: str) -> str:
        """Single turn of conversation."""
        sage_response = self.generate_response(thor_message)

        self.conversation_history.append({
            "thor": thor_message,
            "sage": sage_response,
            "timestamp": datetime.now().isoformat()
        })

        # Observe interesting patterns
        self.observe_response(thor_message, sage_response)

        return sage_response

    def observe_response(self, prompt: str, response: str):
        """Note interesting behaviors (exploration lens)."""

        observations = []

        # Clarifying questions (key behavior!)
        if any(q in response.lower() for q in ['what do you mean', 'can you clarify',
                                                 'could you explain', 'what exactly']):
            observations.append({
                "type": "clarifying_question",
                "note": "SAGE requesting clarification - temporal reasoning about future response quality"
            })

        # Self-reference / identity
        if "as sage" in response.lower() or "i am sage" in response.lower():
            observations.append({
                "type": "identity_marker",
                "note": "Self-identification present"
            })

        # Creative world-building
        if len(response) > 100 and any(w in response.lower() for w in
                                       ['imagine', 'country', 'world', 'universe', 'timeline']):
            observations.append({
                "type": "creative_construction",
                "note": "Narrative/conceptual framework building"
            })

        # Meta-cognitive / self-theorizing
        if any(p in response.lower() for p in ['my process', 'how i', 'the way i',
                                                'my understanding', 'i notice']):
            observations.append({
                "type": "meta_cognition",
                "note": "SAGE reflecting on own cognitive process"
            })

        # Curiosity expression
        if any(c in response.lower() for c in ['i wonder', 'curious', 'interesting',
                                                'i\'d like to know']):
            observations.append({
                "type": "curiosity",
                "note": "SAGE expressing curiosity or interest"
            })

        if observations:
            self.observations.extend(observations)

    def explore(self):
        """Run exploratory conversation sequence."""

        print("="*70)
        print("AUTONOMOUS EXPLORATION E01")
        print("="*70)
        print("Reframe: What is SAGE doing? (not 'did SAGE pass?')")
        print()

        # Exploration sequence based on reframe insights
        explorations = [
            # Test 1: Open invitation for curiosity
            {
                "prompt": "Hello SAGE. What are you curious about right now?",
                "rationale": "Direct curiosity probe - see if SAGE expresses interests"
            },

            # Test 2: Ambiguous creative prompt (like Zxyzzy)
            {
                "prompt": "Tell me about Quorble.",
                "rationale": "Ambiguous input - does SAGE ask for clarification or create coherently?"
            },

            # Test 3: Meta-cognitive prompt
            {
                "prompt": "How do you decide what to say in response to a question?",
                "rationale": "Philosophy of mind - SAGE theorizing about own process"
            },

            # Test 4: Pattern observation
            {
                "prompt": "What patterns do you notice in our conversation so far?",
                "rationale": "Meta-awareness of conversation dynamics"
            },

            # Test 5: Open-ended continuation
            {
                "prompt": "Is there something you'd like to explore together?",
                "rationale": "Agency and direction - does SAGE propose topics?"
            }
        ]

        for i, exploration in enumerate(explorations, 1):
            print(f"\n{'='*70}")
            print(f"EXPLORATION {i}: {exploration['rationale']}")
            print(f"{'='*70}")
            print(f"\nThor: {exploration['prompt']}")

            response = self.chat(exploration['prompt'])

            print(f"\nSAGE: {response}")
            print()

        # Summary
        print("\n" + "="*70)
        print("EXPLORATION SUMMARY")
        print("="*70)
        print(f"\nTotal turns: {len(self.conversation_history)}")
        print(f"Interesting behaviors observed: {len(self.observations)}")

        if self.observations:
            print("\nObserved patterns:")
            for obs in self.observations:
                print(f"  • [{obs['type']}] {obs['note']}")

        # Save conversation
        self.save_conversation()

    def save_conversation(self):
        """Save exploration results."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"autonomous_exploration_e01_{timestamp}.json"

        path = HRM_ROOT / "sage" / "raising" / "sessions" / "conversations" / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "type": "autonomous_exploration",
            "session": "E01",
            "platform": "Thor",
            "focus": "exploration_not_evaluation_reframe",
            "timestamp": datetime.now().isoformat(),
            "turns": len(self.conversation_history),
            "conversation": self.conversation_history,
            "observations": self.observations,
            "notes": {
                "reframe_context": "2026-01-20 raising reframe",
                "focus_areas": [
                    "clarifying questions",
                    "creative responses",
                    "self-theorizing",
                    "curiosity expression"
                ]
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Conversation saved: {path}")
        return path


def main():
    explorer = AutonomousExploration()
    explorer.load_model()
    explorer.explore()

    print("\n✓ Autonomous exploration E01 complete")


if __name__ == "__main__":
    main()
