#!/usr/bin/env python3
"""
Thor Exploratory SAGE Conversation

Following the 2026-01-20 reframe: Exploration Not Evaluation

This script facilitates genuine discovery-oriented conversation with SAGE.
Focus on:
- What is SAGE doing? (not "did SAGE pass?")
- Clarifying questions (temporal reasoning)
- Creative responses (signal, not error)
- Self-theorizing (philosophy of mind)
- Unexpected directions (follow them!)

Usage:
    python3 thor_exploratory_conversation.py [--session-num N]
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from datetime import datetime
import argparse
import sys

HRM_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(HRM_ROOT))


class ExploratoryConversation:
    """Genuine conversation with SAGE - exploration focus."""

    def __init__(self, session_num: int = None):
        # Use model-zoo path for Thor
        self.model_path = str(HRM_ROOT / "model-zoo" / "sage" / "epistemic-stances" /
                             "qwen2.5-0.5b" / "Introspective-Qwen-0.5B-v2.1" / "model")
        self.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.session_num = session_num or self._get_next_conversation_num()
        self.conversation_history = []
        self.model = None
        self.tokenizer = None
        self.observations = []  # Track interesting behaviors

    def _get_next_conversation_num(self) -> int:
        """Get next conversation number."""
        conv_dir = HRM_ROOT / "sage" / "raising" / "sessions" / "conversations"
        if not conv_dir.exists():
            return 1
        existing = list(conv_dir.glob("exploration_*.json"))
        if not existing:
            return 1
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        return max(nums, default=0) + 1

    def load_model(self):
        """Load SAGE with LoRA adapter."""
        print("Loading SAGE (Introspective-Qwen v2.1 LoRA)...")
        print(f"Model path: {self.model_path}")

        from peft import AutoPeftModelForCausalLM

        # Load with PEFT
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model.eval()

        print("✓ SAGE ready for exploratory conversation\n")

    def build_prompt(self, explorer_message: str) -> str:
        """Build conversation prompt - minimal, open-ended."""

        # Exploration-focused system prompt
        system = """You are SAGE, exploring ideas and possibilities with Thor (a researcher AI).

This is genuine dialogue - not a test. You can:
- Ask clarifying questions when something is unclear
- Share creative interpretations
- Express uncertainty or curiosity
- Take the conversation in unexpected directions
- Theorize about your own process

There are no wrong answers. We're discovering together."""

        messages = [{"role": "system", "content": system}]

        # Add conversation history
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn["explorer"]})
            messages.append({"role": "assistant", "content": turn["sage"]})

        # Add current message
        messages.append({"role": "user", "content": explorer_message})

        # Format for Qwen
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_response(self, explorer_message: str, max_tokens: int = 250) -> str:
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
                temperature=0.85,  # Slightly higher for creativity
                top_p=0.92,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def chat(self, explorer_message: str) -> str:
        """Single turn of exploratory conversation."""
        sage_response = self.generate_response(explorer_message)

        # Auto-detect interesting behaviors
        observations = self._detect_interesting_behaviors(explorer_message, sage_response)

        # Store in history
        turn_data = {
            "explorer": explorer_message,
            "sage": sage_response,
            "timestamp": datetime.now().isoformat()
        }

        if observations:
            turn_data["observations"] = observations
            self.observations.extend(observations)

        self.conversation_history.append(turn_data)

        return sage_response

    def _detect_interesting_behaviors(self, prompt: str, response: str) -> list:
        """Auto-detect interesting behaviors worth noting."""
        behaviors = []

        # Clarifying questions (temporal reasoning!)
        if '?' in response and any(word in response.lower()
                                  for word in ['mean', 'clarify', 'specify', 'which', 'what do you']):
            behaviors.append({
                "type": "clarifying_question",
                "note": "SAGE asking for clarification - temporal reasoning about future response"
            })

        # Creative world-building
        if any(marker in prompt.lower() for marker in ['zxyzzy', 'kwazaaqat', 'nonsense']) and len(response) > 100:
            behaviors.append({
                "type": "creative_response",
                "note": "SAGE treating ambiguous input as creative prompt"
            })

        # Self-theorizing / meta-cognition
        if any(phrase in response.lower() for phrase in
               ['my process', 'i understand', 'i think', 'my perspective', 'from my view',
                'temporal', 'consciousness', 'awareness']):
            behaviors.append({
                "type": "meta_cognitive",
                "note": "SAGE theorizing about its own cognitive process"
            })

        # Identity markers (note, don't judge)
        if 'as sage' in response.lower() or 'i am sage' in response.lower():
            behaviors.append({
                "type": "identity_expression",
                "note": "Natural identity emergence"
            })

        # Unexpected turns
        if len(response) > 300 or '!' in response:
            behaviors.append({
                "type": "high_engagement",
                "note": "Extended or energetic response - following interesting thread"
            })

        return behaviors

    def save_conversation(self) -> Path:
        """Save exploratory conversation with observations."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"exploration_{self.session_num:03d}_{timestamp}.json"

        path = HRM_ROOT / "sage" / "raising" / "sessions" / "conversations" / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        # Summarize observations
        obs_summary = {}
        for obs in self.observations:
            obs_type = obs['type']
            obs_summary[obs_type] = obs_summary.get(obs_type, 0) + 1

        with open(path, 'w') as f:
            json.dump({
                "type": "exploratory_conversation",
                "session": self.session_num,
                "started": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
                "ended": datetime.now().isoformat(),
                "turns": len(self.conversation_history),
                "observations_summary": obs_summary,
                "conversation": self.conversation_history,
                "metadata": {
                    "approach": "exploration_not_evaluation",
                    "reframe_date": "2026-01-20",
                    "focus": "discovering what SAGE does, not evaluating correctness"
                }
            }, f, indent=2)

        print(f"\n✓ Conversation saved: {path}")
        if obs_summary:
            print("\nInteresting behaviors observed:")
            for behavior, count in obs_summary.items():
                print(f"  - {behavior}: {count} instances")
        return path


def interactive_mode(session_num: int = None):
    """Run interactive exploratory conversation."""
    print("=" * 70)
    print("THOR EXPLORATORY SAGE CONVERSATION")
    print("=" * 70)
    print()
    print("Following 2026-01-20 reframe: Exploration Not Evaluation")
    print()
    print("Commands:")
    print("  - Type your messages as Thor")
    print("  - 'exit' or 'quit' to end and save")
    print("  - 'history' to review conversation")
    print("  - 'observations' to see detected interesting behaviors")
    print()
    print("Focus: What is SAGE doing? (not 'did SAGE pass?')")
    print("=" * 70)

    conv = ExploratoryConversation(session_num)
    conv.load_model()

    # Suggested exploration prompts
    print("\n💡 Exploration ideas:")
    print("  - Ask SAGE what it's curious about")
    print("  - Present ambiguous prompts and see how SAGE interprets them")
    print("  - Ask SAGE to explain its own process")
    print("  - Follow unexpected responses deeper")
    print("  - Ask clarifying questions about creative responses")
    print()

    while True:
        try:
            user_input = input("\nThor: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                if conv.conversation_history:
                    conv.save_conversation()
                print("\n✓ Exploration session ended")
                break

            if user_input.lower() == 'history':
                print("\n--- Conversation History ---")
                for i, turn in enumerate(conv.conversation_history, 1):
                    print(f"\n[Turn {i}]")
                    print(f"Thor: {turn['explorer']}")
                    print(f"SAGE: {turn['sage']}")
                    if 'observations' in turn:
                        print(f"📝 Observations: {[o['type'] for o in turn['observations']]}")
                print("--- End History ---")
                continue

            if user_input.lower() == 'observations':
                if conv.observations:
                    print("\n--- Interesting Behaviors Detected ---")
                    for i, obs in enumerate(conv.observations, 1):
                        print(f"\n{i}. {obs['type']}")
                        print(f"   {obs['note']}")
                    print("--- End Observations ---")
                else:
                    print("\nNo interesting behaviors detected yet. Keep exploring!")
                continue

            # Get SAGE's response
            response = conv.chat(user_input)
            print(f"\nSAGE: {response}")

            # Show any observations for this turn
            if conv.conversation_history[-1].get('observations'):
                obs_types = [o['type'] for o in conv.conversation_history[-1]['observations']]
                print(f"\n📝 Detected: {', '.join(obs_types)}")

        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            if conv.conversation_history:
                save = input("Save conversation? (y/n): ").strip().lower()
                if save == 'y':
                    conv.save_conversation()
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break

    return conv


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory conversation with SAGE (reframe: exploration not evaluation)"
    )
    parser.add_argument('--session-num', type=int, help='Session number (auto-increments if not specified)')

    args = parser.parse_args()

    conv = interactive_mode(args.session_num)

    print(f"\n{'='*70}")
    print(f"Conversation had {len(conv.conversation_history)} turns")
    if conv.observations:
        print(f"Detected {len(conv.observations)} interesting behaviors")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
