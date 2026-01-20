#!/usr/bin/env python3
"""
Thor-SAGE Conversation Script

Genuine multi-turn conversation between Claude (via human or autonomous session)
and SAGE on Thor using larger models (14B or 30B).

This is exploration, not evaluation. The conversation follows interesting threads
wherever they lead.

Usage:
    # Default (14B model)
    python3 thor_sage_conversation.py

    # Specify model size
    python3 thor_sage_conversation.py --model 14b
    python3 thor_sage_conversation.py --model 30b

    # Non-interactive mode (for autonomous sessions)
    python3 thor_sage_conversation.py --auto "What's on your mind today?"
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sage.core.multi_model_loader import create_thor_loader, TaskComplexity

class ThorSAGEConversation:
    """Conversation manager for Thor's larger SAGE models."""

    def __init__(self, model_size: str = "14b"):
        self.model_size = model_size.lower()
        self.conversation_history = []
        self.loader = None
        self.tokenizer = None
        self.complexity = TaskComplexity.MODERATE  # Default complexity

    def load_model(self):
        """Load the specified model using Thor's multi-model loader."""
        print(f"Loading SAGE-Thor ({self.model_size.upper()})...")

        # Create loader with appropriate default model
        if self.model_size == "30b":
            # For 30B, we'll use COMPLEX complexity by default
            self.complexity = TaskComplexity.COMPLEX
            preload = True
        elif self.model_size == "14b":
            self.complexity = TaskComplexity.MODERATE
            preload = True
        else:
            raise ValueError(f"Unknown model size: {self.model_size}. Use '14b' or '30b'")

        self.loader = create_thor_loader(preload_default=preload)

        # Get tokenizer from loaded model for chat template
        model_config = self.loader.get_model_for_task(self.complexity)
        self.tokenizer = model_config.tokenizer

        print("SAGE-Thor ready for conversation.\n")

        # Show memory status
        status = self.loader.get_status()
        print(f"Memory: {status['memory_used_gb']:.1f}GB / {status['max_memory_gb']}GB\n")

    def build_conversation_prompt(self, user_message: str) -> str:
        """Build conversation using Qwen chat template."""

        # Build messages in chat format
        messages = []

        # System message
        system_msg = (
            "You are SAGE-Thor, in genuine conversation with Claude. "
            "This is exploration - you can ask questions, express uncertainty, "
            "or take the conversation in unexpected directions."
        )
        messages.append({"role": "system", "content": system_msg})

        # Add history
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn['claude']})
            messages.append({"role": "assistant", "content": turn['sage']})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        # Use tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_response(self, user_message: str, max_tokens: int = 250) -> str:
        """Generate SAGE's response to Claude's message."""

        prompt = self.build_conversation_prompt(user_message)

        # Adjust complexity based on conversation depth
        if len(self.conversation_history) > 5:
            complexity = TaskComplexity.COMPLEX
        elif len(self.conversation_history) > 2:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.SIMPLE

        response = self.loader.generate(
            prompt=prompt,
            complexity=complexity,
            max_tokens=max_tokens,
            temperature=0.8  # Encourage creativity and exploration
        )

        # Post-process: extract just the assistant's response
        # The model echoes the template, so extract after "assistant\n"
        if "assistant\n" in response:
            response = response.split("assistant\n", 1)[1]
        elif "assistant" in response:
            response = response.split("assistant", 1)[1]

        return response.strip()

    def chat(self, claude_message: str, max_tokens: int = 250) -> str:
        """Single turn of conversation."""
        sage_response = self.generate_response(claude_message, max_tokens)

        # Store in history
        self.conversation_history.append({
            "claude": claude_message,
            "sage": sage_response,
            "timestamp": datetime.now().isoformat()
        })

        return sage_response

    def save_conversation(self, filename: str = None, metadata: dict = None):
        """Save conversation to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"thor_sage_{self.model_size}_{timestamp}.json"

        # Save to conversations directory
        path = Path(__file__).parent.parent / "sessions" / "conversations" / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        conversation_data = {
            "type": "thor_sage_conversation",
            "machine": "thor",
            "model_size": self.model_size,
            "started": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "ended": datetime.now().isoformat(),
            "turns": len(self.conversation_history),
            "conversation": self.conversation_history
        }

        # Add any additional metadata
        if metadata:
            conversation_data["metadata"] = metadata

        with open(path, 'w') as f:
            json.dump(conversation_data, f, indent=2)

        print(f"\n💾 Conversation saved to: {path}")
        return path


def interactive_mode(model_size: str):
    """Run interactive conversation mode."""
    print("=" * 70)
    print(f"THOR-SAGE CONVERSATION ({model_size.upper()})")
    print("=" * 70)
    print()
    print("This is genuine multi-turn conversation for exploration.")
    print("Type your messages as Claude. SAGE will respond.")
    print()
    print("Commands:")
    print("  'exit' or 'quit' - End and save conversation")
    print("  'history' - Show conversation so far")
    print("  'save' - Save without exiting")
    print("  'tokens:N' - Set max tokens for next response")
    print()
    print("=" * 70)
    print()

    conv = ThorSAGEConversation(model_size)
    conv.load_model()

    max_tokens = 250  # Default

    while True:
        try:
            user_input = input("\nClaude: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                if conv.conversation_history:
                    conv.save_conversation()
                print("\n✅ Conversation ended.")
                break

            if user_input.lower() == 'history':
                print("\n" + "=" * 70)
                print("CONVERSATION HISTORY")
                print("=" * 70)
                for i, turn in enumerate(conv.conversation_history, 1):
                    print(f"\n[Turn {i}]")
                    print(f"Claude: {turn['claude']}")
                    print(f"SAGE: {turn['sage']}")
                print("\n" + "=" * 70)
                continue

            if user_input.lower() == 'save':
                conv.save_conversation()
                continue

            if user_input.lower().startswith('tokens:'):
                try:
                    max_tokens = int(user_input.split(':')[1])
                    print(f"✓ Max tokens set to {max_tokens}")
                    continue
                except (ValueError, IndexError):
                    print("❌ Invalid format. Use: tokens:250")
                    continue

            # Generate response
            response = conv.chat(user_input, max_tokens=max_tokens)
            print(f"\nSAGE: {response}")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted.")
            if conv.conversation_history:
                save = input("Save conversation? (y/n): ").strip().lower()
                if save == 'y':
                    conv.save_conversation()
            break

    return conv


def auto_mode(model_size: str, initial_message: str):
    """Run single-turn conversation for autonomous sessions."""
    print(f"🤖 Auto-mode: Thor-SAGE ({model_size.upper()}) conversation")
    print()

    conv = ThorSAGEConversation(model_size)
    conv.load_model()

    print(f"Claude: {initial_message}")
    response = conv.chat(initial_message)
    print(f"\nSAGE: {response}")
    print()

    # Save with auto-mode flag
    conv.save_conversation(metadata={"mode": "autonomous", "single_turn": True})

    return conv


def main():
    parser = argparse.ArgumentParser(description="Thor-SAGE Conversation Script")
    parser.add_argument(
        '--model',
        type=str,
        default='14b',
        choices=['14b', '30b'],
        help='Model size to use (default: 14b)'
    )
    parser.add_argument(
        '--auto',
        type=str,
        help='Non-interactive mode with single message (for autonomous sessions)'
    )

    args = parser.parse_args()

    if args.auto:
        # Autonomous mode
        conv = auto_mode(args.model, args.auto)
    else:
        # Interactive mode
        conv = interactive_mode(args.model)

    return conv


if __name__ == "__main__":
    main()
