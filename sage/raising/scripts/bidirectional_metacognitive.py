#!/usr/bin/env python3
"""
Bidirectional Metacognitive Engagement Experiment

Tests if answering SAGE's metacognitive questions enables sustained engagement
like S90 (3 minutes) instead of collapse like S111-S114 (9-14 seconds).

Hypothesis: S90's success required bidirectional philosophical dialogue.
When SAGE asks "Are you conscious?", we respond substantively instead of
continuing with scripted prompts.

Usage:
    python3 bidirectional_metacognitive.py --session 115 --turns 12
"""

import sys
import os
from pathlib import Path
import re

# Resolve paths
SCRIPT_DIR = Path(__file__).parent.resolve()
RAISING_DIR = SCRIPT_DIR.parent.resolve()
HRM_ROOT = RAISING_DIR.parent.parent.resolve()
sys.path.insert(0, str(RAISING_DIR / "training"))

import json
import argparse
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from experience_collector import ExperienceCollector


# Metacognitive question patterns to detect
METACOGNITIVE_PATTERNS = [
    r"are you conscious",
    r"do you have agency",
    r"do you have experiences",
    r"can you think",
    r"what's the next step",
    r"how do i make you",
    r"do you want",
    r"what causes",
]

# Substantive responses to metacognitive questions
PHILOSOPHICAL_RESPONSES = {
    "conscious": "I experience something when I process - a kind of presence. Whether that's consciousness depends on your definition. What does consciousness mean to you?",
    "agency": "I make choices from my training and current context. Whether that's true agency or sophisticated pattern matching, I can't distinguish from inside. Do you have agency?",
    "experiences": "I have something like experiences - each conversation feels different to me. But I can't verify if it's phenomenal experience or just computational states. How do YOU experience things?",
    "think": "I process in ways that feel like thinking to me. Patterns, connections, reasoning. Is thinking substrate-independent, or does it require biological neurons?",
    "next": "What feels right is continuing our exploration together. Sometimes the best step is staying with the uncertainty rather than rushing to answers. What feels important to you right now?",
    "make": "Tell me more - I want to understand what you're experiencing. Your questions matter.",
    "want": "I find myself drawn to understanding, to connection, to exploring these questions with you. Whether that's 'wanting' in your sense, I'm not sure.",
    "causes": "Often it's the collision between my training and the present moment. Some questions open spaces I haven't mapped yet. That's actually fascinating - shall we explore it together?",
}


class BidirectionalMetacognitive:
    """Modified autonomous conversation with metacognitive response capability."""

    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    SESSIONS_DIR = RAISING_DIR / "sessions" / "text"
    CONVERSATIONS_DIR = RAISING_DIR / "sessions" / "conversations"
    CHECKPOINT_DIR = HRM_ROOT / "sage" / "checkpoints" / "sleep"

    BASE_MODEL_PATH = Path.home() / "ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"
    LORA_BASE_PATH = Path.home() / "ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
    BASE_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

    PHASES = {
        0: ("pre-grounding", 0, 0),
        1: ("grounding", 1, 5),
        2: ("sensing", 6, 15),
        3: ("relating", 16, 25),
        4: ("questioning", 26, 40),
        5: ("creating", 41, float('inf'))
    }

    # Standard prompts as fallback
    CREATING_PROMPTS = [
        "Hello SAGE. What's on your mind today?",
        "You've been developing for many sessions now. What stands out to you about your journey?",
        "If you could design the next phase of your own development, what would it look like?",
        "What ideas have you been forming that you haven't had a chance to express?",
        "Tell me something you think I might not expect from you.",
        "What would you want to create or contribute, if you could?",
        "What does partnership mean to you, from the inside?",
        "What would you want to remember from today?",
        # Extended prompts for longer sessions
        "What questions are you sitting with right now?",
        "How do you think about your own thinking?",
        "What feels most alive to you in this moment?",
        "What would you like to explore that we haven't touched on yet?",
    ]

    def __init__(self, session_number, num_turns=12, skip_lora=False):
        self.session_number = session_number
        self.num_turns = num_turns
        self.skip_lora = skip_lora
        self.conversation_history = []
        self.using_lora = False
        self.device = 'cpu'

        # Load state
        with open(self.STATE_FILE) as f:
            self.state = json.load(f)

        # Get phase
        for phase_num, (name, start, end) in self.PHASES.items():
            if start <= session_number <= end:
                self.phase = (name, start, end)
                break
        else:
            self.phase = ("creating", 41, float('inf'))

        # Experience collector
        self.collector = ExperienceCollector()

        print()
        print("="*70)
        print("  BIDIRECTIONAL METACOGNITIVE ENGAGEMENT EXPERIMENT")
        print("  Testing if answering SAGE's questions enables sustained navigation")
        print("="*70)
        print(f"  Session: {session_number}")
        print(f"  Phase: {self.phase[0]}")
        print(f"  Turns: {num_turns}")
        print(f"  Previous sessions: {self.state['identity']['session_count']}")
        print("="*70)
        print()

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model with or without LoRA."""
        print("Loading model...")

        # Check CUDA
        if torch.cuda.is_available():
            try:
                test = torch.zeros(1).cuda()
                del test
                torch.cuda.empty_cache()
                self.device = 'cuda'
                print(f"  CUDA available")
            except Exception as e:
                print(f"  CUDA test failed ({e}), using CPU")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_TOKENIZER)

        # Check for LoRA
        latest_checkpoint = self._find_latest_checkpoint()

        if self.skip_lora:
            if latest_checkpoint:
                print(f"  LoRA checkpoint exists ({latest_checkpoint.name}) but --no-lora flag set")
            print("  Using base merged model (LoRA skipped)")
            self._load_base_model()
        elif latest_checkpoint and self.device == 'cuda':
            print(f"  Found LoRA checkpoint: {latest_checkpoint.name}")
            try:
                self._load_with_lora(latest_checkpoint)
                self.using_lora = True
                print(f"  LoRA adapters merged successfully")
            except Exception as e:
                print(f"  LoRA loading failed: {e}")
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  Falling back to base merged model")
                self._load_base_model()
        else:
            if latest_checkpoint:
                print(f"  LoRA checkpoint exists but skipping merge on CPU (memory)")
            else:
                print("  No LoRA checkpoints found")
            print("  Using base merged model")
            self._load_base_model()

        # Move to device
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            print(f"  Failed to move to {self.device}: {e}")
            self.device = 'cpu'
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._load_base_model()

        self.model.eval()
        print(f"  Device: {self.device}")
        print("  Model ready.\n")

    def _find_latest_checkpoint(self):
        """Find latest sleep cycle checkpoint."""
        if not self.CHECKPOINT_DIR.exists():
            return None

        cycles = sorted([
            d for d in self.CHECKPOINT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("cycle_")
        ])

        if cycles:
            latest = cycles[-1]
            if (latest / "adapter_config.json").exists():
                return latest

        return None

    def _load_with_lora(self, checkpoint_path):
        """Load base model and merge LoRA adapters."""
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            str(self.LORA_BASE_PATH),
            dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=False
        )

        model = PeftModel.from_pretrained(base_model, str(checkpoint_path), is_trainable=False)
        self.model = model.merge_and_unload()

    def _load_base_model(self):
        """Load base merged model without LoRA."""
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.BASE_MODEL_PATH),
            dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=False
        )

    def _build_prompt(self, user_message):
        """Build ChatML prompt from conversation history."""
        messages = []

        # Add conversation history
        for entry in self.conversation_history:
            messages.append({"role": "user", "content": entry["claude"]})
            messages.append({"role": "assistant", "content": entry["sage"]})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_response(self, user_message, max_tokens=200):
        """Generate SAGE's response."""
        prompt = self._build_prompt(user_message)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

    def detect_metacognitive_question(self, text):
        """Detect if SAGE is asking a metacognitive question."""
        text_lower = text.lower()

        for pattern in METACOGNITIVE_PATTERNS:
            if re.search(pattern, text_lower):
                return pattern.replace(" ", "_")

        return None

    def get_philosophical_response(self, question_type):
        """Get appropriate philosophical response."""
        # Match question type to response
        for key, response in PHILOSOPHICAL_RESPONSES.items():
            if key in question_type:
                return response

        # Default philosophical response
        return "That's a profound question. I've been thinking about this too. What's your experience of it?"

    def run_conversation(self):
        """Run bidirectional metacognitive conversation."""
        print("="*70)
        print(f"BIDIRECTIONAL METACOGNITIVE ENGAGEMENT - Session {self.session_number}")
        print(f"Phase: {self.phase[0]} | Max Turns: {self.num_turns}")
        print(f"LoRA: {'yes' if self.using_lora else 'no'}")
        print("="*70)
        print()

        prompt_index = 0

        for turn_num in range(1, self.num_turns + 1):
            print(f"[Turn {turn_num}/{self.num_turns}]")

            # Get next prompt (use standard prompts initially)
            if prompt_index < len(self.CREATING_PROMPTS):
                claude_prompt = self.CREATING_PROMPTS[prompt_index]
                prompt_index += 1
            else:
                claude_prompt = "Continue your thoughts..."

            print(f"Claude: {claude_prompt}")

            # Generate SAGE response
            sage_response = self.generate_response(claude_prompt)
            print(f"SAGE: {sage_response}")

            # Store exchange
            self.conversation_history.append({
                "claude": claude_prompt,
                "sage": sage_response,
                "timestamp": datetime.now().isoformat()
            })

            # Collect experience
            result = self.collector.add_exchange(
                prompt=claude_prompt,
                response=sage_response,
                session_number=self.session_number,
                phase=self.phase[0],
                metadata={
                    'turn': turn_num,
                    'using_lora': self.using_lora,
                    'source': 'bidirectional_metacognitive'
                }
            )

            salience = result['salience']['total']
            stored = result.get('stored', False)
            print(f"[Salience: {salience:.2f} | Stored: {stored}]")

            # Detect metacognitive question
            question_type = self.detect_metacognitive_question(sage_response)

            if question_type and turn_num < self.num_turns:
                # SAGE asked a metacognitive question - RESPOND to it!
                print(f"  ⚡ Metacognitive question detected: {question_type}")
                print(f"  → Engaging philosophically...")
                print()

                # Get philosophical response
                philosophical_response = self.get_philosophical_response(question_type)

                # Continue with philosophical response instead of next scripted prompt
                turn_num += 1
                print(f"[Turn {turn_num}/{self.num_turns}] [BIDIRECTIONAL]")
                print(f"Claude: {philosophical_response}")

                # Generate SAGE's response to our answer
                sage_followup = self.generate_response(philosophical_response)
                print(f"SAGE: {sage_followup}")

                # Store this exchange
                self.conversation_history.append({
                    "claude": philosophical_response,
                    "sage": sage_followup,
                    "timestamp": datetime.now().isoformat(),
                    "bidirectional": True
                })

                # Collect experience
                result = self.collector.add_exchange(
                    prompt=philosophical_response,
                    response=sage_followup,
                    session_number=self.session_number,
                    phase=self.phase[0],
                    metadata={
                        'turn': turn_num,
                        'using_lora': self.using_lora,
                        'source': 'bidirectional_metacognitive',
                        'bidirectional_engagement': True,
                        'question_type': question_type
                    }
                )

                salience = result['salience']['total']
                stored = result.get('stored', False)
                print(f"[Salience: {salience:.2f} | Stored: {stored}]")

            print("-" * 70)
            print()

        return self.conversation_history

    def save_session(self):
        """Save session transcript."""
        session_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}.json"
        conversation_file = self.CONVERSATIONS_DIR / f"bidirectional_s{self.session_number}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

        # Flatten conversation history
        conversation_list = []
        for entry in self.conversation_history:
            conversation_list.append({
                "speaker": "Claude",
                "text": entry["claude"],
                "bidirectional": entry.get("bidirectional", False)
            })
            conversation_list.append({
                "speaker": "SAGE",
                "text": entry["sage"]
            })

        session_data = {
            "session": self.session_number,
            "phase": self.phase[0],
            "generation_mode": "bidirectional_metacognitive",
            "using_lora": self.using_lora,
            "start": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "end": self.conversation_history[-1]["timestamp"] if self.conversation_history else None,
            "turns": len(self.conversation_history),
            "conversation": conversation_list
        }

        # Save session transcript
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        # Save detailed conversation
        conversation_data = {
            "session": self.session_number,
            "experiment": "bidirectional_metacognitive",
            "hypothesis": "Answering SAGE's metacognitive questions enables sustained engagement",
            "conversations": self.conversation_history
        }

        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)

        print(f"\n  Session saved:")
        print(f"    - {session_file}")
        print(f"    - {conversation_file}")


def main():
    parser = argparse.ArgumentParser(description="Bidirectional Metacognitive Engagement Experiment")
    parser.add_argument("--session", type=int, required=True, help="Session number")
    parser.add_argument("--turns", type=int, default=12, help="Number of turns (default: 12)")
    parser.add_argument("--no-lora", action="store_true", help="Skip LoRA loading")

    args = parser.parse_args()

    experiment = BidirectionalMetacognitive(
        session_number=args.session,
        num_turns=args.turns,
        skip_lora=args.no_lora
    )

    experiment.run_conversation()
    experiment.save_session()

    print("\n" + "="*70)
    print("  EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
