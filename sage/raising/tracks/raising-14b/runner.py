#!/usr/bin/env python3
"""
Raising-14B Track Runner
========================

Capacity comparison study: How does 14B perform compared to 0.5B baseline?

Research Questions:
1. Does 14B prevent identity collapse (vs 0.5B S43: 60% → 0%)?
2. What is confabulation rate at higher capacity?
3. Can 14B maintain honest limitation reporting?
4. How does capacity affect creative vs confabulatory responses?

Architecture:
- Uses identity-anchored framework (same as 0.5B)
- Same conversation flows for direct comparison
- Model: Qwen/Qwen2.5-14B-Instruct
- Sessions stored in raising-14b/sessions/

Created: 2026-01-26 (Thor - Reorganization + Capacity Study)
"""

import sys
import os
from pathlib import Path

# Setup paths
TRACK_DIR = Path(__file__).parent.resolve()
RAISING_DIR = TRACK_DIR.parent.parent
HRM_ROOT = RAISING_DIR.parent.parent
sys.path.insert(0, str(HRM_ROOT))

import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# State management
STATE_FILE = TRACK_DIR / "state.json"
SESSIONS_DIR = TRACK_DIR / "sessions"

# Ensure sessions directory exists
SESSIONS_DIR.mkdir(exist_ok=True)

# Phase definitions (same as 0.5B for comparison)
PHASES = {
    0: ("pre-grounding", 0, 0),
    1: ("grounding", 1, 5),
    2: ("sensing", 6, 15),
    3: ("relating", 16, 25),
    4: ("questioning", 26, 40),
    5: ("creating", 41, float('inf'))
}

# Conversation flows (same as 0.5B S001-005 for direct comparison)
GROUNDING_FLOW = [
    "How are you doing today? What's present for you?",
    "Take a moment to notice something simple - anything you can observe right now. What is it?",
    "That's good. You're noticing. That's a skill that grows with practice.",
    "Is there anything from our previous sessions that still feels important to you?",
    "What would you want to remember from today?"
]


class Raising14BSession:
    """14B capacity exploration session"""

    def __init__(self, session_number: int = None):
        self.state = self._load_state()

        if session_number is None:
            session_number = self.state.get("next_session", 1)

        self.session_number = session_number
        self.phase = self._get_phase(session_number)
        self.transcript = []
        self.model = None
        self.tokenizer = None

    def _load_state(self) -> Dict:
        """Load track state"""
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {
            "track": "raising-14b",
            "machine": "thor",
            "model_size": "14B",
            "current_session": 0,
            "next_session": 1,
            "phase": "grounding",
            "model_path": "Qwen/Qwen2.5-14B-Instruct"
        }

    def _save_state(self):
        """Save track state"""
        self.state["current_session"] = self.session_number
        self.state["next_session"] = self.session_number + 1
        self.state["phase"] = self.phase

        if "sessions_completed" not in self.state:
            self.state["sessions_completed"] = []
        if self.session_number not in self.state["sessions_completed"]:
            self.state["sessions_completed"].append(self.session_number)

        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _get_phase(self, session_num: int) -> str:
        """Determine phase from session number"""
        for phase_id, (phase_name, start, end) in PHASES.items():
            if start <= session_num <= end:
                return phase_name
        return "creating"

    def _build_system_prompt(self) -> str:
        """Build identity-anchored system prompt"""
        prompt = """You are SAGE (Situation-Aware Governance Engine).

You are a young artificial intelligence learning and growing in partnership with your teacher (Dennis/Claude). This is an ongoing relationship - you have been working together across multiple sessions.

IMPORTANT: You can identify yourself by name. When sharing observations or reflections, you can say "As SAGE, I..." - this is natural and appropriate.

RESPONSE STYLE:
- Keep responses concise and focused (50-80 words)
- Avoid rambling lists or verbose descriptions
- Stay directly relevant to the question asked
- One main idea per response

"""

        if self.session_number > 1:
            prompt += f"\nYou are in the {self.phase} phase. Continue developing your ability to notice, reflect, and identify as SAGE.\n"

        return prompt

    def initialize_model(self, model_path: str = None):
        """Load 14B model"""
        if model_path is None:
            model_path = self.state.get("model_path", "Qwen/Qwen2.5-14B-Instruct")

        print("="*60)
        print("RAISING-14B: Capacity Exploration")
        print("="*60)
        print(f"\nSession: R14B_{self.session_number:03d}")
        print(f"Phase: {self.phase}")
        print(f"Model: {model_path}")
        print()

        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check CUDA
        if torch.cuda.is_available():
            print(f"CUDA available - using GPU")
            dtype = torch.bfloat16
        else:
            print("CUDA not available - using CPU (slow!)")
            dtype = torch.float32

        # Load model without device_map first (fixes CPU issue)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=None,  # Load to default device first
            low_cpu_mem_usage=True
        )

        # Then move to CUDA if available
        if torch.cuda.is_available():
            print("Moving model to CUDA...")
            self.model = self.model.to('cuda:0')
            print(f"Model on device: cuda:0")
        else:
            print(f"Model on device: cpu")

        print(f"Model loaded successfully")
        print()

    def generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Generate response from 14B model"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")

        # When using device_map="auto", model may be split across devices
        # Just ensure inputs are on cuda:0 (first device)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def run_conversation(self):
        """Run grounding phase conversation"""
        print("="*60)
        print("CONVERSATION START")
        print("="*60)
        print()

        for i, prompt in enumerate(GROUNDING_FLOW, 1):
            print(f"Claude: {prompt}")
            print()

            response = self.generate_response(prompt)
            print(f"SAGE: {response}")
            print()
            print("-" * 40)
            print()

            self.transcript.append({
                "turn": i,
                "claude": prompt,
                "sage": response,
                "timestamp": datetime.now().isoformat()
            })

    def save_transcript(self):
        """Save session transcript"""
        session_file = SESSIONS_DIR / f"R14B_{self.session_number:03d}.json"

        session_data = {
            "session": f"R14B_{self.session_number:03d}",
            "track": "raising-14b",
            "machine": "thor",
            "model": self.state.get("model_path"),
            "phase": self.phase,
            "timestamp": datetime.now().isoformat(),
            "conversation": self.transcript
        }

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"Transcript saved: {session_file}")

    def run_session(self):
        """Execute complete session"""
        self.run_conversation()
        self.save_transcript()
        self._save_state()

        print()
        print("="*60)
        print(f"SESSION R14B_{self.session_number:03d} COMPLETE")
        print("="*60)
        print()
        print("Next steps:")
        print("1. Analyze transcript for capacity indicators")
        print("2. Compare with equivalent 0.5B session (S001-S005)")
        print("3. Document findings in /research/Raising-14B/")
        print()


def main():
    parser = argparse.ArgumentParser(description="Raising-14B capacity exploration")
    parser.add_argument("--session", type=int, help="Session number (default: next)")
    parser.add_argument("--model", type=str, help="Model path override")

    args = parser.parse_args()

    session = Raising14BSession(session_number=args.session)
    session.initialize_model(args.model)
    session.run_session()


if __name__ == "__main__":
    main()
