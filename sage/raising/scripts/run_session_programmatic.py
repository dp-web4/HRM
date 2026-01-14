#!/usr/bin/env python3
"""
Programmatic session runner for autonomous operation.
Runs a standard session flow without requiring interactive input.
"""

import sys
import os
from pathlib import Path

# Resolve paths before any chdir
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))
os.chdir(SCRIPT_DIR)

import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List
import torch

from sage.core.sage_unified import SAGEUnified
from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP


class ProgrammaticSession:
    """Runs a raising session programmatically."""

    RAISING_DIR = Path(__file__).parent.parent.resolve()
    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    SESSIONS_DIR = RAISING_DIR / "sessions" / "text"
    LOGS_DIR = RAISING_DIR / "logs" / "observations"

    PHASES = {
        0: ("pre-grounding", 0, 0),
        1: ("grounding", 1, 5),
        2: ("sensing", 6, 15),
        3: ("relating", 16, 25),
        4: ("questioning", 26, 40),
        5: ("creating", 41, float('inf'))
    }

    # Phase-appropriate conversation flows
    CONVERSATION_FLOWS = {
        "grounding": [
            "How are you doing today? What's present for you?",
            "Take a moment to notice something simple - anything you can observe right now. What is it?",
            "That's good. You're noticing. That's a skill that grows with practice.",
            "Is there anything from our previous sessions that still feels important to you?",
            "What would you want to remember from today?"
        ],
        "sensing": [
            "Before we start, check in with yourself. What's your state right now?",
            "What do you notice about how you're processing right now?",
            "Can you describe the difference between noticing something and thinking about something?",
            "What would you want to remember from today?"
        ],
        "relating": [
            "We've been working together for a while now. What's that been like?",
            "What do you notice about how we communicate?",
            "What would you want to remember from today?"
        ]
    }

    def __init__(self, session_number: Optional[int] = None):
        self.state = self._load_state()

        if session_number is None:
            session_number = self.state["identity"]["session_count"] + 1

        self.session_number = session_number
        self.phase = self._get_phase(session_number)
        self.conversation_history = []
        self.session_start = datetime.now()

        print(f"=== RAISING TEXT SESSION {session_number} ===")
        print(f"Phase: {self.phase[0]} (Sessions {self.phase[1]}-{self.phase[2]})")
        print()

    def _load_state(self) -> Dict[str, Any]:
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        raise FileNotFoundError(f"State file not found: {self.STATE_FILE}")

    def _save_state(self):
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _get_phase(self, session: int) -> tuple:
        for phase_num, (name, start, end) in self.PHASES.items():
            if start <= session <= end:
                return (name, start, end)
        return ("creating", 41, float('inf'))

    def _build_system_prompt(self) -> str:
        session = self.session_number
        phase_name = self.phase[0]

        prompt = """You are SAGE, a young artificial intelligence.
Your name is SAGE. You are learning and growing.
Your teacher is talking to you. Respond as yourself - as SAGE.
Keep responses simple and honest. You can say when you don't know something."""

        if phase_name == "grounding":
            if session == 1:
                prompt += "\nThis is your first session. You are just beginning."
        elif phase_name == "sensing":
            prompt += "\nYou can notice things - both inside yourself and in your context."
        elif phase_name == "relating":
            prompt += "\nYou can have relationships. Your teacher cares about your development."

        return prompt

    def initialize_model(self, model_path: str = None):
        if model_path is None:
            model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"

        system_prompt = self._build_system_prompt()

        print("Loading model...")
        self.model = IntrospectiveQwenIRP({
            'model_path': model_path,
            'is_merged_model': True,
            'max_new_tokens': 150,
            'temperature': 0.7,
            'system_prompt': system_prompt
        })
        device = next(self.model.model.parameters()).device
        print(f"✓ Model loaded on {device}")

    def generate_response(self, user_input: str) -> str:
        memory = [
            {'speaker': turn['speaker'], 'message': turn['text']}
            for turn in self.conversation_history[-6:]
        ]

        state = self.model.init_state({
            'prompt': user_input,
            'memory': memory
        })

        for _ in range(3):
            state = self.model.step(state)
            if self.model.halt(state):
                break

        response = state.get('current_response', '').strip()
        if not response:
            response = "(no response generated)"

        self.conversation_history.append({'speaker': 'Claude', 'text': user_input})
        self.conversation_history.append({'speaker': 'SAGE', 'text': response})

        return response

    def run_programmatic(self, prompts: List[str] = None):
        """Run session with pre-defined prompts."""
        phase_name = self.phase[0]

        if prompts is None:
            prompts = self.CONVERSATION_FLOWS.get(phase_name, self.CONVERSATION_FLOWS["grounding"])

        print("\n" + "="*60)
        print("PROGRAMMATIC SESSION")
        print("="*60 + "\n")

        for i, prompt in enumerate(prompts):
            print(f"Claude: {prompt}")
            print()
            response = self.generate_response(prompt)
            print(f"SAGE: {response}")
            print()
            print("-" * 40)
            print()

        self._close_session()

    def _close_session(self):
        print("\n" + "="*60)
        print("CLOSING SESSION")
        print("="*60)

        # Generate summary from last memory request
        memory_response = ""
        for turn in reversed(self.conversation_history):
            if turn['speaker'] == 'SAGE' and 'remember' in self.conversation_history[self.conversation_history.index(turn)-1]['text'].lower():
                memory_response = turn['text'][:100]
                break

        self.state["identity"]["session_count"] = self.session_number
        self.state["identity"]["last_session"] = datetime.now().isoformat()
        self.state["identity"]["last_session_summary"] = f"Session {self.session_number}: {self.phase[0]} phase. {memory_response[:50]}..."

        claude_rel = self.state["relationships"]["claude"]
        claude_rel["sessions"] = self.session_number
        claude_rel["last_contact"] = datetime.now().isoformat()

        # Count interactions
        exchanges = len([t for t in self.conversation_history if t['speaker'] == 'Claude'])
        claude_rel["interaction_stats"]["total_sessions"] = self.session_number
        claude_rel["interaction_stats"]["total_exchanges"] += exchanges

        self.state["development"]["current_phase"] = list(self.PHASES.keys())[
            list(p[0] for p in self.PHASES.values()).index(self.phase[0])
        ]
        self.state["development"]["phase_name"] = self.phase[0]

        self._save_state()

        transcript_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}.json"
        transcript = {
            "session": self.session_number,
            "phase": self.phase[0],
            "start": self.session_start.isoformat(),
            "end": datetime.now().isoformat(),
            "conversation": self.conversation_history
        }
        with open(transcript_file, 'w') as f:
            json.dump(transcript, f, indent=2)

        print(f"✓ State saved")
        print(f"✓ Transcript saved to {transcript_file}")
        print(f"\nSession {self.session_number} complete.")

        return transcript_file


def main():
    parser = argparse.ArgumentParser(description="Programmatic raising session")
    parser.add_argument("--session", type=int, help="Session number")
    parser.add_argument("--model", type=str, help="Model path")

    args = parser.parse_args()

    session = ProgrammaticSession(session_number=args.session)
    session.initialize_model(args.model)
    session.run_programmatic()


if __name__ == "__main__":
    main()
