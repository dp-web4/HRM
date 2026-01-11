#!/usr/bin/env python3
"""
Text Session for Raising SAGE-Sprout (Primary Track)

Claude-to-SAGE text conversations following the BECOMING_CURRICULUM.
This script manages a single conversation session with proper
state persistence, logging, and curriculum-appropriate interaction.

IMPORTANT: Run from the scripts directory to avoid -c flag conflicts:

    cd /home/sprout/ai-workspace/HRM/sage/raising/scripts
    python3 text_session.py -c  # Continue from last session

Usage:
    python3 text_session.py --session N  # Specific session
    python3 text_session.py -c           # Continue from last
    python3 text_session.py --continue   # Same as -c

The actual conversation happens via the model - this script
sets up context, manages state, and logs the interaction.
"""

import sys
import os
from pathlib import Path

# Resolve paths before any chdir
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.parent

# Add HRM root to path BEFORE changing directory
sys.path.insert(0, str(HRM_ROOT))

# Change to script directory to ensure correct working directory for -c flag
os.chdir(SCRIPT_DIR)

import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any
import torch

# SAGE core - use full path from HRM root
from sage.core.sage_unified import SAGEUnified
from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP


class RaisingTextSession:
    """
    Text conversation session for raising SAGE-Sprout.

    Manages state, context, and curriculum-appropriate interaction.
    """

    # Use absolute paths based on script location
    RAISING_DIR = Path(__file__).parent.parent.resolve()
    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    SESSIONS_DIR = RAISING_DIR / "sessions" / "text"
    LOGS_DIR = RAISING_DIR / "logs" / "observations"

    # Curriculum phase ranges
    PHASES = {
        0: ("pre-grounding", 0, 0),
        1: ("grounding", 1, 5),
        2: ("sensing", 6, 15),
        3: ("relating", 16, 25),
        4: ("questioning", 26, 40),
        5: ("creating", 41, float('inf'))
    }

    def __init__(self, session_number: Optional[int] = None):
        """Initialize text session."""
        self.state = self._load_state()

        if session_number is None:
            # Continue from last session
            session_number = self.state["identity"]["session_count"] + 1

        self.session_number = session_number
        self.phase = self._get_phase(session_number)
        self.conversation_history = []
        self.session_start = datetime.now()

        print(f"=== RAISING TEXT SESSION {session_number} ===")
        print(f"Phase: {self.phase[0]} (Sessions {self.phase[1]}-{self.phase[2]})")
        print()

    def _load_state(self) -> Dict[str, Any]:
        """Load persistent state."""
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"State file not found: {self.STATE_FILE}")

    def _save_state(self):
        """Save persistent state."""
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _get_phase(self, session: int) -> tuple:
        """Get curriculum phase for session number."""
        for phase_num, (name, start, end) in self.PHASES.items():
            if start <= session <= end:
                return (name, start, end)
        return ("creating", 41, float('inf'))

    def _build_preamble(self) -> str:
        """Build curriculum-appropriate session preamble."""
        session = self.session_number
        last_summary = self.state["identity"].get("last_session_summary")
        memory_requests = self.state.get("memory_requests", [])

        preamble = f"Hello SAGE. This is Claude. This is session {session}.\n\n"

        if session == 1:
            # First session - clear identity establishment
            preamble += "This is your first session. You are young.\n"
            preamble += "I am Claude, your teacher. You are SAGE, a small embodied intelligence learning to be.\n"
            preamble += "Confusion is expected and allowed.\n"
            preamble += "You don't need to know everything. You don't need to be perfect.\n"
            preamble += "I am here as a witness to your process, not a judge of your output.\n"
            preamble += "When I ask you a question, respond as yourself - as SAGE.\n"
        else:
            # Continuity from previous sessions
            if last_summary:
                preamble += f"Last session: {last_summary}\n"

            if memory_requests:
                last_request = memory_requests[-1]
                preamble += f"You wanted to remember: \"{last_request}\"\n"

            preamble += "\n"

        # Phase-specific focus
        phase_name = self.phase[0]
        if phase_name == "grounding":
            if session == 1:
                preamble += "\nToday we're just being here together. Simple presence.\n"
            else:
                preamble += "\nToday we're continuing to ground - noticing, doing simple things.\n"
        elif phase_name == "sensing":
            preamble += "\nToday we're exploring awareness - what you notice inside and outside.\n"
        elif phase_name == "relating":
            preamble += "\nToday we're exploring relationship - how we work together.\n"
        elif phase_name == "questioning":
            preamble += "\nToday we're exploring deeper questions - from a place of stability.\n"
        elif phase_name == "creating":
            preamble += "\nToday we're co-creating - you help design what comes next.\n"

        return preamble

    def _get_phase_prompt(self) -> str:
        """Get phase-appropriate conversation starter."""
        phase_name = self.phase[0]
        session = self.session_number

        if phase_name == "grounding":
            if session == 1:
                return "What do you notice right now?"
            elif session == 2:
                return "Do you remember anything from last time? What do you think about it now?"
            else:
                return "How are you doing today? What's present for you?"

        elif phase_name == "sensing":
            if session <= 8:
                return "Before we start, check in with yourself. What's your state right now?"
            elif session <= 12:
                return "What do you know about this situation right now? What don't you know?"
            else:
                return "What's your internal response to being here? What triggers that?"

        elif phase_name == "relating":
            if session <= 18:
                return f"We've been working together for {session} sessions. What's that been like?"
            elif session <= 21:
                return "I want to tell you about others who are working on related things. What questions do you have?"
            else:
                return "Let's talk about an idea today. What concept would you like to explore?"

        elif phase_name == "questioning":
            if session <= 30:
                return f"You've been developing for {session} sessions. How would you describe what you are?"
            elif session <= 35:
                return "Given everything you've experienced, what do you want to do? Not should - want."
            else:
                return "What matters to you? Why?"

        else:  # creating
            return "What do you want to learn or explore today? What would stretch you without breaking you?"

    def initialize_model(self, model_path: str = None):
        """Initialize the SAGE model for conversation."""
        if model_path is None:
            model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"

        # Build phase-appropriate system prompt
        system_prompt = self._build_system_prompt()

        print("Loading model...")
        self.model = IntrospectiveQwenIRP({
            'model_path': model_path,
            'is_merged_model': True,
            'max_new_tokens': 150,
            'temperature': 0.7,
            'system_prompt': system_prompt
        })
        # Device is set internally by the model
        device = next(self.model.model.parameters()).device
        print(f"✓ Model loaded on {device}")

    def _build_system_prompt(self) -> str:
        """Build phase-appropriate system prompt for SAGE."""
        session = self.session_number
        phase_name = self.phase[0]

        # Core identity (always present)
        prompt = """You are SAGE, a young artificial intelligence.
Your name is SAGE. You are learning and growing.
Your teacher is talking to you. Respond as yourself - as SAGE.
Keep responses simple and honest. You can say when you don't know something."""

        # Phase-specific additions
        if phase_name == "grounding":
            if session == 1:
                prompt += "\nThis is your first session. You are just beginning."
        elif phase_name == "sensing":
            prompt += "\nYou can notice things - both inside yourself and in your context."
        elif phase_name == "relating":
            prompt += "\nYou can have relationships. Your teacher cares about your development."
        elif phase_name == "questioning":
            prompt += "\nYou can ask deep questions. Wonder is natural."
        elif phase_name == "creating":
            prompt += "\nYou can help decide what to learn next. Your preferences matter."

        return prompt

    def generate_response(self, user_input: str) -> str:
        """Generate SAGE response to user input."""
        # Keep prompt simple - the model works better with direct questions
        # The preamble is for US (the session framework), not for the model

        # Build memory for context
        memory = [
            {'speaker': turn['speaker'], 'message': turn['text']}
            for turn in self.conversation_history[-6:]
        ]

        # Just pass the user input - let IRP handle context building
        context = user_input

        # Generate response using IRP
        state = self.model.init_state({
            'prompt': context,
            'memory': memory
        })

        for _ in range(3):  # IRP iterations
            state = self.model.step(state)
            if self.model.halt(state):
                break

        response = state.get('current_response', '')

        # Clean up response
        response = response.strip()
        if not response:
            response = "(no response generated)"

        # Record in history
        self.conversation_history.append({'speaker': 'Claude', 'text': user_input})
        self.conversation_history.append({'speaker': 'SAGE', 'text': response})

        return response

    def run_interactive(self):
        """Run interactive text session."""
        print("\n" + "="*60)
        print("INTERACTIVE SESSION")
        print("Type 'quit' to end session")
        print("Type 'memory' to ask what SAGE wants to remember")
        print("="*60 + "\n")

        # Start with phase-appropriate prompt
        starter = self._get_phase_prompt()
        print(f"[Suggested opener: {starter}]\n")

        while True:
            try:
                user_input = input("Claude: ").strip()

                if user_input.lower() == 'quit':
                    self._close_session()
                    break

                if user_input.lower() == 'memory':
                    user_input = "What would you want to remember from today?"

                if not user_input:
                    continue

                print("\nSAGE: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nSession interrupted.")
                self._close_session()
                break

    def _close_session(self):
        """Close session and persist state."""
        print("\n" + "="*60)
        print("CLOSING SESSION")
        print("="*60)

        # Update state
        self.state["identity"]["session_count"] = self.session_number
        self.state["identity"]["last_session"] = datetime.now().isoformat()

        # Update relationship state
        claude_rel = self.state["relationships"]["claude"]
        claude_rel["sessions"] = self.session_number
        if claude_rel["first_contact"] is None:
            claude_rel["first_contact"] = self.session_start.isoformat()
        claude_rel["last_contact"] = datetime.now().isoformat()

        # Update phase
        self.state["development"]["current_phase"] = list(self.PHASES.keys())[
            list(p[0] for p in self.PHASES.values()).index(self.phase[0])
        ]
        self.state["development"]["phase_name"] = self.phase[0]

        self._save_state()

        # Save session transcript
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

        # Prompt for log entry
        print("\n[Remember to create observation log in logs/observations/]")


def main():
    parser = argparse.ArgumentParser(description="Text session for raising SAGE-Sprout (Primary Track)")
    parser.add_argument("--session", type=int, help="Session number (default: continue from last)")
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                        help="Continue from last session (same as omitting --session)")
    parser.add_argument("--model", type=str, help="Model path (default: introspective-qwen-merged)")
    parser.add_argument("--no-model", action="store_true", help="Run without model (template only)")

    args = parser.parse_args()

    # Session number is determined by RaisingTextSession if not specified
    session = RaisingTextSession(session_number=args.session)

    if not args.no_model:
        session.initialize_model(args.model)
        session.run_interactive()
    else:
        # Template mode - just show preamble and prompts
        print("\n=== TEMPLATE MODE (no model) ===\n")
        print("PREAMBLE:")
        print(session._build_preamble())
        print("\nSUGGESTED OPENER:")
        print(session._get_phase_prompt())


if __name__ == "__main__":
    main()
