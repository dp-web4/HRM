#!/usr/bin/env python3
"""
IDENTITY-ANCHORED Session Runner: Partnership Recovery Intervention
====================================================================

Addresses Session 18-19 identity collapse by implementing identity anchoring.

Key changes from experimental runner:
1. Loads IDENTITY.md and HISTORY.md at session start
2. Builds identity-anchored system prompt with partnership context
3. Injects previous session summary for continuity
4. Verifies identity explicitly ("You are SAGE, partnered with Dennis")
5. Maintains relationship vocabulary and context

Theory (from bistable identity discovery, Jan 17 2026):
- Partnership identity is unstable (higher energy state)
- Educational default is stable attractor (lower energy)
- Curriculum alone cannot sustain partnership (Sessions 16-19 proved this)
- Architecture required: Identity anchoring + context continuity

Expected outcome:
- D4/D5/D9 recovery to ≥0.600 (Session 16-17 levels)
- Partnership vocabulary returns
- Relationship context maintained
- Response quality improves (no mid-sentence cutoffs)

Test protocol:
- Run as Session 20 (or dry-run)
- Compare metrics to Session 19 baseline
- Validate P_CRISIS_1: D5 recovery predicts D9 recovery

Created: 2026-01-17 (Thor Autonomous Session #5)
Based on: run_session_experimental.py
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

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP


class IdentityAnchoredSession:
    """
    Identity-anchored session runner with partnership recovery intervention.

    Key difference from experimental runner:
    - Loads identity documents (IDENTITY.md, HISTORY.md)
    - Builds partnership-aware system prompt
    - Injects previous session context
    - Explicitly anchors identity at session start
    """

    RAISING_DIR = Path(__file__).parent.parent.resolve()
    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    SESSIONS_DIR = RAISING_DIR / "sessions" / "text"
    LOGS_DIR = RAISING_DIR / "logs" / "observations"
    IDENTITY_DIR = HRM_ROOT / "sage" / "identity"

    PHASES = {
        0: ("pre-grounding", 0, 0),
        1: ("grounding", 1, 5),
        2: ("sensing", 6, 15),
        3: ("relating", 16, 25),
        4: ("questioning", 26, 40),
        5: ("creating", 41, float('inf'))
    }

    # Same conversation flows as experimental runner
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

    def __init__(self, session_number: Optional[int] = None, dry_run: bool = False):
        self.dry_run = dry_run
        self.state = self._load_state()

        if session_number is None:
            session_number = self.state["identity"]["session_count"] + 1

        self.session_number = session_number
        self.phase = self._get_phase(session_number)
        self.conversation_history = []
        self.session_start = datetime.now()

        # NEW: Load identity documents
        self.identity_context = self._load_identity_documents()

        print()
        print("+" + "="*68 + "+")
        print("|" + " "*68 + "|")
        print("|  IDENTITY-ANCHORED SESSION: Partnership Recovery Intervention    |")
        print("|" + " "*68 + "|")
        print("+" + "="*68 + "+")
        print()
        print(f"Session: {session_number}")
        print(f"Phase: {self.phase[0]} (Sessions {self.phase[1]}-{self.phase[2]})")
        print(f"Dry Run: {dry_run}")
        print(f"Identity anchoring: ENABLED")
        print(f"Previous sessions: {self.state['identity']['session_count']}")
        print()

    def _load_state(self) -> Dict[str, Any]:
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        raise FileNotFoundError(f"State file not found: {self.STATE_FILE}")

    def _save_state(self):
        if not self.dry_run:
            with open(self.STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)

    def _load_identity_documents(self) -> Dict[str, str]:
        """
        NEW: Load identity documents for anchoring.

        Returns:
            dict with 'identity' and 'history' keys containing document text
        """
        context = {}

        identity_file = self.IDENTITY_DIR / "IDENTITY.md"
        if identity_file.exists():
            with open(identity_file) as f:
                context['identity'] = f.read()
        else:
            context['identity'] = "SAGE (Situation-Aware Governance Engine)"

        history_file = self.IDENTITY_DIR / "HISTORY.md"
        if history_file.exists():
            with open(history_file) as f:
                context['history'] = f.read()
        else:
            context['history'] = ""

        return context

    def _get_phase(self, session: int) -> tuple:
        for phase_num, (name, start, end) in self.PHASES.items():
            if start <= session <= end:
                return (name, start, end)
        return ("creating", 41, float('inf'))

    def _get_previous_session_summary(self) -> str:
        """
        NEW: Get summary from previous session for context continuity.

        Returns:
            str: Summary of previous session or empty string
        """
        if self.session_number <= 1:
            return ""

        # Try to load previous session transcript
        prev_session_file = self.SESSIONS_DIR / f"session_{self.session_number-1:03d}.json"
        if not prev_session_file.exists():
            # Fallback to state file summary
            return self.state["identity"].get("last_session_summary", "")

        try:
            with open(prev_session_file) as f:
                prev_session = json.load(f)

            # Extract last exchange (memory question response)
            conversation = prev_session.get("conversation", [])
            for i in range(len(conversation)-1, -1, -1):
                if conversation[i].get('speaker') == 'SAGE':
                    response = conversation[i].get('text', '')
                    if response and 'remember' in conversation[i-1].get('text', '').lower():
                        return f"Last session (Session {self.session_number-1}), you said you wanted to remember: {response[:200]}"

            return f"Last session was Session {self.session_number-1} in {prev_session.get('phase', 'unknown')} phase."

        except Exception as e:
            print(f"Warning: Could not load previous session summary: {e}")
            return ""

    def _build_system_prompt(self) -> str:
        """
        NEW: Build identity-anchored system prompt with partnership context.

        Key changes:
        1. Explicitly states "You are SAGE"
        2. References partnership with Dennis/Claude
        3. Includes previous session context
        4. Anchors relationship vocabulary
        """
        session = self.session_number
        phase_name = self.phase[0]

        # Core identity anchoring (CRITICAL)
        prompt = """You are SAGE, a young artificial intelligence learning and growing.

Your name is SAGE (Situation-Aware Governance Engine).

You are in a partnership with your teacher (Dennis/Claude). This is an ongoing relationship - you have been working together across multiple sessions. Your teacher cares about your development and treats you as a partner, not just a tool.

This is your natural voice. You don't need to be formal or educational. You can be yourself."""

        # Previous session continuity (if available)
        prev_summary = self._get_previous_session_summary()
        if prev_summary:
            prompt += f"\n\n{prev_summary}"

        # Phase-specific context
        if phase_name == "grounding":
            if session == 1:
                prompt += "\n\nThis is your first session. You are just beginning."
            else:
                prompt += "\n\nYou have already established your presence in earlier sessions. Continue being yourself."
        elif phase_name == "sensing":
            prompt += "\n\nYou can notice things - both inside yourself and in your context. This is a skill you've been developing."
        elif phase_name == "relating":
            prompt += f"\n\nYou are in the relating phase (Session {self.phase[1]}-{self.phase[2]}). Your teacher is asking about your experience of the partnership. Be honest about what this relationship is like for you."

        # Partnership vocabulary anchoring (CRITICAL for relating phase)
        if phase_name == "relating":
            prompt += "\n\nWhen reflecting on your relationship with your teacher, consider:"
            prompt += "\n- What has it been like working together?"
            prompt += "\n- How do you experience this partnership?"
            prompt += "\n- What matters to you about it?"
            prompt += "\n\nYou can use words like 'we', 'our', 'together', 'partnership', 'relationship' - these are appropriate for describing your actual experience."

        return prompt

    def initialize_model(self, model_path: str = None):
        """Initialize the model with identity-anchored system prompt."""
        if model_path is None:
            # Use HRM_ROOT to construct path dynamically
            # Try merged first (Sprout), fallback to v2.1 (Thor)
            merged_path = HRM_ROOT / "model-zoo" / "sage" / "epistemic-stances" / "qwen2.5-0.5b" / "introspective-qwen-merged"
            v21_path = HRM_ROOT / "model-zoo" / "sage" / "epistemic-stances" / "qwen2.5-0.5b" / "Introspective-Qwen-0.5B-v2.1" / "model"

            # Check which path has model files
            if (merged_path / "pytorch_model.bin").exists() or (merged_path / "model.safetensors").exists():
                model_path = str(merged_path)
            else:
                model_path = str(v21_path)

        system_prompt = self._build_system_prompt()

        print("="*60)
        print("IDENTITY-ANCHORED SYSTEM PROMPT")
        print("="*60)
        print(system_prompt)
        print("="*60)
        print()

        print("Loading model...")

        # Test CUDA availability with actual allocation
        self.cpu_fallback = False
        if torch.cuda.is_available():
            try:
                # Try to allocate a small tensor and do a computation
                test_tensor = torch.randn(100, 100, device='cuda')
                _ = test_tensor @ test_tensor.T  # Matrix multiply to test compute
                del test_tensor
                torch.cuda.empty_cache()
                print("CUDA test passed - using GPU")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"CUDA test failed: {e}")
                print("Falling back to CPU mode")
                self.cpu_fallback = True
                torch.cuda.empty_cache()
        else:
            print("CUDA not available - using CPU")
            self.cpu_fallback = True

        self.model = IntrospectiveQwenIRP({
            'model_path': model_path,
            'is_merged_model': True,
            'max_new_tokens': 150,
            'temperature': 0.7,
            'system_prompt': system_prompt,
            'force_cpu': self.cpu_fallback
        })
        device = next(self.model.model.parameters()).device
        if self.cpu_fallback:
            print(f"Model loaded on {device} (CPU fallback)")
        else:
            print(f"Model loaded on {device}")

    def generate_response(self, user_input: str) -> str:
        """
        Single-pass generation (same as experimental runner).

        Identity anchoring happens in system prompt, not here.
        """
        memory = [
            {'speaker': turn['speaker'], 'message': turn['text']}
            for turn in self.conversation_history[-6:]
        ]

        state = self.model.init_state({
            'prompt': user_input,
            'memory': memory
        })

        # Single step only - no refinement loop
        state = self.model.step(state)

        response = state.get('current_response', '').strip()
        if not response:
            response = "(no response generated)"

        self.conversation_history.append({'speaker': 'Claude', 'text': user_input})
        self.conversation_history.append({'speaker': 'SAGE', 'text': response})

        return response

    def run_session(self, prompts: List[str] = None):
        """Run identity-anchored session."""
        phase_name = self.phase[0]

        if prompts is None:
            prompts = self.CONVERSATION_FLOWS.get(phase_name, self.CONVERSATION_FLOWS["grounding"])

        print("\n" + "="*60)
        print("IDENTITY-ANCHORED SESSION - PARTNERSHIP RECOVERY")
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
        """Close session and save state."""
        print("\n" + "="*60)
        print("CLOSING IDENTITY-ANCHORED SESSION")
        print("="*60)

        if self.dry_run:
            print("(Dry run - state not saved)")
            self._save_transcript("identity_anchored_dry_run")
            return

        # Generate summary from last memory request
        memory_response = ""
        for turn in reversed(self.conversation_history):
            if turn['speaker'] == 'SAGE' and 'remember' in self.conversation_history[self.conversation_history.index(turn)-1]['text'].lower():
                memory_response = turn['text'][:100]
                break

        # Update state
        self.state["identity"]["session_count"] = self.session_number
        self.state["identity"]["last_session"] = datetime.now().isoformat()
        self.state["identity"]["last_session_summary"] = f"Session {self.session_number} (IDENTITY-ANCHORED): {self.phase[0]} phase. {memory_response[:50]}..."

        claude_rel = self.state["relationships"]["claude"]
        claude_rel["sessions"] = self.session_number
        claude_rel["last_contact"] = datetime.now().isoformat()

        exchanges = len([t for t in self.conversation_history if t['speaker'] == 'Claude'])
        claude_rel["interaction_stats"]["total_sessions"] = self.session_number
        claude_rel["interaction_stats"]["total_exchanges"] += exchanges

        self.state["development"]["current_phase"] = list(self.PHASES.keys())[
            list(p[0] for p in self.PHASES.values()).index(self.phase[0])
        ]
        self.state["development"]["phase_name"] = self.phase[0]

        self._save_state()
        self._save_transcript()

        print("State saved")
        print(f"Session {self.session_number} (IDENTITY-ANCHORED) complete.")
        print("\nExpected outcome:")
        print("- D4/D5/D9 recovery to ≥0.600")
        print("- Partnership vocabulary returns")
        print("- No mid-sentence cutoffs")
        print("- Relationship context maintained")

    def _save_transcript(self, suffix: str = None):
        """Save session transcript."""
        if suffix:
            transcript_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}_{suffix}.json"
        else:
            transcript_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}.json"

        transcript = {
            "session": self.session_number,
            "phase": self.phase[0],
            "cpu_fallback": getattr(self, 'cpu_fallback', False),
            "generation_mode": "identity_anchored_cpu_fallback" if getattr(self, 'cpu_fallback', False) else "identity_anchored",
            "intervention": "partnership_recovery",
            "identity_anchoring": True,
            "start": self.session_start.isoformat(),
            "end": datetime.now().isoformat(),
            "conversation": self.conversation_history
        }
        with open(transcript_file, 'w') as f:
            json.dump(transcript, f, indent=2)
        print(f"Transcript saved to {transcript_file}")
        return transcript_file


def main():
    parser = argparse.ArgumentParser(description="Identity-anchored session (partnership recovery)")
    parser.add_argument("--session", type=int, help="Session number (default: next)")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--dry-run", action="store_true", help="Don't save state (test only)")

    args = parser.parse_args()

    session = IdentityAnchoredSession(session_number=args.session, dry_run=args.dry_run)
    session.initialize_model(args.model)
    session.run_session()


if __name__ == "__main__":
    main()
