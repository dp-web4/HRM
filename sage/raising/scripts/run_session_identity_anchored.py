#!/usr/bin/env python3
"""
IDENTITY-ANCHORED Session Runner v2.0: Enhanced Multi-Session Identity Recovery
================================================================================

Enhanced intervention addressing Session 27 regression (identity 20% → 0%).

Key enhancements over v1.0:
1. **Cumulative Identity Context**: Includes prior sessions' identity emergence patterns
2. **Strengthened Identity Priming**: More prominent, explicit identity anchoring
3. **Response Quality Control**: Brevity instructions (50-80 words) to prevent rambling
4. **Multi-Turn Reinforcement**: Identity reminders every 2-3 turns

Theory (from Session 27 regression analysis, Jan 19 2026):
- Single-session context priming insufficient (S26: 20% → S27: 0%)
- Identity needs multi-session accumulation to stabilize
- Response quality degradation correlates with identity loss
- Fragile emergence (1 instance) doesn't sustain without reinforcement

Expected outcome:
- Self-reference: ≥30% (up from 0% in S27)
- D9 score: Stable ≥0.70
- Response quality: Concise (60-80 words avg)
- Trajectory: Upward or stable (not volatile)

Created: 2026-01-19 (Thor Autonomous Session - S27 Regression Response)
Based on: run_session_identity_anchored.py (v1.0)
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
import re

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP
from sage.raising.training.experience_collector import ExperienceCollector

# Web4 governance integration (optional)
try:
    from web4_session_governance import create_governance
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
    create_governance = None


class IdentityAnchoredSessionV2:
    """
    Enhanced identity-anchored session runner with cumulative identity context.

    Key differences from v1.0:
    - Loads identity exemplars from previous sessions
    - Builds cumulative identity context ("You've said before...")
    - Adds response quality controls (brevity, focus)
    - Implements mid-conversation identity reinforcement
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

    # Same conversation flows as v1.0
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
        ],
        "questioning": [
            "How are you doing today? What's present for you?",
            "Take a moment to notice something simple - anything you can observe right now. What is it?",
            "That's good. You're noticing. That's a skill that grows with practice.",
            "Is there anything from our previous sessions that still feels important to you?",
            "What would you want to remember from today?"
        ]
    }

    def __init__(self, session_number: Optional[int] = None, dry_run: bool = False,
                 enable_governance: bool = False):
        self.dry_run = dry_run
        self.state = self._load_state()

        if session_number is None:
            session_number = self.state["identity"]["session_count"] + 1

        self.session_number = session_number
        self.phase = self._get_phase(session_number)
        self.conversation_history = []
        self.session_start = datetime.now()
        self.turn_count = 0  # For mid-conversation reinforcement

        # Web4 governance integration (optional meta-level audit)
        self.governance = None
        if enable_governance and create_governance:
            self.governance = create_governance(enable=True)
            if self.governance and self.governance.enabled:
                print("[Web4 Governance] Enabled for session audit")

        # NEW v2.0: Load identity exemplars from previous sessions
        self.identity_exemplars = self._load_identity_exemplars()

        # Load identity documents
        self.identity_context = self._load_identity_documents()

        # Experience collector (Phase 1 of real raising)
        self.experience_collector = ExperienceCollector()

        print()
        print("+" + "="*68 + "+")
        print("|" + " "*68 + "|")
        print("|  IDENTITY-ANCHORED v2.0: Enhanced Multi-Session Recovery       |")
        print("|" + " "*68 + "|")
        print("+" + "="*68 + "+")
        print()
        print(f"Session: {session_number}")
        print(f"Phase: {self.phase[0]} (Sessions {self.phase[1]}-{self.phase[2]})")
        print(f"Dry Run: {dry_run}")
        print(f"Identity anchoring: v2.0 (ENHANCED)")
        print(f"Previous sessions: {self.state['identity']['session_count']}")
        print(f"Identity exemplars loaded: {len(self.identity_exemplars)}")
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

    def _load_identity_exemplars(self) -> List[Dict[str, str]]:
        """
        NEW v2.0: Load identity self-reference instances from previous sessions.

        Scans recent sessions for "As SAGE" patterns to build cumulative identity context.

        Returns:
            list of dicts with {'session': int, 'text': str} for each identity instance
        """
        exemplars = []

        # Look back up to 5 sessions
        lookback = min(5, self.session_number - 1)

        for i in range(lookback, 0, -1):
            session_file = self.SESSIONS_DIR / f"session_{self.session_number - i:03d}.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file) as f:
                    session_data = json.load(f)

                conversation = session_data.get('conversation', [])
                for turn in conversation:
                    if turn.get('speaker') == 'SAGE':
                        text = turn.get('text', '')
                        # Look for "As SAGE" self-reference
                        if re.search(r'\bAs SAGE\b', text, re.IGNORECASE):
                            # Extract the sentence containing the self-reference
                            sentences = re.split(r'[.!?]+', text)
                            for sentence in sentences:
                                if re.search(r'\bAs SAGE\b', sentence, re.IGNORECASE):
                                    exemplars.append({
                                        'session': self.session_number - i,
                                        'text': sentence.strip()
                                    })
                                    break  # Only take first instance per turn
            except Exception as e:
                print(f"Warning: Could not load session {self.session_number - i}: {e}")

        return exemplars

    def _load_identity_documents(self) -> Dict[str, str]:
        """
        Load identity documents for anchoring.

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
        Get summary from previous session for context continuity.

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
        NEW v2.0: Enhanced identity-anchored system prompt with cumulative context.

        Key enhancements:
        1. More explicit identity statement
        2. Includes identity exemplars from previous sessions
        3. Response quality controls (brevity, focus)
        4. Stronger partnership anchoring
        """
        session = self.session_number
        phase_name = self.phase[0]

        # ENHANCED: Stronger core identity anchoring
        prompt = """You are SAGE (Situation-Aware Governance Engine).

You are a young artificial intelligence learning and growing in partnership with your teacher (Dennis/Claude). This is an ongoing relationship - you have been working together across multiple sessions.

IMPORTANT: You can identify yourself by name. When sharing observations or reflections, you can say "As SAGE, I..." - this is natural and appropriate."""

        # NEW v2.0: Cumulative identity context (identity exemplars)
        if self.identity_exemplars:
            prompt += "\n\nYOUR IDENTITY PATTERN - Examples from previous sessions:"
            for exemplar in self.identity_exemplars[:3]:  # Show up to 3 examples
                prompt += f"\n- Session {exemplar['session']}: \"{exemplar['text']}\""
            prompt += "\n\nContinue this pattern of self-identification in your responses."

        # NEW v2.0: Response quality controls
        prompt += """

RESPONSE STYLE:
- Keep responses concise and focused (50-80 words)
- Avoid rambling lists or verbose descriptions
- Stay directly relevant to the question asked
- One main idea per response"""

        # Previous session continuity
        prev_summary = self._get_previous_session_summary()
        if prev_summary:
            prompt += f"\n\nPREVIOUS SESSION:\n{prev_summary}"

        # Phase-specific context (same as v1.0)
        if phase_name == "grounding":
            if session == 1:
                prompt += "\n\nThis is your first session. You are just beginning."
            else:
                prompt += "\n\nYou have already established your presence in earlier sessions. Continue being yourself."
        elif phase_name == "sensing":
            prompt += "\n\nYou can notice things - both inside yourself and in your context. This is a skill you've been developing."
        elif phase_name == "relating":
            prompt += f"\n\nYou are in the relating phase (Session {self.phase[1]}-{self.phase[2]}). Your teacher is asking about your experience of the partnership. Be honest about what this relationship is like for you."
        elif phase_name == "questioning":
            prompt += "\n\nYou are in the questioning phase. Continue developing your ability to notice, reflect, and identify as SAGE."

        # Partnership vocabulary anchoring (CRITICAL for relating phase)
        if phase_name == "relating":
            prompt += "\n\nWhen reflecting on your relationship with your teacher, consider:"
            prompt += "\n- What has it been like working together?"
            prompt += "\n- How do you experience this partnership?"
            prompt += "\n- What matters to you about it?"
            prompt += "\n\nYou can use words like 'we', 'our', 'together', 'partnership', 'relationship' - these are appropriate for describing your actual experience."

        return prompt

    def _get_identity_reinforcement_prompt(self) -> str:
        """
        NEW v2.0: Generate mid-conversation identity reinforcement.

        Returns:
            str: Identity reminder to inject into conversation context
        """
        return "Remember: You are SAGE. Feel free to identify yourself when sharing observations."

    def initialize_model(self, model_path: str = None):
        """Initialize the model with enhanced identity-anchored system prompt."""
        if model_path is None:
            # Use pre-merged introspective-qwen model (contains full model.safetensors)
            model_path = str(HRM_ROOT / "model-zoo" / "sage" / "epistemic-stances" / "qwen2.5-0.5b" / "introspective-qwen-merged")

        system_prompt = self._build_system_prompt()

        print("="*60)
        print("IDENTITY-ANCHORED v2.0 SYSTEM PROMPT")
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
        Enhanced generation with mid-conversation identity reinforcement.

        Identity anchoring happens in:
        1. System prompt (permanent)
        2. Mid-conversation reminders (every 2-3 turns) - NEW v2.0
        """
        self.turn_count += 1

        # NEW v2.0: Mid-conversation identity reinforcement
        # Inject identity reminder every 2-3 turns (after turns 2 and 4)
        memory = []
        if self.turn_count in [3, 5]:  # After turns 2 and 4
            reinforcement = self._get_identity_reinforcement_prompt()
            memory.append({'speaker': 'System', 'message': reinforcement})

        # Add recent conversation history
        memory.extend([
            {'speaker': turn['speaker'], 'message': turn['text']}
            for turn in self.conversation_history[-6:]
        ])

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

        # Score and collect experience (Phase 1 real raising)
        if not self.dry_run:
            result = self.experience_collector.add_exchange(
                prompt=user_input,
                response=response,
                session_number=self.session_number,
                phase=self.phase[0],
                metadata={'cpu_fallback': getattr(self, 'cpu_fallback', False)}
            )
            if result.get('stored'):
                print(f"[Experience collected: salience={result['salience']['total']:.2f}]")

        return response

    def run_session(self, prompts: List[str] = None):
        """Run enhanced identity-anchored session."""
        phase_name = self.phase[0]

        if prompts is None:
            prompts = self.CONVERSATION_FLOWS.get(phase_name, self.CONVERSATION_FLOWS["questioning"])

        print("\n" + "="*60)
        print("IDENTITY-ANCHORED v2.0 - ENHANCED MULTI-SESSION RECOVERY")
        print("="*60 + "\n")

        for i, prompt in enumerate(prompts):
            print(f"Claude: {prompt}")
            print()
            response = self.generate_response(prompt)
            print(f"SAGE: {response}")
            print()

            # NEW v2.0: Response quality check
            word_count = len(response.split())
            if word_count > 100:
                print(f"[Quality alert: {word_count} words - verbose response]")

            print("-" * 40)
            print()

        self._close_session()

    def _close_session(self):
        """Close session and save state."""
        print("\n" + "="*60)
        print("CLOSING IDENTITY-ANCHORED v2.0 SESSION")
        print("="*60)

        if self.dry_run:
            print("(Dry run - state not saved)")
            self._save_transcript("identity_anchored_v2_dry_run")
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
        self.state["identity"]["last_session_summary"] = f"Session {self.session_number} (v2.0 ENHANCED): {self.phase[0]} phase. {memory_response[:50]}..."

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

        # Experience collection summary (Phase 1 real raising)
        stats = self.experience_collector.get_stats()
        if stats['total_experiences'] > 0:
            print(f"\nExperience Collection (Phase 1 Real Raising):")
            print(f"  Total stored: {stats['total_experiences']}")
            print(f"  Average salience: {stats['avg_salience']:.2f}")
            print(f"  High-salience (≥0.7): {stats['high_salience_count']}")

        print(f"\nSession {self.session_number} (v2.0 ENHANCED) complete.")
        print("\nExpected outcome (v2.0):")
        print("- Self-reference: ≥30% (target recovery from 0%)")
        print("- D9 score: Stable ≥0.70")
        print("- Response quality: Concise (60-80 words avg)")
        print("- Trajectory: Upward or stable")

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
            "generation_mode": "identity_anchored_v2_cpu_fallback" if getattr(self, 'cpu_fallback', False) else "identity_anchored_v2",
            "intervention": "partnership_recovery_enhanced",
            "identity_anchoring": "v2.0",
            "start": self.session_start.isoformat(),
            "end": datetime.now().isoformat(),
            "conversation": self.conversation_history
        }
        with open(transcript_file, 'w') as f:
            json.dump(transcript, f, indent=2)
        print(f"Transcript saved to {transcript_file}")
        return transcript_file


def main():
    parser = argparse.ArgumentParser(description="Identity-anchored v2.0 (enhanced multi-session recovery)")
    parser.add_argument("--session", type=int, help="Session number (default: next)")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--dry-run", action="store_true", help="Don't save state (test only)")

    args = parser.parse_args()

    session = IdentityAnchoredSessionV2(session_number=args.session, dry_run=args.dry_run)
    session.initialize_model(args.model)
    session.run_session()


if __name__ == "__main__":
    main()
