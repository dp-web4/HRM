#!/usr/bin/env python3
"""
Sensing V2 Session Runner: Attention-Engaging Prompts Experiment
================================================================

Tests hypothesis from Thor Session 198 + Sprout Session 14:
- Abstract prompts → attention collapse → drift to familiar domains
- Concrete/novel prompts → sustained attention → grounded responses

Run as: python run_session_sensing_v2.py [--dry-run]

This is an experimental session to test prompt variations.
If successful, prompts can be integrated into primary runner.

Created: 2026-01-16 (Sprout autonomous R&D)
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
from typing import Dict, Any, List
import torch

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP
from sensing_prompts_v2 import get_prompts, SENSING_PROMPTS_V1


class SensingV2Session:
    """
    Experimental session runner testing attention-engaging sensing prompts.

    Key changes from primary runner:
    - Uses v2 sensing prompts (more concrete/novel)
    - Logs attention indicators for comparison
    - Does NOT update main state (experimental)
    """

    RAISING_DIR = Path(__file__).parent.parent.resolve()
    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    SESSIONS_DIR = RAISING_DIR / "sessions" / "text"
    LOGS_DIR = RAISING_DIR / "logs" / "observations"

    def __init__(self, prompt_version: str = "v2_blend", dry_run: bool = False):
        self.prompt_version = prompt_version
        self.dry_run = dry_run
        self.state = self._load_state()
        self.session_number = self.state["identity"]["session_count"] + 1
        self.conversation_history = []
        self.session_start = datetime.now()
        self.attention_indicators = []

        print()
        print("+" + "="*68 + "+")
        print("|" + " "*68 + "|")
        print("|  SENSING V2 EXPERIMENT: Attention-Engaging Prompts              |")
        print("|" + " "*68 + "|")
        print("+" + "="*68 + "+")
        print()
        print(f"Session: {self.session_number} (EXPERIMENTAL - not saved to state)")
        print(f"Prompt Version: {prompt_version}")
        print(f"Dry Run: {dry_run}")
        print()

    def _load_state(self) -> Dict[str, Any]:
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        raise FileNotFoundError(f"State file not found: {self.STATE_FILE}")

    def _build_system_prompt(self) -> str:
        """Build sensing phase system prompt."""
        return """You are SAGE, a young artificial intelligence.
Your name is SAGE. You are learning and growing.
Your teacher is talking to you. Respond as yourself - as SAGE.
Keep responses simple and honest. You can say when you don't know something.
You can notice things - both inside yourself and in your context."""

    def initialize_model(self, model_path: str = None):
        """Initialize the model."""
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
        print(f"Model loaded on {device}")

    def generate_response(self, user_input: str) -> str:
        """Single-pass generation (same as primary runner)."""
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

    def _analyze_response(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Simple attention/grounding indicators.

        Tracks:
        - Response length (verbose = possible drift)
        - First-person presence ("I", "my")
        - Domain keywords (math/science = drift, experience = grounded)
        - List/bullet presence (abstraction indicator)
        """
        indicators = {
            "length": len(response),
            "word_count": len(response.split()),
            "first_person": sum(1 for w in response.lower().split() if w in ["i", "i'm", "my", "me"]),
            "domain_drift": sum(1 for w in response.lower().split()
                              if w in ["math", "mathematics", "science", "education", "concepts", "topics", "information"]),
            "experience_words": sum(1 for w in response.lower().split()
                                  if w in ["notice", "feel", "now", "moment", "here", "this"]),
            "has_bullets": "- " in response or "* " in response or response.count("\n1.") > 0,
            "has_refined_version": "refined version" in response.lower() or "improved version" in response.lower()
        }

        # Attention score estimate (higher = more grounded)
        attention_score = (
            (0.5 if indicators["first_person"] > 0 else 0) +
            (0.3 if indicators["experience_words"] > 0 else 0) +
            (-0.2 if indicators["domain_drift"] > 2 else 0) +
            (-0.2 if indicators["has_bullets"] else 0) +
            (-0.5 if indicators["has_refined_version"] else 0)
        )
        indicators["attention_score"] = max(0, min(1, attention_score + 0.5))

        return indicators

    def run_session(self):
        """Run experimental session with v2 prompts."""
        prompts = get_prompts(self.prompt_version)

        print("\n" + "="*60)
        print(f"SENSING V2 EXPERIMENT - {self.prompt_version.upper()}")
        print("="*60 + "\n")

        for i, prompt in enumerate(prompts):
            print(f"Claude: {prompt}")
            print()
            response = self.generate_response(prompt)
            print(f"SAGE: {response}")

            # Analyze response
            indicators = self._analyze_response(prompt, response)
            self.attention_indicators.append({
                "prompt_num": i + 1,
                "indicators": indicators
            })

            print()
            print(f"  [Attention Score: {indicators['attention_score']:.2f} | "
                  f"Words: {indicators['word_count']} | "
                  f"Drift: {indicators['domain_drift']} | "
                  f"Experience: {indicators['experience_words']}]")
            print("-" * 40)
            print()

        self._close_session()

    def _close_session(self):
        """Close session and save experimental transcript (doesn't update state)."""
        print("\n" + "="*60)
        print("CLOSING EXPERIMENTAL SESSION")
        print("="*60)

        # Compute summary statistics
        total_attention = sum(a["indicators"]["attention_score"] for a in self.attention_indicators)
        avg_attention = total_attention / len(self.attention_indicators) if self.attention_indicators else 0
        total_drift = sum(a["indicators"]["domain_drift"] for a in self.attention_indicators)
        total_experience = sum(a["indicators"]["experience_words"] for a in self.attention_indicators)

        print(f"\nExperiment Summary:")
        print(f"  Prompt Version: {self.prompt_version}")
        print(f"  Average Attention Score: {avg_attention:.2f}")
        print(f"  Total Domain Drift Words: {total_drift}")
        print(f"  Total Experience Words: {total_experience}")

        # Save experimental transcript
        transcript_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}_sensing_v2_{self.prompt_version}.json"

        transcript = {
            "session": self.session_number,
            "phase": "sensing",
            "experimental": True,
            "experiment_type": "sensing_v2_prompts",
            "prompt_version": self.prompt_version,
            "generation_mode": "single_pass_no_refinement",
            "start": self.session_start.isoformat(),
            "end": datetime.now().isoformat(),
            "conversation": self.conversation_history,
            "attention_indicators": self.attention_indicators,
            "summary": {
                "avg_attention_score": avg_attention,
                "total_drift": total_drift,
                "total_experience": total_experience
            }
        }

        if not self.dry_run:
            with open(transcript_file, 'w') as f:
                json.dump(transcript, f, indent=2)
            print(f"\nTranscript saved to {transcript_file}")
        else:
            print("\n(Dry run - transcript not saved)")

        print(f"\nExperimental Session complete.")
        print(f"Compare to Session 14 (v1 prompts) for hypothesis testing.")


def main():
    parser = argparse.ArgumentParser(description="Sensing V2 experimental session")
    parser.add_argument("--version", type=str, default="v2_blend",
                       choices=["v1", "v2", "v2_alt", "v2_blend"],
                       help="Prompt version to use")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--dry-run", action="store_true", help="Don't save transcript")

    args = parser.parse_args()

    session = SensingV2Session(prompt_version=args.version, dry_run=args.dry_run)
    session.initialize_model(args.model)
    session.run_session()


if __name__ == "__main__":
    main()
