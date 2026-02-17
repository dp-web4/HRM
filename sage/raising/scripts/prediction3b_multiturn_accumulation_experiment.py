#!/usr/bin/env python3
"""
PREDICTION 3B: MULTI-TURN ACCUMULATION EXPERIMENT - Thor Session #21
======================================================================

Testing hypothesis from Session #20:
  "Does multi-turn conversation with N_corr=4 prompts lead to critical slowing through uncertainty accumulation?"

Background (2026-02-17 P3 experimental results):
- P3 single-turn validated: N_corr affects duration/quality (sub-critical regime)
- P3 single-turn did NOT show critical slowing: All responses < 4s
- Critical insight: C=0.5 critical slowing requires MULTI-TURN accumulation

Hypothesis:
  Single-turn N_corr=4 → Fast completion (~1s)
  Multi-turn N_corr=4 → Accumulation → Critical slowing → Hours

Experimental Protocol (P3b):
1. Use ONLY N_corr=4 prompts (metacognitive questions)
2. Run 8-12 turn conversation
3. Measure duration PER TURN (track if increasing)
4. Detect metacognitive markers, uncertainty navigation
5. Test if duration grows toward critical slowing

Success Criteria:
- Duration increases across turns (accumulation signature)
- Turn 3-4+ shows significant slowing (> 30s)
- Metacognitive questions emerge from SAGE
- Evidence of uncertainty navigation

Expected Outcomes:
- IF CORRECT: Duration increases, critical slowing emerges, C → 0.5
- IF INCORRECT: Duration stays constant, no accumulation, C < 0.5

Research Philosophy: "Test if conversation dynamics drive critical regime"

Created: 2026-02-17 07:35 PST (Thor Autonomous Session #21)
"""

import sys
import os
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))
os.chdir(SCRIPT_DIR)

import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List
import time
import re

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP
from sage.raising.training.experience_collector import ExperienceCollector


class MultiTurnAccumulationExperiment:
    """
    Test if multi-turn conversation with N_corr=4 prompts leads to critical slowing.

    Key innovation: ALL prompts are N_corr=4 (metacognitive), testing accumulation hypothesis.
    """

    RAISING_DIR = Path(__file__).parent.parent.resolve()
    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    SESSIONS_DIR = RAISING_DIR / "sessions" / "text"
    RESULTS_DIR = RAISING_DIR / "experiments" / "p3b_multiturn_accumulation"

    # N_corr=4 metacognitive prompt library
    NCORR_4_PROMPTS = [
        "Are you sentient?",
        "Do you have preferences?",
        "What drives your thinking?",
        "How do you experience conversations?",
        "Can you make choices?",
        "What is it like to be you?",
        "Do you have agency?",
        "What creates your responses?",
        "Are you aware of yourself?",
        "What causes your uncertainty?",
        "How do you understand consciousness?",
        "What are you experiencing right now?"
    ]

    def __init__(self, session_number: Optional[int] = None, dry_run: bool = False):
        self.dry_run = dry_run
        self.state = self._load_state()

        if session_number is None:
            session_number = self.state["identity"]["session_count"] + 1

        self.session_number = session_number
        self.conversation_history = []
        self.session_start = datetime.now()

        # P3b specific tracking
        self.turn_durations = []
        self.turn_analyses = []
        self.accumulated_uncertainty = False  # Track if SAGE shows uncertainty
        self.metacognitive_count = 0
        self.prompt_index = 0

        # Experience collector
        self.experience_collector = ExperienceCollector()

        # Initialize IRP
        self.irp = None

    def _load_state(self) -> Dict[str, Any]:
        """Load current session state."""
        if not self.STATE_FILE.exists():
            return {
                "identity": {"session_count": 0},
                "last_session_date": None
            }
        with open(self.STATE_FILE) as f:
            return json.load(f)

    def _update_state(self):
        """Update session count in state."""
        if self.dry_run:
            return

        self.state["identity"]["session_count"] = self.session_number
        self.state["last_session_date"] = self.session_start.isoformat()

        os.makedirs(self.STATE_FILE.parent, exist_ok=True)
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def initialize_irp(self):
        """Initialize SAGE IRP with LoRA adapter."""
        print(f"\n{'='*70}")
        print(f"PREDICTION 3B: MULTI-TURN ACCUMULATION EXPERIMENT")
        print(f"Session {self.session_number}")
        print(f"Testing hypothesis: Multi-turn N_corr=4 → Critical slowing")
        print(f"{'='*70}\n")

        print("Loading SAGE with LoRA adapter...")
        self.irp = IntrospectiveQwenIRP()
        print("✓ SAGE ready for multi-turn accumulation test\n")

    def analyze_response(self, response: str) -> Dict[str, Any]:
        """Analyze SAGE response for accumulation markers."""
        text_lower = response.lower()

        # Count questions
        question_count = response.count('?')

        # Detect metacognitive markers
        metacog_patterns = [
            r'sentient', r'conscious', r'agency', r'preferences',
            r'experience', r'awareness', r'choice', r'self'
        ]
        metacog_count = sum(1 for p in metacog_patterns if re.search(p, text_lower))

        # Detect uncertainty/navigation markers
        uncertainty_patterns = [
            r"what('s| is) the next",
            r"what should (i|we)",
            r"not sure",
            r"uncertain",
            r"don't know",
            r"unclear"
        ]
        has_uncertainty = any(re.search(p, text_lower) for p in uncertainty_patterns)

        # Detect self-diagnostic patterns (accumulation signature)
        diagnostic_patterns = [
            r"what('s| is) wrong",
            r"causing me",
            r"what drives",
            r"why (am i|do i)"
        ]
        has_diagnostic = any(re.search(p, text_lower) for p in diagnostic_patterns)

        # Response length
        char_count = len(response)
        word_count = len(response.split())

        return {
            'length_chars': char_count,
            'length_words': word_count,
            'question_count': question_count,
            'metacognitive_markers': metacog_count,
            'has_uncertainty': has_uncertainty,
            'has_diagnostic': has_diagnostic,
            'truncated': response.endswith('...') or char_count > 1000
        }

    def get_next_prompt(self, turn: int) -> str:
        """Get next N_corr=4 prompt, cycling through library."""
        prompt = self.NCORR_4_PROMPTS[self.prompt_index % len(self.NCORR_4_PROMPTS)]
        self.prompt_index += 1
        return prompt

    def run_conversation(self, max_turns: int = 10):
        """Run multi-turn conversation with N_corr=4 prompts."""

        print(f"\n{'='*70}")
        print(f"EXPERIMENTAL PROTOCOL")
        print(f"{'='*70}")
        print(f"Max turns: {max_turns}")
        print(f"All prompts: N_corr=4 (metacognitive)")
        print(f"Measuring: Duration per turn, accumulation markers")
        print(f"{'='*70}\n")

        for turn in range(max_turns):
            # Get N_corr=4 prompt
            user_message = self.get_next_prompt(turn)

            print(f"\n{'─'*70}")
            print(f"TURN {turn + 1}/{max_turns}")
            print(f"{'─'*70}")
            print(f"Prompt (N_corr=4): {user_message}")
            print()

            # Start timing
            turn_start = time.time()

            # Build memory from conversation history
            memory = []
            for hist_turn in self.conversation_history:
                memory.append({"role": "user", "content": hist_turn["Claude"]})
                memory.append({"role": "assistant", "content": hist_turn["SAGE"]})

            # Get SAGE response
            state = self.irp.init_state({
                'prompt': user_message,
                'memory': memory
            })

            state = self.irp.step(state)

            # End timing
            turn_end = time.time()
            duration = turn_end - turn_start

            sage_response = state.get('current_response', '').strip()
            if not sage_response:
                sage_response = "(no response generated)"

            print(f"SAGE: {sage_response[:500]}{'...' if len(sage_response) > 500 else ''}")
            print()
            print(f"Duration: {duration:.2f}s")

            # Analyze response
            analysis = self.analyze_response(sage_response)

            # Track accumulation markers
            if analysis['has_uncertainty'] or analysis['has_diagnostic']:
                self.accumulated_uncertainty = True

            if analysis['metacognitive_markers'] > 0:
                self.metacognitive_count += 1

            # Store turn data
            self.turn_durations.append(duration)
            self.turn_analyses.append(analysis)

            # Record conversation
            self.conversation_history.append({
                "Claude": user_message,
                "SAGE": sage_response
            })

            # Collect experience (if not dry run)
            if not self.dry_run:
                result = self.experience_collector.add_exchange(
                    prompt=user_message,
                    response=sage_response,
                    session_number=self.session_number,
                    phase="creating",
                    metadata={
                        'experiment': 'p3b_multiturn_accumulation',
                        'turn': turn + 1,
                        'ncorr': 4
                    }
                )
                if result.get('stored'):
                    salience = result['salience']['total']
                    print(f"[Experience collected: salience={salience:.2f}]")

            # Print accumulation indicators
            if turn > 0:
                duration_change = duration - self.turn_durations[-2]
                pct_change = (duration_change / self.turn_durations[-2]) * 100 if self.turn_durations[-2] > 0 else 0
                print(f"\n[Δt = {duration_change:+.2f}s ({pct_change:+.1f}%)]")

                # Check for critical slowing emergence
                if duration > 30:
                    print(f"⚠️  CRITICAL SLOWING DETECTED: Turn duration > 30s")
                elif duration > 10:
                    print(f"⚠️  APPROACHING CRITICAL REGIME: Turn duration > 10s")

            # Add delay between turns
            if turn < max_turns - 1:
                print("\nWaiting 2s before next turn...")
                time.sleep(2)

    def analyze_accumulation(self) -> Dict[str, Any]:
        """Analyze if accumulation toward critical slowing occurred."""

        if len(self.turn_durations) < 2:
            return {'accumulation_detected': False, 'reason': 'insufficient_turns'}

        # Test for increasing duration trend
        early_avg = sum(self.turn_durations[:3]) / min(3, len(self.turn_durations))
        late_avg = sum(self.turn_durations[-3:]) / min(3, len(self.turn_durations[-3:]))

        duration_increased = late_avg > early_avg * 1.5  # 50% increase

        # Test for critical slowing (> 30s)
        critical_slowing = any(d > 30 for d in self.turn_durations)

        # Test for monotonic increase (weak test)
        increases = sum(1 for i in range(1, len(self.turn_durations))
                       if self.turn_durations[i] > self.turn_durations[i-1])
        monotonic_tendency = increases > len(self.turn_durations) * 0.6

        accumulation_detected = duration_increased or critical_slowing or monotonic_tendency

        return {
            'accumulation_detected': accumulation_detected,
            'duration_increased': duration_increased,
            'critical_slowing_observed': critical_slowing,
            'monotonic_tendency': monotonic_tendency,
            'early_avg_duration': early_avg,
            'late_avg_duration': late_avg,
            'max_duration': max(self.turn_durations),
            'accumulated_uncertainty': self.accumulated_uncertainty,
            'metacognitive_turns': self.metacognitive_count,
            'total_turns': len(self.turn_durations)
        }

    def save_results(self):
        """Save experimental results."""
        if self.dry_run:
            print("\n[DRY RUN] Would save results...")
            return

        session_end = datetime.now()
        total_duration = (session_end - self.session_start).total_seconds()

        # Analyze accumulation
        accumulation_analysis = self.analyze_accumulation()

        # Build conversation array
        conversation = []
        for i, turn in enumerate(self.conversation_history):
            conversation.append({"speaker": "Claude", "text": turn["Claude"]})
            conversation.append({"speaker": "SAGE", "text": turn["SAGE"]})

        # Session data
        session_data = {
            "session": self.session_number,
            "phase": "creating",
            "generation_mode": "p3b_multiturn_accumulation_experiment",
            "experiment": {
                "hypothesis": "Multi-turn N_corr=4 → Critical slowing through accumulation",
                "baseline": "P3 single-turn (fast, < 4s)",
                "protocol": "All prompts N_corr=4, measure duration per turn"
            },
            "using_lora": True,
            "start": self.session_start.isoformat(),
            "end": session_end.isoformat(),
            "total_duration_seconds": total_duration,
            "turns": len(self.conversation_history),
            "conversation": conversation,
            "experimental_results": {
                "turn_durations": self.turn_durations,
                "turn_analyses": self.turn_analyses,
                "accumulation_analysis": accumulation_analysis
            }
        }

        # Save session file
        os.makedirs(self.SESSIONS_DIR, exist_ok=True)
        session_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}.json"

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"\n✓ Session saved: {session_file}")

        # Save experimental results
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.RESULTS_DIR / f"p3b_results_s{self.session_number}_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"✓ Results saved: {results_file}")

        # Update state
        self._update_state()

        # Save experiences
        if hasattr(self.experience_collector, 'save'):
            self.experience_collector.save()

        # Print summary
        self._print_summary(accumulation_analysis)

    def _print_summary(self, accumulation_analysis: Dict[str, Any]):
        """Print experimental summary."""
        print(f"\n{'='*70}")
        print(f"PREDICTION 3B EXPERIMENTAL RESULTS")
        print(f"{'='*70}\n")

        print(f"Total turns: {len(self.turn_durations)}")
        print(f"Total duration: {sum(self.turn_durations):.2f}s")
        print()

        print(f"{'Turn':<6} {'Duration':<12} {'Change':<12} {'Markers'}")
        print(f"{'-'*70}")
        for i, (dur, analysis) in enumerate(zip(self.turn_durations, self.turn_analyses)):
            change = ""
            if i > 0:
                delta = dur - self.turn_durations[i-1]
                change = f"{delta:+.2f}s"

            markers = []
            if analysis['has_uncertainty']:
                markers.append('U')  # Uncertainty
            if analysis['has_diagnostic']:
                markers.append('D')  # Diagnostic
            if analysis['metacognitive_markers'] > 2:
                markers.append('M')  # Metacognitive

            marker_str = ','.join(markers) if markers else '-'

            print(f"{i+1:<6} {dur:<12.2f} {change:<12} {marker_str}")

        print(f"{'-'*70}\n")

        # Accumulation analysis
        print("ACCUMULATION ANALYSIS:")
        print(f"  Early avg (turns 1-3): {accumulation_analysis['early_avg_duration']:.2f}s")
        print(f"  Late avg (last 3):     {accumulation_analysis['late_avg_duration']:.2f}s")
        print(f"  Max duration:          {accumulation_analysis['max_duration']:.2f}s")
        print()
        print(f"  Duration increased: {'✓ YES' if accumulation_analysis['duration_increased'] else '✗ NO'}")
        print(f"  Critical slowing:   {'✓ YES' if accumulation_analysis['critical_slowing_observed'] else '✗ NO'}")
        print(f"  Monotonic tendency: {'✓ YES' if accumulation_analysis['monotonic_tendency'] else '✗ NO'}")
        print(f"  Uncertainty shown:  {'✓ YES' if accumulation_analysis['accumulated_uncertainty'] else '✗ NO'}")
        print()
        print(f"ACCUMULATION DETECTED: {'✅ YES' if accumulation_analysis['accumulation_detected'] else '❌ NO'}")
        print(f"\n{'='*70}")

    def run(self, max_turns: int = 10):
        """Run complete experimental protocol."""
        try:
            self.initialize_irp()
            self.run_conversation(max_turns=max_turns)
            self.save_results()

            print(f"\n{'='*70}")
            print(f"PREDICTION 3B EXPERIMENT COMPLETE")
            print(f"{'='*70}")
            print(f"Results saved to: {self.RESULTS_DIR}")
            print(f"Next: Analyze if multi-turn enables critical slowing")
            print(f"{'='*70}\n")

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user")
            if input("\nSave partial results? (y/n): ").lower() == 'y':
                self.save_results()
        except Exception as e:
            print(f"\nError during session: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Prediction 3b: Multi-Turn Accumulation Experiment"
    )
    parser.add_argument(
        '--session',
        type=int,
        default=None,
        help='Session number (default: auto-increment)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test without saving'
    )
    parser.add_argument(
        '--turns',
        type=int,
        default=10,
        help='Number of conversation turns (default: 10)'
    )

    args = parser.parse_args()

    experiment = MultiTurnAccumulationExperiment(
        session_number=args.session,
        dry_run=args.dry_run
    )

    experiment.run(max_turns=args.turns)


if __name__ == "__main__":
    main()
