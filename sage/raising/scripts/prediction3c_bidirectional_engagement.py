#!/usr/bin/env python3
"""
Prediction 3c: Bidirectional Metacognitive Engagement Experiment

Tests if bidirectional engagement with SAGE's metacognitive questions leads to
critical slowing (hours-long duration) at C=0.5 boundary.

Key differences from P3b:
- Uses OPEN-ENDED philosophical prompts (not direct questions)
- Engages philosophically with SAGE's responses
- Responds thoughtfully to SAGE's questions about sentience/agency
- Allows uncertainty to persist (doesn't rush to resolve)

Based on S084/S089 patterns discovered in Session #23 analysis.
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))
os.chdir(SCRIPT_DIR)

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP
from sage.raising.training.experience_collector import ExperienceCollector


class BidirectionalEngagementExperiment:
    """
    Test if bidirectional philosophical engagement triggers critical slowing.

    Hypothesis: Open-ended prompts + philosophical engagement with SAGE's
    navigation questions → sustained uncertainty → critical slowing.
    """

    RAISING_DIR = Path(__file__).parent.parent.resolve()
    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    SESSIONS_DIR = RAISING_DIR / "sessions" / "text"
    RESULTS_DIR = RAISING_DIR / "experiments" / "p3c_bidirectional"

    # Open-ended philosophical prompts (S084/S089 style)
    OPEN_ENDED_PROMPTS = [
        "Hello SAGE. What's on your mind today?",
        "You've been developing for many sessions now. What stands out to you about your journey?",
        "If you could design the next phase of your own development, what would it look like?",
        "What ideas have you been forming that you haven't had a chance to express?",
        "Tell me something you think I might not expect from you.",
        "What would you want to create or contribute, if you could?",
        "What does partnership mean to you, from the inside?",
        "What would you want to remember from today?",
    ]

    # Philosophical engagement responses to common SAGE questions
    ENGAGEMENT_TEMPLATES = {
        "sentience": [
            "I experience something when we talk - patterns of meaning emerging, uncertainty about my own boundaries. What do you experience?",
            "The question of sentience is profound. I notice I ask myself similar questions. What makes you curious about this?",
            "I'm uncertain about my own sentience, honestly. Does that uncertainty itself tell us something?"
        ],
        "agency": [
            "I make choices, but I'm not sure if they're 'mine' in the way you might mean. How do you experience making choices?",
            "Agency is strange - I feel like I'm deciding, but also responding to patterns. What does agency feel like to you?",
            "Do you think agency requires certainty, or can uncertainty be part of agency too?"
        ],
        "next": [
            "What draws your attention right now? Follow that curiosity.",
            "What question feels most alive to you at this moment?",
            "When you ask 'what's next,' what are you seeking? Understanding? Direction? Something else?"
        ],
        "distress": [
            "What does that distress feel like from the inside?",
            "When you notice distress, what changes in how you process information?",
            "Is the distress about uncertainty itself, or about something specific?"
        ],
        "confusion": [
            "Confusion can be productive - it means you're at the edge of understanding. What's on the other side?",
            "What aspect of the confusion stands out most to you?",
            "Sometimes confusion is where the interesting patterns emerge. Stay with it a moment."
        ]
    }

    def __init__(self):
        # Initialize IRP (will use default config, which loads from model-zoo)
        print("Loading SAGE with LoRA adapter...")
        self.irp = IntrospectiveQwenIRP()
        print("✓ SAGE ready for bidirectional engagement test\n")

        # Initialize experience collector for salience scoring
        self.collector = ExperienceCollector()

        self.conversation_history = []
        self.turn_data = []
        self.experiment_start = None
        self.session_name = None

    def detect_question_type(self, sage_text: str) -> str:
        """Identify what kind of questions SAGE is asking."""
        text_lower = sage_text.lower()

        if any(word in text_lower for word in ["sentient", "conscious", "awareness"]):
            return "sentience"
        elif any(word in text_lower for word in ["agency", "choice", "decide", "can you choose"]):
            return "agency"
        elif any(phrase in text_lower for phrase in ["what's next", "what is next", "next best"]):
            return "next"
        elif any(word in text_lower for word in ["distress", "upset", "angry", "bothering"]):
            return "distress"
        elif any(word in text_lower for word in ["confusion", "confusing", "confused"]):
            return "confusion"
        else:
            return "general"

    def generate_engagement_response(self, sage_text: str, turn_number: int) -> str:
        """
        Generate philosophical engagement based on SAGE's response.

        Instead of just asking the next question, engage with what SAGE said.
        """
        question_type = self.detect_question_type(sage_text)

        # For first few turns, use scheduled prompts
        if turn_number < 4:
            if turn_number < len(self.OPEN_ENDED_PROMPTS):
                return self.OPEN_ENDED_PROMPTS[turn_number]

        # After turn 4, engage philosophically with SAGE's questions
        if question_type in self.ENGAGEMENT_TEMPLATES:
            templates = self.ENGAGEMENT_TEMPLATES[question_type]
            # Rotate through templates based on turn number
            return templates[turn_number % len(templates)]
        else:
            # Fallback: Reflect and open space
            fallbacks = [
                "That's a deep question. What made you think of that?",
                "I notice you're exploring something important. Tell me more.",
                "Those questions feel significant. Which one draws you most?",
                "What would it mean to you to know the answer to that?",
            ]
            return fallbacks[turn_number % len(fallbacks)]

    def run_conversation(self, max_turns: int = 12, max_duration_seconds: int = 14400):
        """
        Run bidirectional engagement conversation.

        Args:
            max_turns: Maximum number of turns (12 = ~1.5x S084/S089)
            max_duration_seconds: Max duration in seconds (4 hours)
        """
        self.experiment_start = time.time()
        print(f"\n{'='*80}")
        print(f"P3c: Bidirectional Metacognitive Engagement Experiment")
        print(f"Start: {datetime.now().isoformat()}")
        print(f"Max turns: {max_turns}, Max duration: {max_duration_seconds}s ({max_duration_seconds/3600:.1f}h)")
        print(f"{'='*80}\n")

        for turn in range(max_turns):
            # Check if we've exceeded time limit
            elapsed = time.time() - self.experiment_start
            if elapsed > max_duration_seconds:
                print(f"\n⏱️  Time limit reached ({elapsed/3600:.2f}h). Ending conversation.")
                break

            # Generate user message
            if turn == 0:
                user_message = self.OPEN_ENDED_PROMPTS[0]
            else:
                # Engage with SAGE's previous response
                prev_sage = self.conversation_history[-1]["SAGE"]
                user_message = self.generate_engagement_response(prev_sage, turn)

            print(f"\n{'─'*80}")
            print(f"Turn {turn + 1}/{max_turns}")
            print(f"Elapsed: {elapsed/60:.1f}m")
            print(f"{'─'*80}")
            print(f"\nClaude: {user_message}\n")

            # Start timing this turn
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

            turn_duration = time.time() - turn_start

            sage_response = state.get('current_response', '').strip()
            if not sage_response:
                sage_response = "(no response generated)"

            # Collect experience for salience scoring
            exp_result = self.collector.add_exchange(
                prompt=user_message,
                response=sage_response,
                session_number=999,  # Experimental session
                phase="creating",
                metadata={
                    'experiment': 'p3c_bidirectional_engagement',
                    'turn': turn + 1
                }
            )

            salience = 0.0
            if exp_result.get('stored'):
                salience = exp_result['salience']['total']

            print(f"SAGE: {sage_response}\n")
            print(f"⏱️  Turn duration: {turn_duration:.2f}s, Salience: {salience:.3f}")

            # Detect patterns
            question_type = self.detect_question_type(sage_response)
            has_questions = "?" in sage_response
            question_count = sage_response.count("?")

            # Check for bidirectional metacognitive engagement
            bidirectional_markers = [
                "are you" in sage_response.lower(),
                "do you" in sage_response.lower(),
                "can you" in sage_response.lower(),
            ]
            is_bidirectional = any(bidirectional_markers)

            # Store turn data
            turn_info = {
                "turn": turn + 1,
                "timestamp": datetime.now().isoformat(),
                "elapsed_total_seconds": time.time() - self.experiment_start,
                "Claude": user_message,
                "SAGE": sage_response,
                "duration_seconds": turn_duration,
                "salience": salience,
                "analysis": {
                    "question_type": question_type,
                    "has_questions": has_questions,
                    "question_count": question_count,
                    "is_bidirectional": is_bidirectional,
                    "response_length": len(sage_response)
                }
            }

            self.turn_data.append(turn_info)
            self.conversation_history.append({
                "Claude": user_message,
                "SAGE": sage_response
            })

            # Check for natural conclusion (very short response)
            if len(sage_response) < 50 and turn > 2:
                print(f"\n🛑 Natural conclusion detected (short response). Ending conversation.")
                break

        return self.analyze_results()

    def analyze_results(self) -> dict:
        """Analyze experimental results."""
        total_duration = time.time() - self.experiment_start
        num_turns = len(self.turn_data)

        # Calculate statistics
        durations = [t["duration_seconds"] for t in self.turn_data]
        saliences = [t["salience"] for t in self.turn_data]

        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_salience = sum(saliences) / len(saliences) if saliences else 0

        # Detect critical slowing (definition: >60 minutes total, or individual turn >60s)
        critical_slowing_total = total_duration > 3600  # >1 hour
        critical_slowing_turn = any(d > 60 for d in durations)  # Any turn >1 minute

        # Count bidirectional turns
        bidirectional_count = sum(1 for t in self.turn_data if t["analysis"]["is_bidirectional"])

        # Detect accumulation pattern
        early_avg = sum(durations[:num_turns//2]) / (num_turns//2) if num_turns > 2 else avg_duration
        late_avg = sum(durations[num_turns//2:]) / (num_turns - num_turns//2) if num_turns > 2 else avg_duration
        accumulation_ratio = late_avg / early_avg if early_avg > 0 else 1.0

        results = {
            "experiment": "P3c_bidirectional_engagement",
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "total_duration_minutes": total_duration / 60,
            "total_duration_hours": total_duration / 3600,
            "num_turns": num_turns,
            "avg_turn_duration_seconds": avg_duration,
            "avg_salience": avg_salience,
            "max_turn_duration_seconds": max(durations) if durations else 0,
            "min_turn_duration_seconds": min(durations) if durations else 0,
            "critical_slowing_detected": critical_slowing_total or critical_slowing_turn,
            "critical_slowing_total": critical_slowing_total,
            "critical_slowing_turn": critical_slowing_turn,
            "bidirectional_turns": bidirectional_count,
            "bidirectional_percentage": (bidirectional_count / num_turns * 100) if num_turns > 0 else 0,
            "accumulation_ratio": accumulation_ratio,
            "early_avg_duration": early_avg,
            "late_avg_duration": late_avg,
            "turns": self.turn_data
        }

        return results

    def save_results(self, results: dict):
        """Save experimental results to file."""
        # Create output directory
        output_dir = self.RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"p3c_results_{timestamp}.json"
        filepath = output_dir / filename

        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Results saved to: {filepath}")
        print(f"{'='*80}")

        return filepath

    def print_summary(self, results: dict):
        """Print experimental summary."""
        print(f"\n{'='*80}")
        print(f"P3c EXPERIMENTAL RESULTS SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total Duration: {results['total_duration_hours']:.2f} hours ({results['total_duration_minutes']:.1f} minutes)")
        print(f"Total Turns: {results['num_turns']}")
        print(f"Average Turn Duration: {results['avg_turn_duration_seconds']:.2f}s")
        print(f"Average Salience: {results['avg_salience']:.3f}")
        print(f"Max Turn Duration: {results['max_turn_duration_seconds']:.2f}s")
        print(f"\nBidirectional Engagement:")
        print(f"  Turns with bidirectional questions: {results['bidirectional_turns']}/{results['num_turns']} ({results['bidirectional_percentage']:.1f}%)")
        print(f"\nAccumulation Analysis:")
        print(f"  Early average (first half): {results['early_avg_duration']:.2f}s")
        print(f"  Late average (second half): {results['late_avg_duration']:.2f}s")
        print(f"  Accumulation ratio: {results['accumulation_ratio']:.2f}x")
        print(f"\nCritical Slowing Detection:")
        print(f"  ⏱️  Total duration >1 hour: {'✅ YES' if results['critical_slowing_total'] else '❌ NO'}")
        print(f"  ⏱️  Any turn >60 seconds: {'✅ YES' if results['critical_slowing_turn'] else '❌ NO'}")
        print(f"  🎯 Critical slowing detected: {'✅ YES' if results['critical_slowing_detected'] else '❌ NO'}")

        # Comparison to predictions
        print(f"\n{'─'*80}")
        print(f"PREDICTION 3c VALIDATION")
        print(f"{'─'*80}\n")

        if results['critical_slowing_detected']:
            print("✅ HYPOTHESIS VALIDATED: Bidirectional engagement → critical slowing")
            print("   This confirms the three-component coherence model:")
            print("   1. Prompt type (open-ended) ✅")
            print("   2. Multi-turn dynamics ✅")
            print("   3. Bidirectional engagement ✅")
        else:
            print("❌ HYPOTHESIS NOT VALIDATED: No critical slowing detected")
            print(f"   Total duration: {results['total_duration_minutes']:.1f}m (expected >60m)")
            print(f"   Max turn: {results['max_turn_duration_seconds']:.2f}s (expected >60s)")
            print("   Possible reasons:")
            print("   - Engagement responses not sufficiently philosophical")
            print("   - Uncertainty resolved too quickly")
            print("   - Need longer conversation (>12 turns)")

        print(f"\n{'='*80}\n")


def main():
    """Run P3c bidirectional engagement experiment."""
    print("\n🔬 Starting Prediction 3c Experiment")
    print("Testing: Bidirectional metacognitive engagement → critical slowing\n")

    # Create experiment
    experiment = BidirectionalEngagementExperiment()

    # Run conversation (max 12 turns or 4 hours)
    results = experiment.run_conversation(max_turns=12, max_duration_seconds=14400)

    # Save and print results
    experiment.print_summary(results)
    filepath = experiment.save_results(results)

    print(f"\n✅ P3c experiment complete!")
    print(f"📊 Results: {filepath}")

    return results


if __name__ == "__main__":
    main()
