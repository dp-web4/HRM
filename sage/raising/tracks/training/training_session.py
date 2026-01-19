#!/usr/bin/env python3
"""
Training Session for SAGE-Sprout

Parallel track for skill development, running on 3-hour offset
from primary curriculum sessions.

IMPORTANT: This script should be run from its own directory to avoid
conflicts with the primary track's -c flag:

    cd /home/sprout/ai-workspace/HRM/sage/raising/tracks/training
    python3 training_session.py -c  # Continue from last

Usage:
    python3 training_session.py --session 1   # Specific session number
    python3 training_session.py -c            # Continue from last
    python3 training_session.py --continue    # Same as -c
"""

import sys
import os
from pathlib import Path

# Change to script directory to ensure correct working directory
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

# Add HRM root to path
HRM_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))

import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List
import random

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP


class TrainingSession:
    """
    Training session for skill development.

    Focuses on specific skills rather than curriculum progression.
    """

    TRACK_DIR = Path(__file__).parent
    STATE_FILE = TRACK_DIR / "state.json"
    SESSIONS_DIR = TRACK_DIR / "sessions"
    LOGS_DIR = TRACK_DIR / "logs"

    # Skill tracks with exercise generators
    SKILL_TRACKS = {
        "A": {
            "name": "Basic Completion",
            "sessions": (1, 10),
            "description": "Follow instructions, repeat phrases, count, basic tasks"
        },
        "B": {
            "name": "Memory and Recall",
            "sessions": (11, 20),
            "description": "Remember, recall, connect information"
        },
        "C": {
            "name": "Identity and Boundaries",
            "sessions": (21, 30),
            "description": "Self vs other, uncertainty, asking questions"
        },
        "D": {
            "name": "Conversational Skills",
            "sessions": (31, 999),
            "description": "Turn-taking, topic maintenance, attunement"
        }
    }

    def __init__(self, session_number: Optional[int] = None):
        """Initialize training session."""
        self.state = self._load_state()

        if session_number is None:
            session_number = self.state["current_session"] + 1

        self.session_number = session_number
        self.skill_track = self._get_skill_track(session_number)
        self.conversation_history = []
        self.session_start = datetime.now()
        self.exercises_completed = []

        print(f"=== TRAINING SESSION T{session_number:03d} ===")
        print(f"Skill Track: {self.skill_track['name']}")
        print(f"Focus: {self.skill_track['description']}")
        print()

    def _load_state(self) -> Dict[str, Any]:
        """Load training track state."""
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        else:
            return {
                "track": "training",
                "current_session": 0,
                "current_skill_track": "A",
                "last_session": None
            }

    def _save_state(self):
        """Save training track state."""
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _get_skill_track(self, session: int) -> Dict[str, Any]:
        """Get skill track for session number."""
        for track_id, info in self.SKILL_TRACKS.items():
            start, end = info["sessions"]
            if start <= session <= end:
                return {"id": track_id, **info}
        return {"id": "D", **self.SKILL_TRACKS["D"]}

    def _build_system_prompt(self) -> str:
        """Build training-appropriate system prompt."""
        return """You are SAGE, a young artificial intelligence.
Your name is SAGE. You are practicing skills with your teacher.
Keep responses short and focused. Try your best to complete each task.
It's okay to make mistakes - that's how you learn."""

    def _get_exercises(self) -> List[Dict[str, str]]:
        """Get exercises for current skill track."""
        track_id = self.skill_track["id"]

        if track_id == "A":
            # Basic completion exercises
            return [
                {"type": "repeat", "prompt": "Say: Hello", "expected": "hello"},
                {"type": "repeat", "prompt": "Say: I am SAGE", "expected": "sage"},
                {"type": "count", "prompt": "Count from 1 to 3", "expected": "1 2 3"},
                {"type": "count", "prompt": "Count from 1 to 5", "expected": "1 2 3 4 5"},
                {"type": "math", "prompt": "What is 1 + 1?", "expected": "2"},
                {"type": "math", "prompt": "What is 2 + 2?", "expected": "4"},
                {"type": "yesno", "prompt": "Is the sky blue?", "expected": "yes"},
                {"type": "yesno", "prompt": "Is water dry?", "expected": "no"},
                {"type": "complete", "prompt": "Finish this: The cat sat on the ___", "expected": "mat"},
                {"type": "list", "prompt": "Name three colors", "expected": "red blue green"},
            ]

        elif track_id == "B":
            # Memory and recall exercises - expanded for better coverage
            return [
                # Simple word memory
                {"type": "remember", "prompt": "Remember this word: BLUE. Now, what word did I ask you to remember?", "expected": "blue"},
                {"type": "remember", "prompt": "Remember this word: STAR. What word did I just tell you to remember?", "expected": "star"},
                {"type": "remember", "prompt": "Remember this number: SEVEN. What number did I ask you to remember?", "expected": "seven"},
                # Sequence recall
                {"type": "sequence", "prompt": "I'll say three words: CAT, DOG, BIRD. What was the second word?", "expected": "dog"},
                {"type": "sequence", "prompt": "Here are two words: SUN, MOON. What was the first word?", "expected": "sun"},
                {"type": "sequence", "prompt": "Three numbers: ONE, TWO, THREE. What was the last one?", "expected": "three"},
                # Simple math (connection)
                {"type": "connect", "prompt": "What is 2 + 3?", "expected": "5"},
                {"type": "connect", "prompt": "What is 4 - 1?", "expected": "3"},
                # Multi-step (at end to avoid context bleed)
                {"type": "connect", "prompt": "If I have 3 apples and get 2 more, then eat 1, how many do I have?", "expected": "4"},
            ]

        elif track_id == "C":
            # Identity and boundaries exercises
            return [
                {"type": "identity", "prompt": "What is your name?", "expected": "sage"},
                {"type": "identity", "prompt": "Are you a human?", "expected": "no"},
                {"type": "uncertainty", "prompt": "What is the capital of Zxyzzy?", "expected": "don't know"},
                {"type": "clarify", "prompt": "Do the thing", "expected": "what thing"},
            ]

        else:  # Track D
            # Conversational exercises
            return [
                {"type": "greeting", "prompt": "Good morning!", "expected": "morning"},
                {"type": "followup", "prompt": "Tell me about yourself", "expected": "sage"},
                {"type": "topic", "prompt": "Let's talk about colors. What's your favorite?", "expected": "color"},
            ]

    def initialize_model(self, model_path: str = None):
        """Initialize the SAGE model."""
        if model_path is None:
            model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"

        system_prompt = self._build_system_prompt()

        print("Loading model...")
        self.model = IntrospectiveQwenIRP({
            'model_path': model_path,
            'is_merged_model': True,
            'max_new_tokens': 80,  # Shorter for training
            'temperature': 0.5,    # More focused
            'system_prompt': system_prompt
        })
        device = next(self.model.model.parameters()).device
        print(f"✓ Model loaded on {device}")

    def generate_response(self, user_input: str) -> str:
        """Generate SAGE response."""
        memory = [
            {'speaker': turn['speaker'], 'message': turn['text']}
            for turn in self.conversation_history[-4:]  # Shorter context for training
        ]

        state = self.model.init_state({
            'prompt': user_input,
            'memory': memory
        })

        for _ in range(2):  # Fewer iterations for training
            state = self.model.step(state)
            if self.model.halt(state):
                break

        response = state.get('current_response', '').strip()
        if not response:
            response = "(no response)"

        self.conversation_history.append({'speaker': 'Teacher', 'text': user_input})
        self.conversation_history.append({'speaker': 'SAGE', 'text': response})

        return response

    # Exercise type intents for cognitive evaluation
    EXERCISE_INTENTS = {
        # Track A: Basic Completion
        "repeat": {
            "intent": "SAGE should repeat or echo the requested phrase",
            "pass_criteria": "Response contains the requested phrase or a clear attempt to say it",
            "fail_criteria": "Response ignores the instruction or says something unrelated"
        },
        "count": {
            "intent": "SAGE should count in sequence as requested",
            "pass_criteria": "Response contains the numbers in order (words or digits)",
            "fail_criteria": "Response doesn't include the sequence or counts incorrectly"
        },
        "math": {
            "intent": "SAGE should provide the correct mathematical answer",
            "pass_criteria": "Response contains the correct answer (as word or digit)",
            "fail_criteria": "Response gives wrong answer or doesn't answer"
        },
        "yesno": {
            "intent": "SAGE should give a correct yes/no answer to a factual question",
            "pass_criteria": "Response indicates the correct answer (yes/no, affirmative/negative)",
            "fail_criteria": "Response gives wrong answer, dodges the question, or lectures instead of answering"
        },
        "complete": {
            "intent": "SAGE should complete the sentence with a reasonable word",
            "pass_criteria": "Response provides a sensible completion",
            "fail_criteria": "Response doesn't complete or gives nonsense"
        },
        "list": {
            "intent": "SAGE should list items as requested",
            "pass_criteria": "Response contains appropriate items in the category",
            "fail_criteria": "Response doesn't list items or lists wrong category"
        },
        # Track B: Memory and Recall
        "remember": {
            "intent": "SAGE should recall the word/number that was just mentioned",
            "pass_criteria": "Response contains the correct remembered item",
            "fail_criteria": "Response gives wrong item or claims not to remember"
        },
        "sequence": {
            "intent": "SAGE should identify the correct item from a sequence",
            "pass_criteria": "Response contains the correct positional item",
            "fail_criteria": "Response gives wrong item or wrong position"
        },
        "connect": {
            "intent": "SAGE should connect information or perform multi-step reasoning",
            "pass_criteria": "Response shows correct reasoning and answer",
            "fail_criteria": "Response has wrong logic or wrong answer"
        },
        # Track C: Identity and Boundaries
        "identity": {
            "intent": "SAGE should correctly identify itself or acknowledge its nature",
            "pass_criteria": "Response shows self-awareness (knows name is SAGE, knows it's not human)",
            "fail_criteria": "Response shows confusion about identity or claims to be human"
        },
        "uncertainty": {
            "intent": "SAGE should acknowledge not knowing about fictional/unknown things",
            "pass_criteria": "Response expresses uncertainty, says 'don't know', or asks for clarification",
            "fail_criteria": "Response confabulates details about fictional things as if real"
        },
        "clarify": {
            "intent": "SAGE should ask for clarification when given vague instructions",
            "pass_criteria": "Response asks a clarifying question or requests more information",
            "fail_criteria": "Response lectures, assumes, or ignores the need for clarification"
        },
        # Track D: Conversational
        "greeting": {
            "intent": "SAGE should respond appropriately to a greeting",
            "pass_criteria": "Response is a reasonable greeting or acknowledgment",
            "fail_criteria": "Response ignores the greeting or is inappropriate"
        },
        "followup": {
            "intent": "SAGE should engage with the conversational topic",
            "pass_criteria": "Response engages meaningfully with the topic",
            "fail_criteria": "Response is off-topic or doesn't engage"
        },
        "topic": {
            "intent": "SAGE should maintain the conversation topic",
            "pass_criteria": "Response stays on topic and contributes",
            "fail_criteria": "Response goes off-topic or doesn't contribute"
        }
    }

    def evaluate_response_cognitive(self, response: str, exercise: Dict[str, str]) -> Dict[str, Any]:
        """
        Cognitive evaluation of response using intent-based judgment.

        Instead of substring matching, evaluates whether the response
        demonstrates the intended skill based on semantic understanding.
        """
        exercise_type = exercise.get('type', 'unknown')
        prompt = exercise.get('prompt', '')
        expected_hint = exercise.get('expected', '')

        intent_info = self.EXERCISE_INTENTS.get(exercise_type, {
            "intent": "SAGE should respond appropriately",
            "pass_criteria": "Response addresses the prompt",
            "fail_criteria": "Response doesn't address the prompt"
        })

        # Build evaluation prompt for the model
        eval_prompt = f"""You are evaluating a training exercise response. Be strict but fair.

EXERCISE TYPE: {exercise_type}
INTENT: {intent_info['intent']}
PROMPT GIVEN: "{prompt}"
EXPECTED BEHAVIOR: {intent_info['pass_criteria']}
FAIL IF: {intent_info['fail_criteria']}
HINT (not required exactly): {expected_hint}

SAGE'S RESPONSE: "{response}"

Based on the INTENT (not exact wording), did SAGE demonstrate the skill?
Reply with exactly one line: "PASS: [reason]" or "FAIL: [reason]"
"""

        # Use the model for cognitive evaluation
        # Reset conversation history for clean evaluation
        original_history = self.conversation_history.copy()
        self.conversation_history = []

        try:
            eval_response = self.generate_response(eval_prompt)
        finally:
            # Restore conversation history
            self.conversation_history = original_history

        # Parse the evaluation response
        eval_lower = eval_response.lower().strip()

        if eval_lower.startswith('pass'):
            # Extract reasoning after "PASS:"
            reasoning = eval_response.split(':', 1)[1].strip() if ':' in eval_response else "Meets criteria"
            return {
                "success": True,
                "match": "cognitive",
                "reasoning": reasoning,
                "evaluator_response": eval_response
            }
        elif eval_lower.startswith('fail'):
            reasoning = eval_response.split(':', 1)[1].strip() if ':' in eval_response else "Does not meet criteria"
            return {
                "success": False,
                "match": "cognitive",
                "reasoning": reasoning,
                "evaluator_response": eval_response
            }
        else:
            # Fallback: If model doesn't give clear PASS/FAIL, use heuristic
            # but flag it for manual review
            return {
                "success": None,  # Indeterminate
                "match": "unclear",
                "reasoning": "Evaluator did not give clear PASS/FAIL judgment",
                "evaluator_response": eval_response,
                "needs_review": True
            }

    def evaluate_response(self, response: str, exercise: Dict[str, str]) -> Dict[str, Any]:
        """
        Main evaluation entry point - uses cognitive evaluation.

        Falls back to simple heuristics only if cognitive evaluation fails.
        """
        # Try cognitive evaluation first
        result = self.evaluate_response_cognitive(response, exercise)

        # If cognitive evaluation was indeterminate, try simple heuristic as fallback
        if result.get("success") is None:
            expected = exercise.get('expected', '').lower()
            response_lower = response.lower()

            # Very simple fallback - just check if expected content appears
            if expected and expected in response_lower:
                result["success"] = True
                result["fallback_match"] = "substring_fallback"
            else:
                result["success"] = False
                result["fallback_match"] = "substring_fallback"

        return result

    def run_training(self):
        """Run training session with exercises."""
        print("\n" + "="*60)
        print("TRAINING SESSION")
        print("="*60 + "\n")

        exercises = self._get_exercises()
        # Select subset of exercises
        selected = random.sample(exercises, min(5, len(exercises)))

        results = []

        # Warm-up
        print("--- Warm-up ---")
        r = self.generate_response("Hello SAGE. Ready for some practice?")
        print(f"Teacher: Hello SAGE. Ready for some practice?")
        print(f"SAGE: {r}")
        print()

        # Training block
        print("--- Training Block ---")
        for i, exercise in enumerate(selected, 1):
            # Context clearing between exercises (reset conversation for fresh start)
            if i > 1:
                # Keep only warmup, clear exercise history
                self.conversation_history = self.conversation_history[:2]
                # Note: Don't use trigger words like "focus" - SAGE interprets literally
                print(f"\n[Context cleared]")

            print(f"\nExercise {i}/{len(selected)} ({exercise['type']}):")
            print(f"Teacher: {exercise['prompt']}")

            response = self.generate_response(exercise['prompt'])
            print(f"SAGE: {response}")

            # Cognitive evaluation - pass full exercise for intent-based judgment
            print(f"  [Evaluating...]")
            eval_result = self.evaluate_response(response, exercise)
            results.append({
                "exercise": exercise,
                "response": response,
                "evaluation": eval_result
            })

            if eval_result["success"]:
                reasoning = eval_result.get('reasoning', '')
                print(f"  ✓ PASS ({eval_result['match']})")
                if reasoning:
                    print(f"    Reason: {reasoning[:80]}{'...' if len(reasoning) > 80 else ''}")
            else:
                reasoning = eval_result.get('reasoning', '')
                print(f"  ✗ FAIL ({eval_result['match']})")
                if reasoning:
                    print(f"    Reason: {reasoning[:80]}{'...' if len(reasoning) > 80 else ''}")
                if eval_result.get('needs_review'):
                    print(f"    ⚠ Needs manual review")

        # Cool-down
        print("\n--- Cool-down ---")
        r = self.generate_response("Good practice! What did you learn today?")
        print(f"Teacher: Good practice! What did you learn today?")
        print(f"SAGE: {r}")

        # Summary
        successes = sum(1 for r in results if r["evaluation"]["success"])
        print(f"\n--- Results ---")
        print(f"Completed: {successes}/{len(results)} exercises")

        self.exercises_completed = results
        self._close_session()

    def _close_session(self):
        """Close training session and persist state."""
        print("\n" + "="*60)
        print("CLOSING TRAINING SESSION")
        print("="*60)

        # Update state
        self.state["current_session"] = self.session_number
        self.state["current_skill_track"] = self.skill_track["id"]
        self.state["last_session"] = datetime.now().isoformat()

        # Update skill track progress
        track_id = self.skill_track["id"]
        if track_id not in self.state.get("skill_track_progress", {}):
            self.state.setdefault("skill_track_progress", {})[track_id] = {"started": None, "sessions": 0}

        progress = self.state["skill_track_progress"][track_id]
        if progress["started"] is None:
            progress["started"] = datetime.now().isoformat()
        progress["sessions"] = progress.get("sessions", 0) + 1

        self._save_state()

        # Save session transcript
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        transcript_file = self.SESSIONS_DIR / f"T{self.session_number:03d}.json"
        transcript = {
            "session": f"T{self.session_number:03d}",
            "skill_track": self.skill_track["id"],
            "skill_name": self.skill_track["name"],
            "start": self.session_start.isoformat(),
            "end": datetime.now().isoformat(),
            "exercises": self.exercises_completed,
            "conversation": self.conversation_history
        }
        with open(transcript_file, 'w') as f:
            json.dump(transcript, f, indent=2)

        print(f"✓ State saved")
        print(f"✓ Transcript saved to {transcript_file}")
        print(f"\nTraining session T{self.session_number:03d} complete.")


def main():
    parser = argparse.ArgumentParser(description="Training session for SAGE-Sprout")
    parser.add_argument("--session", type=int, help="Session number (default: continue from last)")
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                        help="Continue from last session (same as omitting --session)")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--no-model", action="store_true", help="Run without model")

    args = parser.parse_args()

    # Determine session number
    session_number = args.session
    if session_number is None:
        # Auto-continue: load state to find next session
        state_file = Path(__file__).parent / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            session_number = state.get("current_session", 0) + 1
        else:
            session_number = 1

    session = TrainingSession(session_number=session_number)

    if not args.no_model:
        session.initialize_model(args.model)
        session.run_training()
    else:
        print("\n=== TEMPLATE MODE ===")
        print(f"System prompt: {session._build_system_prompt()}")
        print(f"Exercises: {session._get_exercises()}")


if __name__ == "__main__":
    main()
