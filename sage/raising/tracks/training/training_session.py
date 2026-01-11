#!/usr/bin/env python3
"""
Training Session for SAGE-Sprout

Parallel track for skill development, running on 3-hour offset
from primary curriculum sessions.

Usage:
    python3 training_session.py --session T001
    python3 training_session.py --continue  # Continue from last
"""

import sys
from pathlib import Path
# Add HRM root to path
HRM_ROOT = Path(__file__).parent.parent.parent.parent.parent
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
            # Memory and recall exercises
            return [
                {"type": "remember", "prompt": "Remember this word: APPLE. Now, what word did I ask you to remember?", "expected": "apple"},
                {"type": "sequence", "prompt": "I'll say three words: CAT, DOG, BIRD. What was the second word?", "expected": "dog"},
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

    def evaluate_response(self, response: str, expected: str) -> Dict[str, Any]:
        """Evaluate if response matches expected content."""
        response_lower = response.lower()
        expected_lower = expected.lower()

        # Check if expected content is present
        if expected_lower in response_lower:
            return {"success": True, "match": "exact"}

        # Check for partial matches
        expected_words = expected_lower.split()
        matches = sum(1 for word in expected_words if word in response_lower)
        if matches >= len(expected_words) * 0.5:
            return {"success": True, "match": "partial", "matched": matches, "total": len(expected_words)}

        return {"success": False, "match": "none"}

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
            print(f"\nExercise {i}/{len(selected)} ({exercise['type']}):")
            print(f"Teacher: {exercise['prompt']}")

            response = self.generate_response(exercise['prompt'])
            print(f"SAGE: {response}")

            eval_result = self.evaluate_response(response, exercise['expected'])
            results.append({
                "exercise": exercise,
                "response": response,
                "evaluation": eval_result
            })

            if eval_result["success"]:
                print(f"  ✓ Good ({eval_result['match']})")
            else:
                print(f"  ✗ Expected something like: {exercise['expected']}")

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
    parser.add_argument("--session", type=int, help="Session number (default: continue)")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--no-model", action="store_true", help="Run without model")

    args = parser.parse_args()

    session = TrainingSession(session_number=args.session)

    if not args.no_model:
        session.initialize_model(args.model)
        session.run_training()
    else:
        print("\n=== TEMPLATE MODE ===")
        print(f"System prompt: {session._build_system_prompt()}")
        print(f"Exercises: {session._get_exercises()}")


if __name__ == "__main__":
    main()
