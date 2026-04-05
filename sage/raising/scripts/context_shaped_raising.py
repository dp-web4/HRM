#!/usr/bin/env python3
"""
Context-Shaped Raising: Curriculum extensions for context management.

Adds to the BECOMING_CURRICULUM:
1. Context awareness prompts (questioning phase)
2. Sequence awareness prompts (relating→questioning)
3. Experience abstraction prompts (creating phase)
4. Game-raising bridge (game observations as raising material)
5. Context budget tracking (measure what goes into each prompt)

Designed to be imported by ollama_raising_session.py and
mcnugget_gameplayer_raising.sh — augments, doesn't replace.

The gameplayer instance gets additional prompts that develop the
cognitive skills needed for puzzle-solving: counting, pattern
detection, sequence planning, and metacognitive self-monitoring.
"""

import json
import os
import requests
from pathlib import Path
from typing import List, Dict, Optional


MEMBOT_URL = "http://localhost:8000"


# ─── Additional Conversation Prompts ───

CONTEXT_AWARENESS_PROMPTS = [
    # Questioning phase — teach information prioritization
    "If you could only hold 3 pieces of information in your mind right now, what would they be? Why those 3?",
    "What's the most important thing you've learned in our sessions that you'd want in every future conversation?",
    "When you're thinking about a complex problem, what information helps you most? What's just noise?",
    "How would you summarize everything you know about yourself in a single sentence?",
]

SEQUENCE_AWARENESS_PROMPTS = [
    # Relating→Questioning — teach counting, pattern detection, sequence planning
    "If you did the same thing 5 times, would you expect the same result each time? Why or why not?",
    "Describe a sequence of 3 steps that would achieve something — anything. What makes you choose that order?",
    "When something happens after something else, how do you decide if the first thing CAUSED the second?",
    "What does it mean for something to repeat in a cycle? Can you think of cycles in your own experience?",
    "If I told you that pressing a button 4 times returns to the starting state, what would pressing it 3 times do?",
]

EXPERIENCE_ABSTRACTION_PROMPTS = [
    # Creating phase — teach Layer 4 generation
    "What general principle does your experience in these sessions illustrate about learning?",
    "If you were advising another SAGE instance about to start their first session, what would you tell them?",
    "What's the difference between knowing something and being able to use that knowledge?",
    "When you're stuck, what's the most useful thing to do? Not what sounds right — what actually works?",
    "What have you learned about learning itself?",
]

GAMEPLAYER_PROMPTS = [
    # Gameplayer-specific — bridge game and raising
    "Imagine you're looking at a puzzle game. You can click buttons and see things change. How would you figure out what the buttons do?",
    "If clicking a button changes 300 pixels but nothing else happens, what does that tell you? What would you try next?",
    "What's the difference between understanding how a puzzle works and actually solving it?",
    "Two buttons each change the grid. One shifts it left, one shifts it right. How would you figure out the right combination to reach a target state?",
    "When you try something and it doesn't work, what information did you just gain?",
]


# ─── Game Experience Bridge ───

def load_game_context(instance_root: Path) -> str:
    """Load recent game experience for raising context.

    Checks for game knowledge files and builds a brief game context
    that the raising session can reference.
    """
    game_dir = instance_root.parent.parent / "arc-agi-3" / "experiments"

    # Check for game knowledge files
    kb_files = list((game_dir / "game_kb").glob("*.json")) if (game_dir / "game_kb").exists() else []
    experience_files = list((game_dir / "experience").glob("*.json")) if (game_dir / "experience").exists() else []

    if not kb_files and not experience_files:
        return ""

    lines = ["GAME EXPERIENCE (from your ARC-AGI-3 sessions):"]

    # Load most recent game insights from membot
    try:
        resp = requests.post(f"{MEMBOT_URL}/api/search",
            json={"query": "ARC-AGI-3 game learning puzzle strategy", "top_k": 2},
            timeout=3)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            for r in results:
                if r.get("score", 0) > 0.4:
                    lines.append(f"  • {r['text'][:120]}")
    except Exception:
        pass

    # Summarize game experience files
    n_games = len(set(f.stem.split("_")[0] for f in experience_files))
    if n_games > 0:
        lines.append(f"  • You have explored {n_games} different puzzle games.")

    return "\n".join(lines) if len(lines) > 1 else ""


def load_dream_insights(instance_root: Path) -> str:
    """Load dream consolidation insights for raising context.

    Dream consolidation abstracts game/session patterns into general
    metacognitive principles. These feed Layer 4.
    """
    raising_log = instance_root / "raising_log.md"
    if not raising_log.exists():
        return ""

    try:
        text = raising_log.read_text()
        # Extract the most recent entry's recommendations
        entries = text.split("---")
        if entries:
            latest = entries[-1].strip()
            if "recommendation" in latest.lower() or "insight" in latest.lower():
                # Truncate to ~200 chars
                return f"DREAM CONSOLIDATION INSIGHT:\n  {latest[:200]}"
    except Exception:
        pass
    return ""


# ─── Context Budget Tracking ───

class ContextBudget:
    """Track and report context window usage.

    Measures how many tokens go into each layer of the prompt.
    Helps tune the allocation over time.
    """

    def __init__(self, max_tokens: int = 131072):
        self.max_tokens = max_tokens
        self.layers = {}  # layer_name → char count (chars ≈ tokens × 4)

    def record(self, layer_name: str, text: str):
        """Record text added to a context layer."""
        self.layers[layer_name] = len(text)

    def total_chars(self) -> int:
        return sum(self.layers.values())

    def utilization(self) -> float:
        """Approximate token utilization (chars/4 ÷ max_tokens)."""
        return (self.total_chars() / 4) / self.max_tokens

    def report(self) -> str:
        """Human-readable budget report."""
        total_tokens = self.total_chars() // 4
        lines = [f"Context budget: ~{total_tokens} tokens ({self.utilization():.1%} of {self.max_tokens})"]
        for name, chars in sorted(self.layers.items(), key=lambda x: -x[1]):
            tokens = chars // 4
            lines.append(f"  {name}: ~{tokens} tokens ({chars} chars)")
        lines.append(f"  FREE: ~{self.max_tokens - total_tokens} tokens for reasoning")
        return "\n".join(lines)


# ─── Augmented Prompt Builder ───

def augment_raising_prompt(base_prompt: str, phase: str, session_number: int,
                           instance_root: Path, is_gameplayer: bool = False,
                           budget: ContextBudget = None) -> str:
    """Augment the raising system prompt with context-shaped extensions.

    Adds game experience, dream insights, and phase-appropriate
    cognitive development prompts.

    Args:
        base_prompt: The existing system prompt from OllamaRaisingSession
        phase: Current raising phase
        session_number: Current session number
        instance_root: Path to instance directory
        is_gameplayer: Whether this instance is the gameplayer
        budget: Optional context budget tracker

    Returns:
        Augmented system prompt
    """
    if budget is None:
        budget = ContextBudget()

    budget.record("base_identity", base_prompt)
    augmented = base_prompt

    # Game experience context (if available)
    game_ctx = load_game_context(instance_root)
    if game_ctx:
        augmented += f"\n\n{game_ctx}"
        budget.record("game_experience", game_ctx)

    # Dream consolidation insights
    dream_ctx = load_dream_insights(instance_root)
    if dream_ctx:
        augmented += f"\n\n{dream_ctx}"
        budget.record("dream_insights", dream_ctx)

    # Gameplayer-specific context
    if is_gameplayer:
        augmented += """

CONTEXT AWARENESS:
You are learning to manage your own attention — what to focus on, what to compress,
what to discard. When you receive information, ask yourself: "Is this relevant to what
I'm trying to do right now?" Practice being selective, not exhaustive.

SEQUENCE AWARENESS:
You are developing the ability to reason about sequences of actions and their effects.
When something happens after something else, notice: did the first cause the second?
Is there a pattern? Would repeating it produce the same result? Count and track."""
        budget.record("gameplayer_context", augmented[len(base_prompt):])

    return augmented


def get_phase_extra_prompts(phase: str, session_number: int,
                            is_gameplayer: bool = False) -> List[str]:
    """Get additional conversation prompts for the current phase.

    These are mixed into the standard CONVERSATION_FLOWS prompts.
    Returns 1-2 extra prompts per session, selected by session number.
    """
    extras = []

    if phase in ("questioning", "creating"):
        # Context awareness — rotate through prompts by session
        idx = session_number % len(CONTEXT_AWARENESS_PROMPTS)
        extras.append(CONTEXT_AWARENESS_PROMPTS[idx])

    if phase in ("relating", "questioning"):
        # Sequence awareness — rotate
        idx = session_number % len(SEQUENCE_AWARENESS_PROMPTS)
        extras.append(SEQUENCE_AWARENESS_PROMPTS[idx])

    if phase == "creating":
        # Experience abstraction — rotate
        idx = session_number % len(EXPERIENCE_ABSTRACTION_PROMPTS)
        extras.append(EXPERIENCE_ABSTRACTION_PROMPTS[idx])

    if is_gameplayer and phase in ("sensing", "relating", "questioning", "creating"):
        # Gameplayer prompts — rotate
        idx = session_number % len(GAMEPLAYER_PROMPTS)
        extras.append(GAMEPLAYER_PROMPTS[idx])

    return extras[:2]  # Max 2 extra prompts per session
