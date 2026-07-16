"""
Crystallization evaluator — distinguishes pathological repetition from
stable propositions using contextual-fit-weighted repetition.

Batch-mode analysis on the experience buffer at consolidation time.
Not per-tick — the per-tick metacog detects in-session dysfunction;
this evaluator detects cross-session patterns.

Metric:
    crystallization_score(P) = frequency(P) × (1 - avg_contextual_fit(P))

A phrase repeating 149 times in identity contexts → low score (stable).
A phrase repeating 20 times leaking into creative writing → high score (pathology).

Design: fleet-learning/nomad/track-f-crystallization-evaluator-design.md
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Context-type classifier (Approach A: keyword-based, deterministic)
# --------------------------------------------------------------------------

CONTEXT_KEYWORDS: Dict[str, List[str]] = {
    "identity": [
        "who are you", "describe yourself", "your identity", "your name",
        "what are you", "tell me about yourself", "self-description",
        "your purpose", "your role", "your function", "as sage",
        "identity", "who you are", "introduce yourself",
    ],
    "partnership": [
        "how do we work", "collaboration", "together", "relationship",
        "our work", "partnership", "collective", "team", "fleet",
        "how we collaborate", "working with",
    ],
    "task": [
        "fix", "implement", "write code", "debug", "build", "create",
        "update", "change", "modify", "refactor", "deploy", "test",
        "analyze", "compute", "calculate", "solve",
    ],
    "creative": [
        "write a story", "imagine", "fiction", "poem", "creative",
        "dragon", "adventure", "fantasy", "scenario", "hypothetical",
        "once upon", "pretend",
    ],
    "meta": [
        "what have you learned", "reflect", "session", "progress",
        "observation", "insight", "discovery", "pattern", "notice",
        "what changed", "what surprised",
    ],
}


def classify_context(prompt: str) -> str:
    """Classify a prompt into one of 5 context types by keyword matching."""
    prompt_lower = prompt.lower()
    scores: Dict[str, int] = {}
    for ctx_type, keywords in CONTEXT_KEYWORDS.items():
        scores[ctx_type] = sum(1 for kw in keywords if kw in prompt_lower)
    if not any(scores.values()):
        return "task"  # default
    return max(scores, key=scores.get)


# Context types where identity/partnership phrases have high fit
HIGH_FIT_CONTEXTS = {"identity", "partnership", "meta"}
MEDIUM_FIT_CONTEXTS = {"task"}
LOW_FIT_CONTEXTS = {"creative"}


def contextual_fit(context_type: str) -> float:
    """How appropriate is an identity-related phrase in this context type?"""
    if context_type in HIGH_FIT_CONTEXTS:
        return 1.0
    if context_type in MEDIUM_FIT_CONTEXTS:
        return 0.5
    return 0.0  # creative, other


# --------------------------------------------------------------------------
# N-gram extraction
# --------------------------------------------------------------------------


def extract_ngrams(text: str, n: int) -> List[str]:
    """Extract word-level n-grams from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------


@dataclass
class CrystallizationVerdict:
    phrase: str
    frequency: int
    context_type_distribution: Dict[str, int]
    avg_contextual_fit: float
    crystallization_score: float
    verdict: str  # "crystallization" | "stable_proposition" | "inconclusive"
    appearances: int = 0  # total contexts checked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase,
            "frequency": self.frequency,
            "context_types": self.context_type_distribution,
            "avg_fit": round(self.avg_contextual_fit, 4),
            "score": round(self.crystallization_score, 4),
            "verdict": self.verdict,
        }


@dataclass
class CrystallizationConfig:
    n_gram_sizes: List[int] = field(default_factory=lambda: [4, 5, 6])
    min_frequency: int = 5
    crystallization_threshold: float = 10.0  # empirical — calibrate on Sprout data
    stable_threshold: float = 3.0  # below this → stable proposition


class CrystallizationEvaluator:
    """Batch evaluator for cross-session repetition patterns."""

    def __init__(self, config: Optional[CrystallizationConfig] = None):
        self.config = config or CrystallizationConfig()

    def evaluate_buffer(
        self, experiences: List[Dict[str, Any]]
    ) -> List[CrystallizationVerdict]:
        """Analyze an experience buffer for crystallization patterns.

        Each experience should have 'prompt' and 'response' fields.
        Returns verdicts sorted by crystallization_score descending.
        """
        # Step 1: extract all n-grams from responses + classify contexts
        phrase_contexts: Dict[str, List[str]] = defaultdict(list)

        for exp in experiences:
            prompt = exp.get("prompt", "")
            response = exp.get("response", "")
            ctx_type = classify_context(prompt)

            for n in self.config.n_gram_sizes:
                for ngram in extract_ngrams(response, n):
                    phrase_contexts[ngram].append(ctx_type)

        # Step 2: filter by frequency
        frequent = {
            phrase: contexts
            for phrase, contexts in phrase_contexts.items()
            if len(contexts) >= self.config.min_frequency
        }

        # Step 3: compute scores
        verdicts: List[CrystallizationVerdict] = []
        for phrase, contexts in frequent.items():
            ctx_dist = Counter(contexts)
            fits = [contextual_fit(ctx) for ctx in contexts]
            avg_fit = sum(fits) / len(fits) if fits else 0.5
            freq = len(contexts)
            score = freq * (1.0 - avg_fit)

            if score >= self.config.crystallization_threshold:
                verdict = "crystallization"
            elif score <= self.config.stable_threshold:
                verdict = "stable_proposition"
            else:
                verdict = "inconclusive"

            verdicts.append(CrystallizationVerdict(
                phrase=phrase,
                frequency=freq,
                context_type_distribution=dict(ctx_dist),
                avg_contextual_fit=avg_fit,
                crystallization_score=score,
                verdict=verdict,
                appearances=len(contexts),
            ))

        # Sort by score descending
        verdicts.sort(key=lambda v: v.crystallization_score, reverse=True)
        return verdicts

    def evaluate_phrase(
        self,
        phrase: str,
        experiences: List[Dict[str, Any]],
    ) -> CrystallizationVerdict:
        """Evaluate a specific phrase against the buffer."""
        phrase_lower = phrase.lower()
        contexts: List[str] = []

        for exp in experiences:
            response = exp.get("response", "").lower()
            if phrase_lower in response:
                ctx_type = classify_context(exp.get("prompt", ""))
                contexts.append(ctx_type)

        if not contexts:
            return CrystallizationVerdict(
                phrase=phrase, frequency=0,
                context_type_distribution={},
                avg_contextual_fit=0.5,
                crystallization_score=0.0,
                verdict="inconclusive",
            )

        ctx_dist = Counter(contexts)
        fits = [contextual_fit(ctx) for ctx in contexts]
        avg_fit = sum(fits) / len(fits)
        freq = len(contexts)
        score = freq * (1.0 - avg_fit)

        cfg = self.config
        if score >= cfg.crystallization_threshold:
            verdict = "crystallization"
        elif score <= cfg.stable_threshold:
            verdict = "stable_proposition"
        else:
            verdict = "inconclusive"

        return CrystallizationVerdict(
            phrase=phrase,
            frequency=freq,
            context_type_distribution=dict(ctx_dist),
            avg_contextual_fit=avg_fit,
            crystallization_score=score,
            verdict=verdict,
            appearances=len(contexts),
        )
