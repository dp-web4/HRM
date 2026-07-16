"""Grounded reasoning metrics for LLM dispatch (Phase 4 B6).

Three metrics capture whether the LLM was reasoning about real things,
not articulately hallucinating:

- **entity_validity_rate**: fraction of entities referenced in the
  rationale that are plausible (appear in world model vocabulary,
  are valid coords, are game-action names)
- **vocabulary_correctness**: fraction of rationale tokens that
  intersect the world model vocabulary (domain engagement)
- **mechanics_alignment**: did the LLM's implicit prediction match
  the observed post-action frame delta

Heuristic first pass — no NLP libraries required. Regex + token-set
operations. Honest signal at minimal compute cost (<5ms per invoke).

Spec: grounded_reasoning_metrics.md
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Tokens that aren't informative for domain-engagement measurement
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "and", "or", "but", "not", "no", "so", "if", "then", "else",
    "to", "from", "of", "in", "on", "at", "by", "with", "for",
    "as", "than", "up", "down", "left", "right", "here", "there",
    "will", "would", "should", "could", "can", "may", "might",
    "have", "has", "had", "do", "does", "did",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "first", "second", "third", "last", "next", "previous",
    "action", "step", "steps", "level", "frame", "grid",  # SAGE-universal
    # Strategy-abstraction words — not domain vocabulary, don't count
    "try", "tried", "trying", "approach", "different", "current",
    "working", "working", "now", "still", "because", "since",
    "before", "after", "isn", "aren", "wasn", "weren", "haven", "hasn",
    "wouldn", "couldn", "shouldn", "won", "don", "doesn", "didn",
    "way", "ways", "thing", "things", "something", "anything", "nothing",
    "good", "bad", "better", "worse", "best", "worst",
}

# Colors are valid entities in any game
_COLORS: Set[str] = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink",
    "brown", "black", "white", "gray", "grey", "cyan", "magenta",
    "teal", "maroon", "lime", "olive", "navy", "silver", "gold",
}

# Game actions are valid entities
_ACTION_NAMES: Set[str] = {
    "a0", "up", "down", "left", "right", "sel", "select", "click",
    "noop",
}

# Generic game-domain terms that are always valid
_GENERIC_DOMAIN: Set[str] = {
    "target", "goal", "piece", "block", "cell", "position", "coord",
    "coordinate", "tile", "border", "edge", "corner", "center",
    "player", "avatar", "agent", "cursor",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens (alphanumeric)."""
    return re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text.lower())


def _content_tokens(text: str) -> Set[str]:
    """Tokens minus stopwords."""
    return {t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def world_model_vocabulary(world_model_text: str) -> Set[str]:
    """Extract domain vocabulary from the world model markdown."""
    toks = _content_tokens(world_model_text)
    return toks


def extract_entity_references(rationale: str) -> List[str]:
    """Extract noun-like tokens from the rationale.

    Simple heuristic: content tokens excluding verbs/adjectives common in
    game rationales. Returns unique entities; counts duplicates at caller.
    """
    # Remove action verb chatter first
    cleaned = re.sub(
        r"\b(click|move|pick|select|press|choose|try|test|check|verify|want|need)\b",
        "", rationale.lower()
    )
    # Coord patterns are valid entities
    entities: List[str] = []
    for m in re.finditer(r"\(\s*(\d+)\s*[,x]\s*(\d+)\s*\)", rationale):
        x, y = int(m.group(1)), int(m.group(2))
        if 0 <= x <= 63 and 0 <= y <= 63:
            entities.append(f"coord({x},{y})")

    # Remaining word entities
    toks = [t for t in _tokenize(cleaned) if t not in _STOPWORDS and len(t) > 2]
    entities.extend(toks)
    return entities


def entity_validity_rate(
    rationale: str, world_model_text: str = "",
) -> float:
    """Fraction of rationale entities that are plausible.

    Plausible = coord in grid, color name, action name, generic domain
    term, or appears in world-model vocabulary.

    Returns 0.0 for empty rationale.
    """
    entities = extract_entity_references(rationale)
    if not entities:
        return 0.0

    wm_vocab = world_model_vocabulary(world_model_text) if world_model_text else set()
    valid_set = _COLORS | _ACTION_NAMES | _GENERIC_DOMAIN | wm_vocab

    valid_count = 0
    for e in entities:
        if e.startswith("coord("):
            valid_count += 1
        elif e in valid_set:
            valid_count += 1

    return valid_count / len(entities)


def vocabulary_correctness(
    rationale: str, world_model_text: str,
) -> float:
    """Fraction of rationale content tokens that intersect the world
    model vocabulary.

    Measures domain engagement — is the LLM using terms from this game's
    vocabulary, or generic pattern-matching filler.

    Returns 0.0 when either side is empty.
    """
    if not world_model_text or not rationale:
        return 0.0
    rat_tokens = _content_tokens(rationale)
    wm_tokens = world_model_vocabulary(world_model_text)
    if not rat_tokens:
        # Rationale had no content tokens (all stopwords / strategy-abstractions).
        # That's generic but not wrong — neutral.
        return 0.5
    # Always-valid domain terms (colors, actions, generics) don't count
    # toward game-specific engagement but don't penalize either.
    scoring_tokens = rat_tokens - _COLORS - _ACTION_NAMES - _GENERIC_DOMAIN
    if not scoring_tokens:
        return 0.5   # neutral — rationale was generic but not wrong
    overlap = scoring_tokens & wm_tokens
    return len(overlap) / len(scoring_tokens)


_PREDICTION_HIGH_CHANGE = re.compile(
    r"\b(will|should|to)\s+(advance|clear|move|launch|toggle|activate|"
    r"break|destroy|complete|win|solve)\b", re.IGNORECASE
)
_PREDICTION_LOW_CHANGE = re.compile(
    r"\b(test|testing|explore|exploring|check|checking|verify|verifying|"
    r"observe|observing|probe|probing|no.effect|nothing)\b", re.IGNORECASE
)


def mechanics_alignment(
    rationale: str, observed_frame_delta_pct: float,
    level_advanced: bool = False,
) -> Optional[float]:
    """Did the LLM's predicted effect match the observed effect?

    Returns None when no prediction detected (can't score). Returns 0.0–1.0
    otherwise:
      - 1.0: prediction matched observation
      - 0.0: prediction was opposite of observation
      - 0.5: ambiguous / partial match

    Rules:
      - If rationale predicts advance/win → high delta or level change scores 1.0
      - If rationale predicts test/explore → low delta scores 1.0
      - Mismatches score 0.0–0.3
    """
    if not rationale:
        return None

    predicts_high = _PREDICTION_HIGH_CHANGE.search(rationale) is not None
    predicts_low = _PREDICTION_LOW_CHANGE.search(rationale) is not None

    if not predicts_high and not predicts_low:
        return None

    if level_advanced:
        # Any prediction that mentioned advance-like language gets full credit
        # Low-change prediction + advance is genuinely wrong (didn't expect win)
        return 1.0 if predicts_high else 0.3

    # No level advance — score by frame delta
    high_delta = observed_frame_delta_pct >= 20.0
    moderate_delta = observed_frame_delta_pct >= 5.0
    low_delta = observed_frame_delta_pct < 2.0

    if predicts_high and predicts_low:
        # Rationale hedged both ways — low information
        return 0.5
    if predicts_high:
        if high_delta:
            return 1.0
        elif moderate_delta:
            return 0.7
        elif low_delta:
            return 0.0
        return 0.5
    # predicts_low
    if low_delta:
        return 1.0
    elif moderate_delta:
        return 0.6
    elif high_delta:
        # Predicted exploratory but got major change — unexpected, not wrong
        return 0.4
    return 0.5


def compute_grounded_metrics(
    rationale: str, world_model_text: str,
    observed_frame_delta_pct: float,
    level_advanced: bool = False,
) -> Dict[str, Any]:
    """Compute all three grounded reasoning metrics for one LLM call.

    Returns dict with:
      - entity_validity_rate: 0.0-1.0
      - vocabulary_correctness: 0.0-1.0
      - mechanics_alignment: 0.0-1.0 or None if no prediction detected
      - grounded_pass: bool (all three above thresholds; None counts as pass)
    """
    ev = entity_validity_rate(rationale, world_model_text)
    vc = vocabulary_correctness(rationale, world_model_text)
    ma = mechanics_alignment(rationale, observed_frame_delta_pct, level_advanced)

    # Grounded pass thresholds per spec:
    #   entity_validity_rate >= 0.7
    #   vocabulary_correctness >= 0.7
    #   mechanics_alignment >= 0.5  (None = pass)
    passed = (
        ev >= 0.7 and
        vc >= 0.7 and
        (ma is None or ma >= 0.5)
    )
    return {
        "entity_validity_rate": ev,
        "vocabulary_correctness": vc,
        "mechanics_alignment": ma,
        "grounded_pass": passed,
    }


def aggregate_grounded_metrics(
    per_invoke_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute sweep-level aggregates from per-invoke metrics."""
    if not per_invoke_metrics:
        return {
            "mean_entity_validity_rate": 0.0,
            "mean_vocabulary_correctness": 0.0,
            "mean_mechanics_alignment": None,
            "grounded_pass_count": 0,
            "grounded_pass_rate": 0.0,
            "n_invokes_measured": 0,
        }

    n = len(per_invoke_metrics)
    ev_vals = [m["entity_validity_rate"] for m in per_invoke_metrics]
    vc_vals = [m["vocabulary_correctness"] for m in per_invoke_metrics]
    ma_vals = [m["mechanics_alignment"] for m in per_invoke_metrics
               if m["mechanics_alignment"] is not None]
    passes = sum(1 for m in per_invoke_metrics if m.get("grounded_pass"))

    return {
        "mean_entity_validity_rate": sum(ev_vals) / n,
        "mean_vocabulary_correctness": sum(vc_vals) / n,
        "mean_mechanics_alignment": (sum(ma_vals) / len(ma_vals)) if ma_vals else None,
        "mechanics_alignment_coverage": len(ma_vals) / n,
        "grounded_pass_count": passes,
        "grounded_pass_rate": passes / n,
        "n_invokes_measured": n,
    }
