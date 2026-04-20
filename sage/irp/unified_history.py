"""Unified conversation-history schema for raising and gameplay sessions.

Raising uses `[{speaker: 'Claude'|'SAGE', text: str}, ...]` — a
speaker-named shape that mirrors its conversational partnership framing.

Gameplay uses `[{role: 'user'|'assistant', content: str}, ...]` — the
Anthropic/OpenAI-standard shape that LLM APIs consume directly.

Both are semantically equivalent lists of turns. This module normalizes
between them so both modes can share history-handling code.

Canonical form: `{role, content}` (LLM-standard). Raising writes its
own `{speaker, text}` format to session JSON for continuity; it
converts to/from canonical at the IRP boundary.

Sprint: Phase 4 P1.4 (alignment audit)
"""
from __future__ import annotations

from typing import Any, Dict, List


# Canonical role mapping from raising's speaker-named form
_SPEAKER_TO_ROLE = {
    "claude": "user",      # Claude is the raising partner (the "user" to SAGE)
    "dennis": "user",      # Dennis too, when present
    "sage": "assistant",   # SAGE's responses
    "user": "user",        # already canonical
    "assistant": "assistant",  # already canonical
    "system": "system",    # system messages pass through
}

# Inverse for writing back to raising format (role → speaker).
# Raising's session JSON preserves the specific speaker name rather than
# the generic 'user' role, so the round-trip needs a default.
_ROLE_TO_DEFAULT_SPEAKER = {
    "user": "Claude",
    "assistant": "SAGE",
    "system": "System",
}


def to_unified(turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert any supported history format to canonical `{role, content}`.

    Accepts:
      - `{role, content}` — returned unchanged
      - `{speaker, text}` — raising format
      - `{speaker, message}` — DaemonIRP memory format
      - mixed lists; each turn converted per its own keys
    """
    out: List[Dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        # Already canonical
        if "role" in turn and "content" in turn:
            out.append({"role": str(turn["role"]), "content": str(turn["content"])})
            continue
        # Raising shape
        if "speaker" in turn:
            speaker = str(turn["speaker"]).lower()
            role = _SPEAKER_TO_ROLE.get(speaker, "user")
            # Content key can be 'text' or 'message'
            content = turn.get("text", turn.get("message", ""))
            out.append({"role": role, "content": str(content)})
            continue
        # Unknown shape — skip defensively rather than guess
    return out


def from_unified(
    turns: List[Dict[str, Any]],
    preserve_speakers: bool = False,
) -> List[Dict[str, str]]:
    """Convert canonical `{role, content}` back to raising's `{speaker, text}`.

    preserve_speakers: if the original turn had a `speaker` alongside
    `role`/`content` (possible after a round-trip), keep it. Otherwise
    defaults: user→Claude, assistant→SAGE.
    """
    out: List[Dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if "role" not in turn:
            # Already raising-shape; pass through
            if "speaker" in turn:
                out.append({
                    "speaker": str(turn["speaker"]),
                    "text": str(turn.get("text", turn.get("message", ""))),
                })
            continue
        role = str(turn["role"])
        content = str(turn.get("content", ""))
        speaker = (
            str(turn["speaker"])
            if preserve_speakers and "speaker" in turn
            else _ROLE_TO_DEFAULT_SPEAKER.get(role, role.capitalize())
        )
        out.append({"speaker": speaker, "text": content})
    return out


def is_unified(turns: List[Dict[str, Any]]) -> bool:
    """Return True iff every turn has role + content (canonical)."""
    if not turns:
        return True
    return all(
        isinstance(t, dict) and "role" in t and "content" in t
        for t in turns
    )
