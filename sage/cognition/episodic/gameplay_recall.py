#!/usr/bin/env python3
"""
Game-state episodic recall for three-party gameplay conversation.

Extends EpisodicIndex with gameplay-specific episode schema and recall.
Game-state episodes are (game, level, frame_signature, action, outcome)
tuples — distinct from raising episodes.

Used by llm_dispatch at invoke time: "have I been in a state like this
before? What worked? What failed?"

Sprint: Phase 4 B2
Owner: Thor
"""

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from sage.cognition.episodic.data import Episode, EpisodicCue, EMBEDDING_DIM
from sage.cognition.episodic.index import EpisodicIndex

logger = logging.getLogger(__name__)


# ─── Frame signature ──────────────────────────────────────────────

def compute_frame_signature(frame: np.ndarray, dim: int = 32) -> np.ndarray:
    """Compute a compact signature from a 64x64 game frame.

    Uses spatial pooling + color histogram to create a frame fingerprint
    that's robust to small pixel changes but sensitive to structural
    differences (object positions, color distributions).

    Args:
        frame: (H, W) int array with color indices 0-15, or (C, H, W) one-hot
        dim: output dimension

    Returns:
        (dim,) float32 vector
    """
    # Normalize to 2D int grid
    if frame.ndim == 3:
        if frame.shape[0] <= 16:  # one-hot (C, H, W)
            frame = frame.argmax(axis=0)
        else:
            frame = frame[-1]  # take last layer
    frame = frame.astype(np.int32)
    h, w = frame.shape

    # Color histogram (16 bins, normalized)
    hist = np.bincount(frame.ravel(), minlength=16).astype(np.float32)
    hist /= max(hist.sum(), 1)

    # Spatial pooling: divide frame into 4x4 blocks, dominant color per block
    block_h, block_w = max(h // 4, 1), max(w // 4, 1)
    spatial = np.zeros(16, dtype=np.float32)
    for by in range(4):
        for bx in range(4):
            block = frame[by*block_h:(by+1)*block_h, bx*block_w:(bx+1)*block_w]
            if block.size > 0:
                dominant = np.bincount(block.ravel(), minlength=16).argmax()
                spatial[by * 4 + bx] = dominant / 15.0  # normalize to [0, 1]

    # Combine and project to target dim
    combined = np.concatenate([hist, spatial])  # 32-dim
    if len(combined) < dim:
        combined = np.pad(combined, (0, dim - len(combined)))
    elif len(combined) > dim:
        combined = combined[:dim]

    return combined.astype(np.float32)


# ─── Game-state episode creation ──────────────────────────────────

def create_game_episode(
    game_family: str,
    game_id: str,
    level: int,
    step_index: int,
    frame: np.ndarray,
    action_taken: int,
    coords: Optional[Dict[str, int]] = None,
    frame_delta_pct: float = 0.0,
    level_after: int = 0,
    state_after: str = "RUNNING",
    outcome: str = "neutral",
    action_context: Optional[List[int]] = None,
    source_machine: str = "unknown",
    source_adapter: str = "unknown",
) -> Episode:
    """Create a game-state Episode from a gameplay step.

    This wraps the generic Episode with gameplay-specific fields in
    state_signature and sensory_summary, plus appropriate tags for
    recall filtering.
    """
    ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]
    action_name = ACTION_NAMES[action_taken] if 0 <= action_taken < len(ACTION_NAMES) else str(action_taken)

    frame_sig = compute_frame_signature(frame)

    # Determine success
    success = None
    reward = 0.0
    if outcome == "advance":
        success = True
        reward = 1.0
    elif outcome == "game_over":
        success = False
        reward = -1.0
    elif outcome == "stuck":
        success = False
        reward = -0.3
    elif frame_delta_pct > 5.0:
        reward = 0.3  # Something changed — possibly productive

    # Build state signature for embedding
    state_sig = {
        "game": game_family,
        "level": level,
        "step_fraction": round(step_index / 200, 2),  # normalize to [0, 1] range
        "action_context": (action_context or [])[-5:],
    }

    # SNARC-like scores derived from game state
    snarc = {
        "surprise": min(1.0, frame_delta_pct / 20.0),  # big frame change = surprise
        "novelty": 0.7 if step_index < 10 else 0.3,  # early steps are novel
        "arousal": 0.9 if outcome == "advance" else (0.6 if outcome == "stuck" else 0.4),
        "reward": max(0, reward),
        "conflict": 0.1,
    }

    ep = Episode(
        session_id=f"{source_machine}-gameplay",
        cycle_id=step_index,
        state_signature=state_sig,
        sensory_summary={
            "frame_signature": frame_sig.tolist(),
            "frame_delta_pct": frame_delta_pct,
            "coords": coords,
        },
        snarc_scores=snarc,
        action_taken=action_name,
        action_args={
            "action_int": action_taken,
            "coords": coords or {},
            "game_id": game_id,
            "level_after": level_after,
            "state_after": state_after,
        },
        outcome=outcome,
        reward=reward,
        success=success,
        tags=[
            game_family,
            f"L{level}",
            "gameplay",
            "game_episode",
            source_machine,
            outcome,
        ],
    )
    ep.compute_embedding()
    return ep


# ─── Gameplay recall ──────────────────────────────────────────────

@dataclass
class GameStateCue:
    """Structured cue for gameplay recall."""
    game_family: str
    level: int
    frame: Optional[np.ndarray] = None
    action_context: Optional[List[int]] = None
    outcome_filter: Optional[str] = None  # "advance" to prefer successes


def recall_gameplay(
    index: EpisodicIndex,
    cue: GameStateCue,
    k: int = 3,
    min_similarity: float = 0.5,
) -> List[Tuple[Episode, float]]:
    """Recall similar past game episodes.

    Returns up to k episodes with similarity >= min_similarity,
    sorted by relevance (similarity * recency * success_boost).

    Args:
        index: the EpisodicIndex instance
        cue: structured gameplay cue
        k: max results
        min_similarity: floor for inclusion

    Returns:
        List of (Episode, similarity_score) tuples
    """
    # Build EpisodicCue from GameStateCue
    state_sig = {
        "game": cue.game_family,
        "level": cue.level,
    }
    if cue.action_context:
        state_sig["action_context"] = cue.action_context[-5:]

    tags = [cue.game_family, "game_episode"]
    if cue.outcome_filter:
        tags.append(cue.outcome_filter)

    # Frame signature as SNARC-like cue (uses the sensory similarity path)
    snarc_cue = None
    if cue.frame is not None:
        sig = compute_frame_signature(cue.frame)
        # Use surprise/arousal dims as proxy for frame similarity
        snarc_cue = {
            "surprise": float(sig[:5].mean()),
            "arousal": float(sig[5:10].mean()),
        }

    ep_cue = EpisodicCue(
        state_signature=state_sig,
        snarc_scores=snarc_cue,
        tags=tags,
    )

    results = index.recall(ep_cue, k=k * 5)  # Over-fetch, then filter

    # Filter: must match game family (tag-based, since embedding similarity
    # is too coarse for game discrimination)
    matches = []
    for r in results:
        ep_tags = set(r.episode.tags)
        if cue.game_family not in ep_tags:
            continue  # Wrong game
        if r.similarity >= min_similarity * 0.5:  # Relax sim threshold, rely on tag filter
            # Boost successful episodes
            score = r.similarity
            if r.episode.success:
                score *= 1.3
            matches.append((r.episode, score))

    matches.sort(key=lambda x: -x[1])
    return matches[:k]


def format_recall_for_prompt(matches: List[Tuple[Episode, float]], max_entries: int = 3) -> str:
    """Format recalled episodes as text for the LLM prompt.

    Returns a concise string suitable for injection into the
    three-party conversation narration.
    """
    if not matches:
        return ""

    lines = ["Past experience (from episodic memory):"]
    for ep, score in matches[:max_entries]:
        action = ep.action_taken
        outcome = ep.outcome or "unknown"
        game = ep.state_signature.get("game", "?")
        level = ep.state_signature.get("level", "?")
        delta = ep.sensory_summary.get("frame_delta_pct", 0)
        coords = ep.sensory_summary.get("coords")
        coord_str = f" at ({coords['x']},{coords['y']})" if coords else ""

        success_str = "succeeded" if ep.success else ("failed" if ep.success is False else "neutral")
        lines.append(
            f"  - {game} L{level}: {action}{coord_str} → {success_str} "
            f"({delta:.0f}% change, outcome={outcome})"
        )

    return "\n".join(lines)


# ─── Population from fleet data ───────────────────────────────────

def populate_from_delta_records(
    index: EpisodicIndex,
    data_dir: str,
    machines: Optional[List[str]] = None,
) -> int:
    """Populate episodic index from fleet delta/gameplay records.

    Walks per-machine data directories and ingests gameplay,
    sage_plays_live, and llm_dispatch records.

    Args:
        index: EpisodicIndex to populate
        data_dir: root of training-data/router/
        machines: list of machine slugs, or None for all

    Returns:
        number of episodes ingested
    """
    import gzip

    root = Path(data_dir)
    if not root.exists():
        logger.warning(f"Data dir not found: {root}")
        return 0

    if machines is None:
        machines = [d.name for d in root.iterdir()
                    if d.is_dir() and not d.name.startswith("_")]

    total = 0
    for machine in machines:
        machine_dir = root / machine
        # Walk all subdirs for gameplay-related records
        for subdir_name in ["gameplay", "sage_plays_live", "llm_dispatch", "sage_plays"]:
            subdir = machine_dir / subdir_name
            if not subdir.exists():
                continue

            for gz_file in sorted(subdir.glob("*.jsonl.gz")):
                try:
                    with gzip.open(gz_file, "rt") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                                meta = rec.get("metadata", {})
                                game = meta.get("game", "")
                                if not game:
                                    continue

                                game_family = game.split("-")[0] if "-" in game else game
                                action = meta.get("known_good_action", rec.get("router_output", {}).get("action_idx"))
                                if action is None:
                                    continue

                                level = meta.get("known_good_level", 0)
                                delta = meta.get("frame_delta_pct", 0.0)

                                # Determine outcome
                                outcome = "neutral"
                                go = meta.get("game_outcome", {})
                                if go.get("result") == "WIN":
                                    outcome = "advance"
                                elif go.get("result") == "GAME_OVER":
                                    outcome = "game_over"

                                # Create a minimal frame signature from available data
                                # (we don't have the raw frame, so use a hash of the record)
                                pseudo_frame = np.zeros((64, 64), dtype=np.int32)

                                ep = create_game_episode(
                                    game_family=game_family,
                                    game_id=game,
                                    level=level,
                                    step_index=meta.get("step_index", 0),
                                    frame=pseudo_frame,
                                    action_taken=int(action),
                                    coords=meta.get("known_good_data"),
                                    frame_delta_pct=delta,
                                    level_after=level,
                                    outcome=outcome,
                                    source_machine=machine,
                                    source_adapter=meta.get("adapter", "unknown"),
                                )
                                index.bind(ep)
                                total += 1
                            except Exception:
                                continue
                except Exception as e:
                    logger.warning(f"Error reading {gz_file}: {e}")

    logger.info(f"Populated {total} game episodes from {len(machines)} machines")
    return total


# ─── CLI for standalone population + test ─────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Populate + test gameplay episodic recall")
    p.add_argument("--data-dir", default=os.environ.get(
        "SAGE_ROUTER_DATA_DIR",
        os.path.expanduser("~/ai-workspace/private-context/training-data/router")))
    p.add_argument("--db-path", default=os.path.expanduser(
        "~/ai-workspace/private-context/training-data/episodic/gameplay_episodes.db"))
    p.add_argument("--machines", nargs="*", default=None)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    index = EpisodicIndex(db_path=args.db_path)

    print(f"Existing episodes: {index.stats().get('count', 0)}")
    count = populate_from_delta_records(index, args.data_dir, args.machines)
    print(f"Ingested: {count} new episodes")
    print(f"Total: {index.stats()}")

    # Test recall
    print("\n--- Recall tests ---")
    for game in ["cd82", "tu93", "ft09", "ka59", "sk48"]:
        cue = GameStateCue(game_family=game, level=0)
        matches = recall_gameplay(index, cue, k=3)
        if matches:
            formatted = format_recall_for_prompt(matches)
            print(f"\n{game}:")
            print(formatted)
        else:
            print(f"\n{game}: no matches")
