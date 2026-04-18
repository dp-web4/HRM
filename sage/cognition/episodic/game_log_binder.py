#!/usr/bin/env python3
"""
Retroactively bind ARC-AGI-3 game play logs as episodic memories.

Converts Thor's game experiment logs into Episode objects and
stores them in an EpisodicIndex. This gives the episodic system
real-world data for testing pattern-completion retrieval.

Usage:
    python3 sage/cognition/episodic/game_log_binder.py [--db-path episodes.db]
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from sage.cognition.episodic.data import Episode
from sage.cognition.episodic.index import EpisodicIndex

LOG_DIR = Path("/home/dp/ai-workspace/shared-context/arc-agi-3/fleet-learning/thor/logs")


def bind_plh_log(log_path: Path, index: EpisodicIndex) -> int:
    """Bind a play-like-human or goalscaffold log."""
    with open(log_path) as f:
        data = json.load(f)

    game = data.get("game", "unknown")
    model = data.get("model", "unknown")
    experiment = data.get("experiment", log_path.stem.split("-")[1] if "-" in log_path.stem else "unknown")
    levels_won = data.get("levels_won", 0)
    entries = data.get("log", [])

    count = 0
    for entry in entries:
        action = entry.get("action", "")
        feedback = entry.get("feedback", entry.get("diff", ""))
        what_i_see = entry.get("what_i_see", "")
        why = entry.get("why", entry.get("plan", entry.get("reasoning", "")))
        level_up = entry.get("level_up", False)
        elapsed = entry.get("elapsed", 0)
        turn = entry.get("turn", 0)

        # Determine success: level_up is clear success, "Nothing changed" is failure
        success = None
        reward = 0.0
        if level_up:
            success = True
            reward = 1.0
        elif "Nothing changed" in feedback or "counter" in feedback:
            success = False
            reward = -0.2
        elif "MOVED" in feedback or "CHANGED" in feedback or "change" in feedback.lower():
            reward = 0.3

        ep = Episode(
            session_id=f"thor-{experiment}",
            cycle_id=turn,
            state_signature={
                "game": game,
                "model": model,
                "experiment": experiment,
                "turn_fraction": round(turn / max(len(entries), 1), 2),
            },
            sensory_summary={
                "what_i_see": what_i_see[:200] if what_i_see else "",
                "feedback": feedback[:200] if feedback else "",
            },
            snarc_scores={
                "surprise": 0.9 if level_up else (0.5 if reward > 0 else 0.2),
                "novelty": 0.7 if turn <= 5 else 0.3,
                "arousal": 0.8 if level_up else 0.4,
                "reward": max(0, reward),
                "conflict": 0.1,
            },
            action_taken=action.split("(")[0],  # Strip coordinates
            action_args={"raw": action, "why": why[:100] if why else ""},
            outcome=feedback[:100] if feedback else None,
            reward=reward,
            success=success,
            tags=[
                game.split("-")[0],  # e.g., "cd82"
                experiment,
                model.replace(":", "_"),
                "game_play",
                "arc_agi_3",
            ] + (["level_up"] if level_up else []),
        )
        index.bind(ep)
        count += 1

    return count


def bind_competition_log(log_path: Path, index: EpisodicIndex) -> int:
    """Bind a competition simulation log."""
    with open(log_path) as f:
        data = json.load(f)

    game = data.get("game", "unknown")
    model = data.get("model", "unknown")
    entries = data.get("log", [])

    count = 0
    for entry in entries:
        if entry.get("type") != "action":
            continue

        action = entry.get("action", "")
        diff = entry.get("diff", "")
        reasoning = entry.get("reasoning", "")
        level = entry.get("level", 0)
        turn = entry.get("turn", 0)

        success = None
        reward = 0.0
        if "no change" in diff:
            success = False
            reward = -0.2
        elif "counter tick" in diff:
            success = False
            reward = -0.1
        elif diff and "px" in diff:
            reward = 0.3

        ep = Episode(
            session_id=f"thor-competition",
            cycle_id=turn,
            state_signature={
                "game": game,
                "model": model,
                "experiment": "competition",
                "level": level,
            },
            sensory_summary={"diff": diff},
            snarc_scores={
                "surprise": 0.5 if reward > 0 else 0.2,
                "novelty": 0.6 if turn <= 10 else 0.3,
                "arousal": 0.4,
                "reward": max(0, reward),
                "conflict": 0.1,
            },
            action_taken=action,
            action_args={"x": entry.get("x"), "y": entry.get("y")},
            outcome=diff[:100] if diff else None,
            reward=reward,
            success=success,
            tags=[game.split("-")[0], "competition", "game_play", "arc_agi_3"],
        )
        index.bind(ep)
        count += 1

    return count


def bind_all_logs(db_path: str = None) -> EpisodicIndex:
    """Bind all Thor game logs into an episodic index."""
    index = EpisodicIndex(db_path=db_path)

    if not LOG_DIR.exists():
        print(f"Log directory not found: {LOG_DIR}")
        return index

    total = 0
    for log_file in sorted(LOG_DIR.glob("*.json")):
        try:
            name = log_file.stem
            if "plh" in name or "goalscaffold" in name or "twopipe" in name:
                count = bind_plh_log(log_file, index)
            elif "competition" in name:
                count = bind_competition_log(log_file, index)
            elif "autonomous" in name:
                continue  # Different format, skip for now
            else:
                count = bind_plh_log(log_file, index)  # Try generic

            total += count
            print(f"  {name}: {count} episodes")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    print(f"\nTotal: {total} episodes bound")
    print(f"Stats: {index.stats()}")

    # Test retrieval
    from sage.cognition.episodic.data import EpisodicCue

    print("\n--- Retrieval tests ---")

    # Find cd82 episodes
    results = index.recall(EpisodicCue(tags=["cd82"]), k=5)
    print(f"cd82 recall: {len(results)} results")
    if results:
        print(f"  Best: sim={results[0].similarity:.3f} action={results[0].episode.action_taken} "
              f"reward={results[0].episode.reward:.1f}")

    # Find level-up episodes
    results = index.recall(EpisodicCue(tags=["level_up"]), k=5)
    print(f"level_up recall: {len(results)} results")

    # Find high-reward episodes
    results = index.recall(EpisodicCue(
        snarc_scores={"reward": 0.8, "surprise": 0.7}
    ), k=5)
    print(f"high-reward recall: {len(results)} results")
    if results:
        for r in results[:3]:
            print(f"  {r.episode.tags[:3]} action={r.episode.action_taken} sim={r.similarity:.3f}")

    # Find "nothing changed" episodes (what NOT to do)
    results = index.recall(EpisodicCue(
        snarc_scores={"reward": 0.0, "surprise": 0.1}
    ), k=5)
    print(f"low-reward recall: {len(results)} results")

    return index


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default=None, help="SQLite path for persistence")
    args = p.parse_args()
    bind_all_logs(args.db_path)
