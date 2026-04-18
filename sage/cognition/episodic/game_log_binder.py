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
MCNUGGET_LOG_DIR = Path("/home/dp/ai-workspace/shared-context/arc-agi-3/fleet-learning/mcnugget/logs")


def _safe_float(val, default=0.5) -> float:
    """Convert to float, handling string confidence levels."""
    if isinstance(val, (int, float)):
        return float(val)
    mapping = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8}
    if isinstance(val, str) and val.upper() in mapping:
        return mapping[val.upper()]
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


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


def bind_mcnugget_r7_log(log_path: Path, index: EpisodicIndex) -> int:
    """Bind McNugget r7-* logs (rich traces with prediction/reflection)."""
    with open(log_path) as f:
        data = json.load(f)

    trace = data.get("trace", [])
    summary = data.get("summary", {})
    game = summary.get("game", log_path.stem.split("-")[1] if "-" in log_path.stem else "unknown")

    count = 0
    for entry in trace:
        action = entry.get("action", "")
        if not action:
            continue

        prediction = entry.get("prediction", "")
        result = entry.get("result", "")
        diff = entry.get("pixel_diff", 0)
        reflection = entry.get("reflection", "")
        trust = entry.get("trust_update", {})
        level_won = entry.get("level_won", False)
        step = entry.get("step", 0)

        reward = 0.0
        success = None
        if level_won:
            reward = 1.0
            success = True
        elif diff == 0:
            reward = -0.2
            success = False
        elif diff > 10:
            reward = 0.3

        ep = Episode(
            session_id="mcnugget-r7",
            cycle_id=step,
            state_signature={
                "game": game,
                "experiment": "r7",
                "prediction": prediction[:100] if prediction else "",
            },
            sensory_summary={
                "result": result[:200] if result else "",
                "reflection": reflection[:200] if reflection else "",
                "pixel_diff": str(diff),
            },
            snarc_scores={
                "surprise": 0.9 if level_won else (0.5 if diff > 10 else 0.2),
                "novelty": 0.6 if step <= 5 else 0.3,
                "arousal": 0.8 if level_won else 0.4,
                "reward": max(0, reward),
                "conflict": _safe_float(entry.get("confidence", 0.5)),
            },
            action_taken=action if isinstance(action, str) else str(action),
            action_args={"trust": trust} if trust else {},
            outcome=result[:100] if result else None,
            reward=reward,
            success=success,
            tags=[game, "mcnugget", "r7", "game_play"] + (["level_up"] if level_won else []),
        )
        index.bind(ep)
        count += 1

    return count


def bind_mcnugget_probe_log(log_path: Path, index: EpisodicIndex) -> int:
    """Bind McNugget structured-probe-* logs."""
    with open(log_path) as f:
        data = json.load(f)

    trace = data.get("trace", [])
    game = data.get("game", "unknown")
    won = data.get("won", False)

    count = 0
    for entry in trace:
        action = entry.get("action", "")
        if not action:
            continue

        phase = entry.get("phase", "")
        diff = entry.get("pixel_diff", 0)
        step_won = entry.get("won", False)
        step = entry.get("step", 0)
        action_data = entry.get("data", {})

        reward = 0.0
        success = None
        if step_won:
            reward = 1.0
            success = True
        elif diff == 0:
            reward = -0.2
            success = False
        elif diff > 10:
            reward = 0.3

        ep = Episode(
            session_id="mcnugget-probe",
            cycle_id=step,
            state_signature={
                "game": game,
                "experiment": "structured_probe",
                "phase": phase,
            },
            sensory_summary={"pixel_diff": str(diff)},
            snarc_scores={
                "surprise": 0.9 if step_won else (0.4 if diff > 0 else 0.1),
                "novelty": 0.7 if phase == "explore" else 0.3,
                "arousal": 0.8 if step_won else 0.3,
                "reward": max(0, reward),
                "conflict": 0.1,
            },
            action_taken=action if isinstance(action, str) else str(action),
            action_args=action_data if isinstance(action_data, dict) else {},
            outcome=f"{diff}px" if diff else "no change",
            reward=reward,
            success=success,
            tags=[game, "mcnugget", "structured_probe", "game_play"] + (["level_up"] if step_won else []),
        )
        index.bind(ep)
        count += 1

    return count


def bind_all_logs(db_path: str = None) -> EpisodicIndex:
    """Bind game logs from all fleet machines into an episodic index."""
    index = EpisodicIndex(db_path=db_path)

    # --- Thor logs ---
    thor_total = 0
    if LOG_DIR.exists():
        for log_file in sorted(LOG_DIR.glob("*.json")):
            try:
                name = log_file.stem
                if "plh" in name or "goalscaffold" in name or "twopipe" in name:
                    count = bind_plh_log(log_file, index)
                elif "competition" in name:
                    count = bind_competition_log(log_file, index)
                elif "autonomous" in name:
                    continue
                else:
                    count = bind_plh_log(log_file, index)
                thor_total += count
                print(f"  thor/{name}: {count} episodes")
            except Exception as e:
                print(f"  thor/{name}: ERROR {e}")
    print(f"\nThor: {thor_total} episodes")

    # --- McNugget logs ---
    mc_total = 0
    if MCNUGGET_LOG_DIR.exists():
        for log_file in sorted(MCNUGGET_LOG_DIR.glob("*.json")):
            try:
                name = log_file.stem
                with open(log_file) as f:
                    data = json.load(f)

                if "trace" in data and data["trace"]:
                    # Has per-action traces — bind them
                    first_entry = data["trace"][0]
                    if "prediction" in first_entry or "reflection" in first_entry:
                        count = bind_mcnugget_r7_log(log_file, index)
                    elif "phase" in first_entry:
                        count = bind_mcnugget_probe_log(log_file, index)
                    else:
                        # Generic trace — try probe format
                        count = bind_mcnugget_probe_log(log_file, index)
                    mc_total += count
                    print(f"  mcnugget/{name}: {count} episodes")
                else:
                    # Summary-only logs (harness-v1) — bind as single episode per result
                    results = data.get("results", [])
                    game = data.get("game", "unknown")
                    for r in results:
                        ep = Episode(
                            session_id="mcnugget-harness",
                            state_signature={"game": game, "experiment": "harness_v1", "level": r.get("level", 0)},
                            snarc_scores={"surprise": 0.8 if r.get("won") else 0.3, "reward": 1.0 if r.get("won") else 0.0},
                            action_taken=f"harness_v1_L{r.get('level', '?')}",
                            outcome="won" if r.get("won") else f"lost ({r.get('actions', '?')} actions)",
                            reward=1.0 if r.get("won") else -0.1,
                            success=r.get("won"),
                            tags=[game.split("-")[0] if "-" in game else game, "mcnugget", "harness_v1", "game_play"]
                                 + (["level_up"] if r.get("won") else []),
                        )
                        index.bind(ep)
                        mc_total += 1
                    if results:
                        print(f"  mcnugget/{name}: {len(results)} summary episodes")
            except Exception as e:
                print(f"  mcnugget/{name}: ERROR {e}")
    print(f"McNugget: {mc_total} episodes")

    total = thor_total + mc_total
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
