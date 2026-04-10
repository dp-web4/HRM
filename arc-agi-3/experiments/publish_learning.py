#!/usr/bin/env python3
"""
Publish Learning — extract game learning and append to fleet-learning dir.

Reads from:
  - sage_solver session.json (interactive play)
  - GameKnowledgeBase .knowledge.json files (model play)

Writes to:
  - shared-context/arc-agi-3/fleet-learning/{machine}/{game}_learning.jsonl

Each entry is attributed to machine + player (model identity).

Usage:
    # After interactive play (sage_solver)
    python3 publish_learning.py --session /tmp/sage_solver/session.json

    # After model play (extract from GameKB)
    python3 publish_learning.py --kb arc-agi-3/experiments/cartridges/lp85.knowledge.json

    # Auto-detect machine name
    SAGE_MACHINE=cbp python3 publish_learning.py --session /tmp/sage_solver/session.json

Environment:
    SAGE_MACHINE  — machine name (default: hostname)
    GAME_PLAYER   — player/model name (default: from session or "unknown")
"""

import json
import os
import time
import socket
from pathlib import Path

def _fs_safe(name):
    """Sanitize a name for filesystem use — replace colons, slashes, spaces."""
    return name.replace(":", "-").replace("/", "-").replace(" ", "_").replace("\\", "-")

MACHINE = _fs_safe(os.environ.get("SAGE_MACHINE", socket.gethostname().split(".")[0].lower()))
FLEET_DIR = Path(os.environ.get(
    "FLEET_LEARNING_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "shared-context", "arc-agi-3", "fleet-learning")
))


def ensure_machine_dir():
    d = FLEET_DIR / MACHINE
    d.mkdir(parents=True, exist_ok=True)
    return d


def publish_from_session(session_path):
    """Extract learning from a sage_solver session."""
    with open(session_path) as f:
        session = json.load(f)

    game = session.get("game_prefix", session.get("game_id", "unknown").split("-")[0])
    player = session.get("player", os.environ.get("GAME_PLAYER", "unknown"))
    levels = session.get("levels_completed", 0)
    win_levels = session.get("win_levels", 0)
    solutions = session.get("level_solutions", {})

    machine_dir = ensure_machine_dir()
    output_path = machine_dir / f"{game}_learning.jsonl"

    entries = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Per-level solutions
    for level_str, sol in solutions.items():
        level = int(level_str)
        entries.append({
            "timestamp": now,
            "machine": MACHINE,
            "player": sol.get("player", player),
            "game": game,
            "level": level,
            "event": "level_solved",
            "actions": sol.get("steps", 0),
            "action_sequence": sol.get("actions", []),
        })

    # Game-level summary
    if levels > 0:
        entries.append({
            "timestamp": now,
            "machine": MACHINE,
            "player": player,
            "game": game,
            "level": -1,
            "event": "game_complete" if levels >= win_levels else "game_progress",
            "levels_solved": levels,
            "levels_total": win_levels,
            "total_actions": session.get("step", 0),
        })

    # Level summaries as insights
    for summary in session.get("level_summaries", []):
        entries.append({
            "timestamp": now,
            "machine": MACHINE,
            "player": player,
            "game": game,
            "level": summary.get("level", -1),
            "event": "level_summary",
            "steps": summary.get("steps", 0),
            "wasted": summary.get("wasted", 0),
        })

    # Append to file
    with open(output_path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Published {len(entries)} entries → {output_path}")
    return entries


def publish_from_kb(kb_path):
    """Extract learning from a GameKnowledgeBase file."""
    with open(kb_path) as f:
        kb = json.load(f)

    game = kb.get("game_family", Path(kb_path).stem.replace(".knowledge", ""))
    player = os.environ.get("GAME_PLAYER", "unknown")

    machine_dir = ensure_machine_dir()
    output_path = machine_dir / f"{game}_learning.jsonl"

    entries = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Best level
    best = kb.get("best_level", 0)
    if best > 0:
        entries.append({
            "timestamp": now,
            "machine": MACHINE,
            "player": player,
            "game": game,
            "level": -1,
            "event": "game_progress",
            "levels_solved": best,
            "sessions": kb.get("session_count", 0),
        })

    # Discovered mechanics
    for mechanic in kb.get("mechanics", [])[:10]:
        entries.append({
            "timestamp": now,
            "machine": MACHINE,
            "player": player,
            "game": game,
            "level": -1,
            "event": "mechanic_discovered",
            "description": mechanic[:200],
        })

    # Failed approaches
    for fail in kb.get("failed_approaches", [])[:5]:
        entries.append({
            "timestamp": now,
            "machine": MACHINE,
            "player": player,
            "game": game,
            "level": -1,
            "event": "approach_failed",
            "description": str(fail)[:200],
        })

    # Level solutions
    for level_str, sol in kb.get("level_solutions", {}).items():
        entries.append({
            "timestamp": now,
            "machine": MACHINE,
            "player": player,
            "game": game,
            "level": int(level_str),
            "event": "level_solved",
            "actions": sol.get("attempts", 0),
            "confidence": sol.get("confidence", 0),
        })

    with open(output_path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Published {len(entries)} entries from KB → {output_path}")
    return entries


if __name__ == "__main__":
    import sys

    if "--session" in sys.argv:
        idx = sys.argv.index("--session")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "/tmp/sage_solver/session.json"
        publish_from_session(path)
    elif "--kb" in sys.argv:
        idx = sys.argv.index("--kb")
        path = sys.argv[idx + 1]
        publish_from_kb(path)
    else:
        print("Usage:")
        print("  python3 publish_learning.py --session /tmp/sage_solver/session.json")
        print("  python3 publish_learning.py --kb cartridges/lp85.knowledge.json")
        print("\nEnv: SAGE_MACHINE=cbp GAME_PLAYER=gemma3:4b")
