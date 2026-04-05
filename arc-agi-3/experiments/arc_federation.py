#!/usr/bin/env python3
"""
ARC-AGI-3 Federation Layer — Git-synced knowledge sharing.

Each machine generates knowledge. This module:
1. EXPORT: After a game session, write discoveries to git-tracked JSON
2. IMPORT: At game start, load discoveries from ALL machines
3. MERGE: Combine knowledge from multiple machines intelligently

The shared knowledge lives in arc-agi-3/shared_knowledge/ (git-tracked).
Each machine writes to its own file. All machines read all files.

No conflicts: each machine owns its own file. Git handles distribution.

Structure:
    shared_knowledge/
        mcnugget.json     — McNugget's discoveries
        sprout.json       — Sprout's discoveries
        thor.json         — Thor's discoveries
        nomad.json        — Nomad's discoveries
        merged.json       — Auto-merged (read-only, rebuilt on import)
"""

import json
import os
import time
import socket
from pathlib import Path
from typing import Dict, List, Optional


SHARED_DIR = Path("arc-agi-3/shared_knowledge")


def _get_machine_name() -> str:
    """Detect which machine we're on."""
    hostname = socket.gethostname().lower()
    if "mcnugget" in hostname:
        return "mcnugget"
    elif "sprout" in hostname or "ubuntu" in hostname:
        return "sprout"
    elif "thor" in hostname:
        return "thor"
    elif "nomad" in hostname:
        return "nomad"
    elif "legion" in hostname:
        return "legion"
    elif "cbp" in hostname:
        return "cbp"
    # Fallback: check for machine-specific paths
    if os.path.exists("/Users/dennispalatov"):
        return "mcnugget"
    return hostname.split(".")[0]


class FederatedKnowledge:
    """Git-synced knowledge sharing between fleet machines.

    Each machine writes to shared_knowledge/<machine>.json.
    All machines read all files and merge at load time.

    Knowledge categories:
    - game_discoveries: per-game facts (interactive objects, cycle lengths, etc.)
    - strategies: per-game-type strategies that worked
    - failures: what was tried and didn't work (prevents repetition)
    - meta_insights: cross-game abstract principles
    """

    def __init__(self, machine: str = None):
        self.machine = machine or _get_machine_name()
        self.shared_dir = SHARED_DIR
        self.shared_dir.mkdir(parents=True, exist_ok=True)

        # Load our machine's knowledge
        self.our_file = self.shared_dir / f"{self.machine}.json"
        self.our_data = self._load_file(self.our_file) or self._empty_knowledge()

        # Load all machines' knowledge
        self.fleet_data: Dict[str, dict] = {}
        self._load_all()

    def _empty_knowledge(self) -> dict:
        return {
            "machine": self.machine,
            "updated_at": time.time(),
            "game_discoveries": {},   # game_prefix → list of discoveries
            "strategies": {},         # game_type → list of strategies
            "failures": {},           # game_prefix → list of failed approaches
            "meta_insights": [],      # cross-game abstract principles
        }

    def _load_file(self, path: Path) -> Optional[dict]:
        try:
            if path.exists():
                return json.loads(path.read_text())
        except Exception:
            pass
        return None

    def _load_all(self):
        """Load knowledge from all machines."""
        for f in self.shared_dir.glob("*.json"):
            if f.name == "merged.json":
                continue
            name = f.stem
            data = self._load_file(f)
            if data:
                self.fleet_data[name] = data

    def save(self):
        """Save our machine's knowledge to git-tracked file."""
        self.our_data["updated_at"] = time.time()
        self.our_file.write_text(json.dumps(self.our_data, indent=2))

    # ─── WRITE (our machine only) ───

    def add_discovery(self, game_prefix: str, discovery: str):
        """Add a game-specific discovery."""
        if game_prefix not in self.our_data["game_discoveries"]:
            self.our_data["game_discoveries"][game_prefix] = []
        entries = self.our_data["game_discoveries"][game_prefix]
        # Dedup: don't add if first 60 chars match existing
        if not any(discovery[:60] == e.get("text", "")[:60] for e in entries):
            entries.append({
                "text": discovery,
                "machine": self.machine,
                "timestamp": time.time(),
            })
            # Keep bounded: max 20 per game
            if len(entries) > 20:
                self.our_data["game_discoveries"][game_prefix] = entries[-20:]

    def add_strategy(self, game_type: str, strategy: str):
        """Add a game-type strategy that worked."""
        if game_type not in self.our_data["strategies"]:
            self.our_data["strategies"][game_type] = []
        entries = self.our_data["strategies"][game_type]
        if not any(strategy[:60] == e.get("text", "")[:60] for e in entries):
            entries.append({
                "text": strategy,
                "machine": self.machine,
                "timestamp": time.time(),
            })
            if len(entries) > 10:
                self.our_data["strategies"][game_type] = entries[-10:]

    def add_failure(self, game_prefix: str, failure: str):
        """Record a failed approach so other machines don't repeat it."""
        if game_prefix not in self.our_data["failures"]:
            self.our_data["failures"][game_prefix] = []
        entries = self.our_data["failures"][game_prefix]
        if not any(failure[:60] == e.get("text", "")[:60] for e in entries):
            entries.append({
                "text": failure,
                "machine": self.machine,
                "timestamp": time.time(),
            })
            if len(entries) > 10:
                self.our_data["failures"][game_prefix] = entries[-10:]

    def add_meta_insight(self, insight: str):
        """Add a cross-game abstract principle."""
        entries = self.our_data["meta_insights"]
        if not any(insight[:60] == e.get("text", "")[:60] for e in entries):
            entries.append({
                "text": insight,
                "machine": self.machine,
                "timestamp": time.time(),
            })
            if len(entries) > 20:
                self.our_data["meta_insights"] = entries[-20:]

    # ─── READ (all machines) ───

    def get_game_knowledge(self, game_prefix: str) -> str:
        """Get all fleet knowledge about a specific game."""
        lines = []
        for machine, data in self.fleet_data.items():
            discoveries = data.get("game_discoveries", {}).get(game_prefix, [])
            for d in discoveries[-3:]:  # Last 3 per machine
                source = f"[{machine}]" if machine != self.machine else "[you]"
                lines.append(f"  {source} {d['text'][:150]}")

        failures = []
        for machine, data in self.fleet_data.items():
            fails = data.get("failures", {}).get(game_prefix, [])
            for f in fails[-2:]:
                source = f"[{machine}]" if machine != self.machine else "[you]"
                failures.append(f"  {source} FAILED: {f['text'][:120]}")

        result = ""
        if lines:
            result += "FLEET KNOWLEDGE (what other machines discovered):\n" + "\n".join(lines)
        if failures:
            result += "\n\nKNOWN FAILURES (don't repeat):\n" + "\n".join(failures)
        return result

    def get_strategy(self, game_type: str) -> str:
        """Get fleet strategies for a game type."""
        lines = []
        for machine, data in self.fleet_data.items():
            strategies = data.get("strategies", {}).get(game_type, [])
            for s in strategies[-2:]:
                source = f"[{machine}]" if machine != self.machine else "[you]"
                lines.append(f"  {source} {s['text'][:150]}")
        if lines:
            return "FLEET STRATEGIES:\n" + "\n".join(lines)
        return ""

    def get_meta_insights(self) -> str:
        """Get all fleet meta-insights."""
        lines = []
        seen = set()
        for machine, data in self.fleet_data.items():
            for insight in data.get("meta_insights", [])[-3:]:
                key = insight["text"][:60]
                if key in seen:
                    continue
                seen.add(key)
                source = f"[{machine}]" if machine != self.machine else "[you]"
                lines.append(f"  {source} {insight['text'][:150]}")
        if lines:
            return "FLEET INSIGHTS (cross-game principles):\n" + "\n".join(lines)
        return ""

    def build_context(self, game_prefix: str, game_type: str = "") -> str:
        """Build full federation context for a game session."""
        parts = []
        game_knowledge = self.get_game_knowledge(game_prefix)
        if game_knowledge:
            parts.append(game_knowledge)
        if game_type:
            strategy = self.get_strategy(game_type)
            if strategy:
                parts.append(strategy)
        meta = self.get_meta_insights()
        if meta:
            parts.append(meta)
        return "\n\n".join(parts)

    def summary(self) -> str:
        """Summary of federation state."""
        machines = list(self.fleet_data.keys())
        total_discoveries = sum(
            sum(len(v) for v in d.get("game_discoveries", {}).values())
            for d in self.fleet_data.values())
        total_insights = sum(
            len(d.get("meta_insights", []))
            for d in self.fleet_data.values())
        return (f"Federation: {len(machines)} machines ({', '.join(machines)}), "
                f"{total_discoveries} discoveries, {total_insights} meta-insights")
