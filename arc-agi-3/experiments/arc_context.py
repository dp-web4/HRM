#!/usr/bin/env python3
"""
ARC-AGI-3 Context Constructor

Builds the situation-relevant context window by querying all memory
layers based on WHAT'S HAPPENING RIGHT NOW.

Membot isn't a static lookup — it's queried adaptively:
- Game start: "What do I know about this type of game?"
- After probe: "I found rotation buttons. What worked on rotation puzzles?"
- When stuck: "10 clicks, no level-up. How did I get unstuck before?"
- After level-up: "What sequence worked? Store it."
- After game end: "What did I learn? Abstract it."

The context constructor is the bridge between lived experience (memory)
and the current situation (observation). It answers: what should be in
the context window RIGHT NOW to help the model make the best decision?
"""

import time
import requests
from typing import Optional

MEMBOT_URL = "http://localhost:8000"


class ContextConstructor:
    """Builds situation-relevant context from all memory layers.

    Queries membot adaptively based on current game state.
    Caches results to avoid redundant queries.
    Stores new discoveries for future sessions.
    """

    def __init__(self, game_prefix: str):
        self.game_prefix = game_prefix
        self.membot_available = None
        self.query_cache = {}  # query → (result, timestamp)
        self.cache_ttl = 300   # cache for 5 minutes

        # Ensure cartridge is mounted
        self._mount_cartridge()

    def _mount_cartridge(self):
        """Mount the sage cartridge if membot is running."""
        try:
            resp = requests.post(f"{MEMBOT_URL}/api/mount",
                json={"name": "sage"}, timeout=5)
            self.membot_available = resp.status_code == 200
        except Exception:
            self.membot_available = False

    def _search(self, query: str, n: int = 3) -> list:
        """Search membot. Returns list of text results."""
        if not self.membot_available:
            return []

        # Check cache
        now = time.time()
        if query in self.query_cache:
            result, ts = self.query_cache[query]
            if now - ts < self.cache_ttl:
                return result

        try:
            resp = requests.post(f"{MEMBOT_URL}/api/search",
                json={"query": query, "top_k": n}, timeout=5)
            if resp.status_code == 200:
                results = [r["text"] for r in resp.json().get("results", [])
                          if r.get("score", 0) > 0.35]
                self.query_cache[query] = (results, now)
                return results
        except Exception:
            pass
        return []

    def store(self, text: str):
        """Store a discovery for future sessions."""
        if not self.membot_available:
            return
        try:
            requests.post(f"{MEMBOT_URL}/api/store",
                json={"content": text, "tags": f"arc-agi-3 {self.game_prefix}"},
                timeout=5)
        except Exception:
            pass

    def save(self):
        """Persist cartridge to disk."""
        if not self.membot_available:
            return
        try:
            requests.post(f"{MEMBOT_URL}/api/save",
                json={}, timeout=10)
        except Exception:
            pass

    # ─── Situation-Aware Queries ───

    def on_game_start(self, available_actions: list, game_type: str = "") -> str:
        """Query: What do I know about this type of game?"""
        queries = [
            f"ARC-AGI-3 {self.game_prefix} game strategy",
            f"ARC-AGI-3 {game_type} puzzle" if game_type else f"ARC-AGI-3 puzzle strategy",
        ]

        results = []
        for q in queries:
            results.extend(self._search(q, n=2))

        if results:
            # Deduplicate
            seen = set()
            unique = []
            for r in results:
                key = r[:60]
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            return "WHAT I KNOW ABOUT THIS GAME (from prior experience):\n" + \
                   "\n".join(f"  • {t[:150]}" for t in unique[:4])
        return ""

    def on_probe_complete(self, interactive_objects: list,
                          action_model_desc: str, game_type: str) -> str:
        """Query: I found these objects/mechanics. What worked before?"""
        queries = []

        if interactive_objects:
            obj_colors = set(o.split("_")[0] for o in interactive_objects)
            queries.append(f"ARC-AGI-3 interactive {' '.join(obj_colors)} button clicking")

        if "rotation" in game_type or "cycle" in game_type or "cursor" in game_type:
            queries.append("ARC-AGI-3 rotation puzzle cycle length solving strategy")
        if "navigation" in game_type or "maze" in game_type:
            queries.append("ARC-AGI-3 maze navigation path planning")

        queries.append(f"ARC-AGI-3 {game_type} solving approach")

        results = []
        for q in queries:
            results.extend(self._search(q, n=2))

        if results:
            seen = set()
            unique = [r for r in results if not (r[:60] in seen or seen.add(r[:60]))]
            return "RELEVANT EXPERIENCE (from similar situations):\n" + \
                   "\n".join(f"  • {t[:150]}" for t in unique[:3])
        return ""

    def on_stuck(self, n_actions: int, interactive_names: list,
                 patterns: str) -> str:
        """Query: I'm stuck. What helped before in similar situations?"""
        queries = [
            "ARC-AGI-3 stuck no level-up repeated clicking strategy change",
            f"ARC-AGI-3 {self.game_prefix} level solution sequence",
        ]

        if "STABLE" in patterns:
            queries.append("ARC-AGI-3 grid similarity stable not progressing")
        if "cycle" in patterns.lower():
            queries.append("ARC-AGI-3 cycle detected rotation period counting")

        results = []
        for q in queries:
            results.extend(self._search(q, n=2))

        if results:
            seen = set()
            unique = [r for r in results if not (r[:60] in seen or seen.add(r[:60]))]
            return "ADVICE FROM PRIOR EXPERIENCE (you've been stuck before):\n" + \
                   "\n".join(f"  • {t[:150]}" for t in unique[:3])
        return ""

    def on_level_up(self, level: int, winning_actions: list):
        """Store: What sequence worked for this level?"""
        action_str = ", ".join(winning_actions[-8:])
        self.store(
            f"ARC-AGI-3 {self.game_prefix}: Level {level} solved. "
            f"Winning sequence: {action_str}. "
            f"This is a confirmed solution — replay on future attempts."
        )

    def on_game_end(self, levels_completed: int, win_levels: int,
                    narrative_patterns: str, object_summary: str):
        """Store: What did I learn from this game session?"""
        if levels_completed > 0:
            self.store(
                f"ARC-AGI-3 {self.game_prefix}: Completed {levels_completed}/{win_levels} levels. "
                f"Key patterns: {narrative_patterns[:200]}"
            )

        # Abstract the patterns for cross-game learning
        if "consistent effect" in narrative_patterns:
            self.store(
                f"ARC-AGI-3 general: On {self.game_prefix}, interactive objects have "
                f"consistent per-click effects. {object_summary[:150]}"
            )
        if "STABLE" in narrative_patterns:
            self.store(
                f"ARC-AGI-3 general: On {self.game_prefix}, repeated clicking of "
                f"interactive objects produces changes but no level progress. "
                f"Need to find the correct SEQUENCE, not just the correct objects."
            )

        self.save()

    # ─── Full Context Assembly ───

    def build_layer3(self, situation: str) -> str:
        """Build Layer 3 context based on current situation description.

        This is the adaptive query — the situation determines what's retrieved.
        """
        results = self._search(situation, n=4)
        if results:
            seen = set()
            unique = [r for r in results if not (r[:60] in seen or seen.add(r[:60]))]
            return "FROM MEMORY (relevant to current situation):\n" + \
                   "\n".join(f"  • {t[:150]}" for t in unique[:4])
        return ""
