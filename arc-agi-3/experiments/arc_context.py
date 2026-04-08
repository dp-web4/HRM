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

import os
import sys
import time
import requests
from typing import Optional, List

MEMBOT_URL = "http://localhost:8000"
FLEET_LEARNING_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "shared-context",
    "arc-agi-3", "fleet-learning")

# Try to import multi-cart federation (from membot)
_multi_cart = None
_federate = None
try:
    membot_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "membot")
    if os.path.isdir(membot_path) and membot_path not in sys.path:
        sys.path.insert(0, membot_path)
    import multi_cart as _multi_cart
    import federate as _federate
except ImportError:
    pass  # multi-cart not available — fall back to HTTP membot only


class ContextConstructor:
    """Builds situation-relevant context from all memory layers.

    Queries membot adaptively based on current game state.
    Caches results to avoid redundant queries.
    Stores new discoveries for future sessions.
    """

    def __init__(self, game_prefix: str):
        self.game_prefix = game_prefix
        self.membot_available = None
        self.fleet_available = False
        self.query_cache = {}  # query → (result, timestamp)
        self.cache_ttl = 60    # cache for 1 minute (short — context evolves fast)
        self.attempt_count = 0

        # Ensure cartridge is mounted
        self._mount_cartridge()
        # Mount fleet carts via multi-cart (works without membot server)
        self._mount_fleet()

    def new_attempt(self):
        """Clear cache for a new attempt — force fresh membot queries."""
        self.query_cache.clear()
        self.attempt_count += 1

    def _mount_cartridge(self):
        """Mount the sage cartridge if membot is running."""
        try:
            resp = requests.post(f"{MEMBOT_URL}/api/mount",
                json={"name": "sage"}, timeout=5)
            self.membot_available = resp.status_code == 200
        except Exception:
            self.membot_available = False

    def _mount_fleet(self):
        """Mount fleet learning carts via multi-cart (no server needed)."""
        if _federate is None or _multi_cart is None:
            return
        fleet_dir = FLEET_LEARNING_DIR
        if not os.path.isdir(fleet_dir):
            return
        try:
            # Try to migrate JSONL to carts (ok if already done or fails)
            try:
                _federate.migrate_jsonl(fleet_dir, in_place=True)
            except Exception:
                pass  # carts may already exist
            # Load all fleet carts
            result = _federate.load_fleet(fleet_dir)
            n = result.get("total_patterns", 0)
            machines = result.get("machines", [])
            self.fleet_available = len(machines) > 0
            self.fleet_machines = machines
            self.fleet_patterns = n
        except Exception:
            self.fleet_available = False

    def _search_fleet(self, query: str, n: int = 3) -> List[str]:
        """Search fleet carts via multi-cart. Returns text results with attribution."""
        if not self.fleet_available or _multi_cart is None:
            return []
        try:
            # Ensure carts are still mounted (may have been unmounted by other code)
            mounts = _multi_cart.list_mounts()
            if not mounts:  # list_mounts returns a list
                self._mount_fleet()

            response = _multi_cart.search(query, top_k=n,
                                          scope="all", role_filter="federated")
            # Response is a dict with "results" key
            hits = response.get("results", []) if isinstance(response, dict) else response
            return [f"[{r.get('cart_id', '?')}#{r.get('address', '?')}] {r.get('text', '')[:200]}"
                    for r in hits if r.get("score", 0) > 0.25]
        except Exception as e:
            return []

    def _search(self, query: str, n: int = 3) -> list:
        """Search all available backends. Returns combined text results."""
        results = []

        # Fleet carts (local, no server needed)
        results.extend(self._search_fleet(query, n))

        # Membot HTTP (if server running)
        if not self.membot_available:
            return results

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
                http_results = [r["text"] for r in resp.json().get("results", [])
                               if r.get("score", 0) > 0.35]
                results.extend(http_results)
        except Exception:
            pass

        self.query_cache[query] = (results, now)
        return results

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
        """Store: What did I learn from this game session?

        Stores at multiple specificity levels so future queries find it:
        - Game-specific: "lp85: cyan and teal are buttons, cycle colors"
        - Category-specific: "rotation puzzle: find cycle length first"
        - General: "repeated clicking without progress = wrong sequence"
        """
        # Game-specific learning (found by on_game_start query)
        self.store(
            f"ARC-AGI-3 {self.game_prefix} attempt {self.attempt_count}: "
            f"{levels_completed}/{win_levels} levels. "
            f"Objects: {object_summary[:200]}. "
            f"Patterns: {narrative_patterns[:200]}"
        )

        # Category learning (found by on_probe_complete query)
        if "consistent effect" in narrative_patterns:
            self.store(
                f"ARC-AGI-3 puzzle strategy: On {self.game_prefix}, interactive objects have "
                f"consistent per-click effects. Each click causes the same change. "
                f"This suggests a cycling/rotation mechanic."
            )

        # General learning (found by on_stuck query)
        if "STABLE" in narrative_patterns:
            self.store(
                f"ARC-AGI-3 stuck strategy: On {self.game_prefix}, clicking interactive "
                f"objects causes grid changes but no level progress. The objects are correct "
                f"but the SEQUENCE is wrong. Try: different counts, different combinations, "
                f"different ordering. Don't just alternate — search systematically."
            )
        if "WARNING" in narrative_patterns:
            self.store(
                f"ARC-AGI-3 perseveration warning: On {self.game_prefix}, repeated clicking "
                f"({narrative_patterns.split('WARNING')[1][:100] if 'WARNING' in narrative_patterns else ''}) "
                f"Change approach: try counting clicks, try only one button at a time, "
                f"try clicking in a specific order."
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
