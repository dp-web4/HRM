"""
MemoryCartridgeIRP — Semantic memory via membot brain cartridges.

IRP plugin that searches membot cartridges for semantically relevant
context during the consciousness loop's attention phase.

EXPERIMENT: Testing whether embedding-based retrieval (membot) provides
materially better context recall than keyword-based FTS5 (SNARC).
See: private-context/plans/membot-integration-experiment-2026-03-26.md

Membot provides:
  - 768-dim Nomic embeddings with cosine similarity
  - Binary Hamming codes (96 bytes/pattern) for fast approximate matching
  - Keyword reranking on top of blended scores
  - Cartridge portability (save/load .npz)

Integration:
  - init_state: mount per-project cartridge
  - step: search with attention target content
  - energy: inverse of best search score
  - halt: when search scores plateau

Membot runs as MCP server (HTTP on localhost:8000 or stdio).

2026-03-26 — Experiment Phase 1 (plumbing)
"""

import time
import json
import logging
import os
from typing import Any, Dict, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from sage.irp.base import IRPPlugin, IRPState

log = logging.getLogger("sage.membot_irp")


class MemoryCartridgeIRP(IRPPlugin):
    """
    IRP plugin wrapping membot's semantic search.

    Config:
        membot_url: HTTP endpoint (default: http://localhost:8000)
        cartridge_name: name of cartridge to mount (default: auto from project hash)
        top_k: number of search results (default: 3)
        min_score: minimum relevance score to consider (default: 0.3)
        project_hash: SNARC-compatible project hash for cartridge naming
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.membot_url = config.get('membot_url', 'http://localhost:8000')
        self.cartridge_name = config.get('cartridge_name', None)
        self.top_k = config.get('top_k', 3)
        self.min_score = config.get('min_score', 0.3)
        self.project_hash = config.get('project_hash', '')
        self._mounted = False
        self._last_scores = []

    # REST API endpoint map (FastMCP 3.x doesn't expose /mcp/v1/tools/call)
    _REST_MAP = {
        "mount_cartridge": ("POST", "/api/mount", lambda a: a),
        "memory_search": ("POST", "/api/search", lambda a: a),
        "memory_store": ("POST", "/api/store", lambda a: a),
        "save_cartridge": ("POST", "/api/save", lambda a: {}),
        "get_status": ("GET", "/api/status", lambda a: None),
    }

    def _call_membot(self, tool: str, args: Dict[str, Any]) -> Optional[str]:
        """Call a membot tool via REST API."""
        if not HAS_REQUESTS:
            log.warning("requests not installed — membot unavailable")
            return None

        route = self._REST_MAP.get(tool)
        if not route:
            log.warning(f"membot: unknown tool {tool}")
            return None

        method, path, arg_fn = route
        url = f"{self.membot_url}{path}"

        try:
            if method == "GET":
                resp = requests.get(url, timeout=10)
            else:
                payload = arg_fn(args)
                resp = requests.post(url, json=payload, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                # REST endpoints return {"status": "ok", "result": "..."} or similar
                if isinstance(data, dict):
                    if "result" in data:
                        return str(data["result"])
                    return json.dumps(data)
                return str(data)
            else:
                log.warning(f"membot {tool} returned {resp.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            log.debug("membot not reachable")
            return None
        except Exception as e:
            log.warning(f"membot call failed: {e}")
            return None

    def _ensure_mounted(self) -> bool:
        """Mount cartridge if not already mounted."""
        if self._mounted:
            return True

        name = self.cartridge_name or f"snarc-{self.project_hash}"
        result = self._call_membot("mount_cartridge", {"name": name})

        if result and "Mounted" in result:
            self._mounted = True
            log.info(f"membot cartridge mounted: {name}")
            return True
        elif result and "not found" in result:
            # No cartridge yet — create an empty one
            log.info(f"membot cartridge '{name}' not found — will be created on first store")
            return False
        else:
            log.debug(f"membot mount failed: {result}")
            return False

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """Initialize: attempt to mount the project cartridge."""
        self._ensure_mounted()
        return IRPState(
            x={"query": "", "results": [], "scores": []},
            step_idx=0,
            energy_val=1.0,
            meta={"mounted": self._mounted},
        )

    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """Search membot with the current query."""
        query = state.x.get("query", "")
        if not query or not self._mounted:
            return state

        t0 = time.time()
        result = self._call_membot("memory_search", {
            "query": query,
            "top_k": self.top_k,
        })
        elapsed_ms = (time.time() - t0) * 1000

        results = []
        scores = []
        if result:
            # Parse membot's text response into structured results
            for line in result.split('\n'):
                line = line.strip()
                if not line or line.startswith('---') or line.startswith('Search'):
                    continue
                # Membot format: "N. [score] text..."
                if line[0].isdigit() and '. ' in line:
                    try:
                        parts = line.split('] ', 1)
                        if len(parts) == 2:
                            score_str = parts[0].split('[')[-1]
                            score = float(score_str)
                            text = parts[1].strip()
                            if score >= self.min_score:
                                results.append(text[:300])
                                scores.append(score)
                    except (ValueError, IndexError):
                        pass

        self._last_scores = scores
        best_score = max(scores) if scores else 0.0

        new_state = IRPState(
            x={"query": query, "results": results, "scores": scores},
            step_idx=state.step_idx + 1,
            energy_val=1.0 - best_score,  # Lower energy = better match
            meta={
                "mounted": self._mounted,
                "elapsed_ms": elapsed_ms,
                "n_results": len(results),
                "best_score": best_score,
            },
        )
        return new_state

    def energy(self, state: IRPState) -> float:
        """Energy = 1 - best_score. Lower is better."""
        return state.energy_val or 1.0

    def halt(self, state: IRPState) -> bool:
        """Halt after one search step (no iterative refinement for retrieval)."""
        return state.step_idx >= 1

    def store(self, content: str, tags: str = "") -> bool:
        """Store content in the mounted cartridge."""
        if not self._ensure_mounted():
            # Try creating the cartridge first
            name = self.cartridge_name or f"snarc-{self.project_hash}"
            # Store will work if writable mode is enabled
            pass

        result = self._call_membot("memory_store", {
            "content": content,
            "tags": tags,
        })
        return result is not None and "Stored" in (result or "")

    def save(self) -> bool:
        """Persist the cartridge to disk."""
        result = self._call_membot("save_cartridge", {})
        return result is not None and "Saved" in (result or "")
