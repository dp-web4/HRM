"""
LLM Pool — dynamic registry of LLM temporal sensors.

Treats each LLM backend (OllamaIRP, IntrospectiveQwenIRP, MultiModelLoader)
as a temporal sensor with trust, health, and capability tracking. The
consciousness loop selects the best available model per-cycle instead of
being locked to a single hardcoded plugin.

Architecture:
    ┌──────────────────────────────────┐
    │          LLMPool                 │
    │  ┌─────────┐ ┌─────────┐        │
    │  │ Entry A  │ │ Entry B │  ...   │
    │  │ gemma3:4b│ │ phi4:14b│        │
    │  │ T2, 0.72 │ │ T1, 0.58│        │
    │  └─────────┘ └─────────┘        │
    │         select(context)          │
    └──────────────────────────────────┘
               ↕ periodic
    ┌──────────────────────────────────┐
    │  Ollama /api/tags (discovery)    │
    └──────────────────────────────────┘

Usage:
    pool = LLMPool()
    pool.discover_ollama()                   # auto-register available models
    pool.register(my_plugin, 'custom:7b')    # manual registration

    entry = pool.select({'metabolic_state': 'WAKE'})
    response = entry.plugin.get_response(prompt)
    pool.record_exchange(entry.model_name, latency_ms=120, success=True)
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMEntry:
    """One registered LLM backend — a temporal sensor with trust."""
    plugin: Any                     # OllamaIRP, IntrospectiveQwenIRP, MultiModelLoader
    model_name: str                 # 'gemma3:4b', 'qwen2.5-14b', etc.
    model_family: str = ''          # 'gemma3', 'qwen2.5'
    backend: str = 'ollama'         # 'ollama', 'local', 'multi'
    capability: Any = None          # ToolCapability (lazy import to avoid cycles)

    # Trust (EMA-updated per exchange)
    trust: float = 0.5
    latency_ema_ms: float = 0.0
    error_rate_ema: float = 0.0
    tool_success_rate: float = 0.5
    exchanges: int = 0

    # Health
    last_health_check: float = 0.0
    healthy: bool = True
    size_gb: float = 0.0

    # Tracking
    registered_at: float = field(default_factory=time.time)
    primary: bool = False  # True = raised model with identity context

    def score(self, context: Optional[Dict] = None) -> float:
        """Compute selection score. Higher = better candidate."""
        if not self.healthy:
            return -1.0

        # Tier score from capability
        tier_scores = {'T1': 1.0, 'T2': 0.7, 'T3': 0.4}
        tier = getattr(self.capability, 'tier', 'T3') if self.capability else 'T3'
        tier_score = tier_scores.get(tier, 0.4)

        # Normalize latency (cap at 10s for scoring)
        norm_latency = min(self.latency_ema_ms / 10000.0, 1.0) if self.latency_ema_ms > 0 else 0.5

        base_score = (
            0.4 * self.trust
            + 0.2 * (1.0 - norm_latency)
            + 0.2 * tier_score
            + 0.1 * self.tool_success_rate
            + 0.1 * (1.0 - self.error_rate_ema)
        )

        # Metabolic state modifiers
        if context:
            metabolic = context.get('metabolic_state', '')
            if metabolic == 'CRISIS':
                # In crisis, strongly prefer low latency
                base_score += 0.3 * (1.0 - norm_latency)
            elif metabolic == 'DREAM':
                # In dream, prefer highest capability tier
                base_score += 0.2 * tier_score
            elif metabolic == 'REST':
                # At rest, prefer most trusted (stable)
                base_score += 0.1 * self.trust

        return base_score

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for health endpoint / persistence."""
        return {
            'model_name': self.model_name,
            'model_family': self.model_family,
            'backend': self.backend,
            'tier': getattr(self.capability, 'tier', 'T3') if self.capability else 'T3',
            'grammar': getattr(self.capability, 'grammar_id', '') if self.capability else '',
            'trust': round(self.trust, 4),
            'latency_ema_ms': round(self.latency_ema_ms, 1),
            'error_rate_ema': round(self.error_rate_ema, 4),
            'tool_success_rate': round(self.tool_success_rate, 4),
            'exchanges': self.exchanges,
            'healthy': self.healthy,
            'size_gb': self.size_gb,
        }


class LLMPool:
    """
    Dynamic registry of LLM temporal sensors.

    Models are registered (manually or via Ollama discovery), scored on
    trust/latency/capability, and selected per-call by the consciousness
    loop. Each entry maintains independent trust metrics updated via EMA
    after every exchange.
    """

    EMA_ALPHA = 0.1  # Smoothing factor for trust/latency/error updates

    def __init__(self, ollama_host: str = 'http://localhost:11434'):
        self._entries: Dict[str, LLMEntry] = {}
        self._active_name: Optional[str] = None
        self._ollama_host = ollama_host

    # ── Registration ────────────────────────────────────────────────────

    def register(
        self,
        plugin: Any,
        model_name: str,
        backend: str = 'ollama',
        capability: Any = None,
        size_gb: float = 0.0,
        primary: bool = False,
    ) -> str:
        """Register an LLM plugin. Returns model_name as key.

        Args:
            primary: If True, this is the instance's raised model — the one
                that has been through the BECOMING_CURRICULUM and carries
                the instance's identity. Gets a trust boost to ensure it's
                preferred for conversation over discovered secondary models.
        """
        family = _extract_family(model_name)

        # Detect capability if not provided and backend is ollama
        if capability is None and backend == 'ollama':
            capability = self._detect_capability(model_name)

        entry = LLMEntry(
            plugin=plugin,
            model_name=model_name,
            model_family=family,
            backend=backend,
            capability=capability,
            size_gb=size_gb,
        )

        # Primary (raised) model gets trust boost — it has identity context
        # from raising sessions that secondary/discovered models lack.
        if primary:
            entry.trust = 0.8
            entry.primary = True

        self._entries[model_name] = entry

        # Auto-select if this is the first entry
        if self._active_name is None:
            self._active_name = model_name

        return model_name

    def unregister(self, model_name: str) -> bool:
        """Remove an LLM from the pool."""
        if model_name not in self._entries:
            return False
        del self._entries[model_name]
        if self._active_name == model_name:
            # Fall back to highest-trust remaining entry
            self._active_name = None
            if self._entries:
                best = max(self._entries.values(), key=lambda e: e.trust)
                self._active_name = best.model_name
        return True

    def list(self) -> List[LLMEntry]:
        """Return all registered entries."""
        return list(self._entries.values())

    def get(self, model_name: str) -> Optional[LLMEntry]:
        """Get a specific entry by model name."""
        return self._entries.get(model_name)

    def __len__(self) -> int:
        return len(self._entries)

    # ── Discovery ───────────────────────────────────────────────────────

    def discover_ollama(self, host: Optional[str] = None) -> List[str]:
        """
        Probe Ollama /api/tags for available models.

        - New models: auto-register with fresh OllamaIRP + capability detect
        - Missing models (were registered, no longer in tags): mark unhealthy
        - Existing models: confirm healthy

        Returns list of newly discovered model names.
        """
        ollama_host = host or self._ollama_host
        try:
            req = urllib.request.Request(f'{ollama_host}/api/tags', method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception:
            # Ollama not reachable — mark all ollama entries unhealthy
            for entry in self._entries.values():
                if entry.backend == 'ollama':
                    entry.healthy = False
            return []

        remote_models = {}
        for m in data.get('models', []):
            name = m.get('name', m.get('model', ''))
            size_bytes = m.get('size', 0)
            size_gb = round(size_bytes / 1024**3, 1) if size_bytes else 0.0
            if name:
                remote_models[name] = size_gb

        newly_discovered = []

        # Register new models
        for model_name, size_gb in remote_models.items():
            if model_name not in self._entries:
                plugin = self._create_ollama_plugin(model_name, ollama_host)
                if plugin is not None:
                    self.register(
                        plugin, model_name,
                        backend='ollama', size_gb=size_gb,
                    )
                    newly_discovered.append(model_name)

        # Update health of existing entries
        for name, entry in self._entries.items():
            if entry.backend != 'ollama':
                continue
            if name in remote_models:
                entry.healthy = True
                entry.last_health_check = time.time()
                if remote_models[name]:
                    entry.size_gb = remote_models[name]
            else:
                entry.healthy = False
                entry.last_health_check = time.time()

        return newly_discovered

    def _create_ollama_plugin(self, model_name: str, ollama_host: str) -> Any:
        """Create an OllamaIRP instance for a discovered model."""
        try:
            import importlib.util
            ollama_path = Path(__file__).parent / 'ollama_irp.py'
            spec = importlib.util.spec_from_file_location('ollama_irp', str(ollama_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            OllamaIRP = mod.OllamaIRP
            return OllamaIRP({
                'model_name': model_name,
                'ollama_host': ollama_host,
                'max_response_tokens': 250,
            })
        except Exception:
            return None

    def _detect_capability(self, model_name: str) -> Any:
        """Detect tool capability for a model (best-effort)."""
        try:
            from sage.tools.tool_capability import ToolCapability
            return ToolCapability.detect(model_name, self._ollama_host)
        except Exception:
            return None

    # ── Selection ───────────────────────────────────────────────────────

    def select(self, context: Optional[Dict] = None) -> Optional[LLMEntry]:
        """
        Pick the best available LLM for the current situation.

        Scores all healthy entries and returns the highest-scoring one.
        Updates self._active_name so subsequent calls to .active return it.
        """
        if not self._entries:
            return None

        candidates = [e for e in self._entries.values() if e.healthy]
        if not candidates:
            # All unhealthy — fall back to any entry
            candidates = list(self._entries.values())

        best = max(candidates, key=lambda e: e.score(context))
        self._active_name = best.model_name
        return best

    @property
    def active(self) -> Optional[LLMEntry]:
        """Currently active entry (last selected or first registered)."""
        if self._active_name and self._active_name in self._entries:
            return self._entries[self._active_name]
        return None

    @property
    def active_name(self) -> Optional[str]:
        return self._active_name

    # ── Health ──────────────────────────────────────────────────────────

    def health_check(self, model_name: Optional[str] = None) -> Dict[str, bool]:
        """
        Check health of entries. For Ollama backends, verifies the model
        is still listed in /api/tags.

        Returns dict of model_name → healthy.
        """
        if model_name:
            entries = {model_name: self._entries[model_name]} if model_name in self._entries else {}
        else:
            entries = self._entries

        # Batch: query Ollama once
        ollama_names = set()
        try:
            req = urllib.request.Request(f'{self._ollama_host}/api/tags', method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            for m in data.get('models', []):
                name = m.get('name', m.get('model', ''))
                if name:
                    ollama_names.add(name)
        except Exception:
            pass

        results = {}
        now = time.time()
        for name, entry in entries.items():
            if entry.backend == 'ollama':
                entry.healthy = name in ollama_names
            # Local/multi backends: check plugin is not None
            elif entry.backend in ('local', 'multi'):
                entry.healthy = entry.plugin is not None
            entry.last_health_check = now
            results[name] = entry.healthy

        return results

    # ── Trust Feedback ──────────────────────────────────────────────────

    def record_exchange(
        self,
        model_name: str,
        latency_ms: float,
        success: bool,
        tool_calls: int = 0,
        tool_successes: int = 0,
    ) -> None:
        """
        Update trust metrics after an LLM call (EMA).

        Called by consciousness._call_llm() after every exchange.
        """
        entry = self._entries.get(model_name)
        if entry is None:
            return

        a = self.EMA_ALPHA
        entry.exchanges += 1

        # Latency EMA
        entry.latency_ema_ms = (1 - a) * entry.latency_ema_ms + a * latency_ms

        # Error rate EMA
        error_val = 0.0 if success else 1.0
        entry.error_rate_ema = (1 - a) * entry.error_rate_ema + a * error_val

        # Trust: success raises it, failure lowers it
        trust_delta = 0.1 if success else -0.2
        entry.trust = max(0.0, min(1.0, entry.trust + a * trust_delta))

        # Tool success rate
        if tool_calls > 0:
            tool_rate = tool_successes / tool_calls
            entry.tool_success_rate = (1 - a) * entry.tool_success_rate + a * tool_rate

    # ── Persistence ─────────────────────────────────────────────────────

    def save_state(self, path: Path) -> None:
        """Save pool trust state to JSON (for cross-session persistence)."""
        state = {
            'active': self._active_name,
            'entries': {
                name: entry.to_dict()
                for name, entry in self._entries.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path) -> None:
        """
        Load saved trust state. Merges into existing entries —
        if a model is registered AND has saved state, restore trust metrics.
        Models in the save file but not registered are ignored (they'll be
        re-discovered by discover_ollama).
        """
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        saved_entries = state.get('entries', {})
        for name, entry in self._entries.items():
            if name in saved_entries:
                saved = saved_entries[name]
                saved_trust = saved.get('trust', entry.trust)
                # Primary (raised) model: never restore trust below its
                # initial level — it has identity context that secondary
                # models lack.
                if entry.primary:
                    entry.trust = max(saved_trust, entry.trust)
                else:
                    entry.trust = saved_trust
                entry.latency_ema_ms = saved.get('latency_ema_ms', entry.latency_ema_ms)
                entry.error_rate_ema = saved.get('error_rate_ema', entry.error_rate_ema)
                entry.tool_success_rate = saved.get('tool_success_rate', entry.tool_success_rate)
                entry.exchanges = saved.get('exchanges', entry.exchanges)

        # Restore active selection — but prefer primary model if it exists
        primary_entries = [n for n, e in self._entries.items() if e.primary]
        if primary_entries:
            self._active_name = primary_entries[0]
        else:
            saved_active = state.get('active')
            if saved_active and saved_active in self._entries:
                self._active_name = saved_active

    # ── String representation ───────────────────────────────────────────

    def __repr__(self) -> str:
        names = [f"{e.model_name}({'*' if e.model_name == self._active_name else ''})"
                 for e in self._entries.values()]
        return f"LLMPool([{', '.join(names)}])"


# ── Utilities ───────────────────────────────────────────────────────────

def _extract_family(model_name: str) -> str:
    """Extract model family: 'gemma3:12b' → 'gemma3'."""
    base = model_name.split(':')[0]
    return base.rstrip('-').rstrip('_')
