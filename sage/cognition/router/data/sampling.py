"""
SNARC-stratified sampling — PRD §4.7.D.

Quoting the PRD:

    Top SNARC quintile: 100% of ticks logged for training
    Middle quintiles: stratified sample at ~20% to control dataset balance
    Bottom quintile (idle/noop): ~5% sampled

The agent-zero hazard this defends against (PRD §0.2, §4.7.D):

    Without active sampling, the natural distribution drowns the rare
    events. The router would learn "noop is almost always right" before
    it learns anything else.

This module owns the salience formula and the keep/drop decision. It does
NOT know about on-disk format or record shape — those live in writer.py.

Design choices:

* **Salience formula**: `max(arousal, conflict, |reward|)` per PRD §4.7.A.
  These are the action-relevant SNARC dimensions. Surprise and novelty
  matter for input features but don't weight the training loss, per PRD.

* **Quintile boundaries**: computed from a rolling window (default 5000
  observations), not configured upfront. The natural SNARC distribution
  is unknown at Phase 0 start; a rolling window lets the sampler
  self-calibrate without a separate bootstrap phase.

* **Warmup behavior**: while fewer than `warmup` observations have been
  seen, EVERYTHING is kept. Short initial burst of data, then the
  sampler tightens. This matches the PRD's "ship with sampling on from
  day one" discipline without starving the first hour of data.

* **Determinism**: `seed` fixes the RNG; identical inputs → identical
  keep/drop decisions. Important for tests and for replaying a training
  run on archived raw data.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional


# PRD §4.7.D target retention per quintile (fraction kept).
# Quintile 0 = lowest salience; Quintile 4 = highest salience.
DEFAULT_QUINTILE_KEEP_RATES: List[float] = [0.05, 0.20, 0.20, 0.20, 1.00]


def salience_score(snarc: Mapping[str, float]) -> float:
    """Compute salience from a SNARC dict.

    PRD §4.7.A salience weight = `max(arousal, conflict, |reward|)`.

    Robust to missing keys — treats absent dims as zero. SNARC may arrive
    partially populated during Phase 0 (e.g. reward not yet wired on some
    machines), and we explicitly do NOT want to drop records in that case.
    """
    arousal = float(snarc.get("arousal", 0.0) or 0.0)
    conflict = float(snarc.get("conflict", 0.0) or 0.0)
    reward = float(snarc.get("reward", 0.0) or 0.0)
    return max(arousal, conflict, abs(reward))


@dataclass
class SamplingStats:
    """Per-quintile retention statistics for observability."""

    seen: int = 0
    kept: int = 0
    per_quintile_seen: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    per_quintile_kept: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])

    def keep_rate(self) -> float:
        return self.kept / self.seen if self.seen > 0 else 0.0

    def per_quintile_keep_rates(self) -> List[float]:
        return [
            (k / s if s > 0 else 0.0)
            for k, s in zip(self.per_quintile_kept, self.per_quintile_seen)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "keep_rate": self.keep_rate(),
            "per_quintile_seen": list(self.per_quintile_seen),
            "per_quintile_kept": list(self.per_quintile_kept),
            "per_quintile_keep_rates": self.per_quintile_keep_rates(),
        }


class SnarcStratifiedSampler:
    """Decide keep/drop per SNARC-stratified sampling (PRD §4.7.D).

    Parameters
    ----------
    keep_rates:
        Per-quintile retention probability, length-5, quintile 0 (lowest
        salience) to quintile 4 (highest). Defaults to PRD spec
        [0.05, 0.20, 0.20, 0.20, 1.00].
    window_size:
        Rolling window for quintile-boundary estimation. Larger window →
        more stable boundaries, slower to track drift. 5000 ≈ 1.4 hours
        at 1 Hz per machine.
    warmup:
        Number of observations to keep unconditionally before quintile
        boundaries are used. Guarantees a seed dataset even at cold-start.
    seed:
        RNG seed for reproducible keep/drop decisions. `None` → nondeterministic.

    Notes
    -----
    * Boundaries are recomputed on every call; O(N log N) in window size.
      5000 × O(N log N) ~ 60µs on a modern CPU — within the router's tick
      budget by orders of magnitude.
    * Top-quintile always keeps at rate 1.0 by default. If a caller sets
      that below 1.0, we honor it — the sampler is a policy enforcer, not
      a policy. But PRD §4.7.D is normative, so deviating from 1.0 on
      quintile 4 should be flagged at review.
    """

    def __init__(
        self,
        keep_rates: Optional[List[float]] = None,
        window_size: int = 5000,
        warmup: int = 100,
        seed: Optional[int] = None,
    ):
        if keep_rates is None:
            keep_rates = DEFAULT_QUINTILE_KEEP_RATES
        if len(keep_rates) != 5:
            raise ValueError(
                f"keep_rates must have 5 entries (one per quintile), got {len(keep_rates)}"
            )
        if not all(0.0 <= r <= 1.0 for r in keep_rates):
            raise ValueError(f"keep_rates must all be in [0, 1]: {keep_rates}")
        if window_size < 5:
            raise ValueError(f"window_size must be >= 5, got {window_size}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")

        self.keep_rates: List[float] = list(keep_rates)
        self.window_size: int = window_size
        self.warmup: int = warmup
        self._rng = random.Random(seed)
        self._window: Deque[float] = deque(maxlen=window_size)
        # Warmup tracked independently of stats — reset_stats must NOT
        # re-trigger warmup.
        self._total_observed: int = 0
        self.stats = SamplingStats()

    # ── main API ────────────────────────────────────────────────────

    def should_keep(self, snarc: Mapping[str, float]) -> bool:
        """Observe a SNARC vector and return True if this tick should be kept.

        Side effects: updates rolling window and stats. Call once per tick.
        """
        score = salience_score(snarc)
        self._window.append(score)
        self._total_observed += 1
        self.stats.seen += 1

        quintile = self._quintile_of(score)
        self.stats.per_quintile_seen[quintile] += 1

        if self._total_observed <= self.warmup:
            # Warmup: keep everything.
            self.stats.per_quintile_kept[quintile] += 1
            self.stats.kept += 1
            return True

        rate = self.keep_rates[quintile]
        keep = self._rng.random() < rate
        if keep:
            self.stats.per_quintile_kept[quintile] += 1
            self.stats.kept += 1
        return keep

    def reset_stats(self) -> None:
        """Zero the stats counter without disturbing the rolling window."""
        self.stats = SamplingStats()

    # ── introspection ──────────────────────────────────────────────

    def quintile_boundaries(self) -> List[float]:
        """Return current [q20, q40, q60, q80] cut points from the window.

        Empty / too-small window → [0, 0, 0, 0] (all scores land in q0).
        """
        if len(self._window) < 5:
            return [0.0, 0.0, 0.0, 0.0]
        sorted_scores = sorted(self._window)
        n = len(sorted_scores)
        return [
            sorted_scores[int(n * 0.20)],
            sorted_scores[int(n * 0.40)],
            sorted_scores[int(n * 0.60)],
            sorted_scores[int(n * 0.80)],
        ]

    # ── internals ──────────────────────────────────────────────────

    def _quintile_of(self, score: float) -> int:
        """Map salience score → quintile index 0-4 using current window.

        Uses strict `<` so that ties land in the LOWER quintile. This is
        deliberate: with a lot of zero-salience ticks (the dream state
        majority), we want them all anchored in quintile 0, not smeared.
        """
        cuts = self.quintile_boundaries()
        for i, cut in enumerate(cuts):
            if score < cut:
                return i
        return 4
