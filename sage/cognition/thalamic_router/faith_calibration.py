"""Strict-mode trust calibration for the faith gate (SAGE_FAITH_TRUST_STRICT).

WHY THIS EXISTS
---------------
Broad mode (default v27, SAGE_FAITH_TRUST_UPDATE=1) feeds the gate the LIVE WM
accessor's signal: `dep_confidence` = avg confidence over *all* causal-rules with
evidence, and `baseline_resid` = the WM's *own running mean* of residuals
(`GameWorldModel.predict_residual_now`). Sprout's v27 result
(`forum/sprout-v27-faith-trust-engagement-result-2026-05-24.md`) showed this
over-grants trust: trust moved on **22/25** games on an 0.8B model that wins
**2/25**. Two compounding looseness sources, both traced to the wire (not theory):

  1. avg-causal-rule-confidence is BROAD (high on most games because the stack
     harvests rules broadly) — a different, looser signal than the v25
     *dependency* confidence the θ/cutoff was calibrated on; and
  2. a self-referential running-mean baseline means "confirm" only requires
     beating the WM's own average, so 5-7px misses still confirm on high-variance
     games.

STRICT MODE
-----------
Strict mode replaces the broad live trust-grant signal with the CALIBRATED
per-game v25 values from Sprout's
`experiments/explorer-2026-05-20/results/sprout_v25_dep_resid_distribution.json`:

  - `dep_confidence` / `dep_disambiguation` ← the v25 in-regime dependency
    confidence (only ka59/r11l/re86 clear conf>=0.45 & disambig>=0.5 at 0.8B), so
    untrusted games resolve to explore-noop instead of accruing trust; and
  - `baseline_resid` ← the v25 in-regime `dep_resid_px` (the gate multiplies by
    D_THETA_MULT, reproducing per_trusted_theta_px: ka59 3.93, r11l 1.41, re86 3.09).

The LIVE per-step residual (`step_resid_px`) is KEPT — it is still the real-time
signal compared against the calibrated θ. Only the trust-GRANT decision and the θ
scale come from calibration. This is the narrow-but-strict arm of the v28
broad-vs-strict A/B that Sprout's caveat called for ("you'd need the stricter v25
dependency cutoff wired to test the scale hypothesis honestly").

MODEL SCOPE
-----------
The calibration table is model-specific (qwen3.5:0.8b). Apply strict mode on that
model for a valid A/B against the broad-0.8b result. On other models it is an
approximation until that model is re-calibrated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

# Default location of Sprout's v25 calibration dump (repo-relative). Override with
# SAGE_FAITH_CALIBRATION_PATH (used by tests + future per-model tables).
_DEFAULT_REL = "experiments/explorer-2026-05-20/results/sprout_v25_dep_resid_distribution.json"

# An untrusted game returns conf below D_CONF_CUTOFF so the gate resolves to
# explore-noop (no trust move) — the correct "I-never-modeled-this" semantics.
_UNTRUSTED = {"dep_confidence": 0.0, "dep_disambiguation": 0.0, "baseline_resid": None}

# Module-level cache keyed by resolved path so tests can swap tables.
_CACHE: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}


def _calibration_path() -> Path:
    override = os.environ.get("SAGE_FAITH_CALIBRATION_PATH")
    if override:
        return Path(override)
    # this file: <repo>/sage/cognition/thalamic_router/faith_calibration.py
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / _DEFAULT_REL


def load_calibration(path: Optional[Path] = None) -> Dict[str, Dict[str, Optional[float]]]:
    """Load + cache the per-game strict trust inputs from the v25 dump.

    Returns {game: {dep_confidence, dep_disambiguation, baseline_resid}}. Maps the
    v25 `dep_resid_px` (in-regime aggregate error) to `baseline_resid` (the gate
    scales it by D_THETA_MULT). Missing/null fields degrade to untrusted.
    """
    p = path if path is not None else _calibration_path()
    key = str(p)
    if key in _CACHE:
        return _CACHE[key]
    table: Dict[str, Dict[str, Optional[float]]] = {}
    try:
        raw = json.loads(Path(p).read_text())
        for game, v in (raw.get("per_game") or {}).items():
            table[game] = {
                "dep_confidence": float(v.get("dep_confidence") or 0.0),
                "dep_disambiguation": float(v.get("dep_disambiguation") or 0.0),
                # in-regime aggregate residual → the gate's per-predictor baseline
                "baseline_resid": (None if v.get("dep_resid_px") is None
                                   else float(v["dep_resid_px"])),
            }
    except (OSError, ValueError, json.JSONDecodeError):
        # No table → every game untrusted → strict mode grants no trust (safe).
        table = {}
    _CACHE[key] = table
    return table


def strict_trust_inputs(game: str, path: Optional[Path] = None) -> Dict[str, Optional[float]]:
    """Calibrated (dep_confidence, dep_disambiguation, baseline_resid) for `game`.

    Unknown or untrusted games return the _UNTRUSTED triple so the faith gate
    resolves them to explore-noop. The caller keeps the LIVE step residual.
    """
    return load_calibration(path).get(game, _UNTRUSTED).copy()


def clear_cache() -> None:
    """Test helper — drop the memoized tables."""
    _CACHE.clear()
