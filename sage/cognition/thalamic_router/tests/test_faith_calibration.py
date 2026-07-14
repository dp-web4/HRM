"""Tests for strict-mode faith trust calibration (SAGE_FAITH_TRUST_STRICT).

Proves the strict path reproduces Sprout's calibrated trusted set (ka59/r11l/re86)
instead of the broad live signal that over-granted trust to 22/25 games in v27.
The headline: feeding the gate the calibrated per-game inputs makes trust move on
exactly the 3 games whose committed WM predictor clears conf>=0.45 & disambig>=0.5
at 0.8B; every other game resolves to explore-noop.

Pure; runs under pytest or as `python3 test_faith_calibration.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from sage.cognition.thalamic_router.faith_calibration import (  # noqa: E402
    strict_trust_inputs, load_calibration, clear_cache, _UNTRUSTED,
)
from sage.cognition.thalamic_router.faith_portfolio import (  # noqa: E402
    FaithPortfolio, D_CONF_CUTOFF, D_DISAMBIG_CUTOFF, D_THETA_MULT,
)

# The calibrated trusted set + in-regime baselines from the v25 dump.
TRUSTED = {"ka59": 1.31, "r11l": 0.47, "re86": 1.03}


def test_trusted_games_carry_calibrated_inputs():
    clear_cache()
    for game, dep_resid in TRUSTED.items():
        c = strict_trust_inputs(game)
        assert c["dep_confidence"] >= D_CONF_CUTOFF, (game, c)
        assert c["dep_disambiguation"] >= D_DISAMBIG_CUTOFF, (game, c)
        assert abs(c["baseline_resid"] - dep_resid) < 1e-6, (game, c)


def test_untrusted_game_falls_below_cutoff():
    clear_cache()
    # m0r0/sp80/tu93 are conf==0 in v25 — WI-confident but no committed WM predictor.
    for game in ("m0r0", "sp80", "tu93", "bp35"):
        c = strict_trust_inputs(game)
        assert c["dep_confidence"] < D_CONF_CUTOFF, (game, c)


def test_borderline_game_is_untrusted_by_cutoff():
    clear_cache()
    # ar25: conf 0.333 (< 0.45). disambig 0.5 passes, but conf gates it out.
    c = strict_trust_inputs("ar25")
    assert not (c["dep_confidence"] >= D_CONF_CUTOFF
                and c["dep_disambiguation"] >= D_DISAMBIG_CUTOFF), c


def test_unknown_game_returns_untrusted():
    clear_cache()
    assert strict_trust_inputs("zzzz") == _UNTRUSTED


def test_strict_gate_reproduces_three_game_trusted_set():
    """The integration proof: across all 25 calibrated games, the gate moves trust
    (confirm/disconfirm) on exactly ka59/r11l/re86 and explore-noops the rest."""
    clear_cache()
    table = load_calibration()
    assert len(table) == 25, f"expected 25 calibrated games, got {len(table)}"
    moved = set()
    for game in table:
        cal = strict_trust_inputs(game)
        pf = FaithPortfolio()
        pf.add("c", "win via wm dynamics", plausibility=0.7, trust=0.30,
               strategy="win_via_wm_dynamics")
        # a tiny live residual: a trusted predictor would CONFIRM on it
        outcome = pf.update_trust_from_residual(
            "c", step_resid_px=0.10,
            dep_confidence=cal["dep_confidence"],
            dep_disambiguation=cal["dep_disambiguation"],
            baseline_resid=cal["baseline_resid"],
        )
        if outcome != "explore-noop":
            moved.add(game)
    assert moved == set(TRUSTED), f"trust moved on {sorted(moved)}, expected {sorted(TRUSTED)}"


def test_strict_theta_scale_applies_calibrated_baseline():
    """re86 baseline 1.03 → θ = 3.0 × 1.03 = 3.09: resid below θ confirms, above disconfirms."""
    clear_cache()
    cal = strict_trust_inputs("re86")
    theta = D_THETA_MULT * cal["baseline_resid"]
    assert abs(theta - 3.09) < 1e-6, theta

    pf = FaithPortfolio()
    pf.add("c", "h", plausibility=0.7, trust=0.30, strategy="win_via_wm_dynamics")
    assert pf.update_trust_from_residual(
        "c", 2.0, cal["dep_confidence"], cal["dep_disambiguation"],
        baseline_resid=cal["baseline_resid"]) == "confirm"          # 2.0 <= 3.09

    pf2 = FaithPortfolio()
    pf2.add("c", "h", plausibility=0.7, trust=0.30, strategy="win_via_wm_dynamics")
    assert pf2.update_trust_from_residual(
        "c", 4.0, cal["dep_confidence"], cal["dep_disambiguation"],
        baseline_resid=cal["baseline_resid"]) == "disconfirm"       # 4.0 > 3.09


def test_missing_table_degrades_to_untrusted():
    clear_cache()
    bad = Path("/nonexistent/sprout_v25.json")
    assert load_calibration(bad) == {}
    assert strict_trust_inputs("ka59", path=bad) == _UNTRUSTED
    clear_cache()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
