"""Tests for MRH-scoped veracity (faith_portfolio v0.2).

Validates dp's generalization (reward → V3.Veracity*MRH = "demonstrated success
in similar contexts") + CBP's review (τ as one readout of ATP): veracity
aggregates over similar MRH contexts, transfers coarse→fine (the prior-transfer
lever), confirm/disconfirm-only (silence never moves it), and τ(mrh,budget)
modulates the persist gate.

Pure; runs under pytest or as `python3 test_veracity_mrh.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from sage.cognition.thalamic_router.faith_portfolio import (  # noqa: E402
    VeracityStore, FaithPortfolio, tau, _mrh_similarity,
)

LOGI = frozenset({"class:logistics"})
SCENARIO_L0 = frozenset({"game:scenario", "lvl:0", "class:logistics"})
NOVEL_LOGI = frozenset({"game:novel", "lvl:0", "class:logistics"})
PUZZLE = frozenset({"game:zz", "class:puzzle"})


def test_no_evidence_returns_prior():
    s = VeracityStore(prior=0.3)
    assert s.veracity("deliver_via_helper", SCENARIO_L0) == 0.3


def test_confirm_raises_disconfirm_lowers():
    s = VeracityStore()
    base = s.veracity("strat", SCENARIO_L0)
    for _ in range(4):
        s.record("strat", SCENARIO_L0, "confirm")
    hi = s.veracity("strat", SCENARIO_L0)
    assert hi > base, (hi, base)
    for _ in range(4):
        s.record("strat", SCENARIO_L0, "disconfirm")
    lo = s.veracity("strat", SCENARIO_L0)
    assert lo < hi, (lo, hi)


def test_coarse_to_fine_transfer():
    """Veracity earned in a BROAD context transfers to a finer, NOVEL context
    that contains it — the prior-transfer lever (works where a level counter can't)."""
    s = VeracityStore()
    for _ in range(5):
        s.record("deliver_via_helper", LOGI, "confirm")     # proven broadly
    v_novel = s.veracity("deliver_via_helper", NOVEL_LOGI)   # never seen this game
    assert v_novel > 0.6, f"broad veracity should transfer to novel logistics ctx, got {v_novel}"


def test_unrelated_context_no_transfer():
    s = VeracityStore()
    for _ in range(5):
        s.record("deliver_via_helper", LOGI, "confirm")
    assert s.veracity("deliver_via_helper", PUZZLE) == s.prior  # logistics ⊄ puzzle ctx


def test_similarity_containment():
    assert _mrh_similarity(SCENARIO_L0, LOGI) == 1.0            # broad ⊂ query -> full
    assert _mrh_similarity(LOGI, SCENARIO_L0) < 1.0             # query narrower than stored
    assert _mrh_similarity(PUZZLE, LOGI) == 0.0             # disjoint
    assert _mrh_similarity(SCENARIO_L0, frozenset()) == 1.0     # global ctx applies (weak prior)


def test_tau_budget_and_breadth():
    # scarce budget raises τ; ample budget lowers it
    assert tau(SCENARIO_L0, budget=0.0) > tau(SCENARIO_L0, budget=1.0)
    # broad context -> slightly lower bar than a fine one (≥3 tags adds)
    assert tau(frozenset({"class:logistics"})) < tau(SCENARIO_L0)
    # bounded
    assert 0.05 <= tau(SCENARIO_L0, budget=0.0) <= 0.95


def test_portfolio_seeds_trust_from_store():
    """A candidate whose strategy has cross-context veracity starts ABOVE the
    cold default (transfer at portfolio construction)."""
    s = VeracityStore()
    for _ in range(5):
        s.record("deliver_via_helper", LOGI, "confirm")
    p = FaithPortfolio(veracity_store=s)
    c = p.add("c1", "stage on walls; helper delivers", plausibility=0.7,
              strategy="deliver_via_helper", mrh=NOVEL_LOGI)
    assert c.trust > 0.6, f"candidate trust should be seeded from transferred veracity, got {c.trust}"


def test_portfolio_record_accrues_to_store_and_silence_doesnt():
    s = VeracityStore()
    p = FaithPortfolio(veracity_store=s)
    p.add("c1", "h", strategy="strat", mrh=SCENARIO_L0)
    before = s.veracity("strat", SCENARIO_L0)
    # silence: no record() -> store unchanged (faith rule)
    assert s.veracity("strat", SCENARIO_L0) == before
    p.record("c1", "confirm")
    assert s.veracity("strat", SCENARIO_L0) > before
    # should_commit_in uses τ(mrh, budget)
    for _ in range(6):
        p.record("c1", "confirm")
    assert p.should_commit_in(SCENARIO_L0, budget=0.8)


def test_backward_compat_no_store():
    """Existing API (no store) unchanged: trust evolves on the candidate only."""
    p = FaithPortfolio()
    c = p.add("c1", "h", plausibility=0.7, trust=0.2)
    assert c.trust == 0.2
    for _ in range(6):
        p.record("c1", "confirm")
    assert p.should_commit() and p.veracity_of("c1") == c.trust


def test_graded_d_trusted_confirm_unfreezes_trust():
    """The binding-gap fix: a trusted predictor that keeps being RIGHT confirms,
    moving trust off the 0.30 seed (Sprout/Legion's frozen-trust gap)."""
    p = FaithPortfolio()
    c = p.add("c1", "h", plausibility=0.85, trust=0.30)
    assert c.trust == 0.30
    for _ in range(3):
        action = p.update_trust_from_residual("c1", step_resid_px=0.5,
                                              dep_confidence=0.6, dep_disambiguation=1.0)
        assert action == "confirm", action
    assert c.trust > 0.30, "trust must unfreeze on trusted-confirm"


def test_graded_d_trusted_miss_disconfirms():
    p = FaithPortfolio()
    c = p.add("c1", "h", plausibility=0.85, trust=0.6)
    action = p.update_trust_from_residual("c1", step_resid_px=20.0,   # ≫ θ
                                          dep_confidence=0.6, dep_disambiguation=1.0)
    assert action == "disconfirm" and c.trust < 0.6


def test_graded_d_low_confidence_is_explore_not_abandon():
    """A big miss against an UNtrusted predictor = 'never modeled this' → no trust
    move (the 2nd not-falsification case, Sprout's insight)."""
    p = FaithPortfolio()
    c = p.add("c1", "h", plausibility=0.85, trust=0.30)
    action = p.update_trust_from_residual("c1", step_resid_px=99.0,
                                          dep_confidence=0.2, dep_disambiguation=0.33)
    assert action == "explore-noop" and c.trust == 0.30


def test_graded_d_makes_commit_faith_reachable():
    """End-to-end of the fix: a high-seed candidate (best_score 0.255, below the
    commit_faith τ=0.55) becomes commit-reachable once trusted confirms unfreeze
    trust. This is exactly what Sprout+Legion found unreachable with frozen trust."""
    p = FaithPortfolio()
    p.add("c1", "h", plausibility=0.85, trust=0.30)
    assert p.best_score() < 0.55                      # frozen-trust ceiling (0.255)
    for _ in range(4):
        p.update_trust_from_residual("c1", step_resid_px=0.5,
                                     dep_confidence=0.6, dep_disambiguation=1.0)
    assert p.best_score() > 0.55, f"trusted confirms should lift best_score past commit τ, got {p.best_score()}"


def test_graded_d_baseline_theta():
    """θ scales with the predictor's in-regime baseline (3×). A miss between the
    flat fallback and 3×baseline disconfirms for a tight predictor, confirms for a loose one."""
    p = FaithPortfolio()
    p.add("tight", "h", plausibility=0.7, trust=0.5)
    p.add("loose", "h2", plausibility=0.7, trust=0.5)
    # resid 2.0 px: tight baseline 0.47 → θ=1.41 → miss (disconfirm); loose baseline 1.03 → θ=3.09 → ok (confirm)
    assert p.update_trust_from_residual("tight", 2.0, 0.6, 1.0, baseline_resid=0.47) == "disconfirm"
    assert p.update_trust_from_residual("loose", 2.0, 0.6, 1.0, baseline_resid=1.03) == "confirm"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}"); ok += 1
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    return ok == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
