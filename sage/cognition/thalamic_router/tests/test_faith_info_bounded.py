"""Tests for SAGE_FAITH_INFO_BOUNDED trust update law.

The flag changes the confirm-trust update to use the rule's per-event
information weight as the trust target. A rule whose residuals concentrate
in a single bucket (degenerate noop-predicting rule) caps trust at near-zero;
a rule with diverse residuals accumulates trust normally.

Test data uses the actual residual sequences observed in
`sweep_results/nomad_v37full_gemma4e2b_20260601_0754.log` on dc22 and cn04
to validate the fix on real production data.

Diagnosis: `forum/nomad-dc22-calibration-diagnosis-2026-06-01.md`.
"""

from __future__ import annotations
import os

from sage.cognition.thalamic_router.faith_portfolio import (
    FaithPortfolio, D_CONF_CUTOFF, D_DISAMBIG_CUTOFF,
)


def _add(p: FaithPortfolio, cid: str, init_trust: float = 0.30) -> None:
    p.add(cid=cid, hypothesis=f"test rule {cid}",
          plausibility=0.5, trust=init_trust,
          strategy=cid, mrh=frozenset(["game:test"]))


def _replay(p: FaithPortfolio, cid: str, resids: list, conf: float = 0.50):
    """Apply update_trust_from_residual once per residual, all with high enough
    confidence that the trusted-predictor branch fires."""
    for r in resids:
        p.update_trust_from_residual(
            cid=cid,
            step_resid_px=r,
            dep_confidence=conf,
            dep_disambiguation=1.0,
            baseline_resid=None,
        )


# Real v37 dc22 rule[CLICK] residuals (FAITH-BORN events with the predictor
# trusted, i.e. confirm/disconfirm class — 17 confirms at resid=0 or 1.67).
DC22_CLICK_RESIDS = [
    1.67, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
]

# Real v37 cn04 rule[DOWN] residuals — informative mix of 0 / 2 / 4 / 5px
# (with some disconfirms at resid=5 that drop trust). Approximated from log.
CN04_DOWN_RESIDS = [
    0.00, 0.00, 0.00, 4.00, 0.00, 4.00, 0.00, 4.00, 0.00,
    5.00,  # > theta=4, disconfirm
    0.00, 5.00, 0.00, 4.00, 0.00, 4.00, 0.00, 5.00,
    2.00, 0.00, 2.00, 0.00, 2.00, 0.00,
]


def test_default_off_preserves_existing_symmetric_update():
    """When SAGE_FAITH_INFO_BOUNDED is unset, the symmetric update fires
    normally — dc22's degenerate confirms accumulate trust to high values."""
    os.environ.pop("SAGE_FAITH_INFO_BOUNDED", None)
    p = FaithPortfolio(commit_tau=0.45)
    _add(p, "dc22__rule__CLICK")
    _replay(p, "dc22__rule__CLICK", DC22_CLICK_RESIDS)
    final = p.candidates["dc22__rule__CLICK"].trust
    # Symmetric update with 17 confirms drives trust very close to 1.0
    assert final > 0.95, (
        f"Default symmetric update should saturate near 1.0; got {final:.3f}"
    )


def test_info_bounded_caps_degenerate_dc22_rule():
    """With SAGE_FAITH_INFO_BOUNDED=1, dc22's degenerate rule[CLICK] (all
    residuals at resid=0) stays at or below the commit threshold."""
    os.environ["SAGE_FAITH_INFO_BOUNDED"] = "1"
    try:
        p = FaithPortfolio(commit_tau=0.45)
        _add(p, "dc22__rule__CLICK")
        _replay(p, "dc22__rule__CLICK", DC22_CLICK_RESIDS)
        final = p.candidates["dc22__rule__CLICK"].trust
        # Degenerate rule should not cross commit_tau under info-bounded update.
        assert final < 0.55, (
            f"Info-bounded should cap dc22 degenerate rule below ~commit_tau; "
            f"got {final:.3f}"
        )
    finally:
        os.environ.pop("SAGE_FAITH_INFO_BOUNDED", None)


def test_info_bounded_preserves_informative_cn04_rule():
    """With SAGE_FAITH_INFO_BOUNDED=1, cn04's informative rule[DOWN] (mixed
    residuals at 0, 2, 4, 5px) accumulates trust normally."""
    os.environ["SAGE_FAITH_INFO_BOUNDED"] = "1"
    try:
        p = FaithPortfolio(commit_tau=0.45)
        _add(p, "cn04__rule__DOWN")
        _replay(p, "cn04__rule__DOWN", CN04_DOWN_RESIDS)
        final = p.candidates["cn04__rule__DOWN"].trust
        # Informative rule should accumulate trust above commit_tau under
        # info-bounded update (its diverse residuals carry near-1 bit each).
        assert final > 0.45, (
            f"Info-bounded should preserve cn04 informative rule above "
            f"commit_tau; got {final:.3f}"
        )
    finally:
        os.environ.pop("SAGE_FAITH_INFO_BOUNDED", None)


def test_info_bounded_separates_degenerate_from_informative():
    """End-to-end separation check: in the same flag-on portfolio, dc22's
    degenerate rule should end up well below cn04's informative rule."""
    os.environ["SAGE_FAITH_INFO_BOUNDED"] = "1"
    try:
        p = FaithPortfolio(commit_tau=0.45)
        _add(p, "dc22__rule__CLICK")
        _add(p, "cn04__rule__DOWN")
        _replay(p, "dc22__rule__CLICK", DC22_CLICK_RESIDS)
        _replay(p, "cn04__rule__DOWN", CN04_DOWN_RESIDS)
        dc22_trust = p.candidates["dc22__rule__CLICK"].trust
        cn04_trust = p.candidates["cn04__rule__DOWN"].trust
        separation = cn04_trust - dc22_trust
        assert separation > 0.10, (
            f"Info-bounded should clearly separate informative from degenerate; "
            f"dc22={dc22_trust:.3f} cn04={cn04_trust:.3f} sep={separation:+.3f}"
        )
    finally:
        os.environ.pop("SAGE_FAITH_INFO_BOUNDED", None)


def test_small_sample_full_weight():
    """Before min_samples observations, info weight is 1.0 — trust accumulates
    normally. Lets candidates bootstrap before the info-weighting kicks in."""
    os.environ["SAGE_FAITH_INFO_BOUNDED"] = "1"
    try:
        p = FaithPortfolio(commit_tau=0.45)
        _add(p, "test__rule__X")
        # 2 confirms — below min_samples=3 of the info weight estimator.
        _replay(p, "test__rule__X", [0.0, 0.0])
        c = p.candidates["test__rule__X"]
        # With full weight on 2 confirms, trust should reach ~0.30 + 0.35*(1-0.30)*(...) twice
        assert c.confirms == 2
        assert c.trust > 0.40, f"Bootstrap confirms should move trust normally; got {c.trust:.3f}"
    finally:
        os.environ.pop("SAGE_FAITH_INFO_BOUNDED", None)


def test_disconfirms_still_full_strength():
    """Disconfirms are NOT info-weighted — strong negative evidence stays strong."""
    os.environ["SAGE_FAITH_INFO_BOUNDED"] = "1"
    try:
        p = FaithPortfolio(commit_tau=0.45)
        _add(p, "test__rule__X", init_trust=0.80)
        # All disconfirms — trust must drop sharply
        _replay(p, "test__rule__X", [10.0, 10.0, 10.0])
        final = p.candidates["test__rule__X"].trust
        assert final < 0.20, (
            f"Disconfirms must still halve trust under info-bounded; got {final:.3f}"
        )
    finally:
        os.environ.pop("SAGE_FAITH_INFO_BOUNDED", None)


if __name__ == "__main__":
    # Standalone smoke (no pytest dependency)
    test_default_off_preserves_existing_symmetric_update()
    test_info_bounded_caps_degenerate_dc22_rule()
    test_info_bounded_preserves_informative_cn04_rule()
    test_info_bounded_separates_degenerate_from_informative()
    test_small_sample_full_weight()
    test_disconfirms_still_full_strength()
    print("[OK] all 6 SAGE_FAITH_INFO_BOUNDED tests passed")
