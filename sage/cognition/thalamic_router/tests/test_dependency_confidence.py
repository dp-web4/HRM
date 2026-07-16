"""Tests for GameWorldModel.dependency_confidence_now() — the live, model-general
strict-trust signal (Sprout, 2026-05-24, per CBP ack).

These prove the *formula behavior* (the looseness fix + 1-of-N disambiguation
gating). They do NOT assert it reproduces the calibrated {toy_a, toy_b, toy_c} set —
that requires a live-strict A/B (CBP wires + runs). See the accessor docstring.

Placeholder game ids ({toy_a, toy_b, toy_c}) stand in for the real calibration set.
"""
from sage.cognition.thalamic_router.wm_schema import GameWorldModel, CausalRule

CONF_CUT, DISAMBIG_CUT = 0.45, 0.5  # the gate's cutoffs (faith_portfolio.py)


def _wm(rules):
    wm = GameWorldModel(game="test")
    wm.causal_rules = rules
    return wm


def _rule(action, conf, evid):
    return CausalRule(action=action, condition="", predicted_effect="",
                      confidence=conf, evidence_count=evid)


def test_none_when_no_verified_rules():
    assert _wm([]).dependency_confidence_now() is None
    assert _wm([_rule("UP", 0.9, 0)]).dependency_confidence_now() is None  # no evidence


def test_high_prior_seen_once_is_untrusted():
    """The looseness fix: a 0.9-prior rule seen ONCE earns 0.9*1/3=0.30, not 0.90.
    Broad avg-confidence would call this trusted (0.9>=0.45); graded must not."""
    out = _wm([_rule("UP", 0.9, 1)]).dependency_confidence_now()
    assert out["dep_confidence"] < CONF_CUT, out  # untrusted, unlike broad


def test_well_held_multi_action_is_trusted():
    """Verified rules across 2 of 2 action classes → trusted on both axes."""
    out = _wm([_rule("UP", 0.8, 3), _rule("DOWN", 0.7, 4)]).dependency_confidence_now()
    assert out["dep_confidence"] >= CONF_CUT
    assert out["dep_disambiguation"] >= DISAMBIG_CUT


def test_one_of_many_actions_is_low_disambiguation():
    """Earned coverage on 1 of 4 action classes → disambiguation 0.25 < 0.5
    (the 1-of-N predictor reads as noise even if that one rule is strong)."""
    rules = [_rule("UP", 0.9, 5),          # earned ~0.9
             _rule("DOWN", 0.2, 1),         # earned ~0.07, below floor
             _rule("LEFT", 0.2, 1),
             _rule("RIGHT", 0.2, 1)]
    out = _wm(rules).dependency_confidence_now()
    assert out["dep_disambiguation"] < DISAMBIG_CUT, out


def test_evidence_monotonicity():
    """More confirmations → higher dep_confidence, holding prior fixed."""
    lo = _wm([_rule("UP", 0.6, 1)]).dependency_confidence_now()["dep_confidence"]
    hi = _wm([_rule("UP", 0.6, 3)]).dependency_confidence_now()["dep_confidence"]
    assert hi > lo


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
    sys.exit(0)
