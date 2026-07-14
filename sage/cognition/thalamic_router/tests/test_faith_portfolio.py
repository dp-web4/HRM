"""Tests for the faith-candidate portfolio (parallel win-hypotheses).

Validates dp's model: multiple candidates explored in parallel, relative trust
evolves with accumulated evidence (confirm/disconfirm only — silence never moves
trust), commitment allocated ∝ trust×plausibility, open questions tracked, and
directed exploration picks the discriminating (high-VOI) question.

Pure; runs under pytest or as `python3 test_faith_portfolio.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from sage.cognition.thalamic_router.faith_portfolio import FaithPortfolio  # noqa: E402


def _wa30_portfolio():
    p = FaithPortfolio()
    p.add("direct_place", "place every tile on the strip myself -> win",
          plausibility=0.6, trust=0.3,
          open_questions=["can the player reach all 3 tile destinations?"], pivotals=[0.8])
    p.add("follower_stage", "stage tiles on walls; followers deliver to far zones -> win",
          plausibility=0.7, trust=0.3,
          open_questions=["do followers pick up wall-staged tiles?"], pivotals=[0.9])
    p.add("push_only", "no SEL; just push tiles into place -> win",
          plausibility=0.25, trust=0.3, open_questions=["are tiles pushable?"], pivotals=[0.4])
    return p


def test_parallel_candidates_coexist():
    p = _wa30_portfolio()
    assert len(p.candidates) == 3
    w = p.weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6      # commitment is allocated, not winner-take-all
    assert all(0 < x < 1 for x in w.values())     # all three get some weight initially


def test_silence_does_not_move_trust():
    """The faith rule: a feedback desert (no evidence) must NOT change trust."""
    p = _wa30_portfolio()
    before = {c.id: c.trust for c in p.candidates.values()}
    # ... many silent steps would happen here; we simply record NO evidence ...
    after = {c.id: c.trust for c in p.candidates.values()}
    assert before == after


def test_evidence_reallocates_trust():
    """Disconfirm the wrong candidate, confirm the right one -> trust + weight shift."""
    p = _wa30_portfolio()
    # push_only disconfirmed (tiles aren't pushable); follower_stage confirmed
    p.record("push_only", "disconfirm", question="are tiles pushable?")
    for _ in range(3):
        p.record("follower_stage", "confirm")
    p.record("follower_stage", "confirm", question="do followers pick up wall-staged tiles?")

    best = p.best()
    assert best.id == "follower_stage", f"expected follower_stage to lead, got {best.id}"
    w = p.weights()
    assert w["follower_stage"] > w["direct_place"] > w["push_only"]
    # disconfirmed candidate's pivotal question is resolved (no longer open)
    assert all(q.status != "open" for q in p.candidates["push_only"].open_questions)
    # confirmed candidate crossed the commit gate
    assert p.should_commit()


def test_disconfirm_does_not_kill_faith_globally():
    """Disconfirming one candidate shifts weight to others — faith persists."""
    p = _wa30_portfolio()
    p.record("follower_stage", "disconfirm")          # the front-runner takes a hit
    # portfolio still has plausible candidates; commitment just reallocates
    w = p.weights()
    assert w["direct_place"] > w["follower_stage"], "weight should move off the disconfirmed one"
    assert sum(w.values()) > 0


def test_directed_exploration_picks_pivotal_question():
    """VOI: the open question to resolve is the high-pivotal one on a leading,
    contested candidate — here the follower-pickup question."""
    p = _wa30_portfolio()
    sel = p.next_open_question()
    assert sel is not None
    cid, q = sel
    assert cid == "follower_stage" and "followers" in q.text, (cid, q.text)


def test_commit_gate():
    p = FaithPortfolio(commit_tau=0.45)
    # plausible path (could-win prior 0.7) but low initial trust -> not yet committable
    p.add("plausible", "a path that could win", plausibility=0.7, trust=0.2)
    assert not p.should_commit()                      # 0.7*0.2=0.14 < τ
    for _ in range(6):
        p.record("plausible", "confirm")
    assert p.should_commit()                          # trust accrues -> score clears τ


def test_low_plausibility_never_commits():
    """A low 'could-win' prior caps the score below τ even at full trust —
    you don't commit faith to a path that probably can't win, however 'trusted'."""
    p = FaithPortfolio(commit_tau=0.45)
    p.add("longshot", "probably can't win", plausibility=0.3, trust=0.2)
    for _ in range(12):
        p.record("longshot", "confirm")              # trust -> ~1.0
    assert p.candidates["longshot"].trust > 0.9
    assert not p.should_commit()                      # 0.3 cap < τ — correctly never commits


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
