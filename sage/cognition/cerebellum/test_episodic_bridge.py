"""Tests for the episodic → cerebellum bridge."""

import time

from sage.cognition.cerebellum.core import Cerebellum
from sage.cognition.cerebellum.episodic_bridge import (
    compile_habits_from_episodes,
    compile_habits_from_trajectories,
    episode_to_cerebellum_dict,
    group_episodes_into_trajectories,
    trajectory_to_cerebellum_dict,
)
from sage.cognition.episodic.data import Episode
from sage.cognition.episodic.index import EpisodicIndex


def _ep(
    state_sig: dict,
    action: str = "UP",
    success: bool = True,
    reward: float = 0.5,
    stance: str = "EXPLORING",
    tags: list | None = None,
    session_id: str = "",
    cycle_id: int = 0,
) -> Episode:
    return Episode(
        state_signature=state_sig,
        action_taken=action,
        action_args={},
        outcome="moved",
        reward=reward,
        success=success,
        cognitive_stance=stance,
        tags=tags or [],
        session_id=session_id,
        cycle_id=cycle_id,
    )


def test_episode_to_dict_roundtrip_keys():
    ep = _ep({"level": 1, "color": "red"}, action="LEFT", stance="arc-game")
    d = episode_to_cerebellum_dict(ep)

    assert d["episode_id"] == ep.episode_id
    assert d["state"]["domain"] == "arc-game"
    assert d["state"]["features"] == {"level": 1, "color": "red"}
    assert d["actions"] == [{"action": "LEFT"}]
    assert d["outcome"]["success"] is True
    assert d["outcome"]["reward"] == 0.5


def test_domain_inference_fallback():
    # No stance, no tags → "episodic"
    ep = _ep({"x": 1}, stance="", tags=[])
    assert episode_to_cerebellum_dict(ep)["state"]["domain"] == "episodic"

    # Tags take over when stance is empty
    ep2 = _ep({"x": 1}, stance="", tags=["navigate"])
    assert episode_to_cerebellum_dict(ep2)["state"]["domain"] == "navigate"


def test_domain_override():
    ep = _ep({"x": 1}, stance="EXPLORING")
    assert episode_to_cerebellum_dict(ep, domain="forced")["state"]["domain"] == "forced"


def test_action_args_preserved():
    ep = Episode(
        state_signature={"n": 1},
        action_taken="CLICK",
        action_args={"x": 10, "y": 20},
        outcome="ok",
        success=True,
        cognitive_stance="play",
    )
    d = episode_to_cerebellum_dict(ep)
    assert d["actions"] == [{"action": "CLICK", "args": {"x": 10, "y": 20}}]


def test_no_action_produces_empty_sequence():
    ep = Episode(state_signature={"n": 1}, action_taken=None, success=True)
    d = episode_to_cerebellum_dict(ep)
    assert d["actions"] == []


def test_compile_requires_maturity_threshold():
    """Fewer than maturity_threshold successful episodes → no habit."""
    cb = Cerebellum(maturity_threshold=3)
    state = {"level": 1}
    episodes = [_ep(state, stance="arc") for _ in range(2)]

    compiled = compile_habits_from_episodes(episodes, cb)
    assert compiled == []
    assert cb.habit_count == 0


def test_compile_creates_habit_from_repeated_success():
    """3+ successful episodes with same state → mature habit."""
    cb = Cerebellum(maturity_threshold=3)
    state = {"level": 1, "color": "red"}
    episodes = [_ep(state, action="UP", stance="arc") for _ in range(4)]

    compiled = compile_habits_from_episodes(episodes, cb)
    assert len(compiled) == 1

    h = compiled[0]
    assert h.training_count == 4
    assert h.success_count == 4
    assert h.is_mature
    assert h.state_sig.domain == "arc"
    assert h.state_sig.features == state


def test_compile_links_source_episodes():
    """Compiled habit stores source episode IDs for provenance."""
    cb = Cerebellum(maturity_threshold=3)
    state = {"level": 2}
    episodes = [_ep(state, stance="arc") for _ in range(3)]
    expected_ids = {e.episode_id for e in episodes}

    compiled = compile_habits_from_episodes(episodes, cb)
    assert len(compiled) == 1
    assert set(compiled[0].source_episodes) == expected_ids


def test_compile_rejects_low_success_rate():
    """Group with <80% success → no habit even at threshold."""
    cb = Cerebellum(maturity_threshold=3)
    state = {"level": 1}
    episodes = [
        _ep(state, success=True, stance="arc"),
        _ep(state, success=False, stance="arc"),
        _ep(state, success=False, stance="arc"),
    ]
    assert compile_habits_from_episodes(episodes, cb) == []


def test_episode_filter():
    """episode_filter prunes before compilation."""
    cb = Cerebellum(maturity_threshold=3)
    state = {"level": 1}
    eps = [_ep(state, stance="arc") for _ in range(3)]
    # One episode has a distinguishing tag we want to exclude
    eps[0].tags = ["skip-me"]

    compiled = compile_habits_from_episodes(
        eps, cb, episode_filter=lambda e: "skip-me" not in e.tags
    )
    # Only 2 episodes passed filter → below maturity_threshold=3
    assert compiled == []


def test_end_to_end_with_episodic_index():
    """Integration: bind episodes to an EpisodicIndex, then compile habits."""
    index = EpisodicIndex()  # in-memory
    cb = Cerebellum(maturity_threshold=3)

    # Three identical-state successful episodes
    for _ in range(3):
        index.bind(_ep({"room": "kitchen"}, action="OPEN_FRIDGE", stance="daily-life"))
    # Plus one unrelated episode
    index.bind(_ep({"room": "garden"}, action="WATER_PLANT", stance="daily-life"))

    all_eps = list(index._episodes.values())
    compiled = compile_habits_from_episodes(all_eps, cb)

    assert len(compiled) == 1
    assert compiled[0].state_sig.features == {"room": "kitchen"}
    assert compiled[0].training_count == 3


# ─── Multi-step trajectory bridge ────────────────────────────────────


def test_grouping_splits_by_session():
    """Episodes from different sessions never end up in the same trajectory."""
    eps = [
        _ep({"x": 1}, session_id="A", cycle_id=0),
        _ep({"x": 2}, session_id="A", cycle_id=1),
        _ep({"x": 3}, session_id="B", cycle_id=0),
    ]
    trajs = group_episodes_into_trajectories(eps)
    assert len(trajs) == 2
    sessions = {t[0].session_id for t in trajs}
    assert sessions == {"A", "B"}
    a = next(t for t in trajs if t[0].session_id == "A")
    assert [e.cycle_id for e in a] == [0, 1]


def test_grouping_splits_on_cycle_gap():
    """Cycle-id jumps larger than max_gap split a session into trajectories."""
    eps = [
        _ep({"x": 1}, session_id="S", cycle_id=0),
        _ep({"x": 2}, session_id="S", cycle_id=1),
        _ep({"x": 3}, session_id="S", cycle_id=5),  # gap 4 > max_gap=1
        _ep({"x": 4}, session_id="S", cycle_id=6),
    ]
    trajs = group_episodes_into_trajectories(eps, max_gap=1)
    assert len(trajs) == 2
    assert [e.cycle_id for e in trajs[0]] == [0, 1]
    assert [e.cycle_id for e in trajs[1]] == [5, 6]

    # Same input with larger max_gap collapses to one trajectory
    trajs2 = group_episodes_into_trajectories(eps, max_gap=5)
    assert len(trajs2) == 1
    assert [e.cycle_id for e in trajs2[0]] == [0, 1, 5, 6]


def test_grouping_orders_by_cycle_id():
    """Within a session, episodes get sorted by cycle_id regardless of input order."""
    eps = [
        _ep({"x": 3}, session_id="S", cycle_id=2),
        _ep({"x": 1}, session_id="S", cycle_id=0),
        _ep({"x": 2}, session_id="S", cycle_id=1),
    ]
    trajs = group_episodes_into_trajectories(eps)
    assert len(trajs) == 1
    assert [e.cycle_id for e in trajs[0]] == [0, 1, 2]


def test_grouping_treats_empty_session_as_singleton():
    """Episodes without a session_id can't claim contiguity — each stands alone."""
    eps = [
        _ep({"x": 1}, session_id="", cycle_id=0),
        _ep({"x": 2}, session_id="", cycle_id=1),
    ]
    trajs = group_episodes_into_trajectories(eps)
    assert len(trajs) == 2
    assert all(len(t) == 1 for t in trajs)


def test_trajectory_to_dict_initial_state_final_outcome():
    """Trajectory dict carries initial state, ordered actions, final outcome."""
    traj = [
        _ep({"room": "hall"}, action="WALK", session_id="S", cycle_id=0,
            stance="navigate"),
        _ep({"room": "kitchen"}, action="OPEN_FRIDGE", session_id="S", cycle_id=1,
            stance="navigate"),
        _ep({"room": "kitchen"}, action="GRAB_MILK", session_id="S", cycle_id=2,
            stance="navigate", reward=1.0),
    ]
    d = trajectory_to_cerebellum_dict(traj)
    # State = initial episode's state
    assert d["state"]["features"] == {"room": "hall"}
    assert d["state"]["domain"] == "navigate"
    # Actions = ordered sequence
    assert d["actions"] == [
        {"action": "WALK"},
        {"action": "OPEN_FRIDGE"},
        {"action": "GRAB_MILK"},
    ]
    # Outcome = final episode's
    assert d["outcome"]["reward"] == 1.0
    assert d["outcome"]["success"] is True
    # Trajectory length recorded for identifiability
    assert "trajectory[3]" in d["outcome"]["summary"]
    # episode_id = initial (stable identifier for the trajectory)
    assert d["episode_id"] == traj[0].episode_id


def test_trajectory_to_dict_skips_empty_actions():
    """Episodes with no action_taken contribute no step to the sequence."""
    traj = [
        _ep({"x": 1}, action="A", session_id="S", cycle_id=0),
        Episode(state_signature={"x": 1}, action_taken=None, success=True,
                session_id="S", cycle_id=1, cognitive_stance="x"),
        _ep({"x": 1}, action="B", session_id="S", cycle_id=2),
    ]
    d = trajectory_to_cerebellum_dict(traj)
    assert d["actions"] == [{"action": "A"}, {"action": "B"}]


def test_trajectory_to_dict_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        trajectory_to_cerebellum_dict([])


def test_compile_trajectories_one_observation_per_trajectory():
    """A 3-cycle trajectory counts as ONE observation, not three.

    Maturity semantics must match single-episode path: maturity_threshold=3
    means three trajectory observations, not three episodes within one.
    """
    cb = Cerebellum(maturity_threshold=3)
    # One long trajectory of 5 cycles. Should NOT yield a habit.
    traj_eps = [
        _ep({"start": True}, action="STEP", session_id="single", cycle_id=i,
            stance="walk")
        for i in range(5)
    ]
    compiled = compile_habits_from_trajectories(traj_eps, cb)
    assert compiled == [], "Single trajectory must not satisfy maturity_threshold"


def test_compile_trajectories_three_matching_yields_multistep_habit():
    """Three trajectories starting at the same state with the same arc → habit."""
    cb = Cerebellum(maturity_threshold=3)
    eps: list[Episode] = []
    for sid in ("s1", "s2", "s3"):
        eps.append(_ep({"start": True}, action="A", session_id=sid, cycle_id=0,
                       stance="walk"))
        eps.append(_ep({"mid": True}, action="B", session_id=sid, cycle_id=1,
                       stance="walk"))
        eps.append(_ep({"end": True}, action="C", session_id=sid, cycle_id=2,
                       stance="walk", reward=1.0))

    compiled = compile_habits_from_trajectories(eps, cb)
    assert len(compiled) == 1
    h = compiled[0]
    # Habit keyed on initial state
    assert h.state_sig.features == {"start": True}
    # Multi-step action sequence preserved in order
    assert h.action_sequence == [
        {"action": "A"},
        {"action": "B"},
        {"action": "C"},
    ]
    # 3 trajectory observations
    assert h.training_count == 3
    assert h.success_count == 3
    assert h.is_mature
    # Provenance: one initial episode_id per contributing trajectory
    initial_ids = {eps[0].episode_id, eps[3].episode_id, eps[6].episode_id}
    assert set(h.source_episodes) == initial_ids


def test_compile_trajectories_mixed_arcs_dont_consensus_unless_dominant():
    """When trajectories from same start state diverge, the dominant arc wins.

    This mirrors the existing single-episode compile semantics: most-common
    action sequence is selected once maturity is reached.
    """
    cb = Cerebellum(maturity_threshold=3)
    state = {"start": True}
    eps = []
    # Three trajectories with arc [A,B], one with arc [A,Z]
    for i, last in enumerate(["B", "B", "B", "Z"]):
        sid = f"s{i}"
        eps.append(_ep(state, action="A", session_id=sid, cycle_id=0, stance="x"))
        eps.append(_ep({"mid": True}, action=last, session_id=sid, cycle_id=1,
                       stance="x"))
    compiled = compile_habits_from_trajectories(eps, cb)
    assert len(compiled) == 1
    assert compiled[0].action_sequence == [{"action": "A"}, {"action": "B"}]


def test_compile_trajectories_respects_max_gap():
    """max_gap controls whether multi-cycle arcs collapse into multi-step habits.

    With a cycle gap larger than max_gap, the compile path sees each cycle
    as its own singleton trajectory and produces a single-step habit on
    the initial state. With max_gap large enough to bridge the gap, the
    full multi-step arc gets compiled.
    """
    state = {"start": True}
    eps = []
    # Three sessions, each with cycle 0 and cycle 3 (gap of 3 cycles).
    for i in range(3):
        sid = f"s{i}"
        eps.append(_ep(state, action="A", session_id=sid, cycle_id=0, stance="x"))
        eps.append(_ep({"x": 9}, action="B", session_id=sid, cycle_id=3, stance="x"))

    # max_gap=1: cycle 0 and cycle 3 are split. The cycle-0 singletons
    # cluster on {start:True} → one single-step habit; the cycle-3
    # singletons cluster on {x:9} → another single-step habit.
    cb1 = Cerebellum(maturity_threshold=3)
    h1 = compile_habits_from_trajectories(eps, cb1, max_gap=1)
    seqs1 = sorted(tuple(s["action"] for s in h.action_sequence) for h in h1)
    assert seqs1 == [("A",), ("B",)]

    # max_gap=3: trajectories collapse into 2-step arcs → one multi-step habit
    # keyed on the initial state.
    cb2 = Cerebellum(maturity_threshold=3)
    h2 = compile_habits_from_trajectories(eps, cb2, max_gap=3)
    assert len(h2) == 1
    assert h2[0].action_sequence == [{"action": "A"}, {"action": "B"}]
    assert h2[0].state_sig.features == {"start": True}


def test_compile_trajectories_end_to_end_with_episodic_index():
    """Integration: bind multi-cycle episodes via index, compile multi-step habits."""
    index = EpisodicIndex()
    cb = Cerebellum(maturity_threshold=3)

    # Three identical 2-step daily-routine trajectories
    for sid in ("morning1", "morning2", "morning3"):
        index.bind(_ep({"location": "bedroom"}, action="WAKE",
                       session_id=sid, cycle_id=0, stance="daily"))
        index.bind(_ep({"location": "kitchen"}, action="MAKE_COFFEE",
                       session_id=sid, cycle_id=1, stance="daily", reward=1.0))

    # Plus an unrelated singleton
    index.bind(_ep({"location": "garden"}, action="WATER",
                   session_id="ad-hoc", cycle_id=0, stance="daily"))

    all_eps = list(index._episodes.values())
    compiled = compile_habits_from_trajectories(all_eps, cb)

    assert len(compiled) == 1
    h = compiled[0]
    assert h.state_sig.features == {"location": "bedroom"}
    assert h.action_sequence == [
        {"action": "WAKE"},
        {"action": "MAKE_COFFEE"},
    ]
    assert h.training_count == 3


def test_trajectory_consensus_threshold_blocks_divergent_arcs():
    """4 trajectories from same start, 4 different arcs → gate skips compile.

    Without the gate, a Counter-picked winner at 1/4 (25%) consensus would
    become a habit, misrepresenting what the agent actually does from
    that state. With threshold 0.5, no habit compiles — the cerebellum
    waits for evidence of a preferred sequence.
    """
    cb = Cerebellum(maturity_threshold=3, consensus_threshold=0.5)
    state = {"start": True}
    eps = []
    for i, last in enumerate(["A", "B", "C", "D"]):
        sid = f"s{i}"
        eps.append(_ep(state, action="X", session_id=sid, cycle_id=0, stance="t"))
        eps.append(_ep({"mid": True}, action=last, session_id=sid, cycle_id=1,
                       stance="t"))
    assert compile_habits_from_trajectories(eps, cb) == []


def test_trajectory_consensus_threshold_admits_dominant_arc():
    """3/4 agreement (ratio 0.75) clears threshold 0.5 → multi-step habit."""
    cb = Cerebellum(maturity_threshold=3, consensus_threshold=0.5)
    state = {"start": True}
    eps = []
    for i, last in enumerate(["B", "B", "B", "Z"]):
        sid = f"s{i}"
        eps.append(_ep(state, action="A", session_id=sid, cycle_id=0, stance="t"))
        eps.append(_ep({"mid": True}, action=last, session_id=sid, cycle_id=1,
                       stance="t"))
    habits = compile_habits_from_trajectories(eps, cb)
    assert len(habits) == 1
    assert habits[0].action_sequence == [{"action": "A"}, {"action": "B"}]
    assert "consensus 3/4" in habits[0].outcome_summary
