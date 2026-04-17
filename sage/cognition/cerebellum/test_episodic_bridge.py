"""Tests for the episodic → cerebellum bridge."""

import time

from sage.cognition.cerebellum.core import Cerebellum
from sage.cognition.cerebellum.episodic_bridge import (
    compile_habits_from_episodes,
    episode_to_cerebellum_dict,
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
