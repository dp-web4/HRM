#!/usr/bin/env python3
"""Tests for the episodic memory index."""

import os
import sys
import tempfile
import time

import numpy as np

# Allow standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from sage.cognition.episodic.data import Episode, EpisodicCue, EMBEDDING_DIM
from sage.cognition.episodic.index import EpisodicIndex


def test_basic_bind_recall():
    """T1: basic bind + recall."""
    index = EpisodicIndex()

    ep = Episode(
        state_signature={"game": "toy_a", "level": 1},
        snarc_scores={"surprise": 0.8, "novelty": 0.5, "arousal": 0.3, "reward": 0.6, "conflict": 0.1},
        action_taken="LEFT",
        tags=["game", "toy_a"],
    )
    eid = index.bind(ep)

    results = index.recall(EpisodicCue(state_signature={"game": "toy_a", "level": 1}))
    assert len(results) >= 1, "Should recall at least one episode"
    assert results[0].episode.episode_id == eid
    assert results[0].similarity > 0.5, f"Self-match should be high, got {results[0].similarity}"
    print("  PASS: basic bind + recall")


def test_partial_cue_snarc():
    """T2: partial cue matching by SNARC profile."""
    index = EpisodicIndex()

    ep_high = Episode(
        snarc_scores={"surprise": 0.9, "novelty": 0.8, "arousal": 0.7, "reward": 0.6, "conflict": 0.1},
        tags=["high_surprise"],
    )
    ep_low = Episode(
        snarc_scores={"surprise": 0.1, "novelty": 0.2, "arousal": 0.1, "reward": 0.3, "conflict": 0.0},
        tags=["low_surprise"],
    )
    index.bind(ep_high)
    index.bind(ep_low)

    results = index.recall(EpisodicCue(snarc_scores={"surprise": 0.85, "novelty": 0.75}))
    assert len(results) >= 1
    assert results[0].episode.episode_id == ep_high.episode_id, "High-surprise cue should match high-surprise episode"
    print("  PASS: partial cue SNARC matching")


def test_partial_cue_action():
    """T2b: partial cue matching by action."""
    index = EpisodicIndex()

    ep_left = Episode(action_taken="LEFT", tags=["movement"])
    ep_right = Episode(action_taken="RIGHT", tags=["movement"])
    ep_click = Episode(action_taken="CLICK", tags=["interaction"])
    index.bind(ep_left)
    index.bind(ep_right)
    index.bind(ep_click)

    results = index.recall(EpisodicCue(action_taken="LEFT"))
    assert results[0].episode.action_taken == "LEFT", "Should match LEFT action"
    print("  PASS: partial cue action matching")


def test_tag_filtering():
    """T2c: tag-based recall."""
    index = EpisodicIndex()

    ep1 = Episode(tags=["game", "toy_a", "level1"])
    ep2 = Episode(tags=["game", "toy_c", "level1"])
    ep3 = Episode(tags=["raising", "identity"])
    index.bind(ep1)
    index.bind(ep2)
    index.bind(ep3)

    results = index.recall(EpisodicCue(tags=["toy_a"]))
    assert any(r.episode.episode_id == ep1.episode_id for r in results), "Should find toy_a episode"
    print("  PASS: tag filtering")


def test_consolidation():
    """T3: consolidation reduces episode count."""
    index = EpisodicIndex()

    # Store 10 very similar episodes
    for i in range(10):
        ep = Episode(
            state_signature={"game": "toy_a", "level": 1},
            snarc_scores={"surprise": 0.5 + i * 0.01, "novelty": 0.3, "arousal": 0.2, "reward": 0.4, "conflict": 0.1},
            action_taken="LEFT",
            success=True,
            tags=["game", "toy_a"],
        )
        index.bind(ep)

    assert index.stats()["count"] == 10

    report = index.consolidate(similarity_threshold=0.8, min_cluster_size=3)
    assert index.stats()["count"] < 10, f"Consolidation should reduce count, got {index.stats()['count']}"
    assert report.prototypes_created > 0, "Should create at least one prototype"
    print(f"  PASS: consolidation {report.episodes_before} → {report.episodes_after} ({report.prototypes_created} prototypes)")


def test_forget():
    """T3b: forget removes old low-utility episodes."""
    index = EpisodicIndex()

    # Create old episode
    old_ep = Episode(tags=["old"])
    old_ep.timestamp = time.time() - 40 * 86400  # 40 days ago
    index.bind(old_ep)

    # Create recent episode
    new_ep = Episode(tags=["new"])
    index.bind(new_ep)

    assert index.stats()["count"] == 2
    forgotten = index.forget(max_age_days=30, min_accesses=0)
    assert forgotten == 1, f"Should forget 1 old episode, forgot {forgotten}"
    assert index.stats()["count"] == 1
    assert list(index._episodes.values())[0].tags == ["new"]
    print("  PASS: forget old episodes")


def test_recency_weighting():
    """T4: recent episodes rank higher than old ones."""
    index = EpisodicIndex(recency_half_life=3600)  # 1 hour half-life

    old_ep = Episode(
        state_signature={"test": "recency"},
        tags=["test"],
    )
    old_ep.timestamp = time.time() - 7200  # 2 hours ago
    index.bind(old_ep)

    new_ep = Episode(
        state_signature={"test": "recency"},
        tags=["test"],
    )
    index.bind(new_ep)

    results = index.recall(EpisodicCue(state_signature={"test": "recency"}))
    assert len(results) == 2
    # New episode should score higher due to recency
    assert results[0].episode.episode_id == new_ep.episode_id, "Recent episode should rank first"
    assert results[0].recency_weight > results[1].recency_weight, "Recent episode should have higher recency"
    print("  PASS: recency weighting")


def test_embedding_dimensions():
    """T5: embedding has correct dimensions."""
    ep = Episode(
        state_signature={"game": "test"},
        snarc_scores={"surprise": 0.5, "novelty": 0.3},
        action_taken="UP",
        outcome="moved",
        tags=["test", "movement"],
    )
    emb = ep.compute_embedding()
    assert emb.shape == (EMBEDDING_DIM,), f"Expected ({EMBEDDING_DIM},), got {emb.shape}"
    assert emb.dtype == np.float32
    print(f"  PASS: embedding shape ({EMBEDDING_DIM},)")


def test_sqlite_persistence():
    """T6: episodes persist across index instances."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        # Create and populate
        index1 = EpisodicIndex(db_path=db_path)
        ep = Episode(
            state_signature={"persist": "test"},
            action_taken="SELECT",
            tags=["persistence"],
        )
        eid = index1.bind(ep)
        assert index1.stats()["count"] == 1

        # Recreate from disk
        index2 = EpisodicIndex(db_path=db_path)
        assert index2.stats()["count"] == 1, "Should load from SQLite"

        results = index2.recall(EpisodicCue(tags=["persistence"]))
        assert len(results) >= 1
        assert results[0].episode.episode_id == eid
        print("  PASS: SQLite persistence")
    finally:
        os.unlink(db_path)


def test_empty_recall():
    """T7: recall on empty index returns empty list."""
    index = EpisodicIndex()
    results = index.recall(EpisodicCue(state_signature={"anything": True}))
    assert results == []
    print("  PASS: empty recall")


def test_access_count_updates():
    """T8: recall updates access count."""
    index = EpisodicIndex()
    ep = Episode(state_signature={"access": "test"}, tags=["test"])
    index.bind(ep)
    assert ep.access_count == 0

    index.recall(EpisodicCue(state_signature={"access": "test"}))
    assert ep.access_count == 1

    index.recall(EpisodicCue(state_signature={"access": "test"}))
    assert ep.access_count == 2
    print("  PASS: access count updates")


def test_many_episodes_performance():
    """T9: 10k episodes — recall should be fast."""
    index = EpisodicIndex()
    t0 = time.time()
    for i in range(10000):
        ep = Episode(
            state_signature={"i": i % 100},
            snarc_scores={"surprise": (i % 10) / 10.0},
            action_taken=["UP", "DOWN", "LEFT", "RIGHT"][i % 4],
            tags=[f"batch_{i // 1000}"],
        )
        index.bind(ep)
    bind_time = time.time() - t0

    t0 = time.time()
    results = index.recall(EpisodicCue(state_signature={"i": 42}), k=10)
    recall_time = time.time() - t0

    assert len(results) == 10
    assert recall_time < 1.0, f"Recall took {recall_time:.3f}s — should be <1s for 10k episodes"
    print(f"  PASS: 10k episodes — bind {bind_time:.2f}s, recall {recall_time:.3f}s, "
          f"memory {index.stats()['memory_mb']:.1f}MB")


def test_from_wm_dump():
    """T10: Episode.from_wm_dump creates valid episode from WM snapshot."""
    # Simulate a WM dump (matches CBP's WorkingMemory.dump() format)
    wm_dump = {
        "capacity": 7,
        "slots": [
            {"slot_id": "wm_001", "content_type": "goal", "content": "solve toy_a level 1",
             "priority": 0.9, "timestamp": time.time(), "goal_id": "g1", "access_count": 3},
            {"slot_id": "wm_002", "content_type": "plan_step", "content": "move basket to S position",
             "priority": 0.7, "timestamp": time.time(), "goal_id": "g1", "access_count": 1},
            {"slot_id": "wm_003", "content_type": "hypothesis", "content": "SELECT launches paint",
             "priority": 0.6, "timestamp": time.time(), "goal_id": "g1", "access_count": 2},
        ],
        "active_plan": None,
        "current_step_index": 0,
        "bindings": [
            {"sensor_id": "vision", "observation": "white rectangle with red border",
             "goal_id": "g1", "relevance": 0.8, "timestamp": time.time()},
        ],
        "stats": {"total_writes": 5},
        "snapshot_at": time.time(),
    }

    ep = Episode.from_wm_dump(
        wm_dump=wm_dump,
        session_id="test_session",
        action_taken="SELECT",
        outcome="painted 5 cells",
        reward=0.5,
        success=True,
        snarc_scores={"surprise": 0.7, "novelty": 0.4, "arousal": 0.3, "reward": 0.6, "conflict": 0.1},
        tags=["toy_a", "game"],
    )

    assert ep.embedding is not None
    assert ep.embedding.shape == (EMBEDDING_DIM,)
    assert ep.state_signature.get("goal") == "solve toy_a level 1"
    assert ep.state_signature.get("hypothesis") == "SELECT launches paint"
    assert ep.sensory_summary.get("vision") == "white rectangle with red border"
    assert ep.action_taken == "SELECT"
    assert ep.success is True
    assert "goal" in ep.tags
    assert "toy_a" in ep.tags

    # Verify it's recallable. Recall via the deterministic tag channel — a
    # partial-dict state_signature cue hashes to a vector unrelated to the
    # episode's full stored signature, so it only adds masked-cosine noise
    # (embedding aliasing) rather than a real match signal.
    index = EpisodicIndex()
    index.bind(ep)
    results = index.recall(EpisodicCue(tags=["toy_a"]))
    assert len(results) >= 1
    assert results[0].episode.action_taken == "SELECT"
    print("  PASS: from_wm_dump integration")


def test_wm_consolidate_integration():
    """T11: WM.consolidate_to_ltm sends to episodic index."""
    try:
        from sage.cognition.working_memory import WorkingMemory
    except ImportError:
        print("  SKIP: working_memory not available")
        return

    index = EpisodicIndex()
    wm = WorkingMemory(capacity=7)

    # Add some slots
    wm.add_item("goal", "solve the puzzle", priority=0.9, goal_id="g1")
    wm.add_item("hypothesis", "arrows rotate", priority=0.7, goal_id="g1")

    # Consolidate to episodic
    count = wm.consolidate_to_ltm(
        goal_id="g1",
        episodic_index=index,
        session_id="test",
        action_taken="LEFT",
        outcome="rotated",
        reward=0.3,
        success=True,
        snarc_scores={"surprise": 0.5},
        tags=["integration_test"],
    )

    assert count >= 1, "Should have high-priority slots"
    assert index.stats()["count"] == 1, "Should have bound one episode"

    results = index.recall(EpisodicCue(tags=["integration_test"]))
    assert len(results) >= 1
    assert results[0].episode.state_signature.get("goal") == "solve the puzzle"
    print("  PASS: WM consolidate → episodic integration")


def _traj_ep(session_id: str, cycle_id: int, action: str = "STEP") -> Episode:
    """Minimal Episode fixture for trajectory-walk tests."""
    return Episode(
        state_signature={"cycle": cycle_id},
        action_taken=action,
        success=True,
        session_id=session_id,
        cycle_id=cycle_id,
    )


def test_walk_trajectory_contiguous_session():
    """Walk forward through a contiguous session, starting at the initial cycle."""
    index = EpisodicIndex()
    ids = []
    for cid in range(3):
        ep = _traj_ep("s1", cid)
        ids.append(index.bind(ep))

    traj = index.walk_trajectory(ids[0])
    assert [e.cycle_id for e in traj] == [0, 1, 2]
    assert traj[0].episode_id == ids[0]
    print("  PASS: walk_trajectory contiguous session")


def test_walk_trajectory_stops_at_gap():
    """Gap wider than max_gap terminates the walk; widening max_gap bridges it."""
    index = EpisodicIndex()
    initial_id = index.bind(_traj_ep("s1", 0))
    index.bind(_traj_ep("s1", 1))
    # cycle 4 is a gap of 3 from cycle 1
    index.bind(_traj_ep("s1", 4))
    index.bind(_traj_ep("s1", 5))

    # Default max_gap=1 stops at the gap
    traj = index.walk_trajectory(initial_id)
    assert [e.cycle_id for e in traj] == [0, 1]

    # max_gap=3 bridges the gap and keeps walking
    traj_wide = index.walk_trajectory(initial_id, max_gap=3)
    assert [e.cycle_id for e in traj_wide] == [0, 1, 4, 5]
    print("  PASS: walk_trajectory stops at gap (and widens with max_gap)")


def test_walk_trajectory_excludes_other_sessions():
    """Episodes from other sessions are never pulled into the walk."""
    index = EpisodicIndex()
    initial_id = index.bind(_traj_ep("s1", 0))
    index.bind(_traj_ep("s1", 1))
    # Decoy episodes in another session with overlapping cycle_ids
    index.bind(_traj_ep("s2", 0))
    index.bind(_traj_ep("s2", 1))
    index.bind(_traj_ep("s2", 2))

    traj = index.walk_trajectory(initial_id)
    assert {e.session_id for e in traj} == {"s1"}
    assert [e.cycle_id for e in traj] == [0, 1]
    print("  PASS: walk_trajectory excludes other sessions")


def test_walk_trajectory_empty_session_is_singleton():
    """No session_id → no contiguity claim; returned as a singleton."""
    index = EpisodicIndex()
    solo_id = index.bind(_traj_ep("", 0))
    # A companion with no session_id and adjacent cycle_id must NOT be pulled in
    index.bind(_traj_ep("", 1))

    traj = index.walk_trajectory(solo_id)
    assert len(traj) == 1
    assert traj[0].episode_id == solo_id
    print("  PASS: walk_trajectory empty session_id is singleton")


def test_walk_trajectory_unknown_id_raises():
    """Unknown initial_id raises KeyError; invalid max_gap raises ValueError."""
    import pytest
    index = EpisodicIndex()
    index.bind(_traj_ep("s1", 0))

    with pytest.raises(KeyError):
        index.walk_trajectory("not-a-real-id")

    real_id = index.bind(_traj_ep("s1", 1))
    with pytest.raises(ValueError):
        index.walk_trajectory(real_id, max_gap=0)
    print("  PASS: walk_trajectory raises on unknown id / bad max_gap")


def test_walk_trajectory_backward_from_mid_session():
    """Backward walk from a mid-trajectory anchor recovers earlier context."""
    index = EpisodicIndex()
    ids = []
    for cid in range(4):
        ids.append(index.bind(_traj_ep("s1", cid)))

    # Anchor at cycle 2; walk backward → [0, 1, 2] in ascending order.
    traj = index.walk_trajectory(ids[2], direction="backward")
    assert [e.cycle_id for e in traj] == [0, 1, 2]
    assert traj[-1].episode_id == ids[2]

    # Anchor at cycle 0 (leftmost): singleton.
    traj0 = index.walk_trajectory(ids[0], direction="backward")
    assert [e.cycle_id for e in traj0] == [0]
    print("  PASS: walk_trajectory backward from mid-session")


def test_walk_trajectory_backward_stops_at_gap():
    """Backward walk terminates at a gap wider than max_gap."""
    index = EpisodicIndex()
    index.bind(_traj_ep("s1", 0))
    index.bind(_traj_ep("s1", 1))
    # cycle 4 is 3 cycles from cycle 1 — gap
    index.bind(_traj_ep("s1", 4))
    anchor_id = index.bind(_traj_ep("s1", 5))

    # Default max_gap=1: backward from 5 stops at the 4→1 gap.
    traj = index.walk_trajectory(anchor_id, direction="backward")
    assert [e.cycle_id for e in traj] == [4, 5]

    # max_gap=3 bridges the gap.
    traj_wide = index.walk_trajectory(anchor_id, direction="backward", max_gap=3)
    assert [e.cycle_id for e in traj_wide] == [0, 1, 4, 5]
    print("  PASS: walk_trajectory backward stops at gap (and widens)")


def test_walk_trajectory_both_reconstructs_full_session():
    """direction='both' from a mid-trajectory anchor reconstructs the full arc."""
    index = EpisodicIndex()
    ids = []
    for cid in range(5):
        ids.append(index.bind(_traj_ep("s1", cid)))
    # Decoy in another session should never enter the walk.
    index.bind(_traj_ep("s2", 2))

    # Anchor in the middle (cycle 2). 'both' spans full session.
    traj = index.walk_trajectory(ids[2], direction="both")
    assert [e.cycle_id for e in traj] == [0, 1, 2, 3, 4]
    assert {e.session_id for e in traj} == {"s1"}
    # Anchor appears exactly once even though it is the join point.
    assert sum(1 for e in traj if e.episode_id == ids[2]) == 1

    # Anchor at leftmost: 'both' equals 'forward'.
    traj_left = index.walk_trajectory(ids[0], direction="both")
    assert [e.cycle_id for e in traj_left] == [0, 1, 2, 3, 4]

    # Anchor at rightmost: 'both' equals 'backward'.
    traj_right = index.walk_trajectory(ids[4], direction="both")
    assert [e.cycle_id for e in traj_right] == [0, 1, 2, 3, 4]
    print("  PASS: walk_trajectory both reconstructs full session from mid-anchor")


def test_walk_trajectory_invalid_direction_raises():
    """Unrecognized direction raises ValueError."""
    import pytest
    index = EpisodicIndex()
    real_id = index.bind(_traj_ep("s1", 0))

    with pytest.raises(ValueError):
        index.walk_trajectory(real_id, direction="sideways")
    with pytest.raises(ValueError):
        index.walk_trajectory(real_id, direction="")
    print("  PASS: walk_trajectory invalid direction raises")


if __name__ == "__main__":
    print("=" * 60)
    print("Episodic Memory Index — Test Suite")
    print("=" * 60)

    tests = [
        test_embedding_dimensions,
        test_basic_bind_recall,
        test_partial_cue_snarc,
        test_partial_cue_action,
        test_tag_filtering,
        test_consolidation,
        test_forget,
        test_recency_weighting,
        test_sqlite_persistence,
        test_empty_recall,
        test_access_count_updates,
        test_many_episodes_performance,
        test_from_wm_dump,
        test_wm_consolidate_integration,
        test_walk_trajectory_contiguous_session,
        test_walk_trajectory_stops_at_gap,
        test_walk_trajectory_excludes_other_sessions,
        test_walk_trajectory_empty_session_is_singleton,
        test_walk_trajectory_unknown_id_raises,
        test_walk_trajectory_backward_from_mid_session,
        test_walk_trajectory_backward_stops_at_gap,
        test_walk_trajectory_both_reconstructs_full_session,
        test_walk_trajectory_invalid_direction_raises,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
