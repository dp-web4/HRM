"""Tests for gameplay_capture — synthetic trace replay + env mocking."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.cognition.router.data import RouterDatasetWriter
from sage.cognition.thalamic_router.gameplay_capture import (
    GameplayCapture,
    Trace,
    TraceStep,
    load_trace,
    synth_router_input,
    _frame_delta,
)


class MockGameState:
    def __init__(self, name):
        self.name = name


class MockFrameData:
    def __init__(self, frame, state_name="NOT_FINISHED", levels_completed=0):
        self.frame = frame
        self.state = MockGameState(state_name)
        self.levels_completed = levels_completed


class MockEnv:
    def __init__(self, initial_frame=None):
        self._tick = 0
        self._frame = initial_frame if initial_frame is not None else [[0] * 8 for _ in range(8)]

    def reset(self):
        self._tick = 0
        return MockFrameData(frame=self._frame)

    def step(self, action, data=None):
        self._tick += 1
        new_frame = [row[:] for row in self._frame]
        new_frame[self._tick % len(new_frame)][self._tick % len(new_frame[0])] = (
            (self._tick * 7) % 16
        )
        self._frame = new_frame
        if self._tick >= 5:
            return MockFrameData(frame=new_frame, state_name="WIN",
                                 levels_completed=self._tick)
        return MockFrameData(frame=new_frame, levels_completed=self._tick // 2)


def _make_env_factory(env=None):
    def factory():
        e = env or MockEnv()
        return e, e.reset()
    return factory


def test_load_trace_solutions_json_format(tmp_path):
    sol = tmp_path / "solutions.json"
    sol.write_text(json.dumps([
        [{"action": 1}, {"action": 3}],
        [{"action": 6, "data": {"x": 48, "y": 36}}],
    ]))
    trace = load_trace(sol, "cd82", "cd82-fb555c5d")
    assert trace.source == "solutions.json"
    assert len(trace.steps) == 3
    assert trace.steps[0].level == 0 and trace.steps[0].action == 1
    assert trace.steps[2].level == 1 and trace.steps[2].action == 6
    assert trace.steps[2].data == {"x": 48, "y": 36}
    assert trace.outcome["inferred_won"] is True


def test_load_trace_run_json_format(tmp_path):
    run = tmp_path / "run.json"
    run.write_text(json.dumps({
        "game_id": "cd82-fb555c5d",
        "steps": [
            {"step": 1, "level": 0, "action": "LEFT"},
            {"step": 2, "level": 0, "action": "CLICK", "x": 48, "y": 36},
            {"step": 3, "level": 1, "action": "UP"},
        ],
        "result": "WIN",
        "levels_completed": 2,
        "win_levels": 2,
        "final_level_index": 1,
    }))
    trace = load_trace(run, "cd82", "cd82-fb555c5d")
    assert trace.source == "run.json"
    assert len(trace.steps) == 3
    assert trace.steps[0].action == 3
    assert trace.steps[1].action == 6
    assert trace.steps[1].data == {"x": 48, "y": 36}
    assert trace.outcome["inferred_won"] is True


def test_load_trace_run_json_unknown_result_infers_from_levels(tmp_path):
    run = tmp_path / "run.json"
    run.write_text(json.dumps({
        "game_id": "x",
        "steps": [{"step": 1, "action": "UP"}],
        "levels_completed": 5,
        "win_levels": 5,
        "final_level_index": 4,
    }))
    trace = load_trace(run, "x", "x-1")
    assert trace.outcome["inferred_won"] is True


def test_load_trace_rejects_unknown_format(tmp_path):
    p = tmp_path / "weird.json"
    p.write_text(json.dumps({"not_steps": "nope"}))
    try:
        load_trace(p, "x", "x-1")
    except ValueError:
        return
    raise AssertionError("should have raised ValueError")


def test_frame_delta_identical_is_zero():
    a = [[0, 1], [2, 3]]
    assert _frame_delta(a, a) == 0.0


def test_frame_delta_different_is_positive():
    a = [[0, 0], [0, 0]]
    b = [[9, 0], [0, 0]]
    assert _frame_delta(a, b) > 0.0


def test_frame_delta_shape_mismatch_returns_one():
    a = [[0]]
    b = [[0, 0]]
    assert _frame_delta(a, b) == 1.0


def test_frame_delta_handles_none():
    assert _frame_delta(None, None) == 0.0


def test_synth_router_input_produces_valid_RouterInput():
    step = TraceStep(index=5, level=0, action=3)
    ri = synth_router_input(
        tick=5, prev_frame=[[0, 0]], curr_frame=[[1, 0]],
        step=step, game="cd82", level=0, atp=75.0,
    )
    assert ri.tick == 5
    assert ri.goal_id == "play:cd82:L0"
    assert ri.metabolic_state == "focus"
    assert ri.wm_goal_active is True
    assert "vision" in ri.sensory_modalities
    assert 0.0 <= ri.snarc_arousal <= 1.0


def _make_trace_3_steps() -> Trace:
    return Trace(
        game="testgame", game_id="testgame-0001",
        source="solutions.json", source_path="/fake/path.json",
        steps=[
            TraceStep(index=1, level=0, action=1),
            TraceStep(index=2, level=0, action=3),
            TraceStep(index=3, level=1, action=6, data={"x": 10, "y": 20}),
        ],
        outcome={"source_format": "solutions.json", "inferred_won": True, "steps_total": 3},
    )


def test_capture_end_to_end_emits_records(tmp_path):
    trace = _make_trace_3_steps()
    writer = RouterDatasetWriter(base_dir=tmp_path / "dataset",
                                 machine="cbp", compress=True)
    capture = GameplayCapture(trace=trace, writer=writer, machine="cbp",
                              env_factory=_make_env_factory())
    result = capture.run()
    writer.close()
    assert result.records_emitted == 3
    assert result.steps_applied == 3
    assert result.errors == []
    for rec in capture.records:
        assert rec.metadata.get("source") == "gameplay"
        assert rec.metadata.get("game") == "testgame"
        assert rec.metadata.get("synthetic_kernel_state") is True
        assert "game_outcome" in rec.metadata
        # Supervised-training labels
        assert "known_good_action" in rec.metadata
        assert rec.metadata["known_good_action"] in (1, 3, 6)   # from _make_trace_3_steps
        assert "known_good_level" in rec.metadata


def test_capture_writes_to_partition(tmp_path):
    trace = _make_trace_3_steps()
    base = tmp_path / "dataset"
    writer = RouterDatasetWriter(base_dir=base, machine="cbp", compress=True)
    capture = GameplayCapture(trace=trace, writer=writer, machine="cbp",
                              env_factory=_make_env_factory())
    capture.run()
    writer.close()
    files = list(base.glob("**/*.jsonl.gz"))
    assert len(files) >= 1
    with gzip.open(files[0], "rt") as f:
        recs = [json.loads(line) for line in f]
    gameplay_recs = [r for r in recs if (r.get("metadata") or {}).get("source") == "gameplay"]
    assert len(gameplay_recs) == 3


def test_capture_in_memory_records_get_outcome_attached(tmp_path):
    trace = _make_trace_3_steps()
    writer = RouterDatasetWriter(base_dir=tmp_path / "dataset",
                                 machine="cbp", compress=False)
    capture = GameplayCapture(trace=trace, writer=writer, machine="cbp",
                              env_factory=_make_env_factory())
    capture.run()
    writer.close()
    for rec in capture.records:
        outcome = rec.metadata.get("game_outcome")
        assert outcome is not None
        assert "source_format" in outcome


def test_capture_handles_env_failure_gracefully():
    trace = _make_trace_3_steps()

    def failing_factory():
        raise RuntimeError("env init failed in test")

    writer = MagicMock()
    capture = GameplayCapture(trace=trace, writer=writer, machine="cbp",
                              env_factory=failing_factory)
    result = capture.run()
    assert result.records_emitted == 0
    assert result.errors != []


def test_capture_atp_decays_per_step(tmp_path):
    trace = _make_trace_3_steps()
    writer = RouterDatasetWriter(base_dir=tmp_path / "dataset",
                                 machine="cbp", compress=False)
    capture = GameplayCapture(
        trace=trace, writer=writer, machine="cbp",
        env_factory=_make_env_factory(),
        atp_initial=80.0, atp_decay_per_step=1.0,
    )
    capture.run()
    writer.close()
    atps = [rec.router_input.atp_level for rec in capture.records]
    assert atps[0] == 80.0
    assert atps[-1] < atps[0]


def test_capture_records_have_machine_stamp(tmp_path):
    trace = _make_trace_3_steps()
    writer = RouterDatasetWriter(base_dir=tmp_path / "dataset",
                                 machine="cbp", compress=False)
    capture = GameplayCapture(trace=trace, writer=writer, machine="cbp",
                              env_factory=_make_env_factory())
    capture.run()
    writer.close()
    for rec in capture.records:
        assert rec.machine == "cbp"


def test_capture_all_records_have_valid_router_output(tmp_path):
    trace = _make_trace_3_steps()
    writer = RouterDatasetWriter(base_dir=tmp_path / "dataset",
                                 machine="cbp", compress=False)
    capture = GameplayCapture(trace=trace, writer=writer, machine="cbp",
                              env_factory=_make_env_factory())
    capture.run()
    writer.close()
    for rec in capture.records:
        ok, reason = rec.router_output.validate()
        assert ok, f"invalid output: {reason}"


def test_capture_known_good_labels_match_trace_actions(tmp_path):
    """Each record's known_good_action must equal the corresponding TraceStep.action."""
    trace = _make_trace_3_steps()
    writer = RouterDatasetWriter(base_dir=tmp_path / "dataset",
                                 machine="cbp", compress=False)
    capture = GameplayCapture(trace=trace, writer=writer, machine="cbp",
                              env_factory=_make_env_factory())
    capture.run()
    writer.close()
    assert len(capture.records) == 3
    assert capture.records[0].metadata["known_good_action"] == 1   # UP
    assert capture.records[1].metadata["known_good_action"] == 3   # LEFT
    assert capture.records[2].metadata["known_good_action"] == 6   # CLICK
    # Click step carries click data
    assert capture.records[2].metadata["known_good_data"] == {"x": 10, "y": 20}
    # Non-click steps have None data
    assert capture.records[0].metadata["known_good_data"] is None


def test_capture_known_good_levels_match_trace_levels(tmp_path):
    """known_good_level passes through the trace's per-step level."""
    trace = _make_trace_3_steps()
    writer = RouterDatasetWriter(base_dir=tmp_path / "dataset",
                                 machine="cbp", compress=False)
    capture = GameplayCapture(trace=trace, writer=writer, machine="cbp",
                              env_factory=_make_env_factory())
    capture.run()
    writer.close()
    assert capture.records[0].metadata["known_good_level"] == 0
    assert capture.records[1].metadata["known_good_level"] == 0
    assert capture.records[2].metadata["known_good_level"] == 1
