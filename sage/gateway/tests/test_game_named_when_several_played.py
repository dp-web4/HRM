"""A `game` call that names no game, from a being that has played several, is refused, not defaulted.

legion-being, 2026-10-14: days into ar25, it left `game` out three times. Each call ran on ft09,
appended moves to ft09's replayed history, and its expectations were checked against ft09's board.
"""
import os
import time

import pytest

from sage.gateway.being_gate_client import game_command

STEPPER = "/opt/seat/being_board_step.py"


def _ctx(root):
    return {"game_stepper": STEPPER, "memory_root": str(root)}


def _played(root, game, age=0.0):
    d = root / "scratch" / "game"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{game}_actions.jsonl"
    p.write_text('{"n": 1}\n')
    t = time.time() - age
    os.utime(p, (t, t))


def test_no_history_still_defaults_to_ft09(tmp_path):
    cmd = game_command({"probes": [["ACTION1"]]}, _ctx(tmp_path))
    assert "ft09" in cmd


def test_one_game_played_is_the_one_meant(tmp_path):
    _played(tmp_path, "ar25")
    cmd = game_command({"probes": [["ACTION1"]]}, _ctx(tmp_path))
    assert "ar25" in cmd and "ft09" not in cmd


def test_several_played_and_none_named_is_refused_naming_them(tmp_path):
    _played(tmp_path, "ft09", age=3600)
    _played(tmp_path, "ar25", age=10)
    with pytest.raises(ValueError) as e:
        game_command({"probes": [["ACTION1"]]}, _ctx(tmp_path))
    msg = str(e.value)
    assert "name the game" in msg and "Nothing ran" in msg
    assert msg.index("ar25") < msg.index("ft09")          # most recent first


def test_naming_the_game_always_works(tmp_path):
    _played(tmp_path, "ft09", age=3600)
    _played(tmp_path, "ar25", age=10)
    cmd = game_command({"game": "ft09", "probes": [["ACTION1"]]}, _ctx(tmp_path))
    assert "ft09" in cmd
