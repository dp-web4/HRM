"""`camera` from the cortex (sage/embodiment/look.py; dp 2026-10-12, "the goal is to have it understand
directly what it's looking at"). On Sprout the eyes are CSI sensors only the running cortex can read, so a
seat declaring `camera_source: "cortex"` gets its frame from the cortex's current image.

The responder runs on real numpy frames and real OpenCV; the client runs as the real subprocess the seat
would run (`python3 -m sage.embodiment.look`), against a responder thread standing in for the cortex loop."""
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sage.embodiment import look  # noqa: E402
from sage.gateway import toolset  # noqa: E402
from sage.gateway.being_gate_client import BeingIntent, camera_command  # noqa: E402
from sage.gateway.hestia_dispatch import HestiaF1aDispatcher  # noqa: E402

FRAME = np.full((720, 1280, 3), 90, np.uint8)


def _ask(body, rid="r1", age=0.0):
    (body / look.REQUEST).write_text(json.dumps({"id": rid, "ts": time.time() - age}))


# ---- the cortex's half --------------------------------------------------------------------------

def test_the_cortex_answers_with_a_resized_jpeg_of_the_open_eye(tmp_path):
    _ask(tmp_path)
    r = look.respond([None, FRAME], [0, 1], "open", body_dir=str(tmp_path))
    assert r["ok"] and r["eye"] == "right" and r["width"] == look.WIDTH and r["height"] == 576
    data = Path(r["path"]).read_bytes()
    assert data[:3] == b"\xff\xd8\xff" and data[-2:] == b"\xff\xd9"
    assert json.loads((tmp_path / look.RESULT).read_text())["id"] == "r1"
    assert not (tmp_path / look.REQUEST).exists()


def test_closed_eyes_are_respected_not_overridden(tmp_path):
    _ask(tmp_path)
    r = look.respond([FRAME], [1], "closed", body_dir=str(tmp_path))
    assert r["ok"] is False and "eyes are closed" in r["why"]
    assert not (tmp_path / look.OUT_DIR).exists()


def test_no_frame_yet_and_stale_requests(tmp_path):
    _ask(tmp_path)
    assert look.respond([None], [1], "open", body_dir=str(tmp_path))["why"] == "no eye has delivered a frame yet"
    _ask(tmp_path, rid="old", age=look.STALE_S + 5)
    assert look.respond([FRAME], [1], "open", body_dir=str(tmp_path)) is None   # its asker is gone
    assert look.respond([FRAME], [1], "open", body_dir=str(tmp_path)) is None   # nothing pending


def test_only_the_last_few_answers_are_kept(tmp_path):
    for i in range(look.KEEP + 3):
        _ask(tmp_path, rid=f"r{i}")
        look.respond([FRAME], [1], "open", body_dir=str(tmp_path))
        time.sleep(0.01)
    assert len(list((tmp_path / look.OUT_DIR).glob("*.jpg"))) == look.KEEP


# ---- the seat's half, as the real subprocess ----------------------------------------------------

def _cortex(body, frames, gaze, stop):
    while not stop.is_set():
        look.respond(frames, [1], gaze, body_dir=str(body))
        time.sleep(0.05)


def _run_client(body, out, timeout=20):
    env = dict(os.environ, SAGE_BODY_DIR=str(body), PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, "-m", "sage.embodiment.look", str(out)], env=env,
                          capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))


@pytest.mark.parametrize("gaze,code", [("open", 0), ("closed", 4)])
def test_the_client_gets_the_frame_or_the_cortexs_reason(tmp_path, gaze, code):
    body, out = tmp_path / "body", tmp_path / "home" / "scratch" / "camera" / "f.jpg"
    body.mkdir()
    stop = threading.Event()
    t = threading.Thread(target=_cortex, args=(body, [FRAME], gaze, stop), daemon=True)
    t.start()
    try:
        p = _run_client(body, out)
    finally:
        stop.set(); t.join()
    assert p.returncode == code, p.stderr
    if code == 0:
        assert json.loads(p.stdout)["eye"] == "right"
        assert out.read_bytes()[:3] == b"\xff\xd8\xff"
        assert not list(out.parent.glob("*.tmp-*"))            # published by rename, nothing half-written
    else:
        assert "eyes are closed" in p.stderr and not out.exists()


def test_no_cortex_is_said_and_leaves_no_request_behind(tmp_path):
    rc = look.request(str(tmp_path / "f.jpg"), body_dir=str(tmp_path), wait_s=0.3)
    assert rc == 3 and not (tmp_path / look.REQUEST).exists()


# ---- the verb: composition, dispatch, availability ----------------------------------------------

def test_a_cortex_seat_composes_the_look_command_without_a_worktree(tmp_path):
    home = str(tmp_path)
    cmd = camera_command({"out_path": "scratch/camera/f.jpg"}, {"memory_root": home, "camera_source": "cortex"})
    assert cmd == f"python3 -m sage.embodiment.look {home}/scratch/camera/f.jpg"
    with pytest.raises(ValueError, match="escapes your home|plain path"):
        camera_command({"out_path": "../x.jpg"}, {"memory_root": home, "camera_source": "cortex"})
    with pytest.raises(ValueError, match="not enabled on this seat"):          # undeclared: the hold stands
        camera_command({"out_path": "scratch/camera/f.jpg"}, {"memory_root": home})


def test_the_dispatcher_runs_it_and_says_it_was_your_eye(tmp_path, monkeypatch):
    monkeypatch.setattr(HestiaF1aDispatcher, "_call",
                        lambda self, n, a: {"actionId": "act"} if n == "hestia_begin_action" else {}, raising=False)
    d = HestiaF1aDispatcher.__new__(HestiaF1aDispatcher)
    d.worktree, d.memory_root, d.camera_source = None, str(tmp_path), "cortex"
    import sage.gateway.hestia_dispatch as hd
    seen = {}

    def ok_run(cmd, **kw):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"\xff\xd8\xffJPEG\xff\xd9")
        return types.SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")
    monkeypatch.setattr(hd.subprocess, "run", ok_run)
    env = d._do_camera(BeingIntent("camera", {"out_path": "scratch/camera/f.jpg"}))
    assert env.ok and env.result["device"] == "cortex" and "YOUR OPEN EYE" in env.result["note"]
    assert seen["cmd"][:3] == ["python3", "-m", "sage.embodiment.look"]

    monkeypatch.setattr(hd.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=4, stdout=b"", stderr=b"no frame: your eyes are closed (gaze closed); open them to look"))
    env = d._do_camera(BeingIntent("camera", {"out_path": "scratch/camera/g.jpg"}))
    assert not env.ok and "eyes are closed" in env.result["stderr"] and "gave no frame" in env.result["note"]
    assert "busy" not in env.result["note"]                     # never the ffmpeg taxonomy


def test_availability_follows_the_running_cortex_on_a_cortex_seat():
    live = {"inventory": {"verbs": ["gaze"], "cortex_live": True}}
    dead = {"inventory": {"verbs": [], "cortex_live": False}}
    cfg = {"camera_source": "cortex"}
    assert "camera" not in toolset.unavailable(live, None, cfg)
    assert "not running" in toolset.unavailable(dead, None, cfg)["camera"]
    assert "worktree" in toolset.unavailable(live, None, {})["camera"]          # undeclared: unchanged
