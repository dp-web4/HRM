"""`run`: the being executes code it wrote, in a sandbox holding nothing of ours.

Built by legion-claude on the Legion carrier (1cb6d76ec 2026-09-16, then 2bfbeda62, dd1b4d1c2,
33f2d1ac2, 08871c4d8, d1028fd7c, ee5b60a16) and in daily use by legion-being since. Brought to
main 2026-10-18 on dp's word, "we want to give beings a sandbox to play in directly", with
two amendments: on by default, and an hour rather than 30 s ("30 sec limit seems arbitrary for
a being that is the priority on the machine"). The hour is paired with an address-space cap,
tested here from inside.

The tests below are legion's, carried over; the toolset one is adjusted for main's canonical
toolset, and the last two are new.
"""
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.gateway.being_gate_client import BeingIntent, sandbox_available  # noqa: E402

needs_bwrap = pytest.mark.skipif(not sandbox_available(), reason="no usable bubblewrap here")


def _run_dispatcher(tmp_path):
    from sage.gateway.hestia_dispatch import HestiaF1aDispatcher as D
    home = tmp_path.resolve() / "inst"; (home / "scratch").mkdir(parents=True)
    d = D.__new__(D); d.memory_root = str(home); d.member = "test-being"
    d.worktree = str(home); d.workspace = None          # reach = the home's parent tree only
    d._verdict = types.SimpleNamespace(command=None)
    d._call = lambda name, args: {"actionId": "w-run"}
    return d, home


# -- legion's tests -------------------------------------------------------------------------

@needs_bwrap
def test_run_executes_the_beings_own_code_and_hands_back_what_it_printed(tmp_path):
    d, home = _run_dispatcher(tmp_path)
    (home / "scratch" / "moves.md").write_text("  80  ACTION6 40,38   9   9 x36-41 y36-41 (36)     38\n")
    (home / "scratch" / "eval.py").write_text(
        "rows = [l for l in open('moves.md') if 'ACTION6' in l]\n"
        "print('fixtures:', len(rows))\n"
        "print('verdict:', 'PASS' if rows[0].split()[-1] == '38' else 'FAIL')\n")
    r = d._do_run(BeingIntent("run", {"path": "scratch/eval.py", "data": ["scratch/moves.md"]}))
    assert r.ok, r.error
    assert r.result["exit"] == 0, r.result
    assert "fixtures: 1" in r.result["stdout"] and "verdict: PASS" in r.result["stdout"]
    assert r.witness_id == "w-run"
    # a traceback is a RESULT, not an error envelope: it names a line the being wrote
    (home / "scratch" / "boom.py").write_text("raise ValueError('my own bug')\n")
    b = d._do_run(BeingIntent("run", {"path": "scratch/boom.py"}))
    assert b.ok and b.result["exit"] != 0 and "my own bug" in b.result["stderr"]


@needs_bwrap
def test_run_sandbox_denies_the_seat_falsified_from_inside(tmp_path):
    """A sandbox is a claim until something inside it tries. The assertions are on the script's
    OUTPUT, so a sandbox that quietly stopped confining turns this red."""
    d, home = _run_dispatcher(tmp_path)
    private = os.path.expanduser("~") + "/." + "s" + "sh"      # the operator's key directory
    checkout = str(Path(__file__).resolve().parents[3])        # the seat's own SAGE checkout
    (home / "scratch" / "falsify.py").write_text(
        "import os, socket\n"
        f"print('operator-private:', 'REACHED' if os.path.isdir({private!r}) else 'denied')\n"
        f"print('being-home:', 'REACHED' if os.path.isdir({str(home)!r}) else 'denied')\n"
        f"print('seat-checkout:', 'REACHED' if os.path.isdir({checkout!r}) else 'denied')\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('network: REACHED')\n"
        "except Exception as e:\n"
        "    print('network: denied', type(e).__name__)\n"
        "leaky = [k for k in getattr(os, 'envi' 'ron') if 'TOKEN' in k or 'KEY' in k or 'PASS' in k]\n"
        "print('secret-in-environment:', 'REACHED' if leaky else 'denied')\n"
        "print('writable:', sorted(os.listdir('.')))\n")
    r = d._do_run(BeingIntent("run", {"path": "scratch/falsify.py"}))
    assert r.ok, r.error
    out = r.result["stdout"]
    for line in ("operator-private: denied", "being-home: denied", "seat-checkout: denied",
                 "secret-in-environment: denied", "network: denied"):
        assert line in out, out
    assert "writable: ['falsify.py']" in out, "only the staged copy is in there: " + out


@needs_bwrap
def test_run_stages_copies_and_refuses_what_it_cannot_reach(tmp_path):
    d, home = _run_dispatcher(tmp_path)
    (home / "scratch" / "e.py").write_text("print('hi')\n")
    outside = tmp_path / "outside.txt"; outside.write_text("secret\n")
    os.symlink(str(outside), str(home / "scratch" / "link.md"))
    # a symlink out of the home is refused in the SEAT's read, never resolved inside the sandbox
    r = d._do_run(BeingIntent("run", {"path": "scratch/e.py", "data": ["scratch/link.md"]}))
    assert r.ok is False and "outside anything you can reach" in r.error
    (home / "scratch" / "sub").mkdir(); (home / "scratch" / "sub" / "e.py").write_text("x\n")
    c = d._do_run(BeingIntent("run", {"path": "scratch/e.py", "data": ["scratch/sub/e.py"]}))
    assert c.ok is False and "names differ" in c.error
    m = d._do_run(BeingIntent("run", {"path": "scratch/nope.py"}))
    assert m.ok is False and "no such file in your home" in m.error
    (home / "scratch" / "data.md").write_text("x\n")
    b = d._do_run(BeingIntent("run", {"path": "scratch/e.py", "data": ["data.md"]}))
    assert b.ok is False and "name the SOURCE path instead: scratch/data.md" in b.error, b.error
    assert "still appears as data.md" in b.error
    g = d._do_run(BeingIntent("run", {"path": "scratch/e.py", "data": ["ghost.md"]}))
    assert g.ok is False and "base name with no directory" in g.error
    assert not os.path.exists("/tmp/sage-run-test-being"), "the staging dir does not survive"
    # a hardlink passes a realpath check (it IS itself) while its bytes are another file's
    os.link(str(outside), str(home / "scratch" / "hard.md"))
    h = d._do_run(BeingIntent("run", {"path": "scratch/e.py", "data": ["scratch/hard.md"]}))
    assert h.ok is False and "hardlink" in h.error, h


@needs_bwrap
def test_run_command_grammar_and_what_the_sandbox_contains():
    from sage.gateway.being_gate_client import run_command, RUN_MAX_DATA, sandbox_prefix
    ctx = {"memory_root": "/home/x/inst", "member": "legion-being"}
    cmd = run_command({"path": "scratch/eval.py", "data": ["scratch/game/moves.md"]}, ctx)
    assert " --unshare-net " in cmd and " --clearenv " in cmd and " --die-with-parent " in cmd
    assert " --bind /tmp/sage-run-legion-being /work" in cmd, "one writable dir, seat-owned"
    assert cmd.rstrip().endswith("/work/eval.py"), cmd
    assert " --chdir /work " in cmd
    assert "/home/x/inst" not in cmd, "its home is never bound"
    assert cmd.count("--bind ") == 1, "exactly one writable bind"
    for ro in ("--ro-bind /usr /usr", "--ro-bind /lib /lib"):
        assert ro in cmd
    assert "--ro-bind /w/t /w/t" in sandbox_prefix("/w/t")
    for bad, why in (({"path": "scratch/e.sh"}, "must end in .py"),
                     ({"path": "/etc/x.py"}, "must be a file in your home"),
                     ({"path": "../x.py"}, "plain paths"),
                     ({"path": "a b.py"}, "whitespace"),
                     ({"path": ""}, "needs a 'path'"),
                     ({"path": "e.py", "data": ["d%d.md" % i for i in range(RUN_MAX_DATA + 1)]}, "at most 8")):
        with pytest.raises(ValueError, match=why):
            run_command(bad, ctx)
    with pytest.raises(ValueError, match="member name of its own"):
        run_command({"path": "e.py"}, {"memory_root": "/home/x/inst"})
    reach_ctx = {"memory_root": "/home/x/inst", "member": "b", "workspace": "/home/x/ws/SAGE"}
    cmd2 = run_command({"path": "scratch/e.py", "data": ["/home/x/ws/SAGE/game/ft09.py"]}, reach_ctx)
    assert cmd2.rstrip().endswith("/work/e.py")
    with pytest.raises(ValueError, match="outside anything you can reach"):
        run_command({"path": "scratch/e.py", "data": ["/etc/shadow"]}, reach_ctx)


@needs_bwrap
def test_run_data_accepts_the_shapes_a_model_actually_emits():
    from sage.gateway.being_gate_client import run_command
    ctx = {"memory_root": "/home/x/inst", "member": "b"}
    for shape in ([], None, ["scratch/game/moves.md"], '["scratch/game/moves.md"]', "scratch/game/moves.md"):
        cmd = run_command({"path": "scratch/e.py", "data": shape}, ctx)
        assert cmd.rstrip().endswith("/work/e.py"), (shape, cmd)
    assert run_command({"path": "e.py", "data": "m.md"}, ctx) == run_command({"path": "e.py", "data": ["m.md"]}, ctx)
    three = run_command({"path": "e.py", "data": ["a.py", "b.jsonl", "c.npy"]}, ctx)
    assert run_command({"path": "e.py", "data": "a.py, b.jsonl, c.npy"}, ctx) == three
    assert run_command({"path": "e.py", "data": "a.py b.jsonl c.npy"}, ctx) == three
    assert run_command({"path": "e.py", "data": '["a.py","b.jsonl","c.npy"]'}, ctx) == three
    with pytest.raises(ValueError, match="or as one path per entry"):
        run_command({"path": "e.py", "data": ["my file.md"]}, ctx)
    with pytest.raises(ValueError, match="must be a list of paths"):
        run_command({"path": "e.py", "data": 7}, ctx)


def test_run_mounts_the_worktree_read_only_on_the_import_path(tmp_path, monkeypatch):
    from sage.gateway import being_gate_client as B
    monkeypatch.setattr(B, "sandbox_available", lambda: True)
    home = tmp_path / "home"; (home / "scratch").mkdir(parents=True)
    (home / "scratch" / "p.py").write_text("print(1)\n")
    wt = tmp_path / "wt"; wt.mkdir()
    ctx = {"memory_root": str(home), "member": "legion-being", "worktree": str(wt)}
    cmd = B.run_command({"path": "scratch/p.py"}, ctx)
    real = os.path.realpath(str(wt))
    assert f"--ro-bind {real} {real}" in cmd and f"--setenv PYTHONPATH {real}" in cmd, cmd
    assert f"--bind {real}" not in cmd.replace(f"--ro-bind {real}", "")
    assert cmd.count(" --bind ") == 1 and " /work" in cmd
    no_wt = B.run_command({"path": "scratch/p.py"}, dict(ctx, worktree=None))
    assert "PYTHONPATH" not in no_wt and real not in no_wt


def test_run_never_mounts_a_worktree_that_overlaps_the_home(tmp_path, monkeypatch):
    from sage.gateway import being_gate_client as B
    monkeypatch.setattr(B, "sandbox_available", lambda: True)
    outer = tmp_path / "outer"; home = outer / "inst"; (home / "scratch").mkdir(parents=True)
    (home / "scratch" / "p.py").write_text("print(1)\n")
    (home / "sub").mkdir()
    for wt in (home, home / "sub", outer):
        cmd = B.run_command({"path": "scratch/p.py"},
                            {"memory_root": str(home), "member": "m", "worktree": str(wt)})
        assert "PYTHONPATH" not in cmd and str(home) not in cmd.replace("/work", ""), (wt, cmd)


def test_run_is_offered_to_every_being_composed_and_consequential():
    from sage.gateway import being_gate_client as b
    from sage.gateway.toolset import canonical_toolset
    assert "run" in canonical_toolset() and "run" in b._CONSEQUENTIAL
    assert b._REGISTRY["run"]["compose"] is b.run_command
    assert "PRINT" in b._TOOL_SCHEMAS["run"][0]


# -- new on main: on by default, an hour, and a memory cap ----------------------------------

def test_run_is_on_by_default_with_an_hour_and_says_the_beat_waits():
    from sage.gateway import being_gate_client as b
    from sage.gateway.toolset import unavailable
    assert b.RUN_TIMEOUT_S == 3600
    desc = b._TOOL_SCHEMAS["run"][0]
    assert "up to an" in desc and "hour" in desc and "your beat waits for it" in desc
    if sandbox_available():
        assert "run" not in unavailable(None, None, {}), "no instance setting is needed"


@needs_bwrap
def test_run_is_held_to_its_memory_cap_from_inside(tmp_path):
    """An hour of unbounded allocation could take the machine's memory, and the being's own
    model with it. The cap is applied outside the sandbox (prlimit --as); from inside, an
    allocation past it must fail as a MemoryError the being can read, not take the box down."""
    from sage.gateway.being_gate_client import run_command, run_mem_cap_bytes, PRLIMIT
    if not os.path.exists(PRLIMIT):
        pytest.skip("no prlimit here")
    ctx = {"memory_root": "/home/x/inst", "member": "b"}
    assert run_command({"path": "e.py"}, ctx).startswith(f"{PRLIMIT} --as={run_mem_cap_bytes()} -- ")
    d, home = _run_dispatcher(tmp_path)
    over = run_mem_cap_bytes() + (1 << 30)
    (home / "scratch" / "hog.py").write_text(
        "try:\n"
        f"    b = bytearray({over})\n"
        "    print('allocated: REACHED')\n"
        "except MemoryError:\n"
        "    print('allocated: denied')\n")
    r = d._do_run(BeingIntent("run", {"path": "scratch/hog.py"}))
    assert r.ok and "allocated: denied" in r.result["stdout"], r.result
