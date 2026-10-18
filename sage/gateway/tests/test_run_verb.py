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
    import glob
    assert not glob.glob(f"/tmp/sage-run-test-being-{os.getpid()}-*"), "no staging dir survives"
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
    # /work is a size-limited tmpfs; each staged copy is bound into it READ-ONLY, from a stage dir
    # that belongs to this invocation alone (legion-claude and Codex on #470)
    from sage.gateway.being_gate_client import run_stage_dir, RUN_WORK_BYTES, RUN_TMP_BYTES
    stage = run_stage_dir("legion-being", ["scratch/eval.py", "scratch/game/moves.md"])
    assert stage.startswith(f"/tmp/sage-run-legion-being-{os.getpid()}-")
    assert f" --size {RUN_WORK_BYTES} --tmpfs /work" in cmd
    assert f" --size {RUN_TMP_BYTES} --tmpfs /tmp" in cmd
    assert f" --ro-bind {stage}/eval.py /work/eval.py" in cmd
    assert f" --ro-bind {stage}/moves.md /work/moves.md" in cmd
    assert cmd.rstrip().endswith("/work/eval.py"), cmd
    assert " --chdir /work " in cmd
    assert "/home/x/inst" not in cmd, "its home is never bound"
    assert cmd.count("--bind ") == 0, "no host directory is writable from inside"
    assert "--ro-bind-try /lib64 /lib64" in cmd, "an aarch64 seat without /lib64 still runs"
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
    assert cmd.count(" --bind ") == 0 and " --tmpfs /work" in cmd
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


def test_a_run_never_outlives_its_beat(monkeypatch):
    """The beat unit kills the whole beat at TimeoutStartSec (1500 s in the example unit), so a
    run gets the hour or the beat's remaining time minus a margin, whichever is less."""
    from sage.gateway import being_gate_client as b
    monkeypatch.setenv("SAGE_BEAT_TIMEOUT_S", "1500")
    t0 = b._PROCESS_T0
    budget, why = b.run_budget_s(now=t0 + 100)
    assert budget == 1500 - 100 - b.RUN_BEAT_MARGIN_S and why == "the time left in this beat"
    # NOVA'S BOUNDARIES: `left` is what remains after the record margin. Below RUN_MIN_S no run
    # starts (0); it used to be floored UP to 30 s, a run granted with no beat budget at all.
    at = lambda left: t0 + 1500 - b.RUN_BEAT_MARGIN_S - left          # noqa: E731
    for left, want in ((-1, 0), (0, 0), (1, 0), (29, 0), (30, 30)):
        assert b.run_budget_s(now=at(left))[0] == want, (left, b.run_budget_s(now=at(left)))
    monkeypatch.setenv("SAGE_BEAT_TIMEOUT_S", "7200")
    assert b.run_budget_s(now=t0 + 100) == (3600, "the run limit"), "a long beat allows the hour"
    # UNKNOWN IS UNKNOWN (legion-claude): no declared bound, no assumed 1500, wherever it runs
    monkeypatch.delenv("SAGE_BEAT_TIMEOUT_S")
    assert b.run_budget_s(now=t0 + 99999) == (3600, "the run limit"), "no declared bound"
    monkeypatch.setenv("SAGE_BEAT_TIMEOUT_S", "infinity")
    assert b.run_budget_s(now=t0 + 99999) == (3600, "the run limit"), "an unbounded unit"


def test_a_spent_beat_starts_no_run_and_touches_nothing(tmp_path, monkeypatch):
    import sage.gateway.being_gate_client as b
    monkeypatch.setattr(b, "sandbox_available", lambda: True)
    monkeypatch.setattr(b, "run_budget_s", lambda now=None: (0, "the time left in this beat"))
    d, home = _run_dispatcher(tmp_path)
    (home / "scratch" / "e.py").write_text("print(1)\n")
    calls = []
    d._call = lambda name, args: calls.append(name) or {"actionId": "w"}
    r = d._do_run(BeingIntent("run", {"path": "scratch/e.py"}))
    assert r.ok is False and "nothing was started" in r.error and "next beat" in r.error
    assert calls == [], "no witness row for a run that never started"


@needs_bwrap
def test_a_stopped_run_keeps_what_it_printed(tmp_path, monkeypatch):
    import sage.gateway.being_gate_client as b
    monkeypatch.setattr(b, "run_budget_s", lambda now=None: (2, "the time left in this beat"))
    d, home = _run_dispatcher(tmp_path)
    (home / "scratch" / "slow.py").write_text(
        "import sys, time\nprint('step 1 done', flush=True)\ntime.sleep(30)\nprint('never')\n")
    r = d._do_run(BeingIntent("run", {"path": "scratch/slow.py"}))
    assert r.ok and r.result["exit"] == -9, r.result
    assert "step 1 done" in r.result["stdout"] and "never" not in r.result["stdout"]
    assert "after 2s (the time left in this beat)" in r.result["stderr"]


@needs_bwrap
def test_run_is_held_to_its_memory_cap_from_inside(tmp_path):
    """An hour of unbounded allocation could take the machine's memory, and the being's own
    model with it. The cap is applied outside the sandbox (prlimit --as); from inside, an
    allocation past it must fail as a MemoryError the being can read, not take the box down."""
    from sage.gateway.being_gate_client import run_command, run_mem_cap_bytes, PRLIMIT
    if not os.path.exists(PRLIMIT):
        pytest.skip("no prlimit here")
    ctx = {"memory_root": "/home/x/inst", "member": "b"}
    from sage.gateway.being_gate_client import RUN_FSIZE_BYTES
    assert run_command({"path": "e.py"}, ctx).startswith(
        f"{PRLIMIT} --as={run_mem_cap_bytes()} --fsize={RUN_FSIZE_BYTES} -- ")
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


# -- the review of #470: what an hour can fill (legion-claude) ------------------------------

@needs_bwrap
def test_both_writable_places_are_size_bounded_from_inside(tmp_path, monkeypatch):
    """legion-claude probed on Legion: tmpfs pages are not address space, so a 300 MiB --as cap let
    640 MiB into /tmp. Both tmpfs are now sized; checked from inside with small sizes."""
    import sage.gateway.being_gate_client as b
    monkeypatch.setattr(b, "RUN_TMP_BYTES", 4 << 20)
    monkeypatch.setattr(b, "RUN_WORK_BYTES", 4 << 20)
    d, home = _run_dispatcher(tmp_path)
    (home / "scratch" / "fill.py").write_text(
        "def fill(p):\n"
        "    try:\n"
        "        with open(p, 'wb') as f:\n"
        "            for _ in range(16):\n"
        "                f.write(b'x' * (1 << 20))\n"
        "        return 'REACHED'\n"
        "    except OSError as e:\n"
        "        return 'denied ' + type(e).__name__\n"
        "print('tmp:', fill('/tmp/big'))\n"
        "print('work:', fill('big'))\n"
        "try:\n"
        "    open('fill.py', 'a').write('#')\n"
        "    print('input: REACHED')\n"
        "except OSError:\n"
        "    print('input: read-only')\n")
    r = d._do_run(BeingIntent("run", {"path": "scratch/fill.py"}))
    assert r.ok, r.error
    out = r.result["stdout"]
    assert "tmp: denied" in out and "work: denied" in out, out
    assert "input: read-only" in out, "the judged copy cannot be changed from inside: " + out


@needs_bwrap
def test_output_goes_to_capped_files_and_only_head_and_tail_come_back(tmp_path, monkeypatch):
    """capture_output read everything into the seat (outside every limit) before trimming it."""
    import sage.gateway.being_gate_client as b
    monkeypatch.setattr(b, "RUN_FSIZE_BYTES", 1 << 20)
    d, home = _run_dispatcher(tmp_path)
    (home / "scratch" / "loud.py").write_text(
        "import sys\nprint('FIRST LINE')\n"
        "for _ in range(64):\n    sys.stdout.write('y' * 65536)\n"
        "print('LAST LINE')\n")
    r = d._do_run(BeingIntent("run", {"path": "scratch/loud.py"}))
    assert r.ok, r.error
    out = r.result["stdout"]
    assert out.startswith("FIRST LINE"), out[:80]
    assert "more bytes not shown" in out and len(out) < d.RUN_OUTPUT_CHARS + 200
    # 4 MiB against a 1 MiB file cap: the write past the cap kills the run (SIGXFSZ), so the seat
    # never holds more than the cap, and it says the run did not finish
    assert r.result["exit"] != 0 and "LAST LINE" not in out


# -- the review of #470: staging and authority (Codex, a44126319, inverted) -----------------

def _codex_setup(tmp_path, monkeypatch, granted=None):
    from sage.gateway import being_gate_client as B
    home = tmp_path / "workspace" / "project" / "home"
    (home / "scratch").mkdir(parents=True)
    (home / "scratch" / "probe.py").write_text("print('synthetic')\n")
    monkeypatch.setattr(B, "sandbox_available", lambda: True)
    monkeypatch.setattr(B, "STAGE_ROOT", str(tmp_path / "stage"))
    monkeypatch.setattr(B, "run_budget_s", lambda now=None: (60, "the run limit"))
    from sage.gateway.hestia_dispatch import HestiaF1aDispatcher as D
    d = D.__new__(D)
    d.memory_root, d.member = str(home), "review-being"
    d.worktree, d.workspace = str(home), str(tmp_path / "workspace" / "project")
    d._call = lambda name, args: {"actionId": "synthetic-action"}
    g = granted if granted is not None else ((str(home), True),)
    d._verdict = types.SimpleNamespace(command=None, granted=tuple(r for r, _ in g), granted_reach=g)
    return d, home


class _FakePopen:
    """Stands in for the sandbox: records what the staged inputs held when it 'ran'."""
    seen = []
    on_wait = None

    def __init__(self, argv, **kw):
        self.argv = argv
        stage = [a for a in argv if "/work/" not in a and "-review-being-" in a]
        _FakePopen.seen.append({os.path.basename(p): Path(p).read_text() for p in stage
                                if os.path.isfile(p)})

    def wait(self, timeout=None):
        if _FakePopen.on_wait:
            f, _FakePopen.on_wait = _FakePopen.on_wait, None
            f()
        return 0

    def kill(self):
        pass


def test_a_data_file_outside_the_grants_is_judged_and_not_copied(tmp_path, monkeypatch):
    from sage.gateway import being_gate_client as B
    d, home = _codex_setup(tmp_path, monkeypatch)
    d.worktree = str(tmp_path / "workspace" / "project")
    external = tmp_path / "workspace" / "project" / "ungranted" / "sample.txt"
    external.parent.mkdir()
    external.write_text("synthetic outside-grant bytes")
    intent = B.BeingIntent("run", {"path": "scratch/probe.py", "data": [str(external)]})
    client = B.BeingGateClient.__new__(B.BeingGateClient)
    client.memory_root, client.worktree, client.workspace = d.memory_root, d.worktree, d.workspace
    client.member_id = d.member
    client._core = types.SimpleNamespace(NormalizedEvent=lambda **kw: types.SimpleNamespace(**kw))
    ev = client._normalize(intent)
    assert str(external.resolve()) in ev.paths, "the law sees every source it would read"
    assert str((home / "scratch" / "probe.py").resolve()) in ev.paths
    _FakePopen.seen = []
    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    r = d._do_run(intent)
    assert r.ok is False and "granted" in r.error, r
    assert _FakePopen.seen == [], "nothing ran, nothing was staged"
    # CONTROL: the same file under a granted root is copied in
    ext2 = tmp_path / "b" / "workspace" / "project" / "granted" / "sample.txt"
    ext2.parent.mkdir(parents=True)
    ext2.write_text("synthetic outside-grant bytes")
    d2, home2 = _codex_setup(tmp_path / "b", monkeypatch,
                             granted=((str(tmp_path / "b" / "workspace" / "project" / "home"), True),
                                      (str(ext2.parent), True)))
    d2.worktree = str(tmp_path / "b" / "workspace" / "project")
    _FakePopen.seen = []
    r2 = d2._do_run(B.BeingIntent("run", {"path": "scratch/probe.py", "data": [str(ext2)]}))
    assert r2.ok, r2.error
    assert _FakePopen.seen and _FakePopen.seen[0].get("sample.txt") == "synthetic outside-grant bytes"


def test_a_source_swapped_for_a_symlink_at_open_is_refused(tmp_path, monkeypatch):
    from sage.gateway import being_gate_client as B
    d, home = _codex_setup(tmp_path, monkeypatch)
    source = home / "scratch" / "data.txt"
    source.write_text("initial safe bytes")
    outside = tmp_path / "outside.txt"
    outside.write_text("synthetic out-of-reach bytes")
    real_open = os.open

    def racing_open(path, flags, *a, **kw):
        if str(path) == str(source.resolve()):
            repl = home / "scratch" / "replacement"
            repl.symlink_to(outside)
            os.replace(repl, source)
        return real_open(path, flags, *a, **kw)

    _FakePopen.seen = []
    monkeypatch.setattr(os, "open", racing_open)
    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    r = d._do_run(B.BeingIntent("run", {"path": "scratch/probe.py", "data": ["scratch/data.txt"]}))
    assert r.ok is False and "not copied in" in r.error, r
    assert all("out-of-reach" not in v for s in _FakePopen.seen for v in s.values())


def test_a_parent_directory_swapped_for_a_symlink_at_open_is_refused(tmp_path, monkeypatch):
    """O_NOFOLLOW guards only the last component; the descriptor's real location catches the rest."""
    from sage.gateway import being_gate_client as B
    d, home = _codex_setup(tmp_path, monkeypatch)
    sub = home / "scratch" / "sub"; sub.mkdir()
    (sub / "data.txt").write_text("initial safe bytes")
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
    (elsewhere / "data.txt").write_text("synthetic out-of-reach bytes")
    real_open = os.open

    def racing_open(path, flags, *a, **kw):
        if str(path) == str((sub / "data.txt").resolve()):
            os.rename(sub, home / "scratch" / "sub-old")
            (home / "scratch" / "sub").symlink_to(elsewhere)
        return real_open(path, flags, *a, **kw)

    _FakePopen.seen = []
    monkeypatch.setattr(os, "open", racing_open)
    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    r = d._do_run(B.BeingIntent("run", {"path": "scratch/probe.py", "data": ["scratch/sub/data.txt"]}))
    assert r.ok is False and "changed while it was being read" in r.error, r
    assert _FakePopen.seen == []


def test_an_overlapping_run_does_not_touch_the_first_runs_staging(tmp_path, monkeypatch):
    from sage.gateway import being_gate_client as B
    d, home = _codex_setup(tmp_path, monkeypatch)
    (home / "scratch" / "second.py").write_text("print('second')\n")
    first = B.run_stage_dir(d.member, ["scratch/probe.py"])
    observed = []

    def inner():
        assert os.path.isfile(os.path.join(first, "probe.py"))
        r = d._do_run(B.BeingIntent("run", {"path": "scratch/second.py"}))
        assert r.ok, r.error
        observed.append(os.path.isfile(os.path.join(first, "probe.py")))

    _FakePopen.seen, _FakePopen.on_wait = [], inner
    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    r = d._do_run(B.BeingIntent("run", {"path": "scratch/probe.py"}))
    assert r.ok and observed == [True], observed
    assert not os.path.exists(first), "and it is cleaned up after its own run"


def test_an_overlapping_run_of_the_same_files_is_refused_and_leaves_the_first_intact(tmp_path, monkeypatch):
    """Nova's re-review of #470: the stage is keyed by process and file NAMES, so a second run
    of the same names in the same process (re-entrant here; two threads alike) derives the SAME
    stage. It must be refused before it touches anything, not rmtree the first run's inputs."""
    from sage.gateway import being_gate_client as B
    d, home = _codex_setup(tmp_path, monkeypatch)
    first = B.run_stage_dir(d.member, ["scratch/probe.py"])
    observed = []

    def inner():
        (home / "scratch" / "probe.py").write_text("print('different bytes, same name')\n")
        r = d._do_run(B.BeingIntent("run", {"path": "scratch/probe.py"}))
        observed.append((r.ok, r.error or "",
                         os.path.isfile(os.path.join(first, "probe.py")),
                         os.path.isdir(first + ".out")))

    _FakePopen.seen, _FakePopen.on_wait = [], inner
    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    r = d._do_run(B.BeingIntent("run", {"path": "scratch/probe.py"}))
    assert r.ok, r.error
    (ok2, err2, inputs_left, out_left), = observed
    assert ok2 is False and "already in progress" in err2, err2
    assert inputs_left and out_left, "the first run's inputs and output survived the second call"
    assert _FakePopen.seen == [{"probe.py": "print('synthetic')\n"}], "only the first ran"
    assert not os.path.exists(first) and not os.path.exists(first + ".lock"), "then all cleaned up"
    # and once the first has answered, the same files run again
    _FakePopen.seen = []
    assert d._do_run(B.BeingIntent("run", {"path": "scratch/probe.py"})).ok


def test_three_actors_cannot_both_hold_the_stage_lock_across_inodes(tmp_path):
    """Nova's re-review at 54d592778, the interleaving it named: A holds inode X; B opens X and
    pauses before flock; A unlinks and unlocks; C creates the path anew (inode Y) and locks it;
    B then locks the unlinked X without contention. B must be refused: the path no longer names
    the file B locked, so the stage belongs to C."""
    from sage.gateway.hestia_dispatch import _lock_opened_stage, _release_stage_lock
    path = str(tmp_path / "stage.lock")
    flags = os.O_CREAT | os.O_RDWR
    a = os.open(path, flags, 0o600)
    assert _lock_opened_stage(path, a), "A takes the lock"
    b = os.open(path, flags, 0o600)                  # B opens inode X ... and pauses here
    x = os.fstat(b).st_ino
    _release_stage_lock(path, a)                     # A unlinks X, unlocks, closes
    c = os.open(path, flags, 0o600)
    assert os.fstat(c).st_ino != x, "C created a new inode at the same path"
    assert _lock_opened_stage(path, c), "C takes the lock on Y"
    import fcntl
    fcntl.flock(b, fcntl.LOCK_EX | fcntl.LOCK_NB)    # THE HAZARD IS REAL: B's flock on X succeeds
    assert _lock_opened_stage(path, b) is False, "B locked dead inode X: refused, not admitted"
    with pytest.raises(OSError):
        os.fstat(b)                                  # and B's fd was closed on refusal
    # C still holds Y: a fresh opener D is refused by plain contention
    d = os.open(path, flags, 0o600)
    assert _lock_opened_stage(path, d) is False
    _release_stage_lock(path, c)
    assert not os.path.exists(path), "nothing left behind"
