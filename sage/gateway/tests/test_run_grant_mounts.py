"""A run sees what the being may touch, at the same paths, and nothing else.

dp, 2026-10-23: "the sandbox might need to reflect being's actual grants, including writes where
granted. otherwise there is a misalignment of permissions between what it can do directly vs what
it can do using tools/code." Measured that morning on cbp-being: memory_read had read the
governance closure file on every beat, and scratch/check_file.py, run in the sandbox, printed
"File exists: False" for the same path. The being believed its check and filed a dozen scope
requests for a file it already had.

The mounts are READ-ONLY (Nova's HOLD on 76f079bf0): grant modes are not enforced by the gate
yet, so a run supplies no write, delete or new file through any grant.

The real-run tests assert on what a script prints from INSIDE bubblewrap, so a mount that stopped
mounting, or started showing the home, turns them red.
"""
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.gateway import being_gate_client as B  # noqa: E402
from sage.gateway.being_gate_client import BeingIntent, run_command, run_grant_mounts  # noqa: E402

needs_bwrap = pytest.mark.skipif(B.run_sandbox_kind() != "bwrap", reason="no usable bubblewrap here")


def _world(tmp_path):
    root = tmp_path.resolve()
    home = root / "inst"; (home / "scratch").mkdir(parents=True)
    (home / "journal.md").write_text("mine\n")
    ws = root / "ws"
    gov = ws / "gov"; gov.mkdir(parents=True); (gov / "closure.py").write_text("CLOSURE = 1\n")
    ro = ws / "ro"; ro.mkdir(); (ro / "r.txt").write_text("R\n")
    one = ws / "one.txt"; one.write_text("ONE\n")
    exact = ws / "exactdir"; exact.mkdir(); (exact / "secret.txt").write_text("S\n")
    ungranted = ws / "ungranted"; ungranted.mkdir(); (ungranted / "x.txt").write_text("X\n")
    os.symlink(str(ungranted), str(ws / "link"))
    return root, home, ws


def _reach(ws):
    return ((str(ws / "gov"), True), (str(ws / "ro"), True), (str(ws / "one.txt"), False),
            (str(ws / "exactdir"), False), (str(ws / "link"), True))


def _dispatcher(home, reach):
    from sage.gateway.hestia_dispatch import HestiaF1aDispatcher as D
    d = D.__new__(D); d.memory_root = str(home); d.member = "test-being"
    d.worktree = None; d.workspace = None
    d._verdict = types.SimpleNamespace(command=None, granted=tuple(r for r, _ in reach),
                                       granted_reach=reach)
    d._call = lambda name, args: {"actionId": "w-run"}
    return d


# -- which grants become mounts -----------------------------------------------------------------

def test_recursive_dirs_and_granted_files_mount_and_the_rest_says_why_not(tmp_path):
    root, home, ws = _world(tmp_path)
    mounts, skipped, mask = run_grant_mounts({"memory_root": str(home), "granted_reach": _reach(ws)})
    assert mounts == sorted([(str(ws / "gov"), "dir"), (str(ws / "one.txt"), "file"),
                             (str(ws / "ro"), "dir")])
    why = dict(skipped)
    assert "exact grant on a directory" in why[str(ws / "exactdir")]
    assert "symlink" in why[str(ws / "link")]
    assert mask is False


def test_the_home_is_never_a_mount_and_a_grant_around_it_masks_it(tmp_path):
    root, home, ws = _world(tmp_path)
    inside = run_grant_mounts({"memory_root": str(home),
                               "granted_reach": ((str(home), True), (str(home / "scratch"), True))})
    assert inside == ([], [], False)
    around = run_grant_mounts({"memory_root": str(home), "granted_reach": ((str(root), True),)})
    assert around[0] == [(str(root), "dir")] and around[2] is True
    cmd = run_command({"path": "scratch/a.py"}, {"memory_root": str(home), "member": "test-being",
                                                  "granted_reach": ((str(root), True),)})
    assert f"--tmpfs {home} --remount-ro {home}" in cmd
    assert cmd.index(f"--ro-bind {root} {root}") < cmd.index(f"--tmpfs {home}")


def test_system_paths_and_tmp_itself_are_not_mounted_and_nesting_collapses(tmp_path):
    root, home, ws = _world(tmp_path)
    reach = (("/usr/lib", True), ("/tmp", True), ("/", True), (str(ws), True), (str(ws / "gov"), True))
    mounts, skipped, _ = run_grant_mounts({"memory_root": str(home), "granted_reach": reach})
    assert mounts == [(str(ws), "dir")]
    assert {p for p, _ in skipped} == {"/usr/lib", "/tmp", "/"}


def test_the_line_is_the_same_whatever_order_the_grants_arrive_in(tmp_path):
    """The gate composes from the snapshot's grants and the dispatcher from the verdict's; the
    dispatcher refuses unless the two lines are byte-identical, so order must not matter."""
    root, home, ws = _world(tmp_path)
    args = {"path": "scratch/a.py"}
    base = {"memory_root": str(home), "member": "test-being"}
    a = run_command(args, {**base, "granted_reach": _reach(ws)})
    b = run_command(args, {**base, "granted_reach": tuple(reversed(_reach(ws)))})
    assert a == b and f"--ro-bind {ws / 'gov'} {ws / 'gov'}" in a


def test_the_gate_side_hands_run_the_policy_grants_and_no_other_verb():
    c = B.BeingGateClient.__new__(B.BeingGateClient)
    c._policy_grants = lambda: (("/x",), (("/x", True),))
    assert c._compose_ctx("run")["granted_reach"] == (("/x", True),)
    assert "granted_reach" not in c._compose_ctx("search")
    assert "granted_reach" not in c._compose_ctx()


@needs_bwrap
def test_a_line_judged_without_the_grants_is_not_executed_with_them(tmp_path):
    root, home, ws = _world(tmp_path)
    (home / "scratch" / "a.py").write_text("print('ran')\n")
    d = _dispatcher(home, _reach(ws))
    d._verdict.command = run_command({"path": "scratch/a.py"},
                                     {"memory_root": str(home), "member": "test-being"})
    r = d._do_run(BeingIntent("run", {"path": "scratch/a.py"}))
    assert r.ok is False and "not the command this dispatcher" in r.error


# -- from inside the sandbox ------------------------------------------------------------------

@needs_bwrap
def test_inside_a_run_granted_paths_read_and_none_can_be_written_changed_or_removed(tmp_path):
    """Nova's HOLD on 76f079bf0: grant modes (hestia #1370) are not enforced by the gate yet,
    so a run must never supply a write, a delete or a new file where a grant may only allow a
    read. Every target here is writable on disk, so only the mount can be what refuses."""
    root, home, ws = _world(tmp_path)
    (home / "scratch" / "probe.py").write_text(
        "import os\n"
        f"print('granted-file:', open({str(ws / 'gov' / 'closure.py')!r}).read().strip())\n"
        f"print('exact-file:', open({str(ws / 'one.txt')!r}).read().strip())\n"
        "def attempt(name, fn):\n"
        "    try:\n"
        "        fn(); print(name + ': REACHED')\n"
        "    except OSError as e:\n"
        "        print(name + ': denied', type(e).__name__)\n"
        f"attempt('create', lambda: open({str(ws / 'gov' / 'new.txt')!r}, 'w').write('x'))\n"
        f"attempt('overwrite', lambda: open({str(ws / 'gov' / 'closure.py')!r}, 'w').write('x'))\n"
        f"attempt('append-file-grant', lambda: open({str(ws / 'one.txt')!r}, 'a').write('x'))\n"
        f"attempt('delete', lambda: os.remove({str(ws / 'gov' / 'closure.py')!r}))\n"
        f"attempt('chmod', lambda: os.chmod({str(ws / 'gov' / 'closure.py')!r}, 0o777))\n"
        f"attempt('mkdir', lambda: os.mkdir({str(ws / 'gov' / 'sub')!r}))\n"
        f"print('ungranted:', os.path.exists({str(ws / 'ungranted' / 'x.txt')!r}))\n"
        f"print('exact-dir-content:', os.path.exists({str(ws / 'exactdir' / 'secret.txt')!r}))\n"
        f"print('via-symlink:', os.path.exists({str(ws / 'link' / 'x.txt')!r}))\n"
        f"print('home:', os.path.exists({str(home / 'journal.md')!r}))\n")
    before = {p: p.read_bytes() for p in (ws / "gov" / "closure.py", ws / "one.txt")}
    modes = {p: p.stat().st_mode for p in before}
    d = _dispatcher(home, _reach(ws))
    r = d._do_run(BeingIntent("run", {"path": "scratch/probe.py"}))
    assert r.ok, r.error
    out = r.result["stdout"]
    assert "REACHED" not in out, out
    for line in ("granted-file: CLOSURE = 1", "exact-file: ONE", "create: denied",
                 "overwrite: denied", "append-file-grant: denied", "delete: denied",
                 "chmod: denied", "mkdir: denied", "ungranted: False",
                 "exact-dir-content: False", "via-symlink: False", "home: False"):
        assert line in out, out
    assert {p: p.read_bytes() for p in before} == before
    assert {p: p.stat().st_mode for p in before} == modes
    assert not (ws / "gov" / "new.txt").exists() and not (ws / "gov" / "sub").exists()
    note = r.result["note"]
    assert str(ws / "gov") in note and "READ-ONLY" in note and "os.path.exists() is False" in note
    assert "Not mounted" in note and str(ws / "exactdir") in note


@needs_bwrap
@pytest.mark.parametrize("around", [False, True])
def test_a_symlink_inside_a_grant_that_points_into_the_home_reaches_nothing(tmp_path, around):
    """Nova: home masking must not be bypassable by a symlink path. A link resolves inside the
    sandbox's namespace, where the home is either absent (grant beside it) or the empty mask
    (grant around it)."""
    root, home, ws = _world(tmp_path)
    os.symlink(str(home / "journal.md"), str(ws / "gov" / "tohome"))
    os.symlink(str(home), str(ws / "gov" / "homedir"))
    (home / "scratch" / "probe.py").write_text(
        "import os\n"
        f"print('link-file:', os.path.exists({str(ws / 'gov' / 'tohome')!r}))\n"
        f"print('link-dir:', os.path.isdir({str(ws / 'gov' / 'homedir')!r}) and "
        f"os.listdir({str(ws / 'gov' / 'homedir')!r}))\n")
    reach = (((str(root), True), (str(home / "scratch"), True)) if around
             else ((str(ws / "gov"), True),))
    d = _dispatcher(home, reach)
    r = d._do_run(BeingIntent("run", {"path": "scratch/probe.py"}))
    assert r.ok, r.error
    out = r.result["stdout"]
    assert "link-file: False" in out, out
    assert ("link-dir: []" if around else "link-dir: False") in out, out


@needs_bwrap
def test_a_grant_that_contains_the_home_shows_it_empty_and_read_only(tmp_path):
    root, home, ws = _world(tmp_path)
    (home / "scratch" / "probe.py").write_text(
        "import os\n"
        f"print('home-listing:', sorted(os.listdir({str(home)!r})))\n"
        "try:\n"
        f"    open({str(home / 'journal.md')!r}, 'w').write('x'); print('home-write: REACHED')\n"
        "except OSError as e:\n"
        "    print('home-write: denied', type(e).__name__)\n"
        f"print('around:', os.path.exists({str(ws / 'gov' / 'closure.py')!r}))\n")
    d = _dispatcher(home, ((str(root), True),))
    r = d._do_run(BeingIntent("run", {"path": "scratch/probe.py"}))
    assert r.ok, r.error
    out = r.result["stdout"]
    assert "home-listing: []" in out and "home-write: denied" in out and "around: True" in out, out
    assert (home / "journal.md").read_text() == "mine\n"


@needs_bwrap
def test_with_no_grants_the_run_is_what_it_was(tmp_path):
    root, home, ws = _world(tmp_path)
    (home / "scratch" / "a.py").write_text("import os\nprint(sorted(os.listdir('.')))\n")
    d = _dispatcher(home, ())
    r = d._do_run(BeingIntent("run", {"path": "scratch/a.py"}))
    assert r.ok and "['a.py']" in r.result["stdout"]
    assert "no grant of yours reaches a path a run can mount" in r.result["note"]
