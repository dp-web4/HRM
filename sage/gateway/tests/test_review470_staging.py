"""Review reproductions for PR470 at 1d0b5a02; synthetic files, no subprocess/gate calls.

These assertions pin observed defects, not desired acceptance behavior. Invert
them when converting the cases to production regression tests.
"""
import builtins
import os
import types
from pathlib import Path

from sage.gateway import being_gate_client as B
from sage.gateway.hestia_dispatch import HestiaF1aDispatcher as D


def setup(tmp_path, monkeypatch):
    home = tmp_path / "workspace" / "project" / "home"
    (home / "scratch").mkdir(parents=True)
    (home / "scratch" / "probe.py").write_text("print('synthetic')\n")
    monkeypatch.setattr(B, "sandbox_available", lambda: True)
    monkeypatch.setattr(B, "STAGE_ROOT", str(tmp_path / "stage"))
    d = D.__new__(D)
    d.memory_root, d.member = str(home), "review-being"
    d.worktree, d.workspace = str(home), str(tmp_path / "workspace" / "project")
    d._call = lambda name, args: {"actionId": "synthetic-action"}
    d._verdict = types.SimpleNamespace(command=None, granted=(str(home),),
                                       granted_reach=((str(home), True),))
    return d, home, Path(str(tmp_path / "stage") + "-review-being")


def test_data_source_absent_from_normalized_paths_and_staged_outside_grants(tmp_path, monkeypatch):
    d, home, stage = setup(tmp_path, monkeypatch)
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
    assert ev.paths == [] and str(external) not in ev.command
    d._verdict.command = ev.command
    seen = []

    def execute(*args, **kwargs):
        seen.append((stage / "sample.txt").read_text())
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", execute)
    result = d._do_run(intent)
    assert result.ok and seen == ["synthetic outside-grant bytes"]


def test_source_swap_after_validation_is_followed_during_staging(tmp_path, monkeypatch):
    d, home, stage = setup(tmp_path, monkeypatch)
    source = home / "scratch" / "data.txt"
    source.write_text("initial safe bytes")
    outside = tmp_path / "outside.txt"
    outside.write_text("synthetic out-of-reach bytes")
    ordinary_open = builtins.open

    def racing_open(path, mode="r", *args, **kwargs):
        if str(path) == str(source) and mode == "rb":
            replacement = home / "scratch" / "replacement"
            replacement.symlink_to(outside)
            os.replace(replacement, source)
        return ordinary_open(path, mode, *args, **kwargs)

    seen = []

    def execute(*args, **kwargs):
        seen.append((stage / "data.txt").read_text())
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(builtins, "open", racing_open)
    monkeypatch.setattr("subprocess.run", execute)
    result = d._do_run(B.BeingIntent("run", {"path": "scratch/probe.py", "data": ["scratch/data.txt"]}))
    assert result.ok and seen == ["synthetic out-of-reach bytes"]


def test_overlapping_run_removes_the_first_runs_staging(tmp_path, monkeypatch):
    d, home, stage = setup(tmp_path, monkeypatch)
    (home / "scratch" / "second.py").write_text("print('second')\n")
    active = False
    observed = []

    def execute(*args, **kwargs):
        nonlocal active
        if not active:
            active = True
            assert (stage / "probe.py").exists()
            inner = d._do_run(B.BeingIntent("run", {"path": "scratch/second.py"}))
            assert inner.ok
            observed.append(stage.exists())
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", execute)
    result = d._do_run(B.BeingIntent("run", {"path": "scratch/probe.py"}))
    assert result.ok and observed == [False]
