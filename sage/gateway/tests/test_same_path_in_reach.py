"""A refused read names the copy the being can already read.

dp, 2026-10-23, choosing the gateway over editing the being's files: cbp-being asked at least
six times for `<ws>/hestia/plugins/_shared/hestia_governance_closure.py` (the editable tree)
while its standing grant on `<ws>/governance/**` held the same file at
`<ws>/governance/hestia/plugins/_shared/...`, and memory_read of that path worked. What it got:

- the gate's refusal: "'hestia' is not granted (granted: ...) Scope request ... is still
  pending", naming the boundary and never the copy;
- a relative `hestia/plugins/...` read: "A file DOES exist at '<ws>/hestia/plugins/...'",
  pointing it AT the editable tree;
- request_scope: filed, for a human or a peer to refuse.

Each of those surfaces now names the copy first, and request_scope files nothing.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.being_gate_client import (BeingGateClient, BeingIntent,  # noqa: E402
                                            GatewayVerdict, same_path_in_reach)
from sage.gateway.reference_f1a import ReferenceF1aDispatcher  # noqa: E402

REL = ("hestia", "plugins", "_shared", "closure.py")


def _ws():
    ws = Path(tempfile.mkdtemp(prefix="spir-")).resolve()
    edit = ws.joinpath(*REL)
    copy = ws.joinpath("governance", *REL)
    for f, text in ((edit, "editable\n"), (copy, "snapshot\n")):
        f.parent.mkdir(parents=True)
        f.write_text(text)
    home = ws / "SAGE" / "sage" / "instances" / "b"
    home.mkdir(parents=True)
    return ws, str(edit), str(copy), home


def _reference(home, reach):
    r = ReferenceF1aDispatcher.__new__(ReferenceF1aDispatcher)
    r.memory_root = home
    r._extra_roots = tuple(Path(p).resolve() for p, _ in reach)
    r._extra_reach = tuple((Path(p).resolve(), rec) for p, rec in reach)
    r._witness = lambda *a, **k: "w-1"
    return r


# -- the matcher ---------------------------------------------------------------------------

def test_the_mirror_under_a_sibling_grant_is_found():
    ws, edit, copy, _ = _ws()
    assert same_path_in_reach(edit, ((str(ws / "governance"), True),)) == [(copy, str(ws / "governance"))]


def test_only_the_mirror_shape_matches_not_any_shared_tail():
    """A loose tail match would call /elsewhere/sage/x.py a copy of SAGE/sage/x.py; request_scope
    files nothing on a match, so a wrong one would swallow a real ask."""
    ws, _, _, _ = _ws()
    (ws / "SAGE" / "sage" / "x.py").write_text("x\n")
    other = Path(tempfile.mkdtemp(prefix="spir-else-")).resolve() / "sage" / "x.py"
    other.parent.mkdir(parents=True)
    other.write_text("not the same file\n")
    assert same_path_in_reach(str(other), ((str(ws / "SAGE"), True),)) == []


def test_exact_grants_absent_copies_and_paths_already_in_reach_do_not_match():
    ws, edit, copy, _ = _ws()
    gov = str(ws / "governance")
    assert same_path_in_reach(edit, ((gov, False),)) == [], "an exact grant does not reach beneath it"
    assert same_path_in_reach(str(ws / "hestia" / "nope.py"), ((gov, True),)) == []
    assert same_path_in_reach(copy, ((gov, True),)) == [], "inside the grant is reach, not a copy"


def test_a_symlink_out_of_the_grant_is_not_a_copy_in_reach():
    ws, edit, _, _ = _ws()
    gov2 = ws / "gov2"
    gov2.mkdir()
    os.symlink(str(ws / "hestia"), str(gov2 / "hestia"))   # gov2/hestia/... -> the editable tree
    assert same_path_in_reach(edit, ((str(gov2), True),)) == []


def test_two_grants_holding_it_are_both_named_whatever_order_they_arrive_in():
    ws, edit, copy, _ = _ws()
    mirror = ws.joinpath("mirror", *REL)
    mirror.parent.mkdir(parents=True)
    mirror.write_text("mirror\n")
    a, b = (str(ws / "governance"), True), (str(ws / "mirror"), True)
    assert same_path_in_reach(edit, (a, b)) == same_path_in_reach(edit, (b, a)) == sorted(
        [(copy, a[0]), (str(mirror), b[0])])


# -- the three surfaces ----------------------------------------------------------------------

def test_the_gates_refusal_of_the_editable_path_names_the_copy_first():
    ws, edit, copy, home = _ws()
    c = BeingGateClient.__new__(BeingGateClient)
    c._dispatcher = None
    c.gate = lambda intent: GatewayVerdict("deny", "mrh.path", stage="law",
                                           reason="'hestia' is not granted (granted: governance)")
    c._policy_grants = lambda: ((str(ws / "governance"),), ((str(ws / "governance"), True),))
    env = c._dispatch_resolved(BeingIntent("memory_read", {"path": edit}))
    assert env.refused
    assert env.error.startswith(f"You can already read this: the same path exists at {copy}"), env.error
    assert "do not request_scope" in env.error and "'hestia' is not granted" in env.error
    # a refused write is not answered with a read-only copy
    env = c._dispatch_resolved(BeingIntent("memory_write", {"path": edit, "content": "x"}))
    assert copy not in env.error


def test_the_dispatchers_out_of_reach_refusal_names_the_copy_first():
    ws, edit, copy, home = _ws()
    r = _reference(home, ((str(ws / "governance"), True),))
    try:
        r._safe_path(edit)
        raise AssertionError("the editable path must stay out of reach")
    except ValueError as e:
        msg = str(e)
    assert msg.index(copy) < msg.index("outside your reach. You can read"), msg
    assert "real boundary" not in msg, "the boundary text is what sent the being to request_scope"


def test_a_relative_read_points_at_the_copy_not_the_editable_tree():
    ws, edit, copy, home = _ws()
    r = _reference(home, ((str(ws / "governance"), True),))
    env = r._do_memory_read(BeingIntent("memory_read", {"path": "/".join(REL)}))
    assert env.ok and copy in env.result and edit not in env.result, env.result
    # with no copy in reach, the pointer to the real file stands, as before
    r = _reference(home, ())
    env = r._do_memory_read(BeingIntent("memory_read", {"path": "/".join(REL)}))
    assert edit in env.result, env.result


def test_request_scope_for_a_path_with_a_copy_in_reach_files_nothing():
    from sage.gateway.tests.test_hestia_dispatch import FakeMcp, _disp
    ws, edit, copy, home = _ws()
    d, _ = _disp()
    FakeMcp.calls.clear()
    gov = str(ws / "governance")
    v = GatewayVerdict("allow", granted=(gov,), granted_reach=((gov, True),))
    env = d(BeingIntent("request_scope", {"path": edit, "reason": "to read the closure table"}), v)
    assert env.ok and env.result["status"] == "same_path_in_reach", env
    assert env.result["in_reach"] == [copy] and "No request was filed" in env.result["next"]
    assert not [n for n, _ in FakeMcp.calls if n == "hestia_request_scope"]
    # a path with no copy is still filed
    env = d(BeingIntent("request_scope", {"path": str(ws / "hestia" / "other.py"), "reason": "to read it"}), v)
    assert [n for n, _ in FakeMcp.calls if n == "hestia_request_scope"], env
