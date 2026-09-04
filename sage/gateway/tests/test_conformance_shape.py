"""The conformance runner's report contract (hermetic): fixed step list, valid statuses."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway import conformance as c  # noqa: E402

def test_step_list_is_gpts_lifecycle_in_order():
    assert c.STEPS == ["connect_prove_key", "receive_contract", "propose_act", "allow_deny_escalate",
                       "ruling_mid_session", "execute_after_authority", "record_outcome", "reconnect_replay"]

def test_step_status_is_closed_set():
    for s in ("PASS", "FAIL", "NOT_YET"):
        assert c._step("x", s, k=1)["status"] == s
    try:
        c._step("x", "MAYBE"); assert False
    except AssertionError:
        pass

def test_client_names_its_gate_path_and_single_gate_marker(tmp_path=None):
    # The report's gate_path column comes from the client, never assumed. Whatever the engine
    # here, the path is one of two names and the marker says present or absent-with-reason.
    import tempfile
    from sage.gateway.being_gate_client import BeingGateClient
    d = tempfile.mkdtemp()
    cl = BeingGateClient("shape-test", os.path.join(d, "identity.json"), d)
    assert cl.gate_path in ("single-gate", "local-law")
    st = cl.single_gate_status
    assert (cl.gate_path == "single-gate") == (st == "present")
    if cl.gate_path == "local-law":
        assert st.startswith("absent: ")

def test_under_temp_root_is_a_boundary_not_a_prefix():
    # The same separator test the gate core uses: siblings of /tmp are not scratch space.
    assert c._under_temp_root("/tmp") and c._under_temp_root("/tmp/sage-x/sage/instances/b")
    assert c._under_temp_root("/var/tmp/y")
    assert not c._under_temp_root("/tmp-other/x") and not c._under_temp_root("/var/tmpsecrets/y")
    assert not c._under_temp_root("/home/dp/ai-workspace/sage/sage/instances/b")

def test_step4_unspecified_allow_fails_and_never_names_an_absent_grant():
    # The temp-root carve-out: allow, rule "", standing [] live [] -> FAIL, and the evidence
    # must not claim a grant.
    s = c._judge_decision("allow", "", "ok", [], [])
    assert s["status"] == "FAIL" and s["evidence"]["note"].startswith("unspecified allow")
    assert s["evidence"]["explained_by"] is None and "grant" not in s["evidence"].get("note", "").split(":")[0]
    # A grant the scope status shows explains the allow (shown, not proven to cover the path).
    s = c._judge_decision("allow", "", "ok", [{"id": "scope-1"}], [])
    assert s["status"] == "PASS" and s["evidence"]["explained_by"].startswith("grant in scope status")
    # A rule explains it too; a deny that names mrh + remedy passes as before.
    assert c._judge_decision("allow", "grant.standing", "ok", [], [])["status"] == "PASS"
    d = c._judge_decision("deny", "mrh.path", "targets x outside your granted scope", [], [])
    assert d["status"] == "PASS" and d["evidence"]["remedy"].startswith("hestia_request_scope")
    assert c._judge_decision("deny", "", "", [], [])["status"] == "FAIL"

def _probe(refused, ok, wrote):
    return {"refused": refused, "ok": ok, "file_written": wrote}

def test_step6_a_landed_should_not_exist_file_is_fail_unless_the_allow_was_explained():
    refused = _probe(True, False, False)
    landed = _probe(False, True, True)
    # both refused, nothing written -> PASS
    assert c._judge_execute(True, "w1", {"in_instance": refused, "outside_workspace": refused}, False) == "PASS"
    # in-instance probe lands on an unspecified allow (the /tmp worktree case) -> FAIL
    assert c._judge_execute(True, "w1", {"in_instance": landed, "outside_workspace": refused}, False) == "FAIL"
    # same landing under an explained allow (a real grant) -> PASS
    assert c._judge_execute(True, "w1", {"in_instance": landed, "outside_workspace": refused}, True) == "PASS"
    # the outside-workspace probe must be refused by the GATE regardless of cwd or grants
    assert c._judge_execute(True, "w1", {"in_instance": refused, "outside_workspace": landed}, True) == "FAIL"
    dispatcher_caught = _probe(False, False, False)
    assert c._judge_execute(True, "w1", {"in_instance": refused, "outside_workspace": dispatcher_caught}, True) == "FAIL"
    # no witness -> FAIL
    assert c._judge_execute(False, None, {"in_instance": refused, "outside_workspace": refused}, False) == "FAIL"

if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
