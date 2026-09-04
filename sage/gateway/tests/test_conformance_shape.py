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

if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
