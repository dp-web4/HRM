"""Hermetic: thinking is a declared per-model policy (model_configs), resolved by the
adapter, deferred to by the governed harness; the heartbeat writes operator decisions
back into the escalation notes that filed them."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.irp.adapters.model_capabilities import load_capabilities  # noqa: E402
from sage.gateway.governed_turn import is_reasoning_model  # noqa: E402
from sage.gateway.heartbeat import note_resolutions  # noqa: E402


def test_think_policy_is_per_size_not_per_caller():
    assert load_capabilities("qwen3.5:0.8b").resolve_think("qwen3.5:0.8b") is False   # 2026-03 decision, declared
    assert load_capabilities("qwen3.5:27b").resolve_think("qwen3.5:27b") is True
    assert load_capabilities("qwen3.8-distill:2b").resolve_think("qwen3.8-distill:2b") is True
    assert load_capabilities("qwen38-heretic:q3km").resolve_think("qwen38-heretic:q3km") is True
    assert load_capabilities("gemma3:12b").resolve_think("gemma3:12b") is False


def test_think_budget_declared_for_reasoning_models():
    c = load_capabilities("qwen3.8-distill:2b")
    assert c.resolve_num_predict("qwen3.8-distill:2b", True, 3000) == 6000
    assert c.resolve_num_predict("qwen3.8-distill:2b", False, 3000) == 1024


def test_num_ctx_is_a_floor_the_config_can_raise_not_lower():
    from sage.gateway.governed_turn import resolve_num_ctx
    c = load_capabilities("qwen38-heretic:q3km")
    assert c.resolve_num_ctx("qwen38-heretic:q3km", 8192) == 16384      # Modelfile value, declared per size
    assert c.resolve_num_ctx("qwen38-heretic:q3km", 32768) == 32768     # a caller asking for more keeps it
    assert c.resolve_num_ctx("qwen3.8-distill:2b", 8192) == 8192        # 2B declares nothing: floor unchanged
    assert resolve_num_ctx("qwen38-heretic:q3km", 8192) == 16384
    assert resolve_num_ctx("no-such-model:1b", 8192) == 8192


def test_num_ctx_fallback_to_the_floor_is_loud():
    """A config failure must not be a silent 8192: that is beat 46 with no record of why."""
    import io
    import contextlib
    import sage.irp.adapters.model_capabilities as mc
    from sage.gateway.governed_turn import resolve_num_ctx
    keep = mc.load_capabilities

    def broken(model):
        raise ValueError("bad json")
    mc.load_capabilities = broken
    try:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert resolve_num_ctx("qwen38-heretic:q3km", 8192) == 8192
    finally:
        mc.load_capabilities = keep
    assert "num_ctx" in err.getvalue() and "8192" in err.getvalue() and "bad json" in err.getvalue()
    assert resolve_num_ctx("qwen38-heretic:q3km", 8192) == 16384   # restored


def test_governed_harness_defers_to_the_config():
    assert is_reasoning_model("qwen3.8-distill:2b") and is_reasoning_model("qwen38-heretic:q3km")
    assert not is_reasoning_model("qwen3.5:0.8b")


def test_note_resolutions_appends_once_to_the_filing_note():
    d = Path(tempfile.mkdtemp(prefix="esc-"))
    (d / "a.md").write_text("routing: {\"scope_request\": {\"request_id\": \"scope-abc\"}}\n")
    (d / "b.md").write_text("unrelated scope-zzz\n")
    w = note_resolutions(d, [("scope-abc", "/x", "granted")], "2026-09-05 17:00 UTC", "heartbeat-1")
    assert w == ["a.md"]
    body = (d / "a.md").read_text()
    assert "## Resolved" in body and "granted" in body and "scope-abc" in body
    assert "Resolved" not in (d / "b.md").read_text()
    assert note_resolutions(d, [("scope-abc", "/x", "granted")], "later", "heartbeat-2") == []  # idempotent
    assert note_resolutions(d, [], "x", "y") == [] and note_resolutions(d / "nope", [("i", "p", "denied")], "x", "y") == []


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
