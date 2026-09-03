"""Hermetic tests for the being tool-use loop. No live gate/model: the gate client
uses an injected fake law + mock dispatcher (F1a stand-in), and `generate` is scripted.
Runnable under pytest or directly."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.being_gate_client import BeingGateClient, BeingIntent, ResultEnvelope  # noqa: E402
from sage.gateway.being_tool_loop import run_tool_turn  # noqa: E402


def _client(dispatcher):
    """Client whose local law + society both ALLOW, with an injected F1a dispatcher."""
    c = BeingGateClient.__new__(BeingGateClient)
    c.member_id = "test-being"
    c.workspace = "/tmp/ws"
    c._import_error = ""
    c._profile = object()
    c._dispatcher = dispatcher
    c._mech = SimpleNamespace(query_society_safety=lambda raw: SimpleNamespace(decision="allow"))
    c._core = SimpleNamespace(
        NormalizedEvent=lambda **kw: SimpleNamespace(raw=kw.get("raw", {})),
        evaluate=lambda ev, prof, ws, policy=None: SimpleNamespace(
            decision="allow", rule="", reason="ok", innate=False),
    )
    return c


def _scripted(*outs):
    """A generate() that returns each scripted output in turn, recording what it saw."""
    calls = {"seen": []}
    seq = list(outs)

    def generate(messages):
        calls["seen"].append(list(messages))
        return seq.pop(0) if seq else {"content": "(nothing more)", "intents": []}
    return generate, calls


OK_DISPATCH = lambda intent, v: ResultEnvelope(ok=True, result=f"result[{intent.effector}]", witness_id="w1")


def test_dispatches_then_speaks():
    gen, calls = _scripted(
        {"content": "let me check", "intents": [BeingIntent("witness", {"event": "x"})]},
        {"content": "done — here is my reply", "intents": []},
    )
    r = run_tool_turn(_client(OK_DISPATCH), gen, [{"role": "user", "content": "hi"}])
    assert r.reply == "done — here is my reply", r
    assert r.steps == 1 and not r.capped
    assert len(r.trace) == 1 and r.trace[0][1].ok and r.acted


def test_result_is_reinjected_before_second_generate():
    gen, calls = _scripted(
        {"content": "checking", "intents": [BeingIntent("witness", {})]},
        {"content": "final", "intents": []},
    )
    run_tool_turn(_client(OK_DISPATCH), gen, [{"role": "user", "content": "hi"}])
    # the 2nd generate must have seen a tool message carrying the dispatched result
    second_call_msgs = calls["seen"][1]
    assert any(m.get("role") == "tool" and "result[witness]" in m.get("content", "")
               for m in second_call_msgs), second_call_msgs


def test_cap_forces_a_spoken_close():
    # a being that never stops reaching for tools must still end its turn in language
    gen, _ = _scripted(
        {"content": "t1", "intents": [BeingIntent("witness", {})]},
        {"content": "t2", "intents": [BeingIntent("witness", {})]},
        {"content": "forced close", "intents": [BeingIntent("witness", {})]},
    )
    r = run_tool_turn(_client(OK_DISPATCH), gen, [{"role": "user", "content": "hi"}], max_steps=2)
    assert r.capped and r.steps == 2
    assert r.reply == "forced close"


def test_refused_intent_recorded_and_loop_continues():
    # an out-of-registry effector is refused BEFORE the gate; the loop keeps going
    gen, _ = _scripted(
        {"content": "trying shell", "intents": [BeingIntent("shell", {"command": "rm -rf /"})]},
        {"content": "ok, i cannot do that — here's words instead", "intents": []},
    )
    r = run_tool_turn(_client(OK_DISPATCH), gen, [{"role": "user", "content": "hi"}])
    assert len(r.refused) == 1 and r.refused[0][1].refused
    assert r.reply.startswith("ok, i cannot")
    assert not r.acted  # nothing actually executed


def test_pending_f1a_when_no_dispatcher():
    # with no F1a dispatcher, an allowed intent comes back pending — never fabricated
    gen, _ = _scripted(
        {"content": "checking", "intents": [BeingIntent("witness", {})]},
        {"content": "done", "intents": []},
    )
    r = run_tool_turn(_client(None), gen, [{"role": "user", "content": "hi"}])
    assert r.trace[0][1].pending and not r.trace[0][1].ok
    assert "not yet executed" in r.trace[0][1].to_tool_message()


def test_run_ollama_tool_turn_with_fake_llm():
    from sage.gateway.being_gate_client import ollama_tools
    from sage.gateway.being_tool_loop import run_ollama_tool_turn
    # exactly the bounded registry, nothing more: §7.2's 5 verbs + memory split r/w (6),
    # + pr_review (2026-09-03, Legion: the being reviews a PR; the seat posts, gated as
    # the `gh` command it runs). Widening this number is a registry decision, not a typo.
    assert len(ollama_tools()) == 7

    calls = {"n": 0}

    class FakeLLM:
        def get_chat_response(self, messages, tools=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"content": "checking",
                        "tool_calls": [{"function": {"name": "witness", "arguments": {"event": "x"}}}]}
            return {"content": "done", "tool_calls": []}

    r = run_ollama_tool_turn(_client(OK_DISPATCH), FakeLLM(), [{"role": "user", "content": "hi"}])
    assert r.reply == "done"
    assert r.trace and r.trace[0][0].effector == "witness" and r.trace[0][1].ok


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
