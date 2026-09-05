"""heartbeat._carry: what the next turn is told about a turn that said nothing."""
import os
import sys
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sage.gateway.heartbeat import _carry  # noqa: E402


def _res(reply, trace):
    return SimpleNamespace(reply=reply, trace=trace)


def test_carry_says_what_happened_instead_of_a_placeholder_the_being_cannot_read():
    intent = SimpleNamespace(effector="witness", args={"event": "x"})
    env = SimpleNamespace(ok=True, refused=False, error=None)
    seed = [{"role": "user", "content": "hi"}]
    out = _carry(seed, _res("", [(intent, env), (intent, env)]))
    assert out[-2]["role"] == "user" and out[-2]["content"].startswith("Record of what you did")
    assert out[-1] == {"role": "assistant", "content": "(made 2 tool calls, then said nothing)"}
    out = _carry(seed, _res("", []))
    assert out[-1] == {"role": "assistant", "content": "(said nothing and called no tool this turn)"}
    out = _carry(seed, _res("I am here.", []))
    assert out[-1] == {"role": "assistant", "content": "I am here."}
