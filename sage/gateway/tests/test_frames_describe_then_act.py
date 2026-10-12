"""A beat that carries frames: the first text-only reply is the description it was asked for, not the
end of the turn (legion-being 2026-10-12: three beats ended after describing their frames)."""
from sage.gateway.being_gate_client import BeingIntent, ResultEnvelope
from sage.gateway.being_tool_loop import run_tool_turn


class _Client:
    def __init__(self):
        self.dispatched = []

    def dispatch(self, intent):
        self.dispatched.append(intent.effector)
        return ResultEnvelope(ok=True, result="ok")


def _seed(images):
    m = {"role": "user", "content": "beat"}
    if images:
        m["images"] = ["b64"]
    return [m]


def test_a_description_on_a_frame_beat_continues_once_and_then_acts():
    seen = []
    steps = iter([{"content": "Frame 1 shows the board ...", "intents": []},
                  {"content": "", "intents": [BeingIntent("memory_write", {"path": "journal.md", "content": "x"})]},
                  {"content": "done", "intents": []}])

    def gen(convo):
        seen.append(convo[-1]["content"])
        return next(steps)
    c = _Client()
    res = run_tool_turn(c, gen, _seed(True), max_steps=None)
    assert c.dispatched == ["memory_write"]
    assert res.reply == "done"
    assert any("vision line" in s for s in seen)
    assert {"step": 0, "nudge": "described_frames"} in res.interjected


def test_only_once_a_second_text_reply_still_ends_the_turn():
    steps = iter([{"content": "Frame 1 ...", "intents": []}, {"content": "nothing to do", "intents": []}])
    c = _Client()
    res = run_tool_turn(c, lambda convo: next(steps), _seed(True), max_steps=None)
    assert c.dispatched == [] and res.reply == "nothing to do"


def test_no_frames_no_change():
    steps = iter([{"content": "all done", "intents": []}])
    res = run_tool_turn(_Client(), lambda convo: next(steps), _seed(False), max_steps=None)
    assert res.reply == "all done" and not res.interjected
