"""A prompt the window cannot hold is never sent, and a server cut is not a measurement.

Sprout, 2026-09-06..10-15: about 16 explore beats got an HTTP 400 instead of a reply. Compaction
cannot touch the system prompt or the first user turn (~14k tokens there, against a 10,240 target
at num_ctx 16,384), so the loop grew past the window; Ollama dropped messages off the front, the
only user turn with them, and the qwen template raised "No user query found in messages". The 400
text became the beat's reply. Hermetic: a fake LLM and a recording client, no model, no gate."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.being_gate_client import ResultEnvelope  # noqa: E402
from sage.gateway.being_tool_loop import (run_ollama_tool_turn, WINDOW_SEND_MARGIN,  # noqa: E402
                                          SERVER_CUT_RATIO, PREFLIGHT_CPT, _UNCOUNTED_CHARS)

NO_USER_400 = ("[OllamaIRP: Connection error: HTTP Error 400: Bad Request — {\"error\":\"Unable to generate "
               "parser for this template. ... raise_exception('No user query found in messages.') ...\"}]")


class _Client:
    """Dispatches every call and returns a fixed-size result (under the elision floor, so
    compaction cannot free room and the window arithmetic in each test is exact)."""
    def __init__(self, result_chars=480):
        self.calls = []
        self.result = "x" * result_chars

    def dispatch(self, intent):
        self.calls.append(intent.effector)
        return ResultEnvelope(ok=True, result=self.result)


def _call(n):
    return [{"function": {"name": "witness", "arguments": {"event": f"step {n}"}}}]


class _LLM:
    """Scripted replies; records every list it was actually sent."""
    num_ctx = 16384

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def get_chat_response(self, messages, tools=None, **kw):
        self.sent.append(list(messages))
        return self.replies.pop(0)


def test_a_prompt_over_the_window_is_not_sent_and_the_turn_ends_with_no_words():
    # step 1 measures 16,100 of 16,384; the next step adds a tool result, which puts the
    # anchored estimate past num_ctx - WINDOW_SEND_MARGIN, so step 2 is never sent.
    llm = _LLM([{"content": "", "tool_calls": _call(1), "raw": {"prompt_eval_count": 16100, "eval_count": 20}}])
    r = run_ollama_tool_turn(_Client(), llm, [{"role": "user", "content": "go"}], max_steps=4)
    assert len(llm.sent) == 1, "the over-window prompt must not reach the server"
    assert r.window_full and r.window_full["cause"] == "over_window"
    assert r.window_full["estimated_prompt"] > llm.num_ctx - WINDOW_SEND_MARGIN
    assert r.reply == "", "nothing is said in the being's name"
    assert r.generates[-1]["window_full"] == "over_window"
    assert len(r.thinking) == len(r.generates), "one thinking entry per generate, as before"


def test_an_oversized_first_step_is_refused_on_the_preflight_estimate():
    # No server count yet. A 60,000-char seed at num_ctx 16,384 is ~25k tokens on the preflight
    # estimate: sent, Ollama would cut the front and the only user turn with it. It is not sent.
    llm = _LLM([{"content": "should not be reached", "tool_calls": []}])
    seed = [{"role": "system", "content": "s" * 60000}, {"role": "user", "content": "go"}]
    r = run_ollama_tool_turn(_Client(), llm, seed, max_steps=2, tools=[])
    assert llm.sent == [], "the oversized first prompt must not reach the server"
    assert r.window_full and r.window_full["cause"] == "preflight" and r.window_full["step"] == 0
    assert r.window_full["estimated_prompt"] > llm.num_ctx - WINDOW_SEND_MARGIN
    assert r.reply == ""


def test_a_seed_the_size_of_a_real_fitted_one_is_still_sent():
    # Sprout's real step-1 prompts measured up to 14,306 tokens at num_ctx 16,384: at the ~3.14
    # chars/token measured on its seeds that is ~45k chars, ~32k of them in messages. The
    # preflight puts that at ~15.5k, under the line, so a beat the heartbeat fitter accepted runs.
    llm = _LLM([{"content": "done", "tool_calls": [], "raw": {"prompt_eval_count": 14306, "eval_count": 5}}])
    seed = [{"role": "system", "content": "s" * 31950}, {"role": "user", "content": "go"}]
    est = (sum(len(m["content"]) for m in seed) + _UNCOUNTED_CHARS) / PREFLIGHT_CPT
    assert est < llm.num_ctx - WINDOW_SEND_MARGIN
    r = run_ollama_tool_turn(_Client(), llm, seed, max_steps=2, tools=[])
    assert len(llm.sent) == 1 and r.reply == "done" and r.window_full is None


def test_the_preflight_errs_high_against_every_ratio_measured_on_this_being():
    # 3.026..3.152 chars/token over 60 beats: an estimate at PREFLIGHT_CPT over-counts them all.
    assert PREFLIGHT_CPT < 3.026


def test_no_user_query_400_is_not_retried_and_never_becomes_the_reply():
    llm = _LLM([{"content": "", "tool_calls": _call(1), "raw": {"prompt_eval_count": 9000, "eval_count": 20}},
                {"content": NO_USER_400, "tool_calls": []}])
    r = run_ollama_tool_turn(_Client(), llm, [{"role": "user", "content": "go"}], max_steps=4)
    assert len(llm.sent) == 2, "the same list cannot succeed again: no transport retry"
    assert r.reply == "" and "OllamaIRP" not in r.reply
    assert r.window_full and r.window_full["cause"] == "server_cut"


def test_other_transport_errors_are_still_retried_once():
    llm = _LLM([{"content": "[OllamaIRP: Connection error: HTTP Error 500: invalid tool call arguments]",
                 "tool_calls": []},
                {"content": "done", "tool_calls": [], "raw": {"prompt_eval_count": 100, "eval_count": 5}}])
    r = run_ollama_tool_turn(_Client(), llm, [{"role": "user", "content": "go"}], max_steps=2, tools=[])
    assert len(llm.sent) == 2 and r.reply == "done" and r.window_full is None


def test_a_count_far_below_what_was_sent_is_not_stored_as_the_measurement():
    # Step 2's count (8,172) is what survived a server cut of a ~16k list. Believed, it would
    # anchor step 3 at ~8.4k and step 3 would be sent; kept out, step 3 is judged on the real
    # anchor (15,800) and is not sent.
    llm = _LLM([{"content": "", "tool_calls": _call(1), "raw": {"prompt_eval_count": 15800, "eval_count": 20}},
                {"content": "", "tool_calls": _call(2), "raw": {"prompt_eval_count": 8172, "eval_count": 20}},
                {"content": "should not be reached", "tool_calls": [], "raw": {"prompt_eval_count": 8400}}])
    # window_warn_at=0.99: on the Legion carrier, step 1's 96% would add the window-filling notice
    # (~150 tokens), and this fixture sits one token under the send margin; the notice is not
    # what is under test here, the anchor is.
    r = run_ollama_tool_turn(_Client(), llm, [{"role": "user", "content": "go"}], max_steps=5,
                             window_warn_at=0.99)
    cut = r.generates[1].get("server_cut")
    assert cut and cut["counted"] == 8172 and cut["counted"] < SERVER_CUT_RATIO * cut["expected"]
    assert len(llm.sent) == 2, "anchored on the real count, the third prompt does not fit"
    assert r.window_full and r.window_full["cause"] == "over_window"


def test_the_beat_record_says_the_window_filled_and_that_nothing_the_being_did_caused_it():
    from types import SimpleNamespace
    from sage.gateway import heartbeat as H
    res = SimpleNamespace(trace=[], reply="", window_full={"cause": "over_window"})
    text = H._beat_record_text(res)
    assert "context window filled" in text and "Nothing you did caused it" in text
    assert H._beat_record_text(SimpleNamespace(trace=[], reply="", window_full=None)) == \
        "You called no tools this beat."
    carried = H._carry([], res)
    assert carried[-1]["role"] == "assistant" and "context window filled" in carried[-1]["content"]
