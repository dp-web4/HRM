"""The fixed prompt can overflow the window on its own; fit_to_window cannot help then.

Legion 2026-09-08 15:42Z-19:22Z: config.context_overcommitted True on eight beats running
(headroom -2.4k..-4k tokens), digest and recall already at their floors, the conversations
block 30.9k chars. Every explore generate: prompt_eval + eval == num_ctx, done_reason
length, zero tool calls. The instrument reported it every beat; nothing acted on it."""
from sage.gateway.heartbeat import fit_state, window_budget_chars, CONV_LADDER, CPT


def _build_factory(sizes):
    """build(per_conv, turn_chars) -> text whose length is sizes[rung]"""
    calls = []

    def build(per_conv, turn_chars):
        calls.append((per_conv, turn_chars))
        return "c" * sizes[(per_conv, turn_chars)]
    return build, calls


def test_budget_is_one_producer():
    assert window_budget_chars(24576, 8000) == int((24576 - 6144 - 512) * CPT)
    assert window_budget_chars(24576, 3000) == int((24576 - 3000 - 512) * CPT)   # reserve never exceeds num_predict
    assert window_budget_chars(1000, 8000) == 0


def test_full_display_when_it_fits():
    sizes = {r: 10_000 for r in CONV_LADDER}
    build, calls = _build_factory(sizes)
    text, rung, iv = fit_state(build, num_ctx=24576, num_predict=8000, other_chars=20_000)
    assert rung == CONV_LADDER[0] and iv is None and calls == [CONV_LADDER[0]]


def test_steps_down_until_it_fits_and_says_what_it_suppressed():
    """The measured case: 30.9k of conversations, 38k of everything else, 60.9k budget."""
    budget = window_budget_chars(24576, 8000)
    sizes = {CONV_LADDER[0]: 30_900, CONV_LADDER[1]: 24_000, CONV_LADDER[2]: 14_000,
             CONV_LADDER[3]: 9_000, CONV_LADDER[4]: 6_000}
    build, calls = _build_factory(sizes)
    text, rung, iv = fit_state(build, num_ctx=24576, num_predict=8000, other_chars=38_000)
    assert 38_000 + len(text) <= budget
    assert rung == CONV_LADDER[2]                        # first rung that fits, not the sparsest
    assert calls == list(CONV_LADDER[:3])
    assert iv["kind"] == "context_fit" and iv["block"] == "conversations"
    assert "16900 chars of conversations" in iv["suppressed"] and "last 6 turns" in iv["suppressed"]
    assert "fits now" in iv["reason"] and "68900 chars" in iv["reason"]


def test_sparsest_rung_is_used_and_named_when_nothing_fits():
    sizes = {r: 50_000 for r in CONV_LADDER}
    build, calls = _build_factory(sizes)
    text, rung, iv = fit_state(build, num_ctx=24576, num_predict=8000, other_chars=38_000)
    assert rung == CONV_LADDER[-1] and calls == list(CONV_LADDER)
    assert "STILL does not fit" in iv["reason"]


def test_unknown_window_means_full_display():
    build, calls = _build_factory({CONV_LADDER[0]: 1})
    text, rung, iv = fit_state(build, num_ctx=None, num_predict=8000, other_chars=0)
    assert rung == CONV_LADDER[0] and iv is None



# -- the next-wake check must not be false by construction -----------------------------
def test_a_running_beat_does_not_read_as_an_unarmed_timer():
    """2026-09-09T15:07Z: the end-of-beat check read `monotonic=infinity` and wrote
    "NOTHING WILL WAKE THE BEING" into the record of a beat whose timer armed correctly
    seconds later. An OnUnitInactiveSec timer CANNOT have a next elapse while the unit it
    watches is running — and this check runs from inside that unit."""
    from sage.gateway.heartbeat import interpret_timer_state

    running = ("NextElapseUSecRealtime=\n"
               "NextElapseUSecMonotonic=infinity\n"
               "LoadState=loaded\nActiveState=active\n")
    armed, why = interpret_timer_state(running)
    assert armed is True, why
    assert "correct while this beat is still running" in why

    scheduled = ("NextElapseUSecRealtime=Wed 2026-09-09 09:03:39 PDT\n"
                 "NextElapseUSecMonotonic=infinity\nLoadState=loaded\nActiveState=active\n")
    armed, why = interpret_timer_state(scheduled)
    assert armed is True and why.startswith("scheduled:")

    # the real failure this exists for: the timer is gone or dead, not merely unscheduled
    for bad in ("NextElapseUSecRealtime=\nNextElapseUSecMonotonic=infinity\n"
                "LoadState=not-found\nActiveState=inactive\n",
                "NextElapseUSecRealtime=\nNextElapseUSecMonotonic=infinity\n"
                "LoadState=loaded\nActiveState=failed\n",
                "NextElapseUSecRealtime=\nNextElapseUSecMonotonic=infinity\n"
                "LoadState=loaded\nActiveState=inactive\n"):
        armed, why = interpret_timer_state(bad)
        assert armed is False, why
        assert "not healthy" in why

def test_fill_headroom_uses_last_not_max(tmp_path):
    """A historical spike must not cap headroom forever.

    _fill_headroom reads prompt_tokens_max from an append-only file that is
    never pruned (heartbeat.py, ~line 315 at cc64c838c). max() over it means a
    single large beat — e.g. a digest spike or a long conversation tail — keeps
    the field high for every later beat until the daemon restarts, and the
    acting guard then reports context_overcommitted on beats whose real prompt
    is far smaller than that old maximum. Observed on this machine (legion-
    gemma3-12b): config.context_overcommitted True on eight consecutive beats,
    headroom -2.4k..-4k tokens, while the fixed prompt fit with room to spare.

    The field feeds the acting guard's overcommit decision for THIS beat; what
    matters is how large the last measured prompt was, not the largest one ever
    recorded. A transient spike should stop counting once a smaller beat has
    followed it."""
    from sage.gateway.heartbeat import _fill_headroom

    # spike then recovery: max() says 9000 (the old spike); last says 3200
    f = tmp_path / "prompt_tokens_max.txt"
    f.write_text("1500\n9000\n3200\n")
    assert _fill_headroom(f) == 3200

    # single value: both semantics agree, so the fix changes nothing there
    f = tmp_path / "prompt_tokens_max.txt"
    f.write_text("4100\n")
    assert _fill_headroom(f) == 4100

    # empty file: no measurement yet, stays None (guard treats it as unknown)
    f = tmp_path / "prompt_tokens_max.txt"
    f.write_text("")
    assert _fill_headroom(f) is None
"""The fixed prompt can overflow the window on its own; fit_to_window cannot help then.

Legion 2026-09-08 15:42Z-19:22Z: config.context_overcommitted True on eight beats running
(headroom -2.4k..-4k tokens), digest and recall already at their floors, the conversations
block 30.9k chars. Every explore generate: prompt_eval + eval == num_ctx, done_reason
length, zero tool calls. The instrument reported it every beat; nothing acted on it."""
from sage.gateway.heartbeat import fit_state, window_budget_chars, CONV_LADDER, CPT


def _build_factory(sizes):
    """build(per_conv, turn_chars) -> text whose length is sizes[rung]"""
    calls = []

    def build(per_conv, turn_chars):
        calls.append((per_conv, turn_chars))
        return "c" * sizes[(per_conv, turn_chars)]
    return build, calls


def test_budget_is_one_producer():
    assert window_budget_chars(24576, 8000) == int((24576 - 6144 - 512) * CPT)
    assert window_budget_chars(24576, 3000) == int((24576 - 3000 - 512) * CPT)   # reserve never exceeds num_predict
    assert window_budget_chars(1000, 8000) == 0


def test_full_display_when_it_fits():
    sizes = {r: 10_000 for r in CONV_LADDER}
    build, calls = _build_factory(sizes)
    text, rung, iv = fit_state(build, num_ctx=24576, num_predict=8000, other_chars=0)
    assert len(text) == 10_000 and rung == CONV_LADDER[0] and iv is None


def test_climbs_to_first_fitting_rung():
    sizes = {CONV_LADDER[0]: 90_000, CONV_LADDER[1]: 40_000, CONV_LADDER[2]: 3_000}
    build, calls = _build_factory(sizes)
    text, rung, iv = fit_state(build, num_ctx=16384, num_predict=8000, other_chars=0)
    assert rung == CONV_LADDER[2] and len(text) == 3_000


def test_investigation_verdict_when_nothing_fits():
    build, calls = _build_factory({CONV_LADDER[0]: 90_000})
    text, rung, iv = fit_state(build, num_ctx=4096, num_predict=8000, other_chars=0)
    assert text == "" and rung is None
    assert "context_overcommitted" in iv


def test_none_num_ctx_is_the_floor_not_an_error():
    build, calls = _build_factory({CONV_LADDER[0]: 1})
    text, rung, iv = fit_state(build, num_ctx=None, num_predict=8000, other_chars=0)
    assert rung == CONV_LADDER[0] and iv is None



# -- the next-wake check must not be false by construction -----------------------------
def test_a_running_beat_does_not_read_as_an_unarmed_timer():
    """A beat that is still running reads NextElapseUSecRealtime as empty (infinity):
    an inactivity timer cannot compute a next elapse until its unit goes inactive, and
    this check runs from inside that unit."""
    from sage.gateway.heartbeat import interpret_timer_state

    running = ("NextElapseUSecRealtime=\n"
               "NextElapseUSecMonotonic=infinity\n"
               "LoadState=loaded\nActiveState=active\n")
    armed, why = interpret_timer_state(running)
    assert armed is True and why.startswith("scheduled:")

    # the real failure this exists for: the timer is gone or dead, not merely unscheduled
    for bad in ("NextElapseUSecRealtime=\nNextElapseUSecMonotonic=infinity\n"
                "LoadState=not-found\nActiveState=inactive\n",
               "NextElapseUSecRealtime=\nNextElapseUSecMonotonic=infinity\n"
               "LoadState=loaded\nActiveState=failed\n"):
        armed, why = interpret_timer_state(bad)
        assert armed is False, why
        assert "not healthy" in why


def test_fill_headroom_is_beat_scoped(tmp_path):
    """_fill_headroom must report the largest prompt sent THIS BEAT only.

    The docstring at heartbeat.py:302-314 (tree cc64c838c) names the regression this
    guards against: the first cut reported the worst of ~500 historical generates as if
    it were this beat's, and a stale max then held headroom_tokens negative for every
    later beat until daemon restart. The loop (heartbeat.py:315-324) skips every line
    whose host_session_id is not this beat's; the fixture below discriminates exactly
    that filter — with it removed, best would be 9000 and both asserts fail."""
    from sage.gateway.heartbeat import _fill_headroom

    partial = tmp_path / "heartbeat.partial.jsonl"
    partial.write_text(
        '{"host_session_id": "beat-OLD", "prompt_eval_count": 9000}\n'
        '{"host_session_id": "beat-NOW", "prompt_eval_count": 7000}\n'
        '{"host_session_id": "beat-NOW", "prompt_eval_count": 5200}\n')
    cfg = {"num_ctx": 13_000}
    out = _fill_headroom(cfg, partial, "beat-NOW")
    assert out is cfg                      # mutates and returns the same dict
    assert cfg["prompt_tokens_max"] == 7000      # max WITHIN this beat: not last (5200), not cross-beat (9000)
    assert cfg["headroom_tokens"] == 13_000 - 7000 - 640   # _ANSWER_RESERVE, heartbeat.py:328-330
    assert cfg["context_overcommitted"] is False


def test_fill_headroom_flags_overcommit(tmp_path):
    """Same beat-scoped read; when the largest prompt of this beat leaves no room for
    the answer reserve, context_overcommitted must be True — the signal that acted on
    nothing in the 2026-09-08 incident."""
    from sage.gateway.heartbeat import _fill_headroom

    partial = tmp_path / "heartbeat.partial.jsonl"
    partial.write_text(
        '{"host_session_id": "beat-NOW", "prompt_eval_count": 12_500}\n'
        '{"host_session_id": "beat-OLD", "prompt_eval_count": 9000}\n')
    cfg = {"num_ctx": 13_000}
    _fill_headroom(cfg, partial, "beat-NOW")
    assert cfg["headroom_tokens"] == -140
    assert cfg["context_overcommitted"] is True


def test_fill_headroom_empty_partial_stays_none(tmp_path):
    """No generate yet this beat: the field stays None (the guard treats it as unknown)."""
    from sage.gateway.heartbeat import _fill_headroom

    partial = tmp_path / "heartbeat.partial.jsonl"
    partial.write_text("")
    cfg = {"num_ctx": 13_000}
    out = _fill_headroom(cfg, partial, "beat-NOW")
    assert out is cfg and cfg["prompt_tokens_max"] is None
