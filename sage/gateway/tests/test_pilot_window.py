import hashlib
import json

import pytest

from sage.gateway.pilot_window import PHASES, summarize

START = "2026-10-15T20:05:39Z"


def raw(ts=START, **kw):
    return json.dumps({"schema": "heartbeat/v2", "ts": ts,
                       **dict.fromkeys(PHASES), **kw}).encode() + b"\r\n"


def report(*rows, **kw):
    return summarize(rows, first_beat=START, paths=["scratch/fixture.md"],
                     conversation="private-conversation", sequence=17, **kw)


def test_window_ends_without_consuming_a_third_beat():
    first = raw(elapsed_s=113.5, selected={"turn": "private-conversation:17", "expects_reply": False},
                conversations_marked={"marked": {"private-conversation": 17}},
                reflect={"trace": [{"effector": "say", "ok": True, "args": {
                    "to": "private-conversation", "text": "I will edit scratch/fixture.md"}}]})
    second = raw("2026-10-15T20:35:25Z", selected={"turn": "other:22"})
    out = report(first, second, b"not even json")
    assert out["window_complete"] and out["observed_beats"] == 2
    a, b = out["beats"]
    assert a["request_selected"] and not b["request_selected"]
    assert a["request_marked_seen"] and a["selected_expects_reply"] is False
    assert a["reported_end"] == "2026-10-15T20:07:32.500000Z"
    assert a["record_sha256"] == hashlib.sha256(first).hexdigest()
    assert a["phases"]["reflect"]["say_mentions_fixture"] == 1
    assert a["phases"]["reflect"]["fixture_edits"] == 0
    assert "private-conversation" not in json.dumps(out) and "I will edit" not in json.dumps(out)


def test_exact_path_counts_attempts_not_success_and_never_opens_path(tmp_path):
    secret = tmp_path / "keep"
    secret.write_text("unchanged")
    trace = [{"effector": "memory_read", "args": {"path": "scratch/fixture.md"}, "ok": False},
             {"effector": "memory_edit", "args": {"path": "scratch/fixture.md"}, "ok": False},
             {"effector": "memory_edit", "args": {"path": str(secret)}, "ok": True}]
    out = report(raw(explore={"trace": trace}), count=1)["beats"][0]["phases"]["explore"]
    assert (out["fixture_reads"], out["fixture_edits"], out["other_receipts"]) == (1, 1, 1)
    assert secret.read_text() == "unchanged"


def test_short_absent_and_malformed_evidence_not_zero_success():
    assert not report(raw())["window_complete"]
    assert report(raw("2026-10-15T19:00:00Z"))["observed_beats"] == 0
    row = json.loads(raw(explore={"trace": [None]}, answer={"unexpected": []}))
    del row["posture"]
    b = report(json.dumps(row).encode(), count=1)["beats"][0]
    assert b["phases"]["posture"]["record"] == "missing"
    assert b["phases"]["answer"]["record"] == "no_valid_trace"
    assert b["phases"]["explore"]["malformed_receipts"] == 1
    assert b["request_marked_seen"] is None


@pytest.mark.parametrize("second", [START, "2026-10-15T19:00:00Z"])
def test_duplicate_or_backward_window_rejected(second):
    with pytest.raises(ValueError):
        report(raw(), raw(second))
