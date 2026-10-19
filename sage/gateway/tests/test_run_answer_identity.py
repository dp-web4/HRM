"""Forward answer matching: one contract for wake decisions and visible receipts.

Synthetic conversations only. No program, live notification or model is run.
Historical seat queue closure is intentionally outside this repair (#464/#478).
"""
import hashlib
import importlib.util
from pathlib import Path

import pytest

from sage.gateway import conversations as conv
from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict
from sage.gateway.hestia_dispatch import HestiaF1aDispatcher
from sage.gateway.tests.test_hestia_dispatch import FakeMcp

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seat_run_requests.py"
_spec = importlib.util.spec_from_file_location("seat_run_match_writer", _SCRIPT)
srr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srr)

BEING, SEAT, CID = "test-being", "test-seat", "test-channel"
BODY = "print('fixture')\n"
SHA = hashlib.sha256(BODY.encode()).hexdigest()[:12]
PATH = "notes/sample.py"


def request(extra=()):
    return f"[request_run] {PATH}\nwhy: check\n(sha256:{SHA})\n" + "\n".join(extra)


def receipt(argv=(), sha=SHA, path=PATH, where="", verdict="exit code 0"):
    return (f"[request_run] I ran {path} (sha {sha}) {srr.ran_line(list(argv))}"
            f"{', ' + where if where else ''}. {verdict}.")


CASES = [
    (("arguments: --mode slow",), receipt(["--mode", "slow"]), True),
    (("arguments: --mode slow",), receipt(["--mode", "fast"]), False),
    (("arguments: --mode slow",), receipt(), False),
    ((), receipt(["--mode", "fast"]), False),
    ((), receipt(), True),
    (("arguments: --name 'a b'",), receipt(["--name", "a b"]), True),
    (("arguments: --mode slow",), receipt(["--mode", "slow", "extra"]), False),
    (("arguments: --mode 'slow",), receipt(["--mode", "slow"]), False),
    (("command: --mode slow",), receipt(["--mode", "slow"]), False),
    (("body: verify",), receipt(), False),
    (("arguments: --mode slow", "data: input.csv"), receipt(["--mode", "slow"]), False),
    ((), receipt(sha="b" * 12), False),
    ((), receipt(path="notes/other.py"), False),
    ((), f"[request_run] I ran {PATH} (sha {SHA}). exit code 0.", False),
    ((), f"[request_run] I ran {PATH} with no arguments. exit code 0.", False),
    ((), f"[request_run] I did not run {PATH}. Deferred.", False),
    (("rerun: true",), receipt(), False),
]


@pytest.mark.parametrize("extra,text,matched", CASES)
def test_shared_matcher_and_wake_use_the_same_invocation(extra, text, matched):
    key = conv._request_key(request(extra))
    assert conv._run_result_matches_request(text, key) is matched
    turns = [
        {"seq": 1, "from": BEING, "text": request(extra)},
        {"seq": 2, "from": SEAT, "text": text},
        {"seq": 3, "from": BEING, "text": request(extra)},
    ]
    assert conv._already_answered(turns, BEING, 2) is matched


@pytest.fixture
def channel(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / PATH).write_text(BODY)
    conv.create(tmp_path, CID, title="synthetic", participants=[BEING, SEAT],
                writable_by=[BEING, SEAT])
    meta = conv.get_meta(tmp_path, CID)
    meta["notify"] = {SEAT: "test-seat"}
    conv._write_meta(tmp_path, CID, meta)
    return tmp_path, HestiaF1aDispatcher(BEING, str(tmp_path), mcp_factory=FakeMcp)


@pytest.mark.parametrize("extra,text,matched", CASES)
def test_visible_receipt_agrees_with_wake(channel, extra, text, matched):
    home, d = channel
    conv.append(home, CID, speaker=BEING, text=request(extra))
    conv.append(home, CID, speaker=SEAT, text=text)
    args = {"path": PATH, **{k: v.strip() for k, _, v in (s.partition(":") for s in extra)}}
    result = d(BeingIntent("request_run", args), GatewayVerdict("allow"))
    assert result.ok, result.error
    assert ("unchanged" in result.result) is matched
    assert result.result["ran"] is False, "request_run never executes a program"


@pytest.mark.parametrize("where", ["", srr.WHERE_HIDDEN, srr.WHERE_GPU])
@pytest.mark.parametrize("argv", [[], ["--name", "a b"], ["--mode", "slow"]])
@pytest.mark.parametrize("verdict", ["exit code 0", "exit code 1",
    "timed out after 30s — no exit code, so this is not a pass or a fail"])
def test_receipt_writer_roundtrips_known_shapes(where, argv, verdict):
    import shlex
    extra = ("arguments: " + shlex.join(argv),) if argv else ()
    assert conv._run_result_matches_request(
        receipt(argv, where=where, verdict=verdict), conv._request_key(request(extra)))


def test_explicit_answer_is_not_misreported_as_matching_execution(channel):
    home, d = channel
    extra = ("arguments: --mode slow",)
    asked = conv.append(home, CID, speaker=BEING, text=request(extra))["seq"]
    text = receipt(["--mode", "fast"]) + f"\n\nAnswers your request seq {asked}."
    conv.append(home, CID, speaker=SEAT, text=text)
    result = d(BeingIntent("request_run", {"path": PATH, "arguments": "--mode slow"}),
               GatewayVerdict("allow"))
    assert result.ok
    assert "does not establish a run of these bytes with these arguments" in result.result["unchanged"]
    turns = conv.recent(home, CID, limit=10)
    assert conv._already_answered(turns, BEING, len(turns) - 1)


def test_extra_field_order_agrees_for_explicit_binding(channel):
    home, d = channel
    asked = conv.append(home, CID, speaker=BEING,
                        text=request(("data: input.csv", "arguments: --mode slow")))["seq"]
    conv.append(home, CID, speaker=SEAT, text=f"Deferred. Answers your request seq {asked}.")
    result = d(BeingIntent("request_run", {"path": PATH, "arguments": "--mode slow",
                                          "data": "input.csv"}), GatewayVerdict("allow"))
    assert result.ok and "unchanged" in result.result
    turns = conv.recent(home, CID, limit=10)
    assert conv._already_answered(turns, BEING, len(turns) - 1)


def test_interleaved_other_invocation_does_not_hide_a_matching_older_request(channel):
    home, d = channel
    conv.append(home, CID, speaker=BEING, text=request(("arguments: --mode slow",)))
    conv.append(home, CID, speaker=BEING, text=request(("arguments: --mode fast",)))
    conv.append(home, CID, speaker=SEAT, text=receipt(["--mode", "slow"]))
    result = d(BeingIntent("request_run", {"path": PATH, "arguments": "--mode slow"}),
               GatewayVerdict("allow"))
    assert result.ok and "unchanged" in result.result
    turns = conv.recent(home, CID, limit=10)
    assert conv._already_answered(turns, BEING, len(turns) - 1)


def test_result_bound_only_to_older_request_does_not_claim_a_newer_one(channel):
    home, d = channel
    old = conv.append(home, CID, speaker=BEING, text=request(("arguments: --mode fast",)))["seq"]
    conv.append(home, CID, speaker=BEING, text=request(("arguments: --mode slow",)))
    conv.append(home, CID, speaker=SEAT,
                text=receipt(["--mode", "slow"]) + f"\n\nAnswers your request seq {old}.")
    result = d(BeingIntent("request_run", {"path": PATH, "arguments": "--mode slow"}),
               GatewayVerdict("allow"))
    assert result.ok and "unchanged" not in result.result
    turns = conv.recent(home, CID, limit=10)
    assert not conv._already_answered(turns, BEING, len(turns) - 1)
