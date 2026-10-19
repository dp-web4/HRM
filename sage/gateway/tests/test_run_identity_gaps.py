"""Offline request/result identity probes for SAGE #464.

These exercise the current queue and wake readers, not a proposed replacement.
Strict xfails are known counterexamples, NOT passing capability evidence. They
must become ordinary assertions when repaired. Run with --runxfail to reproduce
the failures directly. All conversations are synthetic, no being program is executed,
and no live being, todo, memory or notification state is changed.

A later result for the same filename is not an answer to every earlier request.
Explicit request references may establish that something was answered; they do
not establish successful execution. This is separate from deciding when an old
result remains useful for a new experiment (inputs/environment can change too).
"""
import importlib.util
from pathlib import Path

import pytest

from sage.gateway import conversations as conv

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seat_run_requests.py"
_spec = importlib.util.spec_from_file_location("seat_run_identity_gap_probe", _SCRIPT)
srr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srr)

BEING, SEAT, CID = "test-being", "test-seat", "test-channel"
SHA_A, SHA_B = "a" * 12, "b" * 12


def request_text(sha=SHA_A, mode="fast", path="notes/sample.py"):
    tag = f"sha256:{sha}" if sha else "digest unavailable"
    return (f"[request_run] {path}\nwhy: measure the requested mode\n"
            f"(10 bytes, {tag}; the seat decides whether to run it and answers here)\n"
            f"arguments: --mode {mode}")


def receipt_text(sha=SHA_A, mode="fast", path="notes/sample.py", answers=()):
    text = (f"[request_run] I ran {path} (sha {sha}) "
            f"{srr.ran_line(['--mode', mode])}. exit code 0.")
    return text + ("\n\n" + srr.answers_line(list(answers)) if answers else "")


@pytest.fixture
def inst(tmp_path, monkeypatch):
    monkeypatch.setenv("SEAT_ID", SEAT)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "sample.py").write_text("print('fixture')\n")
    conv.create(tmp_path, CID, title="synthetic identity probe",
                participants=[BEING, SEAT], writable_by=[BEING, SEAT])
    return tmp_path


def append(inst, speaker, text):
    return conv.append(inst, CID, speaker=speaker, text=text)["seq"]


def pending(inst):
    return [t["seq"] for t in srr.pending(inst, CID)]


@pytest.mark.parametrize("old_sha,old_mode", [
    (SHA_B, "fast"),  # same path/arguments, different source version
    (SHA_A, "slow"),  # same path/source, different invocation
    (None, "fast"),   # legacy request without enough evidence to link
], ids=["different-source", "different-arguments", "unknown-source"])
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#464: pending() closes an unnamed older request by filename")
def test_result_for_newer_request_does_not_close_unmatched_older_one(inst, old_sha, old_mode):
    old = append(inst, BEING, request_text(old_sha, old_mode))
    new = append(inst, BEING, request_text())
    append(inst, SEAT, receipt_text(answers=[new]))
    assert pending(inst) == [old]


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#464: wake fallback compares result path/digest but not arguments")
def test_result_for_other_arguments_does_not_suppress_the_unanswered_repeat():
    turns = [
        {"seq": 1, "from": BEING, "text": request_text(mode="slow")},
        {"seq": 2, "from": SEAT, "text": receipt_text(mode="fast")},
        {"seq": 3, "from": BEING, "text": request_text(mode="slow")},
    ]
    assert not conv._already_answered(turns, BEING, 2)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#464: dispatcher UNCHANGED fallback ignores result arguments")
def test_other_arguments_do_not_become_the_being_facing_answer(inst):
    import hashlib
    from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict
    from sage.gateway.hestia_dispatch import HestiaF1aDispatcher
    from sage.gateway.tests.test_hestia_dispatch import FakeMcp

    meta = conv.get_meta(inst, CID)
    meta["notify"] = {SEAT: "test-seat"}
    conv._write_meta(inst, CID, meta)
    d = HestiaF1aDispatcher(BEING, str(inst), mcp_factory=FakeMcp)
    intent = BeingIntent("request_run", {"path": "notes/sample.py", "arguments": "--mode slow"})
    first = d(intent, GatewayVerdict("allow"))
    assert first.ok, first.error
    sha = hashlib.sha256((inst / "notes" / "sample.py").read_bytes()).hexdigest()[:12]
    append(inst, SEAT, receipt_text(sha, "fast"))
    second = d(intent, GatewayVerdict("allow"))
    assert second.ok, second.error
    assert "unchanged" not in second.result


def test_explicit_answer_closes_named_request_without_claiming_execution(inst):
    old = append(inst, BEING, request_text(SHA_B, "slow"))
    new = append(inst, BEING, request_text())
    append(inst, SEAT, "Deferred; no execution occurred. " + srr.answers_line([old]))
    assert pending(inst) == [new]


def test_other_filename_and_unrelated_prose_do_not_close_request(inst):
    old = append(inst, BEING, request_text())
    append(inst, SEAT, "I saw your request; this is unrelated conversation.")
    append(inst, SEAT, receipt_text(path="notes/other.py"))
    assert pending(inst) == [old]


def test_exact_duplicate_request_can_share_a_bound_answer():
    turns = [
        {"seq": 1, "from": BEING, "text": request_text()},
        {"seq": 2, "from": SEAT, "text": receipt_text(answers=[1])},
        {"seq": 3, "from": BEING, "text": request_text()},
    ]
    assert conv._already_answered(turns, BEING, 2)


def test_wake_reader_already_rejects_a_different_source_digest():
    turns = [
        {"seq": 1, "from": BEING, "text": request_text()},
        {"seq": 2, "from": SEAT, "text": receipt_text(SHA_B)},
        {"seq": 3, "from": BEING, "text": request_text()},
    ]
    assert not conv._already_answered(turns, BEING, 2)
