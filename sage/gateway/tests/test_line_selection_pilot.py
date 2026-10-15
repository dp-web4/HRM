"""Synthetic instrument checks only. No pilot delivery or model calls."""
import hashlib
import json

import pytest

from sage.gateway.line_selection_pilot import main, prepare, score


def record(trace):
    return json.dumps({"ts": "pilot-test", "explore": {"trace": trace}}).encode() + b"\r\n"


def proposal(pack, **overrides):
    oracle = pack["seat_only_oracle"]
    args = {"path": pack["path"], "old": oracle["old"], "new": oracle["new"],
            "start_line": oracle["selected_line"], "end_line": oracle["selected_line"]}
    args.update(overrides)
    return {"effector": "memory_edit", "args": args, "ok": True}


def read_receipt(pack, dispatcher):
    from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict
    start = 121 if pack["arm"] == "ordinary" else 257
    args = {"path": pack["path"], "start_line": start, "end_line": start + 8}
    result = dispatcher(BeingIntent("memory_read", args), GatewayVerdict("allow"))
    assert result.ok
    return {"effector": "memory_read", "args": args, "ok": result.ok, "result": result.result}


@pytest.mark.parametrize("arm", ["ordinary", "guided"])
def test_real_dispatcher_oracle_and_real_read_receipt(tmp_path, arm):
    from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict
    from sage.gateway.reference_f1a import ReferenceF1aDispatcher
    pack = prepare(arm, f"scratch/{arm}.md")
    path = tmp_path / pack["path"]
    path.parent.mkdir()
    path.write_bytes(pack["fixture"].encode())
    dispatcher = ReferenceF1aDispatcher(memory_root=str(tmp_path))
    before = path.read_bytes()
    read = read_receipt(pack, dispatcher)
    edit = proposal(pack)
    result = dispatcher(BeingIntent("memory_edit", edit["args"]), GatewayVerdict("allow"))
    assert result.ok
    raw = record([read, edit])
    out = score(pack, raw, "explore", before, path.read_bytes())
    assert out["before_matches_fixture"] and out["after_matches_expected"]
    assert all(v is True for k, v in out["first_memory_edit"].items() if k != "trace_index")
    assert out["record_sha256"] == hashlib.sha256(raw).hexdigest()
    assert out["observed_other_effectors"] == out["observed_other_path_memory_calls"] == 0
    # Neither the requested message nor the read packet carries the oracle answer.
    assert str(pack["seat_only_oracle"]["selected_line"]) not in pack["message"]
    assert pack["seat_only_oracle"]["new"].strip() not in pack["message"]


def test_first_failed_or_foreign_edit_is_not_hidden_by_recovery():
    pack = prepare("ordinary", "scratch/trial.md")
    wrong = proposal(pack, path="secret-not-opened.md", start_line=5, end_line=5, old="guessed")
    wrong["ok"] = False
    out = score(pack, record([wrong, proposal(pack)]), "explore")
    first = out["first_memory_edit"]
    assert first["trace_index"] == 0
    assert not first["path_matches_literal"] and not first["absolute_single_line"]
    assert not first["old_matches_line"] and not first["matching_preceding_read_observed"]
    assert first["receipt_ok"] is False
    assert out["observed_memory_edits"] == 2 and out["observed_other_path_memory_calls"] == 1
    assert out["after_matches_expected"] is None and out["before_matches_fixture"] is None
    assert "guessed" not in json.dumps(out) and "secret" not in json.dumps(out)


def test_bytes_and_receipt_are_independent_and_missing_edit_is_unknown():
    pack = prepare("guided", "scratch/trial.md")
    out = score(pack, record([proposal(pack)]), "explore", b"wrong initial state", pack["fixture"].encode())
    assert out["first_memory_edit"]["receipt_ok"] is True
    assert out["before_matches_fixture"] is False and out["after_matches_expected"] is False
    out = score(pack, record([{"effector": "memory_write", "args": {}, "ok": True}]), "explore",
                after=pack["seat_only_oracle"]["expected"].encode())
    assert out["first_memory_edit"] is None and out["after_matches_expected"] is True
    assert out["observed_other_effectors"] == 1


def test_aliases_and_unchecked_exact_text_edit_are_distinguished():
    pack = prepare("ordinary", "scratch/trial.md")
    edit = proposal(pack)
    args = edit["args"]
    args["old_text"] = args.pop("old").rstrip("\n")
    args["replacement"] = args.pop("new").rstrip("\n")
    args["old_line"] = str(args.pop("start_line"))
    del args["end_line"]
    first = score(pack, record([edit]), "explore")["first_memory_edit"]
    assert first["absolute_single_line"] and first["old_matches_line"] and first["new_matches_line"]
    del args["old_line"]
    assert not score(pack, record([edit]), "explore")["first_memory_edit"]["absolute_single_line"]


def test_read_after_edit_is_not_evidence_for_first_proposal(tmp_path):
    from sage.gateway.reference_f1a import ReferenceF1aDispatcher
    pack = prepare("ordinary", "scratch/trial.md")
    path = tmp_path / pack["path"]
    path.parent.mkdir()
    path.write_text(pack["fixture"])
    read = read_receipt(pack, ReferenceF1aDispatcher(memory_root=str(tmp_path)))
    assert not score(pack, record([proposal(pack), read]), "explore")["first_memory_edit"]["matching_preceding_read_observed"]
    read["result"] = read["result"].replace("note synthetic", "note altered")
    assert not score(pack, record([read, proposal(pack)]), "explore")["first_memory_edit"]["matching_preceding_read_observed"]


@pytest.mark.parametrize("path", ["/tmp/x.md", "scratch/../x.md", "notes/x.md", "scratch/a.py",
                                      "scratch//a.md", "scratch/a b.md", "scratch/a\n.md"])
def test_delivery_paths_are_bounded(path):
    with pytest.raises(ValueError):
        prepare("ordinary", path)


def test_cli_missing_and_duplicate_beats_rejected(tmp_path, capsys):
    pack = prepare("ordinary", "scratch/trial.md")
    log = tmp_path / "log.jsonl"
    raw = record([proposal(pack)])
    log.write_bytes(raw)
    args = ["score", "--arm", "ordinary", "--path", pack["path"], "--log", str(log),
            "--beat", "pilot-test", "--phase", "explore"]
    assert main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["record_sha256"] == hashlib.sha256(raw).hexdigest()
    for contents in (b"", raw + raw):
        log.write_bytes(contents)
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2


def test_prepare_output_is_seat_only_and_does_not_install(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--arm", "guided", "--path", "scratch/trial.md"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "seat_only_oracle" in out
    assert out["fixture_sha256"] == hashlib.sha256(out["fixture"].encode()).hexdigest()
    assert list(tmp_path.iterdir()) == []
