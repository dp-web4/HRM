"""Selftest for sage/memory/consolidation.py — the pre-registered properties F1..F5.

Hermetic: builds a toy instance under tmp_path with KNOWN contents, so every number the
graft reports is checked against ground truth, not against itself.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.memory import consolidation as C  # noqa: E402

JOURNAL = (
    "2026-09-04 01:00 UTC: first beat. I recalled first, then was refused mrh.path on the forum.\n"
    "I want a grant on my home next time.\n"
    "2026-09-04 01:30 UTC: second beat. Filed request_scope for my home with one sentence.\n"
)
REFLECT_REPLY = "I wrote one journal line and one memory. The refusal named the rule."
JOURNAL_WRITE = "2026-09-04 02:00 UTC: third beat. Nothing new; the grant is still pending."
REMEMBER = "[grants] zero standing grants; every read outside home refused by mrh.path"


def _beat(ts, host, explore_trace, reflect_trace, reflect_reply="", salvaged=(), thinking=()):
    def turn(tr, reply, salv=(), think=()):
        return {"reply": reply, "steps": len(tr), "capped": False, "trace": tr,
                "thinking": list(think), "salvaged": list(salv)}
    return {"ts": ts, "t0": 0, "elapsed_s": 1.0, "member": "toy-being", "model": "toy:1b", "window_h": 1.0,
            "host_session_id": host, "gate_only": False, "act_first": False, "scope": {},
            "explore": turn(explore_trace, "", salvaged, thinking), "posture": None,
            "reflect": turn(reflect_trace, reflect_reply)}


def _eff(name, args, ok=True, refused=False, rule=None, pending=False, error=None):
    return {"effector": name, "args": args, "ok": ok, "refused": refused, "pending": pending,
            "error": error, "witness_id": None, "rule": rule, "result": "ok" if ok else None}


def make_toy(root, with_journal=True):
    inst = root / "toy-being"
    inst.mkdir()
    beats = [
        # beat 1: a recall, a refused memory_read, a remember; reflect writes the journal
        _beat("2026-09-04T01:00:00Z", "hb-1",
              [_eff("recall", {"query": "grants"}),
               _eff("memory_read", {"path": "/forum/x.md"}, ok=False, refused=True, rule="mrh.path"),
               _eff("remember", {"content": REMEMBER})],
              [_eff("memory_write", {"path": "journal.md", "content": JOURNAL_WRITE}, ok=False, refused=True, rule="mrh.path")],
              reflect_reply=REFLECT_REPLY, salvaged=["memory_read"], thinking=["hmm"]),
        # beat 2: empty — the model returned nothing on both turns
        _beat("2026-09-04T01:30:00Z", "hb-2", [], []),
        # beat 3: one witness; a post-S3 record with the explicit ledger, join, account, wake
        {**_beat("2026-09-04T02:00:00Z", "hb-3", [_eff("witness", {"content": "I noticed the grant died."})], []),
         "interventions": [{"kind": "think_suffix", "suppressed": "thinking (model resolves think off)"},
                           {"kind": "salvage", "phase": "explore", "effector": "witness", "form": "fenced",
                            "suppressed": "text-channel narration in place of a native tool call"}],
         "account": {"present": True, "sha256": "ab" * 32}, "join": {"session": {"n": 2}, "presence": None},
         "wake": {"by": "timer"}},
    ]
    (inst / "heartbeats.jsonl").write_text("\n".join(json.dumps(b) for b in beats) + "\n")
    if with_journal:
        (inst / "journal.md").write_text(JOURNAL)
    (inst / "todo.md").write_text("- ask for the home grant\n\n- read the forum\n")
    # identity.json names the being differently from the transcripts, as legion-gemma3-12b does
    (inst / "identity.json").write_text(json.dumps({"identity": {"name": "toy", "machine": "toy"}}))
    (inst / "experience_buffer.json").write_text(json.dumps([
        {"id": "e1", "prompt": "What would you want to remember from today?", "response": "The grant."},
        {"id": "e2", "prompt": "What would you want to remember from today?", "response": "The refusal."},
        {"id": "e3", "prompt": "How are you?", "response": ""},
    ]))
    (inst / "sessions").mkdir()
    for n, conv in ((1, [{"speaker": "Claude", "text": "Hello."}, {"speaker": "Toy", "text": "Hello back."}]),
                    (2, [{"speaker": "Claude", "text": "Again."}, {"speaker": "Toy", "text": "Yes."}, {"speaker": "Toy", "text": "And more."}])):
        (inst / "sessions" / f"session_{n:03d}.json").write_text(json.dumps(
            {"session": str(n), "phase": "grounding", "model": "toy:1b", "start": f"2026-08-0{n}T00:00:00", "conversation": conv}))
    return inst


# ---------------------------------------------------------------------------- F3 ground truth

def test_f3_ground_truth(tmp_path):
    inst = make_toy(tmp_path)
    ev = C.consolidate(inst, seat="test")
    assert ev["event"] == "graft" and ev["version"] == 1 and ev["supersedes"] is None
    g = json.loads((inst / "grafts" / ev["file"]).read_text())
    t = g["table"]
    assert t["beats"]["n"] == 3 and t["beats"]["empty"] == 1
    assert t["beats"]["span"] == ["2026-09-04T01:00:00Z", "2026-09-04T02:00:00Z"]
    assert t["beats"]["effectors"]["memory_read"] == {"trials": 1, "ok": 0, "refused": 1, "pending": 0, "error": 0}
    assert t["beats"]["effectors"]["memory_write"]["refused"] == 1
    assert t["beats"]["refusal_rules"] == {"mrh.path": 2}
    assert t["beats"]["interventions"] == {
        "salvaged_beats": 1, "salvaged_calls": 1, "thinking_beats": 1, "beats_with_ledger": 1,
        "kinds": {"salvage": 1, "think_suffix": 1},
        "suppressed": {"text-channel narration in place of a native tool call": 1,
                       "thinking (model resolves think off)": 1}}
    assert t["beats"]["join"] == {"session_attributed": 1, "presence_attributed": 0}
    assert t["beats"]["account"] == {"present": 1, "distinct_sha256": 1}
    assert t["beats"]["wake_by"] == {"timer": 1}
    assert t["journal"] == {"present": True, "entries": 2}
    assert t["todo"] == {"present": True, "lines": 2}
    assert t["sessions"]["n"] == 2 and t["sessions"]["being"] == "Toy" and t["sessions"]["being_turns"] == 3
    assert t["sessions"]["speakers"] == {"Claude": 2, "Toy": 3}
    assert t["experience"]["n"] == 3 and t["experience"]["top_prompts"][0]["n"] == 2
    # own words: reflect reply (1) + remember (1) + journal write (1) + witness (1) + journal entries (2)
    #            + being session turns (3) + experience responses with text (2) = 11
    assert t["own_account"]["lines"] == 11
    assert t["own_account"]["per_source"] == {"beats": 4, "experience": 2, "journal": 2, "sessions": 3}
    # the own-word index is a sidecar, named by sha; one line per own-word line
    ix = t["own_account"]["index"]
    ib = (inst / "grafts" / ix["file"]).read_bytes()
    assert ix["lines"] == 11 and ib.count(b"\n") == 11 and C._sha_bytes(ib) == ix["sha256"]
    # training data named exactly: every source has path + sha256 + count
    names = {s["source"] for s in g["training_data"]}
    assert names == {"beats", "journal", "todo", "account", "experience", "sessions"}
    for s in g["training_data"]:
        if s["source"] == "account":       # S1 writes it; the toy has none, and says so
            assert s["present"] is False
            continue
        assert s["present"] and re.fullmatch(r"[0-9a-f]{64}", s["sha256"]) and s["count"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", g["instrument"]["sha256"])


def test_f3_absent_source_is_named_as_absent(tmp_path):
    inst = make_toy(tmp_path, with_journal=False)
    ev = C.consolidate(inst, seat="test")
    g = json.loads((inst / "grafts" / ev["file"]).read_text())
    j = [s for s in g["training_data"] if s["source"] == "journal"][0]
    assert j["present"] is False and j["sha256"] is None and j["count"] == 0
    assert g["table"]["journal"] == {"present": False, "entries": 0}
    assert g["table"]["own_account"]["per_source"].get("journal", 0) == 0


# ---------------------------------------------------------------------------- F1 idempotence

def test_f1_idempotence_logged_noop(tmp_path):
    inst = make_toy(tmp_path)
    ev1 = C.consolidate(inst, seat="test")
    ev2 = C.consolidate(inst, seat="test")
    assert ev1["event"] == "graft" and ev2["event"] == "noop"
    assert ev2["latest"] == ev1["file"]
    log = [json.loads(l) for l in (inst / "grafts" / "consolidation_log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log] == ["graft", "noop"]
    assert len(list((inst / "grafts").glob("self-account-*.v*.json"))) == 1
    # a changed source makes v2 that supersedes v1; v1 is still there, byte-identical
    v1 = (inst / "grafts" / ev1["file"]).read_bytes()
    with open(inst / "journal.md", "a") as f:
        f.write("2026-09-04 03:00 UTC: fourth beat. The grant landed.\n")
    ev3 = C.consolidate(inst, seat="test")
    assert ev3["event"] == "graft" and ev3["version"] == 2 and ev3["supersedes"] == ev1["file"]
    assert (inst / "grafts" / ev1["file"]).read_bytes() == v1
    idx = json.loads((inst / "grafts" / "index.json").read_text())
    assert idx["members"]["toy-being"]["latest"] == ev3["file"]
    assert [v["version"] for v in idx["members"]["toy-being"]["versions"]] == [1, 2]


def test_f1_instrument_change_relands_on_unchanged_sources(tmp_path, monkeypatch):
    """#42 finding 2: the graft is a function of sources AND instrument. A changed instrument over
    the same sources is vN+1 (reason instrument_changed), not a no-op that keeps the old reading."""
    inst = make_toy(tmp_path)
    ev1 = C.consolidate(inst, seat="test")
    assert ev1["reason"] == "first"
    monkeypatch.setattr(C, "_self_sha", lambda: "11" * 32)
    ev2 = C.consolidate(inst, seat="test")
    assert ev2["event"] == "graft" and ev2["version"] == 2 and ev2["reason"] == "instrument_changed"
    assert ev2["source_set_sha256"] == ev1["source_set_sha256"]
    ev3 = C.consolidate(inst, seat="test")
    assert ev3["event"] == "noop" and ev3["latest"] == ev2["file"]
    # a v1 index entry (no instrument sha in the index) is read from the graft file
    idx = json.loads((inst / "grafts" / "index.json").read_text())
    for v in idx["members"]["toy-being"]["versions"]:
        v.pop("instrument_sha256", None)
    (inst / "grafts" / "index.json").write_text(json.dumps(idx))
    assert C.consolidate(inst, seat="test")["event"] == "noop"
    monkeypatch.setattr(C, "_self_sha", lambda: "22" * 32)
    assert C.consolidate(inst, seat="test")["reason"] == "instrument_changed"


def test_f1_read_once_names_what_it_distills(tmp_path, monkeypatch):
    """#42 finding 1: training_data hashes the bytes the table was built from. A beat appended
    while the cycle runs lands in the NEXT graft, not half in this one."""
    inst = make_toy(tmp_path)
    real = C._read_sources
    def racing(instance, cm=None):
        out = real(instance, cm)
        with open(Path(instance) / "heartbeats.jsonl", "a") as f:     # a beat lands mid-cycle
            f.write(json.dumps(_beat("2026-09-04T02:30:00Z", "hb-4", [], [])) + "\n")
        return out
    monkeypatch.setattr(C, "_read_sources", racing)
    ev = C.consolidate(inst, seat="test")
    g = json.loads((inst / "grafts" / ev["file"]).read_text())
    named = [s for s in g["training_data"] if s["source"] == "beats"][0]
    assert named["count"] == 3 and g["table"]["beats"]["n"] == 3        # both say 3: what was hashed
    monkeypatch.setattr(C, "_read_sources", real)
    ev2 = C.consolidate(inst, seat="test")                              # the 4th beat is the next version
    assert ev2["event"] == "graft" and ev2["reason"] == "sources_changed"


def test_error_excludes_refused_and_paths_are_portable(tmp_path):
    """#42 findings 3, 4, 5: a refusal carries an error string too and is not an error; the graft
    names no absolute source path; the excluded tutor speakers are in the table."""
    inst = make_toy(tmp_path)
    b = _beat("2026-09-04T02:30:00Z", "hb-4",
              [_eff("memory_write", {"path": "/x"}, ok=False, refused=True, rule="mrh.path", error="refused: mrh.path"),
               _eff("mesh", {"to": "y"}, ok=False, error="ConnectionError: down")], [])
    with open(inst / "heartbeats.jsonl", "a") as f:
        f.write(json.dumps(b) + "\n")
    ev = C.consolidate(inst, seat="test")
    g = json.loads((inst / "grafts" / ev["file"]).read_text())
    eff = g["table"]["beats"]["effectors"]
    assert eff["memory_write"] == {"trials": 2, "ok": 0, "refused": 2, "pending": 0, "error": 0}
    assert eff["mesh"] == {"trials": 1, "ok": 0, "refused": 0, "pending": 0, "error": 1}
    for s in g["training_data"]:
        assert not os.path.isabs(s["path"]), s
    assert g["training_data"][0]["path"] == "heartbeats.jsonl"
    assert g["table"]["sessions"]["tutor_speakers"] == sorted(C.TUTOR_SPEAKERS)
    assert "Claude" in g["table"]["sessions"]["tutor_speakers"]


# ---------------------------------------------------------------------------- F2 determinism

def test_f2_determinism(tmp_path):
    inst = make_toy(tmp_path)
    a = C.consolidate(inst, tmp_path / "ga", seat="test")
    b = C.consolidate(inst, tmp_path / "gb", seat="test")
    ga = json.loads((tmp_path / "ga" / a["file"]).read_text())
    gb = json.loads((tmp_path / "gb" / b["file"]).read_text())
    assert C._canon(ga["table"]) == C._canon(gb["table"])
    assert C._canon(ga["training_data"]) == C._canon(gb["training_data"])
    ia = ga["table"]["own_account"]["index"]["file"]; ib = gb["table"]["own_account"]["index"]["file"]
    assert (tmp_path / "ga" / ia).read_bytes() == (tmp_path / "gb" / ib).read_bytes()
    assert ga["source_set_sha256"] == gb["source_set_sha256"]


# ---------------------------------------------------------------------------- F4 hygiene

# call-shaped tokens: clients, process spawns, network, summarisers. A tutor's NAME in a transcript
# (TUTOR_SPEAKERS) is data the organ has to recognise to leave it out; it is not a call.
FORBIDDEN = ("ollama", "anthropic", "openai", "subprocess", "requests", "urllib", "http",
             "summar", "paraphras", "prompt(", "complete(", "generate(", "chat(")


def test_f4_no_seat_voice_no_model_call():
    src = open(C.__file__, encoding="utf-8").read()
    # strip the docstring/comments: the rule is about what the code DOES, not what it explains
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    low = code.lower()
    hits = [tok for tok in FORBIDDEN if tok in low]
    assert not hits, f"seat-voice / model-call tokens in consolidation.py: {hits}"
    imports = re.findall(r"^\s*(?:import|from)\s+([\w.]+)", code, flags=re.M)
    allowed = {"__future__", "argparse", "hashlib", "json", "os", "re", "sys", "datetime", "pathlib", "typing"}
    assert set(imports) <= allowed, set(imports) - allowed


# ---------------------------------------------------------------------------- F5 verbatim

def test_f5_carry_is_verbatim_substring_of_source(tmp_path):
    inst = make_toy(tmp_path)
    ev = C.consolidate(inst, seat="test")
    g = json.loads((inst / "grafts" / ev["file"]).read_text())
    carry = g["table"]["own_account"]["carry"]
    assert carry, "nothing carried"
    raw = {"beats": (inst / "heartbeats.jsonl").read_text(), "journal": (inst / "journal.md").read_text()}
    for line in carry:
        src = raw[line["source"]]
        # beats are JSON-encoded on disk; compare against the decoded form
        hay = json.dumps(line["text"])[1:-1] if line["source"] == "beats" else line["text"]
        assert hay in src, (line["source"], line["ref"])
    # the reflect reply and the journal write both carried, plus both journal entries
    assert {c["ref"] for c in carry} >= {"hb-1/reflect.reply", "hb-1/reflect.memory_write", "entry:0", "entry:1"}


# ---------------------------------------------------------------------------- dark by default

def test_dark_by_default(tmp_path, monkeypatch):
    inst = make_toy(tmp_path)
    C.consolidate(inst, seat="test")
    monkeypatch.delenv(C.ENV_SWITCH, raising=False)
    assert C.read_graft(inst) is None
    monkeypatch.setenv(C.ENV_SWITCH, "on")
    g = C.read_graft(inst)
    assert g and g["schema"] == C.SCHEMA_GRAFT and g["member"] == "toy-being"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
