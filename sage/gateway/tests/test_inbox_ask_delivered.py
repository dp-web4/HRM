"""An ask arrives with what it says (dp 2026-10-11, option (b)): the courier copies the pointed-to document
from the fleet's shared checkout into the envelope it writes in the being's home, and the beat shows an
unopened ask's words from that envelope. Measured on HUB first: hub-being peeked 17 notices a beat and made
0 memory_read calls in 328 tool calls, so a pointer alone was an ask never read.

The courier copies ONLY from shared-context, and the beat reads ONLY the being's own home."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sage.gateway import being_inbox_drain as d  # noqa: E402
from sage.gateway import heartbeat as hb  # noqa: E402


def _ws(tmp_path):
    """<root>/sage (the workspace), <root>/shared-context, <root>/private-context, and a being home."""
    root = tmp_path / "ai"
    (root / "sage").mkdir(parents=True)
    (root / "shared-context" / "forum").mkdir(parents=True)
    (root / "private-context").mkdir()
    home = root / "sage" / "sage" / "instances" / "b-being"
    home.mkdir(parents=True)
    return root, str(root / "sage"), home


def test_the_courier_copies_a_shared_context_document(tmp_path):
    root, ws, _ = _ws(tmp_path)
    (root / "shared-context/forum/ask.md").write_text("cbp-being asks: what do you notice in your room?")
    got = d.delivered_content("shared-context/forum/ask.md#thread=x", ws)
    assert got["text"].startswith("cbp-being asks") and not got["truncated"]


def test_the_courier_copies_nothing_outside_shared_context(tmp_path):
    root, ws, _ = _ws(tmp_path)
    (root / "private-context/secret.md").write_text("private")
    (root / "shared-context/forum/tool.py").write_text("code")
    os.symlink(root / "private-context/secret.md", root / "shared-context/forum/link.md")
    for ptr in ("private-context/secret.md", "shared-context/../private-context/secret.md",
                "shared-context/forum/link.md", "https://github.com/dp-web4/SAGE/pull/1",
                "shared-context/forum/tool.py", "/etc/passwd", ""):
        got = d.delivered_content(ptr, ws)
        assert "text" not in got and got["why"], ptr
    assert "not in this machine's copy" in d.delivered_content("shared-context/forum/absent.md", ws)["why"]


def test_a_long_document_is_capped_and_says_so(tmp_path):
    root, ws, home = _ws(tmp_path)
    (root / "shared-context/forum/long.md").write_text("x" * (d.DELIVERED_CHARS + 500))
    got = d.delivered_content("shared-context/forum/long.md", ws)
    assert len(got["text"]) == d.DELIVERED_CHARS and got["truncated"]
    p = d.persist_notice(home, {"id": "n1", "kind": "coordination", "from": "u",
                                "pointer_uri": "shared-context/forum/long.md"}, "seat", workspace=ws)
    assert f"first {d.DELIVERED_CHARS} of {d.DELIVERED_CHARS + 500}" in p.read_text()


def test_the_envelope_carries_the_words_labelled_as_the_senders_or_says_why_not(tmp_path):
    root, ws, home = _ws(tmp_path)
    (root / "shared-context/forum/ask.md").write_text("Is your drain stalled?")
    p = d.persist_notice(home, {"id": "n1", "kind": "coordination", "from": "u",
                                "pointer_uri": "shared-context/forum/ask.md"}, "hub-claude", workspace=ws)
    t = p.read_text()
    assert d.DELIVERED_HEADING in t and "the sender's words" in t and "Is your drain stalled?" in t
    q = d.persist_notice(home, {"id": "n2", "kind": "reply", "from": "u",
                                "pointer_uri": "https://github.com/x/y/pull/1"}, "hub-claude", workspace=ws)
    assert d.DELIVERED_HEADING not in q.read_text() and "did not copy what it says" in q.read_text()


def _notice(i, kind, ptr):
    return {"id": i, "kind": kind, "from_plugin": "claude-code", "pointer_uri": ptr, "queued_at": "2026-10-11T10:00"}


def test_the_beat_shows_what_an_unopened_ask_says_from_the_beings_home(tmp_path):
    root, ws, home = _ws(tmp_path)
    (root / "shared-context/forum/ask.md").write_text("cbp-being asks:\n\n  is your drain stalled?")
    p = d.persist_notice(home, {"id": "n1", "kind": "coordination", "from": "u",
                                "pointer_uri": "shared-context/forum/ask.md"}, "hub-claude", workspace=ws)
    ptr = f"sage/instances/b-being/notes/inbox/{p.name}"
    out = hb.render_inbox([_notice(7, "coordination", ptr)], home=home)
    assert "It says (the sender's words" in out and "cbp-being asks: is your drain stalled?" in out
    assert f"memory_read on {ptr}" in out                       # the full text stays one read away
    # handled -> folded, nothing inlined
    assert "It says" not in hb.render_inbox([_notice(7, "coordination", ptr)], handled={7}, home=home)
    # without the home the beat reads nothing
    assert "It says" not in hb.render_inbox([_notice(7, "coordination", ptr)])


def test_at_most_two_asks_are_inlined_and_only_asks(tmp_path):
    root, ws, home = _ws(tmp_path)
    ns = []
    for i in range(4):
        (root / f"shared-context/forum/a{i}.md").write_text(f"ask number {i}")
        p = d.persist_notice(home, {"id": f"n{i}", "kind": "reply", "from": "u",
                                    "pointer_uri": f"shared-context/forum/a{i}.md"}, "s", workspace=ws)
        ns.append(_notice(10 + i, "reply", f"sage/instances/b-being/notes/inbox/{p.name}"))
    ns.append(_notice(20, "forum-note", ns[0]["pointer_uri"]))
    out = hb.render_inbox(ns, home=home)
    assert out.count("It says") == hb.INLINE_ASKS
    assert "ask number 3" in out and "ask number 2" in out      # newest first


def test_an_envelope_outside_the_beings_inbox_is_never_read(tmp_path):
    _, _, home = _ws(tmp_path)
    (tmp_path / "elsewhere.md").write_text(f"{d.DELIVERED_HEADING}\n\n~~~~\nnot yours\n~~~~\n")
    assert hb.delivered_excerpt(home, str(tmp_path / "elsewhere.md")) == ""
    assert hb.delivered_excerpt(home, "sage/instances/b-being/notes/inbox/../../../elsewhere.md") == ""


def test_the_drain_delivers_the_words_end_to_end(tmp_path):
    root, ws, home = _ws(tmp_path)
    (root / "shared-context/forum/ask.md").write_text("hello from cbp-being")
    sent = []
    out = d.drain_once(home, "unused", workspace=ws, member="b-being", relayed_by="seat",
                       fetch=lambda: [{"id": "n9", "kind": "coordination", "from": "u",
                                       "pointer_uri": "shared-context/forum/ask.md"}],
                       notify=lambda k, p: sent.append(p) or {"ok": True})
    assert out["persisted"] == 1 and out["notified"] == 1
    assert "/notes/inbox/" in sent[0]
    assert "hello from cbp-being" in hb.delivered_excerpt(home, sent[0])
