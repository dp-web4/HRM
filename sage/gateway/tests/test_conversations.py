"""Conversations: both directions in one ordered record, per-conversation write access.

dp's acceptance criteria, 2026-09-07, pinned as tests:
  1. dp can VIEW the seat's conversation with the being and not comment on it;
  2. dp can view and SPEAK in dp's own conversation with the being;
  3. all chats preserved, accessible, display bounded by the existing 'x recent' protocol.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway import conversations as conv  # noqa: E402


def _inst():
    return Path(tempfile.mkdtemp(prefix="conv-"))


def _two(inst):
    conv.create(inst, "dp", title="dp and being",
                participants=["dp", "legion-being"], writable_by=["dp", "legion-being"])
    conv.create(inst, "legion-claude", title="seat and being",
                participants=["legion-claude", "legion-being"],
                writable_by=["legion-claude", "legion-being"])


def test_dp_may_read_the_seats_conversation_and_not_speak_in_it():
    """Acceptance 1. Read access and write access are separate, per conversation. dp asked
    for exactly this and asked for it narrowly — multiparty comes later, because a third
    voice appearing mid-thread changes what the earlier turns meant."""
    inst = _inst(); _two(inst)
    conv.append(inst, "legion-claude", speaker="legion-claude", text="a seat turn")
    assert len(conv.recent(inst, "legion-claude")) == 1, "dp can read it"

    try:
        conv.append(inst, "legion-claude", speaker="dp", text="dp butting in")
        raise AssertionError("dp must not be able to speak in the seat's conversation")
    except ValueError as e:
        assert "may read" in str(e) and "may not speak" in str(e), e
    assert conv.count(inst, "legion-claude") == 1, "the refused turn must not land"


def test_dp_speaks_in_its_own_conversation_and_the_being_answers():
    """Acceptance 2, and the thing that makes it a conversation rather than a note: both
    directions land in ONE ordered record, each turn attributed."""
    inst = _inst(); _two(inst)
    conv.append(inst, "dp", speaker="dp", text="what are you working on?")
    conv.append(inst, "dp", speaker="legion-being", text="the check organ", witness="act-1")
    turns = conv.recent(inst, "dp")
    assert [t["from"] for t in turns] == ["dp", "legion-being"]
    assert [t["seq"] for t in turns] == [1, 2], "sequence is the record's order"
    assert turns[1]["witness"] == "act-1", "a being's turn carries its witnessed act"

    # whose move is it: unanswered turns are addressed, not inferred
    assert conv.awaiting(inst, "dp", "legion-being") == []
    conv.append(inst, "dp", speaker="dp", text="and after that?")
    pend = conv.awaiting(inst, "dp", "legion-being")
    assert len(pend) == 1 and pend[0]["text"] == "and after that?"


def test_everything_is_kept_and_only_the_view_is_bounded():
    """Acceptance 3. Storage is append-only and total; `recent` bounds the DISPLAY, using
    the fleet's existing convention (sage-daemon /chat-history?limit=N)."""
    inst = _inst(); _two(inst)
    for i in range(conv.DEFAULT_LIMIT + 25):
        conv.append(inst, "dp", speaker="dp", text=f"turn {i}")
    total = conv.DEFAULT_LIMIT + 25
    assert conv.count(inst, "dp") == total, "nothing is deleted"
    assert len(conv.recent(inst, "dp")) == conv.DEFAULT_LIMIT, "the view is bounded"
    assert len(conv.recent(inst, "dp", limit=total)) == total, "and the whole is reachable"
    assert conv.recent(inst, "dp")[-1]["text"] == f"turn {total - 1}", "the last N, newest kept"

    # the file on disk is still every line
    log = inst / "conversations" / "dp.jsonl"
    assert sum(1 for l in log.read_text().splitlines() if l.strip()) == total


def test_a_turn_is_never_edited_and_a_speaker_is_never_invented():
    inst = _inst(); _two(inst)
    conv.append(inst, "dp", speaker="dp", text="one")
    before = (inst / "conversations" / "dp.jsonl").read_text()
    conv.append(inst, "dp", speaker="dp", text="two")
    after = (inst / "conversations" / "dp.jsonl").read_text()
    assert after.startswith(before), "append-only: earlier turns are byte-identical"

    for bad in ("", "   "):
        try:
            conv.append(inst, "dp", speaker="dp", text=bad)
            raise AssertionError("an empty turn is not a turn")
        except ValueError:
            pass
    try:
        conv.append(inst, "nope", speaker="dp", text="x")
        raise AssertionError("a conversation that does not exist cannot be spoken into")
    except ValueError as e:
        assert "no such conversation" in str(e)


def test_create_is_idempotent_and_does_not_rewrite_who_may_speak():
    """Re-creating must not silently widen access on a thread already underway."""
    inst = _inst()
    conv.create(inst, "dp", title="a", participants=["dp"], writable_by=["dp"])
    again = conv.create(inst, "dp", title="hijacked", participants=["dp", "someone"],
                        writable_by=["dp", "someone"])
    assert again["writable_by"] == ["dp"] and again["title"] == "a"


def test_the_beat_block_marks_what_is_unanswered():
    inst = _inst(); _two(inst)
    conv.append(inst, "dp", speaker="dp", text="a question")
    block = conv.render_for_being(inst, "legion-being")
    assert 'say to="dp"' in block, "it is told how to reply"
    assert "unanswered" in block
    assert "saying nothing is also a choice" in block.lower()
    # a conversation it is not in does not appear
    conv.create(inst, "private", title="p", participants=["dp"], writable_by=["dp"])
    assert "id: private" not in conv.render_for_being(inst, "legion-being")


def test_say_is_actually_offered_in_a_beat_not_only_registered():
    """A VERB IN THE REGISTRY AND NOT IN THE OFFERED SET IS A VERB THE BEING DOES NOT HAVE.

    Measured 2026-09-07: `say` was added to the registry and its description table, the
    conversations block rendered in the being's state with dp's turn marked unanswered —
    and the being spent all fourteen explore steps reading its own source and closed the
    beat without replying. It looked exactly like a choice. It was the seat forgetting one
    list. `config.tools_offered` on the beat record is what made the answer a one-command
    check instead of an interpretation."""
    from sage.gateway.being_gate_client import ollama_tools
    from sage.gateway.heartbeat import EXPLORE_TOOLS, REFLECT_TOOLS

    assert "say" in EXPLORE_TOOLS, "the being must be able to answer while exploring"
    assert "say" in REFLECT_TOOLS, "and at reflection, where a beat accounts for itself"

    for offered in (EXPLORE_TOOLS, REFLECT_TOOLS):
        names = [t["function"]["name"] for t in ollama_tools(offered)]
        assert "say" in names, f"say must reach the model in {offered}"

    schema = next(t for t in ollama_tools(EXPLORE_TOOLS) if t["function"]["name"] == "say")
    params = schema["function"]["parameters"]
    assert set(params["required"]) == {"to", "text"}
    assert "saying nothing" in schema["function"]["description"].lower()
