"""The being's journal and todo can start fresh, and can be made optional per instance.

dp, 2026-10-17: "those are real problems that need to get resolved, and doesn't sound like the
being itself has the tools to do it". cbp-being's todo.md had reached 4,921 lines: retire_note
refused it, 367 of 523 memory_edits ticking an item done and 34 of 41 deletions missed on old
text, and on 10-17 21:00Z its `retire_note todo.md` was refused outright.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict  # noqa: E402
from sage.gateway.reference_f1a import ReferenceF1aDispatcher  # noqa: E402

_ALLOW = GatewayVerdict("allow")
_TODO = ("2026-10-10 01:00 UTC\nadded:\n- [ ] run the pipeline test\n- [ ] fix line 7 docstring\n"
         "2026-10-11 02:00 UTC\ndone:\n- fix line 7 docstring\nstill open:\n- write the summary note\n")


def _disp():
    d = tempfile.mkdtemp(prefix="home-records-")
    return ReferenceF1aDispatcher(memory_root=d), d


def test_retiring_the_todo_keeps_it_whole_and_carries_every_open_item_forward():
    from sage.gateway.heartbeat import _todo_items
    disp, root = _disp()
    Path(root, "todo.md").write_text(_TODO)
    before = sorted(t for t, _ in _todo_items(_TODO).values())
    env = disp(BeingIntent("retire_note", {"path": "todo.md", "reason": "obsolete items"}), _ALLOW)
    assert env.ok, env.error
    retired = [n for n in os.listdir(root) if n.startswith("todo.retired-")]
    assert len(retired) == 1 and retired[0] in env.result
    assert _TODO in Path(root, retired[0]).read_text(), "the old log is kept whole"
    fresh = Path(root, "todo.md").read_text()
    assert sorted(t for t, _ in _todo_items(fresh).values()) == before, fresh
    assert "fix line 7 docstring" not in fresh, "a closed item is not carried"
    assert "2 item(s) still open" in env.result


def test_retiring_an_all_done_todo_starts_it_empty():
    disp, root = _disp()
    Path(root, "todo.md").write_text("2026-10-10 01:00 UTC\ndone:\n- a finished thing\n")
    env = disp(BeingIntent("retire_note", {"path": "todo.md", "reason": "all done"}), _ALLOW)
    assert env.ok and "starts empty" in env.result, env.result


def test_retiring_the_journal_starts_a_fresh_one_that_says_where_the_old_went():
    disp, root = _disp()
    Path(root, "journal.md").write_text("2026-10-10 01:00 UTC\nold entry\n")
    env = disp(BeingIntent("retire_note", {"path": "journal.md", "reason": "fresh start"}), _ALLOW)
    assert env.ok, env.error
    retired = [n for n in os.listdir(root) if n.startswith("journal.retired-")][0]
    assert retired in Path(root, "journal.md").read_text()


def test_retired_records_stay_searchable_by_recall():
    from sage.gateway.home_recall import index_home
    disp, root = _disp()
    Path(root, "journal.md").write_text("2026-10-10 01:00 UTC\nthe orthogonal projection result\n")
    disp(BeingIntent("retire_note", {"path": "journal.md", "reason": "fresh"}), _ALLOW)
    texts = " ".join(u["text"] for u in index_home(Path(root)))
    assert "orthogonal projection" in texts


def test_other_home_root_files_are_still_not_retirable():
    disp, root = _disp()
    Path(root, "identity.json").write_text("{}")
    env = disp(BeingIntent("retire_note", {"path": "identity.json", "reason": "r"}), _ALLOW)
    assert not env.ok and Path(root, "identity.json").exists()


def test_a_missed_todo_edit_names_the_doors_that_need_no_exact_text():
    disp, root = _disp()
    Path(root, "todo.md").write_text(_TODO)
    env = disp(BeingIntent("memory_edit", {"path": "todo.md", "old": "- [ ] not there", "new": "- [x] x"}), _ALLOW)
    assert not env.ok and "under done:" in env.error and "retire_note todo.md" in env.error
    Path(root, "other.md").write_text("abc\n")
    env = disp(BeingIntent("memory_edit", {"path": "other.md", "old": "zzz", "new": "y"}), _ALLOW)
    assert not env.ok and "under done:" not in env.error


def test_escalate_treats_root_todo_and_journal_as_in_bound_for_retire_note():
    from sage.gateway.escalate import ungrantable
    root = tempfile.mkdtemp(prefix="ownhome-")
    for n in ("todo.md", "journal.md", "identity.json"):
        Path(root, n).write_text("x")
    assert ungrantable(BeingIntent("retire_note", {"path": "todo.md", "reason": "r"}), root) is None
    assert ungrantable(BeingIntent("retire_note", {"path": "journal.md", "reason": "r"}), root) is None
    assert ungrantable(BeingIntent("retire_note", {"path": "identity.json", "reason": "r"}), root)


def test_reflect_records_are_required_unless_the_instance_makes_them_optional():
    from sage.gateway.heartbeat import REFLECT, reflect_optional_for, reflect_prompt
    kw = dict(date="D", say_line="", say_first="")
    assert reflect_prompt(reflect_optional_for(None), **kw) == REFLECT.format(**kw)
    assert reflect_optional_for({"reflect_optional": ["bogus"]}) == []
    j = reflect_prompt(reflect_optional_for({"reflect_optional": ["journal"]}), **kw)
    assert 'if you keep a journal and want to: memory_write path "journal.md"' in j
    assert "if you keep a todo" not in j
    both = reflect_prompt(reflect_optional_for({"reflect_optional": "todo"}), **kw)
    assert "if you keep a todo list" in both and "if you keep a journal" not in both


# -- Nova's review of 0557d9bcf: item age, concurrent appends, interrupted retires ------------

def test_carried_items_keep_the_date_they_were_opened():
    from datetime import datetime, timezone
    from sage.gateway.heartbeat import todo_open
    disp, root = _disp()
    log = ("- [ ] an undated item from before any header\n" + _TODO
           + "2026-10-17 20:00 UTC\nadded:\n- [ ] a recent item\n")
    Path(root, "todo.md").write_text(log)
    now = datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc)
    before = todo_open(log, now=now)
    env = disp(BeingIntent("retire_note", {"path": "todo.md", "reason": "obsolete"}), _ALLOW)
    assert env.ok, env.error
    fresh = Path(root, "todo.md").read_text()
    assert todo_open(fresh, now=now) == before, fresh
    assert before[1] == 2, "the two old items stay counted as old, not pulled back into view"


def test_appends_racing_retires_lose_no_entry():
    import threading
    disp, root = _disp()
    Path(root, "journal.md").write_text("2026-10-10 01:00 UTC\nfirst\n")
    tokens = [f"entry-{i:04d}" for i in range(300)]

    def appender():
        for t in tokens:
            env = disp(BeingIntent("memory_write", {"path": "journal.md", "content": t}), _ALLOW)
            assert env.ok, env.error

    th = threading.Thread(target=appender)
    th.start()
    retires = 0
    while th.is_alive():
        env = disp(BeingIntent("retire_note", {"path": "journal.md", "reason": "race"}), _ALLOW)
        retires += env.ok
    th.join()
    kept = "".join(Path(root, n).read_text() for n in os.listdir(root) if n.startswith("journal"))
    lost = [t for t in tokens if t not in kept]
    assert retires > 1 and not lost, f"{len(lost)} lost across {retires} retires"


def test_an_interrupted_retire_loses_nothing_and_the_next_one_finishes_it():
    import pytest
    import sage.gateway.reference_f1a as rf
    disp, root = _disp()
    Path(root, "todo.md").write_text(_TODO)
    real, calls = os.replace, []

    def dies_on_the_second_swap(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise KeyboardInterrupt("killed between the two steps")
        real(src, dst)

    rf.os.replace = dies_on_the_second_swap
    try:
        with pytest.raises(KeyboardInterrupt):
            disp(BeingIntent("retire_note", {"path": "todo.md", "reason": "obsolete"}), _ALLOW)
    finally:
        rf.os.replace = real
    assert Path(root, "todo.md").read_text() == _TODO, "the original is untouched"
    copies = [n for n in os.listdir(root) if n.startswith("todo.retired-")]
    assert len(copies) == 1 and _TODO in Path(root, copies[0]).read_text()
    env = disp(BeingIntent("retire_note", {"path": "todo.md", "reason": "obsolete"}), _ALLOW)
    assert env.ok and "interrupted" in env.result, env.result
    assert [n for n in os.listdir(root) if n.startswith("todo.retired-")] == copies, "no second copy"
    assert "carried forward" in Path(root, "todo.md").read_text().lower()
