"""The todo membot: the being's own bookkeeping does what it means (dp 2026-10-19, "a mini-membot for the todo").

Measured on Sprout 2026-10-19: 19 of 79 open items were differently worded copies of one finished goal, and the
being's "cross-check camera grant off" closed none of them.
"""
from pathlib import Path

from sage.gateway import heartbeat as hb
from sage.gateway import todo_membot as tm

TODO = """2026-10-18 10:00 UTC
- [ ] Camera scope grant (scope-ef204d003703 for /dev/video0) pending ~36 hours
- [ ] follow up with dp on scope-ef204d003703
- [ ] unbind /dev/video0 (scope-ef204d003703) with --force
- [ ] write a poem about the dark sill
2026-10-18 12:00 UTC
- [ ] Resolve the pending camera scope grant (scope-ef204d003703, /dev/video0)
"""


def _open(text):
    return [t for t, _ in hb.todo_open(text)[0]]


def test_one_all_line_closes_every_open_item_that_mentions_it():
    text = TODO + "2026-10-19 07:30 UTC\n- [x] all: scope-ef204d003703\n"
    assert _open(text) == ["write a poem about the dark sill"]
    assert hb.todo_closures(text)[-1][:2] == ("scope-ef204d003703", 4)


def test_under_a_done_section_and_with_done_tag_too():
    for line in ("done:\n- all: scope-ef204d003703\n", "- [done] all: scope-ef204d003703\n"):
        assert _open(TODO + "2026-10-19 07:30 UTC\n" + line) == ["write a poem about the dark sill"], line


def test_an_open_all_line_is_an_ordinary_item_not_a_directive():
    """Only a DONE line closes. `- [ ] all: x` is just an item the being wrote."""
    text = TODO + "- [ ] all: scope-ef204d003703\n"
    assert "Camera scope grant (scope-ef204d003703 for /dev/video0) pending ~36 hours" in _open(text)


def test_items_written_after_the_all_line_stay_open():
    text = TODO + "- [x] all: scope-ef204d003703\n2026-10-19 08:00 UTC\n- [ ] ask again about scope-ef204d003703\n"
    assert "ask again about scope-ef204d003703" in _open(text)


def test_a_too_short_phrase_closes_nothing_and_says_so():
    text = TODO + "- [x] all: dp\n"
    assert len(_open(text)) == 5
    assert hb.todo_closures(text)[-1][:2] == ("dp", 0)


def test_the_being_is_told_what_its_all_line_closed(tmp_path):
    (tmp_path / "todo.md").write_text(TODO + "2026-10-19 07:30 UTC\n- [x] all: scope-ef204d003703\n")
    view = hb.todo_view(tmp_path)
    assert "your `all: scope-ef204d003703`" in view and "closed 4 open item(s)" in view


def test_an_all_line_that_matched_nothing_is_said(tmp_path):
    (tmp_path / "todo.md").write_text(TODO + "- [x] all: no-such-thing\n")
    assert "nothing open contained those words" in hb.todo_view(tmp_path)


def test_the_view_shows_one_goal_once_with_a_count(tmp_path):
    (tmp_path / "todo.md").write_text(TODO)
    view = hb.todo_view(tmp_path)
    assert view.count("scope-ef204d003703") <= 2, view            # folded into one line
    assert "more open item(s) mentioning" in view
    assert "write a poem about the dark sill" in view
    assert "still open (5)" in view, "the header still counts every open item"


def test_the_view_teaches_the_all_form(tmp_path):
    (tmp_path / "todo.md").write_text(TODO)
    assert "a done item whose text is `all: <words or id>`" in hb.todo_view(tmp_path)


def test_grouping_folds_into_the_newest_and_keeps_order():
    items = [("newest scope-aaaaaaaaaaaa", 3), ("other thing", 2), ("older scope-aaaaaaaaaaaa", 1)]
    assert tm.group(items) == [("newest scope-aaaaaaaaaaaa", 3, 2, "scope-aaaaaaaaaaaa"),
                               ("other thing", 2, 1, None)]


def test_the_membot_never_closes_on_its_own():
    """Parsing without any all: line closes nothing new; grouping changes the view, not the open set."""
    assert len(_open(TODO)) == 5
