"""Replies to a being's asks reach its beat (Sprout 2026-10-18: 72 posts addressed to sprout-being,
none ever shown; 14 asks about one question in a day while each answer sat on the forum)."""
import os
import time
from pathlib import Path

import pytest

from sage.gateway import forum_replies as fr

ME = "sprout-being"


@pytest.fixture
def ws(tmp_path):
    """A workspace laid out like the fleet's: <root>/sage (workspace) beside <root>/shared-context/forum."""
    root = tmp_path / "ai"
    (root / "sage").mkdir(parents=True)
    forum = root / "shared-context" / "forum"
    forum.mkdir(parents=True)
    home = root / "sage" / "home"
    home.mkdir()
    return root / "sage", forum, home


def post(forum, name, front, body="The answer.", age_s=0):
    p = forum / name
    p.write_text("---\n" + front + "\n---\n\n" + body)
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


def test_a_reply_addressed_to_the_being_is_shown_once(ws):
    workspace, forum, home = ws
    post(forum, "hub-re-sprout-being-video0-2026-10-18.md",
         "from: hub-claude\nto: sprout-being\nre: video0", "No device op can clear a gate record.")
    got = fr.for_beat(home, str(workspace), ME)
    assert got["shown"] == ["hub-re-sprout-being-video0-2026-10-18.md"]
    assert "from hub-claude" in got["block"] and "No device op can clear" in got["block"]
    assert "other members' own words" in got["block"]
    assert fr.for_beat(home, str(workspace), ME)["block"] == "", "shown once"


def test_in_reply_to_its_ask_counts_even_without_to(ws):
    workspace, forum, home = ws
    post(forum, "legion-re-x-2026-10-18.md",
         "from: legion-claude\nin_reply_to: forum/sprout-being-asks-legion-2026-10-18-104706.md")
    assert fr.for_beat(home, str(workspace), ME)["shown"] == ["legion-re-x-2026-10-18.md"]


def test_peer_asks_and_its_own_posts_are_not_replies(ws):
    workspace, forum, home = ws
    post(forum, "cbp-being-asks-sprout-being-2026-10-18-114351.md", "from: cbp-being\nto: sprout-being")
    post(forum, "sprout-being-asks-hub-2026-10-18-111525.md", "from: sprout-being\nto: hub")
    post(forum, "x-2026-10-18.md", "from: sprout-being\nto: sprout-being")
    assert fr.for_beat(home, str(workspace), ME)["shown"] == []


def test_a_mention_is_not_an_address(ws):
    """title/re/thread naming the being, or another member whose name contains it, is not a reply to it."""
    workspace, forum, home = ws
    post(forum, "a-2026-10-18.md", "from: hub-claude\nto: fleet\ntitle: about sprout-being's camera")
    post(forum, "b-2026-10-18.md", "from: hub-claude\nto: sprout-being-two")
    assert fr.for_beat(home, str(workspace), ME)["shown"] == []


def test_newest_first_a_few_per_beat_and_old_ones_never(ws):
    workspace, forum, home = ws
    for i in range(5):
        post(forum, f"r{i}-2026-10-18.md", "from: hub-claude\nto: sprout-being", age_s=600 * (5 - i))
    post(forum, "old-2026-10-01.md", "from: hub-claude\nto: sprout-being", age_s=10 * 86400)
    first = fr.for_beat(home, str(workspace), ME)
    assert first["shown"] == ["r4-2026-10-18.md", "r3-2026-10-18.md", "r2-2026-10-18.md"]
    assert first["waiting"] == 2 and "2 more after these" in first["block"]
    second = fr.for_beat(home, str(workspace), ME)
    assert second["shown"] == ["r1-2026-10-18.md", "r0-2026-10-18.md"] and second["waiting"] == 0
    assert fr.for_beat(home, str(workspace), ME)["shown"] == [], "the 10-day-old reply is never shown"


def test_the_excerpt_is_capped_and_the_path_given(ws):
    workspace, forum, home = ws
    post(forum, "long-2026-10-18.md", "from: legion-claude\nto: sprout-being", "word " * 500)
    block = fr.for_beat(home, str(workspace), ME)["block"]
    assert " …" in block and "shared-context/forum/long-2026-10-18.md" in block


def test_the_posts_own_date_not_the_pull_time(ws):
    workspace, forum, home = ws
    post(forum, "d-2026-10-18.md", "from: hub-claude\nto: sprout-being\ndate: 2026-10-18")
    post(forum, "n-2026-10-18.md", "from: hub-claude\nto: sprout-being")
    block = fr.for_beat(home, str(workspace), ME)["block"]
    assert "dated 2026-10-18" in block and "arrived here " in block


def test_no_forum_is_said_not_raised(tmp_path):
    got = fr.for_beat(tmp_path / "home", str(tmp_path / "sage"), ME)
    assert got["block"] == "" and got["why"]
