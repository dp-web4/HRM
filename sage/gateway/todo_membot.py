"""The todo membot: the being's own bookkeeping does what the being means. It has no judgment of its own.

dp, 2026-10-19: "we need a way for it to do that. should we have a mini-membot for the todo?"

MEASURED ON SPROUT, 2026-10-19. The being accepted that a goal was finished and tried to close it: "cross-check
camera grant off (scope-ef204d003703 expired undecided on Oct 17, closed without approval)". Nothing closed:
- memory_write appends, and the parser counts an item done only on [x], [done] or a `done:` section;
- a done line closed ONE matching item;
- 19 of the 79 open items it was shown every beat were differently worded copies of that one goal.

Within the hour the goal was back ("pending ~36 hours"), and the open copies grew from 17 to 19. The intent was
the being's; the mechanics failed it (dev research GOAL_ATTRACTORS.md).

What this does, and only this:
- **close by the being's act.** A done line whose text is `all: <words or id>` closes EVERY open item containing
  those words. The membot never closes anything by itself.
- **show the open list grouped.** Open items naming the same id (a scope request, a file, a device) are shown as
  one line with a count, so the being sees one goal once, not 19 times. The file is not changed.
- **say what it closed.** The view reports each recent `all:` closure and how many items it closed, so the
  being can check the effect of its act.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

ALL = re.compile(r"^(?:all|every)\s*:\s*(.+)$", re.I)
MIN_PHRASE = 4          # shorter phrases would close too much by accident ("ok", "dp")

# What makes two items "the same goal" for display: a scope request or a device they name. Script and file
# names are deliberately NOT ids here: they chain unrelated tasks together (measured on Sprout's 80 open items,
# file names merged 36 items, sandbox and shell-script tasks included, into one "camera" line).
_ID = re.compile(r"scope-[0-9a-f]{12}|/dev/[A-Za-z]+\d*", re.I)


def norm(s: str) -> str:
    s = re.sub(r"[`*_]", "", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip(" .:;-")


def close_all(items: Dict[str, Tuple[str, object]], body: str) -> Optional[Tuple[str, List[str]]]:
    """If `body` is an `all: <phrase>` directive, delete every open item containing the phrase and return
    (phrase, [closed item texts]). Return None when it is not a directive (the caller then treats the line as an
    ordinary done item)."""
    m = ALL.match((body or "").strip())
    if not m:
        return None
    phrase = norm(m.group(1))
    if len(phrase) < MIN_PHRASE:
        return (phrase, [])
    closed = [items.pop(k)[0] for k in [k for k in items if phrase in k]]
    return (phrase, closed)


def ids(text: str) -> List[str]:
    return [m.group(0).lower() for m in _ID.finditer(text or "")]


def group(open_items: List[Tuple[str, object]]) -> List[Tuple[str, object, int, Optional[str]]]:
    """[(shown text, since, count, id)], in the given order (newest first). Items that share ANY id belong to one
    group (connected through shared ids: "unbind /dev/video0 (scope-x)" joins both the /dev/video0 items and the
    scope-x items). A group is shown as its newest item, with the id most of its items name."""
    parent = list(range(len(open_items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_with: Dict[str, int] = {}
    item_ids = [ids(t) for t, _ in open_items]
    for i, ks in enumerate(item_ids):
        for k in ks:
            if k in first_with:
                a, b = find(first_with[k]), find(i)
                if a != b:
                    parent[max(a, b)] = min(a, b)       # the root is always the newest member
            else:
                first_with[k] = i
    members: Dict[int, List[int]] = {}
    for i in range(len(open_items)):
        members.setdefault(find(i), []).append(i)
    out = []
    for root in sorted(members):
        idx = members[root]
        counts: Dict[str, int] = {}
        for i in idx:
            for k in set(item_ids[i]):
                counts[k] = counts.get(k, 0) + 1
        label = max(sorted(counts), key=lambda k: counts[k]) if counts else None
        text, since = open_items[root]
        out.append((text, since, len(idx), label))
    return out


def closures_line(closures: List[Tuple[str, int, object]], limit: int = 3) -> List[str]:
    """Lines reporting the being's recent `all:` closures, newest last: what it asked to close and how many
    items that closed. Zero is reported too: an `all:` that matched nothing is worth knowing."""
    out = []
    for phrase, n, when in closures[-limit:]:
        at = f" ({when:%Y-%m-%d %H:%M} UTC)" if hasattr(when, "strftime") else ""
        out.append(f"- your `all: {phrase}`{at} closed {n} open item(s)"
                   + ("" if n else "; nothing open contained those words"))
    return out
