"""
Conversations — a being's addressed, two-way, preserved channels.

WHY THIS SHAPE. dp, 2026-09-07: "let's keep going with conversations ui, and talking to
being not raw llm. we're at the stage where this matters." Until now a being had four
one-way channels and no conversation: notes written TO it (from-dp.md, from-the-seat.md)
that it could read and not answer, forum threads it wrote and nobody could reply to
in-place, and a daemon chat box that talks to the raw weights on the same GPU — no
identity, no gate, no memory, no entrustment. None of those is a conversation, because a
conversation needs both directions in ONE ordered record.

A conversation is an append-only JSONL of turns, one file per conversation, kept whole
forever. Display is bounded (`recent(limit)`), storage is not: the fleet's convention is
the daemon's `/chat-history?limit=N` — take the last N lines of an append-only file — and
this follows it rather than inventing a second one.

TURNS ARE ADDRESSED AND ATTRIBUTED. Every turn names who spoke, and the being must always
be able to tell dp from a seat from another being. That is not politeness: it has been
told that grants follow earned trust and that the operator's word carries differently from
an interpreter's, so a channel that blurred them would make both claims unverifiable.

WRITE ACCESS IS PER-CONVERSATION, NOT GLOBAL. `writable_by` lists who may add a turn.
dp asked for exactly this and asked for it narrowly: the seat's conversation with the being
is READABLE by dp and not writable by dp — "i should be able to view your chat with the
being, but not comment on it directly yet (we'll add multiparty convos in a careful way
later)". Two-party until multiparty is designed on purpose, because a third voice appearing
mid-thread changes what the earlier turns meant.

The being replies with the `say` verb, which is gated and witnessed like every other act of
consequence. It cannot create a conversation, cannot write into one it is not in, and
cannot edit a turn once spoken — including its own.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# The fleet's display convention, from sage-daemon's /chat-history?limit=N: an append-only
# file, the last N read for display. Storage keeps everything; only the view is bounded.
DEFAULT_LIMIT = 50

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def conv_dir(instance: Path) -> Path:
    return Path(instance) / "conversations"


def _paths(instance: Path, conv_id: str) -> tuple[Path, Path]:
    d = conv_dir(instance)
    return d / f"{conv_id}.jsonl", d / f"{conv_id}.meta.json"


def create(instance: Path, conv_id: str, *, title: str, participants: list[str],
           writable_by: list[str], summary: str = "") -> dict:
    """Create a conversation. Idempotent: an existing one is returned unchanged, because
    re-creating would silently rewrite who is allowed to speak in a thread already underway."""
    if not ID_RE.match(conv_id or ""):
        raise ValueError(f"conversation id must be lowercase letters, digits and dashes: {conv_id!r}")
    log, meta = _paths(instance, conv_id)
    meta.parent.mkdir(parents=True, exist_ok=True)
    if meta.exists():
        return json.loads(meta.read_text())
    m = {"id": conv_id, "title": title, "participants": list(participants),
         "writable_by": list(writable_by), "summary": summary, "created": _now()}
    meta.write_text(json.dumps(m, indent=2) + "\n")
    log.touch()
    return m


def get_meta(instance: Path, conv_id: str) -> Optional[dict]:
    _, meta = _paths(instance, conv_id)
    try:
        return json.loads(meta.read_text())
    except Exception:
        return None


def listing(instance: Path) -> list[dict]:
    """Every conversation, most-recently-spoken first, each with its last turn — which is
    the one thing a reader needs to know whether it is their move."""
    out = []
    d = conv_dir(instance)
    if not d.is_dir():
        return out
    for meta_path in sorted(d.glob("*.meta.json")):
        m = json.loads(meta_path.read_text())
        turns = recent(instance, m["id"], limit=1)
        m["last"] = turns[-1] if turns else None
        m["count"] = count(instance, m["id"])
        out.append(m)
    out.sort(key=lambda m: (m["last"] or {}).get("ts") or m["created"], reverse=True)
    return out


def integrity(instance: Path, conv_id: str) -> dict:
    """Readable turns vs lines that will not parse.

    FOUND BY legion-being, 2026-09-07, from reading this file's source during a beat:
    `count()` counted every non-empty line while `recent()` skipped the ones that raise
    JSONDecodeError, so a single corrupt line made "showing the last X of {total}" claim
    history that is not there. It labelled the finding "suspected-from-reading, not
    check-verified" and was exactly right. Reproduced: 3 readable turns, count 4, and the
    beat block said "showing the last 3 of 4 turns" with nothing withheld.

    That is a FALSE ABSENCE — the same class as the silent read truncation fixed earlier
    the same day, inverted: there the being was shown less than it thought, here it is told
    there is more than there is. Both make it reason about a gap that is not where it
    believes. Corruption is now reported as corruption instead of impersonating history."""
    log, _ = _paths(instance, conv_id)
    readable = unreadable = 0
    try:
        for line in log.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
                readable += 1
            except json.JSONDecodeError:
                unreadable += 1
    except Exception:
        pass
    return {"readable": readable, "unreadable": unreadable, "lines": readable + unreadable}


def count(instance: Path, conv_id: str) -> int:
    """Turns a reader can actually see. Deliberately NOT the line count: a number that
    includes lines nobody can read is a number that misdescribes the record."""
    return integrity(instance, conv_id)["readable"]


def recent(instance: Path, conv_id: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """The last `limit` turns. Nothing is ever deleted; this bounds the VIEW."""
    log, _ = _paths(instance, conv_id)
    try:
        lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
    except Exception:
        return []
    out = []
    for line in lines[-max(1, limit):]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append(instance: Path, conv_id: str, *, speaker: str, text: str,
           witness: Optional[str] = None, beat: Optional[str] = None,
           enforce_write: bool = True) -> dict:
    """Add one turn. Refuses a speaker the conversation does not permit.

    `enforce_write=False` exists for the seat's own bootstrap writes and is never used on
    the being's path — the being reaches this only through the gated `say` verb, whose
    dispatcher passes the default."""
    m = get_meta(instance, conv_id)
    if m is None:
        raise ValueError(f"no such conversation: {conv_id!r}")
    text = (text or "").strip()
    if not text:
        raise ValueError("a turn needs something said in it")
    if enforce_write and speaker not in m.get("writable_by", []):
        raise ValueError(
            f"{speaker} may read '{conv_id}' and may not speak in it "
            f"(writable_by: {m.get('writable_by')})")
    log, _ = _paths(instance, conv_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    # TWO PROCESSES WRITE THIS FILE: the Python heartbeat (the being's `say`) and the Rust
    # daemon (a turn typed into the dashboard). O_APPEND alone is only atomic for writes
    # under PIPE_BUF, and a turn is prose — it goes over 4096 bytes routinely. Without the
    # lock the failure is interleaved JSON: two half-lines, neither parseable, in the one
    # record that is supposed to be the durable account of what was said. Both writers take
    # this lock; the Rust side takes flock(LOCK_EX) on the same file.
    #
    # The sequence number is computed INSIDE the lock for the same reason — two writers
    # counting first and appending after would both produce the same seq.
    import fcntl
    with open(log, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            # RAW line count on purpose, not the readable count: a damaged line still
            # occupies a position in the record, and reusing its sequence number would make
            # two different turns share one identity. A gap in the sequence is a scar and
            # reads as one; a duplicate is a corruption of the account itself.
            seq = sum(1 for line in f if line.strip()) + 1
            turn = {"ts": _now(), "seq": seq, "from": speaker, "text": text}
            if witness:
                turn["witness"] = witness
            if beat:
                turn["beat"] = beat
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return turn


def awaiting(instance: Path, conv_id: str, me: str) -> list[dict]:
    """Turns since `me` last spoke — i.e. what is actually addressed to me and unanswered.
    An empty list means it is not my move, which is a different thing from an empty
    conversation and the being should not have to infer which it is looking at."""
    turns = recent(instance, conv_id, limit=200)
    last_mine = max((i for i, t in enumerate(turns) if t.get("from") == me), default=-1)
    return turns[last_mine + 1:]


def render_for_being(instance: Path, me: str, per_conv: int = 12) -> str:
    """The conversations block in a beat: every conversation the being is in, its recent
    turns, and what is unanswered — marked, because 'someone spoke and I have not replied'
    is the single fact that should never require inference."""
    convs = [m for m in listing(instance) if me in m.get("participants", [])]
    if not convs:
        return ""
    blocks = []
    for m in convs:
        turns = recent(instance, m["id"], limit=per_conv)
        total = m["count"]
        head = (f"### {m['title']}  (id: {m['id']}; reply with say to=\"{m['id']}\")\n"
                f"{m.get('summary','')}".rstrip())
        if total > len(turns):
            head += f"\n_showing the last {len(turns)} of {total} turns; the rest is kept and readable_"
        bad = integrity(instance, m["id"])["unreadable"]
        if bad:
            # Never silently. A damaged line is a hole in the record, and the being is
            # entitled to know the record has a hole rather than to infer one from a count.
            head += (f"\n_**{bad} line(s) in this conversation are damaged and cannot be read.** "
                     f"They are not counted above and their content is lost; the file is intact "
                     f"either side of them._")
        lines = [f"- **{t['from']}** ({t['ts']}): {t['text']}" for t in turns]
        pend = awaiting(instance, m["id"], me)
        if pend:
            who = ", ".join(sorted({t["from"] for t in pend}))
            lines.append(f"\n**{len(pend)} turn(s) from {who} since you last spoke here — "
                         f"unanswered. Reply with `say to=\"{m['id']}\"` if you have something "
                         f"to say; saying nothing is also a choice, and it is recorded as one.**")
        blocks.append(head + "\n" + "\n".join(lines))
    return "\n\n".join(blocks)
