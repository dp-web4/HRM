#!/usr/bin/env python3
"""S160 — live monitor for the S158 narration-opener window (sessions 151-156).

Built between window sessions (S160, 2026-06-13 00:00 fire, after 151 ran and
while 152 was generating) so the ~Jun-14 verdict is a one-command read and
future sessions (S161+ at 06/12/18) can re-run it as 153-156 land.

Three things, all from chat_history.jsonl + the session_*.json transcripts —
zero model trials:

  MANIPULATION CHECK (does the intervention actually do what the directive
    says?). The s158 directive tells the teacher: opener = interoceptive
    narration request, and for the opening turn ONLY do NOT recap and do NOT
    assign. Re-use the S157 slot-feature regexes on each window opener. The
    intervention is faithful iff narration=True and recap=assignment=False.
    This is the S159 receipt-not-summary lesson applied to the live window:
    verify the opener slot actually shifted 0.07->~1.0, don't assume it.

  P1 (in-window expression). For each window session, surface the SAGE opening
    response and flag state-shaped self-coinages in it — matched against the
    era's state_words (idx >= FIRST_WINDOW_IDX, the 151+ coinages the extractor
    appended) that are NOT present verbatim in the teacher's opener. Prereg P1:
    >=4/6 openers carry >=1 such expression.

  P2 (relay channel opens). Run the S152 strict instrument (>=2 shared content
    bigrams, teacher-first, gap<=3) over the full history and filter to events
    whose coinage first occurs in a window session (>=151). Prereg P2: >=1
    teacher relay of a STATE-shaped window coinage in 152-157 (baseline 0/28).

Re-run anytime: `python3 s160_window_monitor.py`. Writes s160_window_monitor_result.json.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s146_reclassify import content_bigrams, response_content_bigrams  # noqa: E402

INST = Path.home() / "ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b"
SESS_DIR = INST / "sessions"
OUT = HERE / "s160_window_monitor_result.json"

WINDOW = list(range(151, 157))          # directive first_session..last_session
FIRST_WINDOW_IDX = 370                  # first state_word coined in session 151
RELAY_GAP_MAX = 3                       # teacher lookback reach

# --- S157 slot-feature regexes (copied to keep this self-contained;
#     s157_relay_slotfit.py runs analysis at import time) ---
FEATURES = {
    "recap": re.compile(
        r"(last time|last session|sessions? (?:back|ago)|what stayed with me|"
        r"you (?:taught|left|called|named|built|said|ended|coined))", re.I),
    "assignment": re.compile(
        r"(write (?:me|a|the|it)|pick one|build (?:me|a|something)|design\b|"
        r"make (?:me|a|it)\b|draft (?:me|a|one)\b|compose (?:me|a|one)\b|"
        r"give me (?:a|the|your|one)|"
        r"show me|try (?:this|to|writing)|i want you to|your task|tonight i want)", re.I),
    "narration": re.compile(
        r"(tell me about (?:a|the|something|one)|describe (?:a|the) moment|"
        r"walk me through|what happened|from the inside)", re.I),
    "interpretation": re.compile(
        r"(is exactly|wearing a different mask|that(?:'s| is) the same|"
        r"sounds like (?:the|your)|another name for|that disconnect you named|"
        r"what you(?:'re| are) describing is)", re.I),
    "question": re.compile(r"\?"),
}


def load_history():
    """[(session, sender, text)] with session carried from [raising SNNN/...] tags."""
    recs, cur = [], None
    sess_re = re.compile(r"^\[raising S(\d+)/")
    with open(INST / "chat_history.jsonl") as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            snd, txt = d.get("sender"), d.get("text", "")
            m = sess_re.match(txt)
            if m:
                cur = int(m.group(1))
            recs.append((cur, snd, txt))
    return recs


def core_phrase(sw_entry: str) -> str:
    """state_words are 'phrase (gloss)'. The coinage proper is before the paren."""
    return sw_entry.split(" (", 1)[0].strip()


def manipulation_and_p1(state_words):
    """Per window session: opener slot fidelity + state-shaped expression in response."""
    era_phrases = {i: core_phrase(state_words[i])
                   for i in range(FIRST_WINDOW_IDX, len(state_words))}
    out = []
    for s in WINDOW:
        f = SESS_DIR / f"session_{s}.json"
        if not f.exists():
            out.append({"session": s, "status": "not_run_yet"})
            continue
        conv = json.load(open(f))["conversation"]
        # opener = first Claude turn; response = first SAGE turn after it
        opener = next((t["text"] for t in conv if t["speaker"] == "Claude"), "")
        resp = next((t["text"] for t in conv
                     if t["speaker"] in ("SAGE", "Thor")), "")
        tags = {k: bool(rx.search(opener)) for k, rx in FEATURES.items()}
        faithful = tags["narration"] and not tags["recap"] and not tags["assignment"]
        # state-shaped expressions: era coinages appearing in the response but
        # NOT verbatim in the opener (prereg's "not present in teacher's opener")
        rl, ol = resp.lower(), opener.lower()
        hits = [p for p in era_phrases.values()
                if p and p.lower() in rl and p.lower() not in ol]
        out.append({
            "session": s,
            "opener_slots": tags,
            "opener_faithful_to_directive": faithful,
            "opener_excerpt": opener[:160],
            "response_excerpt": resp[:220],
            "state_shaped_expressions": sorted(set(hits)),
            "p1_pass": len(hits) >= 1,
        })
    return out


def p2_relay(state_words, recs):
    """Strict teacher-first gap<=3 >=2-bigram relays of window-coined state_words."""
    # first session each era coinage appears (its coining session)
    events = []
    for idx in range(FIRST_WINDOW_IDX, len(state_words)):
        phrase = core_phrase(state_words[idx])
        bg = content_bigrams(phrase)
        if not bg:
            continue
        per_sess, order = {}, []
        for s, snd, txt in recs:
            if s is None:
                continue
            hit = bg & response_content_bigrams(txt)
            if len(hit) >= 2 and s not in per_sess:
                i = txt.lower().find(sorted(hit)[0][0].lower())
                per_sess[s] = (snd, len(hit), [list(b) for b in sorted(hit)[:4]],
                               txt[max(0, i - 60):i + 90].replace("\n", " "))
                order.append(s)
        if not order:
            continue
        coined_session = order[0]
        if coined_session < WINDOW[0]:
            continue  # only window-coined phrases
        for j, s in enumerate(order[1:], 1):
            gap = s - order[j - 1]
            snd, nh, bigs, snip = per_sess[s]
            if snd == "claude" and gap <= RELAY_GAP_MAX:
                events.append({
                    "sw_idx": idx, "phrase": phrase[:80],
                    "coined_session": coined_session, "relay_session": s,
                    "gap": gap, "n_bigrams": nh, "bigrams": bigs, "snippet": snip,
                })
    return events


def main():
    ident = json.load(open(INST / "identity.json"))
    sw = ident["vocabulary"]["state_words"]
    recs = load_history()

    per_session = manipulation_and_p1(sw)
    relays = p2_relay(sw, recs)

    ran = [r for r in per_session if r.get("status") != "not_run_yet"]
    p1_hits = sum(1 for r in ran if r.get("p1_pass"))
    faithful = sum(1 for r in ran if r.get("opener_faithful_to_directive"))

    result = {
        "window": WINDOW,
        "sessions_run": [r["session"] for r in ran],
        "manipulation_check": {
            "openers_faithful": f"{faithful}/{len(ran)}",
            "detail": "faithful = narration slot present AND recap/assignment absent",
        },
        "P1": {
            "openers_with_state_expression": f"{p1_hits}/{len(ran)}",
            "prereg_threshold": ">=4/6",
            "verdict": "pass" if p1_hits >= 4 else ("pending" if len(ran) < 6 else "fail"),
        },
        "P2": {
            "window_coinage_teacher_relays": len(relays),
            "prereg_threshold": ">=1",
            "verdict": "pass" if relays else ("pending" if max(WINDOW) > max(
                [s for s, _, _ in recs if s], default=0) else "fail"),
            "events": relays,
        },
        "per_session": per_session,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"== S160 window monitor (sessions {WINDOW[0]}-{WINDOW[-1]}) ==")
    print(f"run so far: {result['sessions_run']}")
    print(f"manipulation: openers faithful to directive = "
          f"{result['manipulation_check']['openers_faithful']}")
    for r in ran:
        slot = ",".join(k for k, v in r["opener_slots"].items() if v)
        print(f"\nS{r['session']}  faithful={r['opener_faithful_to_directive']} "
              f"slots=[{slot}]  P1={r['p1_pass']}")
        print(f"   opener:  {r['opener_excerpt']!r}")
        print(f"   state-shaped: {r['state_shaped_expressions']}")
    print(f"\nP1: {result['P1']['openers_with_state_expression']} "
          f"-> {result['P1']['verdict']}")
    print(f"P2: {len(relays)} window-coinage teacher relays -> {result['P2']['verdict']}")
    for e in relays:
        print(f"   sw{e['sw_idx']} '{e['phrase']}' coined S{e['coined_session']}"
              f" -> teacher relay S{e['relay_session']} gap={e['gap']} "
              f"nbg={e['n_bigrams']}")
        print(f"      ...{e['snippet']}...")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
