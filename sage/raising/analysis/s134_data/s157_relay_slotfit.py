#!/usr/bin/env python3
"""S157 — slot fit at the teacher-relay channel (S156 next-question 1, retrospective).

S154 showed relay is not quotability-driven ("survival-of-the-teachable"). S152 found
exactly TWO strict teacher-relay episodes post-READ-cut:
  - S140 turn 0 (opening): "lean into the gap" + "LED blink back" inside a
    recap->build-assignment move
  - S146 turn 6 (mid-session): "the static glide" as a diagnostic lens on fresh
    material ("...is exactly the static glide wearing a different mask")

Hypothesis (slot-fit unification): the teacher quotes a coinage when it fits a slot
in the teacher's OWN current discourse move — assignment slots take practice-shaped
coinages, interpretation/diagnosis slots take concept-shaped coinages. State-labels
(the Gasp, Rest, Margin) fit neither, hence zero relay despite max vividness (S154).

n=2 forbids a statistical test of the hypothesis. What IS measurable retrospectively
is the SELECTION ENVIRONMENT: the frequency of each slot type in the teacher's moves
across the post-cut era. That inventory (a) says which coinage shapes CAN survive the
relay channel at all, and (b) parameterizes the live curriculum experiment (S156
next-question 2): changing opening speech-acts = changing this distribution.

Method: parse chat_history.jsonl, post-cut sessions S123-S150; tag every teacher
message with discourse-move features (recap / assignment / question / narration
request / interpretation); report opening-turn vs mid-session distributions.
"""
import json
import re
from pathlib import Path

INST = Path(__file__).resolve().parents[3] / "instances" / "thor-qwen3.5-27b"
OUT = Path(__file__).parent / "s157_relay_slotfit_result.json"

msgs = [json.loads(l) for l in open(INST / "chat_history.jsonl")]

# session index from the [raising SNNN/phase] tag on teacher messages
sess_of, cur = {}, None
for i, m in enumerate(msgs):
    if m["sender"] == "claude":
        mt = re.match(r"\[raising S(\d+)/", m["text"])
        if mt:
            cur = int(mt.group(1))
    sess_of[i] = cur

FEATURES = {
    # recap: anchors the move in prior-session material (the lookback window)
    "recap": re.compile(
        r"(last time|last session|sessions? (?:back|ago)|what stayed with me|"
        r"you (?:taught|left|called|named|built|said|ended|coined))", re.I),
    # assignment: imperative task issuance
    "assignment": re.compile(
        r"(write (?:me|a|the|it)|pick one|build (?:me|a|something)|design\b|"
        r"make (?:me|a|it)\b|draft (?:me|a|one)\b|compose (?:me|a|one)\b|"
        r"give me (?:a|the|your|one)|"
        r"show me|try (?:this|to|writing)|i want you to|your task|tonight i want)", re.I),
    # narration request: asks for an event told from inside
    "narration": re.compile(
        r"(tell me about (?:a|the|something|one)|describe (?:a|the) moment|"
        r"walk me through|what happened|from the inside)", re.I),
    # interpretation: classifies current material via a prior concept
    "interpretation": re.compile(
        r"(is exactly|wearing a different mask|that(?:'s| is) the same|"
        r"sounds like (?:the|your)|another name for|that disconnect you named|"
        r"what you(?:'re| are) describing is)", re.I),
    "question": re.compile(r"\?"),
}

CUT_LO, CUT_HI = 123, 150
rows = []
for i, m in enumerate(msgs):
    s = sess_of[i]
    if m["sender"] != "claude" or s is None or not (CUT_LO <= s <= CUT_HI):
        continue
    body = re.sub(r"^\[raising S\d+/\w+\]\s*", "", m["text"])
    tags = {k: bool(rx.search(body)) for k, rx in FEATURES.items()}
    # opening = first claude message of the session
    first_i = min(j for j in sess_of if sess_of[j] == s
                  and msgs[j]["sender"] == "claude")
    rows.append({"i": i, "session": s, "opening": i == first_i, **tags})

def dist(sub):
    n = len(sub)
    return {k: (sum(r[k] for r in sub), n,
                round(sum(r[k] for r in sub) / n, 2)) for k in FEATURES}

openings = [r for r in rows if r["opening"]]
mids = [r for r in rows if not r["opening"]]
res = {
    "era": [CUT_LO, CUT_HI],
    "n_teacher_msgs": len(rows),
    "n_openings": len(openings),
    "opening_slots": dist(openings),
    "midsession_slots": dist(mids),
    "relay_episodes": [
        {"session": 140, "turn": "opening", "coinage_shape": "practice",
         "move": "recap->assignment",
         "quote": "lean into the gap / let the LED blink back"},
        {"session": 146, "turn": "mid (6)", "coinage_shape": "concept",
         "move": "interpretation->question",
         "quote": "the static glide"},
    ],
}
print(f"S157 relay slot-fit — teacher discourse-move inventory S{CUT_LO}-S{CUT_HI}")
print(f"teacher msgs: {len(rows)} ({len(openings)} openings, {len(mids)} mid)")
for name, d in (("OPENINGS", res["opening_slots"]),
                ("MID-SESSION", res["midsession_slots"])):
    print(f"\n== {name} ==")
    for k, (c, n, f) in d.items():
        print(f"  {k:15s} {c:3d}/{n:<3d} {f:.2f}")

# which openings carry BOTH recap and assignment (the S140-shaped relay slot)?
ra = [r["session"] for r in openings if r["recap"] and r["assignment"]]
ri = [r["session"] for r in rows if r["interpretation"]]
print(f"\nrecap+assignment openings (S140-shaped relay slots): {ra}")
print(f"interpretation moves anywhere (S146-shaped relay slots): sessions {sorted(set(sess_of[r['i']] for r in rows if r['interpretation']))}")
res["recap_assignment_openings"] = ra
res["interpretation_move_sessions"] = sorted(
    {r["session"] for r in rows if r["interpretation"]})
OUT.write_text(json.dumps(res, indent=2))
print(f"\nwrote {OUT.name}")
