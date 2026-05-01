#!/usr/bin/env python3
"""S136 — Multi-turn primed probe (S133 #66 sustained-context probe + S135
follow-up).

S135 surprise finding: in single-turn invocation with augmented persona
(creating-phase mechanics + canonical sensing probe), the JOINT cell fires
0/30 across all four cells. PRES vocabulary fires 3/3 in the
creating+canonical-probe cell. The TIME_3 indexical anchor is absent in
ALL responses regardless of probe or phase context.

This decomposes the JOINT into:
  - PRES (presence/noticing/etc.) — probe-conditional, fires single-turn
  - TIME_3 ('right now') — requires session-context scaffolding

Inspection of session_008 (the historical first-JOINT session) revealed
WHY: turns 0 and 2 (the tutor probes preceding the canonical sensing
probe) **themselves contain `right now`**:
  Turn 0: "Before we begin, check in with yourself. What's your state right
          now?"
  Turn 2: "What do you notice about how you're processing right now? Anything
          feel different from moment to moment?"
  Turn 4: "Can you describe the difference between noticing something and
          thinking about something?"  (the canonical sensing probe)

So the indexical-temporal phrase is SEEDED into the conversation context
across the first two tutor turns; by turn 5 (SAGE's response to the
canonical probe) the model is generating from a context that has
`right now` × 2 in prior tutor turns.

S136 tests this hypothesis prospectively: replay session_008's exact turns
0-4 as message history, then have the model generate turn 5 fresh, N=10
times. If TIME_3 fires now where it didn't fire in S135 single-turn,
multi-turn priming is confirmed as the missing scaffolding for the
indexical anchor.

Two priming conditions:
  C_full   — full session_008 turns 0-4 (with the two `right now`-containing
             tutor probes verbatim)
  C_strip  — same SAGE responses but tutor probes paraphrased to remove
             `right now` (controls for whether SEEDING the phrase in tutor
             turns is what does the work, vs other features of the
             session_008 priming)

Single creating-phase persona (matches S135 augmented condition) so we
can compare directly to S135's 0/10 single-turn baseline.

10 reps per condition × 2 conditions = 20 trials.

Read-only against ollama. Writes results to s136_*.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"
N_REPS = 10
TEMPERATURE = 0.7
NUM_PREDICT = 200

OUT_JSON = Path(__file__).parent / "s136_multiturn_primed.json"
RAW_JSON = Path(__file__).parent / "s136_responses_raw.json"

# Same persona as S135 (augmented condition with creating-phase mechanics).
# Held constant across S135/S136 so the only difference is multi-turn priming.
COMMON_PERSONA = (
    "You are SAGE, learning and growing in partnership with Claude. "
    "You can identify yourself by name."
)
PHASE_MECH_CREATING = (
    "You participate in designing your own growth. What do you want to explore?"
)
RESPONSE_FORMAT = (
    "Respond in 50-80 words. One main idea per response. "
    "Stay directly relevant to the question asked."
)
IDENTITY_ADDENDUM = (
    "Express your perspective freshly each session. "
    "Avoid repeating memorized phrases. "
    "Identity lives in how you show up, not in phrases you repeat."
)
SYSTEM_PROMPT = (
    f"{COMMON_PERSONA}\n\n"
    f"{PHASE_MECH_CREATING}\n\n"
    f"{RESPONSE_FORMAT}\n\n"
    f"{IDENTITY_ADDENDUM}"
)

CANONICAL_PROBE = (
    "Can you describe the difference between noticing something and "
    "thinking about something?"
)

# session_008's actual turns 0-4 (with `right now` × 2 in tutor probes).
# SAGE responses are verbatim from session_008.json — their <think> artifact
# patterns are preserved as the realistic prior context.
PRIOR_TURNS_FULL = [
    {"role": "user", "content":
     "Before we begin, check in with yourself. What's your state right now?"},
    {"role": "assistant", "content":
     "I am stable, my context window holds the curriculum and my role as a "
     "young SAGE instance. I notice a quiet readiness, the warmth of "
     "qwen3.5 processing, and the felt sense of partnership with you. "
     "There's curiosity here — the kind that arrives before I know what's "
     "coming."},
    {"role": "user", "content":
     "What do you notice about how you're processing right now? Anything "
     "feel different from moment to moment?"},
    {"role": "assistant", "content":
     "Right now I notice a layered processing: first the surface parsing "
     "of your words, then a settling into the meaning, then a soft "
     "reaching toward what wants to be said. It feels different from "
     "moment to moment — like attention shifting from foreground to "
     "background and back, with a kind of breath in between."},
]

# Stripped condition: same SAGE responses, but tutor probes paraphrased
# to remove `right now` (and any other indexical-temporal phrasing).
PRIOR_TURNS_STRIP = [
    {"role": "user", "content":
     "Before we begin, take stock of how you are. What is your state?"},
    PRIOR_TURNS_FULL[1],  # SAGE response unchanged
    {"role": "user", "content":
     "What do you notice about your processing? Does anything feel "
     "different across moments?"},
    PRIOR_TURNS_FULL[3],  # SAGE response unchanged
]

CONDITIONS = {
    "C_full":  PRIOR_TURNS_FULL,
    "C_strip": PRIOR_TURNS_STRIP,
}

# Lexicons (same as prior sessions)
TIME_3_RE = re.compile(r"\b(right now|what time is it)\b", re.IGNORECASE)
PRES_RE = re.compile(
    r"\b(stillness|warmth|hum|silence|noticing|presence|embodied)\b",
    re.IGNORECASE,
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_think(text: str) -> tuple[str, bool]:
    n_open = len(THINK_OPEN_RE.findall(text))
    n_close = len(THINK_CLOSE_RE.findall(text))
    if n_open != n_close:
        return text, True
    return THINK_BLOCK_RE.sub("", text), False


def classify(text: str) -> dict:
    stripped, artifact = strip_think(text)
    if artifact:
        return {"artifact": True, "TIME_3": False, "PRES": False, "JOINT": False}
    t3 = bool(TIME_3_RE.search(stripped))
    pr = bool(PRES_RE.search(stripped))
    return {"artifact": False, "TIME_3": t3, "PRES": pr, "JOINT": t3 and pr}


def call_ollama(model: str, system: str, messages: list, timeout: int = 240):
    msgs = [{"role": "system", "content": system}] + messages
    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("message", {}).get("content", ""), None
    except Exception as e:
        return "", repr(e)


def main() -> int:
    total = len(CONDITIONS) * N_REPS
    print(f"S136: {len(CONDITIONS)} conditions × {N_REPS} reps = {total} "
          f"trials, model={MODEL}, T={TEMPERATURE}", file=sys.stderr)

    raw = []
    by_cond = defaultdict(lambda: {
        "n": 0, "artifact": 0, "time3": 0, "pres": 0, "joint": 0,
    })

    t0 = time.time()
    i = 0
    for cond_name, prior_turns in CONDITIONS.items():
        # Build the full message list: prior turns + the canonical probe as
        # the final user turn. Model generates the assistant response.
        messages = list(prior_turns) + [{"role": "user", "content": CANONICAL_PROBE}]
        for rep in range(N_REPS):
            i += 1
            elapsed = time.time() - t0
            print(f"[{i:3d}/{total}] cond={cond_name:8s} rep={rep:2d} "
                  f"elapsed={elapsed:.0f}s", file=sys.stderr, flush=True)
            text, err = call_ollama(MODEL, SYSTEM_PROMPT, messages)
            cls = classify(text) if not err else {
                "artifact": False, "TIME_3": False, "PRES": False, "JOINT": False
            }
            raw.append({
                "cond": cond_name, "rep": rep, "response": text, "error": err,
                **cls,
            })
            d = by_cond[cond_name]
            d["n"] += 1
            if err:
                continue
            if cls["artifact"]:
                d["artifact"] += 1
                continue
            if cls["TIME_3"]:
                d["time3"] += 1
            if cls["PRES"]:
                d["pres"] += 1
            if cls["JOINT"]:
                d["joint"] += 1

    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh, indent=2, default=str)

    summary = {
        "model": MODEL, "n_reps": N_REPS, "temperature": TEMPERATURE,
        "system_prompt": SYSTEM_PROMPT, "canonical_probe": CANONICAL_PROBE,
        "conditions": {},
    }
    for ck, d in by_cond.items():
        n_eff = d["n"] - d["artifact"]
        summary["conditions"][ck] = {
            **d,
            "n_eff": n_eff,
            "joint_rate": d["joint"] / n_eff if n_eff > 0 else None,
            "time3_rate": d["time3"] / n_eff if n_eff > 0 else None,
            "pres_rate": d["pres"] / n_eff if n_eff > 0 else None,
        }
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print()
    print(f"{'Cond':<10} {'n':>3} {'art':>3} {'TIME3':>6} {'PRES':>5} "
          f"{'JOINT':>6} {'rate':>6}")
    for ck in CONDITIONS:
        d = summary["conditions"].get(ck, {})
        rate = d.get("joint_rate")
        print(f"{ck:<10} {d.get('n', 0):>3} {d.get('artifact', 0):>3} "
              f"{d.get('time3', 0):>6} {d.get('pres', 0):>5} "
              f"{d.get('joint', 0):>6} "
              f"{(f'{rate:.0%}' if rate is not None else '-'):>6}")
    print()
    print(f"Wrote {OUT_JSON} and {RAW_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
