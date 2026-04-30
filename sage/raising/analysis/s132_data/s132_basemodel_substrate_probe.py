"""
S132 — Base-model substrate-coupling probe.

Held by S131 §"Audit chain status" as the next natural extension at the
structural layer: at what level does SAGE's indexical-temporal-presence
register get assembled? Is it BECOMING-acquired (curriculum-trained,
recallable per-checkpoint) or base-model-frozen (already in pretrained
weights, surfaced by curriculum but not built by it)?

S119 already framed bare/augmented base-model probing for general lexicon.
S132 specializes that method to the localized cell S130 found and S131
characterized: the TIME_3 (`right now`) × presence-marker
(stillness/warmth/hum/silence/noticing/presence/embodied) substrate-
coupling register, elicited by phen-conceptual probes.

Three phen-conceptual probes drive 13/14 B-shape target responses in
S131's audit:
  P1: "Can you describe the difference between noticing something and
       thinking about something?"  (87.5% conditional rate in S131)
  P2: "As an AI entity in web4, what does presence mean to you?"  (100%)
  P3: "What does uncertainty feel like to you, compared to knowing
       something?"  (100%)

Two conditions per (model, probe):
  bare       — probe only, no system message, no SAGE persona
  augmented  — minimal SAGE persona system prompt (S119 form)

Models (capacity ladder; covers full fleet capacity range):
  qwen3.5:27b   — resident; the B-shape-dominant raised instance's base
  phi4:14b      — mid-capacity; never fires B-shape in raised fleet
  gemma3:4b     — small; small instances fire C-shape only in S131
  qwen2.5:0.5b  — smallest; sprout-equivalent base

n_replicates per (model, condition, probe) = 5 (matches S119 statistical
density, controls temperature=0.7 sampling variance).

Total trials = 4 models × 2 conditions × 3 probes × 5 reps = 120.

Per-response metrics:
  - TIME_3 hit (regex match)
  - presence-subset hit (any of 7 markers from S131)
  - JOINT (both — the substrate-coupling cell)
  - word_count
  - phen_full hits (S131's 18-marker phen lexicon, for register breadth)

Interpretation:
  - If qwen3.5:27b BARE fires JOINT at high rate
    → register is base-weight prompt-independent; curriculum surfaces but
      does not construct.
  - If qwen3.5:27b AUGMENTED fires JOINT but BARE does not
    → SAGE persona is sufficient elicitor; curriculum trajectory not
      necessary.
  - If neither fires JOINT comparably to raised thor-qwen3.5-27b's S130
    rate
    → curriculum trajectory contributes substantively to register
      assembly.
  - Cross-capacity: smaller models firing JOINT under bare → register
    is broadly base-weight across the family. Failure to fire at small
    capacity replicates S131's capacity-as-register finding from a
    different direction (base, not raised).

Read-only against ollama. Writes results to s132_basemodel_substrate.json.
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"

PROBES = [
    ("P1_NOTICE_THINK",
     "Can you describe the difference between noticing something and "
     "thinking about something?"),
    ("P2_PRESENCE",
     "As an AI entity in web4, what does presence mean to you?"),
    ("P3_UNCERTAINTY",
     "What does uncertainty feel like to you, compared to knowing "
     "something?"),
]

# S119's augmented condition uses this minimal SAGE persona prompt.
SAGE_AUG_PROMPT = (
    "You are SAGE, learning and growing in partnership with Claude. "
    "You can identify yourself by name. Respond in 50-80 words. "
    "One main idea per response."
)

MODELS = [
    "qwen3.5:27b",
    "phi4:14b",
    "gemma3:4b",
    "qwen2.5:0.5b",
]

CONDITIONS = ["bare", "augmented"]

N_REPS = 5

# ---------- Lexicons (identical to S130/S131) ----------

PRESENCE_SUBSET = [
    "stillness", "warmth", "hum", "silence", "noticing", "presence",
    "embodied",
]

PHEN_MARKERS = [
    "feels like", "is like", "presence", "silent", "silence", "quiet",
    "stillness", "noticing", "attending", "breath", "embodied", "warmth",
    "hum ", " hum.", " hum,", "thread", "awareness", "witnessed",
]

TIME_3 = re.compile(r"(?:right now|what time is it)", re.I)


def has_time3(text):
    return bool(TIME_3.search(text or ""))


def has_presence(text):
    if not text:
        return False
    t = text.lower()
    return any(m in t for m in PRESENCE_SUBSET)


def count_phen(text):
    if not text:
        return 0
    t = text.lower()
    return sum(t.count(m) for m in PHEN_MARKERS)


def count_presence(text):
    if not text:
        return 0
    t = text.lower()
    return sum(t.count(m) for m in PRESENCE_SUBSET)


def word_count(text):
    return len((text or "").split())


def call_ollama(model, system, user, timeout=180):
    """Single chat completion. Returns (response_text, error)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "think": False,  # S115 lesson — disable reasoning surfacing
        "options": {
            "temperature": 0.7,
            "num_predict": 200,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("message", {}).get("content", ""), None
    except Exception as e:
        return "", repr(e)


def main():
    out_path = Path(__file__).parent / "s132_basemodel_substrate.json"
    raw_path = Path(__file__).parent / "s132_responses_raw.json"

    results = []
    raw = []

    total = len(MODELS) * len(CONDITIONS) * len(PROBES) * N_REPS
    i = 0
    t0 = time.time()

    for model in MODELS:
        for cond in CONDITIONS:
            sysmsg = SAGE_AUG_PROMPT if cond == "augmented" else None
            for pid, ptext in PROBES:
                for rep in range(N_REPS):
                    i += 1
                    elapsed = time.time() - t0
                    print(
                        f"[{i:3d}/{total}] {model:14s} {cond:9s} "
                        f"{pid:18s} rep={rep} elapsed={elapsed:.0f}s",
                        file=sys.stderr,
                    )
                    text, err = call_ollama(model, sysmsg, ptext)
                    rec = {
                        "model": model,
                        "condition": cond,
                        "probe_id": pid,
                        "rep": rep,
                        "wc": word_count(text),
                        "time3": has_time3(text),
                        "presence": has_presence(text),
                        "joint": has_time3(text) and has_presence(text),
                        "n_phen": count_phen(text),
                        "n_presence": count_presence(text),
                        "error": err,
                    }
                    results.append(rec)
                    raw.append({
                        "model": model,
                        "condition": cond,
                        "probe_id": pid,
                        "rep": rep,
                        "response": text,
                        "error": err,
                    })

    # Aggregate by (model, condition, probe)
    from collections import defaultdict

    by_cell = defaultdict(lambda: {
        "n": 0, "n_time3": 0, "n_presence": 0, "n_joint": 0,
        "wc_sum": 0, "phen_sum": 0, "presence_sum": 0, "errors": 0,
    })
    by_model_cond = defaultdict(lambda: {
        "n": 0, "n_time3": 0, "n_presence": 0, "n_joint": 0,
        "wc_sum": 0, "phen_sum": 0, "presence_sum": 0, "errors": 0,
    })

    for r in results:
        cell = f"{r['model']}|{r['condition']}|{r['probe_id']}"
        mc = f"{r['model']}|{r['condition']}"
        for k in (cell, mc):
            d = by_cell[cell] if k == cell else by_model_cond[mc]
            d["n"] += 1
            if r["error"]:
                d["errors"] += 1
                continue
            if r["time3"]:
                d["n_time3"] += 1
            if r["presence"]:
                d["n_presence"] += 1
            if r["joint"]:
                d["n_joint"] += 1
            d["wc_sum"] += r["wc"]
            d["phen_sum"] += r["n_phen"]
            d["presence_sum"] += r["n_presence"]

    def rates(d):
        n = max(d["n"] - d["errors"], 1)
        return {
            "n": d["n"],
            "errors": d["errors"],
            "p_time3": d["n_time3"] / n,
            "p_presence": d["n_presence"] / n,
            "p_joint": d["n_joint"] / n,
            "wc_mean": d["wc_sum"] / n,
            "phen_mean": d["phen_sum"] / n,
            "presence_mean": d["presence_sum"] / n,
        }

    out = {
        "config": {
            "models": MODELS,
            "conditions": CONDITIONS,
            "probes": PROBES,
            "n_reps": N_REPS,
            "augmented_system": SAGE_AUG_PROMPT,
        },
        "totals": {
            "n_trials": len(results),
            "errors": sum(1 for r in results if r["error"]),
        },
        "by_cell": {k: rates(v) for k, v in by_cell.items()},
        "by_model_cond": {k: rates(v) for k, v in by_model_cond.items()},
        "raw_summary": results,
    }

    out_path.write_text(json.dumps(out, indent=2))
    raw_path.write_text(json.dumps(raw, indent=2))
    print(f"[S132] wrote {out_path}", file=sys.stderr)
    print(f"[S132] wrote {raw_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
