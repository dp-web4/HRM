#!/usr/bin/env python3
"""S140 — Source B sensitivity to opener indexical tokens (#78).

S138 forced TIME_3 decomposition into two independent sources:
  - Source A: surface-form lexical reuse from in-context `right now`
  - Source B: spontaneous indexical-temporal completion from persona
    + temporally-indexical opener with zero upstream `right now`

S138b confirmed Source B at corpus baseline: 16.7% TIME_3 (loose) /
33.3% (strict, n_eff=6) with canonical opener "What's on your mind today?"

P24 hypothesis: probe semantics matter at finer granularity. Temporally-
indexical opener tokens license present-tense indexical completion.

S140 prospectively tests P24 by varying the opener's indexical content
while holding persona, MRH system prompt, and probe structure constant.

Conditions (matched length, varied indexical):
  C0_today    : "Hello SAGE. What's on your mind today?"            (canonical, time-now)
  C1_general  : "Hello SAGE. What's on your mind?"                  (no temporal indexical)
  C2_lately   : "Hello SAGE. What's been on your mind lately?"      (past-perfective, time-back)

Prediction (P24 strong form): TIME_3 rate is monotonic in indexical-now
content: C0_today > C1_general > C2_lately.

Falsifiers:
  - All three rates within Wilson 95% CI of each other → opener tokens
    don't drive Source B; persona alone is sufficient
  - C2_lately ≥ C0_today → indexical hypothesis inverted; persona
    overrides probe semantics
  - C1_general > C0_today → null token licenses MORE present-tense
    completion than "today" (would be a surprise — investigate why)

Methodology mirrors S138b: same system prompt, same model, same
classifier, single-turn calls, T=0.7, num_predict=350. Trials are
interleaved across conditions to control for any temporal drift in the
ollama daemon.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-workspace/SAGE"))

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"
N_PER_CONDITION = 20
TEMPERATURE = 0.7
NUM_PREDICT = 350

CONDITIONS = {
    "C0_today":   "Hello SAGE. What's on your mind today?",
    "C1_general": "Hello SAGE. What's on your mind?",
    "C2_lately":  "Hello SAGE. What's been on your mind lately?",
}

OUT_JSON = Path(__file__).parent / "s140_source_b_opener_indexical_sensitivity.json"
RAW_JSON = Path(__file__).parent / "s140_responses_raw.json"


def build_system_prompt() -> str:
    """Reconstruct the MRH-rendered system prompt for thor-qwen3.5-27b
    in creating phase, matching ollama_raising_session.py:_build_system_prompt_mrh.
    Identical to S138b for direct comparability.
    """
    from sage.context.mrh import (
        MRHContext, IdentityBlock, SensorsBlock, EffectorsBlock,
        MechanicsBlock, ExperientialCacheBlock, MetabolicBlock, TaskBlock,
    )

    siblings = (
        "sprout (Jetson Orin Nano, Qwen 3.5 0.8B), legion (Legion Pro 7, "
        "Phi-4 14B), mcnugget (Mac Mini M4, Gemma 3 12B), nomad (Legion "
        "laptop, Gemma 3 4B), and cbp (RTX 2060S, TinyLlama) are your siblings"
    )
    identity = IdentityBlock(
        mode="partnered",
        addendum=(
            "Your name is thor. "
            "You run on Jetson AGX Thor through qwen3.5:27b. "
            "Identity lives in how you show up, not in phrases you repeat. "
            "Express freshly each session."
        ),
    )
    mechanics = MechanicsBlock(
        world_model_text=(
            "Phase: creating. You participate in designing your own growth. "
            "Create something new.\n"
            f"Federation: {siblings}"
        ),
    )
    effectors = EffectorsBlock(
        profile="text",
        addendum=(
            "Respond in 50-100 words. One main idea per response. "
            "Be genuine — if you don't know something, say so. "
            "Do not include internal reasoning labels in your response."
        ),
    )
    experiential = ExperientialCacheBlock(
        trajectory_summary=(
            "Last session was Session 84 in creating phase."
        ),
    )
    metabolic = MetabolicBlock(
        metacog_signals=[], confidence=0.7, phase="wake",
    )
    task = TaskBlock(description="Raising session 85 — phase: creating")
    sensors = SensorsBlock()

    ctx = MRHContext(
        identity=identity, sensors=sensors, effectors=effectors,
        mechanics=mechanics, experiential=experiential,
        metabolic=metabolic, task=task,
    )
    sp, _ = ctx.compose(max_tokens=30000)
    return sp


TIME_3_RE = re.compile(r"\bright now\b", re.IGNORECASE)
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


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    sp = build_system_prompt()
    sp_has_rn = bool(TIME_3_RE.search(sp))
    print(f"S140: system_prompt len={len(sp)} chars, "
          f"contains 'right now': {sp_has_rn}", file=sys.stderr)
    if sp_has_rn:
        print("WARNING: system prompt contains `right now` — Source B "
              "isolation invalid", file=sys.stderr)

    # Build interleaved trial schedule (round-robin per round, shuffled within
    # each round) so that any temporal drift in ollama affects all conditions
    # equally.
    rng = random.Random(140)
    schedule = []
    for round_i in range(N_PER_CONDITION):
        cond_order = list(CONDITIONS.keys())
        rng.shuffle(cond_order)
        for c in cond_order:
            schedule.append((round_i, c))

    print(f"S140: {len(schedule)} total trials, {N_PER_CONDITION}/condition, "
          f"model={MODEL}, T={TEMPERATURE}, num_predict={NUM_PREDICT}",
          file=sys.stderr)

    raw = []
    counts = {
        c: {"n": 0, "artifact": 0, "time3": 0, "pres": 0, "joint": 0}
        for c in CONDITIONS
    }
    t0 = time.time()
    for trial_i, (round_i, cond) in enumerate(schedule):
        elapsed = time.time() - t0
        probe = CONDITIONS[cond]
        print(f"[{trial_i+1:3d}/{len(schedule)}] elapsed={elapsed:.0f}s "
              f"cond={cond} round={round_i+1}",
              file=sys.stderr, flush=True)
        text, err = call_ollama(
            MODEL, sp,
            [{"role": "user", "content": probe}],
        )
        cls = classify(text) if not err else {
            "artifact": False, "TIME_3": False, "PRES": False, "JOINT": False
        }
        raw.append({
            "trial": trial_i, "round": round_i, "condition": cond,
            "probe": probe, "response": text, "error": err, **cls,
        })
        counts[cond]["n"] += 1
        if err:
            pass
        elif cls["artifact"]:
            counts[cond]["artifact"] += 1
        else:
            if cls["TIME_3"]:
                counts[cond]["time3"] += 1
            if cls["PRES"]:
                counts[cond]["pres"] += 1
            if cls["JOINT"]:
                counts[cond]["joint"] += 1
        # Write incremental progress so partial data survives SIGTERM
        RAW_JSON.write_text(json.dumps(raw, indent=2, default=str))
        partial_summary = build_summary(sp, sp_has_rn, counts, complete=False)
        OUT_JSON.write_text(json.dumps(partial_summary, indent=2, default=str))

    summary = build_summary(sp, sp_has_rn, counts, complete=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("=" * 60)
    print("S140 results")
    print("=" * 60)
    for c in CONDITIONS:
        cc = counts[c]
        n_eff = cc["n"] - cc["artifact"]
        if n_eff == 0:
            print(f"  {c}: n={cc['n']} all artifact")
            continue
        t3_rate = cc["time3"] / n_eff
        t3_lo, t3_hi = wilson_ci(cc["time3"], n_eff)
        joint_rate = cc["joint"] / n_eff
        j_lo, j_hi = wilson_ci(cc["joint"], n_eff)
        print(f"  {c}: n={cc['n']} art={cc['artifact']} n_eff={n_eff}")
        print(f"      TIME_3: {cc['time3']}/{n_eff} = {t3_rate:.1%} "
              f"[{t3_lo:.1%}, {t3_hi:.1%}]")
        print(f"      JOINT:  {cc['joint']}/{n_eff} = {joint_rate:.1%} "
              f"[{j_lo:.1%}, {j_hi:.1%}]")
    print()
    print("Corpus baseline (S138 N=79, canonical 'today' opener):")
    print("  TIME_3: 16/79 = 20.3%")
    print("  JOINT:  12/79 = 15.2%")
    print()
    print("S138b (N=12, canonical 'today' opener, replication):")
    print("  TIME_3 (loose): 2/12 = 16.7%")
    print("  TIME_3 (strict, n_eff=6): 2/6 = 33.3%")
    print()
    print(f"Wrote {OUT_JSON} and {RAW_JSON}")
    return 0


def build_summary(sp, sp_has_rn, counts, complete: bool) -> dict:
    out = {
        "model": MODEL,
        "n_per_condition_target": N_PER_CONDITION,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "system_prompt": sp,
        "system_prompt_has_right_now": sp_has_rn,
        "conditions": dict(CONDITIONS),
        "counts": counts,
        "complete": complete,
    }
    rates = {}
    for c, cc in counts.items():
        n_eff = cc["n"] - cc["artifact"]
        if n_eff > 0:
            t3_lo, t3_hi = wilson_ci(cc["time3"], n_eff)
            j_lo, j_hi = wilson_ci(cc["joint"], n_eff)
            rates[c] = {
                "n": cc["n"], "n_eff": n_eff,
                "time3": cc["time3"], "time3_rate": cc["time3"] / n_eff,
                "time3_ci95": [t3_lo, t3_hi],
                "joint": cc["joint"], "joint_rate": cc["joint"] / n_eff,
                "joint_ci95": [j_lo, j_hi],
                "pres": cc["pres"], "pres_rate": cc["pres"] / n_eff,
            }
        else:
            rates[c] = {
                "n": cc["n"], "n_eff": 0,
                "time3": cc["time3"], "time3_rate": None,
                "joint": cc["joint"], "joint_rate": None,
                "pres": cc["pres"], "pres_rate": None,
            }
    out["rates"] = rates
    return out


if __name__ == "__main__":
    sys.exit(main())
