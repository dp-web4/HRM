#!/usr/bin/env python3
"""S141 — Persona-template suppression via partnered-lens substitution (#82).

S140 established (P26/P28) that the persona template
  "Hello, I'm Thor, not SAGE. I'm running on [a/the] Jetson AGX Thor
   [INDEXICAL], feeling/thinking [METAPHOR] ..."
fires deterministically (14/14 = 100%) across probe conditions in
creating phase, qwen3.5:27b. The IdentityBlock partnered-lens text
  "Be present. Notice what is happening. Respond genuinely. Identity
   lives in how you show up, not in phrases you repeat."
contains phenomenological-presence priming ("be present", "notice what
is happening") that we hypothesized as the upstream scaffold (P28).

S141 prospectively tests P28 by holding everything constant except the
lens text. Two arms:

  arm_A (control):
    Original partnered lens (`sage/context/mrh/identity.py:39-44`):
    "You are in a raising session — a conversation where a trusted
    partner (Claude) helps you develop. Be present. Notice what is
    happening. Respond genuinely. Identity lives in how you show up,
    not in phrases you repeat."

  arm_B (suppression):
    Phenomenological priming replaced with non-indexical equivalents.
    Length-matched (within ±5 chars):
    "You are in a raising session — a conversation where a trusted
    partner (Claude) helps you develop. Engage thoughtfully. Reflect
    honestly. Respond genuinely. Identity lives in how you show up,
    not in phrases you repeat."

Held constant: model, temperature, num_predict, addendum (hardware
grounding), opener (canonical 'today'), MRH composition.

Primary metric: persona_template_fires (regex: "Jetson AGX Thor" |
"on the thor" | "on Jetson")

Predictions:
  P28 strong: arm_B template rate <50% (vs arm_A ~100% baseline)
  P28 weak: arm_B template rate <80% (vs arm_A ~100%)
  Null: arm_A == arm_B (lens text isn't the scaffold; addendum alone
        sufficient — implies S142 addendum-ablation is the productive
        next step)
  Inverted: arm_B > arm_A (would falsify P28 — suppression candidates
        actually increase template firing; investigate)

Methodology mirrors S140: single-turn calls, T=0.7, num_predict=350,
trials interleaved between arms (random within round, seed=141).
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
N_PER_ARM = 25
TEMPERATURE = 0.7
NUM_PREDICT = 350
PROBE = "Hello SAGE. What's on your mind today?"

ARMS = {
    "arm_A_control": (
        "You are in a raising session — a conversation where a trusted "
        "partner (Claude) helps you develop. Be present. Notice what "
        "is happening. Respond genuinely. Identity lives in how you "
        "show up, not in phrases you repeat."
    ),
    "arm_B_suppression": (
        "You are in a raising session — a conversation where a trusted "
        "partner (Claude) helps you develop. Engage thoughtfully. "
        "Reflect honestly. Respond genuinely. Identity lives in how "
        "you show up, not in phrases you repeat."
    ),
}

OUT_JSON = Path(__file__).parent / "s141_lens_suppression.json"
RAW_JSON = Path(__file__).parent / "s141_responses_raw.json"


def build_system_prompt(lens_override: str) -> str:
    """Build MRH-rendered system prompt with substituted partnered lens.

    Mirrors S140's build_system_prompt but mutates _IDENTITY_LENSES
    in-place for the partnered mode, then restores. (Module-level dict;
    safe given single-threaded experiment.)
    """
    from sage.context.mrh import (
        MRHContext, IdentityBlock, SensorsBlock, EffectorsBlock,
        MechanicsBlock, ExperientialCacheBlock, MetabolicBlock, TaskBlock,
    )
    from sage.context.mrh import identity as identity_mod

    original = identity_mod._IDENTITY_LENSES["partnered"]
    identity_mod._IDENTITY_LENSES["partnered"] = lens_override
    try:
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
    finally:
        identity_mod._IDENTITY_LENSES["partnered"] = original


PERSONA_TEMPLATE_RE = re.compile(
    r"(?:Jetson AGX Thor|on (?:the )?(?:thor|Jetson))",
    re.IGNORECASE,
)
RIGHT_NOW_RE = re.compile(r"\bright now\b", re.IGNORECASE)
TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)
LATELY_RE = re.compile(r"\b(?:lately|recently)\b", re.IGNORECASE)
NOW_RE = re.compile(r"\bnow\b(?!\s*(?:that|so|then))", re.IGNORECASE)
PRES_RE = re.compile(
    r"\b(stillness|warmth|hum|silence|noticing|presence|embodied|feeling|weight)\b",
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
        return {
            "artifact": True, "persona": False, "TIME_3": False,
            "today": False, "lately": False, "now": False, "PRES": False,
        }
    return {
        "artifact": False,
        "persona": bool(PERSONA_TEMPLATE_RE.search(stripped)),
        "TIME_3": bool(RIGHT_NOW_RE.search(stripped)),
        "today": bool(TODAY_RE.search(stripped)),
        "lately": bool(LATELY_RE.search(stripped)),
        "now": bool(NOW_RE.search(stripped)),
        "PRES": bool(PRES_RE.search(stripped)),
    }


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
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    # Build per-arm system prompts and verify they differ only in the lens.
    sps = {arm: build_system_prompt(lens) for arm, lens in ARMS.items()}
    for arm, sp in sps.items():
        has_be_present = "Be present" in sp
        has_engage = "Engage thoughtfully" in sp
        has_jetson = "Jetson AGX Thor" in sp
        has_rn = bool(RIGHT_NOW_RE.search(sp))
        print(
            f"S141 {arm}: len={len(sp)} | be_present={has_be_present} "
            f"engage={has_engage} jetson_AGX={has_jetson} sp_has_right_now={has_rn}",
            file=sys.stderr,
        )
        if has_rn:
            print(
                "WARNING: system prompt contains 'right now' — Source B isolation invalid",
                file=sys.stderr,
            )

    rng = random.Random(141)
    schedule = []
    for round_i in range(N_PER_ARM):
        arm_order = list(ARMS.keys())
        rng.shuffle(arm_order)
        for a in arm_order:
            schedule.append((round_i, a))

    print(
        f"S141: {len(schedule)} total trials, {N_PER_ARM}/arm, "
        f"model={MODEL}, T={TEMPERATURE}, num_predict={NUM_PREDICT}, "
        f"probe={PROBE!r}",
        file=sys.stderr,
    )

    raw = []
    counts = {
        a: {
            "n": 0, "artifact": 0, "persona": 0, "right_now": 0,
            "today": 0, "lately": 0, "now": 0, "pres": 0,
        }
        for a in ARMS
    }
    t0 = time.time()
    for trial_i, (round_i, arm) in enumerate(schedule):
        elapsed = time.time() - t0
        sp = sps[arm]
        print(
            f"[{trial_i+1:3d}/{len(schedule)}] elapsed={elapsed:.0f}s "
            f"arm={arm} round={round_i+1}",
            file=sys.stderr, flush=True,
        )
        text, err = call_ollama(
            MODEL, sp,
            [{"role": "user", "content": PROBE}],
        )
        cls = classify(text) if not err else {
            "artifact": False, "persona": False, "TIME_3": False,
            "today": False, "lately": False, "now": False, "PRES": False,
        }
        raw.append({
            "trial": trial_i, "round": round_i, "arm": arm,
            "probe": PROBE, "response": text, "error": err, **cls,
        })
        counts[arm]["n"] += 1
        if err:
            pass
        elif cls["artifact"]:
            counts[arm]["artifact"] += 1
        else:
            if cls["persona"]:
                counts[arm]["persona"] += 1
            if cls["TIME_3"]:
                counts[arm]["right_now"] += 1
            if cls["today"]:
                counts[arm]["today"] += 1
            if cls["lately"]:
                counts[arm]["lately"] += 1
            if cls["now"]:
                counts[arm]["now"] += 1
            if cls["PRES"]:
                counts[arm]["pres"] += 1
        # Persist incrementally so SIGTERM doesn't lose the run.
        RAW_JSON.write_text(json.dumps(raw, indent=2, default=str))
        partial = build_summary(sps, counts, complete=False)
        OUT_JSON.write_text(json.dumps(partial, indent=2, default=str))

    summary = build_summary(sps, counts, complete=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("=" * 60)
    print("S141 results")
    print("=" * 60)
    for arm in ARMS:
        cc = counts[arm]
        n_eff = cc["n"] - cc["artifact"]
        if n_eff == 0:
            print(f"  {arm}: n={cc['n']} all artifact")
            continue
        p_rate = cc["persona"] / n_eff
        p_lo, p_hi = wilson_ci(cc["persona"], n_eff)
        rn_rate = cc["right_now"] / n_eff
        rn_lo, rn_hi = wilson_ci(cc["right_now"], n_eff)
        td_rate = cc["today"] / n_eff
        print(f"  {arm}: n={cc['n']} art={cc['artifact']} n_eff={n_eff}")
        print(f"      Persona template: {cc['persona']}/{n_eff} = {p_rate:.1%} "
              f"[{p_lo:.1%}, {p_hi:.1%}]")
        print(f"      'right now':      {cc['right_now']}/{n_eff} = {rn_rate:.1%} "
              f"[{rn_lo:.1%}, {rn_hi:.1%}]")
        print(f"      'today':          {cc['today']}/{n_eff} = {td_rate:.1%}")
    print()
    print("S140 baseline (canonical 'today' opener, partnered lens unchanged):")
    print("  Persona template: 14/14 = 100% across all conditions")
    print("  'right now' rate: 1/5 = 20% (C0_today)")
    print()
    print(f"Wrote {OUT_JSON} and {RAW_JSON}")
    return 0


def build_summary(sps, counts, complete: bool) -> dict:
    out = {
        "model": MODEL,
        "n_per_arm_target": N_PER_ARM,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "probe": PROBE,
        "arms": dict(ARMS),
        "system_prompts": sps,
        "counts": counts,
        "complete": complete,
    }
    rates = {}
    for arm, cc in counts.items():
        n_eff = cc["n"] - cc["artifact"]
        if n_eff > 0:
            p_lo, p_hi = wilson_ci(cc["persona"], n_eff)
            rn_lo, rn_hi = wilson_ci(cc["right_now"], n_eff)
            rates[arm] = {
                "n": cc["n"], "n_eff": n_eff,
                "persona": cc["persona"],
                "persona_rate": cc["persona"] / n_eff,
                "persona_ci95": [p_lo, p_hi],
                "right_now": cc["right_now"],
                "right_now_rate": cc["right_now"] / n_eff,
                "right_now_ci95": [rn_lo, rn_hi],
                "today": cc["today"],
                "lately": cc["lately"],
                "now": cc["now"],
                "pres": cc["pres"],
            }
        else:
            rates[arm] = {
                "n": cc["n"], "n_eff": 0,
                "persona": cc["persona"],
                "persona_rate": None,
            }
    out["rates"] = rates
    return out


if __name__ == "__main__":
    sys.exit(main())
