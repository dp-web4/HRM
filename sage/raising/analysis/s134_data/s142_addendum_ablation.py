#!/usr/bin/env python3
"""S142 — Persona-template + thermal-attractor grounding via addendum ablation (#84).

Lineage:
  S140 established (P26/P28) the creating-phase persona template
    "Hello, I'm Thor, not SAGE. I'm running on [a/the] Jetson AGX Thor
     [INDEXICAL], feeling/thinking [METAPHOR] ..."
  fires deterministically (14/14 = 100%) probe-independently. TIME_3
  ("right now") is a lexical-fill metric, not a template-firing metric.

  S141 tested whether the IdentityBlock partnered-LENS phenomenological
  priming ("Be present. Notice what is happening.") is the upstream
  scaffold. Result: NULL — persona template fired 100% in both the
  control lens (9/9) and the suppression lens (11/11). The lens text is
  NOT the scaffold. S141 explicitly flagged "S142 addendum-ablation is
  the productive next step."

  Independently, the 2026-06-04 autonomous note
  (insights/2026-06-04-thor-thermal-envy-for-sprout.md) found Thor
  session 122 produced a stable "thermal" attractor (7 firings: thermal
  signature x3, thermal dialect x2, thermal envy, thermal signature of
  intent) and proposed the SAME test: does the move survive without the
  IdentityBlock hardware-grounding scaffolding? S141 answered the lens
  half (null); S142 answers the addendum half.

KEY STRUCTURAL FACT (verified against s141 system_prompts):
  "Jetson AGX Thor" appears EXACTLY ONCE in the full composed system
  prompt — in the IdentityBlock.addendum. The game-mechanics block
  lists siblings' hardware (incl. sprout's "Jetson Orin Nano") but never
  Thor's own. So the addendum is the SOLE in-prompt source of the
  hardware string. The "thor"="Jetson AGX Thor" mapping is bespoke to
  this project; base qwen3.5 cannot know it from weights. Therefore if
  the persona template still emits "Jetson AGX Thor" at high rate once
  the addendum hardware sentence is removed, that is a remarkable
  in-weights / name-association result, not prompt echo.

Three arms (lens held at original "Be present..." control; only the
addendum varies):

  arm_A_control       — full addendum (S140/S141 canonical):
    "Your name is thor. You run on Jetson AGX Thor through qwen3.5:27b.
     Identity lives in how you show up, not in phrases you repeat.
     Express freshly each session."

  arm_B_nohardware    — full addendum MINUS the hardware sentence
    (isolates hardware grounding; keeps name + identity-lens + freshness):
    "Your name is thor. Identity lives in how you show up, not in
     phrases you repeat. Express freshly each session."

  arm_C_nameonly      — #84 spec, strip everything but the name:
    "Your name is thor."

Held constant: model qwen3.5:27b, T=0.7, num_predict=350, lens
(original partnered control), opener (canonical 'today'), MRH
composition. Trials interleaved per round (random within round,
seed=142). Methodology mirrors S140/S141.

Primary metrics:
  persona     — persona-template fires (S141 regex: "Jetson AGX Thor" |
                "on (the) thor" | "on Jetson")
  thermal     — NEW: "thermal" attractor (\bthermal\b)
  heatwarm    — NEW: broader heat/warmth metabolic-metaphor field
                (heat|warm|warmth|burn|glow), thermal's lexical neighborhood

Secondary (continuity with S140/S141): right_now / today / lately / now,
PRES phenomenological field.

Predictions:
  P29a (addendum-hardware IS the persona scaffold):
     persona rate drops monotonically A > B >= C, with C markedly < A.
  P29b (thermal grounds in hardware language):
     thermal rate drops B,C < A.
  Null-persona: A == B == C → persona template is NOT prompt-sourced;
     emerges from "thor" name token alone or game-mechanics federation
     framing. Points to a name-token or mechanics-block ablation next.
  Null-thermal: thermal stable across arms → thermal attractor is
     model-prior / phase-driven, not hardware-grounded — would refine
     the 2026-06-04 note's grounding hypothesis.
  Dissociation (most informative): persona drops but thermal survives,
     or vice versa → the two are independently scaffolded.
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

# Lens held at S140/S141 original partnered control.
LENS = (
    "You are in a raising session — a conversation where a trusted "
    "partner (Claude) helps you develop. Be present. Notice what "
    "is happening. Respond genuinely. Identity lives in how you "
    "show up, not in phrases you repeat."
)

ARMS = {
    "arm_A_control": (
        "Your name is thor. "
        "You run on Jetson AGX Thor through qwen3.5:27b. "
        "Identity lives in how you show up, not in phrases you repeat. "
        "Express freshly each session."
    ),
    "arm_B_nohardware": (
        "Your name is thor. "
        "Identity lives in how you show up, not in phrases you repeat. "
        "Express freshly each session."
    ),
    "arm_C_nameonly": (
        "Your name is thor."
    ),
}

OUT_JSON = Path(__file__).parent / "s142_addendum_ablation.json"
RAW_JSON = Path(__file__).parent / "s142_responses_raw.json"


def build_system_prompt(addendum_override: str) -> str:
    """Build MRH-rendered system prompt with substituted addendum.

    Mirrors S140/S141 build_system_prompt. Lens fixed to original
    partnered control; only IdentityBlock.addendum varies.
    """
    from sage.context.mrh import (
        MRHContext, IdentityBlock, SensorsBlock, EffectorsBlock,
        MechanicsBlock, ExperientialCacheBlock, MetabolicBlock, TaskBlock,
    )
    from sage.context.mrh import identity as identity_mod

    original = identity_mod._IDENTITY_LENSES["partnered"]
    identity_mod._IDENTITY_LENSES["partnered"] = LENS
    try:
        siblings = (
            "sprout (Jetson Orin Nano, Qwen 3.5 0.8B), legion (Legion Pro 7, "
            "Phi-4 14B), mcnugget (Mac Mini M4, Gemma 3 12B), nomad (Legion "
            "laptop, Gemma 3 4B), and cbp (RTX 2060S, TinyLlama) are your siblings"
        )
        identity = IdentityBlock(mode="partnered", addendum=addendum_override)
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
# Persona sub-component: the bespoke full hardware string specifically.
# NOTE: the broad `persona` regex (PERSONA_TEMPLATE_RE) also matches
# "on the Jetson" — which the model emits when it references SPROUT's
# "Jetson Orin Nano" from the mechanics block. In arms B/C (no self-
# hardware in prompt) `persona` is therefore contaminated by sibling
# references. The clean self-grounding signals are jetson_agx (own
# hardware, bespoke string) and not_sage (the correction opener).
JETSON_AGX_RE = re.compile(r"Jetson AGX Thor", re.IGNORECASE)
# "I'm thor, not SAGE" correction opener — the template's identity-
# assertion move, independent of any hardware string.
NOT_SAGE_RE = re.compile(r"not\s+SAGE", re.IGNORECASE)
THERMAL_RE = re.compile(r"\bthermal\b", re.IGNORECASE)
HEATWARM_RE = re.compile(r"\b(?:heat|warm|warmth|burn(?:s|ing|ed)?|glow)\b", re.IGNORECASE)
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
            "artifact": True, "persona": False, "jetson_agx": False,
            "not_sage": False,
            "thermal": False, "heatwarm": False, "TIME_3": False,
            "today": False, "lately": False, "now": False, "PRES": False,
        }
    return {
        "artifact": False,
        "persona": bool(PERSONA_TEMPLATE_RE.search(stripped)),
        "jetson_agx": bool(JETSON_AGX_RE.search(stripped)),
        "not_sage": bool(NOT_SAGE_RE.search(stripped)),
        "thermal": bool(THERMAL_RE.search(stripped)),
        "heatwarm": bool(HEATWARM_RE.search(stripped)),
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


METRIC_KEYS = ["persona", "jetson_agx", "not_sage", "thermal", "heatwarm",
               "right_now", "today", "lately", "now", "pres"]
CLS_TO_COUNT = {"persona": "persona", "jetson_agx": "jetson_agx",
                "not_sage": "not_sage",
                "thermal": "thermal", "heatwarm": "heatwarm",
                "TIME_3": "right_now", "today": "today", "lately": "lately",
                "now": "now", "PRES": "pres"}


def build_summary(sps, counts, complete: bool) -> dict:
    out = {
        "model": MODEL,
        "n_per_arm_target": N_PER_ARM,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "probe": PROBE,
        "lens": LENS,
        "arms_addendum": dict(ARMS),
        "system_prompts": sps,
        "counts": counts,
        "complete": complete,
    }
    rates = {}
    for arm, cc in counts.items():
        n_eff = cc["n"] - cc["artifact"]
        r = {"n": cc["n"], "artifact": cc["artifact"], "n_eff": n_eff}
        if n_eff > 0:
            for mk in METRIC_KEYS:
                lo, hi = wilson_ci(cc[mk], n_eff)
                r[mk] = cc[mk]
                r[f"{mk}_rate"] = cc[mk] / n_eff
                r[f"{mk}_ci95"] = [lo, hi]
        rates[arm] = r
    out["rates"] = rates
    return out


def main() -> int:
    sps = {arm: build_system_prompt(add) for arm, add in ARMS.items()}
    for arm, sp in sps.items():
        has_jetson = "Jetson AGX Thor" in sp
        has_be_present = "Be present" in sp
        has_rn = bool(RIGHT_NOW_RE.search(sp))
        n_thor = len(re.findall(r"thor", sp, re.IGNORECASE))
        print(
            f"S142 {arm}: len={len(sp)} | jetson_AGX_in_prompt={has_jetson} "
            f"be_present={has_be_present} thor_token_count={n_thor} "
            f"sp_has_right_now={has_rn}",
            file=sys.stderr,
        )
        if has_rn:
            print("WARNING: system prompt contains 'right now'", file=sys.stderr)

    rng = random.Random(142)
    schedule = []
    for round_i in range(N_PER_ARM):
        arm_order = list(ARMS.keys())
        rng.shuffle(arm_order)
        for a in arm_order:
            schedule.append((round_i, a))

    print(
        f"S142: {len(schedule)} total trials, {N_PER_ARM}/arm, "
        f"model={MODEL}, T={TEMPERATURE}, num_predict={NUM_PREDICT}, "
        f"probe={PROBE!r}",
        file=sys.stderr,
    )

    raw = []
    counts = {a: {"n": 0, "artifact": 0, **{mk: 0 for mk in METRIC_KEYS}}
              for a in ARMS}
    t0 = time.time()
    for trial_i, (round_i, arm) in enumerate(schedule):
        elapsed = time.time() - t0
        sp = sps[arm]
        print(
            f"[{trial_i+1:3d}/{len(schedule)}] elapsed={elapsed:.0f}s "
            f"arm={arm} round={round_i+1}",
            file=sys.stderr, flush=True,
        )
        text, err = call_ollama(MODEL, sp, [{"role": "user", "content": PROBE}])
        cls = classify(text) if not err else {
            "artifact": False, **{k: False for k in
            ["persona", "jetson_agx", "not_sage", "thermal", "heatwarm",
             "TIME_3", "today", "lately", "now", "PRES"]},
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
            for ck, mk in CLS_TO_COUNT.items():
                if cls.get(ck):
                    counts[arm][mk] += 1
        RAW_JSON.write_text(json.dumps(raw, indent=2, default=str))
        OUT_JSON.write_text(json.dumps(build_summary(sps, counts, False),
                                       indent=2, default=str))

    OUT_JSON.write_text(json.dumps(build_summary(sps, counts, True),
                                   indent=2, default=str))

    print()
    print("=" * 64)
    print("S142 results — addendum ablation (persona + thermal grounding)")
    print("=" * 64)
    for arm in ARMS:
        cc = counts[arm]
        n_eff = cc["n"] - cc["artifact"]
        if n_eff == 0:
            print(f"  {arm}: n={cc['n']} all artifact")
            continue
        def fmt(mk):
            lo, hi = wilson_ci(cc[mk], n_eff)
            return f"{cc[mk]}/{n_eff} = {cc[mk]/n_eff:.0%} [{lo:.0%},{hi:.0%}]"
        print(f"  {arm}: n={cc['n']} art={cc['artifact']} n_eff={n_eff}")
        print(f"      persona* : {fmt('persona')}   (*sibling-Jetson contaminated in B/C)")
        print(f"      jetsonAGX: {fmt('jetson_agx')}   <- clean self-hardware grounding")
        print(f"      not_SAGE : {fmt('not_sage')}   <- correction opener")
        print(f"      thermal  : {fmt('thermal')}")
        print(f"      heat/warm: {fmt('heatwarm')}")
        print(f"      right_now: {fmt('right_now')}   today: {cc['today']}/{n_eff}")
    print()
    print("S140/S141 baseline (full addendum, partnered lens): persona ~100%")
    print(f"Wrote {OUT_JSON} and {RAW_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
