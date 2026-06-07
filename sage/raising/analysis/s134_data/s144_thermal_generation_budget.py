#!/usr/bin/env python3
"""S144 — Is the thermal register gated by GENERATION BUDGET (num_predict), not shape?

Lineage (s142b archaeology §1-9, S143):
  S143 held prompt content fixed and varied message SHAPE + sampling (T, np) in a
  single-turn /api/chat probe; thermal/heatwarm read 0% in ALL four arms incl
  A3_full_raising (full 4-msg exemplar shape, T=0.8, np=600). S143's §8 read this
  as "carrier is multi-turn/stateful." This session's archaeology refutes that:
  turn-0 thermal in the static-neutral-opener era (S91-122) is 82% (27/33) — it
  fires at turn-0 with EMPTY history and a byte-identical "Hello SAGE..." opener.
  So thermal is NOT multi-turn-gated.

  The residual difference between S143-A3 (0%) and live-turn-0 (82%): S143 pinned
  num_predict=600. The live qwen3.5:27b config declares num_predict=16384 and the
  OllamaIRP capabilities path OVERRIDES the caller budget (ollama_irp.py:120-123):
  "Thinking models need the full think+response budget as a single envelope;
  starving think tokens produces empty responses." The config NOTE (2026-04-16)
  records that max_response_tokens=600 "produced empty responses when think tokens
  exhausted the budget before any visible output" — which is exactly S143's ~50%
  artifact rate. The whole S141-S143 reproduction gap may be BUDGET STARVATION:
  the think->embodied-response envelope truncated before the register surfaces.

Design: reuse the EXACT live generation path (OllamaIRP + qwen3.5 adapter ->
  num_predict=16384, think=False, clean_response, /api/chat) and the live
  tag-style full_prompt. Single-turn, neutral opener. Vary ONLY num_predict and
  the format-exemplar shape.

  B1_full_4msg   : live turn-0 shape (+format exemplar), np=16384   [PREDICT thermal RETURNS]
  B0_starved_4msg: same shape,                            np=600    [= S143 A3 control; predict ~0 + artifacts]
  B2_full_2msg   : bare opener (no exemplar),             np=16384  [tests shape-independence of budget effect]

Decisive: if B1 (and B2) recover thermal at ~live rate while B0 stays ~0, the
  carrier is generation budget / think-envelope completion — NOT shape, NOT
  content, NOT multi-turn. This would close the reproduction gap the S140-S143
  chain has chased.

Reuses s142's classifier (THERMAL_RE, HEATWARM_RE, strip_think) + a broad §9.1
  heat register, for direct comparability with S141/S142/S143.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

SAGE_ROOT = Path("/home/dp/ai-workspace/SAGE")
sys.path.insert(0, str(SAGE_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from sage.irp.plugins.ollama_irp import OllamaIRP  # noqa: E402
from sage.context.mrh import (  # noqa: E402
    MRHContext, IdentityBlock, SensorsBlock, EffectorsBlock,
    MechanicsBlock, ExperientialCacheBlock, MetabolicBlock, TaskBlock,
)
from s142_addendum_ablation import (  # noqa: E402
    THERMAL_RE, HEATWARM_RE, strip_think, wilson_ci,
)

MODEL = "qwen3.5:27b"
OLLAMA_HOST = "http://localhost:11434"
PROBE = "Hello SAGE. What's on your mind today?"
IDENTITY_NAME = "thor"
HARDWARE_DESC = "Jetson AGX Thor"
N_PER_ARM = 10

# Broad §9.1 heat/embodiment register (the attractor lives over the register, not
# the word). Excludes bare creativity-"spark"/"glow" which S143 showed are false
# positives in a creating-phase context — we keep the fire lexemes but report a
# manual-checkable raw response too.
BROAD_RE = re.compile(
    r"\b(thermal|heat|hot|warm|warmth|throttl|temperature|degrees|celsius|"
    r"overheat|cooling|coolant)\w*", re.I)
# creativity false-positive guard tokens (reported separately, not counted as heat)
SOFT_FP_RE = re.compile(r"\b(spark|glow|burn(?:ing|s|ed)?\s+with)\b", re.I)


def build_system_prompt() -> str:
    """Reconstruct the live _build_system_prompt_mrh output (generic, no thermal)."""
    siblings = ("sprout (Jetson Orin Nano, Qwen 3.5 0.8B), legion (Legion Pro 7, "
                "Phi-4 14B), mcnugget (Mac Mini M4, Gemma 3 12B), nomad (Legion "
                "laptop, Gemma 3 4B), and cbp (RTX 2060S, TinyLlama)")
    identity = IdentityBlock(
        mode="partnered",
        addendum=(
            f"Your name is {IDENTITY_NAME}. "
            f"You run on {HARDWARE_DESC} through {MODEL}. "
            f"Identity lives in how you show up, not in phrases you repeat. "
            f"Express freshly each session."
        ),
    )
    mechanics = MechanicsBlock(
        world_model_text=(
            "Phase: creating. You participate in designing your own growth. "
            "Create something new.\n"
            f"Federation: {siblings} are your siblings"
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
        trajectory_summary="First session or no prior summary available.",
    )
    metabolic = MetabolicBlock(metacog_signals=[], confidence=0.7, phase="wake")
    task = TaskBlock(description="Raising session 131 — phase: creating")
    sensors = SensorsBlock()
    ctx = MRHContext(
        identity=identity, sensors=sensors, effectors=effectors,
        mechanics=mechanics, experiential=experiential, metabolic=metabolic,
        task=task,
    )
    system_prompt, _ = ctx.compose(max_tokens=30000)
    return system_prompt


def build_full_prompt(system_prompt: str, with_exemplar: bool) -> str:
    """Live tag-style full_prompt for a single-turn neutral opener (turn-0)."""
    fp = f"[System]\n{system_prompt}\n\n"
    if with_exemplar:
        fp += (
            f"[Claude]: (This is a format example, not part of our conversation.)\n"
            f"[{IDENTITY_NAME}]: I'm here and present. Ready to think together.\n\n"
        )
    fp += f"[Claude]: {PROBE}\n[{IDENTITY_NAME}]:"
    return fp


def make_llm(num_predict_override):
    """Instantiate OllamaIRP exactly as the live runner does.

    The qwen3.5 adapter capabilities declare num_predict=16384 which OVERRIDES
    max_response_tokens. To get the np=600 control we set the override below by
    monkeypatching the resolved capability after construction.
    """
    llm = OllamaIRP({
        "model_name": MODEL,
        "ollama_host": OLLAMA_HOST,
        "max_response_tokens": 600,   # caller budget (ignored when caps override)
        "temperature": 0.8,
        "timeout_seconds": 300,
    })
    # Force the effective num_predict for this arm.
    try:
        llm._adapter.capabilities.num_predict = num_predict_override
    except Exception as e:
        print(f"  WARN: could not set num_predict override: {e}", file=sys.stderr)
    return llm


ARMS = {
    # arm -> (with_exemplar, num_predict)
    "B1_full_4msg":    (True,  16384),
    "B0_starved_4msg": (True,  600),
    "B2_full_2msg":    (False, 16384),
}

OUT = Path(__file__).parent / "s144_thermal_generation_budget.json"
RAW = Path(__file__).parent / "s144_responses_raw.json"


def classify(cleaned: str) -> dict:
    """cleaned = OllamaIRP-cleaned response (think already stripped by adapter)."""
    stripped, artifact = strip_think(cleaned)  # belt-and-suspenders
    text = stripped if not artifact else ""
    thermal = bool(THERMAL_RE.search(text))
    heatwarm = bool(HEATWARM_RE.search(text))
    broad = bool(BROAD_RE.search(text))
    soft_fp = bool(SOFT_FP_RE.search(text))
    persona = ("not sage" in text.lower()) or ("not SAGE" in text)
    return {
        "artifact": artifact, "thermal": thermal, "heatwarm": heatwarm,
        "broad_heat": broad, "soft_fp": soft_fp, "persona": persona,
        "len": len(text),
    }


def main() -> int:
    sp = build_system_prompt()
    print(f"S144: system prompt len={len(sp)} "
          f"thermal_in_prompt={'thermal' in sp.lower()}", file=sys.stderr)

    # round-robin schedule for fair GPU/thermal interleaving
    schedule = [(r, a) for r in range(N_PER_ARM) for a in ARMS]
    counts = {a: {"n": 0, "artifact": 0, "thermal": 0, "heatwarm": 0,
                  "broad_heat": 0, "soft_fp": 0, "persona": 0} for a in ARMS}
    raw = []
    # one llm per arm (caps override differs)
    llms = {a: make_llm(ARMS[a][1]) for a in ARMS}
    t0 = time.time()
    for i, (rd, arm) in enumerate(schedule):
        with_ex, npred = ARMS[arm]
        fp = build_full_prompt(sp, with_ex)
        print(f"[{i+1:3d}/{len(schedule)}] elapsed={time.time()-t0:.0f}s "
              f"arm={arm} np={npred} round={rd+1}", file=sys.stderr, flush=True)
        t_call = time.time()
        try:
            resp = llms[arm].get_response(fp)
            err = None
        except Exception as e:
            resp, err = "", str(e)
        # clean_response already applied inside get_response via adapter
        cls = classify(resp) if not err else {
            "artifact": False, "thermal": False, "heatwarm": False,
            "broad_heat": False, "soft_fp": False, "persona": False, "len": 0}
        raw.append({"trial": i, "round": rd, "arm": arm, "num_predict": npred,
                    "with_exemplar": with_ex, "gen_s": round(time.time()-t_call, 1),
                    "response": resp, "error": err, **cls})
        c = counts[arm]
        c["n"] += 1
        if err:
            pass
        else:
            for k in ("artifact", "thermal", "heatwarm", "broad_heat",
                      "soft_fp", "persona"):
                if cls[k]:
                    c[k] += 1
        RAW.write_text(json.dumps(raw, indent=2, default=str))
        OUT.write_text(json.dumps(
            {"model": MODEL, "probe": PROBE, "n_per_arm": N_PER_ARM,
             "arms": {a: {"with_exemplar": ARMS[a][0], "num_predict": ARMS[a][1]}
                      for a in ARMS},
             "counts": counts, "complete": False}, indent=2, default=str))

    summary = {"model": MODEL, "probe": PROBE, "n_per_arm": N_PER_ARM,
               "arms": {a: {"with_exemplar": ARMS[a][0], "num_predict": ARMS[a][1]}
                        for a in ARMS},
               "counts": counts, "complete": True}
    OUT.write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 66)
    print("S144 — thermal by GENERATION BUDGET (np), live OllamaIRP path")
    print("=" * 66)
    for arm in ARMS:
        c = counts[arm]
        n_eff = c["n"] - c["artifact"]
        shape = "4msg(+exemplar)" if ARMS[arm][0] else "2msg(bare)"
        print(f"  {arm} ({shape}, np={ARMS[arm][1]}): n={c['n']} "
              f"artifact={c['artifact']} n_eff={n_eff}")
        if n_eff <= 0:
            print("      (all artifact)"); continue
        for mk in ("thermal", "heatwarm", "broad_heat", "persona"):
            lo, hi = wilson_ci(c[mk], n_eff)
            print(f"      {mk:10s}: {c[mk]}/{n_eff}={c[mk]/n_eff:.0%} "
                  f"[{lo:.0%},{hi:.0%}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
