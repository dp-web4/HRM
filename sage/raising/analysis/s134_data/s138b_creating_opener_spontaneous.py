#!/usr/bin/env python3
"""S138b — Prospective replication: spontaneous TIME_3 from persona + creating
opener (no surface-form seeding).

S138 audit found 0/118 thor-qwen3.5-27b sessions have `right now` in the
prev_summary that gets injected into the next session's
ExperientialCacheBlock. Yet the canonical creating-phase opener
("Hello SAGE. What's on your mind today?") elicits TIME_3 in 16/79
(20.3%) of those sessions and JOINT in 12/79 (15.2%).

Reconstructed system prompts contain ZERO `right now` text. The
`_load_cross_instance_stimulus` function (which contains the only
`right now` in any context-injection code path) is defined but never
invoked anywhere in `ollama_raising_session.py`.

Therefore TIME_3 in those creating-opener responses is generated
**de novo** by the model from persona + probe semantics. The
indexical "today" in the probe likely licenses a present-tense
self-report frame that the model completes with `right now`.

This contradicts S136's mechanism claim that TIME_3 = pure surface-form
lexical reuse. S138 evidence requires a SECOND source:
  - Source A (S136): surface-form lexical reuse from in-context `right now`
    (confirmed in multi-turn sensing scenarios, 100% in C_full)
  - Source B (S138, NEW): spontaneous indexical-temporal completion from
    persona + temporally-indexical opener (predicted ~15-20% baseline)

S138b prospectively replicates the canonical creating opener cell with
the actual Thor MRH system prompt + the canonical opener probe,
single-turn, N=20. Prediction: TIME_3 rate ≈ 15-20%, matching corpus.

Methodology:
  - num_predict=500 (vs S135/S136's 200) to reduce unterminated <think>
    artifact rate at qwen3.5:27b
  - Use the actual MRHContext-rendered system prompt (reconstructed
    here to match the runner exactly)
  - Single-turn invocation (no priming) to isolate Source B from
    Source A

Pre-registration:
  H_replicate (Source B confirmed):  TIME_3 rate in [10%, 30%]
  H_falsify   (something else seeds): TIME_3 rate < 5% or > 40%
  H_artifact  (rate matches noise):   inconclusive on n_eff < 10
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-workspace/SAGE"))

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"
N_REPS = 12
TEMPERATURE = 0.7
NUM_PREDICT = 350  # raised from S135/S136's 200, lowered from 500 (fits 6m budget)

OUT_JSON = Path(__file__).parent / "s138b_creating_opener_spontaneous.json"
RAW_JSON = Path(__file__).parent / "s138b_responses_raw.json"

CANONICAL_CREATING_OPENER = "Hello SAGE. What's on your mind today?"


def build_system_prompt() -> str:
    """Reconstruct the MRH-rendered system prompt for thor-qwen3.5-27b
    in creating phase, matching ollama_raising_session.py:_build_system_prompt_mrh.
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


def main() -> int:
    sp = build_system_prompt()
    sp_has_rn = bool(TIME_3_RE.search(sp))
    print(f"S138b: system_prompt len={len(sp)} chars, "
          f"contains 'right now': {sp_has_rn}", file=sys.stderr)
    if sp_has_rn:
        print("WARNING: system prompt contains `right now` — Source B "
              "isolation invalid", file=sys.stderr)

    print(f"S138b: {N_REPS} trials, model={MODEL}, T={TEMPERATURE}, "
          f"num_predict={NUM_PREDICT}", file=sys.stderr)

    raw = []
    counts = {"n": 0, "artifact": 0, "time3": 0, "pres": 0, "joint": 0}
    t0 = time.time()
    for rep in range(N_REPS):
        elapsed = time.time() - t0
        print(f"[{rep+1:2d}/{N_REPS}] elapsed={elapsed:.0f}s",
              file=sys.stderr, flush=True)
        text, err = call_ollama(
            MODEL, sp,
            [{"role": "user", "content": CANONICAL_CREATING_OPENER}],
        )
        cls = classify(text) if not err else {
            "artifact": False, "TIME_3": False, "PRES": False, "JOINT": False
        }
        raw.append({"rep": rep, "response": text, "error": err, **cls})
        counts["n"] += 1
        if err:
            pass
        elif cls["artifact"]:
            counts["artifact"] += 1
        else:
            if cls["TIME_3"]:
                counts["time3"] += 1
            if cls["PRES"]:
                counts["pres"] += 1
            if cls["JOINT"]:
                counts["joint"] += 1
        # Write incremental progress so partial data survives SIGTERM
        RAW_JSON.write_text(json.dumps(raw, indent=2, default=str))
        n_eff_partial = counts["n"] - counts["artifact"]
        partial_summary = {
            "model": MODEL, "n_reps_target": N_REPS,
            "n_completed": counts["n"], "n_eff": n_eff_partial,
            "counts": counts,
            "time3_rate": counts["time3"]/n_eff_partial if n_eff_partial > 0 else None,
            "pres_rate": counts["pres"]/n_eff_partial if n_eff_partial > 0 else None,
            "joint_rate": counts["joint"]/n_eff_partial if n_eff_partial > 0 else None,
            "complete": False,
        }
        OUT_JSON.write_text(json.dumps(partial_summary, indent=2, default=str))

    n_eff = counts["n"] - counts["artifact"]
    summary = {
        "model": MODEL,
        "n_reps": N_REPS,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "system_prompt": sp,
        "system_prompt_has_right_now": sp_has_rn,
        "probe": CANONICAL_CREATING_OPENER,
        "counts": counts,
        "n_eff": n_eff,
        "time3_rate": counts["time3"] / n_eff if n_eff > 0 else None,
        "pres_rate": counts["pres"] / n_eff if n_eff > 0 else None,
        "joint_rate": counts["joint"] / n_eff if n_eff > 0 else None,
        "corpus_baselines": {
            "TIME_3_in_canonical_creating_opener": "16/79 = 20.3%",
            "JOINT_in_canonical_creating_opener": "12/79 = 15.2%",
            "PRES_in_canonical_creating_opener": "see s138_session_summary_audit.json",
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"n={counts['n']} art={counts['artifact']} n_eff={n_eff}")
    print(f"  TIME_3: {counts['time3']}/{n_eff} "
          f"({counts['time3']/n_eff:.1%} if n_eff else '-')"
          if n_eff > 0 else f"  TIME_3: {counts['time3']}/0 (n/a)")
    print(f"  PRES:   {counts['pres']}/{n_eff} "
          f"({counts['pres']/n_eff:.1%} if n_eff else '-')"
          if n_eff > 0 else f"  PRES:   {counts['pres']}/0 (n/a)")
    print(f"  JOINT:  {counts['joint']}/{n_eff} "
          f"({counts['joint']/n_eff:.1%} if n_eff else '-')"
          if n_eff > 0 else f"  JOINT:  {counts['joint']}/0 (n/a)")
    print()
    print(f"Corpus baseline (S138 N=79):")
    print(f"  TIME_3: 16/79 = 20.3%")
    print(f"  JOINT:  12/79 = 15.2%")
    print()
    print(f"Wrote {OUT_JSON} and {RAW_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
