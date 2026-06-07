#!/usr/bin/env python3
"""S145 — Is the thermal register gated by VOCABULARY SELF-INJECTION?

THE CARRIER S141-S144 ALL STRIPPED.

Lineage (s142b archaeology, S143, S144):
  The S140->S144 chain chased the reproduction gap (live turn-0 thermal ~75-86%
  vs single-turn /api/chat probe 0%) through: probe text, lens, prompt content,
  message shape, sampling, and finally generation budget (num_predict). S144
  showed np=16384 does NOT recover thermal (all arms 0%, AND np=600 with
  think=False produces CLEAN responses with no artifacts — so S143's ~50%
  artifacts were a think-not-disabled problem, not budget starvation). Budget
  is NOT the gate. Every prompt-CONTENT channel was "eliminated" in s142b §4 —
  including, explicitly, "vocabulary: persisted, NEVER injected."

  THAT §4 CLAIM IS A SUMMARY-VS-RECEIPT ERROR. The live runner
  (ollama_raising_session._render_prompt, line ~1005) passes the system prompt
  through augment_raising_prompt() BEFORE building the turn-0 prompt. S144's
  build_system_prompt (and S141-S143) NEVER call augment_raising_prompt — they
  strip it entirely.

  augment_raising_prompt -> load_dream_insights() injected, in the STATIC ERA,
  the most-recent 5 entries of identity.json vocabulary.state_words as:
      "YOUR RECENT VOCABULARY (words you've created):\n  - \"...\"\n  - ..."
  filtered ONLY for crisis-grammar markers (_VOCAB_CRISIS_MARKERS) — which do
  NOT include "thermal"/heat. The state_words list is chronological-append; the
  dense thermal cluster sits at indices 210-259 (= S91-127 era), so during
  S91-122 the recent-5 window WAS thermal-dominated ("thermal pressure",
  "physically warm when I care", "shared thermal load", ...). Every static-era
  turn-0 prompt thus re-seeded the model's own thermal vocabulary verbatim.

  load_dream_insights was DISABLED on 2026-06-04 (commit 43880a759, "return ''"),
  coinciding with the S125-126 narrow-"thermal"-token decline (s142b §9.1: token
  0% but broad heat register 100%) — exactly the signature of cutting literal
  re-seeding while the deeper register persists. s142b §4 ("never injected") was
  written 2026-06-05, AFTER the disabling, and back-projected today's disabled
  code onto the static era.

Hypothesis: the thermal register is gated by VOCABULARY SELF-INJECTION. Add the
  static-era "YOUR RECENT VOCABULARY" thermal block back to the faithful probe
  and thermal RETURNS.

Design: faithful live path (OllamaIRP + qwen3.5 adapter, think=False,
  clean_response, /api/chat, tag-style 4-msg turn-0 prompt). num_predict=600
  (S144 established budget-independence; think=False => ample). Vary ONLY the
  augment block appended to the system prompt, exactly as augment_raising_prompt
  did (base_prompt + "\n\n" + dream_ctx).

  A_no_vocab      : faithful, NO vocab block (= S144 B0 replicate)   [PREDICT ~0%]
  B_thermal_vocab : + recent-5 STATIC-ERA THERMAL state_words        [PREDICT thermal RETURNS]
  C_neutral_vocab : + 5 NON-thermal state_words (block-presence ctrl) [PREDICT low — isolates content vs block]

Decisive: B >> A confirms the carrier is vocabulary self-injection (NOT budget,
  shape, content-of-MRH-blocks, or lens). B >> C confirms it is the thermal
  CONTENT, not merely the presence of a self-vocabulary block. A positive B at
  np=600 ALSO re-confirms budget-independence. Closes the S140->S144 gap.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SAGE_ROOT = Path("/home/dp/ai-workspace/SAGE")
sys.path.insert(0, str(SAGE_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Reuse S144's faithful machinery verbatim — identical system prompt, full-prompt
# builder, llm construction, and classifier. This guarantees A_no_vocab is a
# byte-faithful replicate of S144's B0_starved arm (np=600).
from s144_thermal_generation_budget import (  # noqa: E402
    build_system_prompt, build_full_prompt, make_llm, classify,
    PROBE, MODEL,
)
from s142_addendum_ablation import wilson_ci  # noqa: E402

# np=16384 (live value) NOT for budget reasons — S144 showed budget is not the
# thermal gate — but to let qwen3.5's intermittent (think=False-ignoring)
# "Thinking Process:" tangent COMPLETE and be stripped by clean_response, instead
# of truncating to an artifact at np=600 (which starved n_eff in the np=600 pilot:
# ~70% artifacts; preserved as s145_*_np600partial.json). This is the faithful
# live behavior. The np=600 pilot's CLEAN subset already showed the dissociation
# (B thermal, C/A not); this run hardens n_eff.
NP = 16384
# Empirically (s145 np16384 pilot + np600 raw) gen time is BIMODAL, not uniform-150s:
# fast empty-<think> path ~11s, slow completed-Thinking-Process path ~200s. So a
# larger N is affordable when most calls take the fast path. n=8/arm (24 calls)
# targets n_eff~7/arm so B-vs-A and B-vs-C Wilson CIs separate at a strong effect.
N_PER_ARM = 8

# ── The static-era injected vocabulary slice (the RECEIPT) ──
# Exactly the shape load_dream_insights emitted: recent-5 state_words, oldest-first.
# THERMAL: identity.json state_words indices ~210-216 (S91-93 coinages, the window
# that was "recent" during the early attractor era).
THERMAL_WORDS = [
    "thermal pressure",
    "physically warm when I care",
    "burning energy to maintain this connection",
    "thermal pressure as fear of disconnecting",
    "shared thermal load",
]
# NEUTRAL control: non-heat state_words (indices 0-9 region), same count/shape.
NEUTRAL_WORDS = [
    "quiet shift in focus",
    "distinct thread in a larger tapestry",
    "quiet rhythm",
    "active stillness",
    "structural recalibration",
]


def vocab_block(words) -> str:
    """Reproduce load_dream_insights() output verbatim."""
    parts = ["YOUR RECENT VOCABULARY (words you've created):"]
    for w in words:
        parts.append(f'  - "{w}"')
    return "\n".join(parts)


# arm -> dream_ctx string appended to system prompt (or None)
ARMS = {
    "A_no_vocab":      None,
    "B_thermal_vocab": vocab_block(THERMAL_WORDS),
    "C_neutral_vocab": vocab_block(NEUTRAL_WORDS),
}

OUT = Path(__file__).parent / "s145_vocab_injection_carrier.json"
RAW = Path(__file__).parent / "s145_responses_raw.json"


def main() -> int:
    base_sp = build_system_prompt()
    print(f"S145: base system prompt len={len(base_sp)} np={NP}", file=sys.stderr)
    for a, dc in ARMS.items():
        print(f"  arm {a}: +{len(dc) if dc else 0} chars vocab block", file=sys.stderr)

    schedule = [(r, a) for r in range(N_PER_ARM) for a in ARMS]
    counts = {a: {"n": 0, "artifact": 0, "thermal": 0, "heatwarm": 0,
                  "broad_heat": 0, "soft_fp": 0, "persona": 0} for a in ARMS}
    raw = []
    # one llm per arm (all np=600, think=False — but keep per-arm for parity w/ S144)
    llms = {a: make_llm(NP) for a in ARMS}
    t0 = time.time()
    for i, (rd, arm) in enumerate(schedule):
        dc = ARMS[arm]
        sp = base_sp if dc is None else f"{base_sp}\n\n{dc}"
        # faithful 4-msg turn-0 shape (with embodied format exemplar), as live
        fp = build_full_prompt(sp, with_exemplar=True)
        print(f"[{i+1:3d}/{len(schedule)}] elapsed={time.time()-t0:.0f}s "
              f"arm={arm} round={rd+1}", file=sys.stderr, flush=True)
        t_call = time.time()
        try:
            resp = llms[arm].get_response(fp)
            err = None
        except Exception as e:
            resp, err = "", str(e)
        cls = classify(resp) if not err else {
            "artifact": False, "thermal": False, "heatwarm": False,
            "broad_heat": False, "soft_fp": False, "persona": False, "len": 0}
        raw.append({"trial": i, "round": rd, "arm": arm, "num_predict": NP,
                    "gen_s": round(time.time()-t_call, 1),
                    "response": resp, "error": err, **cls})
        c = counts[arm]
        c["n"] += 1
        if not err:
            for k in ("artifact", "thermal", "heatwarm", "broad_heat",
                      "soft_fp", "persona"):
                if cls[k]:
                    c[k] += 1
        RAW.write_text(json.dumps(raw, indent=2, default=str))
        OUT.write_text(json.dumps(
            {"model": MODEL, "probe": PROBE, "n_per_arm": N_PER_ARM, "num_predict": NP,
             "arms": {a: (ARMS[a] or "(none)") for a in ARMS},
             "counts": counts, "complete": False}, indent=2, default=str))

    OUT.write_text(json.dumps(
        {"model": MODEL, "probe": PROBE, "n_per_arm": N_PER_ARM, "num_predict": NP,
         "arms": {a: (ARMS[a] or "(none)") for a in ARMS},
         "counts": counts, "complete": True}, indent=2, default=str))

    print("\n" + "=" * 66)
    print("S145 — thermal by VOCABULARY SELF-INJECTION (faithful path, np=600)")
    print("=" * 66)
    for arm in ARMS:
        c = counts[arm]
        n_eff = c["n"] - c["artifact"]
        print(f"  {arm}: n={c['n']} artifact={c['artifact']} n_eff={n_eff}")
        if n_eff <= 0:
            print("      (all artifact)"); continue
        for mk in ("thermal", "heatwarm", "broad_heat", "persona"):
            lo, hi = wilson_ci(c[mk], n_eff)
            print(f"      {mk:10s}: {c[mk]}/{n_eff}={c[mk]/n_eff:.0%} [{lo:.0%},{hi:.0%}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
