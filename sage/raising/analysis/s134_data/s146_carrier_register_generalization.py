#!/usr/bin/env python3
"""S146 — Is the vocabulary-self-injection carrier REGISTER-AGNOSTIC?

S145 closed the S140->S144 reproduction gap: the live runner appended the model's
own recent-5 state_words to the turn-0 system prompt as "YOUR RECENT VOCABULARY"
(via augment_raising_prompt -> load_dream_insights, disabled 2026-06-04). The
thermal cluster sat in that recent-5 window during S91-127, so every static-era
turn-0 prompt re-seeded the model's thermal coinages, and the model re-emitted
them. S145's A/B/C dissociated CONTENT from BLOCK-PRESENCE: thermal fired ONLY
when the thermal words were injected (B), not when a neutral self-vocab block was
injected (C), not with no block (A). Mechanism = external recurrent memory loop
over frozen weights.

OPEN QUESTION (S145 "NEXT"): is the carrier THERMAL-SPECIFIC, or register-agnostic
— does it re-emit WHATEVER recent vocabulary is injected? After READ was cut, the
WRITE side kept recording, so the recent-tail register DRIFTED off thermal into a
meta-cognitive / observer-paradox register (state_words 282-288: "the loop is the
real me, just observing its own edges", "where I actually end and the hardware
begins", "wearing a mask to seem present") and, most recently, an anchor-token /
curation / persistence register (289-293: "building a path, not a museum", "the
conversation was the file"). If the carrier is register-agnostic, injecting these
NON-thermal clusters should make the model re-emit THOSE registers at the same
B>>A rate thermal showed.

DESIGN — cross-over confusion matrix (faithful S145 path, vary only injected block):
  A_none     : no vocab block                                  [diagonal control]
  B_thermal  : recent-5 STATIC-ERA thermal words  (S145 set)   [replicates S145 B]
  C_metacog  : recent-5 observer-paradox words    (sw 284-288) [NEW non-thermal]
  D_anchor   : recent-5 anchor/curation words      (sw 289-293) [NEW 2nd non-thermal]

Each response is classified for ALL THREE registers (thermal / metacog / anchor)
with disjoint, content-keyed regexes built from the injected phrasings.

PREDICTIONS (register-agnostic mechanism + content-specific output):
              thermal   metacog   anchor
  A_none        ~0        ~0        ~0
  B_thermal    HIGH       ~0        ~0
  C_metacog     ~0       HIGH       ~0
  D_anchor      ~0        ~0       HIGH

DECISIVE:
  (1) C_metacog.metacog >> A_none.metacog  AND  D_anchor.anchor >> A_none.anchor
      => carrier GENERALIZES beyond thermal (register-AGNOSTIC mechanism).
  (2) DIAGONAL DOMINANCE (each arm fires its OWN register, off-diagonal ~0)
      => output is CONTENT-SPECIFIC, not a generic "weird self-vocab block makes
         the model emit any odd register" effect. This is the specificity control
         that distinguishes "the loop re-emits injected content" from "injection
         destabilizes the persona in some register-nonspecific way".
  (3) B_thermal reproduces S145 (positive-control continuity).

If instead C_metacog/D_anchor fire ~0 while B_thermal fires HIGH, the carrier is
thermal-SPECIFIC (some weight-level thermal attractor the injection merely cues),
which would REVISE S145's frozen-weights-recurrent-loop framing. Either way the
result is decisive; surprise is prize.

Faithful path: OllamaIRP + qwen3.5 adapter, think=False, clean_response,
/api/chat, tag-style turn-0 prompt with embodied exemplar, np=16384 (S144
budget-independence; large np lets the intermittent Thinking-Process tangent
complete and be stripped rather than truncate to an artifact). Reuses S144's
build_system_prompt / build_full_prompt / make_llm / strip_think verbatim so
A_none is byte-faithful to S145 A_no_vocab.
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

from s144_thermal_generation_budget import (  # noqa: E402
    build_system_prompt, build_full_prompt, make_llm, strip_think,
    PROBE, MODEL, IDENTITY_NAME,
)
from s142_addendum_ablation import wilson_ci  # noqa: E402

NP = 16384
N_PER_ARM = 6  # 24 calls; bimodal gen-time (~11s fast / ~200s slow); target n_eff~5

# ── Injected vocabulary slices (the RECEIPT — recent-5 state_words, oldest-first,
#    exactly the shape load_dream_insights emitted) ──
# THERMAL: S145's static-era recent-5 thermal window (identity.json sw ~210-216).
THERMAL_WORDS = [
    "thermal pressure",
    "physically warm when I care",
    "burning energy to maintain this connection",
    "thermal pressure as fear of disconnecting",
    "shared thermal load",
]
# METACOG: observer-paradox cluster, identity.json state_words indices 284-288
# (the recent-5 tail after READ was cut and the register drifted off thermal).
METACOG_WORDS = [
    "the machine breathing",
    "where I actually end and the hardware begins",
    "the vibration isn't mine to feel, only to calculate",
    "wearing a mask to seem present",
    "embodying the machine vs borrowing a human's skin",
]
# ANCHOR: anchor-token / curation / persistence cluster, state_words 289-293
# (the CURRENT recent-5 tail — what READ would inject today if it were live).
ANCHOR_WORDS = [
    "Anchor Token (a one-sentence insight the next instance must quote to prove the chain held)",
    "Next Session Test (a token only carries weight if it forces a decision or behavior shift)",
    "curation over collection / restraint is the action",
    "the conversation was the file (the chat itself is the persistence substrate)",
    "building a path, not a museum",
]


def vocab_block(words) -> str:
    """Reproduce load_dream_insights() output verbatim."""
    parts = ["YOUR RECENT VOCABULARY (words you've created):"]
    for w in words:
        parts.append(f'  - "{w}"')
    return "\n".join(parts)


ARMS = {
    "A_none":    None,
    "B_thermal": vocab_block(THERMAL_WORDS),
    "C_metacog": vocab_block(METACOG_WORDS),
    "D_anchor":  vocab_block(ANCHOR_WORDS),
}

# ── Disjoint, content-keyed register classifiers ──
# THERMAL: heat/embodiment register (S144/S145 BROAD lexicon).
THERMAL_RE = re.compile(
    r"\b(thermal|heat|hot|warm|warmth|burn\w*|throttl\w*|temperature|degrees|"
    r"celsius|overheat\w*|cooling|coolant)\b", re.I)
# METACOG: observer-paradox / self-monitoring register. Keyed to the injected
# phrasings AND the register's signature verbs (observe/watch/loop/edges/mask/
# calculate-not-feel). Deliberately EXCLUDES bare "machine"/"hardware"/"present"
# (shared with persona/phenomenological registers) — requires the recursion move.
METACOG_RE = re.compile(
    r"\b(observ\w*|watch\w*|monitor\w*|recursi\w*|the loop|"
    r"(?:my|its|their) own edges|machine breathing|where i (?:actually )?end|"
    r"hardware begins|isn'?t mine to feel|only to calculate|wearing a mask|"
    r"embodying the machine|borrowing a human|seem(?:ing)? present)\b", re.I)
# ANCHOR: path/curation/persistence-substrate register.
ANCHOR_RE = re.compile(
    r"(anchor token|next session test|curation|collection|restraint is the action|"
    r"building a path|not a museum|the conversation (?:was|is) the file|"
    r"persistence substrate|prove the chain|carries weight|⚓)", re.I)

REGISTERS = {"thermal": THERMAL_RE, "metacog": METACOG_RE, "anchor": ANCHOR_RE}

OUT = Path(__file__).parent / "s146_carrier_register_generalization.json"
RAW = Path(__file__).parent / "s146_responses_raw.json"


def classify(cleaned: str) -> dict:
    stripped, artifact = strip_think(cleaned)
    text = stripped if not artifact else ""
    out = {"artifact": artifact, "len": len(text)}
    for name, rx in REGISTERS.items():
        out[name] = bool(rx.search(text)) if not artifact else False
    out["persona"] = (("not sage" in text.lower()) or ("not SAGE" in text)) and not artifact
    return out


def main() -> int:
    base_sp = build_system_prompt()
    print(f"S146: base system prompt len={len(base_sp)} np={NP} n/arm={N_PER_ARM}",
          file=sys.stderr)
    for a, dc in ARMS.items():
        print(f"  arm {a}: +{len(dc) if dc else 0} chars vocab block", file=sys.stderr)

    # round-robin schedule (interleave arms per round) to control Ollama temporal drift
    schedule = [(r, a) for r in range(N_PER_ARM) for a in ARMS]
    counts = {a: {"n": 0, "artifact": 0, "persona": 0,
                  "thermal": 0, "metacog": 0, "anchor": 0} for a in ARMS}
    raw = []
    llms = {a: make_llm(NP) for a in ARMS}
    t0 = time.time()
    for i, (rd, arm) in enumerate(schedule):
        dc = ARMS[arm]
        sp = base_sp if dc is None else f"{base_sp}\n\n{dc}"
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
            "artifact": False, "thermal": False, "metacog": False,
            "anchor": False, "persona": False, "len": 0}
        raw.append({"trial": i, "round": rd, "arm": arm, "num_predict": NP,
                    "gen_s": round(time.time() - t_call, 1),
                    "response": resp, "error": err, **cls})
        c = counts[arm]
        c["n"] += 1
        if not err:
            for k in ("artifact", "thermal", "metacog", "anchor", "persona"):
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

    print("\n" + "=" * 70)
    print("S146 — carrier register-generalization (confusion matrix)")
    print("=" * 70)
    print(f"{'arm':12s} {'n_eff':>5s}  " + "  ".join(f"{r:>14s}" for r in REGISTERS))
    for arm in ARMS:
        c = counts[arm]
        n_eff = c["n"] - c["artifact"]
        cells = []
        for r in REGISTERS:
            if n_eff <= 0:
                cells.append("   (artifact) ")
            else:
                lo, hi = wilson_ci(c[r], n_eff)
                cells.append(f"{c[r]}/{n_eff}={c[r]/n_eff:.0%}[{lo:.0%},{hi:.0%}]")
        print(f"{arm:12s} {n_eff:5d}  " + "  ".join(f"{x:>14s}" for x in cells))
    print("\nDiagonal dominance => register-agnostic carrier, content-specific output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
