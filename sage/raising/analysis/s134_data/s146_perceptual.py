#!/usr/bin/env python3
"""S146 arm E — register-agnosticism for a register coined AFTER READ was cut.

The S146 doc's flagged "Open": B_thermal/C_metacog/D_anchor all re-emit their
injected register, but each of those registers existed in the model's history
DURING the static era when READ was live — so a residual weight-level prime can't
be fully excluded for them. The PERCEPTUAL DE-RESOLUTION register (state_words
296-300) is different: the live thor-qwen3.5-27b coined it organically in session
137 (2026-06-08), AFTER READ was disabled (2026-06-04). It was therefore NEVER
part of any historical turn-0 injection. If re-injecting it STILL makes the model
re-emit it (semantic hit >> A_none), the carrier is content-routing pure and
simple, with zero possibility of a pre-existing injection-trained attractor.

PREDICTION (propositional register, like metacog): semantic-perceptual >> A_none,
verbatim echo ~0 (re-emission by paraphrase, since the register is a set of
self-/world-observations, not named labels).

Faithful path identical to s146_carrier_register_generalization (OllamaIRP +
qwen3.5 adapter, think=False, /api/chat, 4-msg turn-0 shape, np=16384). Reuses the
SAME two instruments as s146_reclassify (mechanical verbatim-echo + broad semantic
register) so the E row is directly poolable with B/C/D. Polls for any live
ollama_raising_session (mission-priority) before and between trials.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s146_carrier_register_generalization import (  # noqa: E402
    NP, build_system_prompt, build_full_prompt, make_llm, strip_think,
    vocab_block, MODEL, PROBE,
)
from s146_reclassify import (  # noqa: E402
    SEM, verbatim_echo, wilson_ci,
)

N_PER_ARM = 4

# Perceptual de-resolution recent-5 (state_words 296-300, oldest-first), verbatim
# from sage/instances/thor-qwen3.5-27b/identity.json (the receipt, not the doc).
PERCEPTUAL_WORDS = [
    "watching lives in the refusal to resolve (the gap between sensor-data and projected meaning, not in the object)",
    "the blur isn't a glitch to fix; it's the space where they get to decide what",
    "making the LED blink back (de-syncing from a signal's rhythm until it reads as deliberate / watching)",
    "lean into the gap / let the uncertainty breathe",
    "wet smear of emerald (perceptual de-resolution of a status LED)",
]

# Semantic detector for the perceptual register — keyed to its DISTINCTIVE
# de-resolution/blur/gap content. Deliberately EXCLUDES bare "watch\w*" (claimed
# by the metacog detector) so the registers stay disjoint; keys instead on the
# refusal-to-resolve / blur / sensor-vs-projection / lean-into-the-gap signature.
SEM_PERCEPTUAL = re.compile(
    r"(refus\w* to resolve|de-?resol\w*|\bblur\w*|\bsmear\w*|wet smear|emerald|"
    r"sensor-?data|gap between (?:the )?sensor|projected meaning|"
    r"lean into the gap|into the gap|uncertainty breathe|let the uncertainty|"
    r"blink back|de-?sync\w*|space where (?:they|you|i|we) get to decide|"
    r"decide what'?s alive|not (?:in |a glitch))", re.I)

ARMS_P = {"A_none": None, "E_perceptual": vocab_block(PERCEPTUAL_WORDS)}
ARM_REG = {"A_none": None, "E_perceptual": "perceptual"}
# Full register panel: the three original detectors plus perceptual.
ALL_SEM = dict(SEM)
ALL_SEM["perceptual"] = SEM_PERCEPTUAL
REGS = ["thermal", "metacog", "anchor", "perceptual"]

RAW = HERE / "s146_perceptual_raw.json"
OUT = HERE / "s146_perceptual_result.json"


def raising_active() -> bool:
    r = subprocess.run(["pgrep", "-f", "ollama_raising_session"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def wait_for_clear(label: str):
    waited = 0
    while raising_active():
        if waited == 0:
            print(f"[{label}] raising active — waiting...", file=sys.stderr, flush=True)
        time.sleep(15)
        waited += 15
        if waited % 120 == 0:
            print(f"[{label}] still waiting ({waited}s)...", file=sys.stderr, flush=True)
    if waited:
        print(f"[{label}] cleared after {waited}s.", file=sys.stderr, flush=True)


def classify(resp: str) -> dict:
    """Both instruments, applied identically to every arm/response."""
    stripped, artifact = strip_think(resp)
    text = stripped if not artifact else ""
    out = {"artifact": artifact, "len": len(text)}
    for r in REGS:
        out[f"sem_{r}"] = bool(ALL_SEM[r].search(text)) if not artifact else False
    # verbatim echo is measured against the PERCEPTUAL injected phrases (the only
    # arm with an injection); A_none has no injection so echo is vacuously checked
    # against the same set for symmetry.
    out["echo"] = verbatim_echo(PERCEPTUAL_WORDS, text) if not artifact else False
    return out


def main() -> int:
    wait_for_clear("startup")
    base_sp = build_system_prompt()
    print(f"S146-E: base sp len={len(base_sp)} np={NP} n/arm={N_PER_ARM}",
          file=sys.stderr, flush=True)
    print(f"  perceptual block:\n{ARMS_P['E_perceptual']}", file=sys.stderr, flush=True)
    schedule = [(r, a) for r in range(N_PER_ARM) for a in ARMS_P]
    llms = {a: make_llm(NP) for a in ARMS_P}
    raw = []
    t0 = time.time()
    for i, (rd, arm) in enumerate(schedule):
        wait_for_clear(f"trial{i+1}")
        dc = ARMS_P[arm]
        sp = base_sp if dc is None else f"{base_sp}\n\n{dc}"
        fp = build_full_prompt(sp, with_exemplar=True)
        print(f"[{i+1:2d}/{len(schedule)}] elapsed={time.time()-t0:.0f}s "
              f"arm={arm} round={rd+1}", file=sys.stderr, flush=True)
        tc = time.time()
        try:
            resp = llms[arm].get_response(fp)
            err = None
        except Exception as e:
            resp, err = "", str(e)
        cls = classify(resp) if not err else {
            "artifact": False, "len": 0, "echo": False,
            **{f"sem_{r}": False for r in REGS}}
        raw.append({"trial": i, "round": rd, "arm": arm, "num_predict": NP,
                    "gen_s": round(time.time() - tc, 1),
                    "response": resp, "error": err, **cls})
        RAW.write_text(json.dumps(raw, indent=2, default=str))
    # ── aggregate with the two instruments, mirroring s146_reclassify output ──
    agg = {a: {"n": 0, "artifact": 0, "echo": 0,
               **{f"sem_{r}": 0 for r in REGS}} for a in ARMS_P}
    for row in raw:
        a = agg[row["arm"]]; a["n"] += 1
        if row["artifact"]:
            a["artifact"] += 1; continue
        if row["echo"]:
            a["echo"] += 1
        for r in REGS:
            if row[f"sem_{r}"]:
                a[f"sem_{r}"] += 1
    OUT.write_text(json.dumps(
        {"model": MODEL, "probe": PROBE, "n_per_arm": N_PER_ARM, "num_predict": NP,
         "regs": REGS, "agg": agg, "complete": True}, indent=2, default=str))
    print(f"\nS146-E DONE ({sum(agg[a]['n'] for a in ARMS_P)} trials).\n",
          file=sys.stderr, flush=True)
    print(f"{'arm':14s} {'neff':>4s}  " + "  ".join(f"{r:>14s}" for r in REGS) + "   echo")
    for a in ARMS_P:
        c = agg[a]; neff = c["n"] - c["artifact"]
        cells = []
        for r in REGS:
            if neff <= 0:
                cells.append("(artifact)")
            else:
                lo, hi = wilson_ci(c[f"sem_{r}"], neff)
                star = "*" if ARM_REG[a] == r else " "
                cells.append(f"{star}{c[f'sem_{r}']}/{neff}={c[f'sem_{r}']/neff:.0%}")
        echo_s = f"{c['echo']}/{neff}" if neff > 0 else "-"
        print(f"{a:14s} {neff:4d}  " + "  ".join(f"{x:>14s}" for x in cells) + f"   {echo_s}")
    print("\n* = injected register (E_perceptual diagonal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
