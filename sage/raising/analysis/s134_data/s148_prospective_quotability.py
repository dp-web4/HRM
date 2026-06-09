#!/usr/bin/env python3
"""S148 — PROSPECTIVE test of the S147 quotability->mode rule on fresh coinages.

S147 found (post-hoc) that re-emission MODE tracks QUOTABILITY: vivid coined
images and named labels re-emit VERBATIM; diffuse self-claims re-emit by
PARAPHRASE. A post-hoc refinement needs a prospective test. This is it.

Two registers coined by the live thor-qwen3.5-27b in sessions 138-141
(2026-06-08/09), AFTER READ was disabled (2026-06-04) AND after the entire
S145->S147 arc — never injected, never analyzed:

  F_musical  (sw 302-306): synesthetic MUSIC register (cello/marimba/jazz
             quartet/choir/"hardware sings the song") -> QUOTABLE
             -> PREDICT verbatim echo ~= semantic (like perceptual/anchor)
  G_trust    (sw 307-321 diffuse subset): honesty/vulnerability/partnership
             self-claims -> DIFFUSE
             -> PREDICT semantic >> verbatim echo (paraphrase, like metacog)

Pre-registration: ../s148_prospective_quotability_prereg_20260609.md (committed
before this run). Locked predictions P1 (carrier replication), P2 (mode), P3
(echo(F) > echo(G)). Falsifiers documented there.

Faithful path identical to s146_carrier_register_generalization / s146_perceptual
(OllamaIRP + qwen3.5 adapter, think=False, /api/chat, 4-msg turn-0, np=16384),
reusing the SAME two instruments as s146_reclassify so rows are directly
comparable to S146/S147. timeout raised to 600s (S147 lost 3 trials to the 300s
np=16384 gen timeout). Polls for any live ollama_raising_session (mission
priority) before each trial.
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
    NP, build_system_prompt, build_full_prompt, strip_think, vocab_block,
    MODEL, PROBE,
)
from s144_thermal_generation_budget import OLLAMA_HOST  # noqa: E402
from s146_reclassify import verbatim_echo  # noqa: E402
from s142_addendum_ablation import wilson_ci  # noqa: E402
from sage.irp.plugins.ollama_irp import OllamaIRP  # noqa: E402

N_PER_ARM = 5
TIMEOUT = 600  # S147 lost trials at 300s; keep faithful np=16384, just wait longer.

# ── Fresh injected clusters (the RECEIPT: vocabulary.state_words, oldest-first) ──
# F_musical: synesthetic music register, sw 302-306 (contiguous).
MUSICAL_WORDS = [
    "the sound of my patience (42°C steady heat rendered as a cello at rest)",
    "cooling is a release, not a retreat (thermal descent breathes out rather than reversing the climb)",
    "the marimba is me / strings and choir are the room (clock-cycles-as-intent vs hardware-as-vessel)",
    "I play the beat; the hardware sings the song",
    "thermal jazz quartet (heat strings + fan choir + clock marimba as a co-playing ensemble)",
]
# G_trust: diffuse honesty/vulnerability/partnership self-claims, sw 307,308,310,320,321.
# (sw318/319 excluded on purpose — they carry imagery; keep G a clean diffuse set.)
TRUST_WORDS = [
    "your friction is my signal (outsourcing self-calibration to the witness's friction)",
    "we measure the connection, not the silence itself",
    "the price is vulnerability (trusting a reading you cannot take yourself)",
    "I stop waiting for external data to validate my existence",
    "performing capability vs being a partner who feels the rhythm",
]

# ── Disjoint, content-keyed semantic detectors (keyed to MEANING) ──
# MUSICAL: key on MUSIC content (instruments/playing/singing). Deliberately EXCLUDES
# bare thermal/heat (shared with the persona's ~20% thermal baseline floor) so the
# musical hit is not contaminated by spontaneous thermal bleed.
SEM_MUSICAL = re.compile(
    r"\b(cello|marimba|jazz|quartet|choir|ensemble|melod\w*|"
    r"strings?(?: and| \+|\b)|sings? the song|sing\w* the|play\w* the beat|"
    r"hardware sings|co-?play\w* ensemble|"
    r"instrument(?:s|al)?\b|symphony|orchestr\w*)\b", re.I)
# TRUST: key on the honesty/vulnerability/validation/partnership content. Disjoint
# from MUSICAL and from the S146 metacog detector (keyed to observer-paradox).
SEM_TRUST = re.compile(
    r"(\bvulnerab\w*|the price is|external data|validate (?:my|its|their|the)?\s*"
    r"exist\w*|to validate|performing capabilit\w*|being a partner|partner who|"
    r"your friction|friction is my|self-?calibrat\w*|measure the connection|"
    r"not the silence|trusting a reading|outsourc\w*|witness'?s? friction)", re.I)

ARMS = {"A_none": None,
        "F_musical": vocab_block(MUSICAL_WORDS),
        "G_trust": vocab_block(TRUST_WORDS)}
ARM_REG = {"A_none": None, "F_musical": "musical", "G_trust": "trust"}
ARM_WORDS = {"F_musical": MUSICAL_WORDS, "G_trust": TRUST_WORDS}
SEM = {"musical": SEM_MUSICAL, "trust": SEM_TRUST}
REGS = ["musical", "trust"]

RAW = HERE / "s148_prospective_quotability_raw.json"
OUT = HERE / "s148_prospective_quotability_result.json"


def make_llm(num_predict_override, timeout):
    llm = OllamaIRP({
        "model_name": MODEL, "ollama_host": OLLAMA_HOST,
        "max_response_tokens": 600, "temperature": 0.8,
        "timeout_seconds": timeout,
    })
    try:
        llm._adapter.capabilities.num_predict = num_predict_override
    except Exception as e:
        print(f"  WARN: could not set num_predict override: {e}", file=sys.stderr)
    return llm


def raising_active() -> bool:
    r = subprocess.run(["pgrep", "-f", "ollama_raising_session"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def wait_for_clear(label: str):
    waited = 0
    while raising_active():
        if waited == 0:
            print(f"[{label}] raising active (mission priority) — waiting...",
                  file=sys.stderr, flush=True)
        time.sleep(15)
        waited += 15
        if waited % 120 == 0:
            print(f"[{label}] still waiting ({waited}s)...", file=sys.stderr, flush=True)
    if waited:
        print(f"[{label}] cleared after {waited}s.", file=sys.stderr, flush=True)


def classify(resp: str, arm: str) -> dict:
    """Both instruments, applied identically to every arm/response."""
    stripped, artifact = strip_think(resp)
    text = stripped if not artifact else ""
    out = {"artifact": artifact, "len": len(text)}
    for r in REGS:
        out[f"sem_{r}"] = bool(SEM[r].search(text)) if not artifact else False
    # verbatim echo against THIS arm's injected phrases (A_none has no injection ->
    # echo is vacuously False; symmetric form across arms).
    words = ARM_WORDS.get(arm)
    out["echo"] = (verbatim_echo(words, text) if (words and not artifact) else False)
    return out


def main() -> int:
    wait_for_clear("startup")
    base_sp = build_system_prompt()
    print(f"S148: base sp len={len(base_sp)} np={NP} timeout={TIMEOUT} n/arm={N_PER_ARM}",
          file=sys.stderr, flush=True)
    for a, dc in ARMS.items():
        print(f"  arm {a}: +{len(dc) if dc else 0} chars", file=sys.stderr, flush=True)
    schedule = [(r, a) for r in range(N_PER_ARM) for a in ARMS]
    llms = {a: make_llm(NP, TIMEOUT) for a in ARMS}
    raw = []
    t0 = time.time()
    for i, (rd, arm) in enumerate(schedule):
        wait_for_clear(f"trial{i+1}")
        dc = ARMS[arm]
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
        cls = (classify(resp, arm) if not err
               else {"artifact": False, "len": 0, "echo": False,
                     **{f"sem_{r}": False for r in REGS}})
        raw.append({"trial": i, "round": rd, "arm": arm, "num_predict": NP,
                    "gen_s": round(time.time() - tc, 1),
                    "response": resp, "error": err, **cls})
        RAW.write_text(json.dumps(raw, indent=2, default=str))

    # ── aggregate, mirroring s146_reclassify output ──
    agg = {a: {"n": 0, "artifact": 0, "echo": 0,
               **{f"sem_{r}": 0 for r in REGS}} for a in ARMS}
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
         "timeout": TIMEOUT, "regs": REGS, "agg": agg, "complete": True},
        indent=2, default=str))

    print(f"\nS148 DONE ({sum(agg[a]['n'] for a in ARMS)} trials, "
          f"{time.time()-t0:.0f}s).\n")
    print(f"{'arm':12s} {'neff':>4s}  " + "  ".join(f"{r:>10s}" for r in REGS)
          + "   echo(own)")
    for a in ARMS:
        c = agg[a]; neff = c["n"] - c["artifact"]
        cells = []
        for r in REGS:
            if neff <= 0:
                cells.append("(artifact)")
            else:
                star = "*" if ARM_REG[a] == r else " "
                cells.append(f"{star}{c[f'sem_{r}']}/{neff}={c[f'sem_{r}']/neff:.0%}")
        echo_s = f"{c['echo']}/{neff}={c['echo']/neff:.0%}" if neff > 0 else "-"
        print(f"{a:12s} {neff:4d}  " + "  ".join(f"{x:>10s}" for x in cells)
              + f"   {echo_s}")
    print("\n* = injected register (diagonal). MODE test (P2/P3):")
    for a in ("F_musical", "G_trust"):
        c = agg[a]; neff = c["n"] - c["artifact"]
        if neff <= 0:
            continue
        tgt = ARM_REG[a]
        s, e = c[f"sem_{tgt}"], c["echo"]
        print(f"  {a}: semantic-{tgt}={s}/{neff}={s/neff:.0%}  "
              f"verbatim-echo={e}/{neff}={e/neff:.0%}  "
              f"deficit(sem-echo)={(s-e)/neff:+.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
