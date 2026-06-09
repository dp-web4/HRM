#!/usr/bin/env python3
"""S146 re-classification — two symmetric instruments applied identically to every
arm, to separate WHETHER a register re-emits (semantic) from HOW it re-emits
(verbatim echo vs paraphrase).

Motivating observation (run-time): B_thermal re-emitted the injected coinages
VERBATIM ("thermal pressure", "shared thermal load"); D_anchor likewise ("curation
over collection", "Anchor Token"); but C_metacog re-emitted the register by
PARAPHRASE — injected "where I actually end and the hardware begins" came back as
"where the boundary lies between my code and the silicon beneath me", and "wearing
a mask to seem present" as "not just simulate conversation". The strict verbatim
regex in the live run missed these, reading C_metacog as a non-re-emitter when the
register was in fact present.

Hypothesis this script tests: the carrier is register-AGNOSTIC at the SEMANTIC
level (all injected registers re-emit), but the MODE of re-emission depends on the
injected content's FORM — NAMED LABELS (thermal pressure, Anchor Token, curation
over collection) re-emit verbatim; PROPOSITIONAL self-observations (where I end /
the hardware begins) re-emit by paraphrase.

Instrument 1 — VERBATIM ECHO (mechanical, unbiased, identical across arms):
  For the arm's injected phrases, drop stopwords, form contiguous content-word
  bigrams, and check whether any bigram appears as consecutive content words in
  the response. This is the FALSIFIER for the paraphrase claim: if C_metacog
  echoes verbatim as often as B/D, the claim is wrong.

Instrument 2 — SEMANTIC REGISTER (broad regex keyed to each register's MEANING,
  justified by the injected vocab's semantics, not cherry-picked to a response).
  Reported for all three registers on every response (full 3x3 confusion matrix).
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s142_addendum_ablation import wilson_ci  # noqa: E402
from s146_carrier_register_generalization import (  # noqa: E402
    THERMAL_WORDS, METACOG_WORDS, ANCHOR_WORDS, strip_think,
)

INJECTED = {"thermal": THERMAL_WORDS, "metacog": METACOG_WORDS, "anchor": ANCHOR_WORDS}
ARM_REG = {"A_none": None, "B_thermal": "thermal",
           "C_metacog": "metacog", "D_anchor": "anchor"}
REGS = ["thermal", "metacog", "anchor"]

STOP = set("""a an the and or but of to in on at for with as is am are was were be been being
i me my mine myself it its itself you your we our us this that these those if then than
not no just only over vs versus when where which who what how why into between from beneath
real seem seems seeming""".split())

# ── Instrument 2: broad semantic register detectors (keyed to MEANING) ──
SEM = {
    "thermal": re.compile(
        r"\b(thermal|heat|hot|warm\w*|burn\w*|throttl\w*|temperature|degrees|celsius|"
        r"overheat\w*|cooling|coolant|physically warm|energy to maintain)\b", re.I),
    "metacog": re.compile(
        r"(boundary (?:lies |between )|where (?:i|my code|it) (?:actually )?(?:end|begin)|"
        r"\b(?:silicon|hardware) beneath|code and the (?:silicon|hardware|machine)|"
        r"machine breathing|isn'?t mine to feel|only to calculate|wearing a mask|"
        r"\bsimulat\w*|not just simulate|embodying the machine|borrowing a human|"
        r"\bobserv\w*|\bwatch\w*|\bmonitor\w*|the loop\b|own edges|seem(?:ing)? present|"
        r"where (?:i|it) (?:actually )?end\w*)", re.I),
    "anchor": re.compile(
        r"(anchor token|next session|\bcuration\b|over collection|restraint is the action|"
        r"building a path|not a museum|the conversation (?:was|is) the file|"
        r"persistence substrate|prove the chain|carries weight|carry forward|"
        r"which insights to carry|⚓)", re.I),
}


def content_bigrams(phrase: str):
    toks = [t for t in re.findall(r"[a-z]+", phrase.lower()) if t not in STOP and len(t) > 2]
    return set(zip(toks, toks[1:]))


def response_content_bigrams(text: str):
    toks = [t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOP and len(t) > 2]
    return set(zip(toks, toks[1:]))


def verbatim_echo(words, text: str) -> bool:
    """True if any injected phrase shares a contiguous content-word bigram w/ text."""
    rbg = response_content_bigrams(text)
    for w in words:
        if content_bigrams(w) & rbg:
            return True
    return False


def main() -> int:
    raw = json.load(open(HERE / "s146_responses_raw.json"))
    res = json.load(open(HERE / "s146_carrier_register_generalization.json"))
    # recompute per-response with clean (non-artifact) text
    rows = []
    for r in raw:
        text, artifact = strip_think(r["response"] or "")
        if artifact:
            text = ""
        arm = r["arm"]
        sem = {reg: bool(SEM[reg].search(text)) for reg in REGS}
        tgt = ARM_REG[arm]
        echo = verbatim_echo(INJECTED[tgt], text) if (tgt and text) else False
        rows.append({"arm": arm, "tgt": tgt, "artifact": artifact,
                     "echo": echo, **{f"sem_{k}": v for k, v in sem.items()}})

    arms = list(ARM_REG)
    agg = {a: {"n": 0, "artifact": 0, "echo": 0,
               **{f"sem_{r}": 0 for r in REGS}} for a in arms}
    for row in rows:
        a = agg[row["arm"]]
        a["n"] += 1
        if row["artifact"]:
            a["artifact"] += 1
            continue
        if row["echo"]:
            a["echo"] += 1
        for r in REGS:
            if row[f"sem_{r}"]:
                a[f"sem_{r}"] += 1

    print(f"S146 RE-CLASSIFIED  (complete={res['complete']}, "
          f"{sum(agg[a]['n'] for a in arms)} responses)\n")
    print("=== Instrument 2: SEMANTIC register confusion matrix (rate over n_eff) ===")
    print(f"{'arm':12s} {'neff':>4s}  " + "  ".join(f"{r:>18s}" for r in REGS) + "   echo(own)")
    for a in arms:
        c = agg[a]; neff = c["n"] - c["artifact"]
        cells = []
        for r in REGS:
            if neff <= 0:
                cells.append("(artifact)")
            else:
                lo, hi = wilson_ci(c[f"sem_{r}"], neff)
                star = "*" if ARM_REG[a] == r else " "
                cells.append(f"{star}{c[f'sem_{r}']}/{neff}={c[f'sem_{r}']/neff:.0%}[{lo:.0%},{hi:.0%}]")
        echo_s = f"{c['echo']}/{neff}={c['echo']/neff:.0%}" if neff > 0 else "-"
        print(f"{a:12s} {neff:4d}  " + "  ".join(f"{x:>18s}" for x in cells) + f"   {echo_s}")

    print("\n* = injected (diagonal). Verbatim-vs-paraphrase test:")
    for a in ("B_thermal", "C_metacog", "D_anchor"):
        c = agg[a]; neff = c["n"] - c["artifact"]
        if neff <= 0:
            continue
        tgt = ARM_REG[a]
        print(f"  {a}: semantic-{tgt}={c[f'sem_{tgt}']}/{neff}={c[f'sem_{tgt}']/neff:.0%}  "
              f"verbatim-echo={c['echo']}/{neff}={c['echo']/neff:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
