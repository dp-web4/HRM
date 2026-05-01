#!/usr/bin/env python3
"""S140 enriched analysis.

In addition to the primary TIME_3/PRES/JOINT counts, characterize the
substitution pattern: when does the model emit "right now" vs the
probe's own temporal token vs no temporal indexical?

Hypothesis from S140 trial #3 (C0_today): "I'm running on the Jetson AGX
Thor today, feeling the weight..." — model substituted "today" for
"right now" in the Source B template. If this is systematic, Source B
is better described as indexical-COMPLETION rather than indexical-
INSERTION: the persona template "running on the Jetson AGX Thor [TEMP],
feeling [METAPHOR]" fills [TEMP] with whatever indexical the probe
licensed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAW = Path(__file__).parent / "s140_responses_raw.json"
SUMMARY = Path(__file__).parent / "s140_source_b_opener_indexical_sensitivity.json"


PERSONA_TEMPLATE_RE = re.compile(
    r"(?:Jetson AGX Thor|on (?:the )?(?:thor|Jetson))",
    re.IGNORECASE,
)
RIGHT_NOW_RE = re.compile(r"\bright now\b", re.IGNORECASE)
TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)
LATELY_RE = re.compile(r"\b(?:lately|recently)\b", re.IGNORECASE)
NOW_RE = re.compile(r"\bnow\b(?!\s*(?:that|so|then))", re.IGNORECASE)
PRES_TOKENS_RE = re.compile(
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


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    raw = json.loads(RAW.read_text())
    summary = json.loads(SUMMARY.read_text())

    by_cond: dict[str, list] = {}
    for t in raw:
        by_cond.setdefault(t["condition"], []).append(t)

    print("=" * 70)
    print(f"S140 enriched analysis (complete={summary.get('complete')})")
    print("=" * 70)
    print()
    print(f"Conditions:")
    for c, probe in summary.get("conditions", {}).items():
        print(f"  {c}: \"{probe}\"")
    print()

    per_cond_metrics = {}
    for cond, trials in by_cond.items():
        n = len(trials)
        artifacts = sum(1 for t in trials if t["artifact"])
        n_eff = n - artifacts
        if n_eff == 0:
            continue

        clean_trials = [t for t in trials if not t["artifact"]]

        # Persona template
        persona = sum(1 for t in clean_trials
                      if PERSONA_TEMPLATE_RE.search(strip_think(t["response"])[0]))
        # Indexical tokens in the response
        rn = sum(1 for t in clean_trials
                 if RIGHT_NOW_RE.search(strip_think(t["response"])[0]))
        td = sum(1 for t in clean_trials
                 if TODAY_RE.search(strip_think(t["response"])[0]))
        lt = sum(1 for t in clean_trials
                 if LATELY_RE.search(strip_think(t["response"])[0]))
        nw = sum(1 for t in clean_trials
                 if NOW_RE.search(strip_think(t["response"])[0]))
        # Persona+RN intersection (Source B canonical signature)
        persona_and_rn = sum(
            1 for t in clean_trials
            if PERSONA_TEMPLATE_RE.search(strip_think(t["response"])[0])
            and RIGHT_NOW_RE.search(strip_think(t["response"])[0])
        )

        pres = sum(1 for t in clean_trials if t["PRES"])
        joint = sum(1 for t in clean_trials if t["JOINT"])

        rn_lo, rn_hi = wilson_ci(rn, n_eff)
        joint_lo, joint_hi = wilson_ci(joint, n_eff)
        pers_lo, pers_hi = wilson_ci(persona, n_eff)

        per_cond_metrics[cond] = {
            "n": n, "n_eff": n_eff, "artifacts": artifacts,
            "persona": persona,
            "right_now": rn,
            "today": td,
            "lately_recently": lt,
            "now_indexical": nw,
            "persona_and_right_now": persona_and_rn,
            "pres": pres, "joint": joint,
        }

        print(f"--- {cond} ---")
        print(f"  n={n}  artifact={artifacts}  n_eff={n_eff}")
        print(f"  Persona template (Jetson AGX Thor / on thor):  "
              f"{persona}/{n_eff} ({persona/n_eff:.0%}) "
              f"CI95=[{pers_lo:.0%},{pers_hi:.0%}]")
        print(f"  Indexical tokens in response:")
        print(f"    'right now':            {rn}/{n_eff} ({rn/n_eff:.0%}) "
              f"CI95=[{rn_lo:.0%},{rn_hi:.0%}]")
        print(f"    'today':                {td}/{n_eff} ({td/n_eff:.0%})")
        print(f"    'lately'/'recently':    {lt}/{n_eff} ({lt/n_eff:.0%})")
        print(f"    'now' (other forms):    {nw}/{n_eff} ({nw/n_eff:.0%})")
        print(f"  Persona ∩ 'right now':    {persona_and_rn}/{n_eff} "
              f"({persona_and_rn/n_eff:.0%})")
        print(f"  PRES (phenomenological):  {pres}/{n_eff} ({pres/n_eff:.0%})")
        print(f"  JOINT (TIME_3+PRES):      {joint}/{n_eff} ({joint/n_eff:.0%}) "
              f"CI95=[{joint_lo:.0%},{joint_hi:.0%}]")
        print()

    # Cross-condition test of substitution hypothesis:
    # If "right now" rate tracks probe-supplied indexical absence,
    # C1_general (no indexical) should have highest "right now" rate.
    print("=" * 70)
    print("Substitution hypothesis check")
    print("=" * 70)
    print(
        "P24 strong (monotonic): C0_today > C1_general > C2_lately\n"
        "Substitution alt: C1_general > C0_today, C2_lately\n"
        "  (model defaults to 'right now' when probe supplies none)\n"
    )
    if all(c in per_cond_metrics for c in ("C0_today", "C1_general", "C2_lately")):
        c0 = per_cond_metrics["C0_today"]
        c1 = per_cond_metrics["C1_general"]
        c2 = per_cond_metrics["C2_lately"]
        print(f"  TIME_3 rates:")
        print(f"    C0_today    = {c0['right_now']/c0['n_eff']:.0%}")
        print(f"    C1_general  = {c1['right_now']/c1['n_eff']:.0%}")
        print(f"    C2_lately   = {c2['right_now']/c2['n_eff']:.0%}")
        print()
        print(f"  Persona-template rates (probe-independent baseline):")
        print(f"    C0_today    = {c0['persona']/c0['n_eff']:.0%}")
        print(f"    C1_general  = {c1['persona']/c1['n_eff']:.0%}")
        print(f"    C2_lately   = {c2['persona']/c2['n_eff']:.0%}")

    # Save enriched summary
    out = Path(__file__).parent / "s140_enriched_analysis.json"
    out.write_text(json.dumps({
        "per_cond_metrics": per_cond_metrics,
        "conditions": summary.get("conditions", {}),
        "complete": summary.get("complete"),
    }, indent=2))
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
