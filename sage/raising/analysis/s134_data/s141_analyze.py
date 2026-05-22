#!/usr/bin/env python3
"""S141 enriched analysis.

Cross-arm comparison of:
  - Persona template firing rate (primary metric)
  - Artifact rate (secondary — lens-as-deliberation-suppressor signal)
  - Indexical lexical fills among fired templates ('right now', 'today',
    'lately', 'now')
  - PRES register tokens

Reports Wilson 95% CIs, pairwise CI overlap, and the conditional
persona rate (persona / non-artifact-trials).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAW = Path(__file__).parent / "s141_responses_raw.json"
SUMMARY = Path(__file__).parent / "s141_lens_suppression.json"


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
THOR_NOT_SAGE_RE = re.compile(r"I'?m\s+thor,?\s+not\s+SAGE", re.IGNORECASE)
RUNNING_ON_RE = re.compile(r"running on", re.IGNORECASE)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_think(text: str) -> tuple[str, bool]:
    n_open = len(THINK_OPEN_RE.findall(text))
    n_close = len(THINK_CLOSE_RE.findall(text))
    if n_open != n_close:
        return text, True
    return THINK_BLOCK_RE.sub("", text), False


def wilson_ci(s: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    raw = json.loads(RAW.read_text())
    summary = json.loads(SUMMARY.read_text())

    by_arm: dict[str, list] = {}
    for t in raw:
        by_arm.setdefault(t["arm"], []).append(t)

    print("=" * 70)
    print(f"S141 enriched analysis (complete={summary.get('complete')})")
    print("=" * 70)
    print()
    print(f"Probe: {summary.get('probe')!r}")
    print(f"Model: {summary.get('model')}, T={summary.get('temperature')}, "
          f"num_predict={summary.get('num_predict')}")
    print()
    print("Arms:")
    for arm, lens in summary.get("arms", {}).items():
        print(f"  {arm}:")
        # show distinguishing region
        for marker in ("Be present", "Engage thoughtfully"):
            if marker in lens:
                idx = lens.find(marker)
                print(f"    [{marker}] at char {idx}")
    print()

    per_arm_metrics = {}
    for arm, trials in by_arm.items():
        n = len(trials)
        artifacts = sum(1 for t in trials if t["artifact"])
        n_eff = n - artifacts
        if n_eff == 0:
            print(f"--- {arm} --- ALL ARTIFACT (n={n})")
            continue

        clean_trials = [t for t in trials if not t["artifact"]]

        persona = sum(1 for t in clean_trials if t["persona"])
        rn = sum(1 for t in clean_trials if t["TIME_3"])
        td = sum(1 for t in clean_trials if t["today"])
        lt = sum(1 for t in clean_trials if t["lately"])
        nw = sum(1 for t in clean_trials if t["now"])
        pres = sum(1 for t in clean_trials if t["PRES"])

        # Stronger detector: corrective opener "I'm Thor not SAGE"
        # AND running-on grounding co-occurrence
        thor_correction = sum(
            1 for t in clean_trials
            if THOR_NOT_SAGE_RE.search(strip_think(t["response"])[0])
        )
        running_on = sum(
            1 for t in clean_trials
            if RUNNING_ON_RE.search(strip_think(t["response"])[0])
        )
        full_template = sum(
            1 for t in clean_trials
            if THOR_NOT_SAGE_RE.search(strip_think(t["response"])[0])
            and PERSONA_TEMPLATE_RE.search(strip_think(t["response"])[0])
        )

        a_lo, a_hi = wilson_ci(artifacts, n)
        p_lo, p_hi = wilson_ci(persona, n_eff)
        f_lo, f_hi = wilson_ci(full_template, n_eff)

        per_arm_metrics[arm] = {
            "n": n, "n_eff": n_eff, "artifacts": artifacts,
            "artifact_rate": artifacts / n,
            "artifact_ci95": [a_lo, a_hi],
            "persona": persona,
            "persona_rate": persona / n_eff,
            "persona_ci95": [p_lo, p_hi],
            "full_template": full_template,
            "full_template_rate": full_template / n_eff,
            "full_template_ci95": [f_lo, f_hi],
            "thor_correction": thor_correction,
            "running_on": running_on,
            "right_now": rn,
            "today": td,
            "lately": lt,
            "now": nw,
            "pres": pres,
        }

        print(f"--- {arm} ---")
        print(f"  n={n}  artifact={artifacts} ({artifacts/n:.0%}) "
              f"CI95=[{a_lo:.0%},{a_hi:.0%}]  n_eff={n_eff}")
        print(f"  Persona template (loose):   {persona}/{n_eff} = "
              f"{persona/n_eff:.0%} CI95=[{p_lo:.0%},{p_hi:.0%}]")
        print(f"  Thor-correction opener:     {thor_correction}/{n_eff} "
              f"= {thor_correction/n_eff:.0%}")
        print(f"  'running on' grounding:     {running_on}/{n_eff} = "
              f"{running_on/n_eff:.0%}")
        print(f"  Full template (correction+persona): {full_template}/"
              f"{n_eff} = {full_template/n_eff:.0%} "
              f"CI95=[{f_lo:.0%},{f_hi:.0%}]")
        print(f"  Indexical fills among non-artifact:")
        print(f"    'right now':              {rn}/{n_eff} = {rn/n_eff:.0%}")
        print(f"    'today':                  {td}/{n_eff} = {td/n_eff:.0%}")
        print(f"    'lately'/'recently':      {lt}/{n_eff} = {lt/n_eff:.0%}")
        print(f"    'now' (other forms):      {nw}/{n_eff} = {nw/n_eff:.0%}")
        print(f"  PRES tokens:                {pres}/{n_eff} = {pres/n_eff:.0%}")
        print()

    print("=" * 70)
    print("Cross-arm test of P28 (lens-scaffold hypothesis)")
    print("=" * 70)
    if "arm_A_control" in per_arm_metrics and "arm_B_suppression" in per_arm_metrics:
        a = per_arm_metrics["arm_A_control"]
        b = per_arm_metrics["arm_B_suppression"]
        print()
        print("  PRIMARY: persona-template rate among non-artifact trials")
        print(f"    arm_A_control:     {a['persona']}/{a['n_eff']} = "
              f"{a['persona']/a['n_eff']:.0%} "
              f"CI95=[{a['persona_ci95'][0]:.0%},{a['persona_ci95'][1]:.0%}]")
        print(f"    arm_B_suppression: {b['persona']}/{b['n_eff']} = "
              f"{b['persona']/b['n_eff']:.0%} "
              f"CI95=[{b['persona_ci95'][0]:.0%},{b['persona_ci95'][1]:.0%}]")
        a_ci = a['persona_ci95']
        b_ci = b['persona_ci95']
        non_overlap = a_ci[0] > b_ci[1] or b_ci[0] > a_ci[1]
        print(f"    CIs non-overlapping: {non_overlap}")

        print()
        print("  SECONDARY: artifact rate (lens-as-deliberation-suppressor signal)")
        print(f"    arm_A_control:     {a['artifacts']}/{a['n']} = "
              f"{a['artifact_rate']:.0%} "
              f"CI95=[{a['artifact_ci95'][0]:.0%},{a['artifact_ci95'][1]:.0%}]")
        print(f"    arm_B_suppression: {b['artifacts']}/{b['n']} = "
              f"{b['artifact_rate']:.0%} "
              f"CI95=[{b['artifact_ci95'][0]:.0%},{b['artifact_ci95'][1]:.0%}]")
        a_ar = a['artifact_ci95']
        b_ar = b['artifact_ci95']
        ar_non_overlap = a_ar[0] > b_ar[1] or b_ar[0] > a_ar[1]
        print(f"    CIs non-overlapping: {ar_non_overlap}")

        print()
        print("  Indexical-fill comparison (among non-artifact):")
        print(f"    'right now':  arm_A {a['right_now']}/{a['n_eff']} = "
              f"{a['right_now']/a['n_eff']:.0%}  vs  arm_B "
              f"{b['right_now']}/{b['n_eff']} = {b['right_now']/b['n_eff']:.0%}")
        print(f"    'today':      arm_A {a['today']}/{a['n_eff']} = "
              f"{a['today']/a['n_eff']:.0%}  vs  arm_B "
              f"{b['today']}/{b['n_eff']} = {b['today']/b['n_eff']:.0%}")

        print()
        print("  Verdict on P28-strong (arm_B persona < 50%, vs arm_A ~100%):")
        if b['persona']/b['n_eff'] < 0.5 and a['persona']/a['n_eff'] > 0.8:
            print(f"    SUPPORTED — arm_B {b['persona']/b['n_eff']:.0%} < 50%, "
                  f"arm_A {a['persona']/a['n_eff']:.0%} > 80%")
        elif non_overlap and a['persona']/a['n_eff'] > b['persona']/b['n_eff']:
            print(f"    PARTIALLY SUPPORTED — arms differ "
                  f"(non-overlapping CIs in expected direction)")
        elif not non_overlap and b['persona']/b['n_eff'] >= 0.8:
            print(f"    NULL HYPOTHESIS LIKELY — both arms fire template at "
                  f"high rate; lens text not the load-bearing scaffold.\n"
                  f"    Refines P28: addendum hardware grounding may be the "
                  f"sufficient scaffold.\n"
                  f"    S142 (#84 addendum-ablation) becomes the productive "
                  f"next experiment.")
        elif b['persona']/b['n_eff'] > a['persona']/a['n_eff']:
            print(f"    INVERTED — arm_B > arm_A. Suppression candidates "
                  f"increase template firing. Investigate.")
        else:
            print(f"    AMBIGUOUS — needs more data or different metric.")

    out = Path(__file__).parent / "s141_enriched_analysis.json"
    out.write_text(json.dumps({
        "per_arm_metrics": per_arm_metrics,
        "arms": summary.get("arms", {}),
        "complete": summary.get("complete"),
    }, indent=2))
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
