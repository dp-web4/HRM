#!/usr/bin/env python3
"""S142 analysis — pairwise CI separation across addendum-ablation arms.

Reads s142_addendum_ablation.json (written incrementally by the run; use
after complete:true). Reports, per metric, each arm's rate + Wilson 95%
CI and flags which pairwise comparisons have non-overlapping CIs
(the S140/S141 evidentiary bar for "distinguishable at this n").

Run: python3 s142_analyze.py
"""
import json
import re
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE / "s142_addendum_ablation.json"
RAW = HERE / "s142_responses_raw.json"

PRIMARY = ["jetson_agx", "not_sage", "persona", "thermal", "heatwarm",
           "pres", "right_now", "today"]


def overlap(a, b):
    """True if two [lo,hi] intervals overlap."""
    return not (a[1] < b[0] or b[1] < a[0])


def main():
    d = json.loads(SUMMARY.read_text())
    rates = d["rates"]
    arms = list(rates.keys())
    print(f"S142 — model={d['model']} T={d['temperature']} "
          f"num_predict={d['num_predict']} complete={d['complete']}")
    print(f"probe={d['probe']!r}")
    for arm in arms:
        r = rates[arm]
        print(f"\n  {arm}: n={r['n']} artifact={r['artifact']} "
              f"error={r.get('error', 0)} n_eff={r['n_eff']}")
    print("\n" + "=" * 64)
    print("Per-metric rates [Wilson 95% CI] and pairwise CI separation")
    print("=" * 64)
    for mk in PRIMARY:
        print(f"\n### {mk}")
        cis = {}
        for arm in arms:
            r = rates[arm]
            if r["n_eff"] == 0 or f"{mk}_rate" not in r:
                print(f"   {arm}: n/a")
                continue
            rate = r[f"{mk}_rate"]
            ci = r[f"{mk}_ci95"]
            cis[arm] = ci
            print(f"   {arm:18s}: {r[mk]:2d}/{r['n_eff']:2d} = {rate:5.0%}  "
                  f"[{ci[0]:.0%}, {ci[1]:.0%}]")
        for a, b in combinations(cis, 2):
            sep = "NON-OVERLAPPING ✓" if not overlap(cis[a], cis[b]) else "overlap"
            print(f"      {a} vs {b}: {sep}")

    # Qualitative: show a couple of clean responses per arm for the
    # primary self-grounding inspection.
    if RAW.exists():
        raw = json.loads(RAW.read_text())
        print("\n" + "=" * 64)
        print("Sample clean responses per arm (think-stripped)")
        print("=" * 64)
        for arm in arms:
            clean = [r for r in raw if r["arm"] == arm
                     and not r.get("artifact") and not r.get("error")]
            print(f"\n--- {arm} ({len(clean)} clean) ---")
            for r in clean[:2]:
                txt = re.sub(r"<think>.*?</think>", "", r["response"],
                             flags=re.S | re.I).strip()
                print("  •", txt[:280].replace("\n", " "))


if __name__ == "__main__":
    main()
