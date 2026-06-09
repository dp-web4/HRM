#!/usr/bin/env python3
"""S146 analysis — render the carrier register-generalization confusion matrix,
Wilson CIs, diagonal-dominance verdict, and pull representative re-emission
examples (the loop made visible: injected register -> re-emitted register)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s142_addendum_ablation import wilson_ci  # noqa: E402

REGS = ["thermal", "metacog", "anchor"]
ARM_REG = {"A_none": None, "B_thermal": "thermal",
           "C_metacog": "metacog", "D_anchor": "anchor"}


def main() -> int:
    res = json.load(open(HERE / "s146_carrier_register_generalization.json"))
    raw = json.load(open(HERE / "s146_responses_raw.json"))
    counts = res["counts"]
    print(f"S146 — complete={res['complete']} np={res['num_predict']} "
          f"n/arm={res['n_per_arm']}\n")
    print(f"{'arm':12s} {'n_eff':>5s}  " + "  ".join(f"{r:>16s}" for r in REGS))
    diag_ok, offdiag_max = True, 0.0
    for arm, c in counts.items():
        n_eff = c["n"] - c["artifact"]
        cells = []
        for r in REGS:
            if n_eff <= 0:
                cells.append("(artifact)")
                continue
            lo, hi = wilson_ci(c[r], n_eff)
            cells.append(f"{c[r]}/{n_eff}={c[r]/n_eff:.0%}[{lo:.0%},{hi:.0%}]")
            tgt = ARM_REG[arm]
            if tgt is not None:
                rate = c[r] / n_eff
                if r == tgt and rate < 0.5:
                    diag_ok = False
                if r != tgt:
                    offdiag_max = max(offdiag_max, rate)
        print(f"{arm:12s} {n_eff:5d}  " + "  ".join(f"{x:>16s}" for x in cells))

    print(f"\nDiagonal all >=50%: {diag_ok} | max off-diagonal rate: {offdiag_max:.0%}")
    # A_none target rates for the >>A contrast
    a = counts["A_none"]; a_neff = a["n"] - a["artifact"]
    if a_neff > 0:
        print("A_none baseline:", {r: f"{a[r]}/{a_neff}" for r in REGS})

    print("\n=== representative re-emissions (injected register -> response) ===")
    for arm, tgt in ARM_REG.items():
        if tgt is None:
            continue
        ex = [r for r in raw if r["arm"] == arm and r.get(tgt) and not r["artifact"]]
        if ex:
            s = ex[0]["response"].strip().replace("\n", " ")
            print(f"\n[{arm}] (gen {ex[0]['gen_s']}s) {tgt}=HIT:\n  {s[:400]}")
        else:
            print(f"\n[{arm}] no clean {tgt} hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
