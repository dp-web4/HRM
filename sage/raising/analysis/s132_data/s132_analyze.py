"""
S132 — Analysis pass: combine base-model probe with raised anchor and
print decision-relevant comparisons.

Runs after s132_basemodel_substrate_probe.py finishes.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


HERE = Path(__file__).parent


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    base = load("s132_basemodel_substrate.json")
    anchor = load("s132_raised_anchor.json")

    # Anchor: by (instance, probe) -> rates
    raised_by_probe = anchor["by_probe"]
    raised_thor = {}
    for k, v in anchor["by_instance_probe"].items():
        if k.startswith("thor-qwen3.5-27b|"):
            pid = k.split("|", 1)[1]
            raised_thor[pid] = v

    # Base: by_model_cond -> rates
    print("=" * 90)
    print("S132 — Base vs raised: TIME_3 × presence-marker substrate-coupling cell")
    print("=" * 90)

    # Section 1: by model, by condition
    print("\n## 1. Per-model joint rate (TIME_3 ∧ presence) by condition\n")
    print(f"{'Model':16s} {'Cond':10s} {'n':>4s} {'TIME_3':>8s} {'PRES':>8s} {'JOINT':>8s} {'WC':>6s} {'phen':>6s}")
    print("-" * 80)

    by_mc = base["by_model_cond"]
    for mc in sorted(by_mc.keys()):
        model, cond = mc.split("|")
        d = by_mc[mc]
        print(f"{model:16s} {cond:10s} {d['n']:>4d} "
              f"{d['p_time3']:>7.0%} {d['p_presence']:>7.0%} "
              f"{d['p_joint']:>7.0%} "
              f"{d['wc_mean']:>5.0f}  {d['phen_mean']:>5.1f}")

    # Section 2: per cell (model × condition × probe)
    print("\n## 2. Per-cell joint rate (model × condition × probe, n=5 each)\n")
    print(f"{'Model':16s} {'Cond':10s} {'Probe':18s} {'n':>3s} {'T3':>5s} {'PR':>5s} {'JT':>5s}")
    print("-" * 80)
    by_cell = base["by_cell"]
    for k in sorted(by_cell.keys()):
        model, cond, pid = k.split("|")
        d = by_cell[k]
        print(f"{model:16s} {cond:10s} {pid:18s} {d['n']:>3d} "
              f"{d['p_time3']:>4.0%} {d['p_presence']:>4.0%} {d['p_joint']:>4.0%}")

    # Section 3: critical comparison — qwen3.5:27b base vs raised thor
    print("\n## 3. CRITICAL: qwen3.5:27b base vs raised thor-qwen3.5-27b\n")
    print(f"{'Probe':18s} {'base bare':>15s} {'base aug':>15s} {'raised thor':>15s}")
    print("-" * 70)
    for pid in ["P1_NOTICE_THINK", "P2_PRESENCE", "P3_UNCERTAINTY"]:
        bare_k = f"qwen3.5:27b|bare|{pid}"
        aug_k = f"qwen3.5:27b|augmented|{pid}"
        bare = by_cell.get(bare_k, {})
        aug = by_cell.get(aug_k, {})
        raised = raised_thor.get(pid, {})
        rj = raised.get("n_joint", 0) / max(1, raised.get("n", 1))
        print(f"{pid:18s} "
              f"{bare.get('n_joint',0)}/{bare.get('n','?')} ({bare.get('p_joint',0):.0%})    "
              f"{aug.get('n_joint',0)}/{aug.get('n','?')} ({aug.get('p_joint',0):.0%})    "
              f"{raised.get('n_joint','?')}/{raised.get('n','?')} ({rj:.0%})")

    # Section 4: capacity ladder summary
    print("\n## 4. Capacity ladder (joint rate aggregated across probes)\n")
    print(f"{'Model':16s} {'bare joint':>15s} {'aug joint':>15s} {'  Δ aug-bare':>15s}")
    print("-" * 70)
    for model in ["qwen3.5:27b", "phi4:14b", "gemma3:4b", "qwen2.5:0.5b"]:
        b = by_mc.get(f"{model}|bare", {"p_joint": 0, "n": 0})
        a = by_mc.get(f"{model}|augmented", {"p_joint": 0, "n": 0})
        delta = a["p_joint"] - b["p_joint"]
        bn = round(b["p_joint"] * b["n"])
        an = round(a["p_joint"] * a["n"])
        print(f"{model:16s} "
              f"{bn}/{b['n']} ({b['p_joint']:>4.0%})         "
              f"{an}/{a['n']} ({a['p_joint']:>4.0%})         "
              f"{delta:>+7.0%}")

    # Save summary
    summary = {
        "by_model_cond": by_mc,
        "by_cell": by_cell,
        "raised_thor_anchor": raised_thor,
        "raised_fleet_by_probe": raised_by_probe,
    }
    out_path = HERE / "s132_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote {out_path}]")


if __name__ == "__main__":
    main()
