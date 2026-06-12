#!/usr/bin/env python3
"""S157 — position-confound re-check on S156 data (analysis-only, no new trials).

S155 carry-over: in S148 (greeting-only, n=5/arm) the within-menu position gradients
looked opposite across arms (last-item-wins in one arm, first-item-wins in another),
which left open a positional reading of "which phrase gets echoed". S156 ran the SAME
menus (same 5-phrase order per arm) under THREE probes, n=5/arm/probe — 15 trials per
arm, 3x the per-phrase n, with probe as a known massive selector.

Logic:
  - Menu order never varied. If position were the selector, the per-position echo
    profile should be stable across probes. S156's headline (zero cross-probe rank
    stability) already contradicts that; here we quantify the residual.
  - Tests: (1) per-arm rho(position, echo) pooled over probes; (2) per arm x probe
    gradients — sign consistency; (3) position profile vs slot-fit profile: how much
    of the per-cell variance does position explain after the probe x phrase cell
    means are known? (eta-style variance decomposition on the 45 phrase x probe cells)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s148_quotability_prospective import ARM_PHRASES, spearman  # noqa: E402

RAW = HERE / "s156_slotfit_raw.json"
OUT = HERE / "s157_position_check_result.json"

raw = [t for t in json.loads(RAW.read_text())
       if not t.get("error") and t["arm"] != "A_none"]
PROBES = sorted({t["probe"] for t in raw})
ARMS = sorted({t["arm"] for t in raw})

# phrase -> (arm, position) from the harness's literal menu order
pos_of = {}
for arm, phrases in ARM_PHRASES.items():
    for i, (p, _q) in enumerate(phrases):
        pos_of[p] = (arm, i)

# cell counts: (arm, probe, position) -> [echoes, n]
cells = {}
for t in raw:
    for p, info in t["per_phrase"].items():
        arm, i = pos_of[p]
        assert arm == t["arm"]
        k = (arm, t["probe"], i)
        e, n = cells.get(k, (0, 0))
        cells[k] = (e + bool(info["echo"]), n + 1)

result = {"n_trials": len(raw), "per_arm_pooled": {}, "per_arm_probe": {},
          "variance_decomposition": {}, "verdict": None}

print(f"S157 position check — {len(raw)} menu trials (S156 raw)")
print("\n== per-arm pooled over probes (n=15/phrase) ==")
pooled_rhos = []
for arm in ARMS:
    xs, ys = [], []
    for i in range(5):
        e = sum(cells[(arm, pk, i)][0] for pk in PROBES)
        n = sum(cells[(arm, pk, i)][1] for pk in PROBES)
        xs.append(i)
        ys.append(e / n)
    rho = spearman(xs, ys)
    pooled_rhos.append(rho)
    result["per_arm_pooled"][arm] = {"echo_rate_by_pos": ys, "rho_pos": rho}
    print(f"  {arm}: pos0..4 = {['%.2f' % y for y in ys]}  rho={rho:+.3f}")

print("\n== per arm x probe gradients (n=5/phrase) ==")
signs = []
for arm in ARMS:
    for pk in PROBES:
        ys = [cells[(arm, pk, i)][0] / cells[(arm, pk, i)][1] for i in range(5)]
        rho = spearman(list(range(5)), ys)
        result["per_arm_probe"][f"{arm}|{pk}"] = {"echo_by_pos": ys, "rho_pos": rho}
        if rho == rho and abs(rho) > 1e-9:  # not nan, not flat
            signs.append(rho > 0)
        print(f"  {arm:8s} {pk:12s}: {['%.1f' % y for y in ys]}  rho={rho:+.3f}")

n_pos = sum(signs)
print(f"\n  gradient signs: {n_pos} positive / {len(signs) - n_pos} negative "
      f"(of {len(signs)} non-flat cells)")

# variance decomposition over the 45 (phrase, probe) cell means
# grand mean, then SS explained by position-within-arm vs by (phrase,probe) cell id
cell_means, positions = [], []
for arm in ARMS:
    for pk in PROBES:
        for i in range(5):
            e, n = cells[(arm, pk, i)]
            cell_means.append(e / n)
            positions.append(i)
gm = sum(cell_means) / len(cell_means)
ss_tot = sum((v - gm) ** 2 for v in cell_means)
# position factor: mean echo per position (pooled over arm,probe)
pos_mean = {i: sum(v for v, p in zip(cell_means, positions) if p == i)
            / sum(1 for p in positions if p == i) for i in range(5)}
ss_pos = sum((pos_mean[p] - gm) ** 2 for p in positions)
result["variance_decomposition"] = {
    "ss_total": ss_tot, "ss_position": ss_pos,
    "eta2_position": ss_pos / ss_tot if ss_tot else None,
    "pos_marginal_means": [pos_mean[i] for i in range(5)],
}
print(f"\n== variance decomposition over 45 phrase x probe cells ==")
print(f"  position marginal means pos0..4: "
      f"{['%.2f' % pos_mean[i] for i in range(5)]}")
print(f"  eta^2(position) = {ss_pos / ss_tot:.3f}  "
      f"(share of cell-mean variance attributable to menu position)")

consistent = (n_pos == len(signs)) or (n_pos == 0)
eta2 = ss_pos / ss_tot if ss_tot else 0.0
if consistent and eta2 > 0.25:
    verdict = "POSITION GRADIENT REAL — consistent sign and large variance share"
elif eta2 < 0.10:
    verdict = ("POSITION CONFOUND DISMISSED — gradients sign-inconsistent across "
               "arm x probe and position explains <10% of cell variance")
else:
    verdict = "POSITION WEAK/MIXED — small variance share or inconsistent signs"
result["verdict"] = verdict
result["gradient_signs"] = {"positive": n_pos, "total_nonflat": len(signs)}
print(f"\nVERDICT: {verdict}")
OUT.write_text(json.dumps(result, indent=2))
print(f"wrote {OUT.name}")
