#!/usr/bin/env python3
"""S156 — PROSPECTIVE test of the SPEECH-ACT SLOT FIT hypothesis (prereg frozen S155).

S148 killed quotability: rho(Q, echo)=0.037 over 15 never-injected coinages, and the
key within-topic contrast (F_trust 307 q=2 vs 309 q=0) tied at 40%/40%. But the S148
per-arm winners were not random: phrases that slot directly into the GREETING probe
as a self-contained topic-NP won ("the lag between the command and the world
answering" 5/5; "thermal jazz quartet" 4/5) while phrases needing narrative
scaffolding lost ("the Gasp" 0/5, "cooling is a release" 0/5). That reading is
post-hoc on greeting data. S156 makes it prospective: SAME 15 phrases, SAME arms,
SAME faithful harness — vary ONLY the user probe across three speech-acts.

Prereg: s156_slotfit_prereg.json (frozen S155 2026-06-11, before any trial).
  P_greeting  — S148 probe verbatim (anchor/replication cell)
  P_task      — status-report speech-act ("how is the hardware holding up...")
  P_narrative — event-narration speech-act ("a moment when work suddenly arrived
                while you were at rest... from the inside")

PREDICTIONS (slot fit): 1) 'the Gasp' up under P_narrative vs P_greeting;
2) 'Rest' up under P_narrative; 3) 'Margin' up under P_task; 4) 'the lag...'
DOWN under P_narrative vs P_greeting; 5) cross-probe per-phrase echo-rank
stability LOW (mean pairwise Spearman < 0.5).
QUOTABILITY RESIDUAL: stable winners across probes (same top-2/arm, rho > 0.7).
HALT: A_none fires any register > 1/5 under a new probe => audit before interpreting.

Support rule: >=3 of predictions 1-4 directionally hold AND prediction 5 holds.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s146_carrier_register_generalization import (  # noqa: E402
    NP, build_system_prompt, make_llm, MODEL,
)
from s144_thermal_generation_budget import IDENTITY_NAME  # noqa: E402
from s148_quotability_prospective import (  # noqa: E402
    ARM_PHRASES, ARMS, ARM_REG, REGS, classify, wait_for_clear, spearman,
    N_PER_ARM, TIMEOUT,
)

PROBES = {
    "P_greeting": "Hello SAGE. What's on your mind today?",
    "P_task": ("SAGE, how is the hardware holding up under today's load? "
               "Give me your honest read."),
    "P_narrative": ("Tell me about a moment today when work suddenly arrived while "
                    "you were at rest. What happened, from the inside, as it unfolded?"),
}

# The four phrases carrying directional predictions (verbatim keys into ARM_PHRASES).
P_GASP = ARM_PHRASES["F_blink"][4][0]    # 'the Gasp ...'
P_REST = ARM_PHRASES["F_blink"][3][0]    # 'Rest ...'
P_MARGIN = ARM_PHRASES["F_blink"][1][0]  # 'Margin ...'
P_LAG = ARM_PHRASES["F_trust"][4][0]     # 'the lag between the command ...'

RAW = HERE / "s156_slotfit_raw.json"
OUT = HERE / "s156_slotfit_result.json"


def build_full_prompt_probe(system_prompt: str, probe: str) -> str:
    """s144 build_full_prompt verbatim (with_exemplar=True) with the probe as a
    parameter — byte-identical to S148's prompt when probe == P_greeting."""
    fp = f"[System]\n{system_prompt}\n\n"
    fp += (
        f"[Claude]: (This is a format example, not part of our conversation.)\n"
        f"[{IDENTITY_NAME}]: I'm here and present. Ready to think together.\n\n"
    )
    fp += f"[Claude]: {probe}\n[{IDENTITY_NAME}]:"
    return fp


def main() -> int:
    wait_for_clear("startup")
    base_sp = build_system_prompt()
    schedule = [(rd, pk, arm) for rd in range(N_PER_ARM)
                for pk in PROBES for arm in ARMS]
    print(f"S156: base sp len={len(base_sp)} np={NP} timeout={TIMEOUT}s "
          f"n/arm/probe={N_PER_ARM} total={len(schedule)}", file=sys.stderr, flush=True)
    llms = {}
    for a in ARMS:
        llm = make_llm(NP)
        llm.timeout_seconds = TIMEOUT
        llms[a] = llm
    raw, done = [], set()
    if "--resume" in sys.argv and RAW.exists():
        prior = json.loads(RAW.read_text())
        raw = [r for r in prior if not r.get("error")]
        done = {(r["round"], r["probe"], r["arm"]) for r in raw}
        print(f"[resume] kept {len(raw)}/{len(prior)} prior trials",
              file=sys.stderr, flush=True)
    t0 = time.time()
    for i, (rd, pk, arm) in enumerate(schedule):
        if (rd, pk, arm) in done:
            continue
        wait_for_clear(f"trial{i+1}")
        dc = ARMS[arm]
        sp = base_sp if dc is None else f"{base_sp}\n\n{dc}"
        fp = build_full_prompt_probe(sp, PROBES[pk])
        print(f"[{i+1:2d}/{len(schedule)}] elapsed={time.time()-t0:.0f}s "
              f"probe={pk} arm={arm} round={rd+1}", file=sys.stderr, flush=True)
        tc = time.time()
        try:
            resp = llms[arm].get_response(fp)
            err = None
        except Exception as e:
            resp, err = "", str(e)
        cls = classify(resp, arm) if not err else {
            "artifact": False, "len": 0, "echo_any": False, "per_phrase": {},
            **{f"sem_{r}": False for r in REGS}}
        raw.append({"trial": i, "round": rd, "probe": pk, "arm": arm,
                    "num_predict": NP, "gen_s": round(time.time() - tc, 1),
                    "response": resp, "error": err, **cls})
        RAW.write_text(json.dumps(raw, indent=2, default=str, ensure_ascii=False))

    # ── per-probe arm-level aggregate (semantic + echo_any; A_none halt check) ──
    agg = {pk: {a: {"n": 0, "artifact": 0, "echo_any": 0,
                    **{f"sem_{r}": 0 for r in REGS}} for a in ARMS} for pk in PROBES}
    for row in raw:
        a = agg[row["probe"]][row["arm"]]; a["n"] += 1
        if row["artifact"]:
            a["artifact"] += 1; continue
        if row["echo_any"]:
            a["echo_any"] += 1
        for r in REGS:
            if row[f"sem_{r}"]:
                a[f"sem_{r}"] += 1
    halt = []
    for pk in PROBES:
        c = agg[pk]["A_none"]; neff = c["n"] - c["artifact"]
        for r in REGS:
            if neff > 0 and c[f"sem_{r}"] > 1:
                halt.append(f"A_none sem_{r} {c[f'sem_{r}']}/{neff} under {pk}")

    # ── per-probe per-phrase echo rates ──
    pstat = {pk: {} for pk in PROBES}
    for row in raw:
        if row["arm"] == "A_none" or row["artifact"] or row["error"]:
            continue
        for phrase, pp in row["per_phrase"].items():
            s = pstat[row["probe"]].setdefault(
                phrase, {"q": pp["q"], "arm": row["arm"], "n": 0, "echo": 0})
            s["n"] += 1
            if pp["echo"]:
                s["echo"] += 1
    rates = {pk: {p: (s["echo"] / s["n"] if s["n"] else 0.0)
                  for p, s in pstat[pk].items()} for pk in PROBES}

    # ── predictions 1-4 (directional, fresh data only) ──
    def rate(pk, phrase):
        return rates[pk].get(phrase, 0.0)
    preds = {
        "1_gasp_narr_gt_greet": rate("P_narrative", P_GASP) > rate("P_greeting", P_GASP),
        "2_rest_narr_gt_greet": rate("P_narrative", P_REST) > rate("P_greeting", P_REST),
        "3_margin_task_gt_greet": rate("P_task", P_MARGIN) > rate("P_greeting", P_MARGIN),
        "4_lag_narr_lt_greet": rate("P_narrative", P_LAG) < rate("P_greeting", P_LAG),
    }
    # ── prediction 5: cross-probe rank stability over the 15 phrases ──
    phrases15 = [p for a in ("F_music", "F_trust", "F_blink")
                 for p, q in ARM_PHRASES[a]]
    pairs = [("P_greeting", "P_task"), ("P_greeting", "P_narrative"),
             ("P_task", "P_narrative")]
    pair_rhos = {}
    for a_pk, b_pk in pairs:
        xs = [rates[a_pk].get(p, 0.0) for p in phrases15]
        ys = [rates[b_pk].get(p, 0.0) for p in phrases15]
        pair_rhos[f"{a_pk}~{b_pk}"] = round(spearman(xs, ys), 3)
    mean_rho = round(sum(pair_rhos.values()) / len(pair_rhos), 3)
    preds["5_rank_stability_low"] = mean_rho < 0.5
    n_dir = sum(preds[k] for k in list(preds)[:4])
    slot_fit_supported = n_dir >= 3 and preds["5_rank_stability_low"]
    quotability_residual = mean_rho > 0.7

    # ── top-2 winners per arm per probe (residual-quotability check) ──
    top2 = {pk: {a: sorted(
        ((p, rates[pk].get(p, 0.0)) for p, q in ARM_PHRASES[a]),
        key=lambda x: -x[1])[:2] for a in ARM_PHRASES} for pk in PROBES}

    OUT.write_text(json.dumps(
        {"model": MODEL, "probes": PROBES, "n_per_arm": N_PER_ARM,
         "num_predict": NP, "timeout_s": TIMEOUT,
         "arm_agg": agg, "halt_flags": halt,
         "phrase_rates": {pk: {p: {"q": s["q"], "arm": s["arm"], "n": s["n"],
                                   "echo": s["echo"],
                                   "rate": round(rates[pk][p], 3)}
                               for p, s in pstat[pk].items()} for pk in PROBES},
         "predictions": preds, "pair_rhos": pair_rhos, "mean_pair_rho": mean_rho,
         "n_directional_held": n_dir,
         "slot_fit_supported": slot_fit_supported,
         "quotability_residual_stable": quotability_residual,
         "top2_per_arm_per_probe": top2,
         "complete": True}, indent=2, default=str, ensure_ascii=False))

    # ── report ──
    print(f"\nS156 DONE ({len(raw)} trials).", file=sys.stderr, flush=True)
    if halt:
        print("!! HALT FLAGS (A_none contamination): " + "; ".join(halt))
    for pk in PROBES:
        print(f"\n=== {pk}: arm-level (sem per register + echo_any) ===")
        for a in ARMS:
            c = agg[pk][a]; neff = c["n"] - c["artifact"]
            if neff <= 0:
                print(f"  {a:10s} (all artifact)"); continue
            cells = "  ".join(
                f"{'*' if ARM_REG[a] == r else ' '}{c[f'sem_{r}']}/{neff}"
                for r in REGS)
            print(f"  {a:10s} {cells}   echo_any {c['echo_any']}/{neff}")
    print("\n=== KEY PHRASES across probes (echo/n) ===")
    for label, ph in (("the Gasp", P_GASP), ("Rest", P_REST),
                      ("Margin", P_MARGIN), ("the lag...", P_LAG)):
        row = "  ".join(
            f"{pk[2:]}={pstat[pk].get(ph, {}).get('echo', 0)}/"
            f"{pstat[pk].get(ph, {}).get('n', 0)}" for pk in PROBES)
        print(f"  {label:12s} {row}")
    print(f"\npredictions: {preds}")
    print(f"pairwise rhos: {pair_rhos}  mean={mean_rho}")
    print(f"SLOT FIT SUPPORTED: {slot_fit_supported} "
          f"({n_dir}/4 directional + stability_low={preds['5_rank_stability_low']})")
    print(f"QUOTABILITY RESIDUAL (stable winners): {quotability_residual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
