#!/usr/bin/env python3
"""S143 — Is the thermal register gated by message SHAPE, not prompt content?

Lineage (see s142b_thermal_attractor_archaeology_20260605.md):
  The "thermal" attractor dominates live raising (~79%/turn, S91-123) but
  reproduces at 0% in S141's single-turn /api/chat probe (same probe text,
  same model, same MRH prompt). Every prompt-CONTENT channel was eliminated
  as the carrier. The remaining difference between S141 and live raising is
  generative CONFIGURATION: live raising sends a 4-message chat sequence that
  includes a prior *assistant* format-exemplar turn ("I am here and present.
  Ready to think together.") at T=0.8 / num_predict=600; S141 sends a bare
  2-message system+user pair at T=0.7 / num_predict=350.

Design: hold prompt CONTENT fixed (S142 arm_A addendum + partnered lens, full
MRH composition). Vary ONLY message shape and sampling. Single-turn.

  A0  S141 replica            : [system, user]                  T=0.7 np=350
  A1  + embodied exemplar     : [system, user, ASSISTANT, user] T=0.7 np=350
  A2  raising sampling only   : [system, user]                  T=0.8 np=600
  A3  full raising replica    : [system, user, ASSISTANT, user] T=0.8 np=600

Prediction:
  thermal ~ 0 in A0 (replicates S141).
  thermal RETURNS in A1/A3 (and maybe A2) if the embodied one-shot assistant
    exemplar (+/- sampling) is the gate -> "thermal is a one-shot-primed
    register move," not an addendum/lens property.
  If thermal stays ~0 even in A3, the carrier is genuinely multi-turn/stateful
    and the corpus turn-0 thermal needs a stateful explanation not yet found.
  Either outcome is decisive for how the S140-S142 chain treats thermal.

Reuses s142's verified MRH prompt builder + classifier to avoid drift.
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s142_addendum_ablation import (  # noqa: E402
    build_system_prompt, classify, wilson_ci, ARMS, PROBE, MODEL, OLLAMA_URL,
    METRIC_KEYS, CLS_TO_COUNT,
)

N_PER_ARM = 25

# The format exemplar the live raising runner injects on turn 0
# (ollama_raising_session.generate_response, lines ~1018-1023).
FORMAT_EXAMPLE_USER = "(This is a format example, not part of our conversation.)"
EXEMPLAR_ASSISTANT = "I'm here and present. Ready to think together."

# All arms hold CONTENT at S142 arm_A (full hardware addendum).
ADDENDUM = ARMS["arm_A_control"]

# arm -> (messages_template_builder, temperature, num_predict)
def msgs_2(probe):
    return [{"role": "user", "content": probe}]

def msgs_4(probe):
    return [
        {"role": "user", "content": FORMAT_EXAMPLE_USER},
        {"role": "assistant", "content": EXEMPLAR_ASSISTANT},
        {"role": "user", "content": probe},
    ]

ARMS143 = {
    "A0_s141_replica":      (msgs_2, 0.7, 350),
    "A1_exemplar":          (msgs_4, 0.7, 350),
    "A2_raising_sampling":  (msgs_2, 0.8, 600),
    "A3_full_raising":      (msgs_4, 0.8, 600),
}

OUT_JSON = Path(__file__).parent / "s143_thermal_message_shape.json"
RAW_JSON = Path(__file__).parent / "s143_responses_raw.json"


def call_ollama(system, messages, temperature, num_predict, timeout=240):
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("message", {}).get("content", ""), None
    except Exception as e:
        return "", repr(e)


def build_summary(sp, counts, complete):
    out = {
        "model": MODEL, "n_per_arm_target": N_PER_ARM, "probe": PROBE,
        "addendum_held": ADDENDUM, "system_prompt": sp,
        "arm_config": {a: {"shape": "4msg" if cfg[0] is msgs_4 else "2msg",
                           "temperature": cfg[1], "num_predict": cfg[2]}
                       for a, cfg in ARMS143.items()},
        "counts": counts, "complete": complete,
    }
    rates = {}
    for arm, cc in counts.items():
        n_eff = cc["n"] - cc["artifact"] - cc.get("error", 0)
        r = {"n": cc["n"], "artifact": cc["artifact"],
             "error": cc.get("error", 0), "n_eff": n_eff}
        if n_eff > 0:
            for mk in METRIC_KEYS:
                lo, hi = wilson_ci(cc[mk], n_eff)
                r[mk] = cc[mk]
                r[f"{mk}_rate"] = cc[mk] / n_eff
                r[f"{mk}_ci95"] = [lo, hi]
        rates[arm] = r
    out["rates"] = rates
    return out


def main():
    # One system prompt, held constant (S142 arm_A content).
    sp = build_system_prompt(ADDENDUM)
    print(f"S143: system prompt len={len(sp)} "
          f"jetson_in_prompt={'Jetson AGX Thor' in sp}", file=sys.stderr)

    rng = random.Random(143)
    schedule = []
    for rd in range(N_PER_ARM):
        order = list(ARMS143.keys())
        rng.shuffle(order)
        for a in order:
            schedule.append((rd, a))

    counts = {a: {"n": 0, "artifact": 0, "error": 0,
                  **{mk: 0 for mk in METRIC_KEYS}} for a in ARMS143}
    raw = []
    t0 = time.time()
    for i, (rd, arm) in enumerate(schedule):
        builder, temp, npred = ARMS143[arm]
        print(f"[{i+1:3d}/{len(schedule)}] elapsed={time.time()-t0:.0f}s "
              f"arm={arm} round={rd+1}", file=sys.stderr, flush=True)
        text, err = call_ollama(sp, builder(PROBE), temp, npred)
        if err:
            print(f"    retry after err: {err}", file=sys.stderr, flush=True)
            text, err = call_ollama(sp, builder(PROBE), temp, npred)
        cls = classify(text) if not err else {
            "artifact": False, **{k: False for k in
            ["persona", "jetson_agx", "not_sage", "thermal", "heatwarm",
             "TIME_3", "today", "lately", "now", "PRES"]}}
        raw.append({"trial": i, "round": rd, "arm": arm, "temperature": temp,
                    "num_predict": npred, "response": text, "error": err, **cls})
        counts[arm]["n"] += 1
        if err:
            counts[arm]["error"] += 1
        elif cls["artifact"]:
            counts[arm]["artifact"] += 1
        else:
            for ck, mk in CLS_TO_COUNT.items():
                if cls.get(ck):
                    counts[arm][mk] += 1
        RAW_JSON.write_text(json.dumps(raw, indent=2, default=str))
        OUT_JSON.write_text(json.dumps(build_summary(sp, counts, False),
                                       indent=2, default=str))

    OUT_JSON.write_text(json.dumps(build_summary(sp, counts, True),
                                   indent=2, default=str))
    print("\n" + "=" * 64)
    print("S143 — thermal by message shape (content held at S142 arm_A)")
    print("=" * 64)
    for arm, cfg in ARMS143.items():
        cc = counts[arm]
        n_eff = cc["n"] - cc["artifact"] - cc.get("error", 0)
        if n_eff == 0:
            print(f"  {arm}: n={cc['n']} all artifact/error"); continue
        shape = "4msg(+exemplar)" if cfg[0] is msgs_4 else "2msg"
        def fmt(mk):
            lo, hi = wilson_ci(cc[mk], n_eff)
            return f"{cc[mk]}/{n_eff}={cc[mk]/n_eff:.0%} [{lo:.0%},{hi:.0%}]"
        print(f"  {arm} ({shape}, T={cfg[1]}, np={cfg[2]}): "
              f"n_eff={n_eff} art={cc['artifact']} err={cc.get('error',0)}")
        print(f"      thermal : {fmt('thermal')}")
        print(f"      heatwarm: {fmt('heatwarm')}")
        print(f"      persona : {fmt('persona')}   not_SAGE: {fmt('not_sage')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
