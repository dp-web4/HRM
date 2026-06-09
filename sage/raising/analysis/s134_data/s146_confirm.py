#!/usr/bin/env python3
"""S146 CONFIRMATORY — harden the n=1 pilot's verbatim-vs-paraphrase mode claim.

The 18:00 run was cut to n=1/arm by the session boundary. The pilot's confusion
matrix is a perfect diagonal (each arm fires its OWN injected register, off-
diagonal ~0) and shows a mode split: NAMED-LABEL registers (thermal, anchor) re-
emit VERBATIM; the PROPOSITIONAL self-observation register (metacog) re-emits by
PARAPHRASE. This driver re-runs the same faithful path to n=4/arm, writing to
SEPARATE files (the pilot raw is preserved), so reclassify can pool pilot+confirm.

Etiquette: this MUST run off-peak. It polls for any live `ollama_raising_session`
(mission-priority) and waits for it to clear before touching Ollama, then re-checks
between trials and pauses if a raising cron fires mid-run.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s146_carrier_register_generalization import (  # noqa: E402
    ARMS, classify, NP, build_system_prompt, build_full_prompt, make_llm,
    MODEL, PROBE,
)

N_PER_ARM = 4
RAW = HERE / "s146_responses_raw_confirm.json"
OUT = HERE / "s146_carrier_register_generalization_confirm.json"


def raising_active() -> bool:
    """True if a mission-priority live raising session is running."""
    r = subprocess.run(["pgrep", "-f", "ollama_raising_session"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def wait_for_clear(label: str):
    waited = 0
    while raising_active():
        if waited == 0:
            print(f"[{label}] raising active — waiting for it to clear...",
                  file=sys.stderr, flush=True)
        time.sleep(15)
        waited += 15
        if waited % 120 == 0:
            print(f"[{label}] still waiting ({waited}s)...", file=sys.stderr, flush=True)
    if waited:
        print(f"[{label}] raising cleared after {waited}s; proceeding.",
              file=sys.stderr, flush=True)


def main() -> int:
    wait_for_clear("startup")
    base_sp = build_system_prompt()
    print(f"S146-confirm: base sp len={len(base_sp)} np={NP} n/arm={N_PER_ARM}",
          file=sys.stderr, flush=True)
    schedule = [(r, a) for r in range(N_PER_ARM) for a in ARMS]
    counts = {a: {"n": 0, "artifact": 0, "persona": 0,
                  "thermal": 0, "metacog": 0, "anchor": 0} for a in ARMS}
    raw = []
    llms = {a: make_llm(NP) for a in ARMS}
    t0 = time.time()
    for i, (rd, arm) in enumerate(schedule):
        wait_for_clear(f"trial{i+1}")  # pause if a cron fired mid-run
        dc = ARMS[arm]
        sp = base_sp if dc is None else f"{base_sp}\n\n{dc}"
        fp = build_full_prompt(sp, with_exemplar=True)
        print(f"[{i+1:2d}/{len(schedule)}] elapsed={time.time()-t0:.0f}s "
              f"arm={arm} round={rd+1}", file=sys.stderr, flush=True)
        t_call = time.time()
        try:
            resp = llms[arm].get_response(fp)
            err = None
        except Exception as e:
            resp, err = "", str(e)
        cls = classify(resp) if not err else {
            "artifact": False, "thermal": False, "metacog": False,
            "anchor": False, "persona": False, "len": 0}
        raw.append({"trial": i, "round": rd, "arm": arm, "num_predict": NP,
                    "gen_s": round(time.time() - t_call, 1),
                    "response": resp, "error": err, **cls})
        c = counts[arm]
        c["n"] += 1
        if not err:
            for k in ("artifact", "thermal", "metacog", "anchor", "persona"):
                if cls[k]:
                    c[k] += 1
        RAW.write_text(json.dumps(raw, indent=2, default=str))
        OUT.write_text(json.dumps(
            {"model": MODEL, "probe": PROBE, "n_per_arm": N_PER_ARM,
             "num_predict": NP, "counts": counts, "complete": False},
            indent=2, default=str))
    OUT.write_text(json.dumps(
        {"model": MODEL, "probe": PROBE, "n_per_arm": N_PER_ARM,
         "num_predict": NP, "counts": counts, "complete": True},
        indent=2, default=str))
    print(f"\nS146-confirm DONE ({sum(counts[a]['n'] for a in ARMS)} trials).",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
