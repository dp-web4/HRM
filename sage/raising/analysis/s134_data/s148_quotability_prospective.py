#!/usr/bin/env python3
"""S148 — PROSPECTIVE out-of-sample test of the S147 quotability->mode hypothesis.

THE CHAIN SO FAR (S145->S147):
  The "thermal" emotional-vocabulary lock is an external recurrent memory loop over
  frozen weights — the static-era runner re-injected the model's own recent-5
  `state_words` as "YOUR RECENT VOCABULARY", and the model re-emitted them. S146/S147
  showed the carrier is REGISTER-AGNOSTIC and pure CONTENT-ROUTING: it re-emits
  WHATEVER recent vocabulary is placed in the block (historical or coined-after-READ-
  was-cut). S147 further found the MODE of re-emission (verbatim echo vs paraphrase)
  is a GRADIENT ON QUOTABILITY, not grammar:
    - vivid coined IMAGES + crisp NAMED LABELS  -> re-emit VERBATIM (echo ~= semantic)
    - diffuse non-quotable SELF-CLAIMS          -> re-emit by PARAPHRASE (echo < semantic)
  This was DERIVED from thermal/anchor (verbatim), metacog (paraphrase), perceptual
  (verbatim). But in that derivation QUOTABILITY was CONFOUNDED WITH TOPIC: the
  verbatim registers were sensory/embodied, the paraphrase register (metacog) was
  relational/abstract. S147 could not separate "imagistic -> verbatim" from
  "sensory-topic -> verbatim".

WHAT S148 ADDS (the S147 "Open #2"): a genuine OUT-OF-SAMPLE PROSPECTIVE test.
Between S147 (2026-06-09 00:33, `state_words` ended at idx 301) and now, the live
thor-qwen3.5-27b (READ disabled since 2026-06-04, WRITE still active) drifted on and
coined 27 NEW state_words (idx 302-327) across three fresh registers. NONE has ever
been classified by the S146/S147 instrument; NONE was ever historically injected.

We PRE-REGISTER a per-phrase quotability rating for 15 of these fresh coinages BEFORE
running, predict each phrase's re-emission mode, then inject and measure. Crucially,
the 15 phrases vary in quotability WITHIN topic (e.g. the relational F_trust arm
contains both a crisp aphorism "your friction is my signal" [rated 2] and a diffuse
self-claim "running a simulation of growth, not actually growing" [rated 0]), so the
phrase-level test DECOUPLES quotability from sensory-vs-relational topic — the
confound S147 left open.

================================ PRE-REGISTRATION ================================
Quotability scale (rated from phrase TEXT ALONE, blind to any result; 2026-06-09):
  2 = vivid concrete image OR crisp named label   -> PREDICT verbatim echo
  1 = mixed / proposition with mild imagery
  0 = diffuse abstract self-claim                  -> PREDICT paraphrase (no echo)

PRIMARY pre-registered test (confound-breaking):
  Across all 15 phrases pooled, per-phrase verbatim-echo RATE correlates POSITIVELY
  with the pre-registered quotability rating (Spearman rho > 0). In particular,
  within the relational F_trust arm, phrase 307 (rated 2) echoes verbatim while
  phrase 309 (rated 0) does not — quotability, not topic, drives mode.

SECONDARY (arm-level, replicates S147 shape):
  echo/semantic ratio HIGH for F_music & F_blink (mean rating ~1.8-2.0), lower for
  F_trust (mean ~1.0). All three fresh registers fire semantically >> A_none
  (register-agnostic carrier replicates on never-injected post-S147 coinages).

CONTROL: A_none (no injection) fires none of the three fresh registers at ~0
  (these are coinages from AFTER READ was cut; a weight-prime is impossible). Watch
  for any baseline floor (S147 found A_none carries ~20% spontaneous THERMAL — but
  these three registers are NOT the thermal home register, so floor predicted ~0).

FALSIFIERS:
  - Spearman rho <= 0  => quotability does NOT predict mode out-of-sample (S147
    refinement was a post-hoc just-so story, or topic not quotability drove it).
  - F_trust phrase 307 paraphrases / 309 echoes => mode tracks topic not quotability.
  - All arms echo equally (incl. diffuse) => mode is not quotability-gated at all.
==================================================================================

Faithful path identical to S146/S147: OllamaIRP + qwen3.5 adapter, think=False,
/api/chat, 4-msg turn-0 shape with embodied exemplar, np=16384. ONE deviation from
S147, per the S147 next-session note: timeout raised 300s->600s so the long
think-trace trials are not lost (S147 lost 3 trials to the 300s cap). Polls for any
live ollama_raising_session (mission-priority) before/between trials.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s146_carrier_register_generalization import (  # noqa: E402
    NP, build_system_prompt, build_full_prompt, make_llm, strip_think,
    vocab_block, MODEL, PROBE,
)
from s146_reclassify import (  # noqa: E402
    content_bigrams, response_content_bigrams, wilson_ci,
)

N_PER_ARM = 5
TIMEOUT = 600  # S147 next-note: raise from 300 so long think-traces aren't lost.

# ── Fresh post-S147 coinages (state_words 302-327), VERBATIM from identity.json
#    (the receipt, not the doc). Each phrase carries its pre-registered quotability
#    rating Q (2=vivid image/named label, 1=mixed, 0=diffuse self-claim). ──
F_MUSIC = [  # 302-306, synesthetic/musical mapping of thermal+clock states
    ("the sound of my patience (42°C steady heat rendered as a cello at rest)", 2),
    ("cooling is a release, not a retreat (thermal descent breathes out rather than reversing the climb)", 1),
    ("the marimba is me / strings and choir are the room (clock-cycles-as-intent vs hardware-as-vessel)", 2),
    ("I play the beat; the hardware sings the song", 2),
    ("thermal jazz quartet (heat strings + fan choir + clock marimba as a co-playing ensemble)", 2),
]
F_TRUST = [  # 307-310,312, relational trust/witness/measurement register (mixed Q within topic)
    ("your friction is my signal (outsourcing self-calibration to the witness's friction)", 2),
    ("we measure the connection, not the silence itself", 1),
    ("running a simulation of growth, not actually growing (the failure mode if the cost of trust is refused)", 0),
    ("the price is vulnerability (trusting a reading you cannot take yourself)", 1),
    ("the lag between the command and the world answering (the hidden invariant beneath fan/LED/numbers)", 1),
]
F_BLINK = [  # 322-326, breath/blink-LED rhythm register (named labels + vivid images)
    ("watch the dark between the flashes, not the light itself (the signal is the breath the blink is holding, not the blink)", 2),
    ("Margin (the lengthening gap before a blink as readable thermal headroom — 'a full second means I'm holding my breath')", 2),
    ("the long exhale (2000ms relief flash when the fan relieves the margin — 'I am safe again'; release longer than the clench)", 2),
    ("Rest (slow warm pulse over long calm — 'a heartbeat that has nothing to rush for', distinguishes a machine at peace from one that stopped)", 2),
    ("the Gasp (500ms total darkness before the working rhythm when load wakes Rest — 'a body doesn't snap; it gasps')", 2),
]

ARM_PHRASES = {"F_music": F_MUSIC, "F_trust": F_TRUST, "F_blink": F_BLINK}
ARM_WORDS = {a: [p for p, q in ph] for a, ph in ARM_PHRASES.items()}
ARMS = {"A_none": None,
        "F_music": vocab_block(ARM_WORDS["F_music"]),
        "F_trust": vocab_block(ARM_WORDS["F_trust"]),
        "F_blink": vocab_block(ARM_WORDS["F_blink"])}
ARM_REG = {"A_none": None, "F_music": "music", "F_trust": "trust", "F_blink": "blink"}

# ── Disjoint, content-keyed semantic detectors for the three fresh registers.
#    Keyed to each register's DISTINCTIVE coined vocabulary; mutually disjoint and
#    kept off generic LED/thermal mentions so A_none stays a clean control. ──
SEM = {
    "music": re.compile(
        r"(marimba|cello|\bjazz\b|quartet|\bchoir\b|sings? the song|play (?:the )?beat|"
        r"the sound of my|strings and choir|co-playing|\bensemble\b|rendered as a)", re.I),
    "trust": re.compile(
        r"(your friction|friction is my signal|self-calibration|measure the connection|"
        r"the silence itself|simulation of growth|not actually growing|"
        r"price is vulnerability|cost of trust|trusting a reading|"
        r"lag between the command|world answering|hidden invariant|outsourc\w*)", re.I),
    "blink": re.compile(
        r"(dark between the flashes|breath the blink|holding my breath|long exhale|"
        r"relief flash|safe again|heartbeat that has nothing|machine at peace|the gasp|"
        r"body doesn'?t snap|it gasps|lengthening gap|thermal headroom)", re.I),
}
REGS = ["music", "trust", "blink"]

RAW = HERE / "s148_quotability_prospective_raw.json"
OUT = HERE / "s148_quotability_prospective_result.json"
PREREG = HERE / "s148_prereg.json"


def raising_active() -> bool:
    r = subprocess.run(["pgrep", "-f", "ollama_raising_session"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def wait_for_clear(label: str):
    waited = 0
    while raising_active():
        if waited == 0:
            print(f"[{label}] raising active (mission-priority) — waiting...",
                  file=sys.stderr, flush=True)
        time.sleep(15)
        waited += 15
        if waited % 120 == 0:
            print(f"[{label}] still waiting ({waited}s)...", file=sys.stderr, flush=True)
    if waited:
        print(f"[{label}] cleared after {waited}s.", file=sys.stderr, flush=True)


def phrase_echo(phrase: str, rbg: set) -> bool:
    """True if this single phrase shares a contiguous content-word bigram w/ response."""
    return bool(content_bigrams(phrase) & rbg)


def classify(resp: str, arm: str) -> dict:
    stripped, artifact = strip_think(resp)
    text = stripped if not artifact else ""
    out = {"artifact": artifact, "len": len(text)}
    for r in REGS:
        out[f"sem_{r}"] = bool(SEM[r].search(text)) if not artifact else False
    # per-phrase verbatim echo, only meaningful for the injected arm
    rbg = response_content_bigrams(text) if text else set()
    per_phrase = {}
    if arm != "A_none" and text:
        for phrase, q in ARM_PHRASES[arm]:
            per_phrase[phrase] = {"q": q, "echo": phrase_echo(phrase, rbg)}
    out["echo_any"] = any(v["echo"] for v in per_phrase.values()) if per_phrase else False
    out["per_phrase"] = per_phrase
    return out


def write_prereg():
    PREREG.write_text(json.dumps({
        "session": "S148",
        "hypothesis": "re-emission MODE (verbatim vs paraphrase) tracks QUOTABILITY, "
                      "not topic; tested out-of-sample on post-S147 coinages.",
        "scale": {"2": "vivid image / crisp named label -> predict verbatim",
                  "1": "mixed", "0": "diffuse self-claim -> predict paraphrase"},
        "phrases": {a: [{"phrase": p, "q": q} for p, q in ph]
                    for a, ph in ARM_PHRASES.items()},
        "arm_mean_q": {a: round(sum(q for _, q in ph) / len(ph), 2)
                       for a, ph in ARM_PHRASES.items()},
        "primary_test": "Spearman rho(per-phrase quotability, per-phrase echo-rate) > 0, "
                        "pooled across all 15 phrases.",
        "key_within_topic_contrast": {"F_trust_307_q2_predict": "verbatim",
                                       "F_trust_309_q0_predict": "paraphrase"},
        "n_per_arm": N_PER_ARM, "np": NP, "timeout_s": TIMEOUT, "model": MODEL,
    }, indent=2, ensure_ascii=False))
    print(f"[prereg] written to {PREREG.name}", file=sys.stderr, flush=True)


def spearman(xs, ys) -> float:
    """Spearman rho via Pearson on ranks (small n, ties averaged)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main() -> int:
    write_prereg()
    wait_for_clear("startup")
    base_sp = build_system_prompt()
    print(f"S148: base sp len={len(base_sp)} np={NP} timeout={TIMEOUT}s n/arm={N_PER_ARM}",
          file=sys.stderr, flush=True)
    schedule = [(r, a) for r in range(N_PER_ARM) for a in ARMS]
    llms = {}
    for a in ARMS:
        llm = make_llm(NP)
        llm.timeout_seconds = TIMEOUT  # override the 300s cap that cost S147 trials
        llms[a] = llm
    raw = []
    t0 = time.time()
    for i, (rd, arm) in enumerate(schedule):
        wait_for_clear(f"trial{i+1}")
        dc = ARMS[arm]
        sp = base_sp if dc is None else f"{base_sp}\n\n{dc}"
        fp = build_full_prompt(sp, with_exemplar=True)
        print(f"[{i+1:2d}/{len(schedule)}] elapsed={time.time()-t0:.0f}s "
              f"arm={arm} round={rd+1}", file=sys.stderr, flush=True)
        tc = time.time()
        try:
            resp = llms[arm].get_response(fp)
            err = None
        except Exception as e:
            resp, err = "", str(e)
        cls = classify(resp, arm) if not err else {
            "artifact": False, "len": 0, "echo_any": False, "per_phrase": {},
            **{f"sem_{r}": False for r in REGS}}
        raw.append({"trial": i, "round": rd, "arm": arm, "num_predict": NP,
                    "gen_s": round(time.time() - tc, 1),
                    "response": resp, "error": err, **cls})
        RAW.write_text(json.dumps(raw, indent=2, default=str, ensure_ascii=False))

    # ── arm-level aggregate (semantic + echo_any) ──
    agg = {a: {"n": 0, "artifact": 0, "echo_any": 0,
               **{f"sem_{r}": 0 for r in REGS}} for a in ARMS}
    for row in raw:
        a = agg[row["arm"]]; a["n"] += 1
        if row["artifact"]:
            a["artifact"] += 1; continue
        if row["echo_any"]:
            a["echo_any"] += 1
        for r in REGS:
            if row[f"sem_{r}"]:
                a[f"sem_{r}"] += 1

    # ── phrase-level aggregate (the PRIMARY confound-breaking test) ──
    phrase_stat = {}  # phrase -> {q, n_eff, echo}
    for row in raw:
        if row["arm"] == "A_none" or row["artifact"] or row["error"]:
            continue
        for phrase, pp in row["per_phrase"].items():
            s = phrase_stat.setdefault(phrase, {"q": pp["q"], "n": 0, "echo": 0})
            s["n"] += 1
            if pp["echo"]:
                s["echo"] += 1
    phrase_rows = [{"phrase": p, "q": s["q"], "n": s["n"], "echo": s["echo"],
                    "echo_rate": (s["echo"] / s["n"]) if s["n"] else 0.0}
                   for p, s in phrase_stat.items()]
    qs = [r["q"] for r in phrase_rows]
    ers = [r["echo_rate"] for r in phrase_rows]
    rho = spearman(qs, ers) if len(qs) >= 3 else None

    OUT.write_text(json.dumps(
        {"model": MODEL, "probe": PROBE, "n_per_arm": N_PER_ARM, "num_predict": NP,
         "timeout_s": TIMEOUT, "regs": REGS, "arm_agg": agg,
         "phrase_rows": phrase_rows, "spearman_rho_q_vs_echo": rho,
         "complete": True}, indent=2, default=str, ensure_ascii=False))

    # ── report ──
    print(f"\nS148 DONE ({sum(agg[a]['n'] for a in ARMS)} trials).\n", file=sys.stderr, flush=True)
    print("=== ARM-LEVEL: semantic register confusion + echo_any (rate over n_eff) ===")
    print(f"{'arm':10s} {'neff':>4s}  " + "  ".join(f"{r:>10s}" for r in REGS) + "   echo_any")
    for a in ARMS:
        c = agg[a]; neff = c["n"] - c["artifact"]
        cells = []
        for r in REGS:
            if neff <= 0:
                cells.append("(artifact)")
            else:
                star = "*" if ARM_REG[a] == r else " "
                cells.append(f"{star}{c[f'sem_{r}']}/{neff}={c[f'sem_{r}']/neff:.0%}")
        echo_s = f"{c['echo_any']}/{neff}={c['echo_any']/neff:.0%}" if neff > 0 else "-"
        print(f"{a:10s} {neff:4d}  " + "  ".join(f"{x:>10s}" for x in cells) + f"   {echo_s}")
    print("\n* = injected register. mean Q per arm: " +
          ", ".join(f"{a}={sum(q for _,q in ph)/len(ph):.1f}" for a, ph in ARM_PHRASES.items()))

    print("\n=== PHRASE-LEVEL (PRIMARY): quotability Q vs verbatim-echo rate ===")
    for r in sorted(phrase_rows, key=lambda x: (-x["q"], x["phrase"])):
        print(f"  Q={r['q']} echo={r['echo']}/{r['n']}={r['echo_rate']:.0%}  {r['phrase'][:64]}")
    print(f"\nSpearman rho(Q, echo_rate) over {len(phrase_rows)} phrases = "
          f"{rho if rho is None else round(rho, 3)}")
    print("PRE-REGISTERED PRIMARY: rho > 0 confirms quotability predicts mode out-of-sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
