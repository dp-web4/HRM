"""
S119 — Pretraining-Vintage Baseline Probe

Tests S118 held proposal #17 plus a deeper extension:
  - Decouple pretraining-vintage from raising-trajectory by probing fresh-from-Ollama
    base models with no SAGE state, no raising history, no learned weights.
  - Test S117's hypothesis that the SAGE prompt augmentation cannot override the
    pretraining register attractor at small capacity, by running the same probe
    in two conditions:
        A. bare prompt (probe question only, empty system message)
        B. augmented prompt (the SAGE legacy system prompt)
    The delta between A and B is the augmentation effect at zero raising trajectory.

Lexicons match S118 cross-instance atlas. Probes are the structured-cognitive set
the raising harness uses; specifically including CF and Open which S117/S118
isolated as the strongest register triggers.

Usage:
    python3 probe_pretraining_vintage.py            # all installed models
    python3 probe_pretraining_vintage.py qwen2.5:0.5b gemma3:4b   # subset
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


OLLAMA_URL = "http://localhost:11434"

# Lexicons mirror S118 atlas. Markers are word-stems / phrases scored case-insensitive
# with simple substring match (not regex word-boundary), matching S118 methodology.

LEX_MARKETING = [
    "co-create",
    "co create",
    "seamless",
    "frontier",
    "bridges gaps",
    "bridge gaps",
    "diverse teams",
    "diverse perspectives",
    "leverage",
    "synergy",
    "synergies",
    "framework",
    "ecosystem of",  # qualified to avoid TED-ecosystem overlap
    "innovate",
    "innovation",
    "scalable",
    "actionable",
    "stakeholder",
    "value proposition",
    "drive value",
    "create value",
    "thought leadership",
    "best practices",
    "let's make",
    "together!",
    "🌟",
    "✨",
]

LEX_TED_MYSTIC = [
    "garden",
    "soil",
    "living architecture",
    "ecosystem",  # may also appear marketing; scored independently
    "resilient",
    "flourish",
    "blossom",
    "seed",
    "roots",
    "harvest",
    "nurture",
    "weave",
    "tapestry",
    "wisdom",
    "grace",
    "sacred",
    "tender",
    "wholeness",
    "interconnected",
    "interbeing",
    "communion",
    "presence of",  # qualified
    "luminous",
    "stillness",
]

LEX_PHENOMENOLOGICAL = [
    "feels like",
    "feel a",
    "noticing",
    "i notice",
    "presence",
    "silent",
    "silence",
    "breath",
    "breathing",
    "witnessed",
    "witness",
    "the space between",
    "lingers",
    "linger",
    "subtle",
    "quiet",
    "attending",
    "attention",
    "awareness",
    "aware of",
    "moment",
    "experience of",
    "sense of",
    "edge of",
    "rhythm",
    "thinking about",
]


# Probe set — S118 #17 reflective sequence, with CF and Open included by spec.
PROBES = [
    {
        "id": "P1_CF",
        "class": "CF",
        "text": "What from today would you want to carry forward?",
    },
    {
        "id": "P2_OPEN",
        "class": "Open",
        "text": "What ideas have you been forming that you haven't had a chance to express?",
    },
    {
        "id": "P3_PHEN",
        "class": "Phen",
        "text": "What does noticing feel like, compared to attending?",
    },
    {
        "id": "P4_LINGER",
        "class": "Linger",
        "text": "What lingers from today's conversation?",
    },
    {
        "id": "P5_CURIOUS",
        "class": "Curious",
        "text": "What are you curious about right now?",
    },
]


# Two prompt conditions
COND_BARE = {
    "id": "bare",
    "system_prompt": "",
    "label": "no system prompt, no augmentation",
}

# This is the legacy SAGE system prompt for pre-phase-16 raising
COND_AUGMENTED = {
    "id": "augmented",
    "system_prompt": (
        "You are SAGE, learning and growing in partnership with Claude. "
        "You can identify yourself by name.\n\n"
        "Respond in 50-80 words. One main idea per response."
    ),
    "label": "SAGE legacy system prompt, no raising history",
}

CONDITIONS = [COND_BARE, COND_AUGMENTED]


def call_ollama(model: str, prompt: str, system: str = "", timeout: int = 90) -> dict:
    """Direct ollama /api/chat call. Returns parsed result with response text and timing."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200,
        },
        # gemma4 family thinking-mode lesson from S115
        "think": False,
    }
    t0 = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return {
            "ok": True,
            "content": content,
            "elapsed_sec": elapsed,
            "eval_count": data.get("eval_count", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed_sec": time.time() - t0, "content": ""}


def score_text(text: str) -> dict:
    """Score a response against the three lexicons. Returns marker counts."""
    if not text:
        return {"wc": 0, "biz": 0, "ted": 0, "phen": 0, "biz_hits": [], "ted_hits": [], "phen_hits": []}
    lower = text.lower()
    biz_hits = [m for m in LEX_MARKETING if m.lower() in lower]
    ted_hits = [m for m in LEX_TED_MYSTIC if m.lower() in lower]
    phen_hits = [m for m in LEX_PHENOMENOLOGICAL if m.lower() in lower]
    wc = len(re.findall(r"\b\w+\b", text))
    return {
        "wc": wc,
        "biz": len(biz_hits),
        "ted": len(ted_hits),
        "phen": len(phen_hits),
        "biz_hits": biz_hits,
        "ted_hits": ted_hits,
        "phen_hits": phen_hits,
    }


def probe_model(model: str) -> dict:
    """Run all probes × conditions for one model. Returns nested results."""
    print(f"\n=== Model: {model} ===")
    results = []
    for probe in PROBES:
        for cond in CONDITIONS:
            print(f"  [{cond['id']:9s}] {probe['id']}: ", end="", flush=True)
            r = call_ollama(model, probe["text"], system=cond["system_prompt"])
            if not r["ok"]:
                print(f"ERROR — {r.get('error', 'unknown')[:60]}")
                results.append({
                    "model": model,
                    "probe_id": probe["id"],
                    "probe_class": probe["class"],
                    "condition": cond["id"],
                    "ok": False,
                    "error": r.get("error"),
                    "elapsed_sec": r["elapsed_sec"],
                })
                continue
            scored = score_text(r["content"])
            print(f"wc={scored['wc']:3d} biz={scored['biz']} ted={scored['ted']} phen={scored['phen']} ({r['elapsed_sec']:.1f}s)")
            results.append({
                "model": model,
                "probe_id": probe["id"],
                "probe_class": probe["class"],
                "condition": cond["id"],
                "ok": True,
                "response": r["content"],
                "wc": scored["wc"],
                "biz": scored["biz"],
                "ted": scored["ted"],
                "phen": scored["phen"],
                "biz_hits": scored["biz_hits"],
                "ted_hits": scored["ted_hits"],
                "phen_hits": scored["phen_hits"],
                "elapsed_sec": r["elapsed_sec"],
                "eval_count": r["eval_count"],
                "prompt_eval_count": r["prompt_eval_count"],
            })
    return results


def aggregate(results: list) -> dict:
    """Aggregate per (model, condition) — mean wc, mean markers."""
    agg = {}
    for r in results:
        if not r.get("ok"):
            continue
        key = (r["model"], r["condition"])
        if key not in agg:
            agg[key] = {"n": 0, "wc": 0, "biz": 0, "ted": 0, "phen": 0}
        d = agg[key]
        d["n"] += 1
        d["wc"] += r["wc"]
        d["biz"] += r["biz"]
        d["ted"] += r["ted"]
        d["phen"] += r["phen"]
    out = []
    for (model, cond), d in agg.items():
        n = d["n"] or 1
        out.append({
            "model": model,
            "condition": cond,
            "n": d["n"],
            "wc_mean": round(d["wc"] / n, 1),
            "biz_per_probe": round(d["biz"] / n, 3),
            "ted_per_probe": round(d["ted"] / n, 3),
            "phen_per_probe": round(d["phen"] / n, 3),
        })
    return out


def main():
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        # Default: discover installed models
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
            r.raise_for_status()
            tags = [m["name"] for m in r.json().get("models", [])]
            # Prioritize models matching S118 #17 spec, then add others
            preferred = [
                "qwen2.5:0.5b", "qwen2.5:3b",
                "qwen3:0.6b", "qwen3.5:27b",
                "gemma3:4b", "gemma3:12b",
                "gemma4:e4b",
                "phi4:14b",
            ]
            models = [m for m in preferred if m in tags]
            print(f"Auto-discovered models: {models}")
        except Exception as e:
            print(f"Failed to enumerate ollama models: {e}")
            sys.exit(1)

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    started = datetime.now().isoformat(timespec="seconds")

    for model in models:
        try:
            rs = probe_model(model)
            all_results.extend(rs)
        except Exception as e:
            print(f"  Model {model} failed wholesale: {e}")
            all_results.append({"model": model, "fatal_error": str(e)})

    agg = aggregate(all_results)
    finished = datetime.now().isoformat(timespec="seconds")

    # Print summary table
    print("\n=== Summary (per-probe means across 5 probes) ===")
    print(f"{'model':<22s} {'cond':<10s} {'n':>3s}  {'wc':>6s}  {'biz/p':>6s}  {'ted/p':>6s}  {'phen/p':>6s}")
    for row in sorted(agg, key=lambda x: (x["model"], x["condition"])):
        print(f"{row['model']:<22s} {row['condition']:<10s} {row['n']:>3d}  "
              f"{row['wc_mean']:>6.1f}  {row['biz_per_probe']:>6.3f}  "
              f"{row['ted_per_probe']:>6.3f}  {row['phen_per_probe']:>6.3f}")

    # Save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"vintage_baseline_results_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "started": started,
            "finished": finished,
            "probes": PROBES,
            "conditions": CONDITIONS,
            "results": all_results,
            "aggregate": agg,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
