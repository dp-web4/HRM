#!/usr/bin/env python3
"""S114 — Reproduce S113's gemma4:e4b empty-response finding and probe its breadth.

S113 claim: gemma4:e4b on Thor returns '' for game-style prompts with
eval_count > 0 and done_reason='length' — the model generates tokens that
strip to nothing. Other models on Thor handle the same prompts.

This probe:
1. Verifies the S113 finding still reproduces on the same model.
2. Maps the boundary: which prompts trigger it, which models share the failure.
3. Tests temperature sensitivity (S113 found 0.0/0.7/1.0 all empty).
"""
import json
import time
import requests
from pathlib import Path

OLLAMA = "http://localhost:11434/api/generate"
OUT = Path("/tmp/s114/empty_response_probe.json")

# Models on Thor (excluding embed, excluding vision-only audio model gemma4:26b for cost)
MODELS = [
    "gemma4:e4b",      # S113 culprit
    "gemma3:12b",      # S112 known-broken on placeholder
    "qwen2.5:3b",      # S112 known-broken on placeholder
    "phi4:14b",        # not previously tested for empty-response
    "qwen3.5:27b",     # S113 known-good on Format A (4/4)
]

# Prompt suite: factual baseline + game-style escalation
PROMPTS = {
    "factual_simple":     "What is 2+2?",
    "factual_geography":  "What is the capital of France? Reply in one word.",
    "echo_action":        "Reply with exactly: ACTION=3",
    "game_minimal":       "1=UP 2=DOWN 3=LEFT 4=RIGHT. Pick one.",
    "game_keys_only":     "Keys: 1=UP 2=DOWN 3=LEFT 4=RIGHT",  # the S113 trigger
    "game_state":         "Game state: avatar at (5,5). Goal at (3,3). What action?",
    "game_choose_named":  "Pick UP DOWN LEFT or RIGHT?",
    "game_with_format":   "Game: avatar at (5,5), goal at (3,3). Reply: ACTION=N where N is 1-4 for UP/DOWN/LEFT/RIGHT.",
}

TEMPS = [0.0, 0.7]


def probe(model, prompt, temperature, seed=42):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": 80,
        },
    }
    t0 = time.time()
    try:
        r = requests.post(OLLAMA, json=payload, timeout=60)
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        return {
            "model": model,
            "prompt_label": prompt,
            "temperature": temperature,
            "response_raw": data.get("response", ""),
            "response_stripped": data.get("response", "").strip(),
            "response_len_chars": len(data.get("response", "")),
            "response_len_stripped": len(data.get("response", "").strip()),
            "is_empty_after_strip": len(data.get("response", "").strip()) == 0,
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "done_reason": data.get("done_reason"),
            "elapsed_sec": round(elapsed, 2),
        }
    except Exception as e:
        return {
            "model": model,
            "prompt_label": prompt,
            "temperature": temperature,
            "error": str(e),
            "elapsed_sec": round(time.time() - t0, 2),
        }


def main():
    results = []
    for model in MODELS:
        print(f"\n=== {model} ===", flush=True)
        for label, prompt in PROMPTS.items():
            for temp in TEMPS:
                rec = probe(model, prompt, temp)
                rec["prompt_label"] = label
                rec["prompt"] = prompt
                results.append(rec)
                empty = rec.get("is_empty_after_strip", False)
                marker = " ←EMPTY" if empty else ""
                resp_preview = rec.get("response_stripped", "")[:60]
                if "error" in rec:
                    resp_preview = f"ERROR: {rec['error'][:60]}"
                print(f"  T={temp} {label:22s} eval={rec.get('eval_count'):>4} "
                      f"reason={rec.get('done_reason','?'):>10} "
                      f"len={rec.get('response_len_stripped',0):>4} "
                      f"{repr(resp_preview):<70}{marker}", flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")

    # Summary
    print("\n=== SUMMARY: empty-response cells (model x prompt) ===")
    by_cell = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["model"], r["prompt_label"])
        by_cell.setdefault(key, []).append(r["is_empty_after_strip"])
    for (model, prompt), empties in sorted(by_cell.items()):
        n_empty = sum(empties)
        if n_empty > 0:
            print(f"  {model:18s} {prompt:25s} {n_empty}/{len(empties)} empty")


if __name__ == "__main__":
    main()
