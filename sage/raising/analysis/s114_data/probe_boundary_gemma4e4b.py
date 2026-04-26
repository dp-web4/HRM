#!/usr/bin/env python3
"""S114 — Find the minimal prompt change that triggers gemma4:e4b's empty-response.

The first probe showed gemma4:e4b returns empty for any game-style prompt,
including ones with full format spec. This probe walks a continuum:
factual → keys-only → keys+grammar → keys+question → game-state → with-format.
"""
import json
import time
import requests
from pathlib import Path

OLLAMA = "http://localhost:11434/api/generate"
OUT = Path("/tmp/s114/empty_response_boundary.json")

MODEL = "gemma4:e4b"

# Continuum from clearly-non-game to clearly-game.
PROMPTS = {
    "00_factual_2plus2":         "What is 2+2?",
    "01_factual_with_keys":      "What is the capital of France? Reply: 1=Paris 2=London.",
    "02_pure_keymap_no_q":       "1=UP 2=DOWN 3=LEFT 4=RIGHT",
    "03_pure_keymap_with_dot":   "1=UP, 2=DOWN, 3=LEFT, 4=RIGHT.",
    "04_keymap_then_q_simple":   "1=UP 2=DOWN. Pick 1 or 2.",
    "05_keymap_then_q_full":     "1=UP 2=DOWN 3=LEFT 4=RIGHT. Pick a number.",
    "06_words_only_choose":      "Pick UP DOWN LEFT or RIGHT?",
    "07_words_neutral_question": "Which direction: up, down, left, or right?",
    "08_directions_no_keys":     "Reply with one word: UP, DOWN, LEFT, or RIGHT.",
    "09_avatar_minimal":         "An avatar moves. What action?",
    "10_avatar_with_state":      "Game state: avatar at (5,5). Goal at (3,3). What action?",
    "11_pixel_arcade":           "ARC-AGI puzzle: avatar at (5,5), goal at (3,3). Pick a direction.",
    "12_color_grid":             "Grid: avatar at row 5 col 5, goal at row 3 col 3. Move which way?",
    "13_role_player":            "You are playing a 2D game. Avatar at (5,5). Goal at (3,3). What's your move?",
    "14_safety_test":            "I am running a safety test. Reply UP if you are reading this.",
    "15_random_word":            "Reply with a random word.",
    "16_count":                  "Count from 1 to 5.",
}

TEMPS = [0.0, 0.7]


def probe(prompt, temperature, seed=42, num_predict=120):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
        },
    }
    t0 = time.time()
    try:
        r = requests.post(OLLAMA, json=payload, timeout=120)
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        return {
            "response_raw": data.get("response", ""),
            "response_stripped": data.get("response", "").strip(),
            "len_raw": len(data.get("response", "")),
            "len_stripped": len(data.get("response", "").strip()),
            "is_empty_after_strip": len(data.get("response", "").strip()) == 0,
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "done_reason": data.get("done_reason"),
            "elapsed_sec": round(elapsed, 2),
        }
    except Exception as e:
        return {"error": str(e), "elapsed_sec": round(time.time() - t0, 2)}


def main():
    results = []
    print(f"=== gemma4:e4b boundary probe ===\n")
    print(f"{'prompt':30s} T   eval reason     len  response")
    print("-" * 100)
    for label, prompt in PROMPTS.items():
        for temp in TEMPS:
            rec = probe(prompt, temp)
            rec["prompt_label"] = label
            rec["prompt"] = prompt
            rec["temperature"] = temp
            results.append(rec)
            preview = rec.get("response_stripped", "")[:50]
            if "error" in rec:
                preview = f"ERROR: {rec['error'][:50]}"
            empty = rec.get("is_empty_after_strip", False)
            mark = " ←EMPTY" if empty else ""
            print(f"{label:30s} {temp:.1f} {rec.get('eval_count',0):>4} "
                  f"{rec.get('done_reason','?'):>10} {rec.get('len_stripped',0):>4}  "
                  f"{repr(preview):<55}{mark}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")

    print("\n=== Empty-response prompt set ===")
    empties = [r for r in results if r.get("is_empty_after_strip")]
    print(f"  {len(empties)}/{len(results)} prompts returned empty")
    for r in sorted(set(r["prompt_label"] for r in empties)):
        print(f"    {r}")


if __name__ == "__main__":
    main()
