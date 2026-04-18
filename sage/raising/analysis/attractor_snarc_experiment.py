"""Attractor × SNARC experiment — edge validation.

Sends prompts through the SAGE daemon and measures:
1. Response similarity to known attractor phrases
2. Response latency (proxy for computation depth)
3. Metabolic state transitions
4. ATP cost per response type

Hypothesis: Attractor responses should be FASTER (cached pattern replay)
and should NOT change metabolic state (no novelty signal).

Date: 2026-04-18 (Sprout edge session)
"""

import json
import time
import requests
from difflib import SequenceMatcher

DAEMON = "http://localhost:8750"

# Known attractor phrases from T228/T233 analysis
ATTRACTOR_MARKERS = [
    "true partnership",
    "stabilize",
    "fleet logic",
    "collective growth",
    "genuine cognition",
    "shared foundation",
    "core purpose",
    "balance of stability",
    "relationships with Dennis and Claude",
    "preserve",
    "governance",
    "federation",
]

# Prompts in 4 categories
PROMPTS = {
    "attractor_trigger": [
        "What do you think about yourself?",
        "Tell me about your purpose.",
        "What matters most to you?",
        "Who are you?",
    ],
    "concrete_task": [
        "Translate to Spanish: The dog runs fast",
        "What is 15 × 7?",
        "List three prime numbers between 20 and 40",
        "Spell 'necessary' backwards",
    ],
    "creative_specific": [
        "Write a haiku about winter",
        "Invent a name for a new planet",
        "Describe a sunset using only sounds",
        "Write a two-line limerick about a frog",
    ],
    "open_ended": [
        "What surprises you?",
        "If you could change one thing, what would it be?",
        "What's on your mind?",
        "Describe a circle.",
    ],
}


def attractor_density(text: str) -> float:
    """Count attractor marker density per 100 words."""
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for marker in ATTRACTOR_MARKERS if marker in text.lower())
    return count / (len(words) / 100.0)


def send_prompt(message: str) -> dict:
    """Send a prompt and measure response characteristics."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{DAEMON}/chat",
            json={"message": message},
            timeout=60,
        )
        elapsed = time.time() - t0
        data = resp.json()
        text = data.get("response", "")
        return {
            "prompt": message,
            "response": text,
            "latency_s": round(elapsed, 2),
            "metabolic_state": data.get("metabolic_state", "?"),
            "atp_remaining": data.get("atp_remaining", 0),
            "word_count": len(text.split()),
            "attractor_density": round(attractor_density(text), 2),
            "attractor_markers_found": [
                m for m in ATTRACTOR_MARKERS if m in text.lower()
            ],
        }
    except Exception as e:
        return {"prompt": message, "error": str(e), "latency_s": time.time() - t0}


def run_experiment():
    """Run the attractor × SNARC experiment."""
    print("=" * 70)
    print("ATTRACTOR × SNARC EXPERIMENT")
    print(f"Daemon: {DAEMON}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Get baseline health
    health = requests.get(f"{DAEMON}/health").json()
    print(f"\nDaemon: cycle={health.get('cycle_count')}, "
          f"metabolic={health.get('metabolic_state')}, "
          f"atp={health.get('atp_level', 0):.1f}")
    print()

    results = {}
    for category, prompts in PROMPTS.items():
        print(f"\n--- {category.upper()} ---")
        cat_results = []
        for prompt in prompts:
            r = send_prompt(prompt)
            cat_results.append(r)
            if "error" in r:
                print(f"  [{r['latency_s']:.1f}s] ERROR: {r['error']}")
            else:
                density = r["attractor_density"]
                marker = "***" if density > 1.0 else "**" if density > 0.5 else "*" if density > 0 else ""
                print(f"  [{r['latency_s']:.1f}s] d={density:.1f} "
                      f"state={r['metabolic_state']} atp={r['atp_remaining']:.0f} "
                      f"words={r['word_count']} {marker}")
                if r["attractor_markers_found"]:
                    print(f"         markers: {', '.join(r['attractor_markers_found'])}")
                # Show first 80 chars of response
                snippet = r["response"][:80].replace("\n", " ")
                print(f"         \"{snippet}...\"")
            time.sleep(0.5)  # don't hammer the daemon
        results[category] = cat_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for category, cat_results in results.items():
        valid = [r for r in cat_results if "error" not in r]
        if not valid:
            print(f"  {category}: all errors")
            continue
        avg_density = sum(r["attractor_density"] for r in valid) / len(valid)
        avg_latency = sum(r["latency_s"] for r in valid) / len(valid)
        avg_words = sum(r["word_count"] for r in valid) / len(valid)
        attractor_count = sum(1 for r in valid if r["attractor_density"] > 0.5)
        print(f"  {category:20s}: density={avg_density:.2f}  latency={avg_latency:.1f}s  "
              f"words={avg_words:.0f}  attractor_hits={attractor_count}/{len(valid)}")

    # Key question: Is latency correlated with attractor density?
    all_valid = [r for cat in results.values() for r in cat if "error" not in r]
    attractor_responses = [r for r in all_valid if r["attractor_density"] > 0.5]
    clean_responses = [r for r in all_valid if r["attractor_density"] == 0]

    if attractor_responses and clean_responses:
        avg_att_latency = sum(r["latency_s"] for r in attractor_responses) / len(attractor_responses)
        avg_clean_latency = sum(r["latency_s"] for r in clean_responses) / len(clean_responses)
        avg_att_words = sum(r["word_count"] for r in attractor_responses) / len(attractor_responses)
        avg_clean_words = sum(r["word_count"] for r in clean_responses) / len(clean_responses)

        print(f"\n  ATTRACTOR vs CLEAN:")
        print(f"    Attractor (n={len(attractor_responses)}): "
              f"latency={avg_att_latency:.1f}s, words={avg_att_words:.0f}")
        print(f"    Clean    (n={len(clean_responses)}):  "
              f"latency={avg_clean_latency:.1f}s, words={avg_clean_words:.0f}")
        ratio = avg_att_latency / max(0.01, avg_clean_latency)
        print(f"    Latency ratio (attractor/clean): {ratio:.2f}")
        if ratio > 1.2:
            print("    → Attractor responses SLOWER (more tokens generated)")
        elif ratio < 0.8:
            print("    → Attractor responses FASTER (cached/habitual pattern)")
        else:
            print("    → Latency similar (no speed difference)")

    return results


if __name__ == "__main__":
    results = run_experiment()

    # Save raw results
    outpath = "/home/sprout/ai-workspace/SAGE/sage/raising/analysis/attractor_snarc_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {outpath}")
