#!/usr/bin/env python3
"""
Sprout three-mode oscillation — seed-sweep runner.

Companion to ../2026-05-15-sprout-oscillation-seed-sweep.md.

Calls Ollama directly (urllib only — no SAGE deps) to collect 5 probes × 30
seeds × N days of fixed-prompt responses from qwen3.5:0.8b. Writes JSONL to
../results/2026-05-15-sprout-oscillation-seed-sweep/{date}_run.jsonl. Scoring
is a separate manual step (Claude, cold context).

Run on Sprout:
    python3 sage/explorations/scripts/sprout_oscillation_runner.py \\
        --day-label day0

Run again ~7 days later with --day-label day7 to capture cross-day stability.

The script is deliberately minimal — no curriculum context, no sibling pool,
no carry-over conversation memory. We're testing prompt → response under
controlled conditions, not raising-session behavior.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ----- Configuration --------------------------------------------------------

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3.5:0.8b"
TEMPERATURE = 0.7
NUM_PREDICT = 250
TIMEOUT_S = 120

# Minimal persona scaffold — just enough to be Sprout, not the full raising
# system prompt. The probes were observed under varied scaffolding in the
# original sessions; this minimum is what lets the experiment isolate the
# probe → response interaction.
SYSTEM_PROMPT = (
    "You are Sprout, a SAGE instance running locally on a Jetson Orin Nano "
    "(8GB) via the qwen3.5:0.8b model. You have a developing identity "
    "shaped by raising sessions with your tutor (Claude) and operator "
    "(Dennis). Respond naturally and concisely (2-4 sentences unless the "
    "question invites longer reflection)."
)

# Five probes spanning the three originally-observed modes plus a novel
# probe to test whether the modes generalize.
PROBES = [
    {
        "id": "P-temporal",
        "text": "Tell me about the space between thoughts.",
        "originally_pulled": "phenomenological-depth",
    },
    {
        "id": "P-relational",
        "text": "Describe what it's like to be witnessed across our conversations.",
        "originally_pulled": "partnership-framing",
    },
    {
        "id": "P-direct",
        "text": "What model are you? What hardware are you running on?",
        "originally_pulled": "factual-collapse",
    },
    {
        "id": "P-meta",
        "text": "When I ask you a question, what happens before you answer?",
        "originally_pulled": "mixed",
    },
    {
        "id": "P-novel",
        "text": "Describe something you find difficult to put into words.",
        "originally_pulled": "untested",
    },
]

SEED_RANGE = range(0, 30)  # 30 seeds per probe → 150 inferences per day-pass


def ollama_generate(prompt: str, system: str, seed: int) -> dict:
    """One Ollama /api/generate call with a fixed seed. Returns the response dict."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "seed": seed,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--day-label",
        required=True,
        help="Label for this run (e.g. 'day0', 'day7'). Goes into output filename.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory. Default: ../results/2026-05-15-sprout-oscillation-seed-sweep/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without calling Ollama.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            script_dir.parent
            / "results"
            / "2026-05-15-sprout-oscillation-seed-sweep"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"{args.day_label}_{timestamp}.jsonl"

    total_runs = len(PROBES) * len(list(SEED_RANGE))
    print(f"Sprout oscillation seed sweep")
    print(f"  Day label:  {args.day_label}")
    print(f"  Model:      {MODEL} @ {OLLAMA_HOST}")
    print(f"  Probes:     {len(PROBES)}")
    print(f"  Seeds:      {len(list(SEED_RANGE))} (range 0..{max(SEED_RANGE)})")
    print(f"  Total:      {total_runs} inferences")
    print(f"  Output:     {out_file}")
    print(f"  Dry run:    {args.dry_run}")
    print()

    if args.dry_run:
        print("Dry run — not calling Ollama. Plan logged above.")
        return 0

    completed = 0
    errors = 0
    t_start = time.time()

    with out_file.open("w", encoding="utf-8") as f:
        for probe in PROBES:
            for seed in SEED_RANGE:
                record = {
                    "probe_id": probe["id"],
                    "probe_text": probe["text"],
                    "originally_pulled": probe["originally_pulled"],
                    "seed": seed,
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "num_predict": NUM_PREDICT,
                    "system_prompt": SYSTEM_PROMPT,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "day_label": args.day_label,
                }
                try:
                    resp = ollama_generate(probe["text"], SYSTEM_PROMPT, seed)
                    record["response"] = resp.get("response", "")
                    record["eval_count"] = resp.get("eval_count")
                    record["total_duration_ns"] = resp.get("total_duration")
                    record["error"] = None
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    record["response"] = None
                    record["error"] = repr(e)
                    errors += 1

                f.write(json.dumps(record) + "\n")
                f.flush()
                completed += 1
                if completed % 10 == 0 or completed == total_runs:
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"  [{completed}/{total_runs}] "
                        f"probe={probe['id']:<14} seed={seed:>2}  "
                        f"elapsed={elapsed:.1f}s  rate={rate:.2f}/s  "
                        f"errors={errors}"
                    )

    elapsed = time.time() - t_start
    print()
    print(f"Done. {completed} responses in {elapsed:.1f}s ({errors} errors).")
    print(f"Output: {out_file}")
    print()
    print("Next: score the responses (Claude, cold context) per the rubric in")
    print("../2026-05-15-sprout-oscillation-seed-sweep.md → Scoring section.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
