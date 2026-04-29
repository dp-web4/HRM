"""
S128b — qualitative classification of intent_heuristic ROUTED responses.

S128 found 11.7% of fleet SAGE responses match at least one pattern.
S128b focuses on the responses that would actually ROUTE to a tool
(pass per-tool validation).

For each routed response, classify the match as:
  - INTENT (model intends to invoke the tool)
  - METAPHOR (verb used metaphorically/cognitively, not as tool invocation)
  - REFLECTIVE (first-person reflective sentence containing the verb form)
  - LEXICAL_COINCIDENCE (e.g., "Phi-4 12B" matching arithmetic regex)
  - DESCRIPTIVE (model describing what the tool does without invoking)
  - AMBIGUOUS (cannot tell from context alone)

Sample 60 routed responses (10 per tool category, weighted by fleet
distribution) for hand-classification. Output the per-class rate.
"""

import json
import glob
import re
import sys
import random
from pathlib import Path
from collections import defaultdict, Counter

# Reuse parse_response_emulation logic
sys.path.insert(0, str(Path(__file__).parent))
from s128_intent_heuristic_audit import (
    parse_response_emulation,
    all_matches,
    ALL_PATTERNS,
)


def collect_routed():
    """Collect every SAGE response that would route to a tool."""
    files = glob.glob('/home/dp/ai-workspace/SAGE/sage/instances/*/sessions/session_*.json')
    routed = defaultdict(list)
    for f in files:
        try:
            with open(f) as fp:
                s = json.load(fp)
        except Exception:
            continue
        instance = f.split('/instances/')[1].split('/sessions/')[0]
        for turn in s.get('conversation', []):
            speaker = (turn.get('speaker') or '').lower()
            text = turn.get('text') or ''
            if not text:
                continue
            if speaker not in ('sage', 'model', instance.split('-')[0]):
                continue
            tool, pid = parse_response_emulation(text)
            if tool:
                routed[tool].append({
                    "instance": instance,
                    "session": Path(f).stem,
                    "pid": pid,
                    "text": text,
                })
    return routed


def find_match_context(text, pid):
    for p, pattern, _ in ALL_PATTERNS:
        if p == pid:
            m = pattern.search(text)
            if m:
                start = max(0, m.start() - 100)
                end = min(len(text), m.end() + 100)
                return text[start:end], m.group(0)
    return text[:200], ""


def main():
    routed = collect_routed()
    print(f"[S128b] routed totals: { {k: len(v) for k, v in routed.items()} }", file=sys.stderr)

    random.seed(20260429)
    sample = []
    for tool, items in routed.items():
        n = min(15, len(items))
        chosen = random.sample(items, n)
        for c in chosen:
            ctx, matched_text = find_match_context(c["text"], c["pid"])
            sample.append({
                "tool": tool,
                "pid": c["pid"],
                "instance": c["instance"],
                "session": c["session"],
                "matched_text": matched_text,
                "context": ctx,
                "full_response": c["text"],
            })

    out = {
        "total_routed": {k: len(v) for k, v in routed.items()},
        "sample_count": len(sample),
        "samples": sample,
    }

    with open('/home/dp/ai-workspace/SAGE/sage/raising/analysis/s128_data/s128b_routing_samples.json', 'w') as fo:
        json.dump(out, fo, indent=2)

    print(f"[S128b] sample written: {len(sample)} cases")
    print()
    for s in sample:
        ctx = s["context"].replace("\n", " / ")[:300]
        print(f"\n[{s['tool']:12s} via {s['pid']:25s}] {s['instance'][:18]} {s['session']}")
        print(f"  matched: '{s['matched_text']}'")
        print(f"  ctx:     '{ctx}'")


if __name__ == "__main__":
    main()
