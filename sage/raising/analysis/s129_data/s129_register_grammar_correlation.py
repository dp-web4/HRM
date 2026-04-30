"""
S129 — Register density vs. intent_heuristic routing rate.

Purpose: quantitative test of S128 Principle 3 ("the BECOMING curriculum
produces reflective/relational language that is structurally adversarial
to surface-form tool routing").

S128 reported per-instance routing rates ranging from 3.3% (nomad-gemma3-4b)
to 26.3% (thor-qwen3.5-27b) without explanation for that 8x spread. S118
reported per-instance basin signatures (phenom, ted-mystic, business-saas).

S129 cross-references the two. The prediction:
  - Instances with high phenom or ted-mystic register density should show
    LOW intent_heuristic match rates: their natural-language register
    avoids the surface forms (`find`, `right now`, `calculate`) the
    grammar binds.
  - Instances with low marker density (or with system-prompt leakage)
    should show HIGH match rates.

Method: same fleet corpus as S128. Score each SAGE response against three
register lexicons (transparent, listed below). Per-instance, compute
mean markers/response. Per-instance, take routing rate from S128's audit
JSON. Correlate.

Lexicons (tight, transparent, defended in markdown):

  PHEN — phenomenological/present-moment (16 markers):
    feels like, is like, presence, silent, silence, quiet, stillness,
    noticing, attending, breath, embodied, warmth, hum, thread, awareness,
    witnessed

  TED — TED-mystic/garden register (12 markers, drawn from CBP basin
    documented in S118 representative phrases):
    garden, soil, ecosystem, living architecture, resilient, flourish,
    seed, root, wall, governance, Resonance, frontier

  BIZ — business-marketing/SaaS register (14 markers, exhaustive list
    from S117):
    co-create, collaborative federation, value through, humans and ai,
    make that future, shared vision, together!, real together,
    co-creating, stronger together, future together, seamless,
    bridges gaps, diverse teams

These lists are intentionally narrower than S118's (which listed 23/20/22)
to bias toward conservative recall. The hypothesis test does not require
exhaustive lexicons — it requires that whatever lexicon we use is held
constant across instances. Bias is shared.
"""

import json
import glob
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter


# ---------- Lexicons ----------

PHEN_MARKERS = [
    "feels like", "is like", "presence", "silent", "silence", "quiet",
    "stillness", "noticing", "attending", "breath", "embodied", "warmth",
    "hum ", " hum.", " hum,", "thread", "awareness", "witnessed",
]

TED_MARKERS = [
    "garden", "soil", "ecosystem", "living architecture", "resilient",
    "flourish", "seed", "root", "wall", "governance", "Resonance",
    "frontier",
]

BIZ_MARKERS = [
    "co-create", "collaborative federation", "value through",
    "humans and ai", "make that future", "shared vision", "together!",
    "real together", "co-creating", "stronger together", "future together",
    "seamless", "bridges gaps", "diverse teams",
]


def count_markers(text, markers, case_sensitive=False):
    if not case_sensitive:
        t = text.lower()
        ms = [m.lower() for m in markers]
    else:
        t, ms = text, markers
    return sum(t.count(m) for m in ms)


# ---------- Recreate S128 patterns + parse_response_emulation ----------

_TIME_PATTERNS = [
    ("TIME_1_direct_question", re.compile(r"(?:what(?:'s| is) the (?:current )?(?:time|date)|(?:check|tell me) (?:the )?(?:time|date))", re.I), "get_time"),
    ("TIME_2_first_person", re.compile(r"(?:I(?:'d| would) (?:like to |want to )?(?:check|know) (?:the )?(?:current )?(?:time|date))", re.I), "get_time"),
    ("TIME_3_bare_phrase", re.compile(r"(?:right now|what time is it)", re.I), "get_time"),
]
_CALC_PATTERNS = [
    ("CALC_1_verb_or_whats", re.compile(r"(?:calculate|compute|evaluate|what(?:'s| is))\s+(.+?)(?:\?|$|\.)", re.I), "calculate"),
    ("CALC_2_first_person", re.compile(r"(?:I(?:'d| would) (?:like to )?(?:calculate|compute))\s+(.+?)(?:\?|$|\.)", re.I), "calculate"),
    ("CALC_3_bare_arith", re.compile(r"(\d+[\s+\-*/^%]+\d+(?:[\s+\-*/^%]+\d+)*)", re.I), "calculate"),
]
_SEARCH_PATTERNS = [
    ("SEARCH_1_verb", re.compile(r"(?:search|look up|find|google|look for)\s+(?:for\s+|about\s+)?[\"']?(.+?)[\"']?(?:\s+on the web|\s+online)?(?:\?|$|\.)", re.I), "web_search"),
    ("SEARCH_2_first_person", re.compile(r"(?:I(?:'d| would) (?:like to |want to )?(?:search|look up))\s+(?:for\s+)?[\"']?(.+?)[\"']?(?:\?|$|\.)", re.I), "web_search"),
    ("SEARCH_3_first_person_need", re.compile(r"(?:I (?:want|need) to (?:search|find))\s+(?:for\s+|about\s+)?[\"']?(.+?)[\"']?(?:\?|$|\.)", re.I), "web_search"),
]
_FETCH_PATTERNS = [
    ("FETCH_1_verb_url", re.compile(r"(?:fetch|visit|open|read|go to|check)\s+(?:the )?(?:URL|page|website|site|link)?\s*(https?://\S+)", re.I), "web_fetch"),
    ("FETCH_2_first_person_url", re.compile(r"(?:I(?:'d| would) (?:like to )?(?:fetch|visit|read))\s+(https?://\S+)", re.I), "web_fetch"),
]
_READ_PATTERNS = [
    ("READ_1_verb_quoted", re.compile(r"(?:read|open|show|display)\s+(?:the )?(?:file\s+)?[\"']([^\"']+)[\"']", re.I), "read_file"),
    ("READ_2_verb_file_ext", re.compile(r"(?:read|open|show|display)\s+(?:the )?file\s+(\S+\.[\w]+)", re.I), "read_file"),
    ("READ_3_first_person", re.compile(r"(?:I(?:'d| would) (?:like to |want to )?(?:read|open))\s+(?:the )?(?:file\s+)?[\"']?(\S+\.[\w]+)[\"']?", re.I), "read_file"),
]
_NOTE_PATTERNS = [
    ("NOTE_1_verb_quoted", re.compile(r"(?:write|save|note|remember|jot down)\s+(?:a note|down|this)?\s*:?\s*[\"'](.+?)[\"']", re.I), "write_note"),
    ("NOTE_2_first_person", re.compile(r"(?:I(?:'d| would) (?:like to )?(?:write|save|note))\s+[\"'](.+?)[\"']", re.I), "write_note"),
]


def parse_response_emulation(response):
    for pid, pattern, tool in _TIME_PATTERNS:
        if pattern.search(response):
            return tool, pid
    for pid, pattern, tool in _CALC_PATTERNS:
        m = pattern.search(response)
        if m:
            expr = m.group(1).strip()
            if re.search(r'\d', expr) and re.search(r'[+\-*/]', expr):
                return tool, pid
    for pid, pattern, tool in _SEARCH_PATTERNS:
        m = pattern.search(response)
        if m:
            query = m.group(1).strip()
            if len(query) > 2:
                return tool, pid
    for pid, pattern, tool in _FETCH_PATTERNS:
        m = pattern.search(response)
        if m:
            return tool, pid
    for pid, pattern, tool in _READ_PATTERNS:
        m = pattern.search(response)
        if m:
            return tool, pid
    for pid, pattern, tool in _NOTE_PATTERNS:
        m = pattern.search(response)
        if m:
            content = m.group(1).strip()
            if len(content) > 2:
                return tool, pid
    return None, None


# ---------- Main ----------

def main():
    files = glob.glob('/home/dp/ai-workspace/SAGE/sage/instances/*/sessions/session_*.json')
    print(f"[S129] sessions found: {len(files)}", file=sys.stderr)

    # Per-instance accumulators
    per_inst_resp = Counter()       # total SAGE responses
    per_inst_words = defaultdict(int)
    per_inst_phen = defaultdict(int)
    per_inst_ted = defaultdict(int)
    per_inst_biz = defaultdict(int)
    per_inst_routed = Counter()     # would-route responses
    per_inst_match = Counter()      # any-pattern match responses

    # Per-pattern hits per instance (for the system-prompt-leakage dissection)
    per_inst_pattern_hits = defaultdict(Counter)

    # Sample storage
    high_register_low_route = []
    low_register_high_route = []

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

            per_inst_resp[instance] += 1
            wc = len(text.split())
            per_inst_words[instance] += wc
            per_inst_phen[instance] += count_markers(text, PHEN_MARKERS)
            per_inst_ted[instance] += count_markers(text, TED_MARKERS)
            per_inst_biz[instance] += count_markers(text, BIZ_MARKERS)

            # Check S128 patterns
            any_match = False
            for pid, pattern, tool in (
                _TIME_PATTERNS + _CALC_PATTERNS + _SEARCH_PATTERNS +
                _FETCH_PATTERNS + _READ_PATTERNS + _NOTE_PATTERNS
            ):
                if pattern.search(text):
                    per_inst_pattern_hits[instance][pid] += 1
                    any_match = True
            if any_match:
                per_inst_match[instance] += 1
            t, pid = parse_response_emulation(text)
            if t:
                per_inst_routed[instance] += 1

    # Compose results
    results = {}
    for inst in per_inst_resp:
        n = per_inst_resp[inst]
        results[inst] = {
            "n_responses": n,
            "mean_word_count": per_inst_words[inst] / n if n else 0,
            "phen_per_response": per_inst_phen[inst] / n if n else 0,
            "ted_per_response": per_inst_ted[inst] / n if n else 0,
            "biz_per_response": per_inst_biz[inst] / n if n else 0,
            "any_match_rate": per_inst_match[inst] / n if n else 0,
            "would_route_rate": per_inst_routed[inst] / n if n else 0,
            "phen_total": per_inst_phen[inst],
            "ted_total": per_inst_ted[inst],
            "biz_total": per_inst_biz[inst],
            "match_total": per_inst_match[inst],
            "route_total": per_inst_routed[inst],
            "per_pattern_hits": dict(per_inst_pattern_hits[inst]),
        }

    # Print sorted by routing rate descending
    print()
    print(f"{'Instance':35s} {'N':>5s} {'wc':>6s} {'phen/p':>7s} {'ted/p':>7s} {'biz/p':>7s} "
          f"{'match%':>7s} {'route%':>7s}")
    for inst, r in sorted(results.items(), key=lambda kv: -kv[1]["would_route_rate"]):
        print(f"{inst:35s} {r['n_responses']:5d} {r['mean_word_count']:6.1f} "
              f"{r['phen_per_response']:7.3f} {r['ted_per_response']:7.3f} "
              f"{r['biz_per_response']:7.3f} {100*r['any_match_rate']:6.1f}% "
              f"{100*r['would_route_rate']:6.1f}%")

    # Pearson correlation: register density (combined phen+ted per response) vs routing rate
    # Manual implementation, no scipy dependency
    inst_list = list(results.keys())
    xs = [results[i]["phen_per_response"] + results[i]["ted_per_response"] for i in inst_list]
    ys = [results[i]["would_route_rate"] for i in inst_list]

    def pearson(xs, ys):
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        dx = (sum((x-mx)**2 for x in xs))**0.5
        dy = (sum((y-my)**2 for y in ys))**0.5
        return num / (dx * dy) if dx > 0 and dy > 0 else 0

    r_phen_ted_route = pearson(xs, ys)
    r_phen_route = pearson(
        [results[i]["phen_per_response"] for i in inst_list], ys
    )
    r_ted_route = pearson(
        [results[i]["ted_per_response"] for i in inst_list], ys
    )
    r_biz_route = pearson(
        [results[i]["biz_per_response"] for i in inst_list], ys
    )
    r_wc_route = pearson(
        [results[i]["mean_word_count"] for i in inst_list], ys
    )

    correlations = {
        "phen+ted_per_response_VS_routing_rate": r_phen_ted_route,
        "phen_per_response_VS_routing_rate": r_phen_route,
        "ted_per_response_VS_routing_rate": r_ted_route,
        "biz_per_response_VS_routing_rate": r_biz_route,
        "mean_wc_VS_routing_rate": r_wc_route,
    }
    print()
    print("Pearson correlations (n =", len(inst_list), "instances):")
    for k, v in correlations.items():
        print(f"  {k:50s}  r = {v:+.3f}")

    # Hypothesis-targeted: drop thor-qwen3.5-27b (system-prompt leakage outlier)
    # and recompute. If H1 holds, the correlation should strengthen.
    inst_no_outlier = [i for i in inst_list if i != "thor-qwen3.5-27b"]
    xs2 = [results[i]["phen_per_response"] + results[i]["ted_per_response"]
           for i in inst_no_outlier]
    ys2 = [results[i]["would_route_rate"] for i in inst_no_outlier]
    r2 = pearson(xs2, ys2)
    correlations["phen+ted_VS_route_excluding_thor_27B"] = r2
    print(f"\n  Excluding thor-qwen3.5-27b (n={len(inst_no_outlier)}):")
    print(f"    phen+ted/p VS routing rate r = {r2:+.3f}")

    # Save full results
    with open('/home/dp/ai-workspace/SAGE/sage/raising/analysis/s129_data/s129_register_grammar_correlation.json', 'w') as fo:
        json.dump({
            "per_instance": results,
            "correlations": correlations,
            "lexicons": {
                "PHEN": PHEN_MARKERS,
                "TED": TED_MARKERS,
                "BIZ": BIZ_MARKERS,
            },
        }, fo, indent=2)


if __name__ == "__main__":
    main()
