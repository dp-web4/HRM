"""
S128 — intent_heuristic.py grammar audit.

Carry-forward from S127 #49: apply S125/S126 audit discipline to
sage/tools/grammars/intent_heuristic.py — list each pattern,
classify by intended function, find function-overlapping pairs.

S125 audit primitive: path-trace every dimension consulted, not just
                      the dimension picked.
S126 audit primitive: are alternations within a single dimension
                      functionally homogeneous?

Method: run every regex over a corpus of SAGE responses; tag every
match with its pattern source; sample matches for qualitative review.

Output: s128_intent_heuristic_audit.json (per-pattern match counts +
sample matches with surrounding context).
"""

import json
import glob
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Recreate the patterns verbatim from intent_heuristic.py.
# Each entry is (pattern_id, regex, intended_tool).

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

ALL_PATTERNS = (
    _TIME_PATTERNS + _CALC_PATTERNS + _SEARCH_PATTERNS +
    _FETCH_PATTERNS + _READ_PATTERNS + _NOTE_PATTERNS
)

# Precedence chain mirrors parse_response: time → calc → search → fetch → read → note
PRECEDENCE = ["TIME", "CALC", "SEARCH", "FETCH", "READ", "NOTE"]


def parse_response_emulation(response):
    """Emulate intent_heuristic.parse_response. Returns the tool that would
    actually be picked given precedence + per-tool validation."""
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


def all_matches(response):
    """Path-trace: return ALL pattern matches across all sets, not just the
    first one in precedence order. This is the S125 audit primitive."""
    hits = []
    for pid, pattern, tool in ALL_PATTERNS:
        m = pattern.search(response)
        if m:
            try:
                arg = m.group(1).strip() if m.lastindex else ""
            except IndexError:
                arg = ""
            hits.append((pid, tool, arg, m.start(), m.end()))
    return hits


def main():
    files = glob.glob('/home/dp/ai-workspace/SAGE/sage/instances/*/sessions/session_*.json')
    print(f"[S128] sessions found: {len(files)}", file=sys.stderr)

    pattern_hits = defaultdict(int)
    pattern_samples = defaultdict(list)
    routed_tool_hits = defaultdict(int)  # what parse_response WOULD pick
    multi_match_count = 0
    total_responses = 0
    total_matched = 0
    by_instance = Counter()
    by_instance_routed = defaultdict(Counter)

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
            # SAGE responses are the ones we audit (only model-generated text
            # would feed parse_response).
            if speaker not in ('sage', 'model', instance.split('-')[0]):
                continue
            total_responses += 1

            hits = all_matches(text)
            if hits:
                total_matched += 1
                by_instance[instance] += 1
                if len(hits) > 1:
                    multi_match_count += 1
                for pid, tool, arg, start, end in hits:
                    pattern_hits[pid] += 1
                    if len(pattern_samples[pid]) < 8:
                        ctx_start = max(0, start - 60)
                        ctx_end = min(len(text), end + 60)
                        pattern_samples[pid].append({
                            "instance": instance,
                            "session": Path(f).stem,
                            "arg": arg[:120],
                            "context": text[ctx_start:ctx_end],
                            "all_hits_in_response": [h[0] for h in hits],
                        })
                tool, pid = parse_response_emulation(text)
                if tool:
                    routed_tool_hits[tool] += 1
                    by_instance_routed[instance][tool] += 1

    out = {
        "summary": {
            "total_sage_responses_audited": total_responses,
            "total_with_any_pattern_match": total_matched,
            "false_positive_rate_upper_bound": (
                total_matched / total_responses if total_responses else 0
            ),
            "responses_with_multiple_pattern_matches": multi_match_count,
            "pct_multi_match_among_matches": (
                multi_match_count / total_matched if total_matched else 0
            ),
        },
        "per_pattern_hits": dict(pattern_hits),
        "routed_tool_distribution": dict(routed_tool_hits),
        "by_instance_total_matches": dict(by_instance),
        "by_instance_routed_tool": {
            k: dict(v) for k, v in by_instance_routed.items()
        },
        "samples_per_pattern": dict(pattern_samples),
    }

    with open('/home/dp/ai-workspace/SAGE/sage/raising/analysis/s128_data/s128_intent_heuristic_audit.json', 'w') as fo:
        json.dump(out, fo, indent=2)

    print(f"[S128] total_responses: {total_responses}")
    print(f"[S128] matched: {total_matched} ({100*total_matched/total_responses:.1f}%)")
    print(f"[S128] multi-match: {multi_match_count} ({100*multi_match_count/max(1,total_matched):.1f}% of matches)")
    print()
    print("Per-pattern hits:")
    for pid in [p[0] for p in ALL_PATTERNS]:
        print(f"  {pid:30s}  {pattern_hits.get(pid,0):6d}")
    print()
    print("Routed tool distribution (S125 precedence pick):")
    for tool, n in sorted(routed_tool_hits.items(), key=lambda kv: -kv[1]):
        print(f"  {tool:15s}  {n}")


if __name__ == "__main__":
    main()
