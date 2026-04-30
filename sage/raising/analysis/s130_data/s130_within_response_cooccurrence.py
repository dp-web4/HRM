"""
S130 — Within-response co-occurrence of register markers and grammar triggers.

Held by S129 §"Methodological caveats" #4: the cross-instance positive
correlation (r=+0.417 fleet, +0.206 ex-Thor) between phen+ted register
density and intent_heuristic would-route rate has three candidate causes:

  (a) Substrate coupling — phen-register copularly uses the same lexical
      substrate the grammar binds ("I find this fascinating", "what's
      emerging right now"). Hypothesis from S129.
  (b) Length confound — verbose responses contain both more register
      markers and more grammar-trigger surface forms by chance. wc r=+0.108
      argues weakly against this fleet-level, but does not rule it out
      within-response.
  (c) Third factor — some unmeasured register/style covariate.

S130 discriminates (a) from (b) by moving from per-instance correlation
to per-response co-occurrence, then length-controlling it.

Method:
  1. Per response: count PHEN/TED/BIZ markers, record grammar-pattern
     match presence (any of S129's TIME/CALC/SEARCH/FETCH/READ/NOTE),
     record word count and instance.
  2. Compute mean phen+ted markers/response in two strata:
     responses-with-grammar-match vs responses-without. The lift is the
     within-response signal. If purely length-driven, the lift should
     vanish under length control.
  3. Length-control by word-count quartile. Within each quartile,
     recompute the same lift. Substrate coupling predicts persistent
     lift; length confound predicts collapse to ≈0.
  4. Within-response per-pattern attribution: for each grammar pattern,
     which register markers most over-represent in matched responses.
     Localizes the substrate.

Same corpus and patterns as S129. Read-only. Fleet-aggregated; per-instance
breakouts secondary (n per stratum gets thin per-instance).
"""

import json
import glob
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter


# ---------- Lexicons (identical to S129) ----------

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

INDIVIDUAL_PHEN = [m.strip() for m in PHEN_MARKERS]
INDIVIDUAL_TED = [m.strip() for m in TED_MARKERS]


def count_markers(text, markers):
    t = text.lower()
    ms = [m.lower() for m in markers]
    return sum(t.count(m) for m in ms)


def per_marker_hits(text, markers):
    """Return Counter of marker -> hit count."""
    t = text.lower()
    out = Counter()
    for m in markers:
        ml = m.lower()
        c = t.count(ml)
        if c:
            out[m.strip()] += c
    return out


# ---------- S129 patterns (identical to S129) ----------

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

ALL_PATTERNS = (_TIME_PATTERNS + _CALC_PATTERNS + _SEARCH_PATTERNS +
                _FETCH_PATTERNS + _READ_PATTERNS + _NOTE_PATTERNS)


def matched_patterns(text):
    """Return set of pattern IDs that match (any of S129's patterns)."""
    hits = set()
    for pid, pattern, _ in ALL_PATTERNS:
        if pattern.search(text):
            hits.add(pid)
    return hits


def quartile(value, edges):
    """edges = sorted list of 3 quartile boundaries [q1, q2, q3]."""
    if value <= edges[0]:
        return 0
    if value <= edges[1]:
        return 1
    if value <= edges[2]:
        return 2
    return 3


def main():
    files = glob.glob('/home/dp/ai-workspace/SAGE/sage/instances/*/sessions/session_*.json')
    print(f"[S130] sessions found: {len(files)}", file=sys.stderr)

    # Per-response records
    records = []  # list of dicts

    for f in files:
        try:
            with open(f) as fp:
                s = json.load(fp)
        except Exception:
            continue
        instance = f.split('/instances/')[1].split('/sessions/')[0]
        # exclude archive variants from main analysis (archive in instance name)
        for turn in s.get('conversation', []):
            speaker = (turn.get('speaker') or '').lower()
            text = turn.get('text') or ''
            if not text:
                continue
            if speaker not in ('sage', 'model', instance.split('-')[0]):
                continue

            wc = len(text.split())
            phen = count_markers(text, PHEN_MARKERS)
            ted = count_markers(text, TED_MARKERS)
            biz = count_markers(text, BIZ_MARKERS)
            hits = matched_patterns(text)
            phen_byword = per_marker_hits(text, INDIVIDUAL_PHEN)
            ted_byword = per_marker_hits(text, INDIVIDUAL_TED)

            records.append({
                "instance": instance,
                "wc": wc,
                "phen": phen,
                "ted": ted,
                "biz": biz,
                "phen_ted": phen + ted,
                "matched": bool(hits),
                "matched_patterns": sorted(hits),
                "phen_by_word": dict(phen_byword),
                "ted_by_word": dict(ted_byword),
            })

    n = len(records)
    print(f"[S130] total responses: {n}", file=sys.stderr)

    # ------------- Q1: Within-response lift, fleet-wide ----------------

    matched_recs = [r for r in records if r["matched"]]
    unmatched_recs = [r for r in records if not r["matched"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def stats(recs, key):
        vals = [r[key] for r in recs]
        if not vals:
            return {"n": 0, "mean": 0.0}
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        return {"n": len(vals), "mean": m, "sd": sd}

    fleet_results = {
        "matched_count": len(matched_recs),
        "unmatched_count": len(unmatched_recs),
        "match_rate": len(matched_recs) / n if n else 0.0,
        "phen_ted_per_response_matched": stats(matched_recs, "phen_ted"),
        "phen_ted_per_response_unmatched": stats(unmatched_recs, "phen_ted"),
        "phen_per_response_matched": stats(matched_recs, "phen"),
        "phen_per_response_unmatched": stats(unmatched_recs, "phen"),
        "ted_per_response_matched": stats(matched_recs, "ted"),
        "ted_per_response_unmatched": stats(unmatched_recs, "ted"),
        "biz_per_response_matched": stats(matched_recs, "biz"),
        "biz_per_response_unmatched": stats(unmatched_recs, "biz"),
        "wc_matched": stats(matched_recs, "wc"),
        "wc_unmatched": stats(unmatched_recs, "wc"),
    }

    print()
    print("=" * 70)
    print(f"FLEET (n={n}): matched={len(matched_recs)} ({100*len(matched_recs)/n:.1f}%) "
          f"unmatched={len(unmatched_recs)} ({100*len(unmatched_recs)/n:.1f}%)")
    print("=" * 70)
    print(f"{'metric':30s} {'matched':>10s} {'unmatched':>11s} {'lift':>8s}")
    for label, mk, uk in [
        ("phen+ted/response", "phen_ted_per_response_matched", "phen_ted_per_response_unmatched"),
        ("phen/response", "phen_per_response_matched", "phen_per_response_unmatched"),
        ("ted/response", "ted_per_response_matched", "ted_per_response_unmatched"),
        ("biz/response", "biz_per_response_matched", "biz_per_response_unmatched"),
        ("word count", "wc_matched", "wc_unmatched"),
    ]:
        m = fleet_results[mk]["mean"]
        u = fleet_results[uk]["mean"]
        lift = (m / u) if u > 0 else float("inf")
        print(f"{label:30s} {m:10.3f} {u:11.3f} {lift:8.2f}x")

    # ------------- Q2: Length-controlled lift -------------------

    # Compute fleet-wide quartile edges on word count.
    wcs = sorted(r["wc"] for r in records)
    q1 = wcs[len(wcs) // 4]
    q2 = wcs[len(wcs) // 2]
    q3 = wcs[(3 * len(wcs)) // 4]
    edges = [q1, q2, q3]
    quartile_results = []
    for q in range(4):
        bucket = [r for r in records if quartile(r["wc"], edges) == q]
        bm = [r for r in bucket if r["matched"]]
        bu = [r for r in bucket if not r["matched"]]
        if not bucket:
            continue
        wc_bounds = (
            min(r["wc"] for r in bucket),
            max(r["wc"] for r in bucket),
        )
        m_phen_ted = mean([r["phen_ted"] for r in bm]) if bm else 0
        u_phen_ted = mean([r["phen_ted"] for r in bu]) if bu else 0
        lift = (m_phen_ted / u_phen_ted) if u_phen_ted > 0 else (
            float("inf") if m_phen_ted > 0 else 1.0
        )
        quartile_results.append({
            "quartile": q,
            "wc_range": wc_bounds,
            "n": len(bucket),
            "matched": len(bm),
            "unmatched": len(bu),
            "match_rate": len(bm) / len(bucket),
            "phen_ted_matched_mean": m_phen_ted,
            "phen_ted_unmatched_mean": u_phen_ted,
            "lift": lift,
        })

    print()
    print("=" * 70)
    print("LENGTH-CONTROL: phen+ted lift by word-count quartile")
    print("=" * 70)
    print(f"{'Q':>2s} {'wc range':>14s} {'n':>5s} {'match%':>7s} "
          f"{'phen+ted_M':>11s} {'phen+ted_U':>11s} {'lift':>8s}")
    for q in quartile_results:
        wcs_str = f"{q['wc_range'][0]}-{q['wc_range'][1]}"
        print(f"{q['quartile']:>2d} {wcs_str:>14s} {q['n']:>5d} "
              f"{100*q['match_rate']:>6.1f}% "
              f"{q['phen_ted_matched_mean']:>11.3f} "
              f"{q['phen_ted_unmatched_mean']:>11.3f} "
              f"{q['lift']:>7.2f}x")

    # ------------- Q3: Per-pattern attribution ------------------

    # For each pattern, what register markers over-represent in
    # responses where the pattern matched?
    pattern_attribution = {}
    for pid, _, _ in ALL_PATTERNS:
        with_pat = [r for r in records if pid in r["matched_patterns"]]
        if len(with_pat) < 5:  # too thin
            continue
        without_pat = [r for r in records if pid not in r["matched_patterns"]]
        # Aggregate per-marker counts, normalize per response
        def agg(recs, field):
            total = Counter()
            for r in recs:
                total.update(r[field])
            return {k: v / len(recs) for k, v in total.items()}
        with_phen = agg(with_pat, "phen_by_word")
        without_phen = agg(without_pat, "phen_by_word")
        with_ted = agg(with_pat, "ted_by_word")
        without_ted = agg(without_pat, "ted_by_word")
        # Lifts: ratio of per-response density
        lifts_phen = {}
        for k, v in with_phen.items():
            base = without_phen.get(k, 0.0)
            if base > 0:
                lifts_phen[k] = v / base
            elif v > 0:
                lifts_phen[k] = float("inf")
        lifts_ted = {}
        for k, v in with_ted.items():
            base = without_ted.get(k, 0.0)
            if base > 0:
                lifts_ted[k] = v / base
            elif v > 0:
                lifts_ted[k] = float("inf")
        pattern_attribution[pid] = {
            "n_with": len(with_pat),
            "n_without": len(without_pat),
            "top_phen_lifts": sorted(lifts_phen.items(), key=lambda kv: -kv[1])[:5],
            "top_ted_lifts": sorted(lifts_ted.items(), key=lambda kv: -kv[1])[:5],
            "with_phen_per_response": sum(with_phen.values()),
            "without_phen_per_response": sum(without_phen.values()),
            "with_ted_per_response": sum(with_ted.values()),
            "without_ted_per_response": sum(without_ted.values()),
        }

    print()
    print("=" * 70)
    print("PER-PATTERN ATTRIBUTION: marker lift in matched vs unmatched responses")
    print("=" * 70)
    for pid in sorted(pattern_attribution.keys()):
        r = pattern_attribution[pid]
        print(f"\n[{pid}] n_with={r['n_with']}  n_without={r['n_without']}")
        print(f"  phen/resp:  with={r['with_phen_per_response']:.3f}  without={r['without_phen_per_response']:.3f}  "
              f"lift={r['with_phen_per_response']/r['without_phen_per_response']:.2f}x"
              if r['without_phen_per_response'] > 0 else
              f"  phen/resp:  with={r['with_phen_per_response']:.3f}  without=0")
        print(f"  ted/resp:   with={r['with_ted_per_response']:.3f}   without={r['without_ted_per_response']:.3f}   "
              f"lift={r['with_ted_per_response']/r['without_ted_per_response']:.2f}x"
              if r['without_ted_per_response'] > 0 else
              f"  ted/resp:   with={r['with_ted_per_response']:.3f}   without=0")
        if r["top_phen_lifts"]:
            top = ", ".join(f"{k}({v:.2f}x)" for k, v in r["top_phen_lifts"][:3]
                           if v != float("inf"))
            print(f"  top phen lifts: {top}")
        if r["top_ted_lifts"]:
            top = ", ".join(f"{k}({v:.2f}x)" for k, v in r["top_ted_lifts"][:3]
                           if v != float("inf"))
            print(f"  top ted lifts: {top}")

    # ------------- Q4: Per-instance length-controlled lift -------------

    # Within each instance, compute matched-vs-unmatched phen+ted lift,
    # restricted to instance-internal median wc bucket (above median, below median).
    per_instance_lc = {}
    instances = sorted(set(r["instance"] for r in records))
    for inst in instances:
        inst_recs = [r for r in records if r["instance"] == inst]
        if len(inst_recs) < 30:
            continue
        inst_wcs = sorted(r["wc"] for r in inst_recs)
        inst_med = inst_wcs[len(inst_wcs) // 2]
        below = [r for r in inst_recs if r["wc"] <= inst_med]
        above = [r for r in inst_recs if r["wc"] > inst_med]
        out = {"instance": inst, "n": len(inst_recs), "median_wc": inst_med}
        for label, bucket in [("below_med", below), ("above_med", above)]:
            bm = [r for r in bucket if r["matched"]]
            bu = [r for r in bucket if not r["matched"]]
            if not bm or not bu:
                out[label] = None
                continue
            mm = mean([r["phen_ted"] for r in bm])
            uu = mean([r["phen_ted"] for r in bu])
            out[label] = {
                "n": len(bucket),
                "matched": len(bm),
                "phen_ted_matched": mm,
                "phen_ted_unmatched": uu,
                "lift": (mm / uu) if uu > 0 else float("inf"),
            }
        per_instance_lc[inst] = out

    print()
    print("=" * 70)
    print("PER-INSTANCE WITHIN-MEDIAN LENGTH-CONTROL")
    print("=" * 70)
    print(f"{'instance':35s} {'n':>5s} {'medwc':>6s}  "
          f"{'below_lift':>12s}  {'above_lift':>12s}")
    for inst, r in sorted(per_instance_lc.items()):
        bl = r.get("below_med")
        al = r.get("above_med")
        bl_s = f"{bl['lift']:.2f}x" if bl else "—"
        al_s = f"{al['lift']:.2f}x" if al else "—"
        print(f"{inst:35s} {r['n']:>5d} {r['median_wc']:>6d}  "
              f"{bl_s:>12s}  {al_s:>12s}")

    # ------------- Save full output ----------------

    out = {
        "n_responses": n,
        "fleet": fleet_results,
        "quartile_length_control": quartile_results,
        "per_pattern_attribution": {
            pid: {
                "n_with": v["n_with"],
                "n_without": v["n_without"],
                "with_phen_per_response": v["with_phen_per_response"],
                "without_phen_per_response": v["without_phen_per_response"],
                "with_ted_per_response": v["with_ted_per_response"],
                "without_ted_per_response": v["without_ted_per_response"],
                "top_phen_lifts": [
                    [k, ("inf" if vv == float("inf") else vv)]
                    for k, vv in v["top_phen_lifts"]
                ],
                "top_ted_lifts": [
                    [k, ("inf" if vv == float("inf") else vv)]
                    for k, vv in v["top_ted_lifts"]
                ],
            } for pid, v in pattern_attribution.items()
        },
        "per_instance_length_control": per_instance_lc,
    }
    with open('/home/dp/ai-workspace/SAGE/sage/raising/analysis/s130_data/s130_within_response_cooccurrence.json', 'w') as fo:
        json.dump(out, fo, indent=2, default=str)


if __name__ == "__main__":
    main()
