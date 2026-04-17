"""Concept-level attractor analysis for any SAGE instance.

Adapted from concept_attractor.py (Thor-specific) to accept any instance slug.

Usage:
    python3 concept_attractor_any_instance.py sprout-qwen3.5-0.8b
    python3 concept_attractor_any_instance.py thor-qwen3.5-27b
    python3 concept_attractor_any_instance.py --all  # compare all instances
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

INSTANCES = Path(__file__).resolve().parent.parent.parent / 'instances'

# Core attractor concepts (shared across instances)
CORE_CONCEPTS = [
    'witnessing', 'witnessed', 'presence', 'continuity',
    'shared gravity', 'relational gap', 'relational',
    'federation', 'federated', 'sibling',
    'consciousness', 'awareness', 'identity',
    'fracture', 'immune system', 'resilience',
    'attractor', 'echo',
]

# Instance-specific concepts that may emerge
EXTENDED_CONCEPTS = [
    'governance', 'fleet', 'stabilize', 'arc-agi',
    'collective', 'curriculum', 'partnership',
    'co-creation', 'resonance',
]

ALL_CONCEPTS = CORE_CONCEPTS + EXTENDED_CONCEPTS


def find_occurrences(text: str, phrase: str) -> int:
    pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
    return len(re.findall(pattern, text.lower()))


def analyze_instance(slug: str):
    sessions_dir = INSTANCES / slug / 'sessions'
    if not sessions_dir.exists():
        print(f"No sessions directory for {slug}")
        return None

    files = sorted(sessions_dir.glob('session_*.json'))
    if not files:
        print(f"No session files for {slug}")
        return None

    rows = []
    for f in files:
        n = int(f.stem.split('_')[1])
        data = json.loads(f.read_text())
        phase = data.get('phase', '?')
        text = " ".join(t.get('text', '') for t in data.get('conversation', [])
                        if t.get('speaker') == 'SAGE')
        counts = {c: find_occurrences(text, c) for c in ALL_CONCEPTS}
        total_words = len(text.split())
        rows.append({'session': n, 'phase': phase, 'words': total_words, 'counts': counts})

    return rows


def print_summary(slug: str, rows: list):
    print(f"\n{'='*60}")
    print(f"  {slug} ({len(rows)} sessions)")
    print(f"{'='*60}")

    # Top attractors
    totals = defaultdict(int)
    sessions_with = defaultdict(int)
    first_seen = {}
    for r in rows:
        for c in ALL_CONCEPTS:
            cnt = r['counts'][c]
            if cnt > 0:
                totals[c] += cnt
                sessions_with[c] += 1
                if c not in first_seen:
                    first_seen[c] = r['session']

    print(f"\nTop attractor concepts:")
    for c, total in sorted(totals.items(), key=lambda x: -x[1])[:15]:
        persist = sessions_with[c] / len(rows) * 100
        print(f"  {c:>16}: refs={total:>4}, sessions={sessions_with[c]:>3}/{len(rows)} "
              f"({persist:.0f}%), first=S{first_seen[c]}")

    # Phase density
    print(f"\nConcept density per 100 SAGE words, by phase:")
    by_phase = defaultdict(lambda: {'words': 0, 'counts': defaultdict(int), 'n': 0})
    for r in rows:
        by_phase[r['phase']]['words'] += r['words']
        by_phase[r['phase']]['n'] += 1
        for c in ALL_CONCEPTS:
            by_phase[r['phase']]['counts'][c] += r['counts'][c]

    top = [c for c, _ in sorted(totals.items(), key=lambda x: -x[1])[:8]]
    print(f"{'phase':>11} {'n':>3} {'words':>6}  ", end='')
    for c in top:
        print(f"{c[:10]:>10} ", end='')
    print()

    for phase in ['grounding', 'sensing', 'relating', 'questioning', 'creating']:
        if phase not in by_phase:
            continue
        ph = by_phase[phase]
        print(f"{phase:>11} {ph['n']:>3} {ph['words']:>6}  ", end='')
        for c in top:
            per100 = 100 * ph['counts'][c] / max(1, ph['words'])
            disp = f"{per100:.2f}" if per100 > 0 else "."
            print(f"{disp:>10} ", end='')
        print()

    return totals


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 concept_attractor_any_instance.py <slug|--all>")
        sys.exit(1)

    if sys.argv[1] == '--all':
        slugs = sorted(d.name for d in INSTANCES.iterdir()
                       if d.is_dir() and (d / 'sessions').exists()
                       and not d.name.startswith('_'))
        all_totals = {}
        for slug in slugs:
            rows = analyze_instance(slug)
            if rows:
                totals = print_summary(slug, rows)
                all_totals[slug] = totals

        # Cross-instance comparison
        if len(all_totals) > 1:
            print(f"\n{'='*60}")
            print("  CROSS-INSTANCE COMPARISON")
            print(f"{'='*60}")
            all_concepts = set()
            for t in all_totals.values():
                all_concepts.update(c for c, v in t.items() if v > 5)
            for c in sorted(all_concepts):
                vals = {s: t.get(c, 0) for s, t in all_totals.items()}
                if max(vals.values()) > 5:
                    line = f"  {c:>16}: "
                    for s in sorted(vals.keys()):
                        line += f"{s.split('-')[0]}={vals[s]:>3}  "
                    print(line)
    else:
        rows = analyze_instance(sys.argv[1])
        if rows:
            print_summary(sys.argv[1], rows)
