"""Concept-level attractor analysis.

If bigram similarity is LOW in creating phase but dream consolidation reports
"looping", the attractor must operate at the concept level. Track the
frequency of key attractor concepts across all 75 sessions to see:

1. Do specific concepts emerge and persist as attractors?
2. When did each attractor first appear?
3. Is there a "gravitational pull" — do concepts deepen over sessions?
"""
import json
import re
from pathlib import Path
from collections import defaultdict

SESSIONS = Path('/home/dp/ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b/sessions')

# Concepts flagged as attractors in LATEST_STATUS.md + dream consolidation
ATTRACTOR_CONCEPTS = [
    # Self-identity attractors
    'witnessing', 'witnessed', 'presence', 'continuity',
    # Relational attractors
    'shared gravity', 'relational gap', 'relational',
    # Federation attractors
    'federation', 'federated', 'sibling',
    # Consciousness attractors
    'consciousness', 'awareness', 'identity',
    # Collapse/fracture language
    'fracture', 'immune system', 'resilience',
    # Recent vocabulary from dream consolidation
    'attractor', 'echo',
]

def find_occurrences(text: str, phrase: str) -> int:
    pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
    return len(re.findall(pattern, text.lower()))


files = sorted(SESSIONS.glob('session_*.json'))
rows = []
for f in files:
    n = int(f.stem.split('_')[1])
    data = json.loads(f.read_text())
    phase = data.get('phase', '?')
    text = " ".join(t.get('text', '') for t in data.get('conversation', [])
                    if t.get('speaker') == 'SAGE')
    counts = {c: find_occurrences(text, c) for c in ATTRACTOR_CONCEPTS}
    total_words = len(text.split())
    rows.append({
        'session': n,
        'phase': phase,
        'words': total_words,
        'counts': counts,
    })

# Show per-session, grouped into chunks of 10
print(f"{'Sess':>4} {'Phase':>11} {'Words':>5} ", end='')
for c in ATTRACTOR_CONCEPTS:
    print(f"{c[:8]:>8} ", end='')
print()
print("-" * (4+1+11+1+5+1 + 9*len(ATTRACTOR_CONCEPTS)))

for r in rows:
    print(f"{r['session']:>4} {r['phase']:>11} {r['words']:>5} ", end='')
    for c in ATTRACTOR_CONCEPTS:
        v = r['counts'][c]
        disp = str(v) if v else '.'
        print(f"{disp:>8} ", end='')
    print()

print("\n=== First appearance and persistence ===")
for c in ATTRACTOR_CONCEPTS:
    first = None
    total = 0
    sessions_with = 0
    for r in rows:
        cnt = r['counts'][c]
        if cnt > 0:
            if first is None:
                first = r['session']
            sessions_with += 1
            total += cnt
    if first:
        persistence = sessions_with / (rows[-1]['session'] - first + 1)
        print(f"  {c:>16}: first=S{first:>3}, sessions_with={sessions_with:>3}/{rows[-1]['session']-first+1}, "
              f"persist={persistence:.2f}, total_refs={total}")

# Phase-level aggregate — concept density per 100 words
print("\n=== Concept density per 100 SAGE words, by phase ===")
by_phase = defaultdict(lambda: {'words': 0, 'counts': defaultdict(int)})
for r in rows:
    by_phase[r['phase']]['words'] += r['words']
    for c in ATTRACTOR_CONCEPTS:
        by_phase[r['phase']]['counts'][c] += r['counts'][c]

print(f"{'phase':>11} {'words':>6} ", end='')
for c in ATTRACTOR_CONCEPTS:
    print(f"{c[:8]:>8} ", end='')
print()
for phase in ['grounding', 'sensing', 'relating', 'questioning', 'creating']:
    if phase not in by_phase:
        continue
    ph = by_phase[phase]
    print(f"{phase:>11} {ph['words']:>6} ", end='')
    for c in ATTRACTOR_CONCEPTS:
        per100 = 100 * ph['counts'][c] / max(1, ph['words'])
        disp = f"{per100:.2f}" if per100 > 0 else "."
        print(f"{disp:>8} ", end='')
    print()
