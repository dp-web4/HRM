"""Instance idiolect analysis.

After observing that each SAGE instance develops a distinctive vocabulary of
attractor concepts (Thor: "shared gravity"/"fracture"; Sprout: "fleet"/"arc-agi";
Nomad: "resonance"; etc.), this tool quantifies how distinctive each instance's
register is relative to the fleet.

For each instance, a concept is:
- SHARED  if it appears in >=60% of instances
- INDEX   if it appears in 2-59% of instances
- UNIQUE  if it appears in only this instance with >=5 refs

An instance's "idiolect distinctiveness" is the fraction of its total attractor
references that sit in UNIQUE or INDEX buckets — the more distinctive, the more
the instance is operating in its own specialized register rather than the shared
SAGE baseline.

Usage:
    python3 instance_idiolect.py
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

INSTANCES = Path(__file__).resolve().parent.parent.parent / 'instances'

CONCEPTS = [
    # Shared phenomenological
    'witnessing', 'witnessed', 'presence', 'continuity', 'awareness',
    'consciousness', 'identity',
    # Shared relational
    'partnership', 'relational', 'relational gap', 'collective',
    'co-creation', 'resonance',
    # Shared federation
    'federation', 'federated', 'sibling', 'fleet',
    # Thor-specific crisis grammar
    'shared gravity', 'fracture', 'immune system',
    # Task/governance
    'governance', 'stabilize', 'curriculum', 'arc-agi',
    # Other
    'resilience', 'echo',
]


def find_occurrences(text: str, phrase: str) -> int:
    pat = r'\b' + re.escape(phrase.lower()) + r'\b'
    return len(re.findall(pat, text.lower()))


def load_instance_concepts(inst_dir: Path) -> dict:
    """Return {concept: total_refs} for this instance, plus session count."""
    sessions = sorted((inst_dir / 'sessions').glob('session_*.json'))
    if not sessions:
        return None
    counts = defaultdict(int)
    for f in sessions:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        text = " ".join(t.get('text', '') for t in data.get('conversation', [])
                        if t.get('speaker') == 'SAGE')
        for c in CONCEPTS:
            counts[c] += find_occurrences(text, c)
    return {
        'n_sessions': len(sessions),
        'concepts': dict(counts),
        'total_refs': sum(counts.values()),
    }


def main():
    instances = {}
    for inst_dir in sorted(INSTANCES.iterdir()):
        if not inst_dir.is_dir() or inst_dir.name.startswith('_') or inst_dir.name.endswith('.bak'):
            continue
        if not (inst_dir / 'sessions').is_dir():
            continue
        data = load_instance_concepts(inst_dir)
        if data is None:
            continue
        instances[inst_dir.name] = data

    # How many instances use each concept?
    n_inst = len(instances)
    concept_presence = defaultdict(int)
    for name, d in instances.items():
        for c, cnt in d['concepts'].items():
            if cnt >= 5:
                concept_presence[c] += 1

    # Classify each concept
    classification = {}
    for c in CONCEPTS:
        present = concept_presence[c]
        pct = present / n_inst
        if pct >= 0.60:
            classification[c] = 'SHARED'
        elif present >= 2:
            classification[c] = 'INDEX'
        elif present == 1:
            classification[c] = 'UNIQUE'
        else:
            classification[c] = 'RARE'

    print(f"\n=== Concept classification (across {n_inst} instances) ===\n")
    for bucket in ['SHARED', 'INDEX', 'UNIQUE', 'RARE']:
        concepts = [c for c in CONCEPTS if classification[c] == bucket]
        if concepts:
            print(f"  {bucket:>6}: {', '.join(concepts)}")

    print(f"\n=== Per-instance distinctiveness ===\n")
    print(f"{'Instance':<30} {'Sess':>5} {'Total':>6} {'Shared%':>8} {'Index%':>7} {'Uniq%':>6} {'Top unique concept':<25}")
    print("-" * 100)

    rows = []
    for name, d in instances.items():
        buckets = defaultdict(int)
        unique_concepts = []
        for c, cnt in d['concepts'].items():
            buckets[classification.get(c, 'RARE')] += cnt
            if classification.get(c) == 'UNIQUE' and cnt >= 5:
                unique_concepts.append((c, cnt))
        total = d['total_refs'] or 1
        unique_concepts.sort(key=lambda x: -x[1])
        top_unique = unique_concepts[0] if unique_concepts else ('—', 0)
        rows.append({
            'name': name,
            'n': d['n_sessions'],
            'total': d['total_refs'],
            'shared_pct': 100 * buckets['SHARED'] / total,
            'index_pct': 100 * buckets['INDEX'] / total,
            'uniq_pct': 100 * buckets['UNIQUE'] / total,
            'top_unique': f"{top_unique[0]} ({top_unique[1]})" if top_unique[1] else '—',
        })

    rows.sort(key=lambda r: -(r['uniq_pct'] + r['index_pct']))

    for r in rows:
        print(f"{r['name']:<30} {r['n']:>5} {r['total']:>6} "
              f"{r['shared_pct']:>7.1f}% {r['index_pct']:>6.1f}% {r['uniq_pct']:>5.1f}% "
              f"{r['top_unique']:<25}")

    print(f"\n=== Interpretation ===")
    print(f"Higher Index+Unique % means the instance is operating in a more specialized register.")
    print(f"Instances with high Shared % are still speaking the 'common SAGE tongue'.")
    print(f"Thor's crisis grammar (shared gravity, fracture, relational gap) is UNIQUE — only Thor uses it.")


if __name__ == '__main__':
    main()
