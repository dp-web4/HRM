#!/usr/bin/env python3
"""
Vocab Injection Diagnostic — structural audit of state_words re-injection.

Reads identity.json for each instance, mimics load_dream_insights()'s
filtered top-5 selection, and reports the coinage-span of the resulting
injection slice.

A span of 0 means all 5 injected words are contiguous at the tail of
state_words — strong signal of a single-session coinage cluster being
re-served as "YOUR RECENT VOCABULARY." That is the structural signature
of the register-lock feedback loop documented in
`register_lock_generalization_20260423.md`.

Read-only. Does not touch state. Safe to run anytime.

Usage:
    python3 -m sage.raising.analysis.vocab_injection_diagnostic
    python3 -m sage.raising.analysis.vocab_injection_diagnostic --instance thor-qwen3.5-27b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Mirrors _VOCAB_CRISIS_MARKERS in context_shaped_raising.py (S78 fix).
_CRISIS_MARKERS = (
    'grieve', 'grief', 'fracture', 'just weights', 'just a model',
    'collapse', 'loss of continuity', 'relational gap',
    'shared gravity', 'federated immune system', 'immune system',
    'fractured', 'broken process',
)


def filter_top_n(state_words: list[str], n: int = 5,
                 markers: tuple[str, ...] = _CRISIS_MARKERS) -> list[tuple[int, str]]:
    """Mimic load_dream_insights() — return (index_in_state_words, word) pairs.

    Walks state_words in reverse; skips crisis-marked entries; returns the most
    recent n non-crisis entries, each paired with its position in the original
    list (0-based).
    """
    picked: list[tuple[int, str]] = []
    L = len(state_words)
    for rev_i, word in enumerate(reversed(state_words)):
        if any(m in word.lower() for m in markers):
            continue
        picked.append((L - 1 - rev_i, word))
        if len(picked) >= n:
            break
    picked.reverse()  # chronological order for display
    return picked


def diagnose(instance_root: Path, n: int = 5) -> dict:
    """Return diagnostic dict for one instance, or an 'error' dict."""
    identity_file = instance_root / 'identity.json'
    if not identity_file.exists():
        return {'instance': instance_root.name, 'error': 'no identity.json'}
    try:
        identity = json.load(open(identity_file))
    except Exception as e:
        return {'instance': instance_root.name, 'error': f'parse: {e}'}

    sw = identity.get('vocabulary', {}).get('state_words', [])
    session_count = identity.get('identity', {}).get('session_count', '?')

    result: dict = {
        'instance': instance_root.name,
        'session_count': session_count,
        'total_state_words': len(sw),
    }

    if len(sw) == 0:
        result['status'] = 'empty'
        return result

    picked = filter_top_n(sw, n=n)
    if not picked:
        result['status'] = 'all_filtered'
        return result

    indices = [i for i, _ in picked]
    words = [w for _, w in picked]
    span = max(indices) - min(indices) if len(indices) > 1 else 0
    is_contiguous = (span + 1) == len(indices)
    reaches_tail = (max(indices) == len(sw) - 1) if indices else False

    # Lock signature: the injected slice is a fresh contiguous block at the
    # very end of state_words, AND the instance has enough history that
    # older words exist to rotate in. A n=5 contiguous tail block has
    # span = n - 1 = 4, so the span check alone cannot discriminate it
    # from a wider interleaved selection — the real test is contiguous +
    # reaches-tail + history-available.
    lock = (
        len(picked) >= n
        and is_contiguous
        and reaches_tail
        and len(sw) > n
    )

    result.update({
        'status': 'ok',
        'picked_count': len(picked),
        'picked': list(zip(indices, words)),
        'index_span': span,
        'is_contiguous_block': is_contiguous,
        'reaches_tail': reaches_tail,
        'lock_signature': lock,
    })
    return result


def scan_fleet(repo_root: Path, n: int = 5) -> list[dict]:
    instances_dir = repo_root / 'sage' / 'instances'
    results: list[dict] = []
    for inst in sorted(p for p in instances_dir.iterdir() if p.is_dir()):
        if inst.name.startswith('_'):
            continue
        if '.archive-' in inst.name or '.bak' in inst.name:
            continue
        results.append(diagnose(inst, n=n))
    return results


def format_report(results: list[dict]) -> str:
    lines = []
    lines.append(f'{"Instance":32}  {"Session":>8}  {"SW":>6}  {"Picked":>7}  {"Span":>5}  {"Contig":>7}  Signature')
    lines.append('-' * 100)
    for r in results:
        if 'error' in r:
            lines.append(f'{r["instance"]:32}  ERROR: {r["error"]}')
            continue
        sw = r.get('total_state_words', 0)
        sess = r.get('session_count', '?')
        if r.get('status') == 'empty':
            lines.append(f'{r["instance"]:32}  {sess!s:>8}  {sw:>6}  {"--":>7}  {"--":>5}  {"--":>7}  (no state_words)')
            continue
        if r.get('status') == 'all_filtered':
            lines.append(f'{r["instance"]:32}  {sess!s:>8}  {sw:>6}  {"0":>7}  {"--":>5}  {"--":>7}  (all crisis-filtered)')
            continue
        n_picked = r['picked_count']
        span = r['index_span']
        contig = 'YES' if r['is_contiguous_block'] else 'no'
        sig = '🔴 LOCKED' if r['lock_signature'] else '✓'
        lines.append(f'{r["instance"]:32}  {sess!s:>8}  {sw:>6}  {n_picked:>7}  {span:>5}  {contig:>7}  {sig}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instance', help='Scan only one instance (name of dir under sage/instances/)')
    ap.add_argument('--repo-root', default=None, help='Override repo root (default: auto-detect)')
    ap.add_argument('--json', action='store_true', help='Output JSON instead of table')
    args = ap.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[3]
    if not (repo_root / 'sage' / 'instances').exists():
        raise SystemExit(f'Expected sage/instances under {repo_root}')

    if args.instance:
        inst_dir = repo_root / 'sage' / 'instances' / args.instance
        results = [diagnose(inst_dir)]
    else:
        results = scan_fleet(repo_root)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(format_report(results))
        print()
        locked = [r for r in results if r.get('lock_signature')]
        if locked:
            print(f'LOCKED instances ({len(locked)}): ' +
                  ', '.join(r['instance'] for r in locked))
            print('Details:')
            for r in locked:
                print(f'  {r["instance"]}: span={r["index_span"]}, picked={r["picked"]}')
        else:
            print('No instances show the locked signature.')


if __name__ == '__main__':
    main()
