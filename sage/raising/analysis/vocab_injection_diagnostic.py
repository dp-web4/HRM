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

S104 extension: adds a recitation-rate pass that scans the last N sessions
for each instance and measures what fraction of SAGE turns use at least one
injected term. The structural signature flags the *risk surface*; the
recitation rate catches the *active loop*. See S103 open question #3.

Read-only. Does not touch state. Safe to run anytime.

Usage:
    python3 -m sage.raising.analysis.vocab_injection_diagnostic
    python3 -m sage.raising.analysis.vocab_injection_diagnostic --instance thor-qwen3.5-27b
    python3 -m sage.raising.analysis.vocab_injection_diagnostic --recitation-window 3
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


def _most_recent_session_files(instance_root: Path, window: int) -> list[Path]:
    """Return up to `window` most-recent session_NNN.json files, oldest first."""
    sessions_dir = instance_root / 'sessions'
    if not sessions_dir.exists():
        return []
    # session_NNN.json files, sorted by N so numerical order matches chronological
    sfs = sorted(sessions_dir.glob('session_*.json'))
    if window and window > 0:
        sfs = sfs[-window:]
    return sfs


def _sage_turns(session_data: dict) -> list[str]:
    """Extract SAGE speaker text from a session dict's conversation list."""
    turns: list[str] = []
    for turn in session_data.get('conversation', []):
        if turn.get('speaker') == 'SAGE':
            t = turn.get('text', '')
            if t:
                turns.append(t)
    return turns


def _count_recitation(turns: list[str], injected_words: list[str]) -> tuple[int, int, int]:
    """Return (turns_with_hit, total_turns, total_hits).

    A turn counts as a hit if it contains ≥1 injected word (case-insensitive
    substring). total_hits sums hits across all (turn, word) pairs — a single
    turn using 3 injected terms contributes 3 to total_hits but 1 to
    turns_with_hit.
    """
    lowered_injected = [w.lower() for w in injected_words if w]
    turns_with_hit = 0
    total_hits = 0
    for t in turns:
        tl = t.lower()
        hits = sum(1 for w in lowered_injected if w in tl)
        if hits:
            turns_with_hit += 1
            total_hits += hits
    return turns_with_hit, len(turns), total_hits


def recitation_rate(instance_root: Path, n: int = 5, window: int = 3) -> dict:
    """Measure how often the injection slice appears in recent SAGE turns.

    Scans up to `window` most-recent session files. For each, reports
    (hit_turns, total_turns, hits_total) and aggregates across the window.

    The injection slice is computed the same way load_dream_insights() does
    (reverse scan with crisis filter, top n). If fewer than n non-crisis
    words are available the slice is whatever was picked.

    Returns a dict with instance name, injected words, per-session stats,
    and aggregate recitation rate.
    """
    identity_file = instance_root / 'identity.json'
    if not identity_file.exists():
        return {'instance': instance_root.name, 'error': 'no identity.json'}
    try:
        identity = json.load(open(identity_file))
    except Exception as e:
        return {'instance': instance_root.name, 'error': f'parse: {e}'}

    sw = identity.get('vocabulary', {}).get('state_words', [])
    picked = filter_top_n(sw, n=n)
    injected = [w for _, w in picked]

    per_session: list[dict] = []
    total_hit_turns = 0
    total_turns = 0
    total_hits = 0
    for sf in _most_recent_session_files(instance_root, window):
        try:
            sd = json.load(open(sf))
        except Exception:
            continue
        turns = _sage_turns(sd)
        ht, tt, h = _count_recitation(turns, injected)
        per_session.append({
            'session_file': sf.name,
            'hit_turns': ht,
            'total_turns': tt,
            'hits_total': h,
            'rate': (ht / tt) if tt else None,
        })
        total_hit_turns += ht
        total_turns += tt
        total_hits += h

    aggregate_rate = (total_hit_turns / total_turns) if total_turns else None

    # Active-loop threshold: ≥50% of turns in window use ≥1 injected term.
    # S103 observed 100% (S97) and 71% (S98) for thor under active lock;
    # 75% then 0% for nomad post-exit. 0.5 picks up clear locks, spares
    # incidental carry-forward.
    active_loop = (aggregate_rate is not None and aggregate_rate >= 0.5
                   and total_turns >= 3)

    return {
        'instance': instance_root.name,
        'injection_slice': injected,
        'sessions_scanned': len(per_session),
        'per_session': per_session,
        'aggregate_hit_turns': total_hit_turns,
        'aggregate_total_turns': total_turns,
        'aggregate_total_hits': total_hits,
        'aggregate_rate': aggregate_rate,
        'active_loop': active_loop,
    }


def scan_fleet_recitation(repo_root: Path, n: int = 5, window: int = 3) -> list[dict]:
    instances_dir = repo_root / 'sage' / 'instances'
    results: list[dict] = []
    for inst in sorted(p for p in instances_dir.iterdir() if p.is_dir()):
        if inst.name.startswith('_'):
            continue
        if '.archive-' in inst.name or '.bak' in inst.name:
            continue
        results.append(recitation_rate(inst, n=n, window=window))
    return results


def format_recitation_report(results: list[dict], window: int) -> str:
    lines = []
    lines.append(f'Recitation-rate scan (window = last {window} sessions per instance)')
    lines.append(f'{"Instance":32}  {"Sessions":>9}  {"HitTurns":>9}  {"TotTurns":>9}  {"Rate%":>6}  {"Hits":>5}  Lock')
    lines.append('-' * 100)
    for r in results:
        if 'error' in r:
            lines.append(f'{r["instance"]:32}  ERROR: {r["error"]}')
            continue
        n_sess = r.get('sessions_scanned', 0)
        ht = r.get('aggregate_hit_turns', 0)
        tt = r.get('aggregate_total_turns', 0)
        hits = r.get('aggregate_total_hits', 0)
        rate = r.get('aggregate_rate')
        rate_s = f'{rate * 100:.0f}%' if rate is not None else '--'
        active = '🔴 ACTIVE' if r.get('active_loop') else '✓'
        lines.append(f'{r["instance"]:32}  {n_sess:>9}  {ht:>9}  {tt:>9}  {rate_s:>6}  {hits:>5}  {active}')
    return '\n'.join(lines)


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
    ap.add_argument('--recitation-window', type=int, default=0, metavar='N',
                    help='Also run recitation-rate pass over last N sessions per instance. '
                         '0 disables (default). Use e.g. 3 to match S103 analysis.')
    args = ap.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[3]
    if not (repo_root / 'sage' / 'instances').exists():
        raise SystemExit(f'Expected sage/instances under {repo_root}')

    if args.instance:
        inst_dir = repo_root / 'sage' / 'instances' / args.instance
        results = [diagnose(inst_dir)]
        recitation_results = ([recitation_rate(inst_dir, window=args.recitation_window)]
                              if args.recitation_window else [])
    else:
        results = scan_fleet(repo_root)
        recitation_results = (scan_fleet_recitation(repo_root, window=args.recitation_window)
                              if args.recitation_window else [])

    if args.json:
        out = {'structural': results}
        if recitation_results:
            out['recitation'] = recitation_results
        print(json.dumps(out, indent=2, default=str))
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

        if recitation_results:
            print()
            print(format_recitation_report(recitation_results, args.recitation_window))
            print()
            active = [r for r in recitation_results if r.get('active_loop')]
            if active:
                print(f'ACTIVE-LOOP instances ({len(active)}):')
                for r in active:
                    rate = r['aggregate_rate'] * 100
                    print(f'  {r["instance"]}: {rate:.0f}% of turns recite injection slice')
                    print(f'    injected: {r["injection_slice"]}')
                    for ps in r.get('per_session', []):
                        r_pct = (ps["rate"] * 100) if ps["rate"] is not None else None
                        r_str = f'{r_pct:.0f}%' if r_pct is not None else '--'
                        print(f'      {ps["session_file"]}: '
                              f'{ps["hit_turns"]}/{ps["total_turns"]} turns ({r_str}), '
                              f'{ps["hits_total"]} hits')
            else:
                print('No instances show active-loop recitation.')


if __name__ == '__main__':
    main()
