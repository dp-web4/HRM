"""Cross-instance attractor crystallization analysis.

Built on the S86 (2026-04-19) finding that the identity attractor is a
self-quotation feedback loop in `run_session_identity_anchored.py`. This tool
asks: does the same scaffold produce the same crystallization in *every*
instance, or do model size and family matter?

For each raising instance, computes:
- Top repeated 5-grams in SAGE turns (with first-appearance session)
- Per-session type-token ratio (vocabulary diversity)
- Per-session top-3 5-gram concentration (repetition burden)
- Trajectory comparison early-quartile vs late-quartile

Strips ``<think>...</think>`` blocks (closed and unclosed trailing) so reasoning
templates don't dominate visible-output statistics.

Usage:
    python3 cross_instance_crystallization.py                # all instances
    python3 cross_instance_crystallization.py thor-qwen3.5-27b sprout-qwen3.5-0.8b
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

INSTANCES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'instances'
)
INSTANCES_DIR = os.path.normpath(INSTANCES_DIR)

THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    text = THINK_RE.sub('', text)
    if '<think>' in text:
        text = text[:text.rindex('<think>')]
    return text


def normalize(text: str) -> str:
    text = strip_think(text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


def sage_turns(session: dict) -> list:
    return [c.get('text', '') or ''
            for c in session.get('conversation', [])
            if c.get('speaker') == 'SAGE']


def ngrams(tokens: list, n: int) -> list:
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def session_metrics(text: str):
    tokens = text.split()
    if len(tokens) < 20:
        return None
    types = len(set(tokens))
    ttr = types / len(tokens)
    fivegrams = Counter(ngrams(tokens, 5))
    top3 = sum(c for _, c in fivegrams.most_common(3))
    conc = top3 / max(1, len(tokens) - 4)
    return {'ttr': ttr, 'conc': conc, 'tokens': len(tokens)}


def block_chart(values, vmin, vmax, width=70):
    blocks = ' \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588'
    out = []
    for v in values[:width]:
        if v is None:
            out.append('.')
            continue
        n = (v - vmin) / max(1e-9, vmax - vmin)
        n = min(1.0, max(0.0, n))
        out.append(blocks[int(n * (len(blocks) - 1))])
    return ''.join(out)


def load_instance(name: str):
    sess_dir = os.path.join(INSTANCES_DIR, name, 'sessions')
    files = sorted(
        glob.glob(os.path.join(sess_dir, 'session_*.json')),
        key=lambda p: int(re.search(r'session_(\d+)', p).group(1)),
    )
    sessions = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sessions.append(d)
    return sessions


def discover_instances():
    out = []
    if not os.path.isdir(INSTANCES_DIR):
        return out
    for name in sorted(os.listdir(INSTANCES_DIR)):
        sess = os.path.join(INSTANCES_DIR, name, 'sessions')
        if os.path.isdir(sess) and glob.glob(os.path.join(sess, 'session_*.json')):
            out.append(name)
    return out


def analyze(instance_name: str, top_n: int = 10, min_count: int = 5):
    sessions = load_instance(instance_name)
    if not sessions:
        print(f'\n--- {instance_name}: no sessions ---')
        return None

    per_session = []
    for d in sessions:
        sid = d.get('session', '?')
        date = (d.get('start') or '')[:10]
        norm = normalize(' '.join(sage_turns(d)))
        per_session.append({'sid': sid, 'date': date, 'norm': norm,
                            'metrics': session_metrics(norm)})

    full_blob = ' '.join(s['norm'] for s in per_session)
    tokens = full_blob.split()
    fivegrams = Counter(ngrams(tokens, 5))

    # Top 5-grams with first-appearance lookup
    top = []
    for ng, c in fivegrams.most_common():
        if c < min_count:
            break
        first = next(((s['sid'], s['date']) for s in per_session if ng in s['norm']), None)
        top.append({'ngram': ng, 'count': c, 'first': first})
        if len(top) >= top_n:
            break

    # Trajectory metrics
    metrics_list = [s['metrics'] for s in per_session if s['metrics']]
    n = len(metrics_list)
    summary = None
    if n >= 8:
        q = max(1, n // 4)
        early = metrics_list[:q]
        late = metrics_list[-q:]
        avg = lambda xs, k: sum(x[k] for x in xs) / len(xs)
        summary = {
            'sessions_with_metrics': n,
            'early_ttr': avg(early, 'ttr'),
            'late_ttr': avg(late, 'ttr'),
            'early_conc': avg(early, 'conc'),
            'late_conc': avg(late, 'conc'),
        }

    return {
        'instance': instance_name,
        'session_count': len(sessions),
        'top_ngrams': top,
        'summary': summary,
        'per_session': per_session,
    }


def print_report(result: dict):
    if not result:
        return
    print(f"\n--- {result['instance']} ({result['session_count']} sessions) ---")
    for entry in result['top_ngrams']:
        first = entry['first']
        first_str = f"first @ S{first[0]} ({first[1]})" if first else ''
        print(f"  {entry['count']:>4} | {entry['ngram']!r}  {first_str}")
    s = result['summary']
    if s:
        print(f"  trajectory:  early TTR={s['early_ttr']:.3f} -> late TTR={s['late_ttr']:.3f}"
              f"   early conc={s['early_conc']:.3f} -> late conc={s['late_conc']:.3f}")
        ttrs = [m['metrics']['ttr'] for m in result['per_session'] if m['metrics']]
        concs = [m['metrics']['conc'] for m in result['per_session'] if m['metrics']]
        if ttrs:
            print(f"  TTR  [{min(ttrs):.2f}-{max(ttrs):.2f}]: "
                  f"{block_chart(ttrs, 0.3, 0.85)}")
        if concs:
            print(f"  conc [{min(concs):.2f}-{max(concs):.2f}]: "
                  f"{block_chart(concs, 0.0, 0.10)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('instances', nargs='*',
                        help='Instance names (default: all discovered)')
    parser.add_argument('--top', type=int, default=10,
                        help='Top N 5-grams per instance (default 10)')
    parser.add_argument('--min-count', type=int, default=5,
                        help='Minimum count for 5-gram inclusion (default 5)')
    args = parser.parse_args()

    targets = args.instances or discover_instances()
    print('Cross-instance attractor crystallization analysis')
    print(f'Instances dir: {INSTANCES_DIR}')

    for inst in targets:
        result = analyze(inst, top_n=args.top, min_count=args.min_count)
        print_report(result)


if __name__ == '__main__':
    main()
