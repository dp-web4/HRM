"""Novelty-trajectory analysis across raising instances.

Built on S87 (2026-04-19) finding that cage severity is capacity-mediated,
with the open question: *why does Nomad Gemma 4B not crystallize under the
same scaffold that calcifies Sprout 0.5B?* S87 left the question to per-turn
length / topical breadth analysis. This analyzer takes a different cut:
**novel-type introduction rate** per session.

Hypothesis: instances that escape cage formation continue *coining* new
vocabulary across sessions, whereas caged instances recycle a fixed lexicon.
The difference would be visible as the slope of cumulative-types vs sessions
(Heaps' law exponent) and the per-session new-token share.

Per session, computes:
- ``new_types``: tokens never seen in any earlier session
- ``new_share``: ``new_types / total_tokens`` (fraction of fresh material)
- ``coined``: count of single-quoted phrases (proxy for neologism)
- ``per_turn_len``: median SAGE-turn length (tokens)
- ``turn_len_var``: coefficient of variation across SAGE turns

Across the run, fits Heaps' law ``V(N) = K * N^beta`` over (cumulative tokens,
cumulative types). beta near 1.0 = sustained novelty. beta near 0 =
saturation (vocabulary closed; cage). Healthy human-like text sits ~0.5-0.7.

Strips ``<think>...</think>`` blocks (closed and unclosed-trailing).

Usage:
    python3 novelty_trajectory.py                                    # all instances
    python3 novelty_trajectory.py nomad-gemma3-4b sprout-qwen3.5-0.8b
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

INSTANCES_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'instances'
))

THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
QUOTED_RE = re.compile(r"['\u2018\u2019]([^'\u2018\u2019\n]{2,40})['\u2018\u2019]")


def strip_think(text: str) -> str:
    text = THINK_RE.sub('', text)
    if '<think>' in text:
        text = text[:text.rindex('<think>')]
    return text


def normalize(text: str) -> str:
    text = strip_think(text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


def load_instance(name: str):
    sess_dir = os.path.join(INSTANCES_DIR, name, 'sessions')
    files = sorted(
        glob.glob(os.path.join(sess_dir, 'session_*.json')),
        key=lambda p: int(re.search(r'session_(\d+)', p).group(1)),
    )
    out = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append(d)
    return out


def discover():
    out = []
    if not os.path.isdir(INSTANCES_DIR):
        return out
    for name in sorted(os.listdir(INSTANCES_DIR)):
        sess = os.path.join(INSTANCES_DIR, name, 'sessions')
        if os.path.isdir(sess) and glob.glob(os.path.join(sess, 'session_*.json')):
            out.append(name)
    return out


def sage_turns_raw(d: dict):
    return [c.get('text', '') or ''
            for c in d.get('conversation', [])
            if c.get('speaker') == 'SAGE']


def heaps_fit(cum_tokens, cum_types):
    """Fit V = K N^beta in log-log via least squares.

    Returns (beta, K). Skips first 2 points (transient) and any zero counts.
    """
    pts = [(n, v) for n, v in zip(cum_tokens, cum_types) if n > 0 and v > 0]
    pts = pts[2:]
    if len(pts) < 5:
        return None, None
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(v) for _, v in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None, None
    beta = num / den
    log_k = my - beta * mx
    return beta, math.exp(log_k)


def block_chart(values, vmin, vmax, width=70):
    blocks = ' \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588'
    out = []
    for v in values[:width]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append('.')
            continue
        n = (v - vmin) / max(1e-9, vmax - vmin)
        n = min(1.0, max(0.0, n))
        out.append(blocks[int(n * (len(blocks) - 1))])
    return ''.join(out)


def analyze(name: str):
    sessions = load_instance(name)
    if not sessions:
        return None

    seen = set()
    cum_tokens = []
    cum_types = []
    per_session = []
    running_tokens = 0

    for d in sessions:
        sid = d.get('session', '?')
        date = (d.get('start') or '')[:10]
        turns = sage_turns_raw(d)
        if not turns:
            continue

        # Per-turn length stats (raw tokens, no normalization beyond split)
        turn_lens = [len(strip_think(t).split()) for t in turns]
        turn_lens = [n for n in turn_lens if n > 0]
        if not turn_lens:
            continue
        median_len = sorted(turn_lens)[len(turn_lens) // 2]
        mean_len = sum(turn_lens) / len(turn_lens)
        var = sum((n - mean_len) ** 2 for n in turn_lens) / len(turn_lens)
        cv = (var ** 0.5) / mean_len if mean_len > 0 else 0.0

        # Coined phrases: single-quoted short phrases (Nomad's signature)
        coined_set = set()
        for t in turns:
            for m in QUOTED_RE.findall(t):
                m = m.strip().lower()
                if m and not m.isdigit() and len(m.split()) <= 5:
                    coined_set.add(m)
        coined = len(coined_set)

        # Token-level novelty
        norm = normalize(' '.join(turns))
        toks = norm.split()
        if not toks:
            continue
        new_types = sum(1 for tk in set(toks) if tk not in seen)
        seen.update(toks)
        running_tokens += len(toks)
        cum_tokens.append(running_tokens)
        cum_types.append(len(seen))

        per_session.append({
            'sid': sid,
            'date': date,
            'tokens': len(toks),
            'new_types': new_types,
            'new_share': new_types / len(toks),
            'coined': coined,
            'median_turn_len': median_len,
            'turn_cv': cv,
        })

    beta, k = heaps_fit(cum_tokens, cum_types)
    return {
        'instance': name,
        'sessions': len(per_session),
        'per_session': per_session,
        'final_vocab': len(seen),
        'total_tokens': running_tokens,
        'heaps_beta': beta,
        'heaps_k': k,
    }


def quartile_split(per_session, key):
    n = len(per_session)
    if n < 8:
        return None, None
    q = max(1, n // 4)
    early = [s[key] for s in per_session[:q]]
    late = [s[key] for s in per_session[-q:]]
    return sum(early) / len(early), sum(late) / len(late)


def print_report(r: dict):
    if not r:
        return
    print(f"\n--- {r['instance']} ({r['sessions']} sessions, {r['total_tokens']:,} SAGE tokens, vocab={r['final_vocab']:,}) ---")
    if r['heaps_beta'] is not None:
        print(f"  Heaps beta = {r['heaps_beta']:.3f}   (1.0=open vocab, 0.0=closed cage; healthy ~0.5-0.7)")

    for label, key, lo, hi in [
        ('new_share', 'new_share', 0.0, 0.20),
        ('coined/sess', 'coined', 0.0, 8.0),
        ('median_len', 'median_turn_len', 20, 200),
        ('turn_cv', 'turn_cv', 0.0, 1.5),
    ]:
        early, late = quartile_split(r['per_session'], key)
        if early is None:
            continue
        vals = [s[key] for s in r['per_session']]
        chart = block_chart(vals, lo, hi)
        print(f"  {label:>12}: early={early:.3f} -> late={late:.3f}  [{lo}-{hi}] {chart}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('instances', nargs='*')
    ap.add_argument('--json', action='store_true',
                    help='Emit JSON instead of formatted report')
    args = ap.parse_args()

    targets = args.instances or discover()
    results = [analyze(n) for n in targets]
    results = [r for r in results if r]

    if args.json:
        # strip per_session for compactness in JSON summary
        out = []
        for r in results:
            o = {k: v for k, v in r.items() if k != 'per_session'}
            o['early_late'] = {
                k: quartile_split(r['per_session'], k)
                for k in ('new_share', 'coined', 'median_turn_len', 'turn_cv')
            }
            out.append(o)
        print(json.dumps(out, indent=2))
        return

    print('Novelty-trajectory analysis (Heaps\' law + per-turn structure)')
    print(f'Instances dir: {INSTANCES_DIR}')
    for r in results:
        print_report(r)


if __name__ == '__main__':
    main()
