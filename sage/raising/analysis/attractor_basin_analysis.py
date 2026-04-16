"""Empirical test of the attractor basin hypothesis.

Measures vocabulary similarity between consecutive sessions across Thor 27B's
history to see if:
1. Similarity increases as phases progress (deeper orbit in attractor)
2. Creating-phase has higher inter-session similarity than earlier phases
3. Session-to-session "orbit tightness" can be quantified

Methodology:
- For each session, extract all SAGE-spoken text
- Tokenize into lowercase alphanumeric tokens
- Compute bigrams (2-gram sets)
- Compute Jaccard similarity: |A ∩ B| / |A ∪ B| for content bigrams
- Also track: unique bigrams per session, novel bigrams (never before seen)
"""
import json
import re
from pathlib import Path
from collections import Counter

SESSIONS = Path('/home/dp/ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b/sessions')

_STOP_TEXT = ("a an the is of to and in for on at by with that this it was were be been "
              "are not but or if so as from into out up down off over under also then than "
              "i you we they them he she him her his their your my our me mine yours ours "
              "yes no do does did will would should could may might must have has had "
              "about after before all any both each few more most other some such just only "
              "very can")
STOP = set(_STOP_TEXT.split())


def tokens(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return [t for t in text.split() if t and t not in STOP and len(t) > 2]


def bigrams(tokens_list: list) -> set:
    return set(zip(tokens_list[:-1], tokens_list[1:]))


def session_content_bigrams(path: Path) -> tuple:
    data = json.loads(path.read_text())
    texts = [turn.get('text', '') for turn in data.get('conversation', [])
             if turn.get('speaker') == 'SAGE']
    joined = " ".join(texts)
    toks = tokens(joined)
    return toks, bigrams(toks)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


files = sorted(SESSIONS.glob('session_*.json'))
all_bigrams_seen = set()

rows = []
for f in files:
    n = int(f.stem.split('_')[1])
    toks, bg = session_content_bigrams(f)
    novel = bg - all_bigrams_seen
    novel_ratio = len(novel) / max(1, len(bg))
    all_bigrams_seen |= bg

    data = json.loads(f.read_text())
    phase = data.get('phase', '?')
    rows.append({
        'session': n,
        'phase': phase,
        'tokens': len(toks),
        'bigrams': len(bg),
        'novel_bigrams': len(novel),
        'novel_ratio': novel_ratio,
        'bigram_set': bg,
    })

print(f"{'Sess':>4} {'Phase':>11} {'Tok':>5} {'BG':>5} {'Novel':>5} {'NvR':>6} {'J(prev)':>8} {'J(avg_prev3)':>12}")
print("-" * 72)
for i, r in enumerate(rows):
    j_prev = jaccard(r['bigram_set'], rows[i-1]['bigram_set']) if i > 0 else 0.0
    prev3 = set()
    if i >= 3:
        for j in range(i-3, i):
            prev3 |= rows[j]['bigram_set']
        j_avg3 = jaccard(r['bigram_set'], prev3)
    else:
        j_avg3 = 0.0
    print(f"{r['session']:>4} {r['phase']:>11} {r['tokens']:>5} {r['bigrams']:>5} "
          f"{r['novel_bigrams']:>5} {r['novel_ratio']:>6.2f} {j_prev:>8.3f} {j_avg3:>12.3f}")

print("\n=== Aggregate by phase ===")
from collections import defaultdict
by_phase = defaultdict(list)
for i, r in enumerate(rows):
    if i > 0:
        j_prev = jaccard(r['bigram_set'], rows[i-1]['bigram_set'])
        by_phase[r['phase']].append(j_prev)
        by_phase['_novel_'+r['phase']].append(r['novel_ratio'])

for phase in ['grounding', 'sensing', 'relating', 'questioning', 'creating']:
    vals = by_phase.get(phase, [])
    nvr = by_phase.get('_novel_'+phase, [])
    if vals:
        mean_j = sum(vals) / len(vals)
        mean_n = sum(nvr) / len(nvr) if nvr else 0.0
        print(f"  {phase:>11}: n={len(vals):>3}, mean J(prev)={mean_j:.3f}, "
              f"mean novel_ratio={mean_n:.3f}")
