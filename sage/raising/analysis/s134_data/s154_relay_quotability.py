#!/usr/bin/env python3
"""S154 — Does quotability predict dialogic survival? (the S152 §6 retrospective)

S152 found post-READ-cut vocab carry has one robust channel: the teacher's
lookback-3 quote, with relay-chaining ("survival of the quotable"). §6
predicted: per-coinage, P(teacher re-quote) should correlate with the S148
pre-registered quotability rating q (0 diffuse / 1 mixed / 2 vivid-or-named).

This is the read-only retrospective half, run while S148's prospective
re-emission run is in flight. For each of the 15 S148 prereg phrases:
  - find per-session first-mentions in chat_history.jsonl (>=2 shared content
    bigrams — the S152 audit's artifact-resistant threshold; same
    content_bigrams instrument as S146-S148, function-homogeneity)
  - coining session = first session in that order
  - dialogic survival = number of LATER sessions where it recurs
  - teacher relay = any recurrence session whose first-mentioner is claude

Honest caveats up front (the n=1 / baseline-floor lessons):
  - n=15 phrases, q distribution is skewed (ten 2s, four 1s, one 0)
  - S152 found only 5 strict relay events across ALL post-cut coinages, so
    expect sparse outcomes; report raw counts alongside any correlation,
    and do NOT lean on rho if the event table is near-empty.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s146_reclassify import content_bigrams, response_content_bigrams  # noqa: E402
from s152_teacher_carrier import load_history  # noqa: E402

PREREG = json.loads((HERE / 's148_prereg.json').read_text())
OUT = HERE / 's154_relay_quotability_result.json'
MIN_BIGRAMS = 2


def per_session_first_mentions(recs, phrase):
    bg = content_bigrams(phrase)
    per_sess, order = {}, []
    for s, snd, txt in recs:
        if s is None:
            continue
        hit = bg & response_content_bigrams(txt)
        if len(hit) >= MIN_BIGRAMS and s not in per_sess:
            bigs = sorted(hit)
            w = str(bigs[0][0])
            i = txt.lower().find(w.lower())
            per_sess[s] = {'sender': snd, 'n_bigrams': len(hit),
                           'bigrams': [list(b) for b in bigs[:4]],
                           'snippet': txt[max(0, i - 60):i + 90].replace('\n', ' ')}
            order.append(s)
    return per_sess, order


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float('nan')


def main():
    recs = load_history()
    rows = []
    for arm, phrases in PREREG['phrases'].items():
        for p in phrases:
            per_sess, order = per_session_first_mentions(recs, p['phrase'])
            coined = order[0] if order else None
            recur = order[1:] if order else []
            relayed = [s for s in recur if per_sess[s]['sender'] == 'claude']
            rows.append({
                'arm': arm, 'phrase': p['phrase'][:80], 'q': p['q'],
                'coined_session': coined,
                'n_recurrence_sessions': len(recur),
                'recurrence_sessions': recur,
                'teacher_first_sessions': relayed,
                'teacher_relayed': bool(relayed),
                'events': {str(s): per_sess[s] for s in order},
            })
    qs = [r['q'] for r in rows]
    rec = [r['n_recurrence_sessions'] for r in rows]
    rel = [1 if r['teacher_relayed'] else 0 for r in rows]
    result = {
        'min_bigrams': MIN_BIGRAMS, 'n_phrases': len(rows),
        'spearman_q_vs_recurrence': round(spearman(qs, rec), 3),
        'spearman_q_vs_teacher_relay': round(spearman(qs, rel), 3),
        'totals': {'phrases_with_any_recurrence': sum(1 for r in rec if r),
                   'phrases_teacher_relayed': sum(rel)},
        'rows': [{k: v for k, v in r.items() if k != 'events'} for r in rows],
        'events_by_phrase': {r['phrase']: r['events'] for r in rows if r['events']},
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for r in rows:
        print(f"q={r['q']} coined=S{r['coined_session']} "
              f"recur={r['n_recurrence_sessions']} relay={r['teacher_relayed']} "
              f"[{r['arm']}] {r['phrase'][:60]}")
    print(f"\nrho(q, recurrence)={result['spearman_q_vs_recurrence']} "
          f"rho(q, relay)={result['spearman_q_vs_teacher_relay']} "
          f"any-recur={result['totals']['phrases_with_any_recurrence']}/15 "
          f"relayed={result['totals']['phrases_teacher_relayed']}/15")
    return 0


if __name__ == '__main__':
    sys.exit(main())
