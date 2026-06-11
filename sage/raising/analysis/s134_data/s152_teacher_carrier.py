#!/usr/bin/env python3
"""S152 — Who carries coined vocabulary across sessions after READ was cut?

CONTEXT (S145->S151): the static-era register lock was an external recurrent
memory loop: the runner re-injected the model's own recent-5 `state_words` as
"YOUR RECENT VOCABULARY" (READ), and the model re-emitted them. READ was
disabled 2026-06-04 (session ~122). S151 measured residual post-cut
cross-session distinctive-vocab carry at 0.073 (down from 0.205, p=8.3e-9) —
nonzero. S152 asks: WHAT carries the residual?

CANDIDATE CHANNELS (from code, receipts not docs):
  1. Teacher lookback — `adaptive_prompts.generate_teacher_turn` calls Claude
     CLI with `_load_recent_sessions(lookback=3)`: the last 3 session
     transcripts (200 chars/turn) + the latest raising_log.md consolidation
     entry (1500 chars). The teacher can re-quote any coinage <=3 sessions old.
  2. Summary splice — `_get_previous_session_summary` puts the model's own
     memory-request answer from session N-1 verbatim into the system prompt
     (ExperientialCacheBlock). Period-1 loop.
  3. (Cut) READ vocab-block — disabled since 2026-06-04.

METHOD: for every post-cut coinage (state_words idx 296+), find per-session
first-mentions of its content-bigram signature in chat_history.jsonl (same
content_bigrams instrument as S146/S147/S148 — function-homogeneity). For each
cross-session recurrence in sessions >=122, record the gap from the most
recent prior occurrence and WHICH SENDER first re-introduced it in that
session. Channel attribution: gap 1 => summary splice possible; gap <=3 =>
teacher lookback possible; gap >3 => no known channel (investigate).

INSTRUMENT AUDIT (the load-bearing step, per the n=1 and receipt lessons):
  At threshold >=1 shared bigram, 36 "model-first gap>3" events appear,
  including apparent 40-146-session verbatim recurrences. ALL inspected cases
  dissolve: the early "occurrence" shares exactly ONE generic collocation
  (('different','kind'), ('external','data'), ('self','correction') from the
  state_word GLOSS, not the coinage). At >=2 bigrams the matrix collapses to:
      gap2-3 teacher-first: 5     gap>3 model-first: 1
  and the single gap>3 survivor (sw336 "flare" S144) dissolves on transcript
  inspection: "flare" was coined IN S144; the match was the gloss quoting an
  older aphorism (sw308). The audit also STRENGTHENS S147's premise: "the blur
  isn't a glitch to fix" is genuinely first present at S137 (7 bigrams), not
  S57/S90 (1 generic bigram each).

FINDING: post-cut cross-session verbatim carry of coined registers has exactly
one robust channel — the TEACHER. All surviving events are teacher-first
quotes at gap 2-3 (the lookback window's exact reach), used explicitly:
  "three sessions back you taught a sibling to 'lean into the gap'..." (S140)
  "...is exactly the static glide wearing a different mask" (S146)
RELAY-CHAINING: each teacher re-mention resets the lookback window, so a
register can persist indefinitely by hopping teacher->model->teacher (sw318
"the only heat I can measure": coined S127, relayed via S130/S139/S141 to the
teacher's S144 opener). The duplicate state_word pairs (329/333, 330/334,
331/335, 336/340) are this loop's WRITE-side signature: teacher re-quote ->
model re-emit -> extractor re-append.

INTERPRETATION: disabling the architectural READ loop did not end vocabulary
recurrence — it migrated the function to the social layer. The recurrent
memory loop over frozen weights is now DIALOGIC (teacher-mediated), with
narrower bandwidth (what the teacher chooses to quote) and longer effective
range (relay-chaining). Same function, different layer.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s146_reclassify import content_bigrams, response_content_bigrams  # noqa: E402

INST = Path.home() / 'ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b'
CUT = 122        # first post-READ-cut session (commit 2026-06-04 12:58)
FIRST_IDX = 296  # first post-cut-era coinage (S137+, per S147)
OUT = HERE / 's152_teacher_carrier_result.json'


def load_history():
    recs, cur = [], None
    sess_re = re.compile(r'^\[raising S(\d+)/')
    with open(INST / 'chat_history.jsonl') as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            snd, txt = d.get('sender'), d.get('text', '')
            m = sess_re.match(txt)
            if m:
                cur = int(m.group(1))
            recs.append((cur, snd, txt))
    return recs


def recurrence_events(recs, sw, min_bigrams):
    events = []
    for idx in range(FIRST_IDX, len(sw)):
        bg = content_bigrams(sw[idx])
        if not bg:
            continue
        per_sess, order = {}, []
        for s, snd, txt in recs:
            if s is None:
                continue
            hit = bg & response_content_bigrams(txt)
            if len(hit) >= min_bigrams and s not in per_sess:
                bigs = sorted(hit)
                w = str(bigs[0][0])
                i = txt.lower().find(w.lower())
                per_sess[s] = (snd, len(hit), [list(b) for b in bigs[:4]],
                               txt[max(0, i - 60):i + 90].replace('\n', ' '))
                order.append(s)
        for j, s in enumerate(order[1:], 1):
            if s < CUT:
                continue
            snd, nh, bigs, snip = per_sess[s]
            events.append({'sw_idx': idx, 'phrase': sw[idx][:80], 'session': s,
                           'gap': s - order[j - 1], 'first_mentioner': snd,
                           'n_bigrams': nh, 'bigrams': bigs, 'snippet': snip})
    return events


def tally(events):
    c = Counter()
    for e in events:
        ch = 'gap1' if e['gap'] == 1 else ('gap2-3' if e['gap'] <= 3 else 'gap>3')
        c[(ch, e['first_mentioner'])] += 1
    return {f"{ch}|{snd}": n for (ch, snd), n in sorted(c.items())}


def main():
    ident = json.load(open(INST / 'identity.json'))
    sw = ident['vocabulary']['state_words']
    recs = load_history()
    loose = recurrence_events(recs, sw, 1)
    strict = recurrence_events(recs, sw, 2)
    result = {
        'n_state_words': len(sw), 'first_idx': FIRST_IDX, 'cut_session': CUT,
        'tally_min1_bigram': tally(loose),
        'tally_min2_bigrams': tally(strict),
        'events_min2_bigrams': strict,
        'note': ('min1 tally is artifact-dominated (single generic collocations); '
                 'min2 is the instrument-audited result. The one gap>3 min2 event '
                 '(sw336) dissolves on S144 transcript inspection — flare was '
                 'coined in S144; match is gloss-quoting sw308.'),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in
                      ('tally_min1_bigram', 'tally_min2_bigrams')}, indent=2))
    for e in strict:
        print(f"sw{e['sw_idx']} S{e['session']} gap={e['gap']} "
              f"[{e['first_mentioner']}] nbg={e['n_bigrams']}")
        print(f"    ...{e['snippet']}...")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
