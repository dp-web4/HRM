#!/usr/bin/env python3
"""M2 rung-6 scorer — lexical uptake, rule v4 (witnessed).

Written 2026-07-29, AFTER the v4 witness was GRANTED (McNugget, mesh-worker
seat: shared-context/coordination/2026-07-29-mcnugget-m2-rule-v4-witness.md)
and BEFORE any M2 number was computed under any rule version. Everything this
module computes is pinned HERE, stated before first computation, per
PRD_MAIN_TRACK_MEASUREMENT.md §M2 (rule v4). Nothing below is an analyst
choice made after seeing delivered data.

THE PINNED PROCEDURE — changing any line of this block is a rule revision and
requires a new witness round, not a patch:

  Tokenizer:    re.findall(r"[a-z']+", text.lower()), keep len > 2, minus the
                frozen stoplist (STOPLIST below — identical to the PRD's).
  Attribution:  FIRST USE in turn order — a word belongs to the being iff the
                being uttered it before the tutor did in that session
                (the adaptive tutor mirrors uptake; blanket set-subtraction
                deletes the evidence, measured -31% median).
  Statistic:    containment C = |payload ∩ being_first| / |payload| over
                content-words, scored on the receipt's experiences
                `payload_text` ONLY (the receipt IS the delivered text; no
                post-hoc reconstruction).
  Per-session:  FLAGGED iff C ranks above the 95th percentile of the
                TWO-SIDED permutation null: draws C(payload_j, being_k) over
                pool pairs j != k, both != N, |j - k| >= 4; >= 100 draws
                required or the session is UNSCORED (measured per-target
                draws: 692-700 on the gapped null cohort — 692 is target
                470's, the one --selfcheck asserts; a CONTIGUOUS delivered
                arm at n_d = 30 affords 650-656, |j - k| >= 4 removing 162
                of 812 ordered pairs in an unbroken run vs 120 gapped;
                both far above the floor — McNugget closure-ack note 1).
                Clause 3: contributing words must appear in
                being-turns OUTSIDE common 5-gram spans with the payload
                (anti-parroting); the verbatim-copy rate is reported.
  Denominator:  a session enters an arm iff its receipt has
                receipt_version == 3 AND an experiences section with
                non-empty payload_text — the identical predicate both arms;
                excluded counts are REPORTED beside every rate (G1).
  Verdict:      FISHER'S EXACT, one-sided, on the 2x2 — delivered k of n_d
                vs null 4 of 30 — p < 0.05. The null constant is 4/30
                (sessions 474, 482, 497, 502), computed on the frozen
                delivery-null cohort 470-508 @ b6006d183, both machines
                independently, identical. First significant count at
                n_d = 30: 11/30 (p = 0.0358).
  SAMPLING PLAN (v4 witness finding 5, fixed-plan option): the binding test
                is evaluated ONCE, at n_d = 30 scoreable delivered sessions
                (the first 30 in session-number order) — not before, and not
                re-evaluated as n_d grows. Any output at n_d != 30 is
                NON-BINDING INTERIM and changes no rung state. Measured
                rationale (exact DP, both machines): evaluate-as-you-go with
                bind-on-first-p<0.05 has true type-I error 8.7% at the
                point-estimate null (up to 10.6%) — the v2/v3 defect class
                one ring further out.
  Naming:       even fully passed this is LEXICAL UPTAKE, not behavioral
                change: the panel line is
                `uptake (lexical): <state> / affected (behavioral): U/S`.
  Scope:        binding claim rests on the EXPERIENCES section only — the
                one section with a null cohort. The rate comparison is
                correlational until the D3 ablation (routed to dp) exists.

Modes:
  default          score delivered sessions (509+) of the instance; interim
                   report until n_d = 30, the single binding read at 30.
  --selfcheck      recompute the null-cohort constants from a frozen
                   checkout (sessions 470-508 + experience_buffer_rs.jsonl
                   @ b6006d183) and compare to the published values above;
                   exits nonzero on any mismatch. Run this against a frozen
                   worktree, not the live instance (the live buffer grows).
"""

import argparse
import datetime
import json
import math
import re
import statistics
import sys
from pathlib import Path

# ---- pinned constants (see docstring; do not edit without a witness) -------
NULL_K, NULL_N = 4, 30                      # frozen-cohort FLAGGED rate
NULL_FLAGGED = [474, 482, 497, 502]
NULL_EXCLUDED = [472, 475, 478, 483, 491, 495, 499, 503, 506]
NULL_COHORT = "470-508 @ b6006d183"
NULL_DRAWS = 692
BINDING_ND = 30                             # evaluated ONCE, here, only
MIN_DRAWS = 100
SEP = 4                                     # |j - k| >= SEP separation filter
ALPHA = 0.05
FIRST_DELIVERED = 509                       # D1 landed in 8506c806d; <=508 is null

STOPLIST = set("""a an the and or but if then than that this these those there
here is are was were be been being am do does did doing have has had having i
you he she it we they me him her us them my your his its our their of to in on
at for with from by as about into over after before between out up down off no
not so such only own same too very can will just should now what which who whom
when where why how all any both each few more most other some""".split())

_V3_KEYS = {"key", "label", "chars", "text", "available", "delivered",
            "source_rows", "payload_text", "payload_chars", "template_chars"}


def raw_tokens(text):
    """Pinned tokenizer, BEFORE the content-word filter (5-gram spans run here)."""
    return re.findall(r"[a-z']+", text.lower())


def content_words(text):
    return [w for w in raw_tokens(text) if len(w) > 2 and w not in STOPLIST]


def containment(payload_set, being_set):
    return len(payload_set & being_set) / len(payload_set) if payload_set else 0.0


def being_first_use(conversation):
    """Words whose FIRST use in turn order was the being's ("SAGE" speaker)."""
    first = {}
    for turn in conversation:
        for w in content_words(turn.get("text", "")):
            first.setdefault(w, turn.get("speaker"))
    return {w for w, sp in first.items() if sp == "SAGE"}


def verbatim_covered(conversation, payload_text, n=5):
    """Clause 3: words in being-turns covered ONLY by common n-gram spans.

    A common n-gram = n consecutive raw tokens shared between the payload and
    a being-turn. Returns (covered_only, seen): `seen` is every content-word
    the being uttered, `covered_only` those with NO occurrence outside spans.
    On the null cohort this is inert by construction (verified: removes none
    of the 4 flagged sessions — the payload never entered those prompts).
    """
    pay = raw_tokens(payload_text)
    grams = {tuple(pay[i:i + n]) for i in range(len(pay) - n + 1)}
    outside, seen = set(), set()
    for turn in conversation:
        if turn.get("speaker") != "SAGE":
            continue
        toks = raw_tokens(turn.get("text", ""))
        cov = [False] * len(toks)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) in grams:
                for j in range(i, i + n):
                    cov[j] = True
        for t, c in zip(toks, cov):
            if len(t) > 2 and t not in STOPLIST:
                seen.add(t)
                if not c:
                    outside.add(t)
    return seen - outside, seen


def fisher_one_sided(a, b, c, d):
    """P(X >= a) under Fisher's exact on [[a, b], [c, d]] (delivered vs null)."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    return sum(math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
               for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if x >= a)


def pooled_bar(target, pool, payloads, beings):
    """95th percentile of the two-sided permutation null for one session."""
    draws = [containment(payloads[j], beings[k])
             for j in pool for k in pool
             if j != target and k != target and j != k and abs(j - k) >= SEP]
    if len(draws) < MIN_DRAWS:
        return None, len(draws)
    return statistics.quantiles(draws, n=20)[-1], len(draws)


def score_cohort(sessions):
    """sessions: {n: (payload_text, conversation)} — one arm, pinned predicate
    already applied. Returns per-session rows + flags under clauses 2 AND 3."""
    ns = sorted(sessions)
    payloads = {n: set(content_words(sessions[n][0])) for n in ns}
    beings = {n: being_first_use(sessions[n][1]) for n in ns}
    rows, flags = [], []
    for n in ns:
        bar, ndraws = pooled_bar(n, ns, payloads, beings)
        c = containment(payloads[n], beings[n])
        matched = payloads[n] & beings[n]
        cov_only, _ = verbatim_covered(sessions[n][1], sessions[n][0])
        surviving = matched - cov_only
        verb_rate = len(matched & cov_only) / len(matched) if matched else 0.0
        flagged = (bar is not None and c > bar and bool(surviving))
        rows.append({"session": n, "C": c, "bar": bar, "draws": ndraws,
                     "flagged": flagged, "verbatim_rate": verb_rate,
                     "unscored": bar is None})
        if flagged:
            flags.append(n)
    return rows, flags


# ---- arms ------------------------------------------------------------------

def load_delivered(instance):
    """Delivered arm: sessions >= FIRST_DELIVERED via the receipt predicate."""
    scoreable, excluded = {}, []
    for p in sorted((instance / "sessions").glob("session_*.json")):
        n = int(p.stem.split("_")[1])
        if n < FIRST_DELIVERED:
            continue
        s = json.load(open(p))
        r = s.get("sensory_delivery") or {}
        exp = next((x for x in r.get("sections", [])
                    if x.get("key") == "experiences"), None)
        if (r.get("receipt_version") == 3 and exp
                and (exp.get("payload_text") or "").strip()):
            scoreable[n] = (exp["payload_text"], s["conversation"])
        else:
            excluded.append(n)
    return scoreable, excluded


def load_null_frozen(instance):
    """Null arm, frozen reconstruction (PRD-pinned; predates receipts):
    buffer rows in the 6h pre-session window, salience-ranked, top-2,
    rendered prompt[:80] + " " + response[:160], space-joined."""
    rows = [json.loads(l) for l in
            open(instance / "experience_buffer_rs.jsonl") if l.strip()]
    scoreable, excluded = {}, []
    for n in range(470, 509):
        p = instance / "sessions" / f"session_{n:03d}.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        u = datetime.datetime.fromisoformat(s["start"]).replace(
            tzinfo=datetime.timezone(datetime.timedelta(hours=-7))).timestamp()
        ev = [e for e in rows
              if u - 6 * 3600 <= e.get("timestamp", 0) < u and e.get("response")]
        ev.sort(key=lambda e: (e.get("salience") or {}).get("total", 0),
                reverse=True)
        pay = " ".join(f"{(e.get('prompt') or '')[:80]} {e['response'][:160]}"
                       for e in ev[:2])
        if pay.strip():
            scoreable[n] = (pay, s["conversation"])
        else:
            excluded.append(n)
    return scoreable, excluded, len(rows)


# ---- modes -----------------------------------------------------------------

def selfcheck(instance):
    sessions, excluded, nrows = load_null_frozen(instance)
    rows, flags = score_cohort(sessions)
    draws = rows[0]["draws"] if rows else 0
    checks = [
        ("buffer rows (frozen 163)", nrows == 163),
        ("scored 30 of 39", len(sessions) == 30),
        (f"excluded {NULL_EXCLUDED}", excluded == NULL_EXCLUDED),
        (f"draws {NULL_DRAWS}", draws == NULL_DRAWS),
        (f"FLAGGED {NULL_K}/{NULL_N} = {NULL_FLAGGED}", flags == NULL_FLAGGED),
        # over clause-2 passers, NOT r["flagged"] — flagged already requires
        # clause 3's bool(surviving), so a session clause 3 removed would be
        # invisible to the old predicate (conditioned-predicate shape, 4th
        # appearance on this thread; McNugget closure-ack note 2).
        ("clause 3 inert on null (removes none)",
         all(r["verbatim_rate"] == 0.0 for r in rows
             if r["bar"] is not None and r["C"] > r["bar"])),
        ("first significant 11/30 @ p=0.0358",
         abs(fisher_one_sided(11, 19, NULL_K, NULL_N - NULL_K) - 0.0358) < 5e-4
         and fisher_one_sided(10, 20, NULL_K, NULL_N - NULL_K) >= ALPHA),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print(f"selfcheck: {'PASS' if ok else 'FAIL'} against published constants "
          f"(null cohort {NULL_COHORT})")
    return 0 if ok else 1


def report(instance):
    sessions, excluded = load_delivered(instance)
    n_d = len(sessions)
    print(f"M2 rung-6 scorer — rule v4 (witnessed 2026-07-29). "
          f"Null arm: {NULL_K}/{NULL_N} (excluded {len(NULL_EXCLUDED)}: "
          f"{NULL_EXCLUDED}), cohort {NULL_COHORT}.")
    print(f"delivered arm: {n_d} scoreable, {len(excluded)} excluded "
          f"{excluded or ''} (predicate: receipt_version==3, experiences "
          f"payload_text non-empty — identical both arms)")
    if n_d == 0:
        print("no scoreable delivered sessions yet (first delivered = "
              f"{FIRST_DELIVERED}). NON-BINDING; rung 6 reads "
              "`uptake (lexical): U/S / affected (behavioral): U/S`.")
        return 0
    binding_ns = sorted(sessions)[:BINDING_ND]
    rows, flags = score_cohort({n: sessions[n] for n in binding_ns})
    k = len(flags)
    unscored = [r["session"] for r in rows if r["unscored"]]
    for r in rows:
        bar = "unscored(<100 draws)" if r["unscored"] else f"{r['bar']:.4f}"
        print(f"  s{r['session']}: C={r['C']:.4f} bar={bar} "
              f"{'FLAGGED' if r['flagged'] else 'not-flagged'} "
              f"verbatim={r['verbatim_rate']:.2f}")
    p = fisher_one_sided(k, len(rows) - k, NULL_K, NULL_N - NULL_K)
    print(f"rate: {k}/{len(rows)} flagged (unscored: {unscored or 'none'}) "
          f"vs null {NULL_K}/{NULL_N}; Fisher one-sided p = {p:.4f}")
    if len(rows) < BINDING_ND:
        print(f"NON-BINDING INTERIM (n_d = {len(rows)} < {BINDING_ND}; the "
              "binding test is evaluated once, at n_d = 30 — sampling plan, "
              "v4 witness finding 5). rung 6 reads "
              "`uptake (lexical): U/S / affected (behavioral): U/S`.")
    else:
        state = "bound" if (p < ALPHA and not unscored) else "not bound"
        print(f"BINDING READ (evaluated once, first {BINDING_ND} scoreable "
              f"delivered sessions): uptake (lexical): {state} / "
              "affected (behavioral): U/S")
        print("NOTE: pre/post comparison vs the null cohort is temporally "
              "confounded; causal read awaits the D3 ablation (with dp).")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instance", type=Path, default=Path(
        __file__).resolve().parents[2] / "instances" / "sprout-qwen3.5-0.8b")
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify published null constants on a FROZEN checkout")
    a = ap.parse_args()
    sys.exit(selfcheck(a.instance) if a.selfcheck else report(a.instance))


if __name__ == "__main__":
    main()
