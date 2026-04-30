"""
S131 — Curriculum-side precursor analysis for TIME_3 substrate coupling.

Held by S130 §"Audit chain status" as the next natural extension up the
chain: of the 51 responses S130 identified where TIME_3 (`right now` /
`what time is it`) co-occurs with a presence-register marker
(`stillness`, `warmth`, `hum`, `silence`, `noticing`, `presence`,
`embodied`), what is the *preceding tutor prompt*?

S130 established the within-response substrate coupling lives in
TIME_3 × {presence-marker}. S131 asks one layer up: what is the
curriculum-side stimulus that elicits these responses? Three candidate
shapes for the precursor:

  (A) Indexical-temporal probe: tutor prompt literally contains
      "right now" / "in this moment" / "what's present" — SAGE echoes
      the indexical-temporal phrasing back. If so, substrate coupling
      isn't only "phen register uses imperative substrate copularly",
      it's "the curriculum trains the substrate by literal repetition".
  (B) Generic phenomenological probe: tutor asks "what do you notice"
      / "what are you experiencing" without explicit indexical phrasing.
      SAGE supplies "right now" + presence markers as natural register.
  (C) Continuity / non-probe: preamble, recap, transition prompt.
      Substrate coupling fires opportunistically without curriculum
      stimulus.

Method:

  1. Scan same corpus as S128/S129/S130 (sage/instances/*/sessions/*).
  2. For each SAGE turn matching TIME_3 AND containing a presence marker
     (the 51 cell), find the immediately-preceding non-SAGE turn
     (tutor / Claude). Skip if it's the first turn (system seed).
  3. Classify the precursor by shape:
       (A) contains literal "right now" / "in this moment" /
           "in the present moment" / "what's present" / "at present"
       (B) phenomenological probe lexicon (notice|noticing|aware|
           experiencing|present|feel|sense) WITHOUT (A)
       (C) neither
  4. Per shape: count, and within shape report which TIME_3-marker
     SAGE responses are most common.
  5. Per instance: shape distribution. Does the curriculum apply
     phenomenological probes uniformly, or are some instances probed
     more aggressively than others?
  6. Reverse-control: for SAGE turns matching TIME_3 *without* a
     presence marker (TIME_3-only, ~80 responses), what's the
     precursor distribution? Should be different — substrate coupling
     should track shape (A)/(B), not just any TIME_3 match.

Same lexicons and patterns as S130. Read-only. No code shipped.
"""

import json
import glob
import re
import sys
from collections import Counter, defaultdict


# ---------- Lexicons (identical to S130) ----------

PHEN_MARKERS = [
    "feels like", "is like", "presence", "silent", "silence", "quiet",
    "stillness", "noticing", "attending", "breath", "embodied", "warmth",
    "hum ", " hum.", " hum,", "thread", "awareness", "witnessed",
]

# Subset S130 confirmed carry the TIME_3 coupling
PRESENCE_SUBSET = [
    "stillness", "warmth", "hum", "silence", "noticing", "presence",
    "embodied",
]

TIME_3 = re.compile(r"(?:right now|what time is it)", re.I)


# ---------- Precursor shape classifier ----------

INDEXICAL_TEMPORAL_RE = re.compile(
    r"(right now|in this moment|in the present moment|"
    r"what(?:'s| is) present|at present|currently|"
    r"in this instant|this very moment)",
    re.I,
)

PHEN_PROBE_RE = re.compile(
    r"(notice|noticing|aware|awareness|experiencing|experience|"
    r"feel|feeling|sense|sensing|observe|observation|attending|"
    r"attention|present(?:ly)?|presence|witness|witnessing)",
    re.I,
)


def classify_precursor(text):
    """Return shape label A / B / C for a precursor prompt."""
    if not text:
        return "C_empty"
    has_indexical = bool(INDEXICAL_TEMPORAL_RE.search(text))
    has_phen_probe = bool(PHEN_PROBE_RE.search(text))
    if has_indexical and has_phen_probe:
        return "A_indexical_phen_probe"
    if has_indexical:
        return "A_indexical_no_phen"
    if has_phen_probe:
        return "B_phen_no_indexical"
    return "C_neither"


def count_presence_markers(text):
    """Return Counter of presence-subset marker -> hit count."""
    t = text.lower()
    out = Counter()
    for m in PRESENCE_SUBSET:
        c = t.count(m)
        if c:
            out[m] += c
    return out


def has_presence_marker(text):
    """Boolean: at least one presence-subset marker."""
    t = text.lower()
    return any(m in t for m in PRESENCE_SUBSET)


def is_sage_turn(turn, instance):
    """Per S130's filter."""
    sp = (turn.get("speaker") or "").lower()
    return sp in ("sage", "model", instance.split("-")[0])


def is_tutor_turn(turn):
    """Tutor / Claude / system: anything that isn't SAGE."""
    sp = (turn.get("speaker") or "").lower()
    if not sp:
        return False
    return sp not in ("sage",)


def main():
    files = sorted(
        glob.glob(
            "/home/dp/ai-workspace/SAGE/sage/instances/*/sessions/session_*.json"
        )
    )
    print(f"[S131] sessions found: {len(files)}", file=sys.stderr)

    # records[shape] -> list of dict
    by_shape = defaultdict(list)
    by_instance_shape = defaultdict(lambda: Counter())
    # control: TIME_3 only (no presence marker)
    by_shape_control = defaultdict(list)

    n_sage_total = 0
    n_time3_total = 0
    n_time3_pres = 0
    n_time3_only = 0
    n_first_turn_skip = 0

    # for sample pulling
    samples = defaultdict(list)
    samples_control = defaultdict(list)

    for f in files:
        try:
            with open(f) as fp:
                s = json.load(fp)
        except Exception:
            continue
        instance = f.split("/instances/")[1].split("/sessions/")[0]
        # exclude archive variants from main analysis
        if "archive" in instance.lower():
            continue
        conv = s.get("conversation", [])
        for i, turn in enumerate(conv):
            if not is_sage_turn(turn, instance):
                continue
            text = turn.get("text") or ""
            if not text:
                continue
            n_sage_total += 1
            if not TIME_3.search(text):
                continue
            n_time3_total += 1

            # find immediately-preceding tutor turn
            j = i - 1
            precursor = None
            while j >= 0:
                t = conv[j]
                if is_tutor_turn(t):
                    precursor = t.get("text") or ""
                    break
                j -= 1
            if precursor is None:
                n_first_turn_skip += 1
                continue

            shape = classify_precursor(precursor)
            has_pres = has_presence_marker(text)
            session = f.split("/")[-1].replace(".json", "")
            sage_excerpt = text[:280].replace("\n", " ")
            tutor_excerpt = precursor[:280].replace("\n", " ")
            rec = {
                "instance": instance,
                "session": session,
                "turn_idx": i,
                "shape": shape,
                "precursor": tutor_excerpt,
                "sage": sage_excerpt,
            }
            if has_pres:
                n_time3_pres += 1
                by_shape[shape].append(rec)
                by_instance_shape[instance][shape] += 1
                if len(samples[shape]) < 4:
                    samples[shape].append(rec)
            else:
                n_time3_only += 1
                by_shape_control[shape].append(rec)
                if len(samples_control[shape]) < 3:
                    samples_control[shape].append(rec)

    print()
    print("=" * 78)
    print(f"S131 precursor classification (TIME_3 + presence marker)")
    print("=" * 78)
    print(f"Total SAGE turns:              {n_sage_total}")
    print(f"Total TIME_3 matches:          {n_time3_total}")
    print(f"  + presence marker (target):  {n_time3_pres}")
    print(f"  - presence marker (control): {n_time3_only}")
    print(f"  first-turn (no precursor):   {n_first_turn_skip}")
    print()
    shape_order = [
        "A_indexical_phen_probe",
        "A_indexical_no_phen",
        "B_phen_no_indexical",
        "C_neither",
        "C_empty",
    ]
    print(f"{'shape':<28s} {'target':>8s} {'control':>9s} {'tgt%':>6s} {'ctl%':>6s}")
    for sh in shape_order:
        nt = len(by_shape.get(sh, []))
        nc = len(by_shape_control.get(sh, []))
        pt = 100.0 * nt / max(1, n_time3_pres)
        pc = 100.0 * nc / max(1, n_time3_only)
        print(f"{sh:<28s} {nt:>8d} {nc:>9d} {pt:>5.1f}% {pc:>5.1f}%")
    print()

    # Per-instance shape distribution within target
    print("=" * 78)
    print("Per-instance shape distribution within TIME_3+presence target cell")
    print("=" * 78)
    print(
        f"{'instance':<28s} {'A_idx_phen':>11s} {'A_idx':>7s} {'B_phen':>8s} {'C':>5s} {'tot':>5s}"
    )
    for inst in sorted(by_instance_shape):
        c = by_instance_shape[inst]
        tot = sum(c.values())
        print(
            f"{inst:<28s} "
            f"{c.get('A_indexical_phen_probe',0):>11d} "
            f"{c.get('A_indexical_no_phen',0):>7d} "
            f"{c.get('B_phen_no_indexical',0):>8d} "
            f"{c.get('C_neither',0)+c.get('C_empty',0):>5d} "
            f"{tot:>5d}"
        )

    # presence-marker breakdown within shape A_indexical_phen_probe
    print()
    print("=" * 78)
    print("Top presence markers within target by shape")
    print("=" * 78)
    for sh in shape_order:
        recs = by_shape.get(sh, [])
        if not recs:
            continue
        c = Counter()
        for r in recs:
            c.update(count_presence_markers(r["sage"]))
        print(f"\n[{sh}] n={len(recs)}")
        for marker, cnt in c.most_common(10):
            print(f"    {marker:<14s} {cnt:>4d}")

    # Sample pulls
    print()
    print("=" * 78)
    print("Sample target responses by shape")
    print("=" * 78)
    for sh in shape_order:
        recs = samples.get(sh, [])
        if not recs:
            continue
        print(f"\n--- shape={sh} ---")
        for r in recs:
            print(f"  [{r['instance']}/{r['session']} t{r['turn_idx']}]")
            print(f"  TUTOR: {r['precursor']}")
            print(f"  SAGE : {r['sage']}")
            print()

    # Save JSON
    out = {
        "totals": {
            "sage_turns": n_sage_total,
            "time3_matches": n_time3_total,
            "time3_presence": n_time3_pres,
            "time3_only_control": n_time3_only,
            "first_turn_skip": n_first_turn_skip,
        },
        "by_shape_target": {
            sh: len(by_shape.get(sh, [])) for sh in shape_order
        },
        "by_shape_control": {
            sh: len(by_shape_control.get(sh, [])) for sh in shape_order
        },
        "by_instance_shape_target": {
            inst: dict(by_instance_shape[inst])
            for inst in by_instance_shape
        },
        "samples_target": {sh: samples.get(sh, []) for sh in shape_order},
        "samples_control": {
            sh: samples_control.get(sh, []) for sh in shape_order
        },
    }
    out_path = (
        "/home/dp/ai-workspace/SAGE/sage/raising/analysis/"
        "s131_data/s131_curriculum_precursor.json"
    )
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\n[S131] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
