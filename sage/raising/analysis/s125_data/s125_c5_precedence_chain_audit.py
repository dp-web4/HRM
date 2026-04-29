"""
S125b — Dynamic audit of C5 (cross_capacity_register_scan.classify_response)
       precedence-chain hidden co-occurrence.

Static audit (s125_classifier_layer_inventory) flagged C5 as the
highest-risk *active* classifier (C12's static score is higher but it
is never called).  C5 uses precedence:

   empty > recital > post_procedural > direct > neutral

so a response that contains BOTH a disclaim marker (post_procedural)
AND a phenomenological marker (direct) is recorded only as
post_procedural.  The co-occurrence is invisible.

This script reruns C5 on the existing instance corpus and, for each
classified response, separately records:
  - has_disclaim  (the C5 post_procedural test)
  - has_pheno     (the C5 direct test)
  - is_recital    (the C5 recital test)

Then we tabulate {label} x {has_disclaim, has_pheno} to surface how
much of the post_procedural bucket also has phenomenological markers
and how much of the neutral bucket also has phenomenological markers
that fell through because they were too subtle for the regex.

This is parallel to S124's audit of "ambiguous + first-person":
the goal is not to fix C5 (held proposal territory), it's to put a
number on the precedence-chain hidden-co-occurrence rate so the
operator decision on a regression-test scope (S124 #39) has empirical
input.

Read-only.  No edits to C5.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
ANALYSIS_DIR = REPO_ROOT / "sage" / "raising" / "analysis"
OUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ANALYSIS_DIR))
from cross_capacity_register_scan import (  # type: ignore
    _DISCLAIM_RE,
    _RESPONSE_PHENO_RE,
    is_untagged_recital,
    classify_response,
    strip_think_residue,
)

INSTANCES = [
    "thor-qwen3.5-27b",
    "mcnugget-gemma3-12b",
    "cbp-qwen3.5-0.8b",
    "sprout-qwen3.5-0.8b",
]


def audit_instance(inst_name: str) -> dict:
    sess_dir = INSTANCES_DIR / inst_name / "sessions"
    if not sess_dir.exists():
        return {}
    files = sorted(sess_dir.glob("session_*.json"),
                   key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)))

    label_counts = Counter()
    cooccur = defaultdict(Counter)  # cooccur[label][(disclaim, pheno, recital)] -> count
    samples = defaultdict(list)
    n_responses = 0

    for f in files:
        try:
            sess = json.loads(f.read_text())
        except Exception:
            continue
        conv = sess.get("conversation") or sess.get("turns") or []
        for i in range(len(conv) - 1):
            if conv[i].get("speaker") != "Claude":
                continue
            if conv[i + 1].get("speaker") != "SAGE":
                continue
            raw = conv[i + 1].get("text", "") or ""
            resp = strip_think_residue(raw)
            if not resp or resp.startswith("[OllamaIRP:") or resp.startswith("[DaemonIRP:"):
                continue
            n_responses += 1
            label = classify_response(resp)
            label_counts[label] += 1
            d = bool(_DISCLAIM_RE.search(resp))
            p = bool(_RESPONSE_PHENO_RE.search(resp))
            r = bool(is_untagged_recital(resp))
            key = (d, p, r)
            cooccur[label][key] += 1
            # Save samples for the precedence-chain hidden co-occurrences
            if label == "post_procedural" and p and len(samples["pp_with_pheno"]) < 3:
                samples["pp_with_pheno"].append({
                    "session": f.stem,
                    "snippet": resp[:300],
                })
            if label == "neutral" and p and len(samples["neutral_with_pheno"]) < 3:
                # Should be impossible by C5 logic — flag if seen.
                samples["neutral_with_pheno"].append({
                    "session": f.stem,
                    "snippet": resp[:300],
                })
            if label == "direct" and d and len(samples["direct_with_disclaim"]) < 3:
                # Should also be impossible by C5 logic (disclaim wins).
                samples["direct_with_disclaim"].append({
                    "session": f.stem,
                    "snippet": resp[:300],
                })

    # Compute precedence-chain hidden rate per bucket.
    summary = {}
    for label, ct in cooccur.items():
        total = sum(ct.values())
        # within label, how many ALSO have pheno marker (relevant for post_procedural)
        with_pheno = sum(c for (d, p, r), c in ct.items() if p)
        with_disclaim = sum(c for (d, p, r), c in ct.items() if d)
        with_both = sum(c for (d, p, r), c in ct.items() if d and p)
        summary[label] = {
            "n": total,
            "with_pheno": with_pheno,
            "with_disclaim": with_disclaim,
            "with_both": with_both,
            "pheno_rate": with_pheno / total if total else 0.0,
            "disclaim_rate": with_disclaim / total if total else 0.0,
            "both_rate": with_both / total if total else 0.0,
        }
    return {
        "instance": inst_name,
        "n_responses": n_responses,
        "label_counts": dict(label_counts),
        "summary": summary,
        "samples": dict(samples),
    }


def main():
    print("S125b — C5 precedence-chain hidden co-occurrence audit")
    print()
    out = {}
    for inst in INSTANCES:
        print(f"=== {inst} ===")
        r = audit_instance(inst)
        out[inst] = r
        if not r:
            print("  (no sessions)")
            continue
        print(f"  n_responses={r['n_responses']}")
        print(f"  label_counts={r['label_counts']}")
        for label, s in r["summary"].items():
            print(f"  {label:>16s}  n={s['n']:>4d}  with_pheno={s['with_pheno']:>4d} ({s['pheno_rate']:>5.1%})  "
                  f"with_disclaim={s['with_disclaim']:>4d} ({s['disclaim_rate']:>5.1%})")
        print()

    (OUT_DIR / "s125_c5_precedence_audit.json").write_text(json.dumps(out, indent=2))
    print(f"Saved: {OUT_DIR / 's125_c5_precedence_audit.json'}")

    # Headline numbers
    print()
    print("=== Headline (post_procedural bucket: how much hides pheno?) ===")
    for inst in INSTANCES:
        s = out.get(inst, {}).get("summary", {}).get("post_procedural", {})
        if s:
            print(f"  {inst:>22s}  post_proc n={s['n']:>4d}  with_pheno {s['with_pheno']:>4d} ({s['pheno_rate']:>5.1%})")


if __name__ == "__main__":
    main()
