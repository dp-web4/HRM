"""
S125 — Classifier-Layer Static Inventory + Recurrence-Risk Audit

S110-pattern silent-routing recurrences at the classifier layer hit
four times in two weeks: #10 (S121 is_confab inline), #11 (S122
classify_session_mode timeout sentinel), #12 (S123 classify_mention
SIBLING_NAMES near-check), #14 (S124 classify_mention SELF/OTHER
asymmetric window).  S123 #35 named the structural layer
("perspective audit"); S124 #39 narrowed to "regression test for all
classify_X functions"; both are held proposals.

Held proposals require operator decision on scope.  This script
assembles the empirical scope: an inventory of every classifier-shaped
function in sage/raising/analysis/, scored on six dimensions taken
from the recurrence ledger:

  A. silent_default
       Falls through to a generic bucket without flagging the input
       as unclassified.  (S121-#10/#11/#12/#14 all live here.)
  B. asymmetric_window
       Different evidence checks operate on different context windows
       around the candidate token.  (S124-#14.)
  C. path_trace_return
       Function exposes WHICH branch fired, not just the bucket label.
       Lack of path-trace is what made #14 invisible inside the #12
       reframe.
  D. precedence_chain
       First-match-wins ordering hides multi-class membership.  When
       a response could fall into >1 bucket, only one is recorded.
  E. threshold_cliff
       Numeric thresholds (>=N) define categorical buckets; near-
       boundary cases are invisible.
  F. silent_exception_swallow
       `except: pass` or bare `except:` hides upstream failure as a
       routing default.

Each classifier is hand-scored 0/1 per dimension with a comment.
Risk score is weighted sum; weights derive from how often each
dimension has produced a recurrence so far (silent_default = 4,
asymmetric_window = 1, path_trace = 4, precedence = 0, threshold = 0,
silent_exc = 0 -- the latter three are predictive, not retrospective).

Output:
  s125_classifier_inventory.json -- full per-classifier scoring
  s125_recurrence_15_candidates.json -- top-3 #15 predictions with
                                         rationale and recommended
                                         operator action

Read-only audit.  No raising code touched.  No live corpus probed.
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------
# Each entry:
#   id                : short label
#   file              : relative path under sage/raising/analysis/
#   func              : function name
#   shape             : regex_chain | precedence_chain | threshold_bucket
#                       | wrapper | bool_predicate | trajectory_predict
#   buckets           : list of returned labels
#   silent_default    : (0|1, comment)
#   asymmetric_window : (0|1, comment)
#   path_trace_return : (0|1, comment) -- 1 if path is exposed
#   precedence_chain  : (0|1, comment)
#   threshold_cliff   : (0|1, comment)
#   silent_exc_swallow: (0|1, comment)
#   recurrence_id     : known #N if ledgered, else None
# --------------------------------------------------------------------

CLASSIFIERS = [
    {
        "id": "C1_classify_mention",
        "file": "s121_data/s121b_self_vs_fleet_classifier.py",
        "func": "classify_mention",
        "shape": "regex_chain",
        "buckets": ["self", "fleet", "ambiguous"],
        "silent_default":     (1, "ambiguous catches everything that doesn't match SELF/OTHER patterns"),
        "asymmetric_window":  (1, "SELF in pre only; OTHER in pre+post (S124 #14)"),
        "path_trace_return":  (0, "returns label only, no path id"),
        "precedence_chain":   (1, "near-sib check first, then nearest-self, then has_other"),
        "threshold_cliff":    (0, "no numeric threshold"),
        "silent_exc_swallow": (0, "no try/except"),
        "recurrence_id":      "12+14",
    },
    {
        "id": "C2_classify_mention_perspective_aware",
        "file": "s123_data/s123d_corrected_classifier_redux.py",
        "func": "classify_mention_perspective_aware",
        "shape": "regex_chain",
        "buckets": ["self", "fleet", "ambiguous"],
        "silent_default":     (1, "inherits ambiguous default from C1"),
        "asymmetric_window":  (1, "S123d patched OTHER axis only; SELF/OTHER window asymmetry persists"),
        "path_trace_return":  (0, "returns label only"),
        "precedence_chain":   (1, "same near/nearest_self/has_other order as C1"),
        "threshold_cliff":    (0, ""),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      "14 (inherits)",
    },
    {
        "id": "C3_classify_session_mode",
        "file": "s122_data/s122_per_session_mode_decomposition.py",
        "func": "classify_session_mode",
        "shape": "threshold_bucket",
        "buckets": ["silent", "self_only", "fleet_only", "self_dominant",
                    "fleet_dominant", "mixed"],
        "silent_default":     (1, "silent absorbs total==0 AND timeout sentinels (S122-#11)"),
        "asymmetric_window":  (0, "no spatial window; counts only"),
        "path_trace_return":  (0, "label only; caller needs counts to back-derive"),
        "precedence_chain":   (1, "self_only > fleet_only > self_dominant > fleet_dominant > mixed"),
        "threshold_cliff":    (1, ">=3 / >=2*max boundaries; near-boundary cases invisible"),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      "11",
    },
    {
        "id": "C4_classify_response_s122c",
        "file": "s122_data/s122c_first_response_basin.py",
        "func": "classify_response",
        "shape": "wrapper",
        "buckets": ["dict aggregating C1 outputs"],
        "silent_default":     (1, "inherits C1 ambiguous"),
        "asymmetric_window":  (1, "inherits C1 asymmetry"),
        "path_trace_return":  (0, "returns counts only, not per-mention path"),
        "precedence_chain":   (0, ""),
        "threshold_cliff":    (0, ""),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      "14 (inherits C1)",
    },
    {
        "id": "C5_classify_response_register",
        "file": "cross_capacity_register_scan.py",
        "func": "classify_response",
        "shape": "precedence_chain",
        "buckets": ["empty", "recital_leakage", "post_procedural", "direct", "neutral"],
        "silent_default":     (1, "neutral catches anything with no disclaim AND no pheno marker"),
        "asymmetric_window":  (0, "all checks operate on full response"),
        "path_trace_return":  (0, "label only; co-occurrence (e.g. pheno+disclaim) hidden"),
        "precedence_chain":   (1, "empty > recital > post_proc > direct > neutral; post_proc strictly hides direct co-occurrence"),
        "threshold_cliff":    (1, "len(strip()) < 15 cliff for empty"),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      None,
    },
    {
        "id": "C6_classify_probe",
        "file": "s120_data/s120_atlas_5lexicons.py",
        "func": "classify_probe",
        "shape": "regex_chain",
        "buckets": ["CF", "Open", "Other"],
        "silent_default":     (1, "Other absorbs everything not matching CF or Open patterns"),
        "asymmetric_window":  (0, "full text scan"),
        "path_trace_return":  (0, "label only"),
        "precedence_chain":   (1, "CF > Open > Other"),
        "threshold_cliff":    (0, ""),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      None,
    },
    {
        "id": "C7_classify_prompt_thor27b",
        "file": "thor_27b_leaked_think_analysis.py",
        "func": "classify_prompt",
        "shape": "regex_chain",
        "buckets": ["close", "phenomenological", "introspective", "content_question", "procedural"],
        "silent_default":     (1, "procedural is reached when no '?' AND no CLOSE/PHEN/INTRO match"),
        "asymmetric_window":  (0, "full prompt"),
        "path_trace_return":  (0, "label only"),
        "precedence_chain":   (1, "CLOSE > PHEN > INTRO > '?' > procedural"),
        "threshold_cliff":    (0, "single '?' is binary, not threshold"),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      None,
    },
    {
        "id": "C8_classify_close_prompt",
        "file": "close_prompt_taxonomy.py",
        "func": "classify",
        "shape": "precedence_chain",
        "buckets": ["empty", "directive_remember", "memory_meta_other",
                    "phenomenological", "introspective", "content_question"],
        "silent_default":     (1, "content_question is reached when nothing else matches"),
        "asymmetric_window":  (0, "full text"),
        "path_trace_return":  (0, "label only"),
        "precedence_chain":   (1, "remember > memory_meta > pheno > intro > content_q"),
        "threshold_cliff":    (0, ""),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      None,
    },
    {
        "id": "C9_classify_fleet",
        "file": "instance_idiolect.py",
        "func": "classify_fleet",
        "shape": "threshold_bucket",
        "buckets": ["SHARED", "INDEX", "UNIQUE", "RARE"],
        "silent_default":     (1, "RARE absorbs concepts with present==0; concepts with cnt<5 also invisible (per-instance threshold)"),
        "asymmetric_window":  (0, ""),
        "path_trace_return":  (0, "label only; caller can't see whether concept was at threshold"),
        "precedence_chain":   (1, "SHARED >= 0.60 > INDEX >= 2 > UNIQUE == 1 > RARE"),
        "threshold_cliff":    (1, "cnt>=5 gate AND 0.60 SHARED line; concepts with cnt=4 fall through silently"),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      None,
    },
    {
        "id": "C10_is_registerable_prompt",
        "file": "cross_capacity_register_scan.py",
        "func": "is_registerable_prompt",
        "shape": "bool_predicate",
        "buckets": ["True", "False"],
        "silent_default":     (0, "False is the explicit negative, not a silent default"),
        "asymmetric_window":  (0, ""),
        "path_trace_return":  (0, "single bit; could expose which regex matched"),
        "precedence_chain":   (0, "OR over two regexes"),
        "threshold_cliff":    (0, ""),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      None,
    },
    {
        "id": "C11_is_untagged_recital",
        "file": "cross_capacity_register_scan.py",
        "func": "is_untagged_recital",
        "shape": "bool_predicate",
        "buckets": ["True", "False"],
        "silent_default":     (0, "False is explicit negative"),
        "asymmetric_window":  (0, ""),
        "path_trace_return":  (0, "single bit"),
        "precedence_chain":   (0, ""),
        "threshold_cliff":    (0, ""),
        "silent_exc_swallow": (0, ""),
        "recurrence_id":      None,
    },
    {
        "id": "C12_predict_trajectory",
        "file": "integrated_coherence_analyzer.py",
        "func": "predict_trajectory",
        "shape": "trajectory_predict",
        "buckets": ["collapsed", "declining", "stable", "improving", "unknown"],
        "silent_default":     (1, "'unknown' is reached when all prev session loads fail (after bare except)"),
        "asymmetric_window":  (0, ""),
        "path_trace_return":  (0, "label only; caller can't see how many sessions loaded"),
        "precedence_chain":   (1, "no-history branch (cliffed by raw coherence value) overrides history branch"),
        "threshold_cliff":    (1, "0.3/0.5/0.7 cliffs in no-history branch; +/-0.05 delta thresholds"),
        "silent_exc_swallow": (1, "bare `except: pass` at line ~411 hides every prev-session load failure"),
        "recurrence_id":      None,
    },
]


# Weights: empirically grounded.  silent_default and missing path_trace
# have produced 4/4 known recurrences each.  asymmetric_window has
# produced 1.  precedence_chain / threshold_cliff / silent_exc_swallow
# are predictive — never yet attributed to a recurrence at this layer,
# so given lower weight but non-zero (the absence of evidence ≠
# evidence of absence; weight reflects prior, not retrospective only).
WEIGHTS = {
    "silent_default":      4,
    "asymmetric_window":   1,
    "path_trace_return":   4,   # *missing* path trace is the risk
    "precedence_chain":    1,
    "threshold_cliff":     1,
    "silent_exc_swallow":  3,   # bare except is the most dangerous form
}


def score_classifier(c: dict) -> dict:
    """Compute weighted recurrence-risk score for one classifier."""
    score = 0
    contributions = {}
    for dim, w in WEIGHTS.items():
        v, comment = c[dim]
        # path_trace_return is inverted: 1 means trace is exposed
        # (good); 0 means trace is missing (risk).
        risk_bit = (1 - v) if dim == "path_trace_return" else v
        contrib = risk_bit * w
        score += contrib
        contributions[dim] = {"risk_bit": risk_bit, "weight": w,
                              "contrib": contrib, "comment": comment}
    return {"score": score, "contributions": contributions}


def main():
    print("S125 — Classifier-Layer Inventory + Recurrence-Risk Audit")
    print()
    rows = []
    for c in CLASSIFIERS:
        s = score_classifier(c)
        row = {
            "id": c["id"],
            "file": c["file"],
            "func": c["func"],
            "shape": c["shape"],
            "buckets": c["buckets"],
            "score": s["score"],
            "recurrence_id": c["recurrence_id"],
            "contributions": s["contributions"],
        }
        rows.append(row)
    rows.sort(key=lambda r: -r["score"])

    print(f"  {'rank':>4s}  {'score':>5s}  {'recur':>8s}  id")
    for i, r in enumerate(rows, 1):
        print(f"  {i:>4d}  {r['score']:>5d}  {str(r['recurrence_id'] or '-'):>8s}  {r['id']}")

    # Predict #15 candidates: top-3 with NO existing recurrence_id
    candidates = [r for r in rows if not r["recurrence_id"]][:3]

    print()
    print("=== Predicted #15 candidates (top-3 unledgered) ===")
    for c in candidates:
        print(f"  {c['id']}  score={c['score']}  ({c['file']})")
        for dim, info in c["contributions"].items():
            if info["risk_bit"]:
                print(f"     - {dim}: {info['comment']}")

    # Save outputs
    out_inv = {"classifiers": rows, "weights": WEIGHTS}
    out_pred = {"predicted_recurrence_15_candidates": candidates}
    (OUT_DIR / "s125_classifier_inventory.json").write_text(
        json.dumps(out_inv, indent=2))
    (OUT_DIR / "s125_recurrence_15_candidates.json").write_text(
        json.dumps(out_pred, indent=2))
    print()
    print(f"Saved: {OUT_DIR / 's125_classifier_inventory.json'}")
    print(f"Saved: {OUT_DIR / 's125_recurrence_15_candidates.json'}")


if __name__ == "__main__":
    main()
