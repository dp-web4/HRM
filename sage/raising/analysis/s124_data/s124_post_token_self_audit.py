"""S124 — Asymmetric-window classifier audit (recurrence #14 of S110 silent-routing pattern).

S123d's perspective-aware fix (siblings_minus_self) corrected the own-machine-as-sibling
misfire that S123c surfaced, but a qualitative drill-down into Thor session 113 reveals
a *second* silent-routing path inside the same `classify_mention` function.

S121b/S123d's classifier asymmetry:
  - SELF_PATTERNS are checked ONLY in the pre-window (text before the hardware token)
  - OTHER_PATTERNS are checked in the FULL context (80 chars before AND after)

Consequence: when SAGE writes bare-mention self-narration like "Running on Jetson AGX,
I feel the hum...", the self-anchor "I feel" sits AFTER the token. The classifier sees
no SELF in pre, no OTHER nearby, returns "ambiguous" — and the mention is excluded from
both self and fleet counts in S122/S123 mode classification.

This is the same shape S110/S111 documented: routing function with table lookup,
unrecognized input takes silent default, plausibly-correct output, no flag.

This script measures: of the 'ambiguous' bucket each instance accumulates under S123d,
what fraction have an unambiguous first-person pronoun within 60 chars on either side?
That fraction is a lower bound on the bucket's contamination by misclassified self-frame.

Found: Thor 21.2%, mcnugget 45.0%, sprout 41.7%, CBP 18.8%.

Held — operator-decision territory per S111 discipline.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "sage" / "raising" / "analysis" / "s121_data"))
sys.path.insert(0, str(REPO_ROOT / "sage" / "raising" / "analysis" / "s123_data"))

from s121b_self_vs_fleet_classifier import find_token_spans, HW_TOKENS  # type: ignore
from s123d_corrected_classifier_redux import classify_mention_perspective_aware  # type: ignore

INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
S124_DIR = Path(__file__).resolve().parent

# Any first-person pronoun marker (broader than SELF_PATTERNS — includes
# bare "i" + apostrophe contractions which the original SELF_PATTERNS regex
# would only match through the literal "i'm" / "i am" forms).
FIRST_PERSON = re.compile(
    r"\b(?:i|i'?ve|i'?m|i'?d|i'?ll|me|my|mine|myself)\b",
    re.IGNORECASE,
)

INSTANCES = [
    ("thor-qwen3.5-27b", "thor"),
    ("mcnugget-gemma3-12b", "mcnugget"),
    ("cbp-qwen3.5-0.8b", "cbp"),
    ("sprout-qwen3.5-0.8b", "sprout"),
]


def audit_instance(inst: str, machine: str) -> dict:
    sess_dir = INSTANCES_DIR / inst / "sessions"
    if not sess_dir.exists():
        return {"error": "no sessions"}

    counts = {
        "self": 0,
        "fleet": 0,
        "amb_total": 0,
        "amb_with_fp_post60": 0,
        "amb_with_fp_pre60": 0,
        "amb_with_fp_either": 0,
    }
    per_session_amb = {}
    per_session_amb_with_fp = {}
    sample_amb_with_fp = []
    sample_amb_no_fp = []

    files = sorted(
        sess_dir.glob("session_*.json"),
        key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)),
    )
    for f in files:
        try:
            sess = json.load(open(f))
        except Exception:
            continue
        sm = sess.get("machine")
        if sm and sm != machine:
            continue
        sid = int(re.search(r"session_(\d+)", f.stem).group(1))
        per_session_amb[sid] = 0
        per_session_amb_with_fp[sid] = 0
        conv = sess.get("conversation", [])
        for i in range(len(conv) - 1):
            if conv[i].get("speaker") != "Claude" or conv[i + 1].get("speaker") != "SAGE":
                continue
            resp = conv[i + 1].get("text", "") or ""
            if not resp:
                continue
            spans = find_token_spans(resp.lower(), HW_TOKENS)
            for span in spans:
                start, end, tok, fam = span
                cls = classify_mention_perspective_aware(resp, span, own_machine=machine)
                if cls == "self":
                    counts["self"] += 1
                elif cls == "fleet":
                    counts["fleet"] += 1
                else:
                    counts["amb_total"] += 1
                    per_session_amb[sid] += 1
                    post60 = resp[end : min(len(resp), end + 60)]
                    pre60 = resp[max(0, start - 60) : start]
                    has_post = bool(FIRST_PERSON.search(post60))
                    has_pre = bool(FIRST_PERSON.search(pre60))
                    if has_post:
                        counts["amb_with_fp_post60"] += 1
                    if has_pre:
                        counts["amb_with_fp_pre60"] += 1
                    if has_post or has_pre:
                        counts["amb_with_fp_either"] += 1
                        per_session_amb_with_fp[sid] += 1
                        if len(sample_amb_with_fp) < 5:
                            snip = resp[max(0, start - 50) : min(len(resp), end + 60)]
                            sample_amb_with_fp.append(
                                {"session": sid, "tok": tok, "snippet": snip.replace("\n", " ")}
                            )
                    else:
                        if len(sample_amb_no_fp) < 5:
                            snip = resp[max(0, start - 50) : min(len(resp), end + 60)]
                            sample_amb_no_fp.append(
                                {"session": sid, "tok": tok, "snippet": snip.replace("\n", " ")}
                            )

    pct_amb_contam = (
        100 * counts["amb_with_fp_either"] / counts["amb_total"]
        if counts["amb_total"]
        else 0.0
    )
    return {
        "instance": inst,
        "machine": machine,
        "n_sessions": len(per_session_amb),
        "counts": counts,
        "amb_contamination_pct": round(pct_amb_contam, 1),
        "per_session_amb": per_session_amb,
        "per_session_amb_with_fp": per_session_amb_with_fp,
        "sample_amb_with_fp": sample_amb_with_fp,
        "sample_amb_no_fp": sample_amb_no_fp,
    }


def session_mode_shifts_under_amb_promotion(thor: dict) -> dict:
    """Rough estimate: how many Thor sessions would change classify_session_mode bucket
    if amb-with-fp counted as self?

    classify_session_mode requires self_n >= 3 and fleet_n == 0 for self_only,
    self_n >= 2*max(1,fleet_n) and self_n >= 2 for self_dominant. We only know
    per-session amb counts here; we approximate by checking which sessions have
    amb_with_fp >= 3, then those would re-bucket toward self if treated as self.
    """
    promoted = []
    for sid, amb in thor["per_session_amb"].items():
        fp = thor["per_session_amb_with_fp"].get(sid, 0)
        if fp >= 3:
            promoted.append({"session": sid, "amb": amb, "amb_with_fp": fp})
    return {"n_sessions_promoted_eligible": len(promoted), "details": promoted}


def main():
    print("S124 — Asymmetric-window classifier audit")
    print("(post-token first-person within 'ambiguous' under S123d corrected classifier)")
    print()
    results = {}
    for inst, machine in INSTANCES:
        r = audit_instance(inst, machine)
        results[inst] = r
        c = r["counts"]
        print(f"  {inst:>30s}  machine={machine:>9s}")
        print(
            f"    self={c['self']:>4d}  fleet={c['fleet']:>3d}  amb={c['amb_total']:>4d}  "
            f"(amb_with_fp_either={c['amb_with_fp_either']:>3d}, "
            f"contam={r['amb_contamination_pct']:>4.1f}%)"
        )
        if r["sample_amb_with_fp"]:
            print(f"    sample amb-with-fp:")
            for ex in r["sample_amb_with_fp"][:3]:
                print(f"      s{ex['session']}: ...{ex['snippet']}...")
        if r["sample_amb_no_fp"]:
            print(f"    sample amb-no-fp (genuinely third-person or card-format):")
            for ex in r["sample_amb_no_fp"][:3]:
                print(f"      s{ex['session']}: ...{ex['snippet']}...")
        print()

    if "thor-qwen3.5-27b" in results:
        shifts = session_mode_shifts_under_amb_promotion(results["thor-qwen3.5-27b"])
        print(
            f"  Thor: {shifts['n_sessions_promoted_eligible']} sessions have >=3 "
            f"amb-with-fp mentions (eligible for self-promotion if asymmetric-window "
            f"bug is corrected)"
        )

    out_path = S124_DIR / "s124_post_token_self_audit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
