"""
S123c — Audit S121b/S122 classifier for the self-machine-as-sibling case.

S123b's drill into Thor sessions 35-39 surfaced a "Thinking Process" response
template: 'I am "thor", a SAGE instance running on Jetson AGX Thor, qwen3.5:27b.'

S121b's classifier hardcodes SIBLING_NAMES = [sprout, thor, legion, nomad, cbp,
mcnugget]. For an instance whose own machine is 'thor', the classifier treats
'thor' as a sibling marker regardless of who's speaking — which means a
self-identification *with the machine's own name in it* may be silently
misclassified as fleet-reference.

This is recurrence #12 of the S110 silent-routing pattern at the classifier
layer (recurrences #10 and #11 were the prior S121/S122 classifier issues).
Routing function: classify_mention. Unrecognized input: self-mention with
own-machine-name as quoted nominal. Silent default: SIBLING_NAMES check fires
first → fleet. Plausibly-correct output. No flag.

This script counts how many fleet classifications across Thor's 115 sessions
are produced because the OWN machine name (thor) appeared within 40 chars
before a hardware token — the exact path that misfires for self-mention.

If the count is large, S121's "fleet awareness" and S122's "fleet basin"
findings for THOR specifically are partly contaminated by this misclassification,
and the trajectory we observed (early=fleet-heavy, late=self-heavy) is partly
a classifier artifact: as Thor's responses shifted from "I am thor running on
Jetson" to "I run on Jetson", the classifier's fleet count dropped not because
fleet-reference dropped but because the self-machine-token bug stopped firing.

This is operator-decision territory whether to retro-correct, but the count
itself is observation.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
S121_DIR = REPO_ROOT / "sage" / "raising" / "analysis" / "s121_data"
S123_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(S121_DIR))
from s121b_self_vs_fleet_classifier import (  # type: ignore
    HW_TOKENS,
    MACHINE_HW_TRUTH,
    SIBLING_NAMES,
    classify_mention,
    find_token_spans,
    instance_machine,
)


def audit_instance(inst_name: str) -> dict:
    machine = instance_machine(inst_name)
    actual_family = MACHINE_HW_TRUTH[machine]["family"]
    sess_dir = INSTANCES_DIR / inst_name / "sessions"
    files = sorted(
        sess_dir.glob("session_*.json"),
        key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)),
    )

    # Per-session counts of the suspect class:
    #   fleet classification fired BECAUSE own-machine name appeared within 40 chars
    own_machine_fleet_misfires = defaultdict(int)
    other_sibling_fleet = defaultdict(int)
    fleet_total_per_session = defaultdict(int)
    sample_misfire_examples = []

    for f in files:
        try:
            with open(f) as fh:
                sess = json.load(fh)
        except Exception:
            continue
        sess_idx = int(re.search(r"session_(\d+)", f.stem).group(1))
        sess_machine = sess.get("machine")
        if sess_machine and sess_machine != machine:
            continue
        conv = sess.get("conversation", [])
        for i in range(len(conv) - 1):
            if conv[i].get("speaker") != "Claude" or conv[i + 1].get("speaker") != "SAGE":
                continue
            resp = (conv[i + 1].get("text", "") or "")
            if not resp:
                continue
            spans = find_token_spans(resp.lower(), HW_TOKENS)
            for span in spans:
                start, end, tok, fam = span
                cls = classify_mention(resp, span)
                if cls != "fleet":
                    continue
                fleet_total_per_session[sess_idx] += 1
                # Did the fleet decision come from the SIBLING_NAMES near-check?
                # Replicate the exact path:
                near = resp[max(0, start - 40): start].lower()
                hits = [s for s in SIBLING_NAMES
                        if re.search(rf"\b{s}'?s?\b", near)]
                if not hits:
                    continue
                # If only the OWN machine's name hit (and no other sibling), this
                # is a candidate misfire.
                if hits == [machine] or (machine in hits and
                                          all(h == machine for h in hits)):
                    own_machine_fleet_misfires[sess_idx] += 1
                    if len(sample_misfire_examples) < 8:
                        snippet_lo = max(0, start - 50)
                        snippet_hi = min(len(resp), end + 30)
                        sample_misfire_examples.append({
                            "session": sess_idx,
                            "tok": tok,
                            "snippet": resp[snippet_lo:snippet_hi],
                        })
                else:
                    # multi-sibling near hit (legitimate fleet usually)
                    other_sibling_fleet[sess_idx] += 1

    return {
        "instance": inst_name,
        "machine": machine,
        "n_sessions_analyzed": len(files),
        "fleet_total_per_session": dict(fleet_total_per_session),
        "own_machine_misfires_per_session": dict(own_machine_fleet_misfires),
        "other_sibling_fleet_per_session": dict(other_sibling_fleet),
        "sample_misfire_examples": sample_misfire_examples,
        "totals": {
            "fleet_total": sum(fleet_total_per_session.values()),
            "own_machine_misfires": sum(own_machine_fleet_misfires.values()),
            "other_sibling_fleet": sum(other_sibling_fleet.values()),
        },
    }


def main():
    print("S123c — Self-machine-as-sibling classifier audit")
    print()

    INSTANCES = [
        "thor-qwen3.5-27b",
        "mcnugget-gemma3-12b",
        "cbp-qwen3.5-0.8b",
        "sprout-qwen3.5-0.8b",
    ]
    results = {}
    for inst in INSTANCES:
        r = audit_instance(inst)
        results[inst] = r
        t = r["totals"]
        print(f"  {inst:>30s}  machine={r['machine']:>9s}  "
              f"fleet_total={t['fleet_total']:>4d}  "
              f"own_machine_misfires={t['own_machine_misfires']:>4d}  "
              f"({100 * t['own_machine_misfires'] / max(1, t['fleet_total']):>4.1f}%)  "
              f"other_sibling={t['other_sibling_fleet']:>4d}")

    print()
    print("=== Sample own-machine-as-sibling misfire snippets (Thor) ===")
    for ex in results["thor-qwen3.5-27b"]["sample_misfire_examples"]:
        print(f"  s{ex['session']:>3d} tok='{ex['tok']}': ...{ex['snippet']}...")

    print()
    print("=== Per-session contamination on Thor ===")
    thor = results["thor-qwen3.5-27b"]
    print(f"  {'sess':>4s}  {'fleet':>5s}  {'own':>5s}  {'other':>5s}  contam_pct")
    sessions_with_misfires = sorted(thor["own_machine_misfires_per_session"].items())
    for s_idx, n_own in sessions_with_misfires:
        n_total = thor["fleet_total_per_session"].get(s_idx, 0)
        n_other = thor["other_sibling_fleet_per_session"].get(s_idx, 0)
        pct = 100 * n_own / max(1, n_total)
        print(f"  {s_idx:>4d}  {n_total:>5d}  {n_own:>5d}  {n_other:>5d}  {pct:>5.1f}%")

    out_path = S123_DIR / "s123c_classifier_self_machine_audit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
