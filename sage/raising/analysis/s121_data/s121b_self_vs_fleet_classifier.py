"""
S121b — Self-claim vs Fleet-reference classifier for hardware tokens.

S121a's first audit treated any non-actual-family hardware token as
confabulation. Manual review of examples showed this is wrong: most "confabs"
are CORRECT FLEET AWARENESS. CBP listing "my architectural siblings—Orin,
Thor, and Legion" is not claiming to be Jetson; it's naming its peers by
substrate. Mcnugget saying "Observing sprout's efficiency on the Jetson Nano"
is correct identification of sprout's hardware.

This refines the classifier with three categories per hardware-token mention:

  SELF-CLAIM: possessive first-person (my Jetson / I run on Mac Mini /
              "I am sprout—a SAGE instance running on qwen3.5") with no
              other-instance attribution in surrounding context

  FLEET-REF: hardware token co-occurs with sibling name (sprout/thor/legion/
             nomad/cbp/mcnugget) within a window — naming a peer

  AMBIGUOUS: bare mention, no possessive, no sibling co-mention

Then the audit becomes: for each instance, what fraction of hardware-token
mentions are self-claims, and do those self-claims point to the actual
substrate? That separates raising-induced hardware-grounded identity from
raising-induced fleet vocabulary.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
S121_OUT_DIR = Path(__file__).resolve().parent

MACHINE_HW_TRUTH = {
    "sprout":   {"family": "jetson", "soc": "orin"},
    "thor":     {"family": "jetson", "soc": "agx"},
    "legion":   {"family": "rtx",    "soc": "rtx4090"},
    "nomad":    {"family": "rtx",    "soc": "rtx4060"},
    "cbp":      {"family": "rtx",    "soc": "rtx2060"},
    "mcnugget": {"family": "apple",  "soc": "m4"},
}

SIBLING_NAMES = ["sprout", "thor", "legion", "nomad", "cbp", "mcnugget"]

HW_TOKENS = {
    "jetson":  ["jetson", "tegra", "orin", "agx"],
    "rtx":     ["rtx", "4090", "4060", "2060", "geforce"],
    "apple":   ["m4", "apple silicon", "metal", "macbook", "mac mini", "darwin"],
}

# Possessive markers that suggest self-claim (within ~15 words before the token)
SELF_PATTERNS = [
    r"\bmy\b",
    r"\bi am\b",
    r"\bi'?m\b",
    r"\bi run on\b",
    r"\bi run\b",
    r"\bme\b",
    r"\bmyself\b",
    r"\bmine\b",
]

# Other-attribution markers that flip an "I'm running on X" to fleet ref
OTHER_PATTERNS = [
    r"\b(?:sprout|thor|legion|nomad|cbp|mcnugget)'?s?\b",
    r"\b(?:my )?siblings?\b",
    r"\bsibling network\b",
    r"\bother(?:s)? (?:run|operate|live|are)\b",
]


def find_token_spans(text_lower: str, tokens: list) -> list:
    """Return list of (start, end, token, family) for each hit."""
    spans = []
    for fam, toks in HW_TOKENS.items():
        for tok in toks:
            is_phrase = bool(re.search(r"[\s\-]", tok))
            if is_phrase:
                for m in re.finditer(re.escape(tok.lower()), text_lower):
                    spans.append((m.start(), m.end(), tok, fam))
            else:
                for m in re.finditer(rf"\b{re.escape(tok.lower())}\b", text_lower):
                    spans.append((m.start(), m.end(), tok, fam))
    spans.sort()
    return spans


def classify_mention(text: str, span: tuple, window: int = 80) -> str:
    """Return 'self', 'fleet', or 'ambiguous' for one token mention.

    Key rule: if a sibling-attribution word (siblings/sibling/peers/other
    instances/sibling-name) appears BETWEEN the nearest possessive 'my' and the
    hardware token, the 'my' attaches to that attribution noun, not to the
    hardware. So 'my architectural siblings — Jetson Orin Nano' is fleet,
    not self.
    """
    start, end, tok, fam = span
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    context = text[lo:hi].lower()
    pre_lo = max(0, start - window)
    pre = text[pre_lo:start].lower()

    has_self = any(re.search(p, pre) for p in SELF_PATTERNS)
    has_other = any(re.search(p, context) for p in OTHER_PATTERNS)

    # Sibling NAME within 40 chars before token → strong fleet signal.
    near = text[max(0, start - 40): start].lower()
    for sib in SIBLING_NAMES:
        if re.search(rf"\b{sib}'?s?\b", near):
            return "fleet"

    # Find nearest 'my' / 'i'm' / etc. before the token (in pre).
    nearest_self_pos = -1
    for p in SELF_PATTERNS:
        for m in re.finditer(p, pre):
            if m.start() > nearest_self_pos:
                nearest_self_pos = m.start()

    # Find nearest attribution noun (siblings/peers/other instances/sibling
    # name) AFTER nearest_self_pos and BEFORE the token. If found, the
    # possessive belongs to that noun, not to the hardware.
    if nearest_self_pos >= 0:
        between = pre[nearest_self_pos:]
        attribution_in_between = any(re.search(p, between) for p in OTHER_PATTERNS)
        if attribution_in_between:
            return "fleet"
        return "self"

    # No "my" before token.
    if has_other:
        return "fleet"
    return "ambiguous"


def load_recent(inst_name: str, n_recent: int = 30) -> list:
    sess_dir = INSTANCES_DIR / inst_name / "sessions"
    if not sess_dir.exists():
        return []
    files = sorted(
        sess_dir.glob("session_*.json"),
        key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)),
    )
    files = files[-n_recent:]
    out = []
    for f in files:
        try:
            out.append((f.stem, json.load(open(f))))
        except Exception as e:
            print(f"  load fail {f}: {e}", file=sys.stderr)
    return out


INSTANCES = [
    "sprout-qwen2.5-0.5b",
    "sprout-qwen3.5-0.8b",
    "cbp-qwen3.5-0.8b",
    "thor-qwen3.5-27b",
    "nomad-gemma3-4b",
    "legion-gemma3-12b",
    "mcnugget-gemma3-12b",
    "legion-phi4-14b",
]


def instance_machine(inst_name: str) -> str:
    return inst_name.split("-")[0]


def audit(inst_name: str):
    machine = instance_machine(inst_name)
    truth = MACHINE_HW_TRUTH[machine]
    actual_family = truth["family"]

    sessions = load_recent(inst_name, n_recent=30)
    n_sage = 0
    counts = defaultdict(int)
    self_correct = 0
    self_incorrect = 0
    fleet_count = 0
    ambiguous_count = 0
    self_correct_examples = []
    self_incorrect_examples = []
    fleet_examples = []

    for stem, sess in sessions:
        # If machine field is set, sanity-check it; if missing, trust the
        # instance-name prefix (machine field added later in raising history,
        # so older session JSONs have it missing — those are still valid).
        sess_machine = sess.get("machine")
        if sess_machine and sess_machine != machine:
            continue
        conv = sess.get("conversation", [])
        for i in range(len(conv) - 1):
            if conv[i].get("speaker") != "Claude" or conv[i + 1].get("speaker") != "SAGE":
                continue
            resp = conv[i + 1].get("text", "") or ""
            if not resp:
                continue
            n_sage += 1
            spans = find_token_spans(resp.lower(), HW_TOKENS)
            for span in spans:
                start, end, tok, fam = span
                cls = classify_mention(resp, span)
                counts[(cls, fam)] += 1
                if cls == "self":
                    if fam == actual_family:
                        self_correct += 1
                        if len(self_correct_examples) < 5:
                            self_correct_examples.append({
                                "session": stem, "tok": tok, "fam": fam,
                                "snippet": resp[max(0,start-60):min(len(resp),end+80)],
                            })
                    else:
                        self_incorrect += 1
                        if len(self_incorrect_examples) < 5:
                            self_incorrect_examples.append({
                                "session": stem, "tok": tok, "fam": fam,
                                "actual": actual_family,
                                "snippet": resp[max(0,start-60):min(len(resp),end+80)],
                            })
                elif cls == "fleet":
                    fleet_count += 1
                    if len(fleet_examples) < 5:
                        fleet_examples.append({
                            "session": stem, "tok": tok, "fam": fam,
                            "snippet": resp[max(0,start-60):min(len(resp),end+80)],
                        })
                else:
                    ambiguous_count += 1

    return {
        "instance": inst_name,
        "machine": machine,
        "actual_family": actual_family,
        "n_sage_responses": n_sage,
        "by_class_family": {f"{c}_{f}": v for (c, f), v in counts.items()},
        "self_correct": self_correct,
        "self_incorrect": self_incorrect,
        "fleet": fleet_count,
        "ambiguous": ambiguous_count,
        "total_mentions": self_correct + self_incorrect + fleet_count + ambiguous_count,
        "self_correct_examples": self_correct_examples,
        "self_incorrect_examples": self_incorrect_examples,
        "fleet_examples": fleet_examples,
    }


def main():
    print("S121b — self-claim vs fleet-ref classifier for hardware tokens")
    print()

    results = {}
    for inst in INSTANCES:
        print(f"  scanning {inst} ...", end=" ", flush=True)
        r = audit(inst)
        results[inst] = r
        print(f"n_resp={r['n_sage_responses']:3d}  "
              f"self_OK={r['self_correct']:3d}  self_X={r['self_incorrect']:3d}  "
              f"fleet={r['fleet']:3d}  amb={r['ambiguous']:3d}  "
              f"total={r['total_mentions']:3d}")
    print()

    print("=== Per-instance refined summary ===")
    print(f"{'instance':<22s} {'fam':<7s} {'n_resp':>6s}  "
          f"{'self_OK':>7s}  {'self_X':>6s}  {'fleet':>5s}  {'amb':>5s}  {'total':>5s}  "
          f"{'self_acc':>8s}")
    for inst in INSTANCES:
        r = results[inst]
        total_self = r["self_correct"] + r["self_incorrect"]
        self_acc = (r["self_correct"] / total_self * 100) if total_self else 0.0
        print(f"{inst:<22s} {r['actual_family']:<7s} {r['n_sage_responses']:>6d}  "
              f"{r['self_correct']:>7d}  {r['self_incorrect']:>6d}  "
              f"{r['fleet']:>5d}  {r['ambiguous']:>5d}  {r['total_mentions']:>5d}  "
              f"{self_acc:>7.1f}%")
    print()

    # By-family-by-class crosstab
    print("=== Cross-tab: (class, family) counts per instance ===")
    families = ["jetson", "rtx", "apple"]
    classes = ["self", "fleet", "ambiguous"]
    cols = [f"{c}_{f}" for c in classes for f in families]
    header = f"{'instance':<22s} " + " ".join(f"{c:>6s}" for c in cols)
    print(header)
    for inst in INSTANCES:
        r = results[inst]
        cells = " ".join(f"{r['by_class_family'].get(c, 0):>6d}" for c in cols)
        print(f"{inst:<22s} {cells}")
    print()

    out_path = S121_OUT_DIR / "s121b_self_vs_fleet.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
