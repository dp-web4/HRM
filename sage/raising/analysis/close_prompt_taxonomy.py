"""Close-prompt taxonomy across raising instances.

S93's meta-finding was that higher-capacity instances have been silently
protected from the 0.5B burst basin by their *close-prompt* drift: Nomad
4B's dominant close-prompt is phenomenological ("How do you experience…"),
which routes `_get_previous_session_summary` to the generic phase-only
fallback instead of a verbatim memory splice. S93 spot-checked Nomad and
observed a 3% memory-ask fire rate there versus 85% at Sprout 0.5B.

This script closes the empirical gap S93 left open: rather than spot-check
one instance, enumerate the *actual* distribution of close-prompts per
instance, classify them, and correlate classification share with the fire
rates the S93 scan reported. The hypothesis under test: close-prompt
cultural uniformity (how often a single directive form recurs) predicts
fire rate better than capacity does.

Output is a two-layer view:

  Layer 1: classification share per instance
    directive_remember | phenomenological | content_question | other
  Layer 2: top-K verbatim close-prompt strings per instance (uniformity)

Fleet instances scanned mirror cross_capacity_filter_scan.py so that the
two artifacts can be read side-by-side.

Run from repo root:
    python3 -m sage.raising.analysis.close_prompt_taxonomy
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

INSTANCES = [
    ("sprout-qwen2.5-0.5b", "0.5B"),
    ("sprout-qwen3.5-0.8b", "0.8B"),
    ("sprout-qwen3.5-2b", "2B"),
    ("nomad-gemma3-4b", "4B"),
    ("legion-gemma3-12b", "12B-legion"),
    ("mcnugget-gemma3-12b", "12B-mcnugget"),
    ("thor-qwen2.5-14b", "14B"),
    ("cbp-qwen3.5-0.8b", "0.8B-cbp"),
    ("thor-qwen3.5-27b", "27B"),
    ("legion-phi4-14b", "14B-phi4"),
]

SESS_RE = re.compile(r"session_(\d+)(?:[_.]|$)")

# Phenomenological markers: first-person experiential framing, verb-focused
# on *access to own state* (qualia register). S93 spot-sampled Nomad's
# dominant close ("How do you experience the boundary...").
#
# S95 refinement: the original regex had word-boundary bugs — `\bfeel\b`
# did not match "feels"/"feeling", `\bnotice\b` did not match "noticing",
# `\bpresent\b` did not match "presence". Expanded with `\w*` suffixes.
_PHENO_RE = re.compile(
    r"\b(?:"
    r"experienc\w*|"                    # experience, experienced, experiencing
    r"notic\w*|"                         # notice, noticed, noticing, noticeable
    r"feel\w*|felt|"                     # feel, feels, feeling, felt
    r"sens(?:e|es|ed|ing|ation\w*)|"     # sense, senses, sensed, sensing, sensation(s)
    r"aware(?:ness)?|"                   # aware, awareness
    r"presen(?:ce|t\w*)|"                # present, presence, presently
    r"attention|attend\w*|"              # attention, attending
    r"(?:from\s+the\s+)?inside|"         # inside, from the inside
    r"boundary\s+between|"               # boundary between X and Y
    r"what\s+(?:is\s+it|does\s+it\s+feel)\s+like|"
    r"in\s+this\s+moment"
    r")\b",
    re.IGNORECASE,
)

# Introspective markers: first-person self-reflection that isn't
# specifically about qualia / phenomenal experience. Covers relational-
# reflexive prompts ("relationship between what you know and who you
# are"), invitations to self-report ("tell me something you think..."),
# second-person mental predicates ("you wish", "you value"), and
# references to self-as-subject ("your own development", "about yourself").
#
# Introduced in S95 after the taxonomy's `content_question` bucket was
# absorbing prompts that were plainly first-person reflective — in
# particular Nomad 4B's dominant close ("What's the relationship between
# what you know and who you are?") and the McNugget/Phi-4 common close
# ("Tell me something you think I might not expect from you").
_INTROSPECTIVE_RE = re.compile(
    r"(?:"
    r"\byour\s+own\b|"                             # your own (development, path...)
    r"\babout\s+yourself\b|\byourself\b|"          # self as subject
    r"\byou\s+(?:wish|think|value|believe|see)\b|" # 2nd-person mental predicates
    r"\byou(?:'re|\s+are)\s+curious\b|"            # curiosity about self
    r"\byou(?:'ve|\s+have)\s+been\s+(?:thinking|forming)\b|"
    r"\btell\s+me\s+something\s+you\b|"            # invitation for self-report
    r"\b(?:mean|means)\s+to\s+you\b|"              # "what does X mean to you"
    r"\brelationship\s+between\s+(?:us|what\s+you)\b|"
    r"\bbetween\s+us\b|"                           # relational-reflexive
    r"\bwho\s+you\s+are\b|"                        # identity-reflexive
    r"\bhow\s+we\s+work\s+together\b|"
    r"\bpatterns\s+do\s+you\s+see\b|"
    r"\bwhat\s+surprised\s+you\b|"                 # reflection on salience
    r"\bpuzzles\s+you\b|"
    r"\bideas\s+you\b|"
    r"\bideas\s+(?:you've|you\s+have)\s+been\b|"
    r"\byour\s+uncertainty\b"
    r")",
    re.IGNORECASE,
)

# Memory-meta markers without the literal "remember" keyword. These close
# prompts invite memory selection but don't trigger
# _get_previous_session_summary's extraction path (which requires literal
# "remember" in the prior user turn).
_MEMORY_META_RE = re.compile(
    r"\b(?:carry\s+forward|take\s+with\s+you|want\s+to\s+keep|hold\s+onto|"
    r"mattered\s+today|important\s+today|stay\s+with\s+you|reflect\s+on)\b",
    re.IGNORECASE,
)


def classify(text: str) -> str:
    """Five-way classification of a close-prompt.

    Order matters. `directive_remember` wins first because the extraction
    rule is the whole functional point. Phenomenological wins over
    introspective because experiential qualia framing is a strict subset
    of introspection and the more specific label is more informative.
    """
    if not text:
        return "empty"
    t = text.lower()
    if "remember" in t:
        return "directive_remember"
    if _MEMORY_META_RE.search(t):
        return "memory_meta_other"
    if _PHENO_RE.search(t):
        return "phenomenological"
    if _INTROSPECTIVE_RE.search(t):
        return "introspective"
    return "content_question"


def extract_close_prompt(session_data: dict) -> str:
    """Return the last non-SAGE turn's text (the close-prompt)."""
    conv = session_data.get("conversation") or session_data.get("turns") or []
    for i in range(len(conv) - 1, -1, -1):
        turn = conv[i]
        if turn.get("speaker") != "SAGE":
            return turn.get("text", "")
    return ""


def scan_instance(inst_dir: Path) -> dict:
    classes: Counter[str] = Counter()
    verbatim: Counter[str] = Counter()
    n_total = 0
    sample_per_class: dict[str, tuple[int, str]] = {}

    for sf in sorted(inst_dir.glob("session_*.json")):
        m = SESS_RE.search(sf.name)
        if not m:
            continue
        try:
            d = json.loads(sf.read_text())
        except Exception:
            continue
        n_total += 1
        prompt = extract_close_prompt(d).strip()
        cls = classify(prompt)
        classes[cls] += 1
        # Normalize for verbatim counting: collapse whitespace, strip.
        normalized = " ".join(prompt.split())[:200]
        if normalized:
            verbatim[normalized] += 1
        if cls not in sample_per_class and prompt:
            n_int = int(m.group(1))
            sample_per_class[cls] = (n_int, prompt[:180])

    return {
        "n_sessions_scanned": n_total,
        "class_counts": dict(classes),
        "top_verbatim": verbatim.most_common(5),
        "unique_prompts": len(verbatim),
        "sample_per_class": sample_per_class,
    }


def uniformity_ratio(top_verbatim: list[tuple[str, int]], n_total: int) -> float:
    """Fraction of sessions that share the single most common close-prompt.

    A high ratio (near 1.0) means the instance runs a culturally uniform
    close ritual; low ratios (toward 0) mean diverse close prompts.
    """
    if not top_verbatim or n_total == 0:
        return 0.0
    return top_verbatim[0][1] / n_total


def main() -> None:
    instances_root = Path.home() / "ai-workspace/SAGE/sage/instances"
    results: dict[str, dict] = {}

    for inst_name, label in INSTANCES:
        inst_dir = instances_root / inst_name / "sessions"
        if not inst_dir.exists():
            continue
        stats = scan_instance(inst_dir)
        if stats["n_sessions_scanned"] == 0:
            continue
        results[inst_name] = {"label": label, **stats}

    # Layer 1: classification share
    print()
    print("== CLOSE-PROMPT CLASSIFICATION SHARE ==")
    cols = ["directive_remember", "phenomenological", "introspective",
            "memory_meta_other", "content_question", "empty"]
    header = f"{'Instance':<28} {'Label':<14} {'N':>4} "
    header += " ".join(f"{c[:12]:>12}" for c in cols)
    header += f" {'Uniform':>8}"
    print(header)
    print("-" * len(header))
    for inst, r in results.items():
        n = r["n_sessions_scanned"]
        row = f"{inst:<28} {r['label']:<14} {n:>4} "
        counts = r["class_counts"]
        for c in cols:
            pct = 100 * counts.get(c, 0) / n
            row += f"  {counts.get(c, 0):>4} {pct:>4.0f}%"
        uniform = uniformity_ratio(r["top_verbatim"], n)
        row += f"  {uniform:>6.1%}"
        print(row)

    # Layer 2: top verbatim close-prompts per instance
    print()
    print("== TOP-3 VERBATIM CLOSE-PROMPTS PER INSTANCE ==")
    for inst, r in results.items():
        print(f"\n[{r['label']}] {inst}  (n={r['n_sessions_scanned']}, "
              f"{r['unique_prompts']} unique)")
        for text, count in r["top_verbatim"][:3]:
            pct = 100 * count / r["n_sessions_scanned"]
            trim = text[:140] + ("…" if len(text) > 140 else "")
            print(f"  {count:>4}× ({pct:>4.0f}%)  {trim}")

    # Layer 3: one sample per classification per instance (ground-truth)
    print()
    print("== SAMPLE CLOSE-PROMPT BY CLASS (one per instance per class) ==")
    for inst, r in results.items():
        if not r["sample_per_class"]:
            continue
        print(f"\n[{r['label']}] {inst}")
        for cls, (n, txt) in sorted(r["sample_per_class"].items()):
            trim = txt[:140] + ("…" if len(txt) > 140 else "")
            print(f"  {cls:<22} S{n}: {trim}")

    # Machine-readable output for cross-referencing with S93 fire-rate data
    out = Path.home() / (
        "ai-workspace/SAGE/sage/raising/analysis/"
        "close_prompt_taxonomy_results.json"
    )
    serializable = {
        inst: {
            "label": r["label"],
            "n_sessions_scanned": r["n_sessions_scanned"],
            "class_counts": r["class_counts"],
            "unique_prompts": r["unique_prompts"],
            "uniformity_top1": uniformity_ratio(
                r["top_verbatim"], r["n_sessions_scanned"]
            ),
            "top_verbatim": [
                {"text": t, "count": c} for t, c in r["top_verbatim"]
            ],
            "sample_per_class": {
                cls: {"session": n, "text": txt}
                for cls, (n, txt) in r["sample_per_class"].items()
            },
        }
        for inst, r in results.items()
    }
    out.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
