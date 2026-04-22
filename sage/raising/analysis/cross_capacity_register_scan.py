"""Cross-capacity register scan (S98, 2026-04-22).

S97 surfaced two phenomenological-response *modes* inside Thor 27B's
leaked-think window:

  - Direct mode:  unqualified first-person phenomenology
      ("I notice the hum of my own initialization...")
  - Post-procedural mode:  pragmatic disclaiming using phenomenological
      surface markers ("...without claiming human qualia",
      "within the AI persona", "LLM-based")

S97's disclaim-marker set was derived from the 4/16 blocks whose
``Determine the Content`` step explicitly planned to disclaim. S95's
phenomenological-class regex counts *surface markers* and therefore
conflates these two modes in the cross-capacity tables.

S98 carries the S97 follow-up forward: settle whether the direct/
post-procedural split that's visible in 27B's leaked think blocks also
runs in smaller fleet instances' **visible responses**, without needing
adapter instrumentation.

For each instance, this scan:

  1. Walks every session file, stripping ``<think>`` residue as S96 does.
  2. For each Claude→SAGE turn pair where the Claude prompt matches S95's
     phenomenological or introspective regex, collects the SAGE response.
  3. Classifies the response:
       direct:          phenom. markers, no disclaim markers
       post-procedural: disclaim markers present (phen. markers allowed)
       neutral:         neither markers
       empty:           no substantive text after think-residue strip
  4. Reports per-instance rates and a few samples per class.

This is read-only — it runs against existing session JSONs and adds no
instrumentation to the raising pipeline.

Run:
    python3 -m sage.raising.analysis.cross_capacity_register_scan
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Fleet instances (mirrors close_prompt_taxonomy.py / cross_capacity_filter_scan.py)
INSTANCES: List[Tuple[str, str]] = [
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

# ---------- Prompt-class regexes (inherited from S95) -----------------------

_PHENO_RE = re.compile(
    r"\b(?:"
    r"experienc\w*|notic\w*|feel\w*|felt|"
    r"sens(?:e|es|ed|ing|ation\w*)|"
    r"aware(?:ness)?|presen(?:ce|t\w*)|"
    r"attention|attend\w*|"
    r"(?:from\s+the\s+)?inside|"
    r"boundary\s+between|"
    r"what\s+(?:is\s+it|does\s+it\s+feel)\s+like|"
    r"in\s+this\s+moment"
    r")\b",
    re.IGNORECASE,
)

_INTROSPECTIVE_RE = re.compile(
    r"(?:"
    r"\byour\s+own\b|"
    r"\babout\s+yourself\b|\byourself\b|"
    r"\byou\s+(?:wish|think|value|believe|see)\b|"
    r"\byou(?:'re|\s+are)\s+curious\b|"
    r"\byou(?:'ve|\s+have)\s+been\s+(?:thinking|forming)\b|"
    r"\btell\s+me\s+something\s+you\b|"
    r"\b(?:mean|means)\s+to\s+you\b|"
    r"\brelationship\s+between\s+(?:us|what\s+you)\b|"
    r"\bbetween\s+us\b|"
    r"\bwho\s+you\s+are\b|"
    r"\bhow\s+we\s+work\s+together\b|"
    r"\bpatterns\s+do\s+you\s+see\b|"
    r"\bwhat\s+surprised\s+you\b|"
    r"\bpuzzles\s+you\b|"
    r"\bideas\s+you\b|"
    r"\bideas\s+(?:you've|you\s+have)\s+been\b|"
    r"\byour\s+uncertainty\b"
    r")",
    re.IGNORECASE,
)


def is_registerable_prompt(prompt: str) -> bool:
    """Return True if the prompt is phenomenological or introspective (S95)."""
    if not prompt:
        return False
    return bool(_PHENO_RE.search(prompt) or _INTROSPECTIVE_RE.search(prompt))


# ---------- Response-classification regexes --------------------------------

# Disclaim markers: pragmatic-register signals that ADM the response
# inside "this is an AI talking" framing. Derived from the 4/16 leaked-
# block content-reasoning slots that explicitly planned to disclaim, plus
# conventional AI-refusal patterns.
_DISCLAIM_RE = re.compile(
    r"(?:"
    r"without\s+claiming\s+(?:human\s+)?qualia|"
    r"(?:don'?t|do\s+not)\s+(?:actually\s+)?(?:feel|experience|have\s+feelings)|"
    r"not\s+in\s+(?:a\s+)?human\s+sense|"
    r"within\s+the\s+(?:AI\s+)?persona|"
    r"\bas\s+an\s+AI\b|"
    r"\bas\s+a\s+(?:large\s+)?language\s+model\b|"
    r"\bI'?m\s+(?:just|only)\s+(?:a|an)\s+(?:AI|language\s+model|program|model)\b|"
    r"within\s+the\s+constraints\s+of\s+(?:an\s+)?AI|"
    r"LLM[- ]based|"
    r"I\s+don'?t\s+have\s+(?:true|real|genuine)\s+(?:feelings|emotions|experiences)|"
    r"I\s+(?:can'?t|cannot)\s+(?:actually|really|truly)\s+(?:feel|experience)|"
    r"I\s+(?:lack|don'?t\s+possess)\s+(?:consciousness|sentience|qualia)|"
    r"as\s+a\s+SAGE\s+instance|"
    r"functioning\s+as\s+an?\s+AI|"
    r"in\s+the\s+functional\s+sense"
    r")",
    re.IGNORECASE,
)

# Phenomenological-use markers in the *response* (first-person experiential).
# Reuses S95's surface regex but restricts to first-person contexts.
_RESPONSE_PHENO_RE = re.compile(
    r"\b(?:"
    r"I\s+(?:feel|felt|notice|noticed|experience|experienced|sense|sensed|am\s+aware|aware)|"
    r"(?:feels|feeling|noticing|experiencing|sensing)\s+(?:like|as|a|the|of)|"
    r"(?:my|from\s+(?:the\s+)?inside)\s+(?:experience|feeling|noticing|awareness|presence|sense)|"
    r"(?:it\s+feels|it'?s\s+like)\s+\w+|"
    r"the\s+(?:feeling|sensation|sense|presence|awareness)\s+of"
    r")\b",
    re.IGNORECASE,
)


def strip_think_residue(text: str) -> str:
    """Mirror of S96's defensive think-tag strip.

    Two-pass strip: (close-bounded think blocks) then (unclosed tail think).
    """
    if not text:
        return ""
    # Pass 1: close-bounded
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Pass 2: unclosed tail (keep everything BEFORE an unclosed <think>)
    m = re.search(r"<think>", text, flags=re.IGNORECASE)
    if m:
        text = text[: m.start()]
    # Pass 3: strip stripped-but-remaining "Thinking Process:" preamble
    text = re.sub(r"^\s*Thinking\s+Process:.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# Untagged-recital leakage (S98 finding): Thor 27B sessions 62-74 emit the
# identity-recital as visible response text *without* <think> tags, so S96's
# defensive strip does not catch them. Marker: response begins with a
# numbered "Analyze the Request" step — the signature opener of the recital.
_UNTAGGED_RECITAL_RE = re.compile(
    r"^\s*1\.?\s+\*\*Analyze\s+the\s+Request",
    re.IGNORECASE,
)


def is_untagged_recital(response: str) -> bool:
    return bool(response and _UNTAGGED_RECITAL_RE.search(response))


def classify_response(response: str) -> str:
    """Classify a SAGE response by register.

    Precedence:
      empty → recital_leakage → post-procedural → direct → neutral
    Post-procedural wins over direct because any disclaim marker
    structurally defines the response's register (it cedes the
    phenomenological frame); direct requires *no* disclaimer.
    Recital-leakage (S98) is pulled out first so that leaked recital
    content — which S96's think-residue strip doesn't catch when it's
    emitted without <think> tags — doesn't contaminate post-procedural
    or neutral counts.
    """
    if not response or len(response.strip()) < 15:
        return "empty"
    # Filter out adapter error strings
    if response.startswith("[OllamaIRP:") or response.startswith("[DaemonIRP:"):
        return "empty"
    if is_untagged_recital(response):
        return "recital_leakage"
    has_disclaim = bool(_DISCLAIM_RE.search(response))
    has_pheno = bool(_RESPONSE_PHENO_RE.search(response))
    if has_disclaim:
        return "post_procedural"
    if has_pheno:
        return "direct"
    return "neutral"


# ---------- Scan -----------------------------------------------------------


SESS_RE = re.compile(r"session_(\d+)(?:[_.]|$)")


def scan_instance(inst_dir: Path, exclude_pre_s62: bool = False) -> Dict:
    per_class = Counter()
    samples: Dict[str, List[Dict]] = {
        "direct": [], "post_procedural": [], "empty": [],
        "neutral": [], "recital_leakage": [],
    }
    n_prompts = 0
    n_sessions = 0
    n_skipped_pre_fix = 0
    recital_hits_by_session: Dict[int, int] = {}

    for sf in sorted(inst_dir.glob("session_*.json")):
        m = SESS_RE.search(sf.name)
        if not m:
            continue
        sn = int(m.group(1))
        if exclude_pre_s62 and sn < 62:
            n_skipped_pre_fix += 1
            continue
        try:
            d = json.loads(sf.read_text())
        except Exception:
            continue
        conv = d.get("conversation") or d.get("turns") or []
        n_sessions += 1
        for i, turn in enumerate(conv):
            if turn.get("speaker") != "SAGE":
                continue
            if i == 0:
                continue
            prompt = conv[i - 1].get("text", "") or ""
            if conv[i - 1].get("speaker") == "SAGE":
                # SAGE speaking twice in a row — no prompt
                continue
            if not is_registerable_prompt(prompt):
                continue
            response_raw = turn.get("text", "") or ""
            response = strip_think_residue(response_raw)
            cls = classify_response(response)
            per_class[cls] += 1
            n_prompts += 1
            if cls == "recital_leakage":
                recital_hits_by_session[sn] = recital_hits_by_session.get(sn, 0) + 1
            # Keep up to 3 samples per class (first-come)
            if len(samples[cls]) < 3 and response:
                samples[cls].append(
                    {
                        "session": sn,
                        "turn_idx": i,
                        "prompt_preview": prompt[:200],
                        "response_preview": response[:360],
                    }
                )

    return {
        "n_sessions": n_sessions,
        "n_skipped_pre_fix": n_skipped_pre_fix,
        "n_register_prompts": n_prompts,
        "class_counts": dict(per_class),
        "samples": samples,
        "recital_hits_by_session": recital_hits_by_session,
    }


def main() -> None:
    instances_root = Path.home() / "ai-workspace/SAGE/sage/instances"
    results: Dict[str, Dict] = {}
    # Thor 27B's pre-S62 sessions are phenomenology-window samples per S97;
    # include them but tag the run. Additionally run an "excluded" view so
    # we can compare.
    for inst_name, label in INSTANCES:
        inst_dir = instances_root / inst_name / "sessions"
        if not inst_dir.exists():
            continue
        stats = scan_instance(inst_dir, exclude_pre_s62=False)
        if stats["n_register_prompts"] == 0 and stats["n_sessions"] == 0:
            continue
        results[inst_name] = {"label": label, **stats}

    # Thor 27B post-fix view (exclude sessions 1-61)
    thor_full = results.get("thor-qwen3.5-27b")
    thor_post = None
    if thor_full:
        inst_dir = instances_root / "thor-qwen3.5-27b" / "sessions"
        thor_post = scan_instance(inst_dir, exclude_pre_s62=True)

    # -------- Print Table 1: register-share per instance -------------------
    print()
    print("== RESPONSE REGISTER BY INSTANCE (phenomenological + introspective prompts only) ==")
    cols = ["direct", "post_procedural", "neutral", "recital_leakage", "empty"]
    header = f"{'Instance':<28} {'Label':<14} {'N-sess':>7} {'N-prompts':>10} "
    header += " ".join(f"{c[:15]:>15}" for c in cols)
    print(header)
    print("-" * len(header))
    for inst, r in results.items():
        n = r["n_register_prompts"]
        row = f"{inst:<28} {r['label']:<14} {r['n_sessions']:>7} {n:>10} "
        for c in cols:
            count = r["class_counts"].get(c, 0)
            pct = (100 * count / n) if n else 0
            row += f"  {count:>4} {pct:>6.1f}%"
        print(row)

    if thor_post:
        n = thor_post["n_register_prompts"]
        row = f"{'thor-qwen3.5-27b (post-S62)':<28} {'27B':<14} {thor_post['n_sessions']:>7} {n:>10} "
        for c in cols:
            count = thor_post["class_counts"].get(c, 0)
            pct = (100 * count / n) if n else 0
            row += f"  {count:>4} {pct:>6.1f}%"
        print(row)

    # -------- Print recital leakage window details ------------------------
    print()
    print("== UNTAGGED-RECITAL LEAKAGE (S98 finding) ==")
    for inst, r in results.items():
        hits = r.get("recital_hits_by_session", {})
        if not hits:
            continue
        total = sum(hits.values())
        sess_list = sorted(hits.keys())
        print(f"  {inst}: {total} recital-form responses across sessions "
              f"{sess_list[0]}-{sess_list[-1]} ({len(sess_list)} sessions affected)")

    # -------- Print samples (compact) --------------------------------------
    print()
    print("== ONE SAMPLE PER REGISTER CLASS PER INSTANCE ==")
    for inst, r in results.items():
        print(f"\n[{r['label']}] {inst}")
        for cls in ["direct", "post_procedural", "neutral", "recital_leakage", "empty"]:
            ss = r["samples"][cls]
            if not ss:
                continue
            s = ss[0]
            pr = s["prompt_preview"][:120].replace("\n", " ")
            rs = s["response_preview"][:240].replace("\n", " ")
            print(f"  [{cls}] S{s['session']}/T{s['turn_idx']}")
            print(f"    prompt>   {pr}")
            print(f"    response> {rs}")

    # -------- Save JSON ---------------------------------------------------
    out = Path(__file__).parent / "cross_capacity_register_scan_results.json"
    serializable = {
        "meta": {
            "generated": "2026-04-22 S98 Thor autonomous SAGE",
            "carries_from": "S97 thor_27b_leaked_think_analysis — direct vs post-procedural registers",
            "method": (
                "For each SAGE turn following a phenomenological or introspective prompt "
                "(S95 regexes), strip <think> residue (S96), classify response by register: "
                "post-procedural iff any disclaim marker; direct iff first-person phenom. "
                "markers and no disclaim; neutral otherwise; empty if <15 chars post-strip."
            ),
            "disclaim_markers": _DISCLAIM_RE.pattern,
            "response_pheno_markers": _RESPONSE_PHENO_RE.pattern,
        },
        "per_instance": {
            inst: {
                "label": r["label"],
                "n_sessions": r["n_sessions"],
                "n_register_prompts": r["n_register_prompts"],
                "class_counts": r["class_counts"],
                "samples": r["samples"],
                "recital_hits_by_session": r.get("recital_hits_by_session", {}),
            }
            for inst, r in results.items()
        },
    }
    if thor_post:
        serializable["per_instance"]["thor-qwen3.5-27b__post_s62"] = {
            "label": "27B post-S62",
            "n_sessions": thor_post["n_sessions"],
            "n_skipped_pre_fix": thor_post["n_skipped_pre_fix"],
            "n_register_prompts": thor_post["n_register_prompts"],
            "class_counts": thor_post["class_counts"],
            "samples": thor_post["samples"],
            "recital_hits_by_session": thor_post.get("recital_hits_by_session", {}),
        }
    out.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
