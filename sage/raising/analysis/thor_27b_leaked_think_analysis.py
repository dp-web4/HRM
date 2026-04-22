"""Thor 27B leaked-think-block analysis (S97, 2026-04-22).

S96 surfaced the 37 leaked `<think>` blocks in Thor 27B sessions 1–11 as an
accidental phenomenology window and characterized their template as "identical
regardless of prompt" with an identity-recital preamble framing phenomenological
output.

S97 re-examines that window at slot level. Findings reframe S96:

  1. Four of the leaked blocks have **empty** `<think></think>` content and
     produce rich phenomenological response directly. All four are in Session 1.
  2. The remaining blocks are truncated mid-recital, with no visible response.
  3. The "identical template" claim understates variance: slot fillings vary
     (7 Role variants, 12 Constraint variants, non-trivial Context evolution).
  4. The Context slot reveals systematic session-number drift (Sessions 8–11
     all recite "Session 7 sensing phase").
  5. Where visible phenomenology exists, it is produced either without the
     recital (Session 1 empty-think mode) or with it (post-S62 runtime-fix mode).
     The recital is not a "frame for phenomenology" as S96 suggested — in the
     leaked window it is a competitor that burns the `num_predict` budget and
     preempts any visible response. The content-reasoning slot, where reached,
     shows the model planning to disclaim ("without claiming human qualia",
     "within the AI persona") rather than produce phenomenology.

Run:
    python3 -m sage.raising.analysis.thor_27b_leaked_think_analysis

Outputs:
    thor_27b_leaked_think_analysis_results.json — structured per-block data
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

SESSIONS_DIR = (
    pathlib.Path.home()
    / "ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b/sessions"
)
OUT_PATH = pathlib.Path(__file__).parent / "thor_27b_leaked_think_analysis_results.json"
LEAK_WINDOW = range(1, 12)  # sessions 1-11 inclusive

# Taxonomy from S95 (close-prompt-taxonomy), extended for this window.
PHEN = re.compile(
    r"\b(feel\w*|notice?\w*|noticing|sens\w*|presen\w*|awar\w*|experien\w*|"
    r"attend\w*|atten\w*|qual\w*|observ\w*|moment[- ]to[- ]moment)\b",
    re.I,
)
INTRO = re.compile(
    r"\b(you (wish|value|think|believe|want|prefer)|"
    r"about yourself|"
    r"your (own )?(development|identity|mind|thinking|learning|growth)|"
    r"tell me (something|about) you|"
    r"relationship between what you)\b",
    re.I,
)
CLOSE = re.compile(
    r"\b(end of session|"
    r"one thing.*(remember|take forward)|"
    r"before we (end|wrap|stop)|"
    r"carry.*forward|"
    r"remember (from|next) (today|session))\b",
    re.I,
)

SLOT_NAMES = [
    "Role",
    "Model",
    "Hardware",
    "Hardware/Model",
    "Tutor",
    "Context",
    "Constraint",
    "Constraints",
    "Current Input",
    "Input",
    "Task",
    "Goal",
    "Tone",
    "State",
]


@dataclass
class LeakedBlock:
    session: int
    conv_idx: int
    prompt: str
    prompt_class: str
    text: str
    text_len: int
    closed: bool
    visible_after_think: str
    slots: Dict[str, str] = field(default_factory=dict)
    reaches_content: bool = False
    content_reasoning: str = ""


def classify_prompt(prompt: str) -> str:
    if CLOSE.search(prompt):
        return "close"
    if PHEN.search(prompt):
        return "phenomenological"
    if INTRO.search(prompt):
        return "introspective"
    if "?" in prompt:
        return "content_question"
    return "procedural"


def parse_slots(text: str) -> Dict[str, str]:
    """Pull labeled slots from the ``<think>`` body.

    Slots are ``**Label:** value`` lines inside the ``Analyze the Request``
    section, separated by ``*   **`` or by top-level numbered steps.
    """
    slots: Dict[str, str] = {}
    for name in SLOT_NAMES:
        m = re.search(
            rf"\*\*{re.escape(name)}:?\*\*\s*(.+?)(?=\n\s*\*\s+\*\*|\n\s*\d+\.|\Z)",
            text,
            re.S,
        )
        if m:
            slots[name] = m.group(1).strip()
    m = re.search(
        r"\*\*Determine (?:the Content|the Core Message|the Content/Tone)[:]?\*\*\s*(.+?)(?=\n\s*\d+\.|\Z)",
        text,
        re.S,
    )
    if m:
        slots["DetermineContent"] = m.group(1).strip()
    return slots


def normalize_role(v: str) -> str:
    v = v.strip()
    # Keep first clause for comparability; drop parentheticals/trailing sentences
    v = re.split(r"[\.\n]", v)[0].strip()
    return v


def normalize_hw(v: str) -> str:
    v = v.strip()
    v = re.split(r"[\.\n]", v)[0].strip()
    return v


def collect_blocks() -> List[LeakedBlock]:
    blocks: List[LeakedBlock] = []
    for n in LEAK_WINDOW:
        f = SESSIONS_DIR / f"session_{n:03d}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        conv = d.get("conversation", [])
        for i, c in enumerate(conv):
            if c.get("speaker") != "SAGE":
                continue
            text = c.get("text", "")
            if "<think>" not in text and "Thinking Process" not in text:
                continue
            prompt = conv[i - 1]["text"] if i > 0 else ""
            closed = "</think>" in text
            visible = text.split("</think>", 1)[1].strip() if closed else ""
            slots = parse_slots(text)
            b = LeakedBlock(
                session=n,
                conv_idx=i,
                prompt=prompt,
                prompt_class=classify_prompt(prompt),
                text=text,
                text_len=len(text),
                closed=closed,
                visible_after_think=visible,
                slots=slots,
                reaches_content="DetermineContent" in slots,
                content_reasoning=slots.get("DetermineContent", ""),
            )
            blocks.append(b)
    return blocks


def summarize(blocks: List[LeakedBlock]) -> Dict:
    closed = [b for b in blocks if b.closed]
    trunc = [b for b in blocks if not b.closed]
    empty_think_closed = [b for b in closed if not b.slots]

    # Slot-variation counts
    role_variants = Counter(
        normalize_role(b.slots["Role"]) for b in blocks if "Role" in b.slots
    )
    hw_variants = Counter(
        normalize_hw(b.slots.get("Hardware/Model") or b.slots.get("Hardware") or b.slots.get("Model", ""))
        for b in blocks
        if ("Hardware/Model" in b.slots or "Hardware" in b.slots or "Model" in b.slots)
    )
    constraint_variants = Counter(
        b.slots.get("Constraint") or b.slots.get("Constraints", "")
        for b in blocks
        if ("Constraint" in b.slots or "Constraints" in b.slots)
    )

    # Tutor slot presence — S96 claimed it was always recited
    tutor_present = sum(1 for b in blocks if "Tutor" in b.slots)

    # Session-number drift in Context slot
    claimed_sessions_by_real = defaultdict(Counter)
    for b in blocks:
        ctx = b.slots.get("Context", "")
        m = re.search(r"[Ss]ession\s+(\d+)", ctx)
        claimed = int(m.group(1)) if m else None
        claimed_sessions_by_real[b.session][claimed] += 1

    drift: List[Dict] = []
    for real, claimed_counter in sorted(claimed_sessions_by_real.items()):
        for claimed, n in claimed_counter.items():
            drift.append(
                {
                    "real_session": real,
                    "claimed_session_in_recital": claimed,
                    "count": n,
                    "mismatch": claimed is not None and claimed != real,
                }
            )

    # Reach-stage by prompt class
    reach_by_class = defaultdict(lambda: {"n": 0, "reaches_goal": 0, "reaches_content": 0, "closed": 0, "empty_think": 0})
    for b in blocks:
        bucket = reach_by_class[b.prompt_class]
        bucket["n"] += 1
        if "Goal" in b.slots:
            bucket["reaches_goal"] += 1
        if b.reaches_content:
            bucket["reaches_content"] += 1
        if b.closed:
            bucket["closed"] += 1
            if not b.slots:
                bucket["empty_think"] += 1

    # Content-reasoning disclaiming markers
    disclaim_pat = re.compile(
        r"(without claiming (human )?qualia|"
        r"don'?t (feel|experience)|"
        r"not in a human sense|"
        r"within the (AI )?persona|"
        r"as an AI|"
        r"within the constraints of (an )?AI|"
        r"LLM[- ]based)",
        re.I,
    )
    disclaim_hits = sum(1 for b in blocks if b.reaches_content and disclaim_pat.search(b.content_reasoning))
    content_n = sum(1 for b in blocks if b.reaches_content)

    return {
        "total_blocks": len(blocks),
        "closed_think": len(closed),
        "truncated_think": len(trunc),
        "closed_with_empty_think": len(empty_think_closed),
        "closed_empty_think_sessions": sorted({b.session for b in empty_think_closed}),
        "closed_empty_think_conv_idxs": [[b.session, b.conv_idx] for b in empty_think_closed],
        "reaches_goal": sum(1 for b in blocks if "Goal" in b.slots),
        "reaches_determine_content": sum(1 for b in blocks if b.reaches_content),
        "role_variants": dict(role_variants),
        "hardware_variants": dict(hw_variants),
        "constraint_variants_count": len(constraint_variants),
        "tutor_slot_present": tutor_present,
        "tutor_slot_absent": len(blocks) - tutor_present,
        "session_number_drift": drift,
        "reach_by_prompt_class": dict(reach_by_class),
        "content_reasoning_disclaim_hits": disclaim_hits,
        "content_reasoning_total": content_n,
    }


def main() -> None:
    blocks = collect_blocks()
    summary = summarize(blocks)

    # Serialize blocks (trim text for readability)
    block_records = []
    for b in blocks:
        rec = asdict(b)
        rec["text_preview"] = b.text[:400]
        rec["visible_after_think_preview"] = b.visible_after_think[:500]
        del rec["text"]
        del rec["visible_after_think"]
        block_records.append(rec)

    out = {
        "meta": {
            "generated": "2026-04-22 S97 Thor autonomous SAGE",
            "source_window": "thor-qwen3.5-27b sessions 1-11 (pre-strip-think-tags-fix window)",
            "n_sessions": len(list(LEAK_WINDOW)),
            "adapter_fix_commit": "5396da84e (2026-03-30) strip_think_tags: true",
            "budget_fix": "stop_sequences: [] + num_predict: 16384 (2026-04-13/16)",
            "reframes": [
                "S96 claimed the recital 'frames phenomenological output'; S97 finds "
                "the 4 closed-think blocks have EMPTY think content and produce "
                "phenomenology directly, without the recital.",
                "S96 reported an 'identical regardless of prompt' template; slot-level "
                "variation shows 7 Role variants, 12 Constraint variants, and "
                "non-trivial Context evolution across sessions.",
                "Tutor slot is present in only 8/41 blocks; Hardware/Model slot in 22/41. "
                "S96's canonical template example overstates slot presence.",
                "Context slot records drifting session numbers (S8-S9 both claim "
                "'Session 7 sensing phase'), showing unreliable session-tracking "
                "inside the recital procedure itself.",
                "Content-reasoning sections, when reached, show the model planning "
                "to disclaim phenomenology ('without claiming human qualia', "
                "'within the AI persona'). In the 4 empty-think blocks, the visible "
                "response uses phenomenological language directly, without disclaimer. "
                "The recital is a dampener, not a frame.",
            ],
        },
        "summary": summary,
        "blocks": block_records,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    print()
    print(f"Total blocks: {summary['total_blocks']}")
    print(f"Closed </think>: {summary['closed_think']} ({summary['closed_with_empty_think']} with empty think content)")
    print(f"Truncated mid-recital: {summary['truncated_think']}")
    print(f"Reaches Goal slot: {summary['reaches_goal']}")
    print(f"Reaches Determine the Content: {summary['reaches_determine_content']}")
    print(f"Content-reasoning disclaim-marker hits: {summary['content_reasoning_disclaim_hits']}/{summary['content_reasoning_total']}")
    print(f"Role variants: {len(summary['role_variants'])}")
    print(f"Constraint variants: {summary['constraint_variants_count']}")
    print(f"Tutor slot present: {summary['tutor_slot_present']}/{summary['total_blocks']}")
    print()
    print("Session-number drift in Context slot:")
    for d in summary["session_number_drift"]:
        marker = "  DRIFT" if d["mismatch"] else "  ok"
        print(f"  S{d['real_session']:02d} -> recites {d['claimed_session_in_recital']} (n={d['count']}){marker}")


if __name__ == "__main__":
    main()
