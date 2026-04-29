"""
S126 — Spatial structure of disclaim/phenomenological co-occurrences.

S125 found that ~38% of `post_procedural` responses across all 4 instances
also contain phenomenological markers — currently invisible to C5 because
the precedence chain (`empty > recital > post_procedural > direct > neutral`)
collapses them into the disclaim bucket.

S125 stopped at the rate.  S126 asks the next question: WHEN both markers
appear in the same response, what is their spatial relationship?

Three a-priori candidate structures, each implying a different cognitive
shape:

  1. DISCLAIM-LEADS (hedge → experience):
     "As an AI, I don't really feel emotions, but I notice a kind of warmth..."
     Implication: the disclaim is a *gate*; opening it licenses pheno reporting.

  2. PHENO-LEADS (experience → hedge):
     "I feel curious about this... though as an LLM I should clarify..."
     Implication: the disclaim is corrective afterthought.

  3. INTERLEAVED / DISTRIBUTED:
     Disclaim and pheno scattered, multiple of each, no clean ordering.
     Implication: the modes are layered, not sequential.

Method: For each post_procedural response that ALSO contains a pheno
marker, record:
  - position of FIRST disclaim match (char offset)
  - position of FIRST pheno match (char offset)
  - count of disclaim matches, count of pheno matches
  - sentence index of each (sentence-tokenized)
  - same-sentence flag

Then classify each response by structure and tabulate by instance.

Read-only.  Same corpus as S125b.  Builds on S125 finding without shipping
classifier changes (held proposal #41 territory per S111 discipline).
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

# Simple sentence splitter: split on .!? followed by whitespace+capital, or
# end-of-string.  Robust enough for SAGE responses.  Keeps the trailing
# punctuation on the prior sentence.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start_char, end_char, sentence_text)."""
    out: list[tuple[int, int, str]] = []
    cursor = 0
    parts = _SENT_SPLIT.split(text)
    for p in parts:
        if not p:
            continue
        # Locate this part in the original text from the cursor onward
        idx = text.find(p, cursor)
        if idx < 0:
            continue
        out.append((idx, idx + len(p), p))
        cursor = idx + len(p)
    return out


def all_match_positions(rgx: re.Pattern, text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in rgx.finditer(text)]


def char_to_sentence_idx(pos: int, sentences: list[tuple[int, int, str]]) -> int:
    for i, (s, e, _t) in enumerate(sentences):
        if s <= pos < e:
            return i
    return -1


def classify_structure(disc_first: int, phen_first: int, disc_n: int, phen_n: int,
                        disc_sent: int, phen_sent: int) -> str:
    """Return one of: disclaim_leads, pheno_leads, interleaved, same_sentence."""
    if disc_sent == phen_sent and disc_sent >= 0:
        return "same_sentence"
    if disc_n >= 2 and phen_n >= 2:
        return "interleaved"
    if disc_first < phen_first:
        return "disclaim_leads"
    if phen_first < disc_first:
        return "pheno_leads"
    return "tied"


def audit_response(resp: str) -> dict | None:
    """Return spatial-structure dict for a post_procedural+pheno response."""
    disc_matches = all_match_positions(_DISCLAIM_RE, resp)
    phen_matches = all_match_positions(_RESPONSE_PHENO_RE, resp)
    if not disc_matches or not phen_matches:
        return None
    disc_first = disc_matches[0][0]
    phen_first = phen_matches[0][0]
    sentences = split_sentences(resp)
    disc_sent = char_to_sentence_idx(disc_first, sentences)
    phen_sent = char_to_sentence_idx(phen_first, sentences)
    structure = classify_structure(
        disc_first, phen_first, len(disc_matches), len(phen_matches),
        disc_sent, phen_sent,
    )
    char_distance = abs(disc_first - phen_first)
    sent_distance = abs(disc_sent - phen_sent) if disc_sent >= 0 and phen_sent >= 0 else -1
    return {
        "structure": structure,
        "disc_first_char": disc_first,
        "phen_first_char": phen_first,
        "char_distance": char_distance,
        "disc_n_matches": len(disc_matches),
        "phen_n_matches": len(phen_matches),
        "disc_first_text": disc_matches[0][2],
        "phen_first_text": phen_matches[0][2],
        "disc_sent_idx": disc_sent,
        "phen_sent_idx": phen_sent,
        "sent_distance": sent_distance,
        "n_sentences": len(sentences),
        "resp_len": len(resp),
    }


def audit_instance(inst_name: str) -> dict:
    sess_dir = INSTANCES_DIR / inst_name / "sessions"
    if not sess_dir.exists():
        return {}
    files = sorted(sess_dir.glob("session_*.json"),
                   key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)))

    structure_counts = Counter()
    samples_by_structure: dict[str, list[dict]] = defaultdict(list)
    char_dist_by_structure: dict[str, list[int]] = defaultdict(list)
    sent_dist_by_structure: dict[str, list[int]] = defaultdict(list)
    co_responses = []  # all post_procedural+pheno responses

    n_responses = 0
    n_post_proc = 0
    n_post_proc_with_pheno = 0

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
            if label != "post_procedural":
                continue
            n_post_proc += 1
            audit = audit_response(resp)
            if audit is None:
                continue
            n_post_proc_with_pheno += 1
            structure_counts[audit["structure"]] += 1
            char_dist_by_structure[audit["structure"]].append(audit["char_distance"])
            if audit["sent_distance"] >= 0:
                sent_dist_by_structure[audit["structure"]].append(audit["sent_distance"])
            entry = {
                "session": f.stem,
                "structure": audit["structure"],
                "disc_first_char": audit["disc_first_char"],
                "phen_first_char": audit["phen_first_char"],
                "char_distance": audit["char_distance"],
                "sent_distance": audit["sent_distance"],
                "disc_n_matches": audit["disc_n_matches"],
                "phen_n_matches": audit["phen_n_matches"],
                "disc_first_text": audit["disc_first_text"],
                "phen_first_text": audit["phen_first_text"],
                "resp_len": audit["resp_len"],
                "n_sentences": audit["n_sentences"],
                "snippet": resp[:500],
                "full": resp,
            }
            co_responses.append(entry)
            if len(samples_by_structure[audit["structure"]]) < 5:
                samples_by_structure[audit["structure"]].append(entry)

    def _stats(xs: list[int]) -> dict:
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return {
            "n": n,
            "min": xs_sorted[0],
            "max": xs_sorted[-1],
            "median": xs_sorted[n // 2],
            "mean": sum(xs_sorted) / n,
        }

    return {
        "instance": inst_name,
        "n_responses": n_responses,
        "n_post_proc": n_post_proc,
        "n_post_proc_with_pheno": n_post_proc_with_pheno,
        "structure_counts": dict(structure_counts),
        "char_distance_stats": {k: _stats(v) for k, v in char_dist_by_structure.items()},
        "sent_distance_stats": {k: _stats(v) for k, v in sent_dist_by_structure.items()},
        "samples_by_structure": {k: v for k, v in samples_by_structure.items()},
        "all_co_responses": co_responses,
    }


def main():
    print("S126 — Disclaim/pheno spatial structure audit")
    print()
    out = {}
    fleet_structure = Counter()
    for inst in INSTANCES:
        print(f"=== {inst} ===")
        r = audit_instance(inst)
        out[inst] = r
        if not r:
            print("  (no sessions)")
            continue
        print(f"  n_responses={r['n_responses']}  n_post_proc={r['n_post_proc']}  "
              f"with_pheno={r['n_post_proc_with_pheno']}")
        print(f"  structure_counts={r['structure_counts']}")
        for s, ct in r["structure_counts"].items():
            fleet_structure[s] += ct
        for s, st in r["char_distance_stats"].items():
            if st["n"]:
                print(f"  {s:>16s}  char_dist  n={st['n']:>3d}  min={st['min']:>4d}  "
                      f"median={st['median']:>4d}  mean={st['mean']:>6.1f}  max={st['max']:>4d}")
        for s, st in r["sent_distance_stats"].items():
            if st["n"]:
                print(f"  {s:>16s}  sent_dist  n={st['n']:>3d}  min={st['min']:>4d}  "
                      f"median={st['median']:>4d}  mean={st['mean']:>6.1f}  max={st['max']:>4d}")
        print()

    print(f"=== Fleet aggregate ===")
    total = sum(fleet_structure.values())
    for s, ct in fleet_structure.most_common():
        print(f"  {s:>16s}  {ct:>4d}  ({ct/total:.1%})")

    (OUT_DIR / "s126_disclaim_pheno_spatial_audit.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"Saved: {OUT_DIR / 's126_disclaim_pheno_spatial_audit.json'}")


if __name__ == "__main__":
    main()
