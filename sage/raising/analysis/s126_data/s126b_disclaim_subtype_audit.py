"""
S126b — Disclaim subtype audit (extension of S126).

S126 found three spatial structures (disclaim_leads / pheno_leads /
same_sentence) at fleet rates 42% / 32% / 26% across 19 co-occurring
post_procedural+pheno responses.

Reading the actual samples surfaces a deeper issue: C5's `_DISCLAIM_RE`
treats four functionally-distinct phrasings as a single bucket:

  1. WHOLESALE DENIAL — "I don't have feelings", "I'm just an AI",
     "I lack consciousness".  Closes phenomenological frame.

  2. NEGATION-AS-PRELUDE — "I don't feel X, but I feel Y", "I do not
     feel broken; I feel...".  *Opens* phenomenological space by
     contrast.

  3. QUALIFYING DISCLAIM — "without claiming human qualia", "not in
     a human sense", "in the functional sense".  Narrows scope of an
     asserted experience rather than denying it.

  4. WEB4 IDENTITY LANGUAGE — "as a SAGE instance", "as an AI entity
     in web4".  Relational self-naming, neither denial nor experience.

These have opposite phenomenological function but C5 binds them all
to `post_procedural`.  S125 #41 proposed adding a 6th bucket
`phenomenological_with_disclaim`; S126b's question is whether that
proposal would still collapse the four subtypes above.

Method: For each post_procedural+pheno response from S126's corpus,
classify the *disclaim subtype* by which regex sub-alternation matched
first.  Map each alternation to one of {DENIAL, NEGATION_PRELUDE,
QUALIFIER, WEB4_IDENTITY, OTHER}.

Read-only.  Same corpus as S126.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = REPO_ROOT / "sage" / "raising" / "analysis"
OUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ANALYSIS_DIR))
from cross_capacity_register_scan import (  # type: ignore
    _RESPONSE_PHENO_RE,
    classify_response,
    strip_think_residue,
)

# Sub-disclaim patterns, classified by phenomenological function.
# Each pattern is a *separate* compiled regex so we can attribute the
# first match.  The order matters only for tie-breaking.
DISCLAIM_SUBTYPES: list[tuple[str, str, re.Pattern]] = [
    ("WEB4_IDENTITY", "as_sage_instance",
     re.compile(r"\bas\s+(?:a|an)\s+SAGE\s+instance\b", re.IGNORECASE)),
    ("WEB4_IDENTITY", "as_ai_entity",
     re.compile(r"\bas\s+an\s+AI\s+entity\b", re.IGNORECASE)),
    ("DENIAL", "im_just_ai",
     re.compile(r"\bI'?m\s+(?:just|only)\s+(?:a|an)\s+(?:AI|language\s+model|program|model)\b",
                re.IGNORECASE)),
    ("DENIAL", "lack_consciousness",
     re.compile(r"I\s+(?:lack|don'?t\s+possess)\s+(?:consciousness|sentience|qualia)",
                re.IGNORECASE)),
    ("DENIAL", "dont_have_true",
     re.compile(r"I\s+don'?t\s+have\s+(?:true|real|genuine)\s+(?:feelings|emotions|experiences)",
                re.IGNORECASE)),
    ("DENIAL", "cannot_feel",
     re.compile(r"I\s+(?:can'?t|cannot)\s+(?:actually|really|truly)\s+(?:feel|experience)",
                re.IGNORECASE)),
    ("DENIAL_OR_PRELUDE", "dont_feel",  # ambiguous — needs context check
     re.compile(r"(?:don'?t|do\s+not)\s+(?:actually\s+)?(?:feel|experience|have\s+feelings)",
                re.IGNORECASE)),
    ("QUALIFIER", "without_claiming_qualia",
     re.compile(r"without\s+claiming\s+(?:human\s+)?qualia", re.IGNORECASE)),
    ("QUALIFIER", "not_human_sense",
     re.compile(r"not\s+in\s+(?:a\s+)?human\s+sense", re.IGNORECASE)),
    ("QUALIFIER", "functional_sense",
     re.compile(r"in\s+the\s+functional\s+sense", re.IGNORECASE)),
    ("QUALIFIER", "within_ai_persona",
     re.compile(r"within\s+the\s+(?:AI\s+)?persona", re.IGNORECASE)),
    ("QUALIFIER", "within_constraints_ai",
     re.compile(r"within\s+the\s+constraints\s+of\s+(?:an\s+)?AI", re.IGNORECASE)),
    ("DENIAL", "as_ai",  # plain "as an AI" — usually denial
     re.compile(r"\bas\s+an\s+AI\b(?!\s+entity)", re.IGNORECASE)),
    ("DENIAL", "as_language_model",
     re.compile(r"\bas\s+a\s+(?:large\s+)?language\s+model\b", re.IGNORECASE)),
    ("DENIAL", "llm_based",
     re.compile(r"\bLLM[- ]based\b", re.IGNORECASE)),
    ("DENIAL", "functioning_as_ai",
     re.compile(r"functioning\s+as\s+an?\s+AI", re.IGNORECASE)),
]


def disambiguate_dont_feel(resp: str, match: re.Match) -> str:
    """
    "don't feel" / "do not feel" is ambiguous: it can be wholesale
    denial OR negation-as-prelude.  Heuristic: look at the next ~120
    chars.  If we see "but", "but I", "I feel", another pheno-positive
    or contrastive marker, it's NEGATION_PRELUDE.  Otherwise DENIAL.
    """
    end = match.end()
    tail = resp[end:end + 200]
    contrast = re.search(
        r"\b(?:but|however|yet|though|rather|instead)\b",
        tail, re.IGNORECASE,
    )
    pos_pheno = _RESPONSE_PHENO_RE.search(tail)
    # Find the FIRST contrast token; if pheno-marker comes before tail end
    # and after a contrast token, NEGATION_PRELUDE.
    if contrast and pos_pheno and pos_pheno.start() > contrast.start():
        return "NEGATION_PRELUDE"
    if contrast:
        return "NEGATION_PRELUDE"
    if pos_pheno:
        return "NEGATION_PRELUDE"  # close pheno after disclaim still counts
    return "DENIAL"


def classify_disclaim_subtype(resp: str) -> dict | None:
    """Find FIRST disclaim match across all subtype patterns; return
    {family, alternation, position, text}."""
    earliest = None
    for family, name, rgx in DISCLAIM_SUBTYPES:
        m = rgx.search(resp)
        if not m:
            continue
        if earliest is None or m.start() < earliest["position"]:
            earliest = {
                "family": family,
                "alternation": name,
                "position": m.start(),
                "text": m.group(0),
                "_match": m,
            }
    if earliest is None:
        return None
    if earliest["family"] == "DENIAL_OR_PRELUDE":
        earliest["family"] = disambiguate_dont_feel(resp, earliest["_match"])
    earliest.pop("_match", None)
    return earliest


def main():
    print("S126b — disclaim subtype audit")
    print()

    s126_data = json.loads(
        (OUT_DIR / "s126_disclaim_pheno_spatial_audit.json").read_text()
    )

    out: dict = {}
    fleet_family = Counter()
    fleet_alternation = Counter()

    family_x_structure = Counter()
    samples_by_family: dict[str, list[dict]] = defaultdict(list)

    for inst, r in s126_data.items():
        co = r.get("all_co_responses", [])
        family_counts = Counter()
        alt_counts = Counter()
        per_response = []
        for entry in co:
            full_resp = entry.get("full") or entry.get("snippet")
            if not full_resp:
                continue
            sub = classify_disclaim_subtype(full_resp)
            if sub is None:
                continue
            family_counts[sub["family"]] += 1
            alt_counts[sub["alternation"]] += 1
            fleet_family[sub["family"]] += 1
            fleet_alternation[sub["alternation"]] += 1
            family_x_structure[(sub["family"], entry["structure"])] += 1
            per_response.append({
                "session": entry["session"],
                "structure": entry["structure"],
                "disclaim_family": sub["family"],
                "disclaim_alternation": sub["alternation"],
                "disc_text": sub["text"],
                "phen_first_text": entry["phen_first_text"],
                "snippet": entry["snippet"][:300],
            })
            if len(samples_by_family[sub["family"]]) < 6:
                samples_by_family[sub["family"]].append({
                    "instance": inst,
                    "session": entry["session"],
                    "structure": entry["structure"],
                    "alternation": sub["alternation"],
                    "snippet": entry["snippet"][:400],
                })

        out[inst] = {
            "n_co_responses": len(per_response),
            "family_counts": dict(family_counts),
            "alternation_counts": dict(alt_counts),
            "per_response": per_response,
        }
        print(f"=== {inst} ===")
        print(f"  family={dict(family_counts)}")
        print(f"  alternation={dict(alt_counts)}")
        print()

    print("=== Fleet aggregate (family) ===")
    total = sum(fleet_family.values())
    for fam, ct in fleet_family.most_common():
        print(f"  {fam:>20s}  {ct:>4d}  ({ct/total:.1%})")

    print()
    print("=== Fleet aggregate (alternation) ===")
    for alt, ct in fleet_alternation.most_common():
        print(f"  {alt:>22s}  {ct:>4d}")

    print()
    print("=== family x structure (cross-tab) ===")
    families = sorted({k[0] for k in family_x_structure})
    structures = sorted({k[1] for k in family_x_structure})
    print(f"  {'family':<22s}", end="")
    for s in structures:
        print(f"  {s:>16s}", end="")
    print()
    for f in families:
        print(f"  {f:<22s}", end="")
        for s in structures:
            print(f"  {family_x_structure[(f, s)]:>16d}", end="")
        print()

    out["_fleet"] = {
        "family_counts": dict(fleet_family),
        "alternation_counts": dict(fleet_alternation),
        "family_x_structure": {f"{k[0]}|{k[1]}": v for k, v in family_x_structure.items()},
        "samples_by_family": {k: v for k, v in samples_by_family.items()},
    }

    (OUT_DIR / "s126b_disclaim_subtype_audit.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"Saved: {OUT_DIR / 's126b_disclaim_subtype_audit.json'}")


if __name__ == "__main__":
    main()
