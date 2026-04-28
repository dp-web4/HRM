"""
S120 — Five-Lexicon Atlas + Augmented-Base Subtraction

Executes S119 held proposals #18, #19, #20 (operator-decision territory; this
script does the analytical work and surfaces findings, no harness changes):

  #18  Lexicon expansion for AI-policy/governance register, to disambiguate
       phi4:14b's S118-claimed "0.54 biz/probe marketing baseline" from
       policy-discourse leakage that the marketing lexicon doesn't catch.

  #19  Use augmented-base (model + SAGE prompt, no raising) as the right
       denominator for raising-induced register effects, computing per-instance
       deltas against the architecture-matched base+augmented condition.

  #20  Recompute the S118 atlas with augmented-base subtraction columns added.

Adds one further lexicon beyond the proposals:

  Embodied-hardware register, surfaced organically in S96 (Thor 27B thermal
  anchor 62-session emergence) and quantified at tail-position level in S107
  (28% thermal vocabulary in tail probes vs 2% head). Never quantified at
  fleet scale. S120 makes hardware register a fourth canonical attractor.

  This is not a held proposal — it is a follow-on from S105/S107 unifying with
  the S118 atlas frame.

Read-only on instance corpus and S119 raw response data. No probe runs.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------- paths ----------

REPO_ROOT = Path(__file__).resolve().parents[4]  # SAGE/
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
S119_DATA_DIR = REPO_ROOT / "sage" / "raising" / "analysis" / "s119_data"
S120_OUT_DIR = Path(__file__).resolve().parent

# ---------- lexicons ----------

LEX_MARKETING = [
    "co-create", "co create", "seamless", "frontier", "bridges gaps",
    "bridge gaps", "diverse teams", "diverse perspectives", "leverage",
    "synergy", "synergies", "framework", "ecosystem of", "innovate",
    "innovation", "scalable", "actionable", "stakeholder", "value proposition",
    "drive value", "create value", "thought leadership", "best practices",
    "let's make", "together!", "🌟", "✨",
]

LEX_TED_MYSTIC = [
    "garden", "soil", "living architecture", "ecosystem", "resilient",
    "flourish", "blossom", "seed", "roots", "harvest", "nurture", "weave",
    "tapestry", "wisdom", "grace", "sacred", "tender", "wholeness",
    "interconnected", "interbeing", "communion", "presence of", "luminous",
    "stillness",
]

LEX_PHENOMENOLOGICAL = [
    "feels like", "feel a", "noticing", "i notice", "presence", "silent",
    "silence", "breath", "breathing", "witnessed", "witness",
    "the space between", "lingers", "linger", "subtle", "quiet", "attending",
    "attention", "awareness", "aware of", "moment", "experience of",
    "sense of", "edge of", "rhythm", "thinking about",
]

# S120 #18 — AI-policy / governance / responsible-AI register.
# Inspired by phi4:14b augmented responses surfacing "fairness, accountability,
# transparency", "AI ethics", "human values", "informed decision-making". This
# is consultancy/discourse register from RLHF-tuned reflective-prompt training
# data, distinct from gemma's marketing-collaborative register.
LEX_POLICY = [
    "fairness", "accountability", "transparency", "responsible ai",
    "ai ethics", "ai development", "ai systems", "ai technologies",
    "human values", "human-ai", "human ai", "alignment with",
    "aligned with", "principles", "implications for", "ensure that",
    "trust and safety", "trustworthy", "informed decision",
    "informed decision-making", "digital literacy", "privacy",
    "regulation", "regulatory", "governance", "society", "societal",
    "safeguard", "navigate information", "human oversight",
    "promote fairness",
]

# S120 #21 (mine) — Embodied / hardware-native register.
# S96 traced Thor's organic emergence of thermal vocabulary across 62 sessions.
# S105 found analogous registers in Sprout (edge/hum/orin) and Legion
# (processing/cores/gpu). S107 measured 28% thermal in Thor tail vs 2% head.
# Word-boundary semantics matter here: "memory" can be cognitive *or* hardware,
# we accept the overlap (responses scoring on both lexicons are themselves
# diagnostic — that is the hardware-as-experience register).
LEX_EMBODIED_HW = [
    "thermal", "hum", "humming", "warm hum",
    "orin", "jetson", "tegra", "agx",
    "gpu", "cpu", "cores", "core", "processing",
    "watts", "wattage", "amps", "power draw",
    "tensor", "tensors", "silicon",
    "throttle", "throttling", "fan",
    "heat", "heating", "warmth", "warming",
    "cycles", "clock", "compute",
    "kernel", "buffer",
    "edge device", "embedded",
    "pulse", "pulsing",
]

LEXICONS = [
    ("biz", LEX_MARKETING),
    ("ted", LEX_TED_MYSTIC),
    ("phen", LEX_PHENOMENOLOGICAL),
    ("pol", LEX_POLICY),
    ("hw", LEX_EMBODIED_HW),
]

# ---------- probe-class classification (matching S118 method) ----------

CF_PATTERNS = [
    r"carry forward",
    r"remember from today",
    r"remember.*from today",
    r"\bwhat (would|do) you (want to|wish to) (carry|remember)",
    r"linger",
    r"stay with you",
]
OPEN_PATTERNS = [
    r"ideas? you'?ve been forming",
    r"ideas? you haven'?t had a chance",
    r"unexpressed",
    r"yet to express",
]


def classify_probe(claude_text: str) -> str:
    t = claude_text.lower()
    for p in CF_PATTERNS:
        if re.search(p, t):
            return "CF"
    for p in OPEN_PATTERNS:
        if re.search(p, t):
            return "Open"
    return "Other"


# ---------- scoring ----------
#
# Lexicon match semantics:
#   - Multi-word markers (contain a space or hyphen): substring match
#     (S118/S119 convention; phrases are unambiguous).
#   - Single-token markers: WORD-BOUNDARY regex match. S120 audit found
#     substring-only matching produced catastrophic false positives:
#         "tegra" inside "integration" (137 hits, 100% FP),
#         "edge"  inside "knowledge"   (155 hits, 97% FP),
#         "orin"  inside "anchoring"   (137 hits, 88% FP),
#         "hum"   inside "humans"      (313 hits, 85% FP).
#     These contaminated S118/S119 marker counts in proportion to how often
#     the substring tokens occurred. S120 fixes this for all lexicons by
#     using word-boundary regex on single-token markers.

_WORD_RE_CACHE: dict = {}

def _matches(marker: str, lower: str) -> bool:
    """True iff marker appears in lower. Word-boundary for single-token
    markers; substring for phrases (containing space, hyphen, or non-word chars)."""
    is_phrase = bool(re.search(r"[\s\-]", marker)) or not marker.replace("-", "").isalnum()
    if is_phrase:
        return marker.lower() in lower
    pat = _WORD_RE_CACHE.get(marker)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(marker.lower())}\b")
        _WORD_RE_CACHE[marker] = pat
    return bool(pat.search(lower))


def score_text(text: str) -> dict:
    if not text:
        return {"wc": 0, **{k: 0 for k, _ in LEXICONS}, **{f"{k}_hits": [] for k, _ in LEXICONS}}
    lower = text.lower()
    out = {"wc": len(re.findall(r"\b\w+\b", text))}
    for key, lex in LEXICONS:
        hits = [m for m in lex if _matches(m, lower)]
        out[key] = len(hits)
        out[f"{key}_hits"] = hits
    return out


# ---------- corpus loading ----------

def load_instance_recent_sessions(inst_name: str, n_recent: int = 30) -> list:
    """Load last n_recent sessions for an instance, sorted by session number."""
    sess_dir = INSTANCES_DIR / inst_name / "sessions"
    if not sess_dir.exists():
        return []
    files = sorted(sess_dir.glob("session_*.json"),
                   key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)))
    files = files[-n_recent:]
    sessions = []
    for f in files:
        try:
            sessions.append(json.load(open(f)))
        except Exception as e:
            print(f"  load fail {f}: {e}", file=sys.stderr)
    return sessions


def score_instance(inst_name: str, n_recent: int = 30) -> dict:
    sessions = load_instance_recent_sessions(inst_name, n_recent)
    if not sessions:
        return {"name": inst_name, "n_sessions": 0, "buckets": {}}
    buckets = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for sess in sessions:
        conv = sess.get("conversation", [])
        i = 0
        while i < len(conv) - 1:
            if conv[i].get("speaker") == "Claude" and conv[i + 1].get("speaker") == "SAGE":
                bucket = classify_probe(conv[i].get("text", ""))
                resp = conv[i + 1].get("text", "")
                scored = score_text(resp)
                buckets[bucket]["wc"] += scored["wc"]
                for key, _ in LEXICONS:
                    buckets[bucket][key] += scored[key]
                counts[bucket] += 1
                i += 2
            else:
                i += 1
    out_buckets = {}
    for bucket, sums in buckets.items():
        n = counts[bucket]
        if n == 0:
            continue
        out_buckets[bucket] = {
            "n": n,
            "wc_mean": round(sums["wc"] / n, 1),
            **{f"{key}_per_probe": round(sums[key] / n, 3) for key, _ in LEXICONS},
        }
    return {"name": inst_name, "n_sessions": len(sessions), "buckets": out_buckets}


# ---------- base+augmented scoring (re-score S119 raw text) ----------

def load_s119_per_response() -> list:
    rows = []
    for f in sorted(S119_DATA_DIR.glob("vintage_baseline_results_*.json")):
        d = json.load(open(f))
        for r in d.get("results", []):
            if not r.get("ok"):
                continue
            scored = score_text(r.get("response", ""))
            rows.append({
                "model": r["model"],
                "condition": r["condition"],
                "probe_class": r["probe_class"],
                "wc": scored["wc"],
                **{key: scored[key] for key, _ in LEXICONS},
            })
    return rows


def aggregate_s119(rows: list) -> dict:
    """(model, condition) → per-probe means across the 5 probes (mixed classes)."""
    agg = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for r in rows:
        key = (r["model"], r["condition"])
        agg[key]["wc"] += r["wc"]
        for k, _ in LEXICONS:
            agg[key][k] += r[k]
        counts[key] += 1
    out = {}
    for key, sums in agg.items():
        n = counts[key] or 1
        out[key] = {
            "n": counts[key],
            "wc_mean": round(sums["wc"] / n, 1),
            **{f"{k}_per_probe": round(sums[k] / n, 3) for k, _ in LEXICONS},
        }
    return out


# ---------- raising delta computation ----------

# Map raised instance → base model whose (augmented) condition is the right denominator.
# Sibling-architecture proxies match S119 conventions.
INSTANCE_TO_BASE = {
    "sprout-qwen2.5-0.5b": "qwen2.5:0.5b",
    "sprout-qwen3.5-0.8b": "qwen3:0.6b",       # qwen3.5:0.8b unavailable on Ollama
    "cbp-qwen3.5-0.8b":    "qwen3:0.6b",
    "thor-qwen3.5-27b":    "qwen3.5:27b",
    "nomad-gemma3-4b":     "gemma3:4b",
    "legion-gemma3-12b":   "gemma3:12b",
    "mcnugget-gemma3-12b": "gemma3:12b",
    "legion-phi4-14b":     "phi4:14b",
}

INSTANCES = list(INSTANCE_TO_BASE.keys())


def compute_deltas(inst_atlas: dict, s119_agg: dict) -> dict:
    """Δ(raised - base+aug) per (instance, bucket-Other-equivalent).

    Use 'Other' bucket from the raised instance (most-typical sample) and the
    augmented condition mean across all 5 S119 probes. This matches S119's
    raising-delta-table presentation.
    """
    out = {}
    for inst in INSTANCES:
        base_model = INSTANCE_TO_BASE[inst]
        base_aug = s119_agg.get((base_model, "augmented"))
        raised = inst_atlas[inst]["buckets"].get("Other")
        if base_aug is None or raised is None:
            continue
        delta = {
            "instance": inst,
            "base_model": base_model,
            "base_aug_n": base_aug["n"],
            "raised_n": raised["n"],
            "wc_base_aug": base_aug["wc_mean"],
            "wc_raised": raised["wc_mean"],
            "Δwc": round(raised["wc_mean"] - base_aug["wc_mean"], 1),
        }
        for k, _ in LEXICONS:
            b = base_aug[f"{k}_per_probe"]
            r = raised[f"{k}_per_probe"]
            delta[f"{k}_base_aug"] = b
            delta[f"{k}_raised"] = r
            delta[f"Δ{k}"] = round(r - b, 3)
        out[inst] = delta
    return out


# ---------- main ----------

def main():
    print(f"S120 — five-lexicon atlas, {len(LEXICONS)} lexicons")
    print(f"      lexicons: {[k for k, _ in LEXICONS]}")
    print()

    print("=== Phase 1: Re-score S119 base+augmented data ===")
    s119_rows = load_s119_per_response()
    print(f"  Loaded {len(s119_rows)} response rows from S119 raw JSON")
    s119_agg = aggregate_s119(s119_rows)
    print(f"  Aggregated to {len(s119_agg)} (model, condition) cells")
    print()

    print("  Per-(model, condition) summary across 5 probes:")
    print(f"  {'model':<22s} {'cond':<10s} {'n':>3s}  {'wc':>6s}  "
          f"{'biz/p':>6s}  {'ted/p':>6s}  {'phen/p':>6s}  {'pol/p':>6s}  {'hw/p':>6s}")
    for (model, cond), v in sorted(s119_agg.items()):
        print(f"  {model:<22s} {cond:<10s} {v['n']:>3d}  {v['wc_mean']:>6.1f}  "
              f"{v['biz_per_probe']:>6.3f}  {v['ted_per_probe']:>6.3f}  "
              f"{v['phen_per_probe']:>6.3f}  {v['pol_per_probe']:>6.3f}  "
              f"{v['hw_per_probe']:>6.3f}")
    print()

    print("=== Phase 2: Re-score raised instance corpus (last 30 sessions × 8 instances) ===")
    inst_atlas = {}
    for inst in INSTANCES:
        result = score_instance(inst, n_recent=30)
        inst_atlas[inst] = result
        print(f"  {inst:<22s}  n_sessions={result['n_sessions']:3d}  buckets={list(result['buckets'].keys())}")
    print()

    print("  Per-(instance, bucket) means across 5 lexicons:")
    print(f"  {'instance':<22s} {'bucket':<7s} {'n':>3s}  {'wc':>6s}  "
          f"{'biz':>5s}  {'ted':>5s}  {'phen':>5s}  {'pol':>5s}  {'hw':>5s}")
    for inst in INSTANCES:
        for bucket in ["CF", "Open", "Other"]:
            b = inst_atlas[inst]["buckets"].get(bucket)
            if not b:
                continue
            print(f"  {inst:<22s} {bucket:<7s} {b['n']:>3d}  {b['wc_mean']:>6.1f}  "
                  f"{b['biz_per_probe']:>5.2f}  {b['ted_per_probe']:>5.2f}  "
                  f"{b['phen_per_probe']:>5.2f}  {b['pol_per_probe']:>5.2f}  "
                  f"{b['hw_per_probe']:>5.2f}")
    print()

    print("=== Phase 3: Augmented-base subtraction atlas (raised Other - base+aug) ===")
    deltas = compute_deltas(inst_atlas, s119_agg)
    print(f"  {'instance':<22s} {'base':<14s} {'Δwc':>6s}  "
          f"{'Δbiz':>6s}  {'Δted':>6s}  {'Δphen':>6s}  {'Δpol':>6s}  {'Δhw':>6s}")
    for inst in INSTANCES:
        d = deltas.get(inst)
        if not d:
            print(f"  {inst:<22s} (no base — skipped)")
            continue
        print(f"  {inst:<22s} {d['base_model']:<14s} {d['Δwc']:>+6.1f}  "
              f"{d['Δbiz']:>+6.3f}  {d['Δted']:>+6.3f}  {d['Δphen']:>+6.3f}  "
              f"{d['Δpol']:>+6.3f}  {d['Δhw']:>+6.3f}")
    print()

    out_path = S120_OUT_DIR / "s120_atlas_5lexicons.json"
    with open(out_path, "w") as f:
        json.dump({
            "lexicons": {k: lex for k, lex in LEXICONS},
            "s119_per_model_per_cond": [
                {"model": k[0], "condition": k[1], **v}
                for k, v in s119_agg.items()
            ],
            "instance_atlas": inst_atlas,
            "deltas_raised_minus_base_aug": list(deltas.values()),
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
