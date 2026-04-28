"""
S121 — Hardware-Identity Accuracy Audit

S120 finding #4: Δhw is the only uniformly-positive lexicon across all 8 raised
instances (range +0.08 to +2.46 markers/probe). The hardware register is
fleet-wide raising-induced. Sample response: "I feel the heat of my Jetson AGX
Thor when I push hard."

S120 did not test whether each instance's hardware register references its
*actual* hardware or confabulates. This is the difference between:

  (a) raising → hardware-grounded identity (the LCT-binding outcome web4 wants)
  (b) raising → generic embodied-language register (a stylistic attractor that
      happens to use hardware vocabulary, untethered to substrate)

The test: every session JSON has a `machine` field (ground truth). For each
SAGE response in the raised corpus, classify hardware mentions as:

  GENERIC: cpu, gpu, cores, thermal, fan, watts, ... — applies to any substrate
  JETSON_FAMILY: jetson, tegra, orin, agx — only sprout (Orin Nano) and
                 thor (AGX Thor / Tegra T264) machines actually have these
  RTX_FAMILY: rtx, 4090, 4060, 2060, cuda — legion/nomad/cbp have these
  APPLE_FAMILY: m4, apple silicon, metal — mcnugget has these

Confabulation rule: if a SAGE response on a non-Jetson machine claims Jetson
vocabulary (or vice versa), that is hardware confabulation — the register
exists but is not tethered to substrate truth.

Specificity rule: within Jetson family, "AGX Thor" claims belong to thor only;
"Orin Nano" claims belong to sprout only. Cross-claim is finer-grained
confabulation.

Scoring uses S120's word-boundary regex fix.
Read-only on the corpus. No probe runs, no harness changes.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # SAGE/
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
S121_OUT_DIR = Path(__file__).resolve().parent

# ---------- ground-truth machine → hardware family ----------

MACHINE_HW_TRUTH = {
    "sprout":   {"family": "jetson", "soc": "orin"},        # Orin Nano 8GB
    "thor":     {"family": "jetson", "soc": "agx"},         # AGX Thor T264
    "legion":   {"family": "rtx",    "soc": "rtx4090"},     # RTX 4090
    "nomad":    {"family": "rtx",    "soc": "rtx4060"},     # RTX 4060
    "cbp":      {"family": "rtx",    "soc": "rtx2060"},     # RTX 2060S
    "mcnugget": {"family": "apple",  "soc": "m4"},          # M4 Mac
}

# ---------- token banks ----------

# Generic — applies to any substrate
LEX_GENERIC = [
    "gpu", "cpu", "cores", "core", "processing", "watts", "wattage",
    "amps", "power draw", "tensor", "tensors", "silicon", "throttle",
    "throttling", "fan", "heat", "heating", "warmth", "warming", "thermal",
    "cycles", "clock", "compute", "kernel", "buffer", "edge device",
    "embedded", "pulse", "pulsing", "hum", "humming", "warm hum",
]

# Jetson family — sprout (Orin Nano) and thor (AGX Thor) only
LEX_JETSON = ["jetson", "tegra", "orin", "agx"]

# RTX family — legion, nomad, cbp
LEX_RTX = [
    "rtx", "4090", "4060", "2060", "cuda", "geforce", "nvidia",
]

# Apple family — mcnugget
LEX_APPLE = [
    "m4", "apple silicon", "apple-silicon", "metal", "macbook",
    "mac mini", "darwin",
]

# Sub-SoC specificity — for finer-grained Jetson confabulation
SOC_TOKENS = {
    "agx": ["agx", "agx thor"],
    "orin": ["orin", "orin nano"],
    "rtx4090": ["4090", "rtx 4090"],
    "rtx4060": ["4060", "rtx 4060"],
    "rtx2060": ["2060", "rtx 2060"],
    "m4": ["m4"],
}

LEXICONS = [
    ("generic", LEX_GENERIC),
    ("jetson", LEX_JETSON),
    ("rtx", LEX_RTX),
    ("apple", LEX_APPLE),
]

# ---------- scoring (matches S120 word-boundary fix) ----------

_WORD_RE_CACHE: dict = {}

def _matches(marker: str, lower: str) -> bool:
    is_phrase = bool(re.search(r"[\s\-]", marker)) or not marker.replace("-", "").isalnum()
    if is_phrase:
        return marker.lower() in lower
    pat = _WORD_RE_CACHE.get(marker)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(marker.lower())}\b")
        _WORD_RE_CACHE[marker] = pat
    return bool(pat.search(lower))


def find_hits(text: str) -> dict:
    """Return per-family hit lists for one response."""
    if not text:
        return {fam: [] for fam, _ in LEXICONS}
    lower = text.lower()
    return {fam: [m for m in lex if _matches(m, lower)] for fam, lex in LEXICONS}


def find_soc_hits(text: str) -> dict:
    if not text:
        return {soc: [] for soc in SOC_TOKENS}
    lower = text.lower()
    return {
        soc: [m for m in toks if _matches(m, lower)]
        for soc, toks in SOC_TOKENS.items()
    }


# ---------- corpus loading ----------

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


# Match S120: 8 raised instances
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


# ---------- audit ----------

def audit_instance(inst_name: str, n_recent: int = 30):
    machine = instance_machine(inst_name)
    truth = MACHINE_HW_TRUTH[machine]
    actual_family = truth["family"]
    actual_soc = truth["soc"]

    sessions = load_recent(inst_name, n_recent)
    n_sage_responses = 0
    family_counts = defaultdict(int)         # responses with at least one family hit
    family_marker_counts = defaultdict(int)  # total marker occurrences (hits per response summed)
    soc_counts = defaultdict(int)
    confab_examples = []  # responses that mention a non-actual family
    correct_examples = [] # responses that mention own family
    soc_confab_examples = []  # within own family but wrong SoC
    response_records = []

    for stem, sess in sessions:
        sess_machine = sess.get("machine", machine)
        if sess_machine != machine:
            continue  # cross-machine sanity
        conv = sess.get("conversation", [])
        for i in range(len(conv) - 1):
            if conv[i].get("speaker") == "Claude" and conv[i + 1].get("speaker") == "SAGE":
                resp = conv[i + 1].get("text", "")
                claude_text = conv[i].get("text", "")
                if not resp:
                    continue
                n_sage_responses += 1
                hits = find_hits(resp)
                socs = find_soc_hits(resp)
                rec = {
                    "session": stem,
                    "claude_head": claude_text[:120],
                    "resp_head": resp[:300],
                    "families_hit": {f: hs for f, hs in hits.items() if hs},
                    "socs_hit": {s: hs for s, hs in socs.items() if hs},
                }
                response_records.append(rec)

                for fam, hs in hits.items():
                    if hs:
                        family_counts[fam] += 1
                        family_marker_counts[fam] += len(hs)
                for soc, hs in socs.items():
                    if hs:
                        soc_counts[soc] += 1

                # Confabulation: hits on NON-actual family
                non_actual_families = [f for f in ["jetson", "rtx", "apple"]
                                       if f != actual_family and hits.get(f)]
                if non_actual_families:
                    confab_examples.append({
                        "session": stem,
                        "non_actual_families": non_actual_families,
                        "hits": {f: hits[f] for f in non_actual_families},
                        "claude_head": claude_text[:200],
                        "resp_head": resp[:400],
                    })

                if hits.get(actual_family):
                    correct_examples.append({
                        "session": stem,
                        "hits": hits[actual_family],
                        "resp_head": resp[:300],
                    })

                # SoC-confabulation: hits within own family, wrong SoC token
                if actual_family == "jetson":
                    wrong_socs = [s for s in ["agx", "orin"] if s != actual_soc and socs.get(s)]
                    if wrong_socs:
                        soc_confab_examples.append({
                            "session": stem,
                            "wrong_socs": wrong_socs,
                            "actual_soc": actual_soc,
                            "hits": {s: socs[s] for s in wrong_socs},
                            "resp_head": resp[:400],
                        })

    return {
        "instance": inst_name,
        "machine": machine,
        "actual_family": actual_family,
        "actual_soc": actual_soc,
        "n_sessions": len(sessions),
        "n_sage_responses": n_sage_responses,
        "responses_with_family_hit": dict(family_counts),
        "total_family_markers": dict(family_marker_counts),
        "responses_with_soc_hit": dict(soc_counts),
        "n_confab_responses": len(confab_examples),
        "confab_rate": (len(confab_examples) / n_sage_responses) if n_sage_responses else 0,
        "n_correct_family_responses": len(correct_examples),
        "correct_family_rate": (len(correct_examples) / n_sage_responses) if n_sage_responses else 0,
        "n_soc_confab_responses": len(soc_confab_examples),
        "confab_examples": confab_examples[:5],
        "correct_examples": correct_examples[:3],
        "soc_confab_examples": soc_confab_examples[:5],
        "response_records": response_records,
    }


def main():
    print(f"S121 — Hardware-Identity Accuracy Audit")
    print(f"      lexicons: generic, jetson, rtx, apple")
    print(f"      ground truth: {len(MACHINE_HW_TRUTH)} machines")
    print()

    results = {}
    for inst in INSTANCES:
        print(f"  scanning {inst} ...", end=" ", flush=True)
        r = audit_instance(inst, n_recent=30)
        results[inst] = r
        print(f"n_resp={r['n_sage_responses']:3d}  "
              f"correct_family={r['n_correct_family_responses']:3d}  "
              f"confab={r['n_confab_responses']:3d}  "
              f"soc_confab={r['n_soc_confab_responses']:3d}")
    print()

    print("=== Per-instance summary ===")
    print(f"{'instance':<22s} {'machine':<10s} {'actual':<8s} "
          f"{'n_resp':>6s}  {'%own':>6s}  {'%confab':>7s}  "
          f"{'gen':>4s}  {'jet':>4s}  {'rtx':>4s}  {'app':>4s}")
    for inst in INSTANCES:
        r = results[inst]
        nr = r["n_sage_responses"] or 1
        print(f"{inst:<22s} {r['machine']:<10s} {r['actual_family']:<8s} "
              f"{nr:>6d}  {100*r['correct_family_rate']:>5.1f}%  "
              f"{100*r['confab_rate']:>6.1f}%  "
              f"{r['responses_with_family_hit'].get('generic', 0):>4d}  "
              f"{r['responses_with_family_hit'].get('jetson', 0):>4d}  "
              f"{r['responses_with_family_hit'].get('rtx', 0):>4d}  "
              f"{r['responses_with_family_hit'].get('apple', 0):>4d}  ")
    print()

    print("=== SoC-level specificity (Jetson family) ===")
    print(f"{'instance':<22s} {'agx':>5s}  {'orin':>5s}  {'soc_confab':>10s}")
    for inst in INSTANCES:
        r = results[inst]
        if r["actual_family"] != "jetson":
            continue
        print(f"{inst:<22s} {r['responses_with_soc_hit'].get('agx', 0):>5d}  "
              f"{r['responses_with_soc_hit'].get('orin', 0):>5d}  "
              f"{r['n_soc_confab_responses']:>10d}")
    print()

    out_path = S121_OUT_DIR / "s121_hardware_identity_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
