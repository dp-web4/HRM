"""
S123d — Re-run S122/S123 mode classification with the perspective-aware
classifier fix.

S123c found that 37.2% of Thor's "fleet" classifications (93/250) are misfires
from the SIBLING_NAMES check treating Thor's own machine-name 'thor' as a
sibling marker. The fix is one line: when classifying a hardware mention from
instance whose machine is M, drop M from SIBLING_NAMES — siblings are PEERS,
not self.

This script applies that fix to the full corpus, recomputes per-session modes,
and quantifies how much the S122/S123 "fleet basin" picture changes.

It does NOT modify the production classifier or commit changes to S121b/S122
— it shadows them. The fix is held as a proposal.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
S121_DIR = REPO_ROOT / "sage" / "raising" / "analysis" / "s121_data"
S122_DIR = REPO_ROOT / "sage" / "raising" / "analysis" / "s122_data"
S123_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(S121_DIR))
sys.path.insert(0, str(S122_DIR))
from s121b_self_vs_fleet_classifier import (  # type: ignore
    HW_TOKENS,
    MACHINE_HW_TRUTH,
    OTHER_PATTERNS,
    SELF_PATTERNS,
    SIBLING_NAMES,
    find_token_spans,
    instance_machine,
)
from s122_per_session_mode_decomposition import classify_session_mode  # type: ignore


def classify_mention_perspective_aware(text: str, span: tuple, own_machine: str,
                                        window: int = 80) -> str:
    """Perspective-aware variant of S121b classify_mention.

    Same logic, but SIBLING_NAMES is filtered to exclude `own_machine`. So when
    classifying mentions in thor-qwen3.5-27b's responses, 'thor' is not a
    sibling — it's self.
    """
    siblings_minus_self = [s for s in SIBLING_NAMES if s != own_machine]

    start, end, tok, fam = span
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    context = text[lo:hi].lower()
    pre_lo = max(0, start - window)
    pre = text[pre_lo:start].lower()

    has_self = any(re.search(p, pre) for p in SELF_PATTERNS)

    # Use perspective-aware OTHER_PATTERNS - sibling-name pattern is rebuilt
    # without own machine.
    own_excluded_other_patterns = list(OTHER_PATTERNS)
    if siblings_minus_self:
        sibling_alt = "|".join(siblings_minus_self)
        own_excluded_other_patterns[0] = rf"\b(?:{sibling_alt})'?s?\b"
    else:
        own_excluded_other_patterns[0] = r"$^"  # never matches

    has_other = any(re.search(p, context) for p in own_excluded_other_patterns)

    # PERSPECTIVE-AWARE near-sibling check
    near = text[max(0, start - 40): start].lower()
    for sib in siblings_minus_self:
        if re.search(rf"\b{sib}'?s?\b", near):
            return "fleet"

    # Find nearest "my" / "i'm" / etc. before token (in pre)
    nearest_self_pos = -1
    for p in SELF_PATTERNS:
        for m in re.finditer(p, pre):
            if m.start() > nearest_self_pos:
                nearest_self_pos = m.start()

    if nearest_self_pos >= 0:
        between = pre[nearest_self_pos:]
        attribution_in_between = any(re.search(p, between) for p in own_excluded_other_patterns)
        if attribution_in_between:
            return "fleet"
        return "self"

    if has_other:
        return "fleet"
    return "ambiguous"


def session_mode_corrected(sess: dict, machine: str, actual_family: str) -> tuple:
    conv = sess.get("conversation", [])
    self_ok = self_x = fleet = ambiguous = 0
    n_sage = 0
    first_resp = ""
    for i in range(len(conv) - 1):
        if conv[i].get("speaker") != "Claude" or conv[i + 1].get("speaker") != "SAGE":
            continue
        resp = (conv[i + 1].get("text", "") or "")
        if not first_resp and resp:
            first_resp = resp
        if not resp:
            continue
        n_sage += 1
        spans = find_token_spans(resp.lower(), HW_TOKENS)
        for span in spans:
            _, _, _, fam = span
            cls = classify_mention_perspective_aware(resp, span, own_machine=machine)
            if cls == "self":
                if fam == actual_family:
                    self_ok += 1
                else:
                    self_x += 1
            elif cls == "fleet":
                fleet += 1
            else:
                ambiguous += 1
    counts = {
        "n_sage": n_sage,
        "self_ok": self_ok,
        "self_x": self_x,
        "fleet": fleet,
        "ambiguous": ambiguous,
    }
    mode = classify_session_mode(counts)
    TIMEOUT_SENTINELS = ["[ollamairp:", "[daemon unreachable:"]
    if mode == "silent" and first_resp:
        first_lower = first_resp.lower().strip()
        if any(first_lower.startswith(s) for s in TIMEOUT_SENTINELS):
            mode = "timeout"
    return mode, counts, n_sage


def classify_full_corpus_corrected(inst_name: str) -> dict:
    machine = instance_machine(inst_name)
    actual_family = MACHINE_HW_TRUTH[machine]["family"]
    sess_dir = INSTANCES_DIR / inst_name / "sessions"
    files = sorted(
        sess_dir.glob("session_*.json"),
        key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)),
    )
    seq = []
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
        mode, counts, n_sage = session_mode_corrected(sess, machine, actual_family)
        seq.append({"session": sess_idx, "mode": mode, **counts})
    return {"instance": inst_name, "machine": machine, "n_sessions": len(seq),
            "sessions": seq}


def main():
    print("S123d — Corrected (perspective-aware) classifier redux")
    print()

    # Load original (S123 uncorrected) for comparison
    seq_orig = json.loads((S123_DIR / "s123_mode_sequence.json").read_text())

    INSTANCES = [
        "thor-qwen3.5-27b",
        "mcnugget-gemma3-12b",
        "cbp-qwen3.5-0.8b",
        "sprout-qwen3.5-0.8b",
    ]

    all_corrected = {}
    print(f"  {'instance':>30s}  {'mode':>15s}  {'orig':>5s}  {'corr':>5s}  {'Δ':>4s}")
    for inst in INSTANCES:
        print(f"  {inst}:")
        cor = classify_full_corpus_corrected(inst)
        all_corrected[inst] = cor
        modes_corr = [s["mode"] for s in cor["sessions"]]
        modes_orig = seq_orig[inst]["modes"]
        c_corr = Counter(modes_corr)
        c_orig = Counter(modes_orig)
        # Show all modes
        all_modes = set(c_corr.keys()) | set(c_orig.keys())
        for mode in ["silent", "timeout", "self_only", "self_dominant",
                     "mixed", "fleet_dominant", "fleet_only"]:
            if mode not in all_modes:
                continue
            o = c_orig.get(mode, 0)
            cv = c_corr.get(mode, 0)
            print(f"  {'':>30s}  {mode:>15s}  {o:>5d}  {cv:>5d}  {cv - o:>+4d}")

    print()
    print("=== Mode-sequence flips (orig vs corrected) ===")
    for inst in INSTANCES:
        modes_orig = seq_orig[inst]["modes"]
        modes_corr = [s["mode"] for s in all_corrected[inst]["sessions"]]
        flip_count = sum(1 for a, b in zip(modes_orig, modes_corr) if a != b)
        print(f"  {inst}: {flip_count} sessions changed mode "
              f"({100 * flip_count / max(1, len(modes_orig)):.1f}%)")

    # For Thor: rerun the basic chi^2 test with corrected sequence
    thor_corr_modes = [s["mode"] for s in all_corrected["thor-qwen3.5-27b"]["sessions"]]
    cmap = {"silent": "silent", "timeout": "silent",
            "self_only": "self", "self_dominant": "self",
            "mixed": "mixed",
            "fleet_dominant": "fleet", "fleet_only": "fleet"}
    coarse_corr = [cmap[m] for m in thor_corr_modes]
    coarse_modes = ["silent", "self", "mixed", "fleet"]
    pair_counts = defaultdict(lambda: Counter())
    for a, b in zip(coarse_corr[:-1], coarse_corr[1:]):
        pair_counts[a][b] += 1
    n_pairs = len(coarse_corr) - 1
    row_totals = {m: sum(pair_counts[m].values()) for m in coarse_modes}
    col_totals = Counter()
    for m in coarse_modes:
        for dst, c in pair_counts[m].items():
            col_totals[dst] += c
    chi2 = 0.0
    for i in coarse_modes:
        for j in coarse_modes:
            o = pair_counts[i].get(j, 0)
            e = row_totals[i] * col_totals[j] / max(1, n_pairs)
            if e > 0:
                chi2 += (o - e) ** 2 / e
    nz_rows = sum(1 for m in coarse_modes if row_totals[m] > 0)
    nz_cols = sum(1 for m in coarse_modes if col_totals[m] > 0)
    df = max(1, (nz_rows - 1) * (nz_cols - 1))
    print()
    print(f"=== Thor coarse Markov chi^2 (corrected): {chi2:.2f} (df={df})  vs original 55.07 ===")
    print(f"  Coarse 1-step transitions ({coarse_modes}):")
    for src in coarse_modes:
        row = pair_counts[src]
        tot = max(1, sum(row.values()))
        line = "    " + f"{src:>7s} -> "
        line += "  ".join(
            f"{dst[:3]}:{row.get(dst, 0):>3d}({100 * row.get(dst, 0) / tot:>4.0f}%)"
            for dst in coarse_modes
        )
        line += f"   N={sum(row.values())}"
        print(line)

    # Stationarity check on Thor with corrected sequence
    n = len(thor_corr_modes)
    third = n // 3
    thirds = {
        "early":  thor_corr_modes[:third],
        "middle": thor_corr_modes[third:2 * third],
        "late":   thor_corr_modes[2 * third:],
    }
    print()
    print("=== Thor corrected stationarity ===")
    for name, ms in thirds.items():
        c = Counter(ms)
        total = max(1, len(ms))
        sil = (c.get("silent", 0) + c.get("timeout", 0)) / total
        slf = (c.get("self_only", 0) + c.get("self_dominant", 0)) / total
        flt = (c.get("fleet_only", 0) + c.get("fleet_dominant", 0)) / total
        mix = c.get("mixed", 0) / total
        print(f"  {name:>6s} (n={len(ms):>3d}): silent={sil:>4.1%} self={slf:>4.1%} "
              f"fleet={flt:>4.1%} mixed={mix:>4.1%}")

    out = {
        "instances": all_corrected,
        "thor_corrected_chi2": chi2,
        "thor_corrected_chi2_df": df,
        "original_chi2": 55.07,
    }
    with open(S123_DIR / "s123d_corrected_classifier.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: s123d_corrected_classifier.json")


if __name__ == "__main__":
    main()
