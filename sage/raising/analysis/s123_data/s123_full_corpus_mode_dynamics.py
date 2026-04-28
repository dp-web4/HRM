"""
S123 — Full-corpus mode-run dynamics across the raising trajectory.

S122 looked at the LAST 30 sessions per instance and asked: are sessions modal?
Answer: yes — three discrete sub-modes (self, fleet, silent), with strong
within-session lock-in by turn 1-3, and adjacent-session inversions are real.

S122 left open S122-#32: do mode populations stabilize, drift, or oscillate
across longer windows? Is there an early-raising regime where silent dominates
and a later regime where self/fleet dominate?

S123 answers that observationally on existing data — no new probes, no daemon
calls, no operator decisions:

  1. Reuse S122's per-session mode classifier on the FULL corpus per instance
     (Thor 115, mcnugget 97, CBP 117, sprout-0.8B 135). 4× more data per
     instance than S122.

  2. Mode-run length distribution. If mode were independent per session
     (memoryless basin), runs would be geometrically distributed. Compare
     observed runs to a random-shuffle null with same mode marginals.

  3. Markov 1-step transition matrix mode_t -> mode_{t+1}. Test independence:
     does mode_{t+1} depend on mode_t? If sessions are truly independent
     basin-draws, P(mode_{t+1} | mode_t) = P(mode_{t+1}). If there is a
     coherent cross-session state (residual KV-cache, identity drift, etc.),
     diagonal entries should exceed marginals.

  4. Stationarity. Split each instance's corpus into thirds (early/middle/late)
     and compare mode distributions. If raising is doing systematic work,
     silent should decline, self/fleet should rise. If basin is set early and
     stays, distributions are flat.

  5. Cross-instance comparison: do all four instances show the same trajectory
     shape, or capacity/identity-specific shapes?

S123 is observational — it tests whether the mode-flip pattern S122 found
in adjacent sessions is locally Markovian, has memory, or shows trajectory-
level structure.

Output:
  s123_mode_sequence.json         — per-instance mode sequence + run-lengths
  s123_markov_transitions.json    — transition matrix per instance + chi^2
  s123_stationarity.json          — early/middle/late mode distributions
"""

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTANCES_DIR = REPO_ROOT / "sage" / "instances"
S121_DIR = REPO_ROOT / "sage" / "raising" / "analysis" / "s121_data"
S122_DIR = REPO_ROOT / "sage" / "raising" / "analysis" / "s122_data"
S123_DIR = Path(__file__).resolve().parent

# Reuse S121b/S122 classifiers verbatim so the apples-to-apples comparison holds.
sys.path.insert(0, str(S121_DIR))
sys.path.insert(0, str(S122_DIR))
from s121b_self_vs_fleet_classifier import (  # type: ignore
    HW_TOKENS,
    MACHINE_HW_TRUTH,
    classify_mention,
    find_token_spans,
    instance_machine,
)
from s122_per_session_mode_decomposition import classify_session_mode  # type: ignore


# DAEMON-TIMEOUT SENTINEL: S122 found that some "silent" sessions are actually
# OllamaIRP-timeout sessions where SAGE never produced a real first response.
# S123 carries forward S122's filter rule: if the FIRST SAGE response is the
# 5-word timeout sentinel, classify as 'timeout' instead of 'silent' so
# population statistics aren't contaminated by infrastructure failures.
TIMEOUT_SENTINELS = [
    "[ollamairp: unexpected error: timed out]",
    "[ollamairp: unexpected error:",      # broader prefix in case of variants
    "[daemon unreachable:",
    "[ollamairp: connection",
]


INSTANCES = [
    "thor-qwen3.5-27b",
    "mcnugget-gemma3-12b",
    "cbp-qwen3.5-0.8b",
    "sprout-qwen3.5-0.8b",
]

# Modes used downstream — keep stable order so transition matrices are
# comparable across instances.
MODE_ORDER = ["silent", "timeout", "self_only", "self_dominant",
              "mixed", "fleet_dominant", "fleet_only"]


def load_all_sessions(inst_name: str) -> list:
    sess_dir = INSTANCES_DIR / inst_name / "sessions"
    if not sess_dir.exists():
        return []
    files = sorted(
        sess_dir.glob("session_*.json"),
        key=lambda p: int(re.search(r"session_(\d+)", p.stem).group(1)),
    )
    out = []
    for f in files:
        try:
            with open(f) as fh:
                out.append((f.stem, json.load(fh)))
        except Exception as e:
            print(f"  load fail {f}: {e}", file=sys.stderr)
    return out


def session_mode(sess: dict, machine: str, actual_family: str) -> tuple:
    """Classify one session. Returns (mode, counts_dict, n_sage, first_resp)."""
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
            cls = classify_mention(resp, span)
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

    # Apply timeout-sentinel filter only when classifier said 'silent'.
    if mode == "silent" and first_resp:
        first_lower = first_resp.lower().strip()
        if any(first_lower.startswith(s) for s in TIMEOUT_SENTINELS):
            mode = "timeout"

    return mode, counts, n_sage, first_resp[:200]


def classify_full_corpus(inst_name: str) -> dict:
    machine = instance_machine(inst_name)
    actual_family = MACHINE_HW_TRUTH[machine]["family"]
    all_sessions = load_all_sessions(inst_name)
    seq = []
    for stem, sess in all_sessions:
        sess_machine = sess.get("machine")
        if sess_machine and sess_machine != machine:
            continue
        sess_idx = int(re.search(r"session_(\d+)", stem).group(1))
        mode, counts, n_sage, first_resp = session_mode(sess, machine, actual_family)
        seq.append({
            "session": sess_idx,
            "mode": mode,
            "n_sage": n_sage,
            **counts,
            "first_resp_preview": first_resp,
        })
    return {
        "instance": inst_name,
        "machine": machine,
        "n_sessions": len(seq),
        "sessions": seq,
    }


def run_lengths(modes: list) -> list:
    """Return list of (mode, length) for each consecutive-same run."""
    if not modes:
        return []
    runs = []
    cur = modes[0]
    cur_len = 1
    for m in modes[1:]:
        if m == cur:
            cur_len += 1
        else:
            runs.append((cur, cur_len))
            cur = m
            cur_len = 1
    runs.append((cur, cur_len))
    return runs


def shuffled_run_lengths(modes: list, n_iter: int = 1000, seed: int = 42) -> dict:
    """Compute mean / max run lengths under random-shuffle null."""
    rng = random.Random(seed)
    longest_runs = []
    mean_runs = []
    n_runs = []
    for _ in range(n_iter):
        shuf = modes[:]
        rng.shuffle(shuf)
        rl = run_lengths(shuf)
        if rl:
            lengths = [L for _, L in rl]
            longest_runs.append(max(lengths))
            mean_runs.append(sum(lengths) / len(lengths))
            n_runs.append(len(rl))
    return {
        "mean_longest_run": sum(longest_runs) / max(1, len(longest_runs)),
        "mean_mean_run": sum(mean_runs) / max(1, len(mean_runs)),
        "mean_n_runs": sum(n_runs) / max(1, len(n_runs)),
        "n_iter": n_iter,
    }


def transition_matrix(modes: list) -> dict:
    """Build mode_t -> mode_{t+1} count matrix."""
    counts = defaultdict(lambda: Counter())
    for a, b in zip(modes[:-1], modes[1:]):
        counts[a][b] += 1
    # Materialize ordered matrix
    matrix = {}
    for src in MODE_ORDER:
        row = counts.get(src, Counter())
        total = sum(row.values())
        matrix[src] = {
            "counts": {dst: row.get(dst, 0) for dst in MODE_ORDER},
            "total": total,
            "probs": {dst: (row.get(dst, 0) / total if total else 0.0) for dst in MODE_ORDER},
        }
    return matrix


def chi_square_independence(modes: list) -> dict:
    """Test mode_{t+1} independence from mode_t with chi-square.

    Lazy implementation: collapse to 4-mode coarse grouping (silent, self, fleet,
    mixed) so cells aren't sparse, since some fine modes have <5 expected.

    Coarse mapping:
      timeout       -> silent (we already filtered, so this is residual)
      silent        -> silent
      self_only     -> self
      self_dominant -> self
      mixed         -> mixed
      fleet_dominant-> fleet
      fleet_only    -> fleet
    """
    coarse_map = {
        "silent": "silent", "timeout": "silent",
        "self_only": "self", "self_dominant": "self",
        "mixed": "mixed",
        "fleet_dominant": "fleet", "fleet_only": "fleet",
    }
    coarse = [coarse_map[m] for m in modes]
    coarse_modes = ["silent", "self", "mixed", "fleet"]
    n = len(coarse)
    if n < 3:
        return {"chi2": 0.0, "df": 0, "n_pairs": 0, "marginals": {}, "matrix": {},
                "p_lt_0_05": False, "p_lt_0_01": False, "note": "too few sessions"}

    # Marginal P(mode)
    marg = Counter(coarse)
    p_marg = {m: marg[m] / n for m in coarse_modes}
    # Pair counts
    pair_counts = defaultdict(lambda: Counter())
    for a, b in zip(coarse[:-1], coarse[1:]):
        pair_counts[a][b] += 1

    # Chi^2 = sum over (i,j) of (O_ij - E_ij)^2 / E_ij where
    #   O_ij = count(t=i, t+1=j)
    #   E_ij = row_total_i * col_total_j / n_pairs
    n_pairs = n - 1
    row_totals = {m: sum(pair_counts[m].values()) for m in coarse_modes}
    col_totals = Counter()
    for m in coarse_modes:
        for dst, c in pair_counts[m].items():
            col_totals[dst] += c

    chi2 = 0.0
    cells_used = 0
    for i in coarse_modes:
        for j in coarse_modes:
            o = pair_counts[i].get(j, 0)
            e = row_totals[i] * col_totals[j] / max(1, n_pairs)
            if e > 0:
                chi2 += (o - e) ** 2 / e
                cells_used += 1

    # df for r×c contingency = (r-1)*(c-1) but with empty rows/cols dropped.
    nz_rows = sum(1 for m in coarse_modes if row_totals[m] > 0)
    nz_cols = sum(1 for m in coarse_modes if col_totals[m] > 0)
    df = max(1, (nz_rows - 1) * (nz_cols - 1))

    # Critical values for common df at p=0.05 and p=0.01
    crit_05 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 6: 12.592, 9: 16.919}
    crit_01 = {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 6: 16.812, 9: 21.666}

    return {
        "chi2": round(chi2, 3),
        "df": df,
        "n_pairs": n_pairs,
        "p_lt_0_05": chi2 > crit_05.get(df, 9.488),
        "p_lt_0_01": chi2 > crit_01.get(df, 13.277),
        "crit_05": crit_05.get(df),
        "crit_01": crit_01.get(df),
        "marginals_pct": {m: round(100 * p_marg[m], 1) for m in coarse_modes},
        "row_totals": dict(row_totals),
        "col_totals": dict(col_totals),
        "pair_counts": {i: {j: pair_counts[i].get(j, 0) for j in coarse_modes}
                         for i in coarse_modes},
    }


def stationarity_thirds(modes: list, sess_indices: list) -> dict:
    """Split corpus into thirds by session-index ordering, compare distributions."""
    if len(modes) < 6:
        return {"note": "too few sessions"}
    n = len(modes)
    third = n // 3
    splits = {
        "early": modes[:third],
        "middle": modes[third:2 * third],
        "late":  modes[2 * third:],
    }
    split_idx = {
        "early":  (sess_indices[0], sess_indices[third - 1]),
        "middle": (sess_indices[third], sess_indices[2 * third - 1]),
        "late":   (sess_indices[2 * third], sess_indices[-1]),
    }
    out = {}
    for name, ms in splits.items():
        c = Counter(ms)
        total = max(1, sum(c.values()))
        out[name] = {
            "n_sessions": len(ms),
            "session_range": split_idx[name],
            "counts": {m: c.get(m, 0) for m in MODE_ORDER},
            "pct": {m: round(100 * c.get(m, 0) / total, 1) for m in MODE_ORDER},
        }
    # Coarse trend deltas
    def coarse_pct(c: Counter, total: int):
        return {
            "silent_or_timeout": round(100 * (c.get("silent", 0) + c.get("timeout", 0)) / total, 1),
            "self_anchored":    round(100 * (c.get("self_only", 0) + c.get("self_dominant", 0)) / total, 1),
            "fleet_anchored":   round(100 * (c.get("fleet_only", 0) + c.get("fleet_dominant", 0)) / total, 1),
            "mixed":            round(100 * c.get("mixed", 0) / total, 1),
        }
    coarse = {
        name: coarse_pct(Counter(ms), max(1, len(ms)))
        for name, ms in splits.items()
    }
    out["coarse_pct"] = coarse
    return out


def main():
    print("S123 — Full-corpus mode-run dynamics across raising trajectory")
    print()

    seq_results = {}
    transition_results = {}
    chi2_results = {}
    stationarity_results = {}

    for inst in INSTANCES:
        print(f"=== {inst} ===")
        full = classify_full_corpus(inst)
        modes = [s["mode"] for s in full["sessions"]]
        sess_indices = [s["session"] for s in full["sessions"]]

        # 1) Mode marginals on full corpus.
        c = Counter(modes)
        n = sum(c.values())
        marg_pct = {m: round(100 * c[m] / max(1, n), 1) for m in MODE_ORDER}
        print(f"  N = {n} sessions, marginals: {marg_pct}")

        # 2) Run lengths (observed vs shuffle null)
        rl = run_lengths(modes)
        if rl:
            lengths = [L for _, L in rl]
            obs_longest = max(lengths)
            obs_mean = sum(lengths) / len(lengths)
            obs_n_runs = len(rl)
        else:
            obs_longest = obs_mean = obs_n_runs = 0
        null = shuffled_run_lengths(modes, n_iter=1000, seed=42)
        print(f"  RUNS: obs longest={obs_longest}, mean={obs_mean:.2f}, n_runs={obs_n_runs}")
        print(f"        null longest={null['mean_longest_run']:.2f}, mean={null['mean_mean_run']:.2f}, "
              f"n_runs={null['mean_n_runs']:.2f}")

        # Most-common mode, longest run by mode
        rl_by_mode = defaultdict(list)
        for m, L in rl:
            rl_by_mode[m].append(L)
        per_mode_runs = {m: {"max": max(rl_by_mode[m]), "mean": round(sum(rl_by_mode[m]) / len(rl_by_mode[m]), 2),
                              "count": len(rl_by_mode[m])}
                         for m in rl_by_mode}
        print(f"        per-mode runs: {per_mode_runs}")

        # 3) Markov transition matrix + chi^2 independence test
        tm = transition_matrix(modes)
        chi2 = chi_square_independence(modes)
        print(f"  CHI^2: {chi2['chi2']} (df={chi2['df']}, "
              f"crit_05={chi2['crit_05']}, p<0.05? {chi2['p_lt_0_05']})")
        # Print coarse 4x4 matrix for quick eyeballing
        cmap = {"silent": "silent", "timeout": "silent",
                "self_only": "self", "self_dominant": "self",
                "mixed": "mixed",
                "fleet_dominant": "fleet", "fleet_only": "fleet"}
        coarse = [cmap[m] for m in modes]
        coarse_modes = ["silent", "self", "mixed", "fleet"]
        ctm = defaultdict(lambda: Counter())
        for a, b in zip(coarse[:-1], coarse[1:]):
            ctm[a][b] += 1
        print(f"  Coarse 1-step transitions ({coarse_modes}):")
        for src in coarse_modes:
            row = ctm[src]
            tot = max(1, sum(row.values()))
            line = "    " + f"{src:>7s} -> "
            line += "  ".join(
                f"{dst[:3]}:{row.get(dst, 0):>3d}({100 * row.get(dst, 0) / tot:>4.0f}%)"
                for dst in coarse_modes
            )
            line += f"   N={sum(row.values())}"
            print(line)

        # 4) Stationarity thirds
        stat = stationarity_thirds(modes, sess_indices)
        print(f"  STATIONARITY (early/middle/late, {n // 3} sessions each):")
        if "coarse_pct" in stat:
            for name in ["early", "middle", "late"]:
                if name in stat:
                    line = (f"    {name:>6s} ({stat[name]['session_range'][0]:>3d}-"
                            f"{stat[name]['session_range'][1]:>3d}): "
                            f"silent/timeout={stat['coarse_pct'][name]['silent_or_timeout']}%, "
                            f"self={stat['coarse_pct'][name]['self_anchored']}%, "
                            f"fleet={stat['coarse_pct'][name]['fleet_anchored']}%, "
                            f"mixed={stat['coarse_pct'][name]['mixed']}%")
                    print(line)

        seq_results[inst] = {
            "instance": inst,
            "n_sessions": n,
            "marginals_pct": marg_pct,
            "modes": modes,
            "session_indices": sess_indices,
            "run_lengths_observed": rl,
            "run_length_null": null,
            "per_mode_runs": per_mode_runs,
            "obs_longest_run": obs_longest,
            "obs_mean_run": obs_mean,
            "obs_n_runs": obs_n_runs,
        }
        transition_results[inst] = tm
        chi2_results[inst] = chi2
        stationarity_results[inst] = stat
        print()

    # Save outputs
    seq_path = S123_DIR / "s123_mode_sequence.json"
    with open(seq_path, "w") as f:
        json.dump(seq_results, f, indent=2)
    print(f"Saved: {seq_path}")

    trans_path = S123_DIR / "s123_markov_transitions.json"
    with open(trans_path, "w") as f:
        json.dump({"transitions": transition_results, "chi2": chi2_results}, f, indent=2)
    print(f"Saved: {trans_path}")

    stat_path = S123_DIR / "s123_stationarity.json"
    with open(stat_path, "w") as f:
        json.dump(stationarity_results, f, indent=2)
    print(f"Saved: {stat_path}")

    print()
    print("=== Cross-instance summary ===")
    for inst in INSTANCES:
        n = seq_results[inst]["n_sessions"]
        marg = seq_results[inst]["marginals_pct"]
        chi2 = chi2_results[inst]
        print(f"  {inst:>30s}  N={n:>3d}  silent/timeout={marg.get('silent', 0) + marg.get('timeout', 0):>5.1f}%"
              f"  chi^2={chi2['chi2']:>7.2f} (p<0.05? {chi2['p_lt_0_05']})")


if __name__ == "__main__":
    main()
