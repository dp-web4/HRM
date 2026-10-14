"""Read-only cost/receipt census of heartbeat/v2 JSONL; no model or gateway imports.

Usage: python3 -m sage.gateway.attempt_baseline heartbeats.jsonl
       tail -n 100 heartbeats.jsonl | python3 -m sage.gateway.attempt_baseline

Attempt arrays replace (never supplement) legacy final-response token counters.
Unknown counts are not zero. Completeness means coverage of RECORDED generations,
not a crash-durable provider ledger, comparable experimental arms, or task success.
Receipt statuses are reported claims, not independently verified effects. Repeated
proposals can be legitimate reads/retries; they are not classified as wasted acts.
Only aggregates are emitted: no prompts, arguments, replies, paths or identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys

PHASES = ("explore", "posture", "account", "reflect", "answer", "act_after_answer")
FIELDS = ("generates", "legacy_generates", "attempts", "input_tokens_known",
          "output_tokens_known", "input_counts_missing", "output_counts_missing",
          "nonfinal_attempts", "nonfinal_output_tokens_known", "nonfinal_output_counts_missing",
          "transport_errors", "length_stops", "retried_generates", "receipts",
          "reported_ok", "reported_refused", "reported_pending", "reported_failed",
          "receipt_unknown", "receipt_conflicting", "witness_refs_present",
          "repeated_proposals")


def _count(value):
    return type(value) is int and value >= 0


def _cost(phase, out, issues):
    if phase.get("error"):
        issues["phase_error"] += 1
    gens = phase.get("generates")
    if not isinstance(gens, list):
        issues["missing_or_invalid_generates"] += 1
        return
    for gen in gens:
        out["generates"] += 1
        if not isinstance(gen, dict):
            issues["invalid_generate"] += 1
            continue
        if _count(gen.get("retried")) and gen["retried"] > 0:
            out["retried_generates"] += 1
        if "attempts" not in gen:
            out["legacy_generates"] += 1
            issues["legacy_generate"] += 1
            continue
        attempts = gen["attempts"]
        if not isinstance(attempts, list) or not attempts:
            issues["empty_or_invalid_attempts"] += 1
            continue
        retried = gen.get("retried", 0)
        if not _count(retried):
            issues["invalid_retry_count"] += 1
        elif len(attempts) < retried + 1:
            issues["retry_attempts_missing"] += 1
        for index, attempt in enumerate(attempts):
            out["attempts"] += 1
            if not isinstance(attempt, dict):
                issues["invalid_attempt"] += 1
                attempt = {}
            for key, label in (("prompt_eval_count", "input"), ("eval_count", "output")):
                value = attempt.get(key)
                if _count(value):
                    out[f"{label}_tokens_known"] += value
                else:
                    out[f"{label}_counts_missing"] += 1
                    issues[f"{label}_count_missing_or_invalid"] += 1
            if index < len(attempts) - 1:
                out["nonfinal_attempts"] += 1
                if _count(attempt.get("eval_count")):
                    out["nonfinal_output_tokens_known"] += attempt["eval_count"]
                else:
                    out["nonfinal_output_counts_missing"] += 1
            out["transport_errors"] += attempt.get("transport_error") is True
            out["length_stops"] += attempt.get("done_reason") == "length"


def _receipts(phase, out, seen):
    trace = phase.get("trace", [])
    if not isinstance(trace, list):
        out["receipt_unknown"] += 1
        return
    for receipt in trace:
        out["receipts"] += 1
        if not isinstance(receipt, dict):
            out["receipt_unknown"] += 1
            continue
        ok, refused, pending = (receipt.get(k) for k in ("ok", "refused", "pending"))
        if any(v is not None and type(v) is not bool for v in (ok, refused, pending)):
            out["receipt_unknown"] += 1
        elif sum(v is True for v in (ok, refused, pending)) > 1:
            out["receipt_conflicting"] += 1
        elif pending is True:
            out["reported_pending"] += 1
        elif refused is True:
            out["reported_refused"] += 1
        elif ok is True:
            out["reported_ok"] += 1
        elif ok is False:
            out["reported_failed"] += 1
        else:
            out["receipt_unknown"] += 1
        if isinstance(receipt.get("witness_id"), str) and receipt["witness_id"].strip():
            out["witness_refs_present"] += 1
        if isinstance(receipt.get("effector"), str) and isinstance(receipt.get("args"), dict):
            # Per-beat only; these strings are neither retained nor printed in the report.
            key = (receipt["effector"], json.dumps(receipt["args"], sort_keys=True))
            out["repeated_proposals"] += key in seen
            seen.add(key)


def _view(counts, issues):
    result = {k: counts[k] for k in FIELDS}
    complete = bool(counts["generates"]) and not issues
    result.update(recorded_cost_complete=complete,
                  input_tokens_total=counts["input_tokens_known"] if complete else None,
                  output_tokens_total=counts["output_tokens_known"] if complete else None,
                  cost_issues=dict(issues))
    return result


def summarize(lines):
    """Stream JSONL once, with bounded aggregate state and per-beat duplicate tracking."""
    total, issues, coverage = Counter(), Counter(), Counter()
    phases = {p: (Counter(), Counter()) for p in PHASES}
    cohorts = {k: (Counter(), Counter()) for k in ("complete_recorded", "incomplete", "no_generations")}
    malformed = beats = 0
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError):
            malformed += 1
            continue
        if not isinstance(record, dict):
            malformed += 1
            continue
        beats += 1
        counts, problems, seen = Counter(), Counter(), set()
        if record.get("schema") != "heartbeat/v2":
            problems["unknown_schema"] += 1
        if record.get("killed"):
            problems["killed_beat"] += 1
        # Do not recursively scan arbitrary tool results: nested generates are data,
        # not another provider ledger. Flag new phases instead of silently dropping them.
        for key, value in record.items():
            if key not in PHASES and isinstance(value, dict) and "generates" in value:
                problems["unknown_generation_phase"] += 1
        for name in PHASES:
            phase = record.get(name)
            if phase is None:
                continue
            pc, pi = Counter(), Counter()
            if not isinstance(phase, dict):
                pi["invalid_phase"] += 1
            else:
                _cost(phase, pc, pi)
                _receipts(phase, pc, seen)
            phases[name][0].update(pc)
            phases[name][1].update(pi)
            counts.update(pc)
            problems.update(pi)
        kind = "incomplete" if problems else "complete_recorded" if counts["generates"] else "no_generations"
        coverage[kind] += 1
        cohorts[kind][0].update(counts)
        cohorts[kind][1].update(problems)
        total.update(counts)
        issues.update(problems)
    if malformed:
        issues["malformed_jsonl_rows"] += malformed
    return {
        "schema": "attempt-baseline/v1", "beats": beats, "malformed_rows": malformed,
        "input_text_sha256": digest.hexdigest(),
        "coverage": {k: coverage[k] for k in cohorts}, "totals": _view(total, issues),
        "cohorts": {k: _view(*v) for k, v in cohorts.items()},
        "phases": {k: _view(*v) for k, v in phases.items()},
        "limits": ["Recorded generations only; crashes before publication can hide calls.",
                   "Reported receipt status and witness references are not verified effects or task success.",
                   "Cohorts separate cost coverage, not model/config/task differences or experimental arms.",
                   "Repeated proposals are not necessarily duplicate executions or waste.",
                   "Nonfinal attempts include useful JSON selection/arguments, not just discarded retries."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="JSONL file, or - for stdin")
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            result = summarize(sys.stdin)
        else:
            with open(args.input, encoding="utf-8", errors="replace") as stream:
                result = summarize(stream)
    except OSError:
        parser.error("could not read input")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
