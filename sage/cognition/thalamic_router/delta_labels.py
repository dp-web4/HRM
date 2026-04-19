"""Derive retrain labels from delta-record streams.

The invoke-head's self-supervised label (first step / level change / ≥10%
pixel diff) describes when the FRAME warrants attention. It cannot see
when SAGE's *own play* got stuck — because that signal only exists after
the model has acted on the world.

Delta records capture exactly that: per (game, game_id, step_index), what
SAGE did and what happened. This module aggregates those into per-step
overrides that the training pipeline folds into the invoke label.

Inputs:
    One or more gzipped JSONL streams written by RouterDatasetWriter
    (subdirs: gameplay, llm_dispatch, sage_plays_live, sage_plays_self).

Output:
    Dict keyed by (game, game_id, step_index) → DeltaLabel with
    invoke_target and a reason code. Train-time lookup lets us override
    the self-supervised label when the substrate has outcome evidence.

Design notes:
    - Step index convention matches the solver trace: step_index 1 = first
      action. Frames align because gameplay/sage_plays streams replay
      solver traces.
    - sage_plays_self records come from real play, not trace replay, and
      step_index there is SAGE's step, not the solver's. We still fold
      them in because the pattern "NN got stuck at game X level Y step N"
      is a transferable prior even when step N doesn't literally align.
    - We prefer positive invoke labels when the substrate disagrees with
      self-supervision (stuck=True but pixel_diff<10%). Missed-invoke is
      a more costly error than over-invoke — over-invoke wastes a few
      Claude calls, missed-invoke loses levels.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DeltaLabel:
    """Per-step retrain label derived from substrate evidence.

    invoke_target and action_target are independent override signals; either
    (or both) may be None for a given step.
    """
    invoke_target: Optional[float] = None  # 0.0 or 1.0 when substrate has signal
    action_target: Optional[int] = None    # override action label if LLM's choice advanced state
    reason: str = ""                       # short code for diagnostics
    source_records: int = 1                # how many delta records voted for this label


def _get_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("metadata") or {}


def _sage_plays_self_label(record: Dict[str, Any]) -> Optional[DeltaLabel]:
    """Extract invoke/action label from a sage_plays_self or llm_dispatch record.

    Positive invoke cases:
      - stuck_triggered=True at this step → the NN's next-action will fail
      - llm_invoked=True AND levels_after > levels_before → LLM saved the run

    Positive action cases (LLM action as supervised target):
      - llm_invoked=True AND state_after != GAME_OVER AND
        (levels_after > levels_before OR stuck_triggered was resolved)
    """
    md = _get_meta(record)
    sd = md.get("sage_plays_self")
    if not isinstance(sd, dict):
        return None

    stuck = bool(sd.get("stuck_triggered"))
    llm_invoked = bool(sd.get("llm_invoked"))
    la, lb = sd.get("levels_after"), sd.get("levels_before")
    levels_up = (isinstance(la, int) and isinstance(lb, int) and la > lb)
    state_after = sd.get("state_after")
    action_target: Optional[int] = None
    reason: str = ""
    invoke: Optional[float] = None

    if stuck:
        invoke = 1.0
        reason = "stuck_triggered"
    if llm_invoked and levels_up:
        invoke = 1.0
        llm_action = sd.get("llm_action")
        if isinstance(llm_action, int) and 0 <= llm_action < 7:
            action_target = llm_action
        reason = "llm_invoke_advanced"
    # Negative invoke evidence: NN played AND levels advanced AND not stuck.
    # The NN was right; don't escalate invoke here.
    if (not llm_invoked) and levels_up and (not stuck):
        invoke = 0.0
        reason = "nn_correct_advance"
    # Broader negative evidence (added for v8 to counter invoke_head paranoia):
    # NN played, no stuck, state didn't degrade. v7's 695+/1- skew taught the
    # invoke_head to over-trigger; these counter-examples rebalance the signal.
    # NB: absence of progress is NOT evidence invoke would've helped — the LLM
    # might have failed too. We only treat "NN played AND things kept moving
    # AND no stuck" as weak negative.
    state_ok = state_after not in ("GAME_OVER",)
    decision = sd.get("decision")
    if invoke is None and decision == "play" and (not stuck) and state_ok:
        invoke = 0.0
        reason = "nn_played_no_stuck"

    if invoke is None:
        return None
    return DeltaLabel(invoke_target=invoke, action_target=action_target, reason=reason)


def _gameplay_label(record: Dict[str, Any]) -> Optional[DeltaLabel]:
    """Extract action label from a gameplay record (teacher-forced replay).

    gameplay records carry `known_good_action` — the solver's choice. This
    is already what frame_train uses via the trace directly, but the delta
    stream can fill in gaps for partial traces.
    """
    md = _get_meta(record)
    if "known_good_action" not in md:
        return None
    kga = md.get("known_good_action")
    if not isinstance(kga, int) or not (0 <= kga < 7):
        return None
    # gameplay records don't carry direct invoke signal — only action truth
    return DeltaLabel(invoke_target=None, action_target=kga, reason="solver_known_good")


def _sage_plays_live_label(record: Dict[str, Any]) -> Optional[DeltaLabel]:
    """sage_plays_live carries NN-vs-solver comparisons.

    `predicted_rank_of_known_good` tells us how wrong the NN was:
      0 = NN's top pick matched solver (correct)
      >0 = solver's choice was ranked lower by the NN

    Since we ALREADY train action_head on the solver's choice via the
    trace replay, this stream is mostly redundant for action. We do use
    it for *diagnostics* — records where rank >> 0 indicate games/levels
    where the NN has weak priors, worth boosting invoke.
    """
    md = _get_meta(record)
    live = md.get("sage_plays_live")
    if not isinstance(live, dict):
        return None
    kga = md.get("known_good_action")
    if not isinstance(kga, int) or not (0 <= kga < 7):
        return None
    rank = live.get("predicted_rank_of_known_good", 0)
    # If NN ranked the solver's pick 3rd or worse, the scalar/frame context
    # at this step isn't a good teacher for the action head — and it's
    # exactly the state where we'd want to invoke an LLM in live play.
    if isinstance(rank, int) and rank >= 3:
        return DeltaLabel(
            invoke_target=1.0, action_target=kga,
            reason=f"nn_rank_{rank}_known_good",
        )
    return None


def _step_key(record: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
    """Return (game_family, game_id, step_index) or None if incomplete."""
    md = _get_meta(record)
    game = md.get("game")
    game_id = md.get("game_id")
    step = md.get("step_index")
    if isinstance(game, str) and isinstance(game_id, str) and isinstance(step, int):
        return (game, game_id, step)
    return None


def load_delta_labels(
    stream_paths: List[Path],
    verbose: bool = False,
) -> Dict[Tuple[str, str, int], DeltaLabel]:
    """Read all gzipped streams and aggregate per-step labels.

    When multiple records cover the same (game, game_id, step_index), the
    aggregation rule is max-invoke (any positive signal wins) and first-
    action-target (earliest-seen LLM correction).
    """
    agg: Dict[Tuple[str, str, int], DeltaLabel] = {}
    counts_by_source: Dict[str, int] = defaultdict(int)
    counts_by_reason: Dict[str, int] = defaultdict(int)

    for path in stream_paths:
        path = Path(path)
        if not path.exists():
            continue
        open_fn = gzip.open if path.suffix == ".gz" else open
        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = _get_meta(record).get("source", "")
                counts_by_source[source] += 1

                label = (
                    _sage_plays_self_label(record)
                    or _sage_plays_live_label(record)
                    or _gameplay_label(record)
                )
                if label is None:
                    continue
                key = _step_key(record)
                if key is None:
                    continue

                counts_by_reason[label.reason] += 1
                existing = agg.get(key)
                if existing is None:
                    agg[key] = label
                else:
                    # Aggregation: max invoke (any positive signal wins);
                    # prefer non-None action target; concat reasons.
                    if existing.invoke_target is None:
                        merged_invoke = label.invoke_target
                    elif label.invoke_target is None:
                        merged_invoke = existing.invoke_target
                    else:
                        merged_invoke = max(existing.invoke_target, label.invoke_target)
                    merged_action = existing.action_target
                    if merged_action is None:
                        merged_action = label.action_target
                    merged_reason = (
                        f"{existing.reason}+{label.reason}"
                        if existing.reason != label.reason else existing.reason
                    )
                    agg[key] = DeltaLabel(
                        invoke_target=merged_invoke,
                        action_target=merged_action,
                        reason=merged_reason,
                        source_records=existing.source_records + 1,
                    )

    if verbose:
        print(f"Loaded {sum(counts_by_source.values())} records across "
              f"{len(stream_paths)} streams:")
        for src, n in sorted(counts_by_source.items(), key=lambda x: -x[1]):
            print(f"  {src or '<none>':25s} {n}")
        print(f"\nDistinct labeled step-keys: {len(agg)}")
        print("Labels by reason:")
        for reason, n in sorted(counts_by_reason.items(), key=lambda x: -x[1]):
            print(f"  {reason:30s} {n}")

    return agg


def summarize_labels(
    labels: Dict[Tuple[str, str, int], DeltaLabel],
) -> Dict[str, Any]:
    """Produce a diagnostic summary of the label set."""
    if not labels:
        return {"total": 0}

    total = len(labels)
    invoke_pos = sum(
        1 for L in labels.values()
        if L.invoke_target is not None and L.invoke_target >= 0.5
    )
    invoke_neg = sum(
        1 for L in labels.values()
        if L.invoke_target is not None and L.invoke_target < 0.5
    )
    invoke_unset = sum(1 for L in labels.values() if L.invoke_target is None)
    action_overrides = sum(1 for L in labels.values() if L.action_target is not None)

    by_game: Dict[str, int] = defaultdict(int)
    invoke_pos_by_game: Dict[str, int] = defaultdict(int)
    for (game, _, _), L in labels.items():
        by_game[game] += 1
        if L.invoke_target is not None and L.invoke_target >= 0.5:
            invoke_pos_by_game[game] += 1

    return {
        "total": total,
        "invoke_positive": invoke_pos,
        "invoke_negative": invoke_neg,
        "invoke_unset": invoke_unset,
        "invoke_positive_pct_of_set": (
            invoke_pos / (invoke_pos + invoke_neg) * 100
            if (invoke_pos + invoke_neg) else 0.0
        ),
        "action_overrides": action_overrides,
        "by_game": dict(by_game),
        "invoke_positive_by_game": dict(invoke_pos_by_game),
    }


def default_stream_paths(training_data_root: Path) -> List[Path]:
    """Return all known delta-record streams under a training-data root."""
    training_data_root = Path(training_data_root)
    subdirs = ["gameplay", "llm_dispatch", "sage_plays_live", "sage_plays_self"]
    paths: List[Path] = []
    for sub in subdirs:
        d = training_data_root / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl.gz")):
            paths.append(f)
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate delta labels from stream files.")
    ap.add_argument(
        "--training-data-root",
        default="/mnt/c/exe/projects/ai-agents/private-context/training-data/router/cbp",
        help="Root containing gameplay/ llm_dispatch/ sage_plays_live/ sage_plays_self/",
    )
    args = ap.parse_args()

    paths = default_stream_paths(Path(args.training_data_root))
    print(f"Found {len(paths)} stream files under {args.training_data_root}")
    labels = load_delta_labels(paths, verbose=True)
    summary = summarize_labels(labels)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
