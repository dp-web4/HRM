#!/usr/bin/env python3
"""sage_plays_live — teacher-forced live play harness.

At each step of a winning trace:
  1. Synthesize RouterInput from the CURRENT env state (live, not captured)
  2. Ask the promoted adapter to propose an action class
  3. Emit a delta record: (proposed, known_good, rank, confidences, ...)
  4. Advance env with `known_good` (teacher forcing — errors don't compound)

This differs from sage_plays.py in one critical way: sage_plays re-scores
already-captured records offline. This module drives the env live and has
SAGE compute its proposal from the live state at each tick. The env is
always advanced by the solver's action, never SAGE's, so we stay inside
the solver's winnable trajectory. Every step has a defined "right" answer.

Why teacher forcing at this stage:
  - If we let SAGE's wrong choices drive the env, we'd create states the
    solver never visited, where no `known_good` is defined. No ground
    truth → no delta → no learning signal.
  - Teacher forcing keeps us on the winnable manifold. Delta = "did SAGE
    choose sagely at each state it actually sees in winning play."
  - Full play (adapter drives env, errors compound, measure win rate) is
    a later phase once coord regression + motor tier exist.

Output: {machine}/sage_plays_live/{date}.jsonl.gz — RouterRecord with
source="sage_plays_live" and metadata.sage_plays_live = {proposed, rank,
confidences, entropy, softmax_probs, teacher_action_applied}.

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-1.5-sage-plays-plan.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sage.cognition.router.data import RouterDatasetWriter
from sage.cognition.router.record import RouterRecord
from sage.cognition.thalamic_router.phase1_training import (
    HeadBAdapter, _feature_vec, HEAD_B_ACTION_NAMES,
)
from sage.cognition.thalamic_router.sage_plays import _compute_delta
from sage.cognition.thalamic_router.gameplay_capture import (
    Trace, load_trace, synth_router_input, _discover_trace,
    ARC_AGI_EXPERIMENTS,
)


@dataclass
class LiveResult:
    game: str
    game_id: str
    records_emitted: int
    steps_total: int
    steps_applied: int
    correct: int
    final_state: Optional[str]
    final_levels_completed: Optional[int]
    rank_distribution: Dict[int, int] = field(default_factory=dict)
    per_action: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def _make_env(trace: Trace):
    """Same version-fallback pattern as gameplay_capture._make_env."""
    if str(ARC_AGI_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(ARC_AGI_EXPERIMENTS))
    from arc_agi import Arcade
    arc = Arcade(operation_mode="offline")
    env = arc.make(trace.game_id)
    fallback_note = None
    if env is None:
        env = arc.make(trace.game)
        if env is not None:
            fallback_note = (f"version_fallback: trace pinned {trace.game_id}, "
                             f"using latest local version for family {trace.game}")
    if env is None:
        raise RuntimeError(
            f"arc.make returned None for both {trace.game_id} and {trace.game}"
        )
    fd = env.reset()
    return env, fd, fallback_note


def run_teacher_forced(
    trace: Trace, adapter: HeadBAdapter, writer: RouterDatasetWriter,
    machine: str, atp_initial: float = 100.0, atp_decay: float = 0.1,
) -> LiveResult:
    """Teacher-forced play: SAGE proposes, teacher advances, delta recorded."""
    errors: List[str] = []
    try:
        env, fd, fallback_note = _make_env(trace)
        if fallback_note:
            errors.append(fallback_note)
    except Exception as e:
        return LiveResult(
            game=trace.game, game_id=trace.game_id,
            records_emitted=0, steps_total=len(trace.steps),
            steps_applied=0, correct=0,
            final_state=None, final_levels_completed=None,
            errors=[f"env init failed: {e!r}"],
        )

    try:
        from arcengine import GameAction
        int_to_action = {ga.value: ga for ga in GameAction}
    except Exception:
        int_to_action = {}

    steps_applied = 0
    records_emitted = 0
    correct = 0
    rank_counts: Dict[int, int] = {}
    per_action: Dict[int, Dict[str, int]] = {}

    atp = atp_initial
    prev_frame = getattr(fd, "frame", None)

    for step in trace.steps:
        curr_frame = getattr(fd, "frame", None)

        # Build RouterInput from CURRENT env state (live — not captured)
        router_input = synth_router_input(
            tick=step.index, prev_frame=prev_frame, curr_frame=curr_frame,
            step=step, game=trace.game, level=step.level, atp=atp,
        )

        # Extract feature vector from the live RouterInput
        ri_dict = {}
        for attr in ("snarc_surprise", "snarc_novelty", "snarc_arousal",
                     "snarc_reward", "snarc_conflict", "sensory_novelty",
                     "sensory_urgency", "atp_level", "wm_goal_active",
                     "wm_pressure", "habit_available", "habit_confidence",
                     "sensory_modalities", "metabolic_state"):
            ri_dict[attr] = getattr(router_input, attr, None)
        features = _feature_vec({"router_input": ri_dict})

        # Adapter proposes; compute delta against known_good (step.action)
        delta = _compute_delta(features, step.action, adapter)
        if delta["correct"]:
            correct += 1
        rank = delta["predicted_rank_of_known_good"]
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        bucket = per_action.setdefault(step.action, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if delta["correct"]:
            bucket["correct"] += 1

        # Use a minimal RouterOutput just to satisfy schema — this record is
        # about the adapter's action proposal, not the programmatic dispatch.
        from sage.cognition.router.outputs import RouterOutput
        router_output = RouterOutput.noop(rationale_code="sage_plays_live")

        metadata = {
            "source": "sage_plays_live",
            "game": trace.game,
            "game_id": trace.game_id,
            "replay_source": trace.source,
            "step_index": step.index,
            "level": step.level,
            "synthetic_kernel_state": True,
            "known_good_action": step.action,
            "known_good_data": step.data,
            "known_good_level": step.level,
            "sage_plays_live": {
                "proposed_action": delta["proposed_action"],
                "proposed_name": delta["proposed_name"],
                "known_good_name": delta["known_good_name"],
                "correct": delta["correct"],
                "predicted_rank_of_known_good": delta["predicted_rank_of_known_good"],
                "confidence_on_known_good": delta["confidence_on_known_good"],
                "confidence_on_proposed": delta["confidence_on_proposed"],
                "entropy": delta["entropy"],
                "teacher_action_applied": step.action,   # env advances on this
            },
            "sage_plays_probs": delta["softmax_probs"],
            "adapter_machine": adapter.machine,
            "adapter_trained_at": adapter.trained_at,
        }

        record = RouterRecord(
            router_input=router_input, router_output=router_output,
            machine=machine, timestamp=time.time(), metadata=metadata,
        )
        try:
            writer.append(record.to_dict())
            records_emitted += 1
        except Exception as e:
            errors.append(f"step {step.index}: write failed: {e!r}")

        # Teacher-forced advance — always use known_good, never proposed
        try:
            ga = int_to_action.get(step.action)
            if ga is not None:
                fd = env.step(ga, data=step.data) if step.data else env.step(ga)
                steps_applied += 1
                prev_frame = curr_frame
                atp = max(10.0, atp - atp_decay)
        except Exception as e:
            errors.append(f"step {step.index}: env.step failed: {e!r}")

    final_state = getattr(getattr(fd, "state", None), "name", None)
    final_levels = getattr(fd, "levels_completed", None)

    return LiveResult(
        game=trace.game, game_id=trace.game_id,
        records_emitted=records_emitted, steps_total=len(trace.steps),
        steps_applied=steps_applied, correct=correct,
        final_state=final_state, final_levels_completed=final_levels,
        rank_distribution=dict(sorted(rank_counts.items())),
        per_action={
            HEAD_B_ACTION_NAMES[k] if 0 <= k < len(HEAD_B_ACTION_NAMES) else str(k): {
                **v, "recall": v["correct"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in sorted(per_action.items())
        },
        errors=errors,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True,
                   help="Path to Head B adapter JSON.")
    p.add_argument("--game", required=True, help="Game family (e.g. cd82).")
    p.add_argument("--game-id", default=None,
                   help="Full versioned id. Looked up from game_coordination.json if omitted.")
    p.add_argument("--trace", default=None,
                   help="Explicit trace path. Auto-discovered if omitted.")
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"))
    p.add_argument("--data-dir", default=os.environ.get(
        "SAGE_ROUTER_DATA_DIR",
        "/mnt/c/exe/projects/ai-agents/private-context/training-data/router"))
    p.add_argument("--json-out", default=None,
                   help="Write result summary to this path.")
    args = p.parse_args()

    # Resolve game_id via the same search pattern gameplay_capture uses
    game_id = args.game_id
    if not game_id:
        for root in [
            Path(os.environ.get("ARC_SAGE_DIR", "")),
            Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE"),
            Path.home() / "ai-workspace" / "ARC-SAGE",
            Path.home() / "repos" / "ARC-SAGE",
        ]:
            coord = root / "knowledge" / "game_coordination.json"
            if coord.exists():
                try:
                    for g in json.loads(coord.read_text()).get("games", []):
                        if g.get("family") == args.game:
                            game_id = g.get("id")
                            break
                except Exception:
                    pass
                if game_id:
                    break
        if not game_id:
            print(f"ERROR: could not resolve game_id for {args.game}")
            return 1

    trace_path = Path(args.trace) if args.trace else _discover_trace(args.game, game_id)
    if not trace_path or not trace_path.exists():
        print(f"ERROR: no trace found for {args.game}")
        return 1
    print(f"Loading trace: {trace_path}")
    trace = load_trace(trace_path, args.game, game_id)
    print(f"  {len(trace.steps)} steps, outcome={trace.outcome.get('inferred_won')}")

    adapter = HeadBAdapter.load(Path(args.adapter))
    print(f"Loaded adapter: machine={adapter.machine}, trained_at={adapter.trained_at}")

    writer = RouterDatasetWriter(
        base_dir=Path(args.data_dir), machine=args.machine,
        compress=True, subdir="sage_plays_live",
    )
    try:
        result = run_teacher_forced(trace, adapter, writer, machine=args.machine)
    finally:
        writer.close()

    acc = result.correct / result.records_emitted if result.records_emitted else 0.0
    total_ranks = sum(result.rank_distribution.values()) or 1
    top4 = sum(v for k, v in result.rank_distribution.items() if k <= 3) / total_ranks

    print("=" * 60)
    print(f"sage_plays_live — {args.game} — {args.machine}")
    print("=" * 60)
    print(f"  Steps (trace / applied / emitted): {len(trace.steps)} / {result.steps_applied} / {result.records_emitted}")
    print(f"  SAGE correct:    {result.correct} / {result.records_emitted} ({acc:.4f})")
    print(f"  Top-4 rank cov:  {top4:.4f}")
    print(f"  Final state:     {result.final_state}  (levels={result.final_levels_completed})")
    print(f"  Rank distribution:")
    for rank, n in result.rank_distribution.items():
        bar = "█" * min(60, n // max(1, result.records_emitted // 60))
        print(f"    rank {rank}: n={n:5d}  {bar}")
    print(f"  Per-action (known_good):")
    for name, stats in result.per_action.items():
        print(f"    {name:6s}: n={stats['n']:5d}  correct={stats['correct']:5d}  recall={stats['recall']:.3f}")
    for err in result.errors[:3]:
        print(f"  ERR: {err}")

    if args.json_out:
        import dataclasses
        payload = dataclasses.asdict(result)
        payload["accuracy"] = acc
        payload["top4_rank_coverage"] = top4
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote: {args.json_out}")

    return 0 if result.records_emitted > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
