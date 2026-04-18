#!/usr/bin/env python3
"""sage_plays_self — self-advance play. No teacher. Errors compound.

SAGE proposes; SAGE's proposal drives env.step(); SAGE sees whatever
state its choice produced; SAGE chooses again. Game ends when the env
ends it (WIN / GAME_OVER / NOT_FINISHED / max-steps).

This is the first time SAGE's choices actually determine outcomes. No
teacher assistance. Every wrong direction step compounds. Every CLICK
goes wherever our coord-fallback policy places it — and we have no
coord head, so CLICKs are center-of-frame and almost always wrong.

That's the point. "Live the consequences" means let the gaps show.

Coord fallback for CLICK: center-of-frame (32, 32 on a 64×64 canvas).
Games whose click targets align with center will occasionally work;
most won't. This localizes "what does the adapter-plus-no-coord-head
combo actually achieve?"

Output: {machine}/sage_plays_self/{date}.jsonl.gz — one record per step
with metadata.source="sage_plays_self" and metadata.sage_plays_self =
  { proposed_action, coords_used, state_before, state_after,
    levels_before, levels_after, confidence, entropy, softmax_probs }

No `known_good_*` fields — SAGE's trajectory diverges from any trace
after step 0, so no ground truth to attach.

Terminal condition: env reports WIN | GAME_OVER | NOT_FINISHED, OR we
hit max_steps (default: 2 × trace length or 1000, whichever is lower).

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-1.5-sage-plays-plan.md
      §"Not in scope" — what we're promoting now.
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
from sage.cognition.router.outputs import RouterOutput
from sage.cognition.thalamic_router.phase1_training import (
    HeadBAdapter, _feature_vec, HEAD_B_ACTION_NAMES,
)
from sage.cognition.thalamic_router.gameplay_capture import (
    load_trace, synth_router_input, _discover_trace, _frame_delta,
    ARC_AGI_EXPERIMENTS, Trace, TraceStep,
)


# Coord fallback when adapter proposes CLICK. No coord regression head
# exists yet, so every CLICK goes here. Games whose click targets
# cluster near center will occasionally succeed; most won't. This is
# the honest gap.
CLICK_FALLBACK_X = 32
CLICK_FALLBACK_Y = 32

# Default absolute cap; per-run cap = min(CAP, 2 * trace_length) if trace available.
HARD_MAX_STEPS = 1000


@dataclass
class SelfPlayResult:
    game: str
    game_id: str
    n_steps: int
    max_steps: int
    final_state: Optional[str]
    final_levels: Optional[int]
    outcome: str                       # 'WIN' | 'GAME_OVER' | 'NOT_FINISHED' | 'MAX_STEPS'
    solver_steps: Optional[int] = None  # trace length, if available (reference)
    solver_levels: Optional[int] = None
    action_counts: Dict[str, int] = field(default_factory=dict)
    click_count: int = 0
    errors: List[str] = field(default_factory=list)


def _make_env_short(game_family: str, game_id: str):
    """Same version-fallback pattern as gameplay_capture."""
    if str(ARC_AGI_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(ARC_AGI_EXPERIMENTS))
    from arc_agi import Arcade
    arc = Arcade(operation_mode="offline")
    env = arc.make(game_id)
    fallback = None
    if env is None:
        env = arc.make(game_family)
        if env is not None:
            fallback = f"version_fallback: {game_id} → latest {game_family}"
    if env is None:
        raise RuntimeError(f"arc.make returned None for {game_id} and {game_family}")
    fd = env.reset()
    return env, fd, fallback


def run_self_play(
    game_family: str, game_id: str, adapter: HeadBAdapter,
    writer: RouterDatasetWriter, machine: str,
    max_steps: int, trace_steps: Optional[int] = None,
) -> SelfPlayResult:
    errors: List[str] = []
    try:
        env, fd, fb = _make_env_short(game_family, game_id)
        if fb:
            errors.append(fb)
    except Exception as e:
        return SelfPlayResult(
            game=game_family, game_id=game_id, n_steps=0,
            max_steps=max_steps, final_state=None, final_levels=None,
            outcome="ENV_INIT_FAILED", errors=[f"env init: {e!r}"],
        )

    try:
        from arcengine import GameAction
        int_to_action = {ga.value: ga for ga in GameAction}
    except Exception:
        int_to_action = {}
        errors.append("arcengine import failed; action ints may not map")

    action_counts: Dict[str, int] = {}
    click_count = 0
    step_idx = 0
    atp = 100.0
    prev_frame = getattr(fd, "frame", None)

    # NOT_FINISHED is the "keep playing" signal (set by env.reset and
    # returned after every non-terminal step). Terminal states are
    # WIN (cleared all levels) and GAME_OVER (lost). MAX_STEPS is our
    # fallback if SAGE just keeps looping without either.
    outcome_terminal = {"WIN", "GAME_OVER"}

    while step_idx < max_steps:
        curr_frame = getattr(fd, "frame", None)
        curr_state = getattr(getattr(fd, "state", None), "name", None) or "RUNNING"
        curr_levels = getattr(fd, "levels_completed", 0) or 0

        # Build a TraceStep-shaped object for synth_router_input compat.
        # SAGE doesn't know the "level" — use current levels_completed as
        # a proxy (what the env says we're on).
        pseudo_step = TraceStep(index=step_idx + 1, level=curr_levels, action=0)
        router_input = synth_router_input(
            tick=step_idx + 1, prev_frame=prev_frame, curr_frame=curr_frame,
            step=pseudo_step, game=game_family, level=curr_levels, atp=atp,
        )

        # Feature vector → adapter → proposal
        ri_dict = {}
        for attr in ("snarc_surprise", "snarc_novelty", "snarc_arousal",
                     "snarc_reward", "snarc_conflict", "sensory_novelty",
                     "sensory_urgency", "atp_level", "wm_goal_active",
                     "wm_pressure", "habit_available", "habit_confidence",
                     "sensory_modalities", "metabolic_state"):
            ri_dict[attr] = getattr(router_input, attr, None)
        features = _feature_vec({"router_input": ri_dict})

        proposed, probs = adapter.predict(features)
        pname = HEAD_B_ACTION_NAMES[proposed] if proposed < len(HEAD_B_ACTION_NAMES) else str(proposed)
        action_counts[pname] = action_counts.get(pname, 0) + 1

        # Coord policy: CLICK → center-of-frame fallback; everything else → None
        coords = None
        if pname == "CLICK":
            coords = {"x": CLICK_FALLBACK_X, "y": CLICK_FALLBACK_Y}
            click_count += 1

        # Advance env — SAGE's proposal drives it
        new_fd = fd
        try:
            ga = int_to_action.get(proposed)
            if ga is None:
                errors.append(f"step {step_idx}: no enum for action {proposed}")
                break
            new_fd = env.step(ga, data=coords) if coords else env.step(ga)
            atp = max(10.0, atp - 0.1)
        except Exception as e:
            errors.append(f"step {step_idx}: env.step({pname}) failed: {e!r}")
            break

        new_state = getattr(getattr(new_fd, "state", None), "name", None) or "RUNNING"
        new_levels = getattr(new_fd, "levels_completed", 0) or 0

        # Softmax entropy for logging
        import math
        entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)

        # Emit a record
        metadata = {
            "source": "sage_plays_self",
            "game": game_family,
            "game_id": game_id,
            "step_index": step_idx + 1,
            "synthetic_kernel_state": True,
            "sage_plays_self": {
                "proposed_action": proposed,
                "proposed_name": pname,
                "coords_used": coords,
                "confidence": float(probs[proposed]),
                "entropy": float(entropy),
                "state_before": curr_state,
                "state_after": new_state,
                "levels_before": curr_levels,
                "levels_after": new_levels,
                "atp": atp,
            },
            "sage_plays_probs": [float(p) for p in probs],
        }
        router_output = RouterOutput.noop(rationale_code="sage_plays_self")
        rec = RouterRecord(
            router_input=router_input, router_output=router_output,
            machine=machine, timestamp=time.time(), metadata=metadata,
        )
        try:
            writer.append(rec.to_dict())
        except Exception as e:
            errors.append(f"step {step_idx}: write failed: {e!r}")

        fd = new_fd
        prev_frame = curr_frame
        step_idx += 1

        if new_state in outcome_terminal:
            return SelfPlayResult(
                game=game_family, game_id=game_id, n_steps=step_idx,
                max_steps=max_steps, final_state=new_state,
                final_levels=new_levels, outcome=new_state,
                solver_steps=trace_steps, action_counts=action_counts,
                click_count=click_count, errors=errors,
            )

    final_state = getattr(getattr(fd, "state", None), "name", None) or "RUNNING"
    final_levels = getattr(fd, "levels_completed", 0) or 0
    return SelfPlayResult(
        game=game_family, game_id=game_id, n_steps=step_idx,
        max_steps=max_steps, final_state=final_state,
        final_levels=final_levels, outcome="MAX_STEPS",
        solver_steps=trace_steps, action_counts=action_counts,
        click_count=click_count, errors=errors,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True,
                   help="Path to Head B adapter JSON.")
    p.add_argument("--game", required=True, help="Game family (e.g. cd82).")
    p.add_argument("--game-id", default=None,
                   help="Full versioned id. Looked up from game_coordination.json if omitted.")
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"))
    p.add_argument("--data-dir", default=os.environ.get(
        "SAGE_ROUTER_DATA_DIR",
        "/mnt/c/exe/projects/ai-agents/private-context/training-data/router"))
    p.add_argument("--max-steps", type=int, default=None,
                   help=f"Hard cap. Default: min(2 × trace_length, {HARD_MAX_STEPS}).")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    # Resolve game_id
    game_id = args.game_id
    if not game_id:
        for root in [Path(os.environ.get("ARC_SAGE_DIR", "")),
                     Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE"),
                     Path.home() / "ai-workspace" / "ARC-SAGE",
                     Path.home() / "repos" / "ARC-SAGE"]:
            coord = root / "knowledge" / "game_coordination.json"
            if coord.exists():
                try:
                    for g in json.loads(coord.read_text()).get("games", []):
                        if g.get("family") == args.game:
                            game_id = g.get("id"); break
                except Exception:
                    pass
                if game_id:
                    break
    if not game_id:
        print(f"ERROR: could not resolve game_id for {args.game}")
        return 1

    # Trace is optional — we just want its length for max_steps reference
    trace_steps = None
    solver_levels = None
    try:
        tp = _discover_trace(args.game, game_id)
        if tp and tp.exists():
            tr = load_trace(tp, args.game, game_id)
            trace_steps = len(tr.steps)
            solver_levels = tr.outcome.get("levels_completed") or tr.outcome.get("win_levels")
    except Exception:
        pass

    if args.max_steps is None:
        max_steps = min(2 * trace_steps, HARD_MAX_STEPS) if trace_steps else HARD_MAX_STEPS
    else:
        max_steps = args.max_steps

    adapter = HeadBAdapter.load(Path(args.adapter))
    writer = RouterDatasetWriter(
        base_dir=Path(args.data_dir), machine=args.machine,
        compress=True, subdir="sage_plays_self",
    )
    try:
        result = run_self_play(
            game_family=args.game, game_id=game_id, adapter=adapter,
            writer=writer, machine=args.machine,
            max_steps=max_steps, trace_steps=trace_steps,
        )
        result.solver_levels = solver_levels
    finally:
        writer.close()

    print("=" * 60)
    print(f"sage_plays_self — {args.game} — {args.machine}")
    print("=" * 60)
    print(f"  Solver reference : {trace_steps} steps → {solver_levels} levels ({result.solver_levels and 'WIN' or 'unknown'})")
    print(f"  SAGE took        : {result.n_steps} / {max_steps} steps")
    print(f"  Final state      : {result.final_state}")
    print(f"  Levels reached   : {result.final_levels}")
    print(f"  Outcome          : {result.outcome}")
    print(f"  Action counts    : {result.action_counts}")
    print(f"  CLICK count      : {result.click_count} (all at {CLICK_FALLBACK_X},{CLICK_FALLBACK_Y})")
    for err in result.errors[:3]:
        print(f"  ERR: {err}")

    if args.json_out:
        import dataclasses
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(result), f, indent=2)
        print(f"\nWrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
