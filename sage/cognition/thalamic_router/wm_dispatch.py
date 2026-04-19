#!/usr/bin/env python3
"""wm_dispatch — the router in its intended role.

At each step:
  1. Compute embedding from live env state
  2. Ask choose_dispatch → {decision: "invoke" | "play", context, hint}
  3. Also check stuck-detection (stateful): if SAGE has been cycling
     without progress, force invoke with reason="stuck"
  4. If invoke → log context+hint package (LLM not called yet, this is
     the training-the-router phase). Fall through to NN's play_action
     so the env still advances.
  5. If play → NN commits its top action
  6. env.step; update stuck-detection state

Each step emits a record with the dispatch decision, the hint package,
and the action actually applied. That's the training substrate for
(a) validating the invoke-head against structural signals and
(b) measuring how well the play head performs on its high-confidence
subset.

Stuck-detection signals (harness-level, stateful):
  - Last K actions identical AND last K states visually identical
    (frame delta ≈ 0 for K steps in a row)
  - levels_completed hasn't advanced in K_progress steps

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-2-world-model-embedding-sprint.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sage.cognition.router.data import RouterDatasetWriter
from sage.cognition.router.record import RouterRecord
from sage.cognition.router.outputs import RouterOutput
from sage.cognition.thalamic_router.phase1_training import _feature_vec
from sage.cognition.thalamic_router.world_model import (
    WorldModel, WorldModelConfig, load_world_model,
    build_input_vector, N_ACTIONS, ACTION_NAMES, choose_dispatch,
)
from sage.cognition.thalamic_router.gameplay_capture import (
    load_trace, synth_router_input, _discover_trace,
    ARC_AGI_EXPERIMENTS, TraceStep,
)


CLICK_FALLBACK_X = 32
CLICK_FALLBACK_Y = 32
HARD_MAX_STEPS = 1000

# Stuck detection: if the last K actions are identical AND the frame
# hasn't changed in K steps, we're stuck.
STUCK_WINDOW = 5
STUCK_FRAME_EPS = 1e-3
# Levels-stalled: no level progress for K_progress steps → stuck
PROGRESS_STALL_WINDOW = 40


@dataclass
class DispatchResult:
    game: str
    game_id: str
    n_steps: int
    max_steps: int
    final_state: Optional[str]
    final_levels: Optional[int]
    outcome: str
    solver_steps: Optional[int] = None
    action_counts: Dict[str, int] = field(default_factory=dict)
    invoke_count: int = 0
    invoke_reasons: Dict[str, int] = field(default_factory=dict)
    stuck_count: int = 0
    unique_embeddings: int = 0
    errors: List[str] = field(default_factory=list)


def _frame_norm(frame: Any) -> Optional[np.ndarray]:
    try:
        arr = np.array(frame)
        return arr.flatten().astype(np.float32)
    except Exception:
        return None


def _frames_equal(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return float(np.abs(a - b).mean()) < STUCK_FRAME_EPS


def _make_env_short(game_family: str, game_id: str):
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
        raise RuntimeError(f"arc.make None for {game_id} and {game_family}")
    fd = env.reset()
    return env, fd, fallback


def run_dispatch(
    model: WorldModel, cfg: WorldModelConfig,
    game_family: str, game_id: str,
    writer: RouterDatasetWriter, machine: str,
    max_steps: int, trace_steps: Optional[int] = None,
    device: str = "cpu",
) -> DispatchResult:
    errors: List[str] = []
    try:
        env, fd, fb = _make_env_short(game_family, game_id)
        if fb:
            errors.append(fb)
    except Exception as e:
        return DispatchResult(
            game=game_family, game_id=game_id, n_steps=0,
            max_steps=max_steps, final_state=None, final_levels=None,
            outcome="ENV_INIT_FAILED", errors=[f"env init: {e!r}"],
        )

    try:
        from arcengine import GameAction
        int_to_action = {ga.value: ga for ga in GameAction}
    except Exception:
        int_to_action = {}

    model = model.to(device)
    game_slugs = cfg.game_slugs
    game_idx = game_slugs.index(game_family) if game_family in game_slugs else 0

    action_counts: Dict[str, int] = {}
    invoke_reasons: Dict[str, int] = {}
    invoke_count = 0
    stuck_count = 0
    unique_embs: set = set()
    step_idx = 0
    atp = 100.0
    prev_frame = getattr(fd, "frame", None)
    last_action = 0
    outcome_terminal = {"WIN", "GAME_OVER"}

    mean = np.array(cfg.feature_mean)
    std = np.array(cfg.feature_std)

    # Stuck-detection sliding windows
    recent_actions: deque = deque(maxlen=STUCK_WINDOW)
    recent_frames: deque = deque(maxlen=STUCK_WINDOW)
    steps_since_progress = 0
    last_levels = 0

    while step_idx < max_steps:
        curr_frame = getattr(fd, "frame", None)
        curr_state = getattr(getattr(fd, "state", None), "name", None) or "RUNNING"
        curr_levels = getattr(fd, "levels_completed", 0) or 0

        pseudo_step = TraceStep(index=step_idx + 1, level=curr_levels, action=0)
        router_input = synth_router_input(
            tick=step_idx + 1, prev_frame=prev_frame, curr_frame=curr_frame,
            step=pseudo_step, game=game_family, level=curr_levels, atp=atp,
        )

        ri_dict = {}
        for attr in ("snarc_surprise", "snarc_novelty", "snarc_arousal",
                     "snarc_reward", "snarc_conflict", "sensory_novelty",
                     "sensory_urgency", "atp_level", "wm_goal_active",
                     "wm_pressure", "habit_available", "habit_confidence",
                     "sensory_modalities", "metabolic_state"):
            ri_dict[attr] = getattr(router_input, attr, None)
        base_feats = _feature_vec({"router_input": ri_dict})
        norm_feats = ((np.array(base_feats) - mean) / std).tolist()

        step_frac = min(1.0, (step_idx + 1) / max(1, trace_steps or 100))
        x_vec = build_input_vector(
            norm_feats, game_idx, cfg.n_games, curr_levels, cfg.n_levels,
            step_frac, last_action,
        )
        x = torch.tensor(x_vec, dtype=torch.float32, device=device)

        dispatch = choose_dispatch(
            model, x, game=game_family, level=curr_levels,
            step_index=step_idx + 1,
        )

        # Harness-level stuck detection overlay
        stuck_triggered = False
        if len(recent_actions) == STUCK_WINDOW:
            same_actions = len(set(recent_actions)) == 1
            frames_same = all(
                _frames_equal(recent_frames[i], recent_frames[0])
                for i in range(1, len(recent_frames))
            )
            if same_actions and frames_same:
                stuck_triggered = True
                stuck_count += 1
        if steps_since_progress >= PROGRESS_STALL_WINDOW:
            stuck_triggered = True
            stuck_count += 1

        if stuck_triggered:
            # Force invoke and add reason
            dispatch["decision"] = "invoke"
            if "stuck" not in dispatch["invoke_reasons"]:
                dispatch["invoke_reasons"].append("stuck")

        # Tally invoke reasons for the summary
        if dispatch["decision"] == "invoke":
            invoke_count += 1
            for r in dispatch["invoke_reasons"]:
                invoke_reasons[r] = invoke_reasons.get(r, 0) + 1

        # Action to apply: NN's play_action (even on invoke — we're not
        # calling the LLM yet, so we fall through to the NN's best guess)
        action = dispatch["play_action"]
        aname = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
        action_counts[aname] = action_counts.get(aname, 0) + 1

        coords = None
        if aname == "CLICK":
            coords = {"x": CLICK_FALLBACK_X, "y": CLICK_FALLBACK_Y}

        # Rough embedding-uniqueness
        unique_embs.add(tuple(
            round(v, 1) for v in dispatch["context"]["embedding"]
        ))

        new_fd = fd
        try:
            ga = int_to_action.get(action)
            if ga is None:
                errors.append(f"step {step_idx}: no enum for {action}")
                break
            new_fd = env.step(ga, data=coords) if coords else env.step(ga)
            atp = max(10.0, atp - 0.1)
        except Exception as e:
            errors.append(f"step {step_idx}: env.step({aname}) failed: {e!r}")
            break

        new_state = getattr(getattr(new_fd, "state", None), "name", None) or "RUNNING"
        new_levels = getattr(new_fd, "levels_completed", 0) or 0

        metadata = {
            "source": "sage_plays_self",       # existing valid source
            "game": game_family,
            "game_id": game_id,
            "step_index": step_idx + 1,
            "synthetic_kernel_state": True,
            "sage_plays_self": {
                "decision": dispatch["decision"],
                "invoke_reasons": dispatch["invoke_reasons"],
                "invoke_prob": dispatch["invoke_prob"],
                "play_action": dispatch["play_action"],
                "play_confidence": dispatch["play_confidence"],
                "play_margin": dispatch["play_margin"],
                "action_ranking": dispatch["action_ranking"],
                "coords_used": coords,
                "state_before": curr_state,
                "state_after": new_state,
                "levels_before": curr_levels,
                "levels_after": new_levels,
                "atp": atp,
                "planner": "wm_dispatch_v2",
                "stuck_triggered": stuck_triggered,
            },
        }
        rec = RouterRecord(
            router_input=router_input,
            router_output=RouterOutput.noop(rationale_code="wm_dispatch"),
            machine=machine, timestamp=time.time(), metadata=metadata,
        )
        try:
            writer.append(rec.to_dict())
        except Exception as e:
            errors.append(f"step {step_idx}: write failed: {e!r}")

        # Update stuck-tracking state
        recent_actions.append(action)
        recent_frames.append(_frame_norm(curr_frame))
        if new_levels > last_levels:
            steps_since_progress = 0
            last_levels = new_levels
        else:
            steps_since_progress += 1

        fd = new_fd
        prev_frame = curr_frame
        last_action = action
        step_idx += 1

        if new_state in outcome_terminal:
            return DispatchResult(
                game=game_family, game_id=game_id, n_steps=step_idx,
                max_steps=max_steps, final_state=new_state,
                final_levels=new_levels, outcome=new_state,
                solver_steps=trace_steps, action_counts=action_counts,
                invoke_count=invoke_count, invoke_reasons=invoke_reasons,
                stuck_count=stuck_count,
                unique_embeddings=len(unique_embs), errors=errors,
            )

    final_state = getattr(getattr(fd, "state", None), "name", None) or "RUNNING"
    final_levels = getattr(fd, "levels_completed", 0) or 0
    return DispatchResult(
        game=game_family, game_id=game_id, n_steps=step_idx,
        max_steps=max_steps, final_state=final_state,
        final_levels=final_levels, outcome="MAX_STEPS",
        solver_steps=trace_steps, action_counts=action_counts,
        invoke_count=invoke_count, invoke_reasons=invoke_reasons,
        stuck_count=stuck_count,
        unique_embeddings=len(unique_embs), errors=errors,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--game", required=True)
    p.add_argument("--game-id", default=None)
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"))
    p.add_argument("--data-dir", default=os.environ.get(
        "SAGE_ROUTER_DATA_DIR",
        "/mnt/c/exe/projects/ai-agents/private-context/training-data/router"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

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

    trace_steps = None
    try:
        tp = _discover_trace(args.game, game_id)
        if tp and tp.exists():
            tr = load_trace(tp, args.game, game_id)
            trace_steps = len(tr.steps)
    except Exception:
        pass

    if args.max_steps is None:
        max_steps = min(2 * trace_steps, HARD_MAX_STEPS) if trace_steps else HARD_MAX_STEPS
    else:
        max_steps = args.max_steps

    model, cfg = load_world_model(Path(args.adapter))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = RouterDatasetWriter(
        base_dir=Path(args.data_dir), machine=args.machine,
        compress=True, subdir="sage_plays_self",
    )
    try:
        result = run_dispatch(
            model, cfg, args.game, game_id, writer, args.machine,
            max_steps=max_steps, trace_steps=trace_steps, device=device,
        )
    finally:
        writer.close()

    print("=" * 60)
    print(f"wm_dispatch — {args.game} — {args.machine}")
    print("=" * 60)
    print(f"  Solver reference : {trace_steps} steps")
    print(f"  SAGE took        : {result.n_steps} / {max_steps} steps")
    print(f"  Final state      : {result.final_state}  levels={result.final_levels}")
    print(f"  Outcome          : {result.outcome}")
    print(f"  Action counts    : {result.action_counts}")
    print(f"  Invokes          : {result.invoke_count}/{result.n_steps} "
          f"({100*result.invoke_count/max(result.n_steps,1):.1f}%)")
    print(f"  Invoke reasons   : {result.invoke_reasons}")
    print(f"  Stuck triggers   : {result.stuck_count}")
    print(f"  Unique embs      : {result.unique_embeddings}")
    for err in result.errors[:3]:
        print(f"  ERR: {err}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nWrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
