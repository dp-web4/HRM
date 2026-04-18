#!/usr/bin/env python3
"""wm_play — self-advance play driven by world-model forward simulation.

At each step:
  1. Encode current state
  2. For each candidate action a in 0..6:
     - simulate next embedding via dynamics_head
     - score via outcome_head
  3. Blend with supervised action prior: score[a] *= softmax(action_head)[a]
  4. Argmax (or temperature-sample) → action
  5. Coord fallback: CLICK → center-of-frame (same as sage_plays_self)
  6. env.step(action)
  7. Terminal check; loop

This is SAGE choosing actions by imagining consequences. The consciousness
loop in miniature.

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-2-world-model-embedding-sprint.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from sage.cognition.router.data import RouterDatasetWriter
from sage.cognition.router.record import RouterRecord
from sage.cognition.router.outputs import RouterOutput
from sage.cognition.thalamic_router.phase1_training import _feature_vec
from sage.cognition.thalamic_router.world_model import (
    WorldModel, WorldModelConfig, load_world_model,
    build_input_vector, N_ACTIONS, ACTION_NAMES, N_BASE_FEATURES,
)
from sage.cognition.thalamic_router.gameplay_capture import (
    load_trace, synth_router_input, _discover_trace, _frame_delta,
    ARC_AGI_EXPERIMENTS, Trace, TraceStep,
)


CLICK_FALLBACK_X = 32
CLICK_FALLBACK_Y = 32
HARD_MAX_STEPS = 1000


@dataclass
class WmPlayResult:
    game: str
    game_id: str
    n_steps: int
    max_steps: int
    final_state: Optional[str]
    final_levels: Optional[int]
    outcome: str
    solver_steps: Optional[int] = None
    action_counts: Dict[str, int] = field(default_factory=dict)
    unique_embeddings: int = 0
    errors: List[str] = field(default_factory=list)


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


def choose_action(
    model: WorldModel, cfg: WorldModelConfig, x: torch.Tensor,
    plan_weight: float = 1.0, prior_weight: float = 0.5,
    temperature: float = 0.0,
) -> tuple[int, List[float]]:
    """Score actions by combining forward-simulated outcome with supervised prior.

    plan_weight=1, prior_weight=0.5:
        score[a] = σ(outcome_head(dynamics(emb, a_oh)))^plan_weight
                 × softmax(action_head(emb))[a]^prior_weight

    temperature > 0 enables stochastic sampling instead of argmax.
    """
    model.eval()
    with torch.no_grad():
        emb = model.encode(x.unsqueeze(0))               # (1, emb_dim)
        action_probs = F.softmax(model.forward_action(emb), dim=-1)[0]  # (7,)

        # Forward-simulate each action, score predicted next-state's outcome
        # Batch the 7 rollouts for speed
        emb_rep = emb.expand(N_ACTIONS, -1)              # (7, emb_dim)
        a_oh = torch.eye(N_ACTIONS, device=x.device)     # (7, 7)
        next_emb = model.forward_dynamics(emb_rep, a_oh)  # (7, emb_dim)
        next_outcome = torch.sigmoid(model.forward_outcome(next_emb))  # (7,)

    plan_scores = next_outcome.pow(plan_weight)
    prior_scores = action_probs.pow(prior_weight)
    combined = plan_scores * prior_scores                # (7,)

    if temperature > 0:
        logits = (combined + 1e-9).log() / temperature
        probs = F.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
    else:
        action = int(combined.argmax().item())

    # Return the combined scores as debugging signal
    return action, combined.cpu().tolist()


def run_wm_play(
    model: WorldModel, cfg: WorldModelConfig,
    game_family: str, game_id: str,
    writer: RouterDatasetWriter, machine: str,
    max_steps: int, trace_steps: Optional[int] = None,
    plan_weight: float = 1.0, prior_weight: float = 0.5,
    temperature: float = 0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> WmPlayResult:
    errors: List[str] = []
    try:
        env, fd, fb = _make_env_short(game_family, game_id)
        if fb:
            errors.append(fb)
    except Exception as e:
        return WmPlayResult(
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
    unique_embs: set = set()
    step_idx = 0
    atp = 100.0
    prev_frame = getattr(fd, "frame", None)
    last_action = 0
    outcome_terminal = {"WIN", "GAME_OVER"}

    mean = np.array(cfg.feature_mean)
    std = np.array(cfg.feature_std)

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

        action, combined_scores = choose_action(
            model, cfg, x,
            plan_weight=plan_weight, prior_weight=prior_weight,
            temperature=temperature,
        )
        aname = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
        action_counts[aname] = action_counts.get(aname, 0) + 1

        # Rough embedding-uniqueness check (round to 1 decimal place to avoid float noise)
        with torch.no_grad():
            emb = model.encode(x.unsqueeze(0))[0].cpu().numpy()
        unique_embs.add(tuple(np.round(emb, 1).tolist()))

        coords = None
        if aname == "CLICK":
            coords = {"x": CLICK_FALLBACK_X, "y": CLICK_FALLBACK_Y}

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
            "source": "sage_plays_self",   # schema compat with existing valid source
            "game": game_family,
            "game_id": game_id,
            "step_index": step_idx + 1,
            "synthetic_kernel_state": True,
            "sage_plays_self": {
                "proposed_action": action,
                "proposed_name": aname,
                "coords_used": coords,
                "state_before": curr_state,
                "state_after": new_state,
                "levels_before": curr_levels,
                "levels_after": new_levels,
                "atp": atp,
                "combined_scores": combined_scores,
                "planner": "wm_play",
            },
        }
        rec = RouterRecord(
            router_input=router_input,
            router_output=RouterOutput.noop(rationale_code="wm_play"),
            machine=machine, timestamp=time.time(), metadata=metadata,
        )
        try:
            writer.append(rec.to_dict())
        except Exception as e:
            errors.append(f"step {step_idx}: write failed: {e!r}")

        fd = new_fd
        prev_frame = curr_frame
        last_action = action
        step_idx += 1

        if new_state in outcome_terminal:
            return WmPlayResult(
                game=game_family, game_id=game_id, n_steps=step_idx,
                max_steps=max_steps, final_state=new_state,
                final_levels=new_levels, outcome=new_state,
                solver_steps=trace_steps, action_counts=action_counts,
                unique_embeddings=len(unique_embs), errors=errors,
            )

    final_state = getattr(getattr(fd, "state", None), "name", None) or "RUNNING"
    final_levels = getattr(fd, "levels_completed", 0) or 0
    return WmPlayResult(
        game=game_family, game_id=game_id, n_steps=step_idx,
        max_steps=max_steps, final_state=final_state,
        final_levels=final_levels, outcome="MAX_STEPS",
        solver_steps=trace_steps, action_counts=action_counts,
        unique_embeddings=len(unique_embs), errors=errors,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="World-model adapter base path (no ext).")
    p.add_argument("--game", required=True)
    p.add_argument("--game-id", default=None)
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"))
    p.add_argument("--data-dir", default=os.environ.get(
        "SAGE_ROUTER_DATA_DIR",
        "/mnt/c/exe/projects/ai-agents/private-context/training-data/router"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--plan-weight", type=float, default=1.0)
    p.add_argument("--prior-weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = argmax; >0 = stochastic sampling (breaks lock-ins).")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    # Resolve game_id (same pattern as sage_plays_self)
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
    writer = RouterDatasetWriter(
        base_dir=Path(args.data_dir), machine=args.machine,
        compress=True, subdir="wm_play",
    )
    try:
        result = run_wm_play(
            model, cfg, args.game, game_id, writer, args.machine,
            max_steps=max_steps, trace_steps=trace_steps,
            plan_weight=args.plan_weight, prior_weight=args.prior_weight,
            temperature=args.temperature,
        )
    finally:
        writer.close()

    print("=" * 60)
    print(f"wm_play — {args.game} — {args.machine}")
    print("=" * 60)
    print(f"  Adapter          : {Path(args.adapter).name}")
    print(f"  Solver reference : {trace_steps} steps")
    print(f"  SAGE took        : {result.n_steps} / {max_steps} steps")
    print(f"  Final state      : {result.final_state}")
    print(f"  Levels reached   : {result.final_levels}")
    print(f"  Outcome          : {result.outcome}")
    print(f"  Action counts    : {result.action_counts}")
    print(f"  Unique embs seen : {result.unique_embeddings}")
    for err in result.errors[:3]:
        print(f"  ERR: {err}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nWrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
