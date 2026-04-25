#!/usr/bin/env python3
"""
Lean dispatch — minimal prompt, maximum signal.

The invoke prompt is a QUESTION, not a MANUAL.
~300 tokens of decision-relevant info per invocation.
System prompt cached once in Ollama's context, not repeated.

Design from playing as the model: what do I need to pick the right action?
1. What game, what level, what step
2. What's the goal (one sentence)
3. What each action does RIGHT NOW (lookahead)
4. Any special signals (level advance, stuck)
5. Pick an action
"""

import copy
import os
import sys
import time
import json
import numpy as np
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sage.cognition.thalamic_router.llm_dispatch import (
    OllamaClient, render_frame_pair_png, parse_llm_response,
    load_frame_router, choose_dispatch,
    onehot_frame, build_scalar_vector,
    N_ACTIONS, ACTION_NAMES, FRAME_H, FRAME_W, N_COLORS, RECENT_ACTIONS_K,
)
from sage.cognition.thalamic_router.lookahead import lookahead
from sage.cognition.thalamic_router.frame_router import FrameRouter


# World model summaries — one paragraph per game, loaded once
def _load_wm_oneliner(game_family: str) -> str:
    """Load a one-paragraph world model summary."""
    for root in [
        Path(os.environ.get("SHARED_CONTEXT_DIR", "")),
        Path.home() / "ai-workspace" / "shared-context",
    ]:
        wm_path = root / "arc-agi-3" / "world-models" / f"{game_family}.md"
        if wm_path.exists():
            text = wm_path.read_text()
            # Extract just Objects + Rules + Win Condition, compress
            sections = {}
            current = None
            for line in text.splitlines():
                if line.startswith("## "):
                    current = line
                elif current and current in ("## Rules", "## Win Condition", "## Strategy"):
                    sections.setdefault(current, []).append(line)

            parts = []
            for s in ["## Rules", "## Win Condition", "## Strategy"]:
                body = " ".join(sections.get(s, [])).strip()
                if body:
                    parts.append(body[:200])
            return " ".join(parts)[:500]
    return ""


def build_lean_prompt(
    game: str, level: int, step: int,
    lookahead_text: str,
    click_targets: str,
    wm_summary: str,
    nn_hint: str, nn_confidence: float,
    recent_actions: List[str],
    special_signals: List[str],
) -> str:
    """Build a ~300 token decision-focused prompt."""

    recent_str = " → ".join(recent_actions[-5:]) if recent_actions else "(none)"
    signals = ""
    if special_signals:
        signals = "\n" + "\n".join(f"  ⚠ {s}" for s in special_signals)

    click_block = ""
    if click_targets:
        click_block = f"\n{click_targets}"

    return f"""{game} level {level}, step {step}.
{wm_summary[:300]}

Actions NOW:
{lookahead_text}{click_block}
{signals}
Recent: {recent_str}
NN hint: {nn_hint} ({nn_confidence:.0%})

Which action makes the most PROGRESS toward winning?
If lookahead shows LEVEL ADVANCE for any action, choose that action.
SEL = launch/activate (the action that commits progress).

1=UP 2=DOWN 3=LEFT 4=RIGHT 5=SEL 6=CLICK
ACTION=N X=x Y=y (for CLICK, give coordinates)
One sentence explaining why."""


def play_lean(
    game: str, game_id: str,
    adapter_path: str,
    llm_model: str = "gemma4:26b",
    max_steps: int = 200,
    machine: str = "thor",
    system_prompt: Optional[str] = None,
) -> Dict:
    """Play a game with lean dispatch.

    system_prompt: optional cached system prompt (e.g. from DIR Phase 1 discovery).
    Passed to Ollama as the system message — cached in KV, not repeated per invoke.
    """
    from arc_agi import Arcade
    from arcengine import GameAction

    GA = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
          4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6}

    arcade = Arcade(operation_mode='offline')
    env = arcade.make(game_id)
    if env is None:
        for ei in arcade.get_environments():
            if game in ei.game_id:
                env = arcade.make(ei.game_id)
                game_id = ei.game_id
                break
    if env is None:
        return {"error": f"Game {game_id} not found"}

    fd = env.reset()
    model, cfg = load_frame_router(Path(adapter_path))
    llm = OllamaClient(model=llm_model)

    # Load world model once
    wm = _load_wm_oneliner(game)

    prev_frame = np.array(fd.frame)[-1].copy()
    recent = deque(maxlen=RECENT_ACTIONS_K)
    recent_names = []
    action_counts = Counter()
    invoke_count = 0
    stuck_count = 0
    llm_responses = []

    # Within-game action outcome tracking
    # Track how often each action produces meaningful frame change
    action_outcomes = {name: {"tried": 0, "productive": 0} for name in ACTION_NAMES[1:]}  # skip A0

    print(f"LEAN DISPATCH: {game_id}, max {max_steps} steps")

    for step in range(max_steps):
        if fd.state.name in ("WON", "GAME_OVER", "LOST"):
            break

        curr_frame = np.array(fd.frame)[-1]
        curr_oh = onehot_frame(curr_frame)
        prev_oh = onehot_frame(prev_frame)
        level = fd.levels_completed

        # Adapter decision
        scalar = build_scalar_vector(
            game_idx=0, n_games=cfg.n_games,
            level=level, n_levels=cfg.n_levels,
            step_frac=step / max(max_steps, 1),
            budget_remaining=1.0 - step / max(max_steps, 1),
            last_action=int(recent[-1]) if recent else 0,
            recent_actions=list(recent),
            available_actions=[1] * N_ACTIONS,
            batch_state=[0.0, 0.0, 0.0],
            include_last_action=cfg.include_last_action,
        )
        dispatch = choose_dispatch(
            model, cfg, prev_oh, curr_oh,
            np.array(scalar, dtype=np.float32),
        )

        if dispatch.decision == "invoke":
            invoke_count += 1

            # Lookahead
            la_text, _ = lookahead(env, fd, actions=[1, 2, 3, 4, 5])

            # Click probes if needed
            click_text = ""
            if dispatch.play_action == 6 or la_text.lower().count("no change") >= 3:
                ct, _ = lookahead(env, fd, actions=[],
                    click_coords=[(x, y) for y in range(8, 56, 16) for x in range(8, 56, 16)])
                active = [l.strip() for l in ct.split('\n')
                          if 'CLICK' in l and 'no change' not in l and 'trivial' not in l and 'error' not in l]
                if active:
                    click_text = "Click targets:\n  " + "\n  ".join(active[:3])

            # Special signals
            signals = []
            stuck_action = None

            # Check if any action produces LEVEL ADVANCE
            if "LEVEL ADVANCE" in la_text:
                signals.append("LEVEL ADVANCE detected in lookahead — check which action!")

            # Stuck detection — single-action repetition
            if len(recent_names) >= 3 and len(set(recent_names[-3:])) == 1:
                stuck_action = recent_names[-1]
                stuck_count += 1

            # Oscillation detection — pair repetition (A→B→A→B)
            if not stuck_action and len(recent_names) >= 4:
                last4 = recent_names[-4:]
                if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
                    stuck_action = last4[-1]  # exclude the most recent
                    stuck_count += 1

            if stuck_action:
                # Tell the model what alternatives do, not just "try something different"
                alt_lines = []
                for line in la_text.split('\n'):
                    stripped = line.strip()
                    if stripped and not stripped.startswith(stuck_action + ':'):
                        if any(stripped.startswith(a + ':') for a in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SEL']):
                            if 'no change' not in stripped.lower():
                                alt_lines.append(stripped)
                if alt_lines:
                    signals.append(f"STUCK on {stuck_action}. Alternatives that produce change: {'; '.join(alt_lines[:3])}")
                else:
                    signals.append(f"STUCK on {stuck_action}. Try a different action.")

            # Action outcome signal — what's been working this game?
            # Show contrast: best vs worst, only after enough data
            total_tried = sum(v["tried"] for v in action_outcomes.values())
            if total_tried >= 15:
                rated = [(name, v["productive"] / v["tried"], v["tried"])
                         for name, v in action_outcomes.items() if v["tried"] >= 3]
                if rated:
                    rated.sort(key=lambda x: -x[1])
                    best = rated[0]
                    worst = rated[-1]
                    if best[1] > worst[1] + 0.2:  # meaningful contrast
                        signals.append(
                            f"This game: {best[0]} works {best[1]:.0%} of the time, "
                            f"{worst[0]} works {worst[1]:.0%}. Favor what works."
                        )

            nn_hint = ACTION_NAMES[dispatch.play_action] if dispatch.play_action < len(ACTION_NAMES) else "?"

            # Adaptive prompt — depth shaped by what lookahead reveals
            from sage.cognition.thalamic_router.adaptive_prompt import (
                build_adaptive_prompt, classify_situation,
            )
            situation = classify_situation(la_text, game)

            # For trivial (LEVEL ADVANCE), skip the LLM entirely
            if situation == "trivial":
                # Find the advancing action
                for line in la_text.split('\n'):
                    if "LEVEL ADVANCE" in line:
                        for name_idx, name in enumerate(ACTION_NAMES[1:], 1):
                            if line.strip().startswith(name):
                                action = name_idx
                                break
                        break
                rationale = "LEVEL ADVANCE — auto-selected"
                coords = None
                elapsed = 0.0
                llm_responses.append({
                    "step": step, "action": action, "coords": coords,
                    "rationale": rationale, "latency_s": 0,
                    "prompt_tokens": 0, "situation": "trivial",
                })
                if step < 15 or step % 20 == 0:
                    print(f"  {step:3d}: {ACTION_NAMES[action]} (0.0s) L{level} \"LEVEL ADVANCE!\"")

                # Execute and continue
                if action in GA:
                    fd = env.step(GA[action])
                action_counts[ACTION_NAMES[action]] += 1
                recent.append(action)
                recent_names.append(ACTION_NAMES[action])
                prev_frame = curr_frame
                continue

            prompt = build_adaptive_prompt(
                game=game, level=level, step=step,
                lookahead_text=la_text,
                click_targets=click_text,
                nn_hint=nn_hint, nn_confidence=dispatch.play_confidence,
                recent_actions=recent_names,
                special_signals=signals,
            )

            pair_png = render_frame_pair_png(prev_frame, curr_frame, scale=4)

            t0 = time.time()
            response = llm.chat(prompt, images_png=[pair_png], system_prompt=system_prompt)
            elapsed = time.time() - t0

            # When stuck, diversify the fallback — don't fall back to the stuck action
            fallback = dispatch.play_action
            if stuck_action and fallback < len(ACTION_NAMES) and ACTION_NAMES[fallback] == stuck_action:
                # Pick a different action that the lookahead shows is active
                import random
                active_actions = []
                for line in la_text.split('\n'):
                    stripped = line.strip()
                    for idx, name in enumerate(ACTION_NAMES[1:], 1):
                        if stripped.startswith(name + ':') and 'no change' not in stripped.lower():
                            active_actions.append(idx)
                if active_actions:
                    fallback = random.choice([a for a in active_actions if ACTION_NAMES[a] != stuck_action] or active_actions)

            action, coords, rationale = parse_llm_response(response, fallback_action=fallback)

            llm_responses.append({
                "step": step, "action": action, "coords": coords,
                "action_name": ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else "?",
                "rationale": rationale, "latency_s": elapsed,
                "prompt_tokens": len(prompt) // 4,
                "situation": situation,
                "raw_response": response[:200],
                "stuck_action": stuck_action,
                "level": level,
            })

            if step < 15 or step % 20 == 0:
                aname = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action)
                print(f"  {step:3d}: {aname} ({elapsed:.1f}s) L{level} \"{rationale[:60]}\"")
        else:
            action = dispatch.play_action
            coords = None
            if action == 6:
                coords = {"x": 32, "y": 32}

        # Execute
        if action in GA:
            try:
                if action == 6 and coords:
                    fd = env.step(GA[action], data=coords)
                else:
                    fd = env.step(GA[action])
            except (KeyError, TypeError, Exception) as e:
                # Some games crash on certain actions (e.g. click with wrong coord format)
                # Fall back to stepping without data
                try:
                    fd = env.step(GA[action])
                except Exception:
                    pass  # fd unchanged, continue
            if fd is None:
                break

        aname = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action)
        action_counts[aname] += 1
        recent.append(action)
        recent_names.append(aname)

        # Track action outcomes — did this action produce meaningful change?
        frame_delta_pct = 0.0
        if fd and hasattr(fd, 'frame') and aname in action_outcomes:
            post_frame = np.array(fd.frame)[-1]
            px_diff = int(np.sum(post_frame != curr_frame))
            total_px = curr_frame.size if curr_frame.size > 0 else 1
            frame_delta_pct = 100.0 * px_diff / total_px
            action_outcomes[aname]["tried"] += 1
            if px_diff > 10:  # meaningful change threshold
                action_outcomes[aname]["productive"] += 1

        # Backfill frame_delta_pct into the last llm_response (if this step had one)
        if llm_responses and llm_responses[-1].get("step") == step:
            llm_responses[-1]["frame_delta_pct"] = frame_delta_pct
            new_level = fd.levels_completed if fd and hasattr(fd, 'levels_completed') else level
            llm_responses[-1]["level_after"] = new_level

        prev_frame = curr_frame

    final_state = fd.state.name if fd and hasattr(fd, 'state') and fd.state else "CRASHED"
    final_levels = fd.levels_completed if fd and hasattr(fd, 'levels_completed') else 0
    result = {
        "game": game, "game_id": game_id,
        "n_steps": step + 1 if final_state in ("WON", "GAME_OVER", "LOST") else step,
        "final_state": final_state,
        "final_levels": final_levels,
        "action_counts": dict(action_counts),
        "invoke_count": invoke_count,
        "stuck_count": stuck_count,
        "llm_responses": llm_responses,
        "avg_prompt_tokens": np.mean([r["prompt_tokens"] for r in llm_responses]) if llm_responses else 0,
        "action_outcomes": {k: v for k, v in action_outcomes.items() if v["tried"] > 0},
    }

    print(f"\nResult: L{final_levels}, {result['n_steps']} steps, {final_state}")
    print(f"Actions: {dict(action_counts)}")
    print(f"Invokes: {invoke_count}, Stuck: {stuck_count}")
    print(f"Avg prompt: ~{result['avg_prompt_tokens']:.0f} tokens")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--game-id", default=None)
    p.add_argument("--adapter", required=True)
    p.add_argument("--llm-model", default="gemma4:26b")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--machine", default="thor")
    p.add_argument("--json-out", default=None)
    p.add_argument("--system-prompt-file", default=None,
                   help="File containing system prompt (e.g. from DIR discovery)")
    args = p.parse_args()

    os.environ.setdefault("ARC_SAGE_DIR", os.path.expanduser("~/ai-workspace/arc-sage"))
    os.environ.setdefault("SHARED_CONTEXT_DIR", os.path.expanduser("~/ai-workspace/shared-context"))

    game_id = args.game_id
    if not game_id:
        coord_path = Path(os.environ["ARC_SAGE_DIR"]) / "knowledge" / "game_coordination.json"
        if coord_path.exists():
            for g in json.loads(coord_path.read_text()).get("games", []):
                if g.get("family") == args.game:
                    game_id = g.get("id"); break

    sys_prompt = None
    if args.system_prompt_file:
        sys_prompt = Path(args.system_prompt_file).read_text().strip()
        print(f"System prompt from {args.system_prompt_file}: {sys_prompt[:100]}...")

    result = play_lean(
        game=args.game, game_id=game_id or args.game,
        adapter_path=args.adapter,
        llm_model=args.llm_model,
        max_steps=args.max_steps,
        machine=args.machine,
        system_prompt=sys_prompt,
    )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Wrote: {args.json_out}")
