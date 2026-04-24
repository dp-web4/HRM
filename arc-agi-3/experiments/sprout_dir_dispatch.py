#!/usr/bin/env python3
"""Sprout DIR dispatch: discover-inhabit-rediscover with text perception.

Uses arc_perception as eyes, Qwen 0.8B as brain.
Discovery phase lets the model describe before acting.
Rediscovery triggers on stuck detection.

Usage:
    sudo systemctl stop sage-daemon-sprout.service
    python3 sprout_dir_dispatch.py --game cd82 --steps 50
    sudo systemctl start sage-daemon-sprout.service
"""
import sys, os, json, time, re, argparse, requests
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import full_perception, get_frame, grid_diff

OLLAMA = "http://localhost:11434/api/generate"
GA_MAP = {a.value: a for a in GameAction}
ACTION_MAP = {
    'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2,
    'LEFT': GameAction.ACTION3, 'RIGHT': GameAction.ACTION4,
    'SELECT': GameAction.ACTION5, 'CLICK': GameAction.ACTION6,
}
ACTION_NAMES = {v: k for k, v in ACTION_MAP.items()}


def ask(prompt, max_tokens=60, temperature=0.4):
    r = requests.post(OLLAMA, json={
        "model": "qwen3.5:0.8b", "prompt": prompt,
        "stream": False, "think": False,
        "options": {"num_predict": max_tokens, "temperature": temperature}
    }, timeout=30)
    return r.json().get("response", "").strip()


def parse_action(response, available):
    text = response.upper().strip()
    click = re.search(r'CLICK\s*\(?(\d+)\s*[,\s]\s*(\d+)', text)
    if click and GameAction.ACTION6 in available:
        return GameAction.ACTION6, {"x": int(click.group(1)), "y": int(click.group(2))}
    for name, action in ACTION_MAP.items():
        if name in text and action in available:
            return action, None
    # Fallback
    for a in available:
        if a != GameAction.ACTION7:
            return a, None
    return available[0], None


def discover(perception, available_names):
    """Phase 1: let the model describe what it sees and form a plan."""
    prompt = (
        f"You are looking at a puzzle game screen.\n\n"
        f"{perception}\n\n"
        f"Available actions: {', '.join(available_names)}\n\n"
        f"Describe what you see in 2 sentences. "
        f"Then say which object looks most important to interact with."
    )
    return ask(prompt, max_tokens=80, temperature=0.5)


def inhabit(perception_short, discovery_context, step, prev_result, available_names):
    """Phase 2: act from the discovered context."""
    prompt = f"Game context: {discovery_context[:200]}\n\n"
    prompt += f"Step {step}. {perception_short}\n"
    if prev_result:
        prompt += f"Last action result: {prev_result}\n"
    prompt += f"\nActions: {', '.join(available_names)}\n"
    prompt += "What action? Reply with just the action."
    return ask(prompt, max_tokens=20, temperature=0.3)


def run_game(game_id, max_steps=50):
    arc = Arcade()
    env = arc.make(game_id)
    obs = env.reset()

    available = [GA_MAP[a] for a in obs.available_actions if a in GA_MAP]
    avail_names = [ACTION_NAMES.get(a, f'A{a.value}') for a in available
                   if a != GameAction.ACTION7]

    prev_grid = None
    discovery_context = None
    stuck_count = 0
    results = []
    level = getattr(env._game, 'level_index', 0)

    print(f"\n{'='*60}")
    print(f"Sprout DIR Dispatch: {game_id}")
    print(f"Actions: {avail_names}")
    print(f"{'='*60}")

    for step in range(1, max_steps + 1):
        grid = get_frame(obs)
        perception = full_perception(grid)
        new_level = getattr(env._game, 'level_index', 0)

        # Trigger (re)discovery on: first step, level change, stuck
        need_discover = (
            discovery_context is None or
            new_level != level or
            stuck_count >= 3
        )

        if need_discover:
            reason = "start" if discovery_context is None else (
                "level_up" if new_level != level else "stuck"
            )
            print(f"\n  [{step}] DISCOVER ({reason}):")
            discovery_context = discover(perception, avail_names)
            print(f"    {discovery_context[:150]}")
            stuck_count = 0
            level = new_level

        # Compact perception for inhabit (just key info)
        perception_short = perception.split('\n')[0]  # first line = grid summary
        regions = [l for l in perception.split('\n') if 'region' in l.lower() or '@' in l]
        if regions:
            perception_short += '\n' + regions[0][:100]

        # Previous action result
        prev_result = None
        if prev_grid is not None:
            diff = grid_diff(prev_grid, grid)
            if diff:
                prev_result = diff[:80]

        # Inhabit: get action
        response = inhabit(perception_short, discovery_context, step,
                           prev_result, avail_names)
        action, data = parse_action(response, available)
        action_name = ACTION_NAMES.get(action, f'A{action.value}')
        if data:
            action_name += f"({data['x']},{data['y']})"

        # Execute
        obs = env.step(action, data=data)
        new_grid = get_frame(obs)
        px_changed = int(np.sum(grid != new_grid))

        # Stuck detection
        if px_changed <= 2:
            stuck_count += 1
        else:
            stuck_count = 0

        new_level = getattr(env._game, 'level_index', 0)
        lvl_up = " *** LEVEL UP! ***" if new_level > level else ""
        level = new_level

        print(f"  [{step:3d}] {action_name:15s} Δ={px_changed:4d}px "
              f"stuck={stuck_count} '{response[:25]}'{lvl_up}")

        results.append({
            'step': step, 'action': action_name, 'response': response[:50],
            'px_changed': px_changed, 'stuck': stuck_count,
            'level': level, 'level_up': bool(lvl_up),
        })

        prev_grid = grid

        state = obs.state
        if str(state) in ('WIN', 'WON', 'GAME_OVER', 'LOST'):
            print(f"\n  Game ended: {state} at step {step}, level {level}")
            break

    # Summary
    levels_won = sum(1 for r in results if r['level_up'])
    action_dist = {}
    for r in results:
        a = r['action'].split('(')[0]
        action_dist[a] = action_dist.get(a, 0) + 1
    discovers = sum(1 for r in results if r.get('level_up')) + 1  # at least initial

    print(f"\n{'='*60}")
    print(f"Steps: {len(results)} | Levels: {levels_won} | Discovers: {discovers}")
    print(f"Actions: {action_dist}")
    print(f"Stuck episodes: {sum(1 for r in results if r['stuck'] >= 3)}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='cd82')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--json-out', default=None)
    args = parser.parse_args()

    results = run_game(args.game, max_steps=args.steps)

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {args.json_out}")
