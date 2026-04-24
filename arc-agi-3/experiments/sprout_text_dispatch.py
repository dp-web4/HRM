#!/usr/bin/env python3
"""Sprout text-vision dispatch: arc_perception + Qwen 0.8B plays games.

No vision LLM needed — arc_perception converts frames to structured
text, Qwen 0.8B reads text and proposes actions.

Usage:
    # Stop daemon first (shares Ollama)
    sudo systemctl stop sage-daemon-sprout.service
    python3 sprout_text_dispatch.py --game cd82 --steps 30
    sudo systemctl start sage-daemon-sprout.service
"""
import sys, os, json, time, re, argparse, requests
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import full_perception, get_frame, grid_diff

OLLAMA = "http://localhost:11434/api/generate"
ACTION_MAP = {
    'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2,
    'LEFT': GameAction.ACTION3, 'RIGHT': GameAction.ACTION4,
    'SELECT': GameAction.ACTION5, 'CLICK': GameAction.ACTION6,
    'UNDO': GameAction.ACTION7,
}
ACTION_NAMES = {v: k for k, v in ACTION_MAP.items()}


def ask_model(prompt, max_tokens=30, temperature=0.3):
    """Query Qwen 0.8B via Ollama."""
    r = requests.post(OLLAMA, json={
        "model": "qwen3.5:0.8b",
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"num_predict": max_tokens, "temperature": temperature}
    }, timeout=30)
    return r.json().get("response", "").strip()


def parse_action(response, available):
    """Parse model response into a GameAction + optional click coords."""
    text = response.upper().strip()

    # Try CLICK(x,y) pattern
    click_match = re.search(r'CLICK\s*\(?(\d+)\s*[,\s]\s*(\d+)', text)
    if click_match and GameAction.ACTION6 in available:
        x, y = int(click_match.group(1)), int(click_match.group(2))
        return GameAction.ACTION6, {"x": x, "y": y}

    # Try named actions
    for name, action in ACTION_MAP.items():
        if name in text and action in available:
            return action, None

    # Fallback: first available non-undo action
    for a in available:
        if a != GameAction.ACTION7:
            return a, None
    return available[0], None


def build_prompt(perception, step, level, available_actions, prev_action=None, prev_diff=None):
    """Build the prompt for the model. Lean — data not strategy."""
    action_names = [ACTION_NAMES.get(a, f'A{a.value}') for a in available_actions
                    if a != GameAction.ACTION7]  # hide UNDO

    prompt = f"You are playing a puzzle game. Step {step}, level {level}.\n\n"
    prompt += f"What you see:\n{perception}\n\n"

    if prev_action and prev_diff:
        prompt += f"Last action: {prev_action}. Result: {prev_diff}\n\n"

    prompt += f"Available: {', '.join(action_names)}\n"
    prompt += "What action? Reply with just the action name"
    if GameAction.ACTION6 in available_actions:
        prompt += " (for CLICK include coordinates: CLICK(x,y))"
    prompt += "."

    return prompt


def run_game(game_id, max_steps=50, verbose=True):
    """Run a game with text-vision dispatch."""
    arc = Arcade()
    env = arc.make(game_id)
    obs = env.reset()

    _ga_map = {a.value: a for a in GameAction}
    available = [_ga_map[a] for a in obs.available_actions if a in _ga_map]
    prev_grid = None
    prev_action_name = None
    results = []

    print(f"\n{'='*60}")
    print(f"Sprout Text-Vision Dispatch: {game_id}")
    print(f"Available actions: {[ACTION_NAMES.get(a, f'A{a.value}') for a in available]}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}\n")

    for step in range(1, max_steps + 1):
        grid = get_frame(obs)
        perception = full_perception(grid)

        # Diff from previous frame
        diff_desc = None
        if prev_grid is not None:
            diff = grid_diff(prev_grid, grid)
            if diff:
                diff_desc = diff[:100]

        # Get current level
        level = getattr(env._game, 'level_index', 0)

        # Build prompt and ask model
        prompt = build_prompt(perception, step, level, available,
                              prev_action_name, diff_desc)
        t0 = time.time()
        response = ask_model(prompt)
        latency = time.time() - t0

        # Parse action
        action, data = parse_action(response, available)
        action_name = ACTION_NAMES.get(action, f'A{action.value}')
        if data:
            action_name += f"({data['x']},{data['y']})"

        # Execute
        obs = env.step(action, data=data)
        new_grid = get_frame(obs)

        # Track
        pixels_changed = int(np.sum(grid != new_grid)) if prev_grid is not None else 0
        state = obs.state
        new_level = getattr(env._game, 'level_index', 0)

        result = {
            'step': step, 'level': level, 'action': action_name,
            'response': response[:50], 'latency': round(latency, 1),
            'pixels_changed': pixels_changed, 'state': str(state),
            'level_advanced': new_level > level,
        }
        results.append(result)

        if verbose:
            lvl_mark = " *** LEVEL UP! ***" if new_level > level else ""
            print(f"  [{step:3d}] {action_name:15s} ({latency:.1f}s) "
                  f"Δ={pixels_changed:4d}px model='{response[:30]}'{lvl_mark}")

        prev_grid = grid
        prev_action_name = action_name

        # Check game state
        if str(state) in ('WIN', 'WON', 'GAME_OVER', 'LOST'):
            print(f"\n  Game ended: {state} at step {step}, level {new_level}")
            break

    # Summary
    levels_won = sum(1 for r in results if r['level_advanced'])
    actions_used = {a: 0 for a in ACTION_NAMES.values()}
    for r in results:
        base_action = r['action'].split('(')[0]
        actions_used[base_action] = actions_used.get(base_action, 0) + 1

    print(f"\n{'='*60}")
    print(f"Summary: {len(results)} steps, {levels_won} levels advanced")
    print(f"Actions: {', '.join(f'{k}={v}' for k, v in actions_used.items() if v > 0)}")
    print(f"Final state: {results[-1]['state'] if results else '?'}")
    print(f"Avg latency: {sum(r['latency'] for r in results)/len(results):.1f}s")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='cd82', help='Game family or full ID')
    parser.add_argument('--steps', type=int, default=30, help='Max steps')
    parser.add_argument('--json-out', default=None, help='Save results JSON')
    args = parser.parse_args()

    results = run_game(args.game, max_steps=args.steps)

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {args.json_out}")
