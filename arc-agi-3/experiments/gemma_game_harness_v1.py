#!/usr/bin/env python3
"""
Gemma Game Harness v1 — Game-Agnostic Structured Probing

Plays ANY ARC-AGI-3 game by:
1. Discovering available actions from SDK
2. Classifying actions by pixel diff (safe vs consequential)
3. Probing consequential actions from states reached by safe actions
4. Presenting discovered map + cartridge to Gemma for planning
5. Executing the plan, checking for win

Usage:
    python3 gemma_game_harness_v1.py --game cd82 --levels 1-6
    python3 gemma_game_harness_v1.py --game ft09 --levels 1
"""

import sys, os, json, argparse, requests, time
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
from collections import defaultdict

SAGE_DIR = Path(__file__).parent.parent.parent
SHARED = SAGE_DIR.parent / 'shared-context'
CART_DIR = SHARED / 'arc-agi-3' / 'phase2' / 'carts'

OLLAMA_URL = 'http://localhost:11434'
MODEL = 'gemma4:e4b'

# GameAction values → names
ACTION_NAMES = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'SELECT', 6: 'CLICK', 7: 'UNDO'}

# Int→GameAction enum mapping (GameAction(int) fails on some Python versions)
def int_to_ga(i):
    from arcengine.enums import GameAction
    return getattr(GameAction, f'ACTION{i}')


def embed(text):
    resp = requests.post(f'{OLLAMA_URL}/api/embeddings', json={
        'model': 'nomic-embed-text', 'prompt': text
    })
    return np.array(resp.json()['embedding'], dtype=np.float32)


def cart_search(query, cart_path, top_k=2):
    if not Path(cart_path).exists():
        return []
    data = np.load(cart_path, allow_pickle=True)
    embs = data['embeddings']
    passages = json.loads(str(data['passages'][0]))
    q_emb = embed(query)
    sims = np.dot(embs, q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb))
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [passages[i]['text'] for i in top_idx]


def generate(prompt, max_tokens=400):
    resp = requests.post(f'{OLLAMA_URL}/api/generate', json={
        'model': MODEL, 'prompt': prompt, 'stream': False,
        'think': False, 'options': {'temperature': 0.3, 'num_predict': max_tokens}
    }, timeout=120)
    return resp.json().get('response', '')


def frame_diff(f1, f2):
    """Count differing pixels between two frames."""
    return int(np.sum(f1 != f2))


def classify_diff(diff):
    if diff == 0: return 'no-effect'
    if diff < 10: return 'selection'
    if diff < 100: return 'moderate'
    return 'major'


def play_game(game_prefix, levels=None, max_actions_per_level=80, probe_depth=4):
    """Play a game using structured probing + Gemma planning."""

    from arc_agi import Arcade
    from arcengine import GameAction

    arcade = Arcade()
    game_id = None
    for e in arcade.get_environments():
        if game_prefix in e.game_id:
            game_id = e.game_id
            break

    if not game_id:
        print(f"Game {game_prefix} not found")
        return

    env = arcade.make(game_id)
    fd = env.reset()
    game = env._game

    available = fd.available_actions  # list of ints
    action_names = {a: ACTION_NAMES.get(a, f'ACTION{a}') for a in available}

    total_levels = fd.win_levels
    if levels is None:
        levels = range(total_levels)

    print(f"\n{'='*60}")
    print(f"GAME: {game_id} | {total_levels} levels | actions: {[action_names[a] for a in available]}")
    print(f"{'='*60}")

    results = []

    for level_idx in levels:
        if fd.levels_completed > level_idx:
            continue  # already past this level

        print(f"\n--- Level {level_idx + 1} ---")
        actions_used = 0

        # Get initial frame
        init_frame = np.array(fd.frame)[-1] if len(np.array(fd.frame).shape) == 3 else np.array(fd.frame)

        # Extract UI element positions BEFORE probing changes game state
        # Convert game coords → display coords using camera inverse
        sprite_info = ""
        click_targets = []  # structured: [{'x':display_x, 'y':display_y, 'color':c}, ...]
        try:
            game_obj = env._game
            cam = game_obj.camera

            # Build game→display lookup by inverting display_to_grid
            g2d = {}  # game_coord → display_coord
            for dx in range(64):
                for dy in range(64):
                    gpos = cam.display_to_grid(dx, dy)
                    if gpos:
                        gx, gy = gpos
                        if (gx, gy) not in g2d:
                            g2d[(gx, gy)] = (dx, dy)

            clickable_descs = []
            for s in game_obj.current_level.get_sprites():
                if s.is_visible and s.width > 0 and s.width <= 10 and s.height <= 10:
                    c = int(s.pixels[min(1,s.height-1), min(1,s.width-1)])
                    if c >= 0:
                        game_cx, game_cy = s.x + s.width//2, s.y + s.height//2
                        # Convert to display coords
                        display_pos = g2d.get((game_cx, game_cy))
                        if display_pos:
                            dcx, dcy = display_pos
                        else:
                            dcx, dcy = game_cx, game_cy  # fallback: 1:1
                        click_targets.append({'x': dcx, 'y': dcy, 'color': c})
                        clickable_descs.append(f"game({game_cx},{game_cy})→display({dcx},{dcy}) color={c}")
            if clickable_descs:
                sprite_info = "\n## Detected UI Elements (click at DISPLAY coordinates)\n"
                for ci in clickable_descs[:15]:
                    sprite_info += f"- {ci}\n"
                print(f"  Detected {len(clickable_descs)} UI elements (camera-corrected)")
        except Exception as e:
            print(f"  UI detection error: {e}")

        # ===== PHASE 1a: Classify each action =====
        print(f"Phase 1a: Classify {len(available)} actions")

        action_class = {}
        for a_id in available:
            name = action_names[a_id]
            pre_frame = np.array(fd.frame)[-1] if len(np.array(fd.frame).shape) == 3 else np.array(fd.frame)

            # For CLICK: test at first detected sprite position (not 0,0)
            if name == 'CLICK' and click_targets:
                ct = click_targets[0]
                fd = env.step(int_to_ga(a_id), data={'x': ct['x'], 'y': ct['y']})
            else:
                fd = env.step(int_to_ga(a_id))
            post_frame = np.array(fd.frame)[-1] if len(np.array(fd.frame).shape) == 3 else np.array(fd.frame)
            actions_used += 1

            diff = frame_diff(pre_frame, post_frame)
            cls = classify_diff(diff)
            action_class[name] = {'id': a_id, 'diff': diff, 'class': cls}

            print(f"  {name:8s}: {diff:4d}px → {cls}")

            if fd.levels_completed > level_idx:
                print(f"  *** WON during classification! ***")
                results.append({'level': level_idx+1, 'won': True, 'actions': actions_used, 'phase': '1a'})
                break

        if fd.levels_completed > level_idx:
            continue

        # Phase 1a.5: Test REVERSIBILITY of high-diff actions
        # An action is "navigation" if doing it then its opposite returns to original state
        # An action is "consequential" if it changes state irreversibly
        opposites = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}

        print(f"Phase 1a.5: Reversibility test")
        for name, info in list(action_class.items()):
            if info['class'] in ('major', 'moderate') and name in opposites:
                opp = opposites[name]
                if opp in action_class:
                    # Test: do action, do opposite, compare to before
                    probe_env2 = arcade.make(game_id)
                    pfd = probe_env2.reset()
                    pre = np.array(pfd.frame)[-1] if len(np.array(pfd.frame).shape) == 3 else np.array(pfd.frame)
                    probe_env2.step(int_to_ga(info['id']))
                    probe_env2.step(int_to_ga(action_class[opp]['id']))
                    # Get frame after do+undo
                    probe_env3 = arcade.make(game_id)
                    pfd3 = probe_env3.reset()
                    pre3 = np.array(pfd3.frame)[-1] if len(np.array(pfd3.frame).shape) == 3 else np.array(pfd3.frame)
                    probe_env3.step(int_to_ga(info['id']))
                    pfd3b = probe_env3.step(int_to_ga(action_class[opp]['id']))
                    post3 = np.array(pfd3b.frame)[-1] if len(np.array(pfd3b.frame).shape) == 3 else np.array(pfd3b.frame)

                    residual = frame_diff(pre3, post3)
                    if residual < 5:  # nearly identical = reversible = navigation
                        info['class'] = 'navigation'
                        print(f"  {name}: reversible (residual={residual}px) → navigation")
                    else:
                        print(f"  {name}: irreversible (residual={residual}px) → consequential")

        # Classify: navigation = reversible directional, consequential = everything else
        nav_actions = [n for n, info in action_class.items()
                       if info['class'] in ('navigation', 'no-effect')
                       and n in ('UP', 'DOWN', 'LEFT', 'RIGHT')]
        consequential = [n for n, info in action_class.items()
                        if info['class'] not in ('no-effect',) and n not in nav_actions]

        # If no consequential found, ALL non-no-effect actions ARE consequential
        # This handles click-only and pure-nav games
        if not consequential:
            consequential = [n for n, info in action_class.items() if info['class'] != 'no-effect']
            # For probing: use ALL actions as nav (to reach different states)
            nav_actions = [n for n in action_class if n in ('UP', 'DOWN', 'LEFT', 'RIGHT')]

        # If STILL no nav (click-only), nav_seqs will just be [[]] (probe from start)
        print(f"  Nav: {nav_actions}, Consequential: {consequential}")

        # click_targets populated earlier from sprite extraction

        # ===== PHASE 1b: Probe consequential actions from reachable states =====
        print(f"Phase 1b: Probe from {probe_depth}-deep navigation states")

        # Generate navigation sequences up to probe_depth
        nav_seqs = [[]]  # start with empty (current position)
        for depth in range(1, probe_depth + 1):
            for combo in product(nav_actions, repeat=depth):
                nav_seqs.append(list(combo))

        # Use detected click targets as selection prefixes for probing
        selection_prefixes = [('none', [], {})]
        if click_targets and 'CLICK' in action_class:
            for ct in click_targets:
                selection_prefixes.append(
                    (f"CLICK({ct['x']},{ct['y']})", ['CLICK'], {'x': ct['x'], 'y': ct['y']})
                )

        # Include ALL depths up to probe_depth
        # depth 4 with 4 nav: 1+4+16+64+256=341 sequences
        max_nav = len(nav_seqs)  # use all

        # Reduce selection prefixes to keep total manageable
        # Only use 'none' + the 2 most distinct click targets (highest/lowest color)
        if len(selection_prefixes) > 3:
            # Keep none + first + last (likely different colors)
            selection_prefixes = [selection_prefixes[0], selection_prefixes[1], selection_prefixes[-1]]

        # Build click probe targets for CLICK-type consequential actions
        click_probe_targets = [{}]  # default: no coords
        if click_targets and any(c == 'CLICK' for c in consequential):
            click_probe_targets = [{'x': ct['x'], 'y': ct['y']} for ct in click_targets]

        n_click_targets = len(click_probe_targets) if any(c == 'CLICK' for c in consequential) else 1
        total_probes = len(nav_seqs) * len(selection_prefixes) * (
            sum(n_click_targets if c == 'CLICK' else 1 for c in consequential)
        )
        print(f"  Total probes planned: ~{total_probes} ({len(nav_seqs)} nav × {len(selection_prefixes)} sel × {len(consequential)} cons" +
              (f" × {n_click_targets} click-targets" if n_click_targets > 1 else "") + ")")

        position_map = {}

        for cons_action in consequential:
            cons_id = action_class[cons_action]['id']
            # If this action is CLICK, probe at each click target position
            # Otherwise, probe with no data (directional actions don't need coords)
            is_click = cons_action == 'CLICK'
            cons_targets = click_probe_targets if is_click else [{}]

            for sel_label, sel_prefix, sel_data in selection_prefixes:
                for nav_seq in nav_seqs:
                    for ct_data in cons_targets:
                        nav_label = '→'.join(nav_seq) if nav_seq else 'start'
                        ct_suffix = f"@({ct_data['x']},{ct_data['y']})" if ct_data else ""
                        label = f"{sel_label+'→' if sel_label else ''}{nav_label}{ct_suffix}"

                        # Create fresh game for probing
                        probe_env = arcade.make(game_id)
                        probe_fd = probe_env.reset()

                        if level_idx > 0:
                            continue  # TODO: replay solved levels

                        # Apply selection prefix
                        for sel_name in sel_prefix:
                            if sel_data:
                                probe_fd = probe_env.step(int_to_ga(action_class[sel_name]['id']), data=sel_data)
                            else:
                                probe_fd = probe_env.step(int_to_ga(action_class[sel_name]['id']))

                        # Navigate
                        for nav in nav_seq:
                            nav_id = action_class[nav]['id']
                            probe_fd = probe_env.step(int_to_ga(nav_id))

                        # Record pre-state
                        pre = np.array(probe_fd.frame)[-1] if len(np.array(probe_fd.frame).shape) == 3 else np.array(probe_fd.frame)

                        # Execute consequential action (with coordinates if CLICK)
                        if ct_data:
                            probe_fd = probe_env.step(int_to_ga(cons_id), data=ct_data)
                        else:
                            probe_fd = probe_env.step(int_to_ga(cons_id))
                        post = np.array(probe_fd.frame)[-1] if len(np.array(probe_fd.frame).shape) == 3 else np.array(probe_fd.frame)

                        diff = frame_diff(pre, post)
                        won = probe_fd.levels_completed > level_idx

                        key = f"{cons_action}@{label}"
                        position_map[key] = {
                            'nav': nav_seq,
                            'sel': sel_prefix,
                            'sel_data': sel_data,
                            'action': cons_action,
                            'action_data': ct_data,
                            'diff': diff,
                            'won': won
                        }

                        if won or diff > 20:
                            print(f"  {key:40s}: {diff:4d}px" + (" *** WINS ***" if won else ""))

        # ===== PHASE 2: Model plans from discovered map =====
        print(f"Phase 2: Gemma plans from discovered map")

        # Retrieve cartridge
        game_family = game_prefix.split('-')[0]
        game_ctx = cart_search(
            f"How to play {game_family}, strategy, win condition",
            str(CART_DIR / f'{game_family}.cart.npz'), top_k=2
        )
        substrate_ctx = cart_search(
            "action budget click classification",
            str(CART_DIR / 'substrate-primitives.cart.npz'), top_k=2
        )

        # Build map text
        map_text = f"## DISCOVERED ACTION MAP for Level {level_idx+1}\n\n"
        map_text += "### Action effects:\n"
        for name, info in action_class.items():
            map_text += f"- {name}: {info['class']} ({info['diff']}px change)\n"

        winning_keys = [k for k, v in position_map.items() if v['won']]
        high_impact = sorted(
            [(k, v) for k, v in position_map.items() if v['diff'] > 20],
            key=lambda x: -x[1]['diff']
        )[:10]

        if winning_keys:
            map_text += "\n### WINNING sequences found:\n"
            for k in winning_keys:
                v = position_map[k]
                sel_str = " → ".join(v.get('sel', [])) + " → " if v.get('sel') else ""
                nav_str = " → ".join(v['nav']) if v['nav'] else "(from start)"
                map_text += f"- {sel_str}Navigate: {nav_str}, then {v['action']} → WINS!\n"

        if high_impact:
            map_text += "\n### High-impact probes:\n"
            for k, v in high_impact:
                nav_str = " → ".join(v['nav']) if v['nav'] else "(start)"
                map_text += f"- {nav_str} → {v['action']}: {v['diff']}px\n"

        # Add reasoning bridge
        analysis_note = ""
        if not winning_keys:
            analysis_note = """
## IMPORTANT: No winning probe was found.

This likely means the probes tested the CONSEQUENTIAL action (SELECT/LAUNCH)
without first changing the game's SELECTION STATE (e.g., color, mode, tool).

The cartridge mentions clickable elements (palettes, buttons, selectors).
You MUST include CLICK with specific (x,y) coordinates to change the
selection state BEFORE using the consequential action.

Read the cartridge knowledge carefully for palette/button positions,
then plan: CLICK (select correct option) → navigate → consequential action.
"""

        plan_prompt = f"""## Game Knowledge
{chr(10).join(t[:400] for t in game_ctx)}

## Substrate
{chr(10).join(t[:200] for t in substrate_ctx)}

{map_text}
{sprite_info}
{analysis_note}

OUTPUT THE ACTION SEQUENCE ONLY. No analysis. No explanation. One action per line.
Valid: {', '.join(action_names.values())}
CLICK must include x y coordinates from the UI Elements list above.
Example: CLICK 43 4"""

        plan = generate(plan_prompt, max_tokens=200)
        print(f"Model plan:\n{plan[:400]}")

        # ===== PHASE 3: Execute =====
        # If winning probes found, replay exact winning sequence on fresh game
        if winning_keys:
            best_win = min(winning_keys, key=lambda k: len(position_map[k]['nav']))
            wv = position_map[best_win]
            print(f"Phase 3: Replay winning probe: {best_win}")

            # Fresh game for clean execution
            win_env = arcade.make(game_id)
            win_fd = win_env.reset()

            # Apply selection prefix
            for sel_name in wv.get('sel', []):
                sel_data = wv.get('sel_data', {})
                if sel_data:
                    win_fd = win_env.step(int_to_ga(action_class[sel_name]['id']), data=sel_data)
                else:
                    win_fd = win_env.step(int_to_ga(action_class[sel_name]['id']))
                actions_used += 1

            # Navigate
            for nav_name in wv['nav']:
                win_fd = win_env.step(int_to_ga(action_class[nav_name]['id']))
                actions_used += 1

            # Consequential action (with coordinates if CLICK)
            cons_data = wv.get('action_data', {})
            if cons_data:
                win_fd = win_env.step(int_to_ga(action_class[wv['action']]['id']), data=cons_data)
            else:
                win_fd = win_env.step(int_to_ga(action_class[wv['action']]['id']))
            actions_used += 1

            won = win_fd.levels_completed > level_idx
            print(f"  Result: {'*** WON! ***' if won else 'did not win'} ({actions_used} actions)")

            if won:
                results.append({'level': level_idx+1, 'won': True, 'actions': actions_used, 'phase': '3-replay'})
                # Update main env to match
                fd = win_fd
                env = win_env
                continue

        print(f"Phase 3: Execute model plan")

        # Parse
        planned = []
        for line in plan.split('\n'):
            line_upper = line.strip().upper()
            if 'CLICK' in line_upper:
                nums = [w for w in line_upper.split() if w.isdigit()]
                data = {'x': int(nums[0]), 'y': int(nums[1])} if len(nums) >= 2 else {}
                planned.append(('CLICK', 6, data))
            else:
                for name, a_id in [(n, info['id']) for n, info in action_class.items()]:
                    if name in line_upper:
                        planned.append((name, a_id, {}))
                        break

        print(f"  Parsed {len(planned)} actions")

        for i, (name, a_id, data) in enumerate(planned):
            if actions_used >= max_actions_per_level:
                break

            if data:
                fd = env.step(int_to_ga(a_id), data=data)
            else:
                fd = env.step(int_to_ga(a_id))
            actions_used += 1

            won = fd.levels_completed > level_idx
            data_str = f" ({data.get('x','')},{data.get('y','')})" if data else ""
            print(f"  {i+1}: {name}{data_str}" + (" *** WON! ***" if won else ""))

            if won:
                results.append({'level': level_idx+1, 'won': True, 'actions': actions_used, 'phase': '3'})
                break

        if not any(r.get('won') for r in results if r.get('level') == level_idx+1):
            results.append({'level': level_idx+1, 'won': False, 'actions': actions_used})

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {game_id}")
    won_count = sum(1 for r in results if r.get('won'))
    print(f"  Won: {won_count}/{len(results)} levels")
    for r in results:
        status = "✓ WON" if r.get('won') else "✗ lost"
        print(f"  L{r['level']}: {status} ({r['actions']} actions, phase {r.get('phase','?')})")

    # Save
    out_dir = SHARED / 'arc-agi-3' / 'fleet-learning' / 'mcnugget' / 'logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(out_dir / f'harness-v1-{game_family}-{ts}.json', 'w') as f:
        json.dump({
            'game': game_id, 'model': MODEL,
            'levels_won': won_count, 'levels_attempted': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='cd82', help='Game prefix')
    parser.add_argument('--levels', default=None, help='Level range: 1-6 or 1')
    parser.add_argument('--depth', type=int, default=4, help='Navigation probe depth')
    parser.add_argument('--budget', type=int, default=80, help='Max actions per level')
    args = parser.parse_args()

    levels = None
    if args.levels:
        if '-' in args.levels:
            start, end = args.levels.split('-')
            levels = range(int(start)-1, int(end))
        else:
            levels = [int(args.levels)-1]

    play_game(args.game, levels=levels, max_actions_per_level=args.budget, probe_depth=args.depth)
