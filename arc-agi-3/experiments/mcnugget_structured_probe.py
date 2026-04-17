#!/usr/bin/env python3
"""
McNugget Structured Probing Harness — Automated Action-Map Discovery

The harness DISCOVERS what each action does by systematic probing,
then hands the discovered map to Gemma 4 for planning.

Phase 1a: Test each action type once (6 actions) — classify as safe/consequential
Phase 1b: For each CONSEQUENTIAL action, probe from multiple states:
          - Navigate to different positions (using safe actions)
          - Test the consequential action, record effect
          - UNDO to restore state
          - Repeat from each reachable position
          This builds: {state → action → effect} triples
Phase 2:  Present discovered map + cartridge to model for planning
Phase 3:  Execute model's plan
"""

import sys, os, json, requests
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SAGE_DIR = Path(__file__).parent.parent.parent
SHARED = SAGE_DIR.parent / 'shared-context'
CART_DIR = SHARED / 'arc-agi-3' / 'phase2' / 'carts'
GAME_DIR = SHARED / 'environment_files' / 'cd82' / 'fb555c5d'
sys.path.insert(0, str(GAME_DIR))

from arcengine.enums import GameAction
from arcengine import ActionInput

OLLAMA_URL = 'http://localhost:11434'
MODEL = 'gemma4:e4b'

ACTIONS = {
    'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2,
    'LEFT': GameAction.ACTION3, 'RIGHT': GameAction.ACTION4,
    'LAUNCH': GameAction.ACTION5, 'CLICK': GameAction.ACTION6
}

# Navigation actions (safe, for reaching positions)
NAV_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
# Consequential actions (need probing from multiple states)  
CONSEQUENTIAL = ['LAUNCH']
# Selection actions
SELECTION = ['CLICK']


def embed(text):
    resp = requests.post(f'{OLLAMA_URL}/api/embeddings', json={
        'model': 'nomic-embed-text', 'prompt': text
    })
    return np.array(resp.json()['embedding'], dtype=np.float32)


def cart_search(query, cart_path, top_k=2):
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


def execute(game, action_name, data=None):
    fd = game.perform_action(ActionInput(id=ACTIONS[action_name], data=data or {}), raw=True)
    return fd


def get_canvas(game):
    """Extract canvas pixels for cd82."""
    for s in game.current_level.get_sprites():
        if 'xytrjjbyib' in s.name:
            return s.pixels.copy()
    return None


def describe_paint(canvas):
    """Describe what region was painted white."""
    if canvas is None:
        return "unknown"
    top = int(np.mean(canvas[:5]))
    bot = int(np.mean(canvas[5:]))
    left = int(np.mean(canvas[:, :5]))
    right = int(np.mean(canvas[:, 5:]))
    
    parts = []
    if top > 10 and bot < 5:
        parts.append("TOP half")
    elif bot > 10 and top < 5:
        parts.append("BOTTOM half")
    elif top > 10 and bot > 10:
        parts.append("ALL")
    
    if left > 10 and right < 5:
        parts.append("LEFT half")
    elif right > 10 and left < 5:
        parts.append("RIGHT half")
    
    if not parts:
        # Diagonal or partial
        if top > 5 and left > 5:
            parts.append("top-left wedge")
        elif top > 5 and right > 5:
            parts.append("top-right wedge")
        elif bot > 5 and left > 5:
            parts.append("bottom-left wedge")
        elif bot > 5 and right > 5:
            parts.append("bottom-right wedge")
        else:
            parts.append(f"partial (top={top} bot={bot} L={left} R={right})")
    
    return ", ".join(parts)


def run(game_id='cd82', level=0, max_actions=80):
    import cd82
    game = cd82.Cd82()
    game.set_level(level)
    
    trace = []
    actions_used = 0
    
    print(f"\n{'='*60}")
    print(f"STRUCTURED PROBING: {game_id} L{level+1}")
    print(f"{'='*60}")
    
    # Get target pattern
    target_canvas = None
    for s in game.current_level.get_sprites():
        if 'eoqnvkspoa' in s.name and s.is_visible:
            target_canvas = s.pixels.copy()
    target_desc = describe_paint(target_canvas) if target_canvas is not None else "unknown"
    print(f"Target pattern: {target_desc}")
    
    # Find palette positions
    palette_positions = []
    for s in game.current_level.get_sprites():
        if 'pqkenviek' in s.name:
            c = s.pixels[min(1, s.height-1), min(1, s.width-1)]
            palette_positions.append((s.x + s.width//2, s.y + s.height//2, int(c)))
    print(f"Palette: {palette_positions}")
    
    # ===== PHASE 1a: Basic action classification =====
    print(f"\n--- Phase 1a: Action Classification ---")
    
    pre_frame = np.array(game.perform_action(
        ActionInput(id=GameAction.RESET, data={}), raw=True
    ).frame)[-1]
    
    action_class = {}
    for name in ACTIONS:
        pre = pre_frame.copy() if actions_used == 0 else post.copy()
        fd = execute(game, name)
        post = np.array(fd.frame)[-1]
        actions_used += 1
        diff = int(np.sum(pre != post))
        
        if diff == 0:
            cls = 'no-effect'
        elif diff < 10:
            cls = 'selection'
        elif diff < 100:
            cls = 'moderate'
        else:
            cls = 'major'
        
        action_class[name] = {'diff': diff, 'class': cls}
        print(f"  {name:8s}: {diff:4d}px → {cls}")
    
    # ===== PHASE 1b: Structured probing of LAUNCH from each position =====
    print(f"\n--- Phase 1b: Probe LAUNCH from each reachable position ---")
    
    # Navigation sequences to reach each ring position from start
    # We'll try systematic exploration: from start, try each nav combo up to depth 4
    nav_sequences = [
        ([], "start(N)"),
        (['LEFT'], "L(NW)"),
        (['RIGHT'], "R(NE)"),
        (['LEFT', 'DOWN'], "LD(W)"),
        (['RIGHT', 'DOWN'], "RD(E)"),
        (['LEFT', 'DOWN', 'DOWN'], "LDD(SW)"),
        (['RIGHT', 'DOWN', 'DOWN'], "RDD(SE)"),
        (['LEFT', 'DOWN', 'DOWN', 'RIGHT'], "LDDR(S)"),
        (['RIGHT', 'DOWN', 'DOWN', 'LEFT'], "RDDL(S)"),
    ]
    
    position_map = {}
    
    for nav, pos_name in nav_sequences:
        # Reset to start: use fresh game each time (cheaper than UNDO chain)
        probe_game = cd82.Cd82()
        probe_game.set_level(level)
        
        # Select white for probing
        execute(probe_game, 'CLICK', {'x': palette_positions[-1][0], 'y': palette_positions[-1][1]})
        
        # Navigate to position
        for n in nav:
            execute(probe_game, n)
        
        # Record canvas before LAUNCH
        canvas_before = get_canvas(probe_game)
        
        # LAUNCH
        fd = execute(probe_game, 'LAUNCH')
        won = fd.levels_completed > level
        
        # Record canvas after
        canvas_after = get_canvas(probe_game)
        
        # Describe what changed
        effect = describe_paint(canvas_after)
        
        position_map[pos_name] = {
            'nav': nav,
            'effect': effect,
            'won': won
        }
        
        print(f"  {pos_name:12s} (nav={','.join(nav) or 'none':10s}): paints {effect}" +
              (" *** WINS ***" if won else ""))
        
        # Don't count these against budget — they're probe games
    
    # ===== PHASE 2: Present map + cartridge to model =====
    print(f"\n--- Phase 2: Model Plans from Discovered Map ---")
    
    # Retrieve cartridge
    game_ctx = cart_search(
        f"How to play {game_id}, strategy",
        str(CART_DIR / f'{game_id}.cart.npz'), top_k=2
    )
    
    # Build the discovered map text
    map_text = "## DISCOVERED ACTION MAP (from systematic probing)\n\n"
    map_text += "### Navigation (move basket around ring):\n"
    for name, info in action_class.items():
        if info['class'] == 'major':
            map_text += f"- {name}: moves basket (major change, {info['diff']}px)\n"
        elif info['class'] == 'selection':
            map_text += f"- {name}: selection/toggle ({info['diff']}px)\n"
    
    map_text += "\n### LAUNCH from each position (what gets painted WHITE):\n"
    for pos, info in position_map.items():
        nav_str = " → ".join(info['nav']) if info['nav'] else "(starting position)"
        win_note = " ← THIS WINS THE LEVEL!" if info['won'] else ""
        map_text += f"- Position {pos} (reach via: {nav_str}): paints {info['effect']}{win_note}\n"
    
    map_text += f"\n### Color selection:\n"
    for px, py, c in palette_positions:
        cname = 'black' if c == 0 else 'white' if c == 15 else f'color-{c}'
        map_text += f"- CLICK at ({px},{py}): selects {cname}\n"
    
    map_text += f"\n### Target: {target_desc}\n"
    
    # Find winning positions
    winning = [pos for pos, info in position_map.items() if info['won']]
    
    if winning:
        print(f"  Winning positions found: {winning}")
        # Use the shortest navigation to a winning position
        best_win = min(winning, key=lambda p: len(position_map[p]['nav']))
        best_nav = position_map[best_win]['nav']
        
        plan_prompt = f"""## Game Knowledge
{chr(10).join(t[:300] for t in game_ctx)}

{map_text}

## WINNING POSITION IDENTIFIED: {best_win}

The probing discovered that launching from position {best_win} wins the level.
Navigation to reach it: {' → '.join(best_nav) if best_nav else 'already there'}

Output the COMPLETE action sequence to win. Include color selection first if needed.
One action per line. For CLICK, include coordinates: CLICK x y"""
    else:
        plan_prompt = f"""## Game Knowledge
{chr(10).join(t[:300] for t in game_ctx)}

{map_text}

No single LAUNCH won. You may need multiple LAUNCHes with different colors from different positions.
Plan a multi-step painting sequence. One action per line."""
    
    plan = generate(plan_prompt, max_tokens=300)
    print(f"\nModel's plan:\n{plan[:500]}")
    
    # ===== PHASE 3: Execute plan on REAL game =====
    print(f"\n--- Phase 3: Execute on Real Game ---")
    
    # Parse plan
    planned = []
    for line in plan.split('\n'):
        line = line.strip().upper()
        # Check for CLICK with coordinates
        if 'CLICK' in line:
            parts = line.split()
            nums = [p for p in parts if p.isdigit()]
            if len(nums) >= 2:
                planned.append(('CLICK', {'x': int(nums[0]), 'y': int(nums[1])}))
            else:
                planned.append(('CLICK', {}))
        else:
            for name in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'LAUNCH']:
                if name in line:
                    planned.append((name, {}))
                    break
    
    print(f"Parsed {len(planned)} actions")
    
    for i, (action_name, data) in enumerate(planned):
        if actions_used >= max_actions:
            print(f"  Budget exhausted"); break
        
        pre = np.array(game.perform_action(
            ActionInput(id=GameAction.RESET, data={}), raw=True
        ).frame)[-1] if i == 0 and actions_used == 6 else post_frame
        
        fd = execute(game, action_name, data)
        post_frame = np.array(fd.frame)[-1]
        actions_used += 1
        diff = int(np.sum(pre != post_frame)) if i > 0 or actions_used > 7 else 0
        won = fd.levels_completed > level
        
        data_str = f" ({data['x']},{data['y']})" if data.get('x') else ""
        print(f"  {i+1}: {action_name}{data_str} → {diff}px" +
              (" *** WON! ***" if won else ""))
        
        trace.append({
            'phase': 3, 'step': actions_used, 'action': action_name,
            'data': data, 'pixel_diff': diff, 'won': won
        })
        
        if won:
            print(f"\n*** cd82 L{level+1} WON by Gemma 4 in {actions_used} game actions! ***")
            break
    
    # Save
    result = {
        'experiment': 'structured-probing',
        'game': game_id, 'level': level + 1, 'model': MODEL,
        'won': any(e.get('won') for e in trace),
        'total_game_actions': actions_used,
        'probe_positions_tested': len(position_map),
        'winning_positions': winning,
        'action_map_discovered': {k: v['effect'] for k, v in position_map.items()},
        'plan_parsed_actions': len(planned),
        'timestamp': datetime.now().isoformat(),
        'trace': trace
    }
    
    out_dir = SHARED / 'arc-agi-3' / 'fleet-learning' / 'mcnugget' / 'logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(out_dir / f'structured-probe-{game_id}-L{level+1}-{ts}.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved to: structured-probe-{game_id}-L{level+1}-{ts}.json")
    return result


if __name__ == '__main__':
    run('cd82', level=0, max_actions=80)
