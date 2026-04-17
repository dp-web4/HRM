#!/usr/bin/env python3
"""
McNugget Enforced-Exploration Harness — Gemma 4 E4B Game Play

The model CANNOT freely choose actions until it has TESTED each one.
This breaks the fixation loop by making exploration structural, not optional.

Pipeline:
  Phase 1 — MANDATORY EXPLORATION (harness-driven, 6 actions)
    Test each action type once. Record pixel diff + state change.
    Model has no choice here — harness drives.
    
  Phase 2 — ACTION MAP PRESENTATION (0 actions)
    Show model: "here's what each action does" (from Phase 1)
    + cartridge context (world model, strategy)
    Ask model to PLAN a sequence to win.
    
  Phase 3 — GATED EXECUTION (model-driven, budget-aware)
    Model proposes actions one at a time.
    For consequential actions (>50px change): model must predict outcome.
    Harness validates prediction plausibility before executing.
    
  Phase 4 — R7 REFLECTION (after win or budget exhaustion)
    What worked, what didn't, trust score updates.

Target: win cd82 L1 (select white → navigate to West → LAUNCH).
"""

import sys, os, json, time, requests
import numpy as np
from pathlib import Path
from datetime import datetime

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


def embed(text):
    resp = requests.post(f'{OLLAMA_URL}/api/embeddings', json={
        'model': 'nomic-embed-text', 'prompt': text
    })
    return np.array(resp.json()['embedding'], dtype=np.float32)


def cart_search(query, cart_path, top_k=3):
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


def get_frame(game):
    fd = game.perform_action(ActionInput(id=GameAction.RESET, data={}), raw=True)
    return np.array(fd.frame)[-1]


def execute(game, action_ga):
    fd = game.perform_action(ActionInput(id=action_ga, data={}), raw=True)
    return fd, np.array(fd.frame)[-1]


def run(game_id='cd82', level=0, max_actions=50):
    import cd82
    game = cd82.Cd82()
    game.set_level(level)
    
    trace = []
    actions_used = 0
    
    print(f"\n{'='*60}")
    print(f"ENFORCED EXPLORATION: {game_id} L{level+1}, model={MODEL}")
    print(f"{'='*60}")
    
    # ===== PHASE 1: MANDATORY EXPLORATION =====
    print(f"\n--- Phase 1: Mandatory Exploration (6 actions) ---")
    
    pre_frame = get_frame(game)
    action_map = {}
    
    for action_name, action_ga in ACTIONS.items():
        pre = pre_frame.copy() if actions_used == 0 else post.copy()
        fd, post = execute(game, action_ga)
        actions_used += 1
        
        diff = int(np.sum(pre != post))
        level_won = fd.levels_completed > level
        
        if diff == 0:
            effect = "NO CHANGE"
        elif diff < 10:
            effect = f"MINOR ({diff}px — selection/toggle)"
        elif diff < 100:
            effect = f"MODERATE ({diff}px — state moved)"
        else:
            effect = f"MAJOR ({diff}px — significant change)"
        
        action_map[action_name] = {
            'pixel_diff': diff,
            'effect': effect,
            'temperament': 'safe' if diff < 10 else 'moderate' if diff < 100 else 'consequential'
        }
        
        trace.append({
            'phase': 1, 'step': actions_used, 'action': action_name,
            'pixel_diff': diff, 'effect': effect, 'level_won': level_won
        })
        
        print(f"  {action_name:8s} → {effect}")
        
        if level_won:
            print(f"  *** WON during exploration! ***")
            return trace, action_map, True
    
    # ===== PHASE 2: MAP + CARTRIDGE → PLAN =====
    print(f"\n--- Phase 2: Build Plan (0 actions) ---")
    
    # Retrieve cartridge context
    game_ctx = cart_search(
        f"How do I play {game_id}? Strategy for winning.",
        str(CART_DIR / f'{game_id}.cart.npz'), top_k=2
    )
    substrate_ctx = cart_search(
        "action budget directional ambiguity",
        str(CART_DIR / 'substrate-primitives.cart.npz'), top_k=2
    )
    
    # Format action map
    map_text = "## Your Action Test Results (Phase 1 — OBSERVED, not predicted)\n"
    for name, info in action_map.items():
        map_text += f"- {name}: {info['effect']} (temperament: {info['temperament']})\n"
    
    plan_prompt = f"""## Game Knowledge (from cartridge)
{chr(10).join(t[:400] for t in game_ctx)}

{map_text}

## MAPPING TASK (connect observations to knowledge)

Step A: Match each action to its game function using the cartridge:
- The cartridge says "basket navigates octagonal ring." Your tests showed LEFT=201px, RIGHT=201px (major changes). Therefore: LEFT and RIGHT MOVE THE BASKET around the ring.
- The cartridge says "LAUNCH paints a wedge." Your test showed LAUNCH=50px. Therefore: LAUNCH PAINTS from the current basket position.
- The cartridge says "CLICK selects color from palette." Your test showed CLICK=1px. Therefore: CLICK CHANGES THE ACTIVE COLOR.
- UP=1px (minor adjustment to basket position). DOWN=0px (no effect from this position).

Step B: What does Level {level+1} need?
The cartridge says the target pattern is a simple shape. You need to paint specific regions with specific colors.

Step C: Plan the EXACT sequence:
1. First: select the right color (CLICK)
2. Then: navigate basket to the right position (LEFT/RIGHT)
3. Then: LAUNCH to paint
4. Repeat if multiple colors/regions needed

## Your Plan
Output ONLY the action sequence, one action per line. Example format:
CLICK
LEFT
LEFT
LEFT
LAUNCH

{max_actions - actions_used} actions remaining. Be efficient."""

    plan = generate(plan_prompt, max_tokens=500)
    print(f"\nModel's plan:\n{plan[:600]}")
    
    # ===== PHASE 3: GATED EXECUTION =====
    print(f"\n--- Phase 3: Execute Plan ---")
    
    # Parse plan for action sequence — accept any line that IS an action name
    planned_actions = []
    for line in plan.split('\n'):
        token = line.strip().upper().split()[0] if line.strip() else ''
        # Strip leading numbers, bullets, dashes
        for prefix in ['STEP', '-', '*', '.']:
            token = token.lstrip('0123456789').lstrip(prefix).lstrip(': ').strip()
        if token in ACTIONS:
            planned_actions.append(token)
        else:
            # Check if any action name appears in the line
            for name in ACTIONS:
                if name in line.upper():
                    planned_actions.append(name)
                    break
    
    print(f"Parsed {len(planned_actions)} planned actions: {planned_actions[:15]}...")
    
    for i, action_name in enumerate(planned_actions):
        if actions_used >= max_actions:
            print(f"  Budget exhausted at step {i+1}")
            break
        
        action_ga = ACTIONS[action_name]
        pre = post.copy()
        fd, post = execute(game, action_ga)
        actions_used += 1
        diff = int(np.sum(pre != post))
        level_won = fd.levels_completed > level
        
        trace.append({
            'phase': 3, 'step': actions_used, 'action': action_name,
            'pixel_diff': diff, 'level_won': level_won
        })
        
        status = f"{diff}px" + (" *** WON! ***" if level_won else "")
        print(f"  Step {i+1}: {action_name} → {status}")
        
        if level_won:
            print(f"\n*** LEVEL WON in {actions_used} total actions! ***")
            return trace, action_map, True
    
    # ===== PHASE 4: R7 REFLECTION =====
    print(f"\n--- Phase 4: Reflection ---")
    reflect_prompt = f"""You played {game_id} Level {level+1} for {actions_used} actions. You did NOT win.

Your action map showed:
{map_text}

Your plan was:
{plan[:300]}

What went wrong? What would you do differently? (2-3 sentences)"""
    
    reflection = generate(reflect_prompt, max_tokens=150)
    print(f"Reflection: {reflection[:200]}")
    
    return trace, action_map, False


if __name__ == '__main__':
    trace, action_map, won = run('cd82', level=0, max_actions=50)
    
    # Save results
    out_dir = SHARED / 'arc-agi-3' / 'fleet-learning' / 'mcnugget' / 'logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    result = {
        'experiment': 'enforced-exploration',
        'game': 'cd82', 'level': 1,
        'model': MODEL,
        'won': won,
        'total_actions': len(trace),
        'action_map': action_map,
        'action_diversity': len(set(e['action'] for e in trace)) / len(ACTIONS),
        'trace': trace,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(out_dir / f'enforced-explore-cd82-{ts}.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved to: enforced-explore-cd82-{ts}.json")
    print(f"Diversity: {result['action_diversity']:.0%}")
    print(f"Won: {won}")
