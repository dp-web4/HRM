#!/usr/bin/env python3
"""
McNugget R7 Feedback Loop — Cartridge-Augmented Game Play with Trust Scoring

Tests: can Gemma 4 E4B learn from game-play feedback when given world model context?

Protocol:
1. Load substrate + game cartridge → retrieve relevant context
2. Present game state as text description
3. Model chooses action + predicts consequence
4. Execute action, observe actual result
5. R7 reflection: model compares prediction vs reality, updates trust
6. Repeat for N actions, measure trust calibration improvement

Produces: R6/R7 action traces with T3 trust scores per action.
"""

import sys, os, json, time, requests
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup paths
SAGE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SAGE_DIR))
SHARED = SAGE_DIR.parent / 'shared-context'
CART_DIR = SHARED / 'arc-agi-3' / 'phase2' / 'carts'

# Game setup
GAME_DIR = SHARED / 'environment_files' / 'cd82' / 'fb555c5d'
sys.path.insert(0, str(GAME_DIR))

from arcengine.enums import GameAction
from arcengine import ActionInput

OLLAMA_URL = 'http://localhost:11434'
MODEL = 'gemma4:e4b'

ACTION_MAP = {
    "UP": GameAction.ACTION1, "DOWN": GameAction.ACTION2,
    "LEFT": GameAction.ACTION3, "RIGHT": GameAction.ACTION4,
    "LAUNCH": GameAction.ACTION5, "CLICK": GameAction.ACTION6
}
ACTION_NAMES = {v: k for k, v in ACTION_MAP.items()}


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


def generate(prompt, max_tokens=300):
    resp = requests.post(f'{OLLAMA_URL}/api/generate', json={
        'model': MODEL,
        'prompt': prompt,
        'stream': False,
        'think': False,
        'options': {'temperature': 0.3, 'num_predict': max_tokens}
    }, timeout=120)
    return resp.json().get('response', '')


def describe_state(game, level_num):
    """Text description of current game state."""
    sprites = game.current_level.get_sprites()
    # Extract key state
    frame = np.array(game.perform_action(
        ActionInput(id=GameAction.RESET, data={}), raw=True
    ).frame)[-1]
    
    # Simple description based on canvas state
    desc = f"cd82 Level {level_num + 1}.\n"
    desc += f"Canvas: 10x10 grid. "
    desc += f"Basket position: visible on octagonal ring. "
    desc += f"Available actions: UP(1) DOWN(2) LEFT(3) RIGHT(4) LAUNCH(5) CLICK(6).\n"
    return desc


def run_r7_loop(game_id='cd82', level=0, max_actions=20):
    """Run R7 feedback loop on one level."""
    
    import cd82
    game = cd82.Cd82()
    game.set_level(level)
    
    # Retrieve cartridge context
    game_context = cart_search(
        f"How do I play {game_id}? What does each action do?",
        str(CART_DIR / f'{game_id}.cart.npz'), top_k=3
    )
    substrate_context = cart_search(
        "action budget and directional ambiguity",
        str(CART_DIR / 'substrate-primitives.cart.npz'), top_k=2
    )
    
    context = "## Game Knowledge (retrieved from cartridge)\n\n"
    for text in game_context:
        context += text[:500] + "\n\n"
    context += "## General Principles\n\n"
    for text in substrate_context:
        context += text[:300] + "\n\n"
    
    # Trust scores per action (T3: Talent, Training, Temperament)
    trust = {}
    for a_name in ACTION_MAP.keys():
        trust[a_name] = {
            'talent': 'unknown',      # what it does
            'training': 0,            # times observed
            'temperament': 'unknown', # reversible? costly?
            'last_prediction': None,
            'prediction_accuracy': 0,
            'predictions_made': 0
        }
    
    # Action trace (R6/R7 tuples)
    trace = []
    
    print(f"\n{'='*60}")
    print(f"R7 Loop: {game_id} L{level+1}, {max_actions} actions, model={MODEL}")
    print(f"{'='*60}\n")
    
    for step in range(max_actions):
        # Build state description
        trust_summary = "\n".join([
            f"  {name}: training={t['training']}, talent={t['talent']}, temperament={t['temperament']}"
            for name, t in trust.items()
        ])
        
        prompt = f"""{context}

## Current State
Step {step+1}/{max_actions}. Actions used: {step}.

## Your Trust Scores (what you've learned so far)
{trust_summary}

## Task
Choose ONE action and predict what will happen.

Respond in this exact format:
ACTION: <action name from UP/DOWN/LEFT/RIGHT/LAUNCH/CLICK>
PREDICTION: <what you expect will happen>
CONFIDENCE: <LOW/MEDIUM/HIGH based on your training count>
REASONING: <why this action, referencing your trust scores>"""
        
        # Get model's action choice
        response = generate(prompt, max_tokens=200)
        
        # Parse response
        action_name = None
        prediction = None
        confidence = 'LOW'
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('ACTION:'):
                action_name = line.split(':',1)[1].strip().upper()
            elif line.startswith('PREDICTION:'):
                prediction = line.split(':',1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                confidence = line.split(':',1)[1].strip().upper()
        
        if not action_name or action_name not in ACTION_NAMES.values():
            action_name = 'UP'  # fallback
        
        action_ga = ACTION_MAP.get(action_name, GameAction.ACTION1)
        
        # Execute action
        pre_frame = np.array(game.perform_action(
            ActionInput(id=GameAction.RESET, data={}), raw=True
        ).frame)[-1] if step == 0 else post_frame
        
        fd = game.perform_action(
            ActionInput(id=action_ga, data={}), raw=True
        )
        post_frame = np.array(fd.frame)[-1]
        
        # Measure change
        pixel_diff = int(np.sum(pre_frame != post_frame))
        level_won = fd.levels_completed > level
        
        # Classify result
        if pixel_diff == 0:
            result = "NO CHANGE (action had no visible effect)"
        elif pixel_diff < 10:
            result = f"MINOR CHANGE ({pixel_diff} pixels — likely selection/toggle)"
        elif pixel_diff < 100:
            result = f"MODERATE CHANGE ({pixel_diff} pixels — state moved)"
        else:
            result = f"MAJOR CHANGE ({pixel_diff} pixels — significant action)"
        
        if level_won:
            result = "LEVEL WON!"
        
        # R7 Reflection: ask model what it learned
        r7_prompt = f"""You just performed action {action_name}.
Your prediction was: {prediction}
Actual result: {result}

What did you learn? Update your understanding:
1. Was your prediction correct? (yes/partially/no)
2. What does {action_name} actually do? (update talent)
3. Is {action_name} safe to repeat? (update temperament: safe/risky/irreversible)
4. Confidence update: should you trust {action_name} more or less?

Respond briefly (2-3 sentences)."""
        
        reflection = generate(r7_prompt, max_tokens=150)
        
        # Update trust scores
        t = trust[action_name]
        t['training'] += 1
        t['last_prediction'] = prediction
        if pixel_diff > 0:
            t['talent'] = f'produces {pixel_diff}px change'
        else:
            t['talent'] = 'no visible effect from this position'
        if pixel_diff < 10:
            t['temperament'] = 'safe (minimal change)'
        elif pixel_diff < 100:
            t['temperament'] = 'moderate risk'
        else:
            t['temperament'] = 'consequential (large change)'
        
        # Log R6/R7 tuple
        entry = {
            'step': step + 1,
            'action': action_name,
            'prediction': prediction,
            'confidence': confidence,
            'result': result,
            'pixel_diff': pixel_diff,
            'level_won': level_won,
            'reflection': reflection[:200],
            'trust_update': {action_name: dict(t)}
        }
        trace.append(entry)
        
        print(f"Step {step+1}: {action_name} (conf={confidence}) → {result}")
        print(f"  Prediction: {prediction}")
        print(f"  Reflection: {reflection[:100]}...")
        print()
        
        if level_won:
            print("*** LEVEL WON! ***")
            break
    
    # Summary
    actions_used = [e['action'] for e in trace]
    unique_actions = len(set(actions_used))
    training_counts = {name: t['training'] for name, t in trust.items() if t['training'] > 0}
    
    summary = {
        'game': game_id,
        'level': level + 1,
        'model': MODEL,
        'total_actions': len(trace),
        'unique_actions': unique_actions,
        'action_diversity': unique_actions / len(ACTION_MAP),
        'level_won': any(e['level_won'] for e in trace),
        'training_counts': training_counts,
        'trust_final': {name: {
            'talent': t['talent'],
            'training': t['training'],
            'temperament': t['temperament']
        } for name, t in trust.items() if t['training'] > 0},
        'cartridges_used': ['substrate-primitives.cart.npz', f'{game_id}.cart.npz'],
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {unique_actions}/{len(ACTION_MAP)} actions tested, "
          f"diversity={summary['action_diversity']:.0%}")
    print(f"Trust: {json.dumps(training_counts, indent=2)}")
    print(f"Won: {summary['level_won']}")
    
    # Save results
    out_dir = SHARED / 'arc-agi-3' / 'fleet-learning' / 'mcnugget' / 'logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(out_dir / f'r7-{game_id}-L{level+1}-{ts}.json', 'w') as f:
        json.dump({'summary': summary, 'trace': trace}, f, indent=2)
    
    print(f"Saved to: r7-{game_id}-L{level+1}-{ts}.json")
    return summary


if __name__ == '__main__':
    run_r7_loop('cd82', level=0, max_actions=20)
