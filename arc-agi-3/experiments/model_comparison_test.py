#!/usr/bin/env python3
"""
Quick comparison test: Gemma 3 12B vs Phi-4 14B vs Qwen 3.5 27B
Tests each model on 3 actions to measure reasoning speed and quality.
"""
import sys
import time
import requests
import numpy as np

sys.path.insert(0, ".")
from arc_agi import Arcade
from arcengine import GameAction

OLLAMA_URL = "http://localhost:11434/api/generate"
INT_TO_GAME_ACTION = {a.value: a for a in GameAction}

def test_model(model_name: str, steps: int = 3):
    """Test a single model on a game."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}\n")
    
    # Initialize game
    arcade = Arcade()
    envs = arcade.get_environments()
    env = arcade.make(envs[0].game_id)
    frame_data = env.reset()
    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]
    
    times = []
    
    for step in range(1, steps + 1):
        available = [int(a) for a in (frame_data.available_actions or [])]
        if not available:
            break
        
        # Simple prompt
        prompt = f"""You are analyzing a grid game. The grid has changed.
Available actions: {available}

Choose ONE action number that seems most likely to progress the game.
Respond ONLY with a number from the available actions."""
        
        t0 = time.time()
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=60,
            )
            response = resp.json().get("response", "").strip()
            elapsed = time.time() - t0
            times.append(elapsed)
            
            # Parse action
            try:
                action = int(response.split()[0])
                if action not in available:
                    action = available[0]
            except:
                action = available[0]
            
            print(f"Step {step}: {elapsed:.1f}s - Action {action}")
            
            # Execute
            frame_data = env.step(INT_TO_GAME_ACTION[action])
            grid = np.array(frame_data.frame)
            if grid.ndim == 3:
                grid = grid[-1]
                
        except Exception as e:
            print(f"Step {step}: ERROR - {e}")
            break
    
    if times:
        avg = sum(times) / len(times)
        print(f"\nAverage: {avg:.1f}s/action")
        return avg
    return None

if __name__ == "__main__":
    models = [
        "gemma3:12b",
        "phi4:14b",
        "qwen3.5:27b",
    ]
    
    results = {}
    for model in models:
        try:
            avg = test_model(model, steps=3)
            if avg:
                results[model] = avg
        except Exception as e:
            print(f"\n{model}: FAILED - {e}")
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    for model, avg in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model:20s}: {avg:5.1f}s/action")
