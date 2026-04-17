#!/usr/bin/env python3
"""Test: Can Qwen 0.8B play bp35 L0 with text descriptions?

This tests the core Phase 3 question: what knowledge format helps
a small model make correct game decisions?

bp35 L0: Player at (3,23), gem at (3,7), gravity UP.
Path: go RIGHT 4 times to x=7, click tiles to clear path, fall to gem.
15 actions total. Simple enough for a baseline test.
"""
import sys, json, time, requests
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')
from arc_agi import Arcade
from arcengine import GameAction

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6

OLLAMA = "http://localhost:11434/api/generate"

def ask_model(prompt, max_tokens=30):
    """Query Qwen 0.8B via Ollama."""
    r = requests.post(OLLAMA, json={
        "model": "qwen3.5:0.8b",
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3}
    }, timeout=30)
    return r.json().get("response", "").strip()

def frame_to_text(frame):
    """Convert 64x64 frame to compact text description."""
    # Just describe what's visible: player position, gem position, obstacles
    from collections import Counter
    colors = Counter()
    for row in frame:
        for c in row:
            colors[c] += 1
    # Find non-background objects by color regions
    bg = colors.most_common(1)[0][0]
    objects = []
    for y in range(64):
        for x in range(64):
            c = frame[y][x]
            if c != bg and c != 2:  # not bg, not padding
                objects.append((x, y, c))
    return f"Frame has {len(objects)} non-background pixels across {len(set(c for _,_,c in objects))} colors."

# ============================================================
print("=== Small Model Game Test: bp35 L0 ===\n")

arc = Arcade()
env = arc.make('bp35-0a0ad940')
obs = env.reset()

engine = env._game.oztjzzyqoek
player = engine.twdpowducb
pp = tuple(player.qumspquyus)
grav = engine.vivnprldht

print(f"Player: {pp}, Gravity: {'UP' if grav else 'DOWN'}")
print(f"Actions: {obs.available_actions}")

# Test different prompt formats
prompts = {
    "minimal": """You are playing a puzzle game. Player at (3,23). Gem at (3,7). Gravity pulls UP.
Actions: LEFT, RIGHT, CLICK(x,y).
What is your first action? Reply with JUST the action.""",

    "with_context": """GAME: Side-scrolling puzzle platformer.
RULES: Player moves LEFT/RIGHT. Gravity pulls UP (player falls upward). CLICK destroys ground tiles. Reach the gem to win.
STATE: Player at (3,23). Gem at (3,7). Between them: solid ground at y=9,12,15,19,22 with gaps.
The gem is UP and to the LEFT of the player.
ACTIONS: LEFT, RIGHT, CLICK(x,y)
What should you do first? Reply with ONE action.""",

    "with_strategy": """GAME: bp35 puzzle platformer. Player moves horizontally, falls vertically (gravity UP).
STRATEGY: Click ground tiles above you to create gaps. Fall through gaps toward the gem.
STATE: Player at (3,23). Gem at (3,7). There's a wall gap at x=7,y=22 that leads upward.
Going RIGHT to x=7 would let you fall through the gap.
What action? Reply: LEFT, RIGHT, or CLICK(x,y)""",
}

print("\n--- Testing different prompt formats ---\n")
for name, prompt in prompts.items():
    t0 = time.time()
    response = ask_model(prompt)
    dt = time.time() - t0
    print(f"  {name:15s}: '{response}' ({dt:.1f}s)")

# Now test: can the model play a sequence?
print("\n--- Sequential play test (with_strategy prompt) ---\n")

# Manually describe state after each action and ask for next
states = [
    ("Player at (3,23). Gap at (7,22) leads UP. Go RIGHT to reach it.", "RIGHT"),
    ("Player at (4,23). Gap at (7,22) is 3 cells RIGHT.", "RIGHT"),
    ("Player at (5,23). Gap at (7,22) is 2 cells RIGHT.", "RIGHT"),
    ("Player at (6,23). Gap at (7,22) is 1 cell RIGHT.", "RIGHT"),
    ("Player at (7,20). You fell UP through the gap! Ground tile at (7,19) blocks you. CLICK it.", "CLICK(7,19)"),
    ("Player at (7,16). Ground at (4,16) blocks LEFT path. Need to clear it.", "CLICK(4,16)"),
    ("Tile cleared. Go LEFT toward x=4.", "LEFT"),
    ("Player at (6,16). Continue LEFT.", "LEFT"),
    ("Player at (5,16). Continue LEFT.", "LEFT"),
    ("Player at (4,13). Fell through! Ground at (4,12) above. CLICK it.", "CLICK(4,12)"),
]

correct = 0
total = 0
for state_desc, expected in states:
    total += 1
    prompt = f"""GAME: bp35 platformer. Click tiles to destroy them. Fall through gaps (gravity UP).
STATE: {state_desc}
ACTIONS: LEFT, RIGHT, CLICK(x,y)
What action? Reply with JUST the action."""

    response = ask_model(prompt, max_tokens=20)
    match = response.upper().replace(" ", "").startswith(expected.upper().replace(" ", ""))
    if match:
        correct += 1
    print(f"  Step {total:2d}: expected={expected:15s} got='{response}' {'✓' if match else '✗'}")

print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.0f}%)")

# Restart daemon when done
import subprocess
subprocess.run(["sudo", "systemctl", "start", "sage-daemon-sprout.service"], capture_output=True)
print("\nSAGE daemon restarted.")
