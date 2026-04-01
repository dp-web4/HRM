#!/usr/bin/env python3
"""
Seed membot with game-category knowledge for SAGE autonomous play.

This encodes our analysis of the 25 ARC-AGI-3 games into membot memories
that SAGE can recall during gameplay to inform its strategy.

Run once (or whenever knowledge updates):
    python3 arc-agi-3/experiments/seed_membot_knowledge.py
"""

import requests
import time

MEMBOT_URL = "http://localhost:8000"


def store(text):
    try:
        resp = requests.post(f"{MEMBOT_URL}/api/store",
            json={"text": text}, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ─── Game category knowledge ───
# These help SAGE identify what type of game it's playing from grid observations

CATEGORY_KNOWLEDGE = [
    # Placement/sequence puzzles (click + submit, multiple sections)
    "ARC-AGI-3 game category: PLACEMENT PUZZLE. "
    "Indicators: actions [5,6,7], colored indicators at top, empty slots in middle, "
    "colored items at bottom. Strategy: click bottom item to select, click middle slot "
    "to place. Match top indicator colors. Submit with action 5 when done. "
    "Games: sb26 (sequence matching).",

    # Rotation puzzles
    "ARC-AGI-3 game category: ROTATION PUZZLE. "
    "Indicators: actions [5,6,7], grid sections that can rotate. "
    "Strategy: click sections to rotate them until pattern matches target. "
    "Submit when arrangement correct. Games: lp85.",

    # Click-to-clear puzzles
    "ARC-AGI-3 game category: CLICK TO CLEAR. "
    "Indicators: actions [5,6,7], colored items scattered on grid. "
    "Strategy: click matching colored items to clear them. May need to click "
    "adjacent same-color items. Games: vc33 (pair matching).",

    # Navigation puzzles
    "ARC-AGI-3 game category: NAVIGATION. "
    "Indicators: actions [1,2,3,4] (directional movement), small character sprite. "
    "Strategy: navigate character through maze or to targets. "
    "Avoid obstacles. Games: tn36, s5i5.",

    # Pipe/connection puzzles
    "ARC-AGI-3 game category: PIPE CONNECTION. "
    "Indicators: actions [5,6,7], pipe-like shapes on grid. "
    "Strategy: click to rotate pipe segments until flow connects from source to dest. "
    "Submit when connected.",

    # Grid physics puzzles
    "ARC-AGI-3 game category: GRID PHYSICS. "
    "Indicators: actions include movement, objects affected by gravity/forces. "
    "Strategy: observe how objects move after actions. Plan sequences to guide objects.",

    # General strategy knowledge
    "ARC-AGI-3 general: action 5 = submit/select, action 6 = click with coordinates, "
    "action 7 = undo (free, no energy cost). Always try undo if a placement seems wrong.",

    "ARC-AGI-3 general: most games have 64x64 grids with 16 colors (0-15). "
    "The most common color is background. Non-background colors are game elements.",

    "ARC-AGI-3 general: games have multiple levels. Solving level 1 often reveals "
    "the pattern for subsequent levels. Each level may add complexity.",

    # Specific game knowledge from prior sessions
    "ARC-AGI-3 sb26: sequence matching puzzle. Row 1 has destination colors. "
    "Rows 57-60 have palette items. Color 2 marks empty slots in y=15-50. "
    "Click palette item then click slot to place. Colors must match destination order. "
    "Level 1+ has multiple frames connected by portals.",

    "ARC-AGI-3 lp85: rotation puzzle. Click grid sections to rotate them. "
    "Match the target pattern shown. Submit with action 5. "
    "McNugget scored 4/8 levels with smart rotation strategy.",

    "ARC-AGI-3 vc33: click-based game. Clicking colored items causes changes. "
    "Scored 1/7 levels with observation-based clicking strategy.",
]


def main():
    print("Seeding membot with ARC-AGI-3 game knowledge...")

    # Mount cartridge
    try:
        requests.post(f"{MEMBOT_URL}/api/mount",
            json={"name": "sage-sprout"}, timeout=3)
        print("Mounted sage-sprout cartridge")
    except Exception:
        print("Warning: Could not mount cartridge")

    stored = 0
    for i, knowledge in enumerate(CATEGORY_KNOWLEDGE):
        if store(knowledge):
            stored += 1
            print(f"  [{i+1}/{len(CATEGORY_KNOWLEDGE)}] Stored")
        else:
            print(f"  [{i+1}/{len(CATEGORY_KNOWLEDGE)}] Failed")
        time.sleep(0.1)  # Don't hammer membot

    # Save cartridge
    try:
        requests.post(f"{MEMBOT_URL}/api/save",
            json={"name": "sage-sprout"}, timeout=5)
        print(f"\nSaved cartridge. {stored}/{len(CATEGORY_KNOWLEDGE)} memories stored.")
    except Exception:
        print(f"\n{stored}/{len(CATEGORY_KNOWLEDGE)} stored (save failed)")


if __name__ == "__main__":
    main()
