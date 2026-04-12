#!/usr/bin/env python3
"""g50t full solver: L0-L5 known solutions (242 actions for 6/7 levels).

Multi-phase clone maze puzzle mechanics:
- Player moves on 6px grid
- ACTION5 = PHASE transition: records moves, creates clone, resets player to start
- Clones replay recorded moves simultaneously with each player move
- Modifiers (buttons) toggle obstacles when stepped on
- Non-toggle obstacles stay shifted while someone is on the modifier
- Toggle obstacles stay shifted until re-triggered (ignore deactivation signal)
- Ghost moves autonomously and can activate modifiers
- Win: player reaches goal position (goal.x+1, goal.y+1)
- Timer ticks every 2 steps — runs out = lose
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
PHASE = GameAction.ACTION5

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
DIR_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', PHASE: 'UNDO'}

KNOWN = {
    # L0: 17 actions. Simple two-phase: right then down+right.
    0: 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT',

    # L1: 31 actions. Three phases with modifier activation.
    1: 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT',

    # L2: 57 actions. Three phases, complex navigation with obstacle toggling.
    2: 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP',

    # L3: 31 actions. Three phases with gate/modifier interaction.
    3: 'DOWN DOWN RIGHT DOWN UNDO DOWN DOWN RIGHT RIGHT UP UP RIGHT RIGHT DOWN DOWN DOWN UNDO LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT LEFT LEFT LEFT',

    # L4: 42 actions. Three phases, portal-swap mechanics (mpreboxmgc pads).
    4: 'UP DOWN DOWN RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN RIGHT LEFT DOWN LEFT LEFT LEFT LEFT LEFT UP UP',

    # L5: 49 actions. Three phases + ghost-activated modifiers.
    # Phase 0 (3 moves): Player walks to mod[0] at (43,25), creating a clone.
    # Phase 1 (5 moves): Player walks to mod[1] at (31,25), creating a second clone.
    # Phase 2 (39 moves): Both clones hold mods 0,1 clearing row-7 obstacles.
    #   Ghost traverses row 7 to toggle mods 2,3 which clear row-25 obstacles.
    #   Player navigates through row 25 to reach mod[4] at (13,49), clearing the
    #   path obstacle at (31,49). Then returns through row 25 and navigates to goal.
    5: 'LEFT LEFT UP UNDO LEFT LEFT UP LEFT LEFT UNDO LEFT LEFT DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT UP DOWN LEFT LEFT UP UP UP UP UP RIGHT RIGHT RIGHT DOWN DOWN RIGHT RIGHT DOWN DOWN RIGHT RIGHT',

    # L6: 69 actions. Portal-swap + ghost-activated obstacle toggling + non-toggle gate holding.
    # Phase 0 (16 moves): ghost0 pacing path. Ends at (31,31) but timing is what matters —
    #   its replay will reach portal pad (13,25) at step 8 of P2, then teleport to (13,13),
    #   walk UP UP to mod[3] at (13,1), and stay there holding obs[0] open.
    # Phase 1 (24 moves): ghost1 path. Step 3 hits mod[2] at (19,37) toggling obs[1] OFF,
    #   letting the wild ghost patrol up through (1,37) to reach mod[0] at (1,25),
    #   firing the left portal swap (13,13)↔(13,25). Ghost1 then navigates via (31,1)
    #   (which is open because ghost0 holds mod[3]) to mod[1] at (49,1), firing the
    #   right portal swap (49,37)↔(49,49).
    # Phase 2 (27 moves): player navigates to portal pad (49,37), paces while waiting
    #   for ghost1 to reach mod[1] and fire the right swap, then teleports to (49,49)
    #   and walks LLL to goal at (31,49).
    6: 'DOWN UP DOWN UP DOWN UP LEFT LEFT RIGHT RIGHT DOWN DOWN RIGHT LEFT UP UP UNDO '
       'DOWN DOWN LEFT RIGHT UP UP DOWN DOWN UP UP DOWN DOWN RIGHT RIGHT UP UP UP UP '
       'LEFT UP UP RIGHT RIGHT RIGHT UNDO '
       'DOWN DOWN RIGHT RIGHT UP UP UP UP RIGHT RIGHT DOWN DOWN DOWN DOWN UP DOWN UP '
       'DOWN UP DOWN UP DOWN UP DOWN LEFT LEFT LEFT',
}


def main():
    print("=" * 60)
    print("g50t Final Solver — L0-L5 (6/7 levels)")
    print("=" * 60)

    arcade = Arcade()
    env = arcade.make('g50t-5849a774')
    fd = env.reset()
    game = env._game

    total_actions = 0
    results = {}

    for level in range(7):
        print(f"\n{'='*50}")
        print(f"Level {level} (engine={game.level_index})")
        print(f"{'='*50}")

        if fd.state.name in ('WON', 'GAME_OVER'):
            print(f"  Game ended: {fd.state.name}")
            break

        if level in KNOWN:
            sol_str = KNOWN[level]
            n = len(sol_str.split())
            print(f"  Using known solution: {n} actions")
            completed_before = fd.levels_completed
            for name in sol_str.split():
                fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
            advanced = fd.levels_completed > completed_before or fd.state.name == 'WON'
            if advanced:
                print(f"  L{level} SOLVED! (levels_completed={fd.levels_completed}, state={fd.state.name})")
                results[level] = n
                total_actions += n
            else:
                print(f"  Known solution FAILED! (levels_completed={fd.levels_completed}, state={fd.state.name})")
                break
        else:
            print(f"  No solution available for L{level}")
            break

    print(f"\n{'='*60}")
    print(f"FINAL: {len(results)}/7 levels solved, {total_actions} total actions")
    for lv, n in sorted(results.items()):
        print(f"  L{lv}: {n} actions")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
