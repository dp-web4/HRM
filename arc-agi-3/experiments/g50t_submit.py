#!/usr/bin/env python3
"""Submit complete g50t solution: all 7 levels."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}

with open(os.path.join(os.path.dirname(__file__), 'g50t_solutions.json')) as f:
    data = json.load(f)

SOLUTIONS = data['solutions']

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()

total_actions = 0
for lv in range(7):
    actions = SOLUTIONS[str(lv)].split()
    total_actions += len(actions)
    for name in actions:
        fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    print(f'After L{lv}: completed={fd.levels_completed}, actions_so_far={total_actions}')
    if fd.levels_completed <= lv:
        print(f'FAILED at L{lv}!')
        sys.exit(1)

print(f'\n=== g50t COMPLETE ===')
print(f'All 7 levels solved in {total_actions} total actions')
print(f'State: {fd.state.name}')
