"""Replay verified re86 solutions for levels 1-7."""
import sys
sys.path.insert(0, 'environment_files/re86/4e57566e')
from re86 import Re86
from arcengine import GameAction, ActionInput

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
SEL = GameAction.ACTION5

SOLUTIONS = {
    # L1: blue RIGHT 4 UP 7, SEL, yellow LEFT 2 UP 6
    1: [RIGHT]*4+[UP]*7+[SEL]+[LEFT]*2+[UP]*6,

    # L2: orange LEFT 3 DOWN 10, SEL, red LEFT 6 UP 6, SEL, blue LEFT 7 DOWN 2
    2: [LEFT]*3+[DOWN]*10+[SEL]+[LEFT]*6+[UP]*6+[SEL]+[LEFT]*7+[DOWN]*2,

    # L3: line LEFT 2 UP 13, SEL, X LEFT 9 UP 6, SEL, diamond RIGHT 8 UP 8
    3: [LEFT]*2+[UP]*13+[SEL]+[LEFT]*9+[UP]*6+[SEL]+[RIGHT]*8+[UP]*8,

    # L4: cross through c12 zone to (2,17), diamond through c14 zone to (29,20)
    4: [UP]*7+[LEFT]*13+[DOWN]*5+[SEL]+[RIGHT]*8+[DOWN]*8+[UP]*5+[LEFT]*3,

    # L5: diamond23 -> c9 paint -> (19,4), cross -> c9 via zones -> (19,37),
    #     diamond19 -> c8 paint -> (42,27)
    5: ([DOWN]+[LEFT]*3+[RIGHT]*2+[UP]*10+[RIGHT]*3+
        [SEL]+[LEFT]*16+[DOWN]*6+[RIGHT]*9+
        [SEL]+[RIGHT]*6+[DOWN]*15+[RIGHT]+[UP]*9),

    # L6: deformable rect UP 3 RIGHT 4 (deform 3x to 10x28), position at (45,30)
    #     cross with border-shifts, position at (6,3) arm=(6,6)
    6: ([UP]*3+[RIGHT]*4+[DOWN]*4+[RIGHT]*9+[UP]*2+
        [SEL]+[LEFT]*12+[DOWN]+[RIGHT]*7+[DOWN]*5+[LEFT]*5+[UP]*6),

    # L7: cross19 shift arm_row, paint c11, position (36,30)
    #     rect13 deform to 19x7, paint c9, position (39,18)
    #     cross37 shift arms, paint c8, position (0,9)
    7: ([RIGHT]*3+[UP]*7+[LEFT]+[UP]*6+[RIGHT]*3+[DOWN]*3+[RIGHT]*6+[DOWN]*7+  # cross19
        [SEL]+
        [RIGHT]*5+[UP]*5+[LEFT]*6+[UP]*11+[LEFT]+[DOWN]*2+[RIGHT]*11+[DOWN]*3+  # rect13
        [SEL]+
        [RIGHT]*2+[UP]*4+[LEFT]*7+[UP]*9+[DOWN]+[RIGHT]*9+[UP]+[RIGHT]*3+[UP]*2+[DOWN]*2+[LEFT]*9),  # cross37
}


def replay_all():
    """Replay all solutions and verify."""
    game = Re86()
    results = {}

    for level_num in sorted(SOLUTIONS.keys()):
        level_idx = level_num - 1
        actions = SOLUTIONS[level_num]
        game.set_level(level_idx)
        budget = game.current_level.get_data('StepCounter')

        passed = False
        for i, a in enumerate(actions):
            prev = game.level_index
            game.perform_action(ActionInput(id=a))
            if game.level_index != prev:
                results[level_num] = (True, i + 1, budget)
                passed = True
                break

        if not passed:
            results[level_num] = (False, len(actions), budget)

    for level_num in sorted(results.keys()):
        ok, steps, budget = results[level_num]
        status = "PASS" if ok else "FAIL"
        print(f"L{level_num}: {status} ({steps} steps, budget {budget})")

    return results


if __name__ == '__main__':
    replay_all()
