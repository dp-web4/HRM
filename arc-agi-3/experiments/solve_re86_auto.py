"""Automated solver for re86 levels 4-8 using BFS."""
import sys
sys.path.insert(0, 'environment_files/re86/4e57566e')
from re86 import Re86, sprites, levels, hfoimipuna
from arcengine import GameAction, ActionInput
import numpy as np
from collections import deque

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
SEL = GameAction.ACTION5

ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', SEL: 'SEL'}


def solve_level_bfs(level_idx, max_steps=None, prefix_actions=None):
    """BFS solver for a level. Returns action sequence or None."""
    game = Re86()
    game.set_level(level_idx)
    budget = game.current_level.get_data('StepCounter')
    if max_steps is None:
        max_steps = budget

    # Apply prefix actions first
    prefix = []
    if prefix_actions:
        for a in prefix_actions:
            game.perform_action(ActionInput(id=a))
            prefix.append(a)
            if game.level_index != level_idx:
                return prefix  # Already solved!

    # BFS state: serialize piece positions, colors, and shapes
    def get_state(g):
        pieces = g.current_level.get_sprites_by_tag('vfaeucgcyr')
        state_parts = []
        for p in pieces:
            try: color = int(hfoimipuna(p))
            except: color = -1
            center = int(p.pixels[p.height//2, p.width//2])
            # Include shape hash for deformable pieces
            shape_hash = hash(p.pixels.tobytes())
            state_parts.append((p.x, p.y, color, center, p.width, p.height, shape_hash))
        return tuple(state_parts)

    initial_state = get_state(game)

    # BFS
    queue = deque()
    visited = {initial_state: []}
    queue.append((game, initial_state, []))

    actions = [UP, DOWN, LEFT, RIGHT, SEL]
    best = None

    iterations = 0
    while queue:
        # Pop from queue, but we need fresh game states
        # Since we can't clone games, we replay from scratch
        _, state, path = queue.popleft()
        iterations += 1

        if iterations % 10000 == 0:
            print(f"  BFS iteration {iterations}, queue size {len(queue)}, path len {len(path)}")

        if len(path) >= max_steps:
            continue

        for action in actions:
            # Replay game state
            g = Re86()
            g.set_level(level_idx)
            for a in prefix + path:
                g.perform_action(ActionInput(id=a))

            # Apply new action
            g.perform_action(ActionInput(id=action))
            new_path = path + [action]

            if g.level_index != level_idx:
                print(f"  SOLVED at {len(new_path)} steps (iterations: {iterations})")
                return prefix + new_path

            new_state = get_state(g)
            if new_state not in visited:
                visited[new_state] = new_path
                queue.append((g, new_state, new_path))

    print(f"  No solution found (iterations: {iterations})")
    return None


def solve_level_dfs_limited(level_idx, max_depth, prefix_actions=None):
    """Iterative deepening DFS solver."""
    for depth in range(1, max_depth + 1):
        print(f"  Trying depth {depth}...")
        result = _dfs(level_idx, depth, prefix_actions or [])
        if result is not None:
            return result
    return None


def _dfs(level_idx, max_depth, prefix, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    # Replay
    g = Re86()
    g.set_level(level_idx)
    for a in prefix + path:
        g.perform_action(ActionInput(id=a))

    if g.level_index != level_idx:
        return prefix + path

    if len(path) >= max_depth:
        return None

    state = _get_state(g)
    if state in visited:
        return None
    visited.add(state)

    for action in [UP, DOWN, LEFT, RIGHT, SEL]:
        result = _dfs(level_idx, max_depth, prefix, path + [action], visited)
        if result is not None:
            return result

    return None


def _get_state(g):
    pieces = g.current_level.get_sprites_by_tag('vfaeucgcyr')
    parts = []
    for p in pieces:
        try: color = int(hfoimipuna(p))
        except: color = -1
        center = int(p.pixels[p.height//2, p.width//2])
        parts.append((p.x, p.y, color, center, p.width, p.height))
    return tuple(parts)


def replay_solution(level_idx, actions, replay_prefix=None):
    """Replay and verify a solution."""
    game = Re86()
    game.set_level(level_idx)

    if replay_prefix:
        for a in replay_prefix:
            game.perform_action(ActionInput(id=a))

    for i, a in enumerate(actions):
        prev = game.level_index
        game.perform_action(ActionInput(id=a))
        if game.level_index != prev:
            return True, i + 1

    return game.level_index != level_idx, len(actions)


def format_actions(actions):
    """Format action list as human-readable string."""
    parts = []
    i = 0
    while i < len(actions):
        a = actions[i]
        count = 1
        while i + count < len(actions) and actions[i + count] == a:
            count += 1
        if count > 1:
            parts.append(f"{ACTION_NAMES[a]} {count}")
        else:
            parts.append(ACTION_NAMES[a])
        i += count
    return ", ".join(parts)


if __name__ == '__main__':
    # Verified solutions for L1-L4
    l1 = [RIGHT]*4 + [UP]*7 + [SEL] + [LEFT]*2 + [UP]*6
    l2 = [LEFT]*3 + [DOWN]*10 + [SEL] + [LEFT]*6 + [UP]*6 + [SEL] + [LEFT]*7 + [DOWN]*2
    l3 = [LEFT]*2 + [UP]*13 + [SEL] + [LEFT]*9 + [UP]*6 + [SEL] + [RIGHT]*8 + [UP]*8
    l4 = [UP]*7 + [LEFT]*13 + [DOWN]*5 + [SEL] + [RIGHT]*8 + [DOWN]*8 + [UP]*5 + [LEFT]*3
    l5 = ([DOWN]*1 + [LEFT]*3 + [RIGHT]*2 + [UP]*10 + [RIGHT]*3 +
          [SEL] + [LEFT]*16 + [DOWN]*6 + [RIGHT]*9 +
          [SEL] + [RIGHT]*6 + [DOWN]*15 + [RIGHT]*1 + [UP]*9)
    l6 = ([UP]*3 + [RIGHT]*4 + [DOWN]*4 + [RIGHT]*9 + [UP]*2 +
          [SEL] + [LEFT]*12 + [DOWN]*1 + [RIGHT]*7 + [DOWN]*5 + [LEFT]*5 + [UP]*6)

    # Verify all
    for name, actions, lvl in [('L1',l1,0),('L2',l2,1),('L3',l3,2),('L4',l4,3),('L5',l5,4),('L6',l6,5)]:
        ok, steps = replay_solution(lvl, actions)
        print(f"{name}: {'PASS' if ok else 'FAIL'} ({steps} steps)")

    # L7 and L8 need solving
    # For now, try the manual approach for L7 and L8
