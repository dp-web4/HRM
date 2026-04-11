#!/usr/bin/env python3
"""Quick probe of unsolved games to identify type and mechanics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
all_envs = arcade.get_environments()

# Games to probe
games = ['ar25', 'bp35', 'cn04', 'dc22', 'ka59', 'lf52', 'lp85', 'm0r0', 'r11l', 're86', 's5i5', 'sk48', 'su15', 'wa30']

for game_id in games:
    try:
        matching = [e for e in all_envs if e.game_id.startswith(game_id + '-') or e.game_id == game_id]
        if not matching:
            print(f'{game_id}: NO INSTANCE')
            continue
        env_info = matching[0]
        env = arcade.make(env_info.game_id)
        fd = env.reset()

        # Frame info
        frame = np.array(fd.frame)
        shape_str = 'x'.join(str(s) for s in frame.shape)

        # Game class name
        game_cls = type(env._game).__name__

        # Try available_actions if exists, else try all
        avail = getattr(env._game, 'available_actions', None)
        if avail:
            actions = [a.name for a in GameAction if a.value in avail]
        else:
            actions = []

        # Try each action and see what happens
        working_actions = {}
        for action in GameAction:
            try:
                env2 = arcade.make(env_info.game_id)
                fd2 = env2.reset()
                f_before = np.array(fd2.frame).copy()
                fd2 = env2.step(action)
                f_after = np.array(fd2.frame)
                diff = int(np.sum(f_before != f_after))
                if diff > 0:
                    working_actions[action.name] = diff
            except:
                pass

        baseline = getattr(env_info, 'baseline_actions', None)
        src_size = 0
        # Check source file size
        src_dir = f'environment_files/{game_id}'
        if os.path.isdir(src_dir):
            for root, dirs, files in os.walk(src_dir):
                for f in files:
                    if f.endswith('.py'):
                        src_size += os.path.getsize(os.path.join(root, f))

        print(f'{game_id}: cls={game_cls} frame={shape_str} working_actions={working_actions} baseline={baseline} src={src_size}B state={fd.state.name}')
    except Exception as e:
        import traceback
        print(f'{game_id}: ERROR {e}')
