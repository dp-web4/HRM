#!/usr/bin/env python3
"""
ls20 Visual Capture — Replay all 7 solved levels through SDK, save every frame.
Outputs to shared-context/arc-agi-3/visual-memory/ls20/ for fleet knowledge.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, get_all_frames

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4}

SDK_PALETTE = {
    0: (255,255,255), 1: (204,204,204), 2: (153,153,153), 3: (102,102,102),
    4: (51,51,51), 5: (0,0,0), 6: (229,58,163), 7: (255,123,204),
    8: (249,60,49), 9: (30,147,255), 10: (136,216,241), 11: (255,220,0),
    12: (255,133,27), 13: (146,18,49), 14: (79,204,48), 15: (163,86,214),
}

SOLUTIONS = {
    1: "LEFT LEFT LEFT UP UP UP UP RIGHT RIGHT RIGHT UP UP UP",
    2: "UP RIGHT UP UP UP UP UP RIGHT RIGHT DOWN RIGHT DOWN DOWN DOWN DOWN DOWN DOWN UP DOWN DOWN LEFT LEFT RIGHT UP RIGHT UP UP UP UP UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT DOWN LEFT DOWN DOWN DOWN DOWN DOWN",
    3: "UP UP UP UP UP UP UP UP LEFT DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN UP UP UP LEFT LEFT UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT UP UP UP LEFT UP DOWN UP RIGHT DOWN",
    4: "LEFT LEFT LEFT DOWN DOWN DOWN LEFT DOWN DOWN LEFT LEFT UP DOWN UP DOWN UP DOWN UP UP LEFT LEFT UP DOWN LEFT LEFT UP UP UP DOWN DOWN RIGHT UP UP UP UP RIGHT UP RIGHT UP UP LEFT LEFT LEFT",
    5: "UP LEFT UP UP LEFT LEFT LEFT RIGHT LEFT RIGHT LEFT RIGHT UP UP LEFT LEFT LEFT LEFT UP LEFT LEFT LEFT RIGHT UP DOWN RIGHT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT UP",
    6: "UP UP DOWN UP DOWN RIGHT RIGHT UP RIGHT UP UP UP LEFT LEFT RIGHT RIGHT UP UP RIGHT RIGHT UP UP RIGHT DOWN DOWN UP UP LEFT UP DOWN LEFT LEFT LEFT LEFT UP DOWN LEFT LEFT DOWN DOWN DOWN DOWN UP RIGHT RIGHT RIGHT LEFT LEFT LEFT UP UP UP UP UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN RIGHT RIGHT UP UP RIGHT DOWN DOWN DOWN DOWN DOWN",
    7: "UP UP DOWN DOWN LEFT LEFT DOWN DOWN DOWN DOWN DOWN UP DOWN RIGHT DOWN UP RIGHT UP DOWN UP DOWN UP DOWN UP DOWN LEFT LEFT UP UP UP RIGHT RIGHT RIGHT RIGHT UP RIGHT RIGHT UP RIGHT RIGHT UP UP RIGHT DOWN DOWN LEFT LEFT LEFT UP DOWN DOWN DOWN DOWN",
}


def render_frame_png(grid, path, scale=4):
    """Save palette-indexed grid as RGB PNG."""
    if grid.ndim == 3:
        grid = grid[-1]
    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = SDK_PALETTE.get(int(grid[r, c]), (128, 128, 128))
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    Image.fromarray(img).save(path)


def render_gif(frames, path, scale=4, duration=100):
    """Save animation frames as GIF."""
    images = []
    for grid in frames:
        if grid.ndim == 3:
            grid = grid[-1]
        h, w = grid.shape
        img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                color = SDK_PALETTE.get(int(grid[r, c]), (128, 128, 128))
                img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
        images.append(Image.fromarray(img))
    if images:
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=duration, loop=0)


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'/home/sprout/ai-workspace/shared-context/arc-agi-3/visual-memory/ls20/run_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    arcade = Arcade()
    env = arcade.make('ls20-9607627b')
    fd = env.reset()

    run_meta = {
        'game_id': 'ls20-9607627b',
        'player': 'sprout-solver',
        'started': timestamp,
        'win_levels': fd.win_levels,
        'baseline': 546,
        'total_actions': 309,
        'steps': [],
    }

    # Save initial frame (level 1 start)
    grid = get_frame(fd)
    init_path = f'level_0_start.png'
    render_frame_png(grid, os.path.join(out_dir, init_path))
    print(f"Saved initial frame: {init_path}")

    step_num = 0
    for level_num in range(1, 8):
        solution = SOLUTIONS[level_num]
        actions = solution.split()
        prev_level = fd.levels_completed

        # Save level start frame
        grid = get_frame(fd)
        start_path = f'level_{level_num - 1}_start.png'
        if step_num > 0:  # don't re-save L1 start
            render_frame_png(grid, os.path.join(out_dir, start_path))

        print(f"\n=== Level {level_num} ({len(actions)} actions) ===")

        for i, action_name in enumerate(actions):
            a = NAME_TO_INT[action_name]
            prev_grid = grid.copy() if grid is not None else None

            fd = env.step(INT_TO_GA[a])
            all_frames = get_all_frames(fd)
            grid = all_frames[-1]

            step_num += 1
            n_changed = int(np.sum(prev_grid != grid)) if prev_grid is not None else 0
            level_up = fd.levels_completed > prev_level

            # Save frame
            png_name = f'step_{step_num:04d}_L{level_num - 1}_{action_name}.png'
            render_frame_png(grid, os.path.join(out_dir, png_name))

            step_entry = {
                'step': step_num,
                'level': level_num - 1,
                'action': action_name,
                'png': png_name,
                'px_changed': n_changed,
            }

            # Save animation GIF if multi-frame
            if len(all_frames) > 1:
                gif_name = f'step_{step_num:04d}_L{level_num - 1}_anim.gif'
                render_gif(all_frames, os.path.join(out_dir, gif_name))
                step_entry['gif'] = gif_name
                step_entry['anim_frames'] = len(all_frames)

            if level_up:
                step_entry['level_up'] = True
                step_entry['new_level'] = fd.levels_completed
                prev_level = fd.levels_completed
                # Save the post-transition frame as new level start
                grid = get_frame(fd)
                new_start = f'level_{fd.levels_completed}_start.png'
                render_frame_png(grid, os.path.join(out_dir, new_start))

            run_meta['steps'].append(step_entry)

            sym = '★' if level_up else ('+' if n_changed > 10 else '·')
            anim = f' [{len(all_frames)}f]' if len(all_frames) > 1 else ''
            print(f"  {sym} {step_num:3d}. {action_name:5s} | {n_changed:4d}px | L={fd.levels_completed}/{fd.win_levels}{anim}")

            if fd.state.name in ('GAME_OVER', 'WON'):
                break

        if fd.state.name == 'WON':
            break

    run_meta['finished'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_meta['result'] = fd.state.name
    run_meta['levels_completed'] = fd.levels_completed
    run_meta['total_steps'] = step_num

    with open(os.path.join(out_dir, 'run.json'), 'w') as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Result: {fd.state.name}, levels: {fd.levels_completed}/{fd.win_levels}")
    print(f"Total steps: {step_num}")
    print(f"Output: {out_dir}")
    print(f"Files: {len(os.listdir(out_dir))}")


if __name__ == '__main__':
    main()
