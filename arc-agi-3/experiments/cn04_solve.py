#!/usr/bin/env python3
"""Solve cn04: jigsaw connector puzzle.

Strategy: fix piece 0 at its initial position, find positions for other pieces
by connector matching. Much faster than brute force.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import Counter

arcade = Arcade()

def get_connectors(pixels, rotation):
    """Get connector positions after CW rotation."""
    p = pixels.copy()
    for _ in range((rotation % 360) // 90):
        p = np.rot90(p, k=-1)
    conns = [(x, y) for y in range(p.shape[0]) for x in range(p.shape[1]) if p[y, x] == 8]
    w, h = p.shape[1], p.shape[0]
    return conns, w, h

def solve_level(sprites_info, grid_w, grid_h):
    """Constraint-based solver. Try all sprite orderings, fix first at init pos."""
    from itertools import permutations
    n = len(sprites_info)

    # Precompute configs per sprite index
    configs = []
    for name, ix, iy, irot, pix in sprites_info:
        cfgs = []
        for rot in [0, 90, 180, 270]:
            conns, w, h = get_connectors(pix, rot)
            cfgs.append((rot, conns, w, h))
        configs.append(cfgs)

    solutions = []

    def search(order, step, placed):
        if solutions:
            return

        if step == len(order):
            all_conns = []
            for _, _, _, _, wc in placed:
                all_conns.extend(wc)
            counts = Counter(all_conns)
            if all(c == 2 for c in counts.values()):
                result = [None] * n
                for si, x, y, rot, wc in placed:
                    result[si] = (sprites_info[si][0], x, y, rot)
                solutions.append(result)
            return

        idx = order[step]

        all_existing = []
        for _, _, _, _, wc in placed:
            all_existing.extend(wc)
        existing_counts = Counter(all_existing)
        unpaired = {pos for pos, cnt in existing_counts.items() if cnt == 1}

        for rot, conns, w, h in configs[idx]:
            if not conns:
                continue

            if unpaired:
                candidate_positions = set()
                for cx, cy in conns:
                    for ux, uy in unpaired:
                        px, py = ux - cx, uy - cy
                        if 0 <= px and 0 <= py and px + w <= grid_w and py + h <= grid_h:
                            candidate_positions.add((px, py))
            else:
                ix, iy = sprites_info[idx][1], sprites_info[idx][2]
                candidate_positions = {(ix, iy)}

            for px, py in candidate_positions:
                world_conns = tuple((px + cx, py + cy) for cx, cy in conns)
                temp = list(all_existing) + list(world_conns)
                counts = Counter(temp)
                if any(c > 2 for c in counts.values()):
                    continue
                if step == len(order) - 1:
                    if any(c == 1 for c in counts.values()):
                        continue

                search(order, step + 1, placed + [(idx, px, py, rot, world_conns)])
                if solutions:
                    return

    # Try all permutations of sprite order
    for perm in permutations(range(n)):
        if solutions:
            break
        # Fix first sprite at its current position, try all its rotations
        first = perm[0]
        ix, iy = sprites_info[first][1], sprites_info[first][2]
        for rot, conns, w, h in configs[first]:
            if not conns:
                continue
            if ix + w > grid_w or iy + h > grid_h:
                continue
            wc = tuple((ix + cx, iy + cy) for cx, cy in conns)
            search(perm, 1, [(first, ix, iy, rot, wc)])
            if solutions:
                break

    if solutions:
        return solutions[0]

    # Fallback: also try moving first sprite
    for perm in permutations(range(n)):
        if solutions:
            break
        first = perm[0]
        for rot, conns, w, h in configs[first]:
            if not conns:
                continue
            for px in range(grid_w - w + 1):
                for py in range(grid_h - h + 1):
                    wc = tuple((px + cx, py + cy) for cx, cy in conns)
                    search(perm, 1, [(first, px, py, rot, wc)])
                    if solutions:
                        break
                if solutions:
                    break
            if solutions:
                break

    return solutions[0] if solutions else None


def click_sprite(env, game, sprite):
    """Click to select a sprite. Returns fd or None."""
    cam = game.camera
    for dx in range(64):
        for dy in range(64):
            result = cam.display_to_grid(dx, dy)
            if result:
                gx, gy = result
                if sprite.x <= gx < sprite.x + sprite.width and sprite.y <= gy < sprite.y + sprite.height:
                    rendered = sprite.render()
                    lx, ly = gx - sprite.x, gy - sprite.y
                    if 0 <= ly < rendered.shape[0] and 0 <= lx < rendered.shape[1]:
                        if rendered[ly, lx] >= 0:
                            return env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
    return None


# === SOLVE ===
env = arcade.make('cn04-65d47d14')
fd = env.reset()
game = env._game

total_actions = 0
all_solutions = {}

for lv in range(5):
    level = game.current_level
    sprites = list(level.get_sprites())
    grid_w, grid_h = level.grid_size

    print(f"\n=== Level {lv} ({len(sprites)} sprites) ===")

    sprites_info = []
    for s in sprites:
        pix = game.npwwu[s.name]
        conns, w, h = get_connectors(pix, s.rotation)
        world_conns = [(s.x + cx, s.y + cy) for cx, cy in conns]
        print(f"  {s.name}: ({s.x},{s.y}) rot={s.rotation} {s.width}x{s.height} conn_world={world_conns}")
        sprites_info.append((s.name, s.x, s.y, s.rotation, pix))

    selected = game.weqid.name if game.weqid else None

    sol = solve_level(sprites_info, grid_w, grid_h)
    if not sol:
        print("  NO SOLUTION")
        break

    print(f"  Solution: {sol}")
    targets = {name: (tx, ty, trot) for name, tx, ty, trot in sol}

    # Execute: process each sprite
    lv_actions = 0

    for sname, tx, ty, trot in sol:
        sprite = None
        for s in game.current_level.get_sprites():
            if s.name == sname:
                sprite = s
                break

        cur_rot = sprite.rotation
        rot_needed = ((trot - cur_rot) % 360) // 90
        dx = tx - sprite.x
        dy = ty - sprite.y

        if dx == 0 and dy == 0 and rot_needed == 0:
            continue

        # Select
        if game.weqid is None or game.weqid.name != sname:
            fd = click_sprite(env, game, sprite)
            if fd is None:
                print(f"  CLICK FAILED for {sname}")
                break
            lv_actions += 1

        # Rotate
        for _ in range(rot_needed):
            fd = env.step(GameAction.ACTION5)
            lv_actions += 1

        # Re-read sprite position (might change after rotation?)
        for s in game.current_level.get_sprites():
            if s.name == sname:
                sprite = s
                break
        dx = tx - sprite.x
        dy = ty - sprite.y

        # Move
        for _ in range(abs(dx)):
            fd = env.step(GameAction.ACTION4 if dx > 0 else GameAction.ACTION3)
            lv_actions += 1
        for _ in range(abs(dy)):
            fd = env.step(GameAction.ACTION2 if dy > 0 else GameAction.ACTION1)
            lv_actions += 1

    total_actions += lv_actions
    print(f"  {lv_actions} actions, completed={fd.levels_completed}, state={fd.state.name}")

    if fd.state.name == 'WIN':
        print(f"\n✓ ALL SOLVED! Total: {total_actions}")
        break

    if fd.levels_completed <= lv:
        # Debug
        print(f"  WIN CHECK: {game.exlcvhdjsf()}")
        for s in game.current_level.get_sprites():
            print(f"    {s.name}: ({s.x},{s.y}) rot={s.rotation} dpmge={game.dpmge.get(s.name, set())}")
        # Check connector overlap
        all_world_conns = []
        for s in game.current_level.get_sprites():
            pix = game.npwwu[s.name]
            conns, w, h = get_connectors(pix, s.rotation)
            wc = [(s.x + cx, s.y + cy) for cx, cy in conns]
            all_world_conns.extend(wc)
            print(f"    → connectors_world: {wc}")
        counts = Counter(all_world_conns)
        print(f"    Overlap counts: {dict(counts)}")
        break

    all_solutions[lv] = sol

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
