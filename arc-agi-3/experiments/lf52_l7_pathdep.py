#!/usr/bin/env python3
"""
lf52 L7 path-dependency investigation (session 6).

Hypothesis: the way we arrive at L7 affects the L7 initial state. Five previous
agents assumed L7 init is deterministic. This test compares full engine state
produced by multiple paths to L7 and reports every field that differs.

Paths tested:
  A: Direct set_level(6) from fresh game.
  B: Sequential play L1-L6 using cached solutions, then observe L7.
  C: Direct set_level(6) after solving L1-L3 (lazy-init probe).
  D: Direct set_level(6) after solving L1 only.
  E: Variation of B with different L5 action ordering (if applicable).

For each, dumps EVERY attribute of the game, the eq, every sprite in the scene
graph, and every sprite's pixels. Diff is reported field by field.
"""

import os, sys, json, importlib
os.chdir('/mnt/c/exe/projects/ai-agents/SAGE')
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

# Re-use solver pieces
from arc_agi import Arcade
from arcengine import GameAction

# Import solver helpers
spec_path = 'arc-agi-3/experiments'
sys.path.insert(0, spec_path)
from lf52_solve_final import (
    extract_state, make_puzzle_state, solve_level, solve_jumps_only,
    solve_unified, execute_actions, DIR_ACTIONS, DIR_NAMES,
)


def dump_full_state(game, label=''):
    """Serialize EVERY observable attribute into a comparable dict."""
    eq = game.ikhhdzfmarl
    snap = {}

    # ---- Game-level attrs ----
    game_attrs = {}
    for k in sorted(vars(game).keys()):
        v = getattr(game, k)
        game_attrs[k] = _repr_val(v)
    snap['game_attrs'] = game_attrs

    # ---- eq-level attrs ----
    eq_attrs = {}
    for k in sorted(vars(eq).keys()):
        v = getattr(eq, k)
        eq_attrs[k] = _repr_val(v)
    snap['eq_attrs'] = eq_attrs

    # ---- Scene graph (full walk) ----
    snap['scene_graph'] = _walk_scene(eq, max_depth=8)

    # ---- Derived game board state ----
    try:
        snap['board'] = extract_state(eq)
        # Make hashable/JSON-able
        snap['board'] = {
            'walkable': sorted(snap['board']['walkable']),
            'pushable': sorted(snap['board']['pushable']),
            'walls': sorted(snap['board']['walls']),
            'pieces': sorted(snap['board']['pieces'].items()),
            'fixed_pegs': sorted(snap['board']['fixed_pegs']),
            'level': snap['board']['level'],
            'offset': snap['board']['offset'],
        }
    except Exception as e:
        snap['board'] = f'ERR: {e}'

    # ---- Frame pixels ----
    try:
        frame = eq.vclswpkbjs()
        snap['frame_hash'] = int(hash(frame.tobytes()) & 0xFFFFFFFF)
        snap['frame_shape'] = frame.shape
        snap['frame_sum'] = int(frame.sum())
    except Exception as e:
        snap['frame_hash'] = f'ERR: {e}'

    # ---- numpy RNG state ----
    try:
        rs = np.random.get_state()
        snap['np_rng'] = (rs[0], int(rs[1][0]), int(rs[2]), int(rs[3]), float(rs[4]))
    except Exception as e:
        snap['np_rng'] = f'ERR: {e}'

    return snap


_seen_ids = None


def _walk_scene(obj, max_depth=6, depth=0, path=''):
    """Walk scene-graph sprites, capturing name/pos/visible/tags/pixels-hash."""
    global _seen_ids
    if depth == 0:
        _seen_ids = set()
    if id(obj) in _seen_ids:
        return {'__cycle__': path}
    _seen_ids.add(id(obj))

    out = {}
    out['__type__'] = type(obj).__name__
    for attr in ('name', 'klxajkxujf', 'grid_x', 'grid_y', 'x', 'y',
                 'visible', 'rotation', 'wslpmugjcyi', 'cdpcbbnfdp',
                 'tags', 'layer'):
        if hasattr(obj, attr):
            try:
                v = getattr(obj, attr)
                out[attr] = _repr_val(v, shallow=True)
            except Exception as e:
                out[attr] = f'ERR: {e}'
    # pixels
    for pattr in ('aujbzpqexr', 'pixels', '_pixels'):
        if hasattr(obj, pattr):
            try:
                p = getattr(obj, pattr)
                if isinstance(p, np.ndarray):
                    out[pattr + '_hash'] = int(hash(p.tobytes()) & 0xFFFFFFFF)
                    out[pattr + '_shape'] = p.shape
            except Exception:
                pass
    # children (common child collections)
    if depth < max_depth:
        for cattr in ('children', 'sprites', '_children', 'qxjawxfuluj', 'khosdevozr'):
            if hasattr(obj, cattr):
                try:
                    c = getattr(obj, cattr)
                    if isinstance(c, (list, tuple)):
                        out[cattr] = [_walk_scene(x, max_depth, depth + 1, f'{path}/{cattr}[{i}]')
                                      for i, x in enumerate(c)]
                    elif isinstance(c, dict):
                        out[cattr] = {str(k): _walk_scene(v, max_depth, depth + 1, f'{path}/{cattr}[{k}]')
                                      for k, v in c.items()}
                except Exception as e:
                    out[cattr] = f'ERR: {e}'
    return out


def _repr_val(v, shallow=False):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, tuple):
        return tuple(_repr_val(x, shallow=True) for x in v)
    if isinstance(v, list):
        if shallow:
            return f'list[{len(v)}]'
        return [_repr_val(x, shallow=True) for x in v[:30]]
    if isinstance(v, dict):
        if shallow:
            return f'dict[{len(v)}]'
        return {str(k): _repr_val(val, shallow=True) for k, val in list(v.items())[:30]}
    if isinstance(v, np.ndarray):
        return {'__nd__': True, 'shape': v.shape, 'dtype': str(v.dtype),
                'hash': int(hash(v.tobytes()) & 0xFFFFFFFF)}
    if callable(v):
        return f'<callable {getattr(v, "__name__", "?")}>'
    try:
        return f'<{type(v).__name__}>'
    except Exception:
        return '<unrep>'


def diff_snapshots(a, b, path='', out=None):
    if out is None:
        out = []
    if type(a) != type(b):
        out.append((path, f'TYPE {type(a).__name__} vs {type(b).__name__}'))
        return out
    if isinstance(a, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys, key=str):
            if k not in a:
                out.append((f'{path}.{k}', f'MISSING_IN_A (b={_short(b[k])})'))
            elif k not in b:
                out.append((f'{path}.{k}', f'MISSING_IN_B (a={_short(a[k])})'))
            else:
                diff_snapshots(a[k], b[k], f'{path}.{k}', out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append((path, f'LEN {len(a)} vs {len(b)}'))
        for i, (x, y) in enumerate(zip(a, b)):
            diff_snapshots(x, y, f'{path}[{i}]', out)
    else:
        if a != b:
            out.append((path, f'{_short(a)} vs {_short(b)}'))
    return out


def _short(v):
    s = repr(v)
    return s if len(s) < 120 else s[:117] + '...'


def path_A_direct():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    game.set_level(6)  # 0-indexed -> L7
    return env, game, dump_full_state(game, 'A')


def path_B_sequential():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for level in range(6):  # solve L1..L6
        fd = solve_level(env, game, level)
        if fd is None or fd.levels_completed <= level:
            return env, game, {'__err__': f'failed at L{level+1}'}
    # Drain animation until eq is on level 7
    for _ in range(60):
        if game.ikhhdzfmarl.whtqurkphir == 7:
            break
        env.step(GameAction.ACTION1)
    return env, game, dump_full_state(game, 'B')


def path_C_partial_then_direct(num_to_solve):
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for level in range(num_to_solve):
        fd = solve_level(env, game, level)
        if fd is None or fd.levels_completed <= level:
            return env, game, {'__err__': f'failed at L{level+1}'}
    # Drain
    for _ in range(30):
        env.step(GameAction.ACTION1)
    game.set_level(6)
    return env, game, dump_full_state(game, f'C{num_to_solve}')


def run():
    print('=== Path A: fresh -> set_level(6) ===')
    _, _, snap_A = path_A_direct()
    print('=== Path B: sequential L1..L6 -> observe L7 ===')
    _, _, snap_B = path_B_sequential()

    # Compare A vs B (most important)
    print('\n--- DIFF A vs B ---')
    diffs_ab = diff_snapshots(snap_A, snap_B)
    if not diffs_ab:
        print('  IDENTICAL — no fields differ')
    else:
        for p, msg in diffs_ab[:400]:
            print(f'  {p}: {msg}')
        print(f'  total diffs: {len(diffs_ab)}')

    # Board-focused diff (this is what would determine solvability)
    print('\n--- BOARD-ONLY DIFF A vs B ---')
    board_diffs = [(p, m) for p, m in diffs_ab if p.startswith('.board')]
    if not board_diffs:
        print('  Board state IDENTICAL between A and B')
    else:
        for p, m in board_diffs:
            print(f'  {p}: {m}')

    # Save snapshots
    out_path = '/tmp/lf52_l7_snapshots.json'
    try:
        with open(out_path, 'w') as f:
            json.dump({'A': snap_A, 'B': snap_B,
                       'diffs_AB': [(p, m) for p, m in diffs_ab]},
                      f, default=str, indent=1)
        print(f'\nsnapshots saved to {out_path}')
    except Exception as e:
        print(f'save err: {e}')

    if diffs_ab and board_diffs:
        print('\n*** PATH DEPENDENCY DETECTED (board differs) ***')
        return 2
    if diffs_ab:
        print('\nNon-board fields differ (may still matter — inspect above)')
        return 1
    print('\nPath dependency NOT detected: A == B everywhere.')
    return 0


if __name__ == '__main__':
    sys.exit(run())
