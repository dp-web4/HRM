#!/usr/bin/env python3
"""
dc22 L5 probe: load L5 fresh (replay L1-L4 from cache), then systematically
probe the state space looking for a click combination that makes the goal
reachable via pure walking.

Strategy:
- Enumerate click targets (crane + jpug buttons)
- BFS over click sequences up to some depth
- For each resulting state, compute player_reachable_cells
- If goal is reachable, print the click sequence + reach info
- Additionally dump per-state diagnostics: which sprites changed, which tiles
  gained/lost walkability
"""
import os, sys, json, time, copy
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
from collections import deque

sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import (
    save_game_state, restore_game_state, player_reachable_cells,
    reconstruct_moves, find_click_targets, save_frame,
)

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

def replay_cached(env, cache_path, up_to_level):
    with open(cache_path) as f:
        raw = json.load(f)
    act_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
               3: GameAction.ACTION3, 4: GameAction.ACTION4,
               6: GameAction.ACTION6}
    for lvl_idx in range(up_to_level):
        for m in raw[lvl_idx]:
            a = act_map[m['action']]
            if 'data' in m:
                env.step(a, data=m['data'])
            else:
                env.step(a)

def describe_state(game):
    """Dump sprite positions + interaction for wbze/jpug/zbhi/itki/kbqq/vckz/qgdz."""
    out = []
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED:
            continue
        tags_of_interest = {'wbze','jpug','zbhi','itki','kbqq','bynyvtuepbt-object','pressure-plate'}
        if any(t in s.tags for t in tags_of_interest) or any(k in s.name for k in ('vckz','qgdz','kbqq','wbze','merged','qeqe')):
            out.append((s.name, s.x, s.y, s.width, s.height, s.interaction.name, list(s.tags)))
    return out

def board_sig(game):
    """Signature of the board for dedup (no player pos)."""
    out = []
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED:
            continue
        if any(t in s.tags for t in ('wbze','jpug','zbhi','itki','bynyvtuepbt-object')):
            out.append((s.name, s.x, s.y, s.interaction.value, s.is_visible))
    nxhz = (game.nxhz_x, game.nxhz_y, game.nxhz_attached_kind,
            game.attached_hhxv_prefix, game.attached_hhxv_x, game.attached_hhxv_y)
    return (tuple(sorted(out)), nxhz)

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    obs = env.reset()
    cache_path = f"{VIS}/solutions.json"
    print("Replaying L1-L4 from cache...")
    replay_cached(env, cache_path, 4)
    game = env._game
    print(f"Now on L5: levels_completed={env.observation_space.levels_completed}")

    p = game.fdvakicpimr
    print(f"Player: ({p.x},{p.y})  Goal: ({game.bqxa.x},{game.bqxa.y})")
    print(f"Steps: {game.step_counter_ui.rdnpeqedga}")

    # Dump click targets
    click_targets = find_click_targets(game)
    print(f"\n{len(click_targets)} click targets:")
    for kind, name, cx, cy in click_targets:
        # Find sprite and look up any single-letter tags
        letter = '?'
        for s in game.current_level.get_sprites():
            if s.name == name:
                for t in s.tags:
                    if len(t) == 1:
                        letter = t
                        break
                break
        print(f"  {kind}: {name} at ({cx},{cy}) letter={letter}")

    # Dump interesting sprites
    print("\nSprites of interest:")
    sig = describe_state(game)
    for item in sig:
        print(f"  {item}")

    # Initial reachability
    initial_state = save_game_state(game)
    parents0 = player_reachable_cells(game)
    goal = (game.bqxa.x, game.bqxa.y)
    print(f"\nInitial reach: {len(parents0)} cells, goal reachable: {goal in parents0}")
    print(f"  Reach cells: {sorted(parents0.keys())[:30]}...")

    # Enumerate click-only BFS up to depth N
    MAX_DEPTH = 6
    click_actions = [(GameAction.ACTION6, {'x': cx, 'y': cy}, name, kind)
                     for kind, name, cx, cy in click_targets]
    print(f"\nBFS click-only up to depth {MAX_DEPTH}, {len(click_actions)} actions each node")

    visited = {board_sig(game)}
    frontier = deque()
    frontier.append(([], initial_state))
    best_reach = len(parents0)
    t0 = time.time()
    winners = []
    nodes = 0

    while frontier:
        seq, st = frontier.popleft()
        nodes += 1
        if len(seq) >= MAX_DEPTH:
            continue
        for (act, data, name, kind) in click_actions:
            restore_game_state(game, st)
            fd = env.step(act, data=data)
            if fd.state.name == 'LOSE':
                continue
            if fd.levels_completed > 4:
                print(f"*** LEVEL COMPLETED BY CLICK SEQ {seq + [(name, kind)]}")
                winners.append(('win', seq + [(name, kind)]))
                return
            bk = board_sig(game)
            if bk in visited:
                continue
            visited.add(bk)

            ns = save_game_state(game)
            # Check reach
            parents = player_reachable_cells(game)
            if goal in parents:
                print(f"!!! GOAL REACHABLE after clicks {[(n,k) for (_,_,n,k) in [(None,None,n,k) for n,k in [(name,kind)]]]}")
                print(f"    Full seq: {seq + [(name, kind)]}")
                winners.append(('reachable', seq + [(name, kind)], ns))
                # Don't return — collect all
            if len(parents) > best_reach:
                best_reach = len(parents)
                print(f"  depth={len(seq)+1} {name}({kind}) -> reach={len(parents)} (new best)")
            frontier.append((seq + [(name, kind)], ns))
        if time.time() - t0 > 120:
            print(f"Time cap hit at nodes={nodes}, visited={len(visited)}")
            break

    print(f"\nDone. nodes={nodes}, visited={len(visited)}, winners={len(winners)}, best_reach={best_reach}, {time.time()-t0:.1f}s")
    for w in winners[:5]:
        print(w[:2])

    # SECOND PHASE: dump reach for each "interesting" state
    print("\n\n=== Phase 2: dump reach after each click path that grows reach ===")
    restore_game_state(game, initial_state)
    # click b once
    env.step(GameAction.ACTION6, data={'x': 52, 'y': 41})  # jpug-jbyz b
    parents = player_reachable_cells(game)
    print(f"After b-click: reach={len(parents)}: {sorted(parents.keys())}")
    # Can we walk anywhere new? Check kbqq positions
    kbqq_cells = [(30,34),(30,35),(30,36),(30,37),(31,34),(31,35),(31,36),(31,37),
                  (32,34),(32,35),(32,36),(32,37),(33,34),(33,35),(33,36),(33,37),
                  (26,34),(26,35),(26,36),(26,37),(27,34),(27,35),(27,36),(27,37),
                  (28,34),(28,35),(28,36),(28,37),(29,34),(29,35),(29,36),(29,37)]
    for c in kbqq_cells:
        if c in parents:
            print(f"  reachable kbqq-ish: {c}")

    # Dump nxhz (crane) state
    print(f"\nCrane: nxhz_x={game.nxhz_x}, nxhz_y={game.nxhz_y}, attached={game.nxhz_attached_kind}")
    print(f"       sprite-10 pos: ", end='')
    for s in game.current_level.get_sprites():
        if s.name == 'sprite-10':
            print(f"({s.x},{s.y}) interaction={s.interaction.name}")
            break

    # Click crane buttons to see what moves crane
    restore_game_state(game, initial_state)
    env.step(GameAction.ACTION6, data={'x': 52, 'y': 41})  # b
    print(f"\nAfter b click, try each crane button and see if reach grows:")
    b_state = save_game_state(game)
    for (act, data, name, kind) in click_actions:
        if kind != 'sys_click':
            continue
        restore_game_state(game, b_state)
        env.step(act, data=data)
        p2 = player_reachable_cells(game)
        print(f"  {name}: reach={len(p2)}, crane={game.nxhz_x},{game.nxhz_y}")

if __name__ == "__main__":
    main()
