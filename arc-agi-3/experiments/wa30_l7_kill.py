#!/usr/bin/env python3
"""L7 opening-book experiment. ONE run per invocation.

Usage: python3 wa30_l7_kill.py [mode]
  modes: baseline, rush, intercept, block
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction
from collections import deque

CELL = 4
UP, DOWN, LEFT, RIGHT, ACT5 = (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5
)
ACTION_NAMES = {UP:'U',DOWN:'D',LEFT:'L',RIGHT:'R',ACT5:'5'}
NAME2A = {'U':UP,'D':DOWN,'L':LEFT,'R':RIGHT,'5':ACT5}
DIR_LIST = [(0,-CELL),(0,CELL),(-CELL,0),(CELL,0)]

MODE = sys.argv[1] if len(sys.argv) > 1 else 'rush'

arcade = Arcade(operation_mode='offline')
env = arcade.make('wa30-ee6fef47')
fd = env.reset()
game = env._game

# Load cached L0-L6 solutions (fresh cache after we fixed writer)
CACHE = "/tmp/wa30_l0_l6_sols.py"
if os.path.exists(CACHE):
    ns = {}
    with open(CACHE) as f: exec(f.read(), ns)
    sols_str = ns['PRE_SOLS_STR']
    sols = [[NAME2A[c] for c in s] for s in sols_str]
    print(f"Loaded L0-L6 from cache: {[len(s) for s in sols]}")
else:
    print("No cache yet; run wa30_solve_final.py first to generate it, or regenerate.")
    # Regenerate via importing helpers
    sys.path.insert(0, 'arc-agi-3/experiments')
    import importlib.util
    spec = importlib.util.spec_from_file_location("wa30sf", "arc-agi-3/experiments/wa30_solve_final.py")
    mod = importlib.util.module_from_spec(spec)
    # Block main loop
    import builtins
    _real_open = builtins.open
    spec.loader.exec_module(mod)
    mod.env = env; mod.game = game; mod.arcade = arcade
    sols = []
    for lv in range(7):
        fd = mod.replay_from_solutions(sols)
        blues = game.current_level.get_sprites_by_tag("kdweefinfi")
        whites = game.current_level.get_sprites_by_tag("ysysltqlke")
        has_ai = len(blues)>0 or len(whites)>0
        if lv in mod.KNOWN_SOLUTIONS:
            sol = mod.string_to_actions(mod.KNOWN_SOLUTIONS[lv])
        elif not has_ai:
            sol = mod.solve_no_ai()
        else:
            sol = mod.solve_with_ai(sols)
        fd = mod.replay_from_solutions(sols)
        cb = fd.levels_completed
        actual = []
        for m in sol:
            fd = env.step(m); actual.append(m)
            if fd.levels_completed > cb: break
        sols.append(actual)
        print(f"L{lv}: {len(actual)} moves")
    A2N = {UP:'U',DOWN:'D',LEFT:'L',RIGHT:'R',ACT5:'5'}
    with open(CACHE,'w') as f:
        f.write("PRE_SOLS_STR = [\n")
        for s in sols:
            f.write("  " + repr(''.join(A2N[a] for a in s)) + ",\n")
        f.write("]\n")

# Replay to L7 start
fd = env.reset()
for s in sols:
    for m in s: fd = env.step(m)

player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
whites = game.current_level.get_sprites_by_tag("ysysltqlke")
print(f"L7 start: P=({player.x},{player.y}) whites={[(w.x,w.y) for w in whites]}")

# ---------- helpers ----------
def bfs_reach(start, targets, obs):
    if start in targets: return [start]
    vis = {start}; q = deque([(start,[start])])
    while q:
        (cx,cy), path = q.popleft()
        for dx,dy in DIR_LIST:
            nx,ny = cx+dx, cy+dy
            if (nx,ny) in vis or (nx,ny) in obs: continue
            if nx<0 or nx>=64 or ny<0 or ny>=64: continue
            vis.add((nx,ny))
            np = path + [(nx,ny)]
            if (nx,ny) in targets: return np
            q.append(((nx,ny), np))
    return None

def snap():
    p = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    ws = game.current_level.get_sprites_by_tag("ysysltqlke")
    bs = game.current_level.get_sprites_by_tag("kdweefinfi")
    gs = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots = game.wyzquhjerd
    placed = sum(1 for g in gs if (g.x,g.y) in slots and g not in game.zmqreragji)
    return p, ws, bs, gs, placed, slots

def facing_of(p):
    rot = p.rotation
    if rot==0: return (p.x, p.y-CELL)
    if rot==180: return (p.x, p.y+CELL)
    if rot==90: return (p.x+CELL, p.y)
    return (p.x-CELL, p.y)

def face_and_kill(target):
    """If adjacent and facing target, ACT5; else turn-face."""
    p,_,_,_,_,_ = snap()
    dx,dy = target[0]-p.x, target[1]-p.y
    if dy < 0: return UP
    if dy > 0: return DOWN
    if dx > 0: return RIGHT
    if dx < 0: return LEFT
    return ACT5

# Load smart_action from solver
import importlib.util
spec = importlib.util.spec_from_file_location("wa30sf", "arc-agi-3/experiments/wa30_solve_final.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.env = env; mod.game = game; mod.arcade = arcade

def smart():
    p, ws, bs, gs, _, slots = snap()
    coll = set(game.pkbufziase)
    blocked = set(game.qthdiggudy)
    slot_aligned = set((sx,sy) for sx,sy in slots if sx%CELL==0 and sy%CELL==0)
    slots2 = game.lqctaojiby
    return mod.smart_action_l7(p, gs, ws, bs, coll, blocked, slots, slots2, slot_aligned)

def run(policy, max_steps, label='', verbose=False):
    actions = []
    prev_placed = 0
    for step in range(max_steps):
        p, ws, bs, gs, placed, slots = snap()
        if placed != prev_placed and verbose:
            print(f"  [{label}] step {step}: placed {prev_placed}->{placed}")
            prev_placed = placed
        a = policy(step)
        if a is None:
            a = smart()
        fd = env.step(a); actions.append(a)
        if fd.levels_completed > 7 or fd.state.name=='WIN':
            print(f"  [{label}] WIN at step {step+1} !!! total={len(actions)}")
            return actions, True
        if fd.state.name in ('LOSE','GAME_OVER'):
            p, ws, bs, gs, placed, slots = snap()
            print(f"  [{label}] LOSE at step {step}, placed={placed}/13")
            print(f"    P=({p.x},{p.y}) whites={[(w.x,w.y) for w in ws]}")
            for g in gs:
                pl = 'PLACED' if (g.x,g.y) in slots and g not in game.zmqreragji else 'free'
                c = game.zmqreragji.get(g)
                ct = '-'
                if c:
                    if 'kdweefinfi' in c.tags: ct='blue@'+str((c.x,c.y))
                    elif 'ysysltqlke' in c.tags: ct='WHITE@'+str((c.x,c.y))
                    elif 'wbmdvjhthc' in c.tags: ct='player'
                print(f"     piece({g.x},{g.y}) {pl} by={ct}")
            return actions, False
    p, ws, bs, gs, placed, slots = snap()
    print(f"  [{label}] exit, placed={placed}/13, steps_left={game.kuncbnslnm.current_steps}")
    return actions, False

# ---------- modes ----------
def pursue_white_policy_v2(target_spawn):
    """Smarter chase:
       - If adjacent: face+kill
       - If manhattan==8 diagonal (1 step horizontal + 1 step vertical): step into
         the cell that reduces BOTH coords. Actually when the white is also
         advancing toward a piece, we may need to STALL to force alignment.
       - Otherwise BFS to the white's ADJACENT cells.
       - Special trick: if white is moving along a row/col and player is on
         same row/col, step toward white — direct collision."""
    def policy(step):
        p, ws, bs, gs, placed, slots = snap()
        if not ws: return smart()
        tw = min(ws, key=lambda w: abs(w.x-target_spawn[0])+abs(w.y-target_spawn[1]))
        coll = set(game.pkbufziase)
        obs = coll - {(p.x,p.y)}
        mh = abs(p.x-tw.x)+abs(p.y-tw.y)
        if mh==CELL:
            fx,fy = facing_of(p)
            if (fx,fy)==(tw.x,tw.y): return ACT5
            dx,dy = tw.x-p.x, tw.y-p.y
            if dy<0: return UP
            if dy>0: return DOWN
            if dx>0: return RIGHT
            return LEFT
        # Same row or col? Walk directly toward
        if p.y==tw.y:
            if tw.x>p.x:
                nx = p.x+CELL
                if (nx,p.y) not in coll: return RIGHT
            else:
                nx = p.x-CELL
                if (nx,p.y) not in coll: return LEFT
        if p.x==tw.x:
            if tw.y>p.y:
                ny=p.y+CELL
                if (p.x,ny) not in coll: return DOWN
            else:
                ny=p.y-CELL
                if (p.x,ny) not in coll: return UP
        # BFS fallback
        adj = set()
        for dx,dy in DIR_LIST:
            c = (tw.x+dx, tw.y+dy)
            if 0<=c[0]<64 and 0<=c[1]<64 and c not in coll:
                adj.add(c)
        if not adj: return smart()
        path = bfs_reach((p.x,p.y), adj, obs)
        if path is None or len(path)<2: return smart()
        nx,ny = path[1]
        if ny<p.y: return UP
        if ny>p.y: return DOWN
        if nx>p.x: return RIGHT
        return LEFT
    return policy

def pursue_white_policy(target_spawn, intercept=False):
    """Chase the white whose current position is nearest to target_spawn.
    If adjacent, face + ACT5. If intercept=True, use 1-step lookahead to predict
    where white will go and path to a cell adjacent to that."""
    state = {'prev_w': None}
    def policy(step):
        p, ws, bs, gs, placed, slots = snap()
        if not ws:
            return smart()
        tw = min(ws, key=lambda w: abs(w.x-target_spawn[0])+abs(w.y-target_spawn[1]))
        # Only target the "correct" white if still near its strip
        # Compute obs
        coll = set(game.pkbufziase)
        obs = coll - {(p.x,p.y)}
        # Adjacent?
        if abs(p.x-tw.x)+abs(p.y-tw.y)==CELL:
            fx,fy = facing_of(p)
            if (fx,fy)==(tw.x,tw.y): return ACT5
            # Face
            dx,dy = tw.x-p.x, tw.y-p.y
            if dy<0: return UP
            if dy>0: return DOWN
            if dx>0: return RIGHT
            return LEFT
        # BFS to cells ADJACENT to white
        adj = set()
        for dx,dy in DIR_LIST:
            c = (tw.x+dx, tw.y+dy)
            if 0<=c[0]<64 and 0<=c[1]<64 and c not in coll:
                adj.add(c)
        if not adj: return smart()
        path = bfs_reach((p.x,p.y), adj, obs)
        if path is None or len(path)<2: return smart()
        nx,ny = path[1]
        if ny<p.y: return UP
        if ny>p.y: return DOWN
        if nx>p.x: return RIGHT
        return LEFT
    return policy

if MODE == 'baseline':
    run(lambda step: smart(), 200, 'BASE', verbose=True)

elif MODE == 'rush':
    # Phase 1: rush upper white (25 moves max)
    # Phase 2: rush lower white (30 moves max)
    # Phase 3: smart cleanup
    state = {'phase': 1, 'ph1_steps': 0, 'ph2_steps': 0}
    pol_upper = pursue_white_policy((32,16))
    pol_lower = pursue_white_policy((32,56))

    def combined(step):
        p, ws, bs, gs, placed, slots = snap()
        # Check if upper white (near y<24) is dead
        upper_alive = any(w.y < 24 for w in ws)
        lower_alive = any(w.y > 36 for w in ws)
        if state['phase']==1:
            if not upper_alive:
                state['phase']=2
                print(f"  [rush] upper dead at step {step}, switching to lower")
            elif state['ph1_steps'] >= 30:
                state['phase']=2
                print(f"  [rush] upper not killed in 30 steps, moving on")
            else:
                state['ph1_steps'] += 1
                return pol_upper(step)
        if state['phase']==2:
            if not lower_alive:
                state['phase']=3
                print(f"  [rush] lower dead at step {step}")
            elif state['ph2_steps'] >= 35:
                state['phase']=3
                print(f"  [rush] lower not killed in 35 steps, moving on")
            else:
                state['ph2_steps'] += 1
                return pol_lower(step)
        return smart()
    run(combined, 150, 'rush', verbose=True)

elif MODE == 'idle':
    def idle_pol(step):
        return UP if step%2==0 else DOWN
    run(idle_pol, 150, 'idle', verbose=True)

elif MODE == 'lower_only':
    # Kill lower white; then smart only
    state = {'phase': 1}
    pol_lower = pursue_white_policy((32,56))
    def combined(step):
        p, ws, bs, gs, placed, slots = snap()
        lower_alive = any(w.y>36 for w in ws)
        if state['phase']==1:
            if not lower_alive:
                state['phase']=2
                print(f"  [lo] lower dead step {step}")
            else:
                return pol_lower(step)
        return smart()
    run(combined, 150, 'lower_only', verbose=True)

elif MODE == 'lower_first':
    # Lower white is closer to player, only 18 moves to kill. Kill lower first.
    UPPER_CAP = int(os.environ.get('UPPER_CAP', '999'))
    state = {'phase': 1, 'ph2_steps': 0}
    pol_lower = pursue_white_policy_v2((32,56))
    pol_upper = pursue_white_policy_v2((32,16))
    def combined(step):
        p, ws, bs, gs, placed, slots = snap()
        lower_alive = any(w.y>36 for w in ws)
        upper_alive = any(w.y<24 for w in ws)
        if state['phase']==1:
            if not lower_alive:
                state['phase']=2
                print(f"  [lf] lower dead step {step}")
            else:
                return pol_lower(step)
        if state['phase']==2:
            if not upper_alive:
                state['phase']=3
                print(f"  [lf] upper dead step {step}")
            elif state['ph2_steps'] >= UPPER_CAP:
                state['phase']=3
                print(f"  [lf] upper cap at step {step}")
            else:
                state['ph2_steps'] += 1
                return pol_upper(step)
        return smart()
    acts, won = run(combined, 150, 'lower_first', verbose=True)
    if won:
        s = ''.join(ACTION_NAMES[a] for a in acts)
        print(f"L7_SOLUTION({len(acts)}): {s}")
        with open('/tmp/wa30_l7_sol.txt','w') as f: f.write(s)

else:
    print(f"Unknown mode: {MODE}")
