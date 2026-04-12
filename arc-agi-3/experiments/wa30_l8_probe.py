#!/usr/bin/env python3
"""L8 probe + experiments."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from collections import deque

CELL = 4
UP, DOWN, LEFT, RIGHT, ACT5 = (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5
)
ACTION_NAMES = {UP:'U',DOWN:'D',LEFT:'L',RIGHT:'R',ACT5:'5'}
NAME2A = {v:k for k,v in ACTION_NAMES.items()}
DIR_LIST = [(0,-CELL),(0,CELL),(-CELL,0),(CELL,0)]
MODE = sys.argv[1] if len(sys.argv)>1 else 'probe'

import importlib.util
spec = importlib.util.spec_from_file_location("wa30sf", "arc-agi-3/experiments/wa30_solve_final.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

arcade = Arcade(operation_mode='offline')
env = arcade.make('wa30-ee6fef47')
game = env._game
mod.env = env; mod.game = game; mod.arcade = arcade
fd = env.reset()

# Solve L0-L7 using baked solver logic
sols = []
for lv in range(8):
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
    print(f"L{lv}: {len(actual)} moves -> completed={fd.levels_completed}")

# Now on L8
p = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
ws = game.current_level.get_sprites_by_tag("ysysltqlke")
bs = game.current_level.get_sprites_by_tag("kdweefinfi")
gs = game.current_level.get_sprites_by_tag("geezpjgiyd")
print(f"\nL8: P=({p.x},{p.y}) rot={p.rotation}")
print(f"Pieces ({len(gs)}): {sorted((g.x,g.y) for g in gs)}")
print(f"Blues: {[(b.x,b.y) for b in bs]}")
print(f"Whites: {[(w.x,w.y) for w in ws]}")
print(f"Steps: {game.kuncbnslnm.current_steps}")

coll = set(game.pkbufziase)
print("Collidable map:")
for y in range(0,64,4):
    row = ''
    for x in range(0,64,4):
        c = (x,y)
        if c==(p.x,p.y): row+='P'
        elif any(c==(w.x,w.y) for w in ws): row+='W'
        elif any(c==(b.x,b.y) for b in bs): row+='B'
        elif any(c==(g.x,g.y) for g in gs): row+='g'
        elif c in coll: row+='#'
        else: row+='.'
    print(f"  y={y:2d} {row}")

slots = game.wyzquhjerd
sa = set((sx,sy) for sx,sy in slots if sx%CELL==0 and sy%CELL==0)
print(f"Slot aligned cells: {sorted(sa)}")

# ---------- policies ----------
def bfs_reach(start, targets, obs):
    if start in targets: return [start]
    vis = {start}; q = deque([(start,[start])])
    while q:
        (cx,cy),path = q.popleft()
        for dx,dy in DIR_LIST:
            nx,ny = cx+dx, cy+dy
            if (nx,ny) in vis or (nx,ny) in obs: continue
            if nx<0 or nx>=64 or ny<0 or ny>=64: continue
            vis.add((nx,ny))
            np = path+[(nx,ny)]
            if (nx,ny) in targets: return np
            q.append(((nx,ny),np))
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
    if p.rotation==0: return (p.x, p.y-CELL)
    if p.rotation==180: return (p.x, p.y+CELL)
    if p.rotation==90: return (p.x+CELL, p.y)
    return (p.x-CELL, p.y)

def smart():
    p, ws, bs, gs, _, slots = snap()
    coll = set(game.pkbufziase)
    blocked = set(game.qthdiggudy)
    slot_aligned = set((sx,sy) for sx,sy in slots if sx%CELL==0 and sy%CELL==0)
    return mod.smart_action_l7(p, gs, ws, bs, coll, blocked, slots, game.lqctaojiby, slot_aligned)

def pursue_white(tw_spawn):
    def pol(step):
        p, ws, bs, gs, _, _ = snap()
        if not ws: return smart()
        tw = min(ws, key=lambda w: abs(w.x-tw_spawn[0])+abs(w.y-tw_spawn[1]))
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
    return pol

def run(policy, max_steps, label):
    acts = []
    last_p = 0
    for step in range(max_steps):
        p, ws, bs, gs, placed, slots = snap()
        if placed != last_p:
            print(f"  [{label}] step{step}: placed {last_p}->{placed}")
            last_p = placed
        a = policy(step)
        if a is None: a = smart()
        fd = env.step(a); acts.append(a)
        if fd.levels_completed > 8 or fd.state.name=='WIN':
            print(f"  [{label}] WIN step {step+1}, total={len(acts)}")
            return acts, True
        if fd.state.name in ('LOSE','GAME_OVER'):
            print(f"  [{label}] LOSE step {step}, placed={placed}")
            for g in gs:
                pl = 'P' if (g.x,g.y) in slots and g not in game.zmqreragji else 'f'
                c = game.zmqreragji.get(g); ct='-'
                if c:
                    if 'kdweefinfi' in c.tags: ct='blue'
                    elif 'ysysltqlke' in c.tags: ct='WH'
                    elif 'wbmdvjhthc' in c.tags: ct='me'
                print(f"    {pl} piece({g.x},{g.y}) by={ct}")
            return acts, False
    return acts, False

def capture():
    """Serialize engine state sufficiently to restart L8. Since env.reset isn't
    clean, we just record from-L0 replay prefix."""
    return None

if MODE == 'probe':
    pass
elif MODE == 'prime_l8':
    # Try hand-crafted primes that direct player toward upper-left pieces
    primes_to_try = [
        '',
        'LLLLU',       # head left
        'LLLLUUU',     # head upper-left
        'LLLLLLLL',
        'LLLLUULL',
        'UUUU',        # head up first
        'UULLLLLL',
        'UUUULLLLL',
        'LLLLLLLLUUUU',
        'LLLL',
        'LL',
    ]
    for pi, ps in enumerate(primes_to_try):
        arcade2 = Arcade(operation_mode='offline')
        env = arcade2.make('wa30-ee6fef47')
        game = env._game
        mod.env = env; mod.game = game; mod.arcade = arcade2
        fd = env.reset()
        for s in sols:
            for m in s: fd = env.step(m)
        acts = []
        for c in ps:
            a = NAME2A[c]
            fd = env.step(a); acts.append(a)
            if fd.state.name in ('LOSE','GAME_OVER') or fd.levels_completed>8: break
        won = False
        while fd.state.name not in ('LOSE','GAME_OVER') and fd.levels_completed<=8:
            a = smart()
            fd = env.step(a); acts.append(a)
            if fd.levels_completed > 8 or fd.state.name=='WIN':
                won = True; break
        p,ws,bs,gs,placed,slots = snap()
        print(f"  prime[{pi}]'{ps}': placed={placed}/9 won={won}")
        if won:
            s = ''.join(ACTION_NAMES[a] for a in acts)
            print(f"L8_SOLUTION({len(acts)}): {s}")
            with open('/tmp/wa30_l8_sol.txt','w') as f: f.write(s)
            break

elif MODE == 'biased_search':
    # Run smart with occasional random deviation
    import random
    best = 0
    for seed in range(300):
        random.seed(seed)
        arcade2 = Arcade(operation_mode='offline')
        env = arcade2.make('wa30-ee6fef47')
        game = env._game
        mod.env = env; mod.game = game; mod.arcade = arcade2
        fd = env.reset()
        for s in sols:
            for m in s: fd = env.step(m)
        acts = []
        rand_rate = 0.05 + 0.25*random.random()
        while fd.state.name not in ('LOSE','GAME_OVER') and fd.levels_completed<=8:
            if random.random() < rand_rate:
                a = random.choice([UP,DOWN,LEFT,RIGHT,ACT5])
            else:
                a = smart()
            fd = env.step(a); acts.append(a)
            if fd.levels_completed>8 or fd.state.name=='WIN': break
        if fd.levels_completed>8 or fd.state.name=='WIN':
            s = ''.join(ACTION_NAMES[a] for a in acts)
            print(f"WIN seed={seed} len={len(acts)}")
            print(f"L8_SOLUTION({len(acts)}): {s}")
            with open('/tmp/wa30_l8_sol.txt','w') as f: f.write(s)
            break
        p,ws,bs,gs,placed,slots = snap()
        if placed > best:
            best = placed; print(f"seed {seed}: new best={placed} rand={rand_rate:.2f}")
        if seed%25==0:
            print(f"seed {seed}: {placed} best={best}")
    print(f"Biased search best: {best}")

elif MODE == 'random_primes':
    import random
    random.seed(0)
    base = sols[:]
    best = 0
    for attempt in range(120):
        # Fresh env each attempt
        arcade2 = Arcade(operation_mode='offline')
        env = arcade2.make('wa30-ee6fef47')
        game = env._game
        mod.env = env; mod.game = game; mod.arcade = arcade2
        fd = env.reset()
        for s in base:
            for m in s: fd = env.step(m)
        # Random prime of 0-5 moves, then smart
        prime_len = random.randint(0, 10)
        # Most primes = no-op oscillation (delay); occasionally real random
        if random.random() < 0.5:
            prime = [UP if i%2==0 else DOWN for i in range(prime_len)]
        else:
            prime = [random.choice([UP,DOWN,LEFT,RIGHT]) for _ in range(prime_len)]
        acts = list(prime)
        for m in prime:
            fd = env.step(m)
            if fd.state.name in ('LOSE','GAME_OVER') or fd.levels_completed>8:
                break
        won = False
        while fd.state.name not in ('LOSE','GAME_OVER') and fd.levels_completed <= 8:
            a = smart()
            fd = env.step(a); acts.append(a)
            if fd.levels_completed > 8 or fd.state.name=='WIN':
                won = True; break
        p,ws,bs,gs,placed,slots = snap()
        if won:
            print(f"  attempt {attempt}: WIN! prime_len={prime_len}, len={len(acts)}")
            s = ''.join(ACTION_NAMES[a] for a in acts)
            print(f"L8_SOLUTION({len(acts)}): {s}")
            with open('/tmp/wa30_l8_sol.txt','w') as f: f.write(s)
            sys.exit(0)
        if placed > best:
            best = placed
        if attempt%10==0:
            print(f"  attempt {attempt}: this={placed} best={best}")
    print(f"best={best}/9")

elif MODE == 'baseline':
    run(lambda s: smart(), 70, 'BASE')
elif MODE == 'kill_white':
    state = {'phase':1}
    pw = pursue_white((60,56))
    def pol(step):
        p,ws,bs,gs,_,_ = snap()
        if not ws:
            if state['phase']==1:
                state['phase']=2
                print(f"  [kw] white dead step {step}")
            return smart()
        return pw(step)
    acts, won = run(pol, 70, 'kw')
    if won:
        s = ''.join(ACTION_NAMES[a] for a in acts)
        print(f"L8_SOLUTION({len(acts)}): {s}")
        with open('/tmp/wa30_l8_sol.txt','w') as f: f.write(s)
