#!/usr/bin/env python3
"""
lf52 L7 CONTEXT probe — question the frame.

Four previous Opus agents converged on "L7 structurally unsolvable under
verified model". They exhaustively verified the model is correct under the
assumption that "the win path is cfilhtifcb counter decrement to 2".

This probe questions the *frame* rather than the work within it.

HYPOTHESES being tested:
  H-A: There is an object in the L7 scene graph whose name matches the counter
       prefix ("fozwvlovdui") but is NOT one of the three pieces. If so,
       removing it (or something causing `ndtvadsrqf("fozwvlovdui")` to drop)
       could trigger the win check.
  H-B: The three "pieces" we see are not the only things the counter counts.
       Cursor sprite `csrvckunbev` has name="fozwvlovdui" (per source), and
       igcydyfanuk highlight sprites too — are any of them in `kdsncymzyeb`
       of the grid?
  H-C: There's an off-grid object at (negative or >29) coordinates we never
       scan. Scan a wider range.
  H-D: A right-N that we THINK is at (22,6) might actually be elsewhere.
       Print ALL `ndtvadsrqf("fozwvlovdui")` objects with their grid coords.
  H-E: Some action path invokes pchvqimdvj() which then DOES reach win().
       Trace what pchvqimdvj leads to from any entry point.
  H-F: Re-read `self.kdsncymzyeb` directly (the grid's child list) and
       compare to ijpoqzvnjt(x,y) scan.
  H-G: `khosdevozr` (highlight sprite dict) objects — are any added to grid?

APPROACH: Advance to L7, then systematically enumerate EVERYTHING in the scene
graph, find anomalies, and print a verdict.
"""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction


def advance_to_l7(env, game):
    """Use existing solver to reach L7."""
    import lf52_solve_final as solver
    for lvl in range(6):
        fd = solver.solve_level(env, game, lvl)
        if fd is None or fd.levels_completed <= lvl:
            print(f"FAIL reaching L7 at L{lvl+1}")
            return False
    # Force refresh of eq reference
    _ = env.observation_space.frame
    return True


def dump_grid_objects(eq, title):
    """Dump every child in the grid's kdsncymzyeb list."""
    grid = eq.hncnfaqaddg
    print(f"\n=== {title} ===")
    print(f"grid type={type(grid).__name__} name={getattr(grid, 'name', '?')}")
    print(f"grid offset={grid.cdpcbbnfdp}")
    # Dive into kdsncymzyeb (children)
    children = getattr(grid, 'kdsncymzyeb', None)
    if children is None:
        print("  NO kdsncymzyeb attr on grid")
    else:
        print(f"  kdsncymzyeb: {len(children)} children")
        by_name = {}
        for c in children:
            name = getattr(c, 'name', repr(c))
            by_name.setdefault(name, []).append(c)
        for name in sorted(by_name):
            objs = by_name[name]
            # Print position of first 3
            positions = []
            for o in objs[:3]:
                gp = getattr(o, 'chahdtpdoz', None)
                pp = getattr(o, 'cdpcbbnfdp', None)
                positions.append(f"grid={gp} pixel={pp}")
            print(f"    {name}: {len(objs)}  | " + "  ;  ".join(positions))


def dump_ndtvadsrqf(eq):
    """What does the win counter see?"""
    grid = eq.hncnfaqaddg
    print(f"\n=== ndtvadsrqf('fozwvlovdui') probe ===")
    hits = grid.ndtvadsrqf("fozwvlovdui")
    print(f"  count={len(hits)}")
    for o in hits:
        gp = getattr(o, 'chahdtpdoz', None)
        pp = getattr(o, 'cdpcbbnfdp', None)
        print(f"    name={o.name!r}  yrxvacxlgrf={getattr(o,'yrxvacxlgrf','?')!r}  grid={gp} pixel={pp}")

    # Now also try each exact-name variant via whdmasyorl
    for name in ("fozwvlovdui", "fozwvlovdui_red", "fozwvlovdui_blue", "csrvckunbev"):
        hits = grid.whdmasyorl(name)
        print(f"  whdmasyorl({name!r}): {len(hits)}")


def wide_scan(eq):
    """ijpoqzvnjt over a wide grid range, to catch off-grid objects."""
    print(f"\n=== wide cell scan (-5..35 x -5..35) ===")
    grid = eq.hncnfaqaddg
    nontrivial = []
    for y in range(-5, 35):
        for x in range(-5, 35):
            objs = grid.ijpoqzvnjt(x, y)
            if objs:
                names = tuple(o.name for o in objs)
                if any("fozwvlovdui" in n for n in names):
                    nontrivial.append(((x, y), names))
    for cell, names in nontrivial:
        print(f"  {cell}: {names}")
    return nontrivial


def deep_walk_scene(eq, max_depth=4):
    """Walk the entire scene graph under eq, collect every object with
    a yrxvacxlgrf/name attribute that contains 'fozwvlovdui'."""
    print(f"\n=== deep scene walk for fozwvlovdui-named objects ===")
    seen_ids = set()
    found = []

    def visit(obj, depth, path):
        if id(obj) in seen_ids or depth > max_depth:
            return
        seen_ids.add(id(obj))
        name = getattr(obj, 'yrxvacxlgrf', None) or getattr(obj, 'name', None)
        if isinstance(name, str) and 'fozwvlovdui' in name:
            gp = getattr(obj, 'chahdtpdoz', None)
            pp = getattr(obj, 'cdpcbbnfdp', None)
            found.append((path, name, gp, pp, type(obj).__name__))
        # walk children
        for attr in ('kdsncymzyeb', 'children', '_children'):
            v = getattr(obj, attr, None)
            if isinstance(v, (list, tuple)):
                for i, c in enumerate(v):
                    visit(c, depth + 1, f"{path}.{attr}[{i}]")

    visit(eq, 0, "eq")
    for path, name, gp, pp, t in found:
        print(f"  {path}: {t} name={name!r} grid={gp} pixel={pp}")
    return found


def check_khosdevozr(eq):
    print(f"\n=== eq.khosdevozr (highlight sprite cache) ===")
    d = getattr(eq, 'khosdevozr', None)
    if d is None:
        print("  missing")
        return
    for k, v in d.items():
        name = getattr(v, 'name', '?')
        parent = getattr(v, 'fcokslsroqg', None)
        parent_name = getattr(parent, 'name', None) if parent else None
        print(f"  {k} -> name={name!r} parent={parent_name!r}")


def check_wpwvsglgmb(eq):
    print(f"\n=== eq.wpwvsglgmb (cursor) ===")
    c = getattr(eq, 'wpwvsglgmb', None)
    if c is None:
        print("  missing")
        return
    name = getattr(c, 'name', '?')
    parent = getattr(c, 'fcokslsroqg', None)
    parent_name = getattr(parent, 'name', None) if parent else None
    qoi = getattr(c, 'qoifrofmiu', None)
    qoi_name = getattr(qoi, 'name', None) if qoi else None
    qoi_grid = getattr(qoi, 'chahdtpdoz', None) if qoi else None
    print(f"  name={name!r}  parent={parent_name!r}  selected={qoi_name!r} at {qoi_grid}")


def try_naive_win_probe(env, game, eq):
    """Concrete hypothesis probe: if we just push UP a bunch, does anything
    happen? Record the counter each step."""
    print(f"\n=== counter over action sequence ===")
    for i, (a, name) in enumerate([
        (GameAction.ACTION1, "UP"),
        (GameAction.ACTION1, "UP"),
        (GameAction.ACTION1, "UP"),
        (GameAction.ACTION2, "DOWN"),
        (GameAction.ACTION3, "LEFT"),
        (GameAction.ACTION4, "RIGHT"),
        (GameAction.ACTION5, "A5"),
    ]):
        fd = env.step(a)
        grid = eq.hncnfaqaddg
        cnt = len(grid.ndtvadsrqf("fozwvlovdui"))
        print(f"  step {i} {name}: count={cnt} won={eq.iajuzrgttrv} lost={eq.evxflhofing} state={fd.state.name}")


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game

    if not advance_to_l7(env, game):
        return
    eq = game.ikhhdzfmarl
    print(f"\nReached level {eq.whtqurkphir}")
    if eq.whtqurkphir != 7:
        print("Not L7, abort")
        return

    dump_grid_objects(eq, "L7 fresh")
    dump_ndtvadsrqf(eq)
    wide_scan(eq)
    deep_walk_scene(eq)
    check_khosdevozr(eq)
    check_wpwvsglgmb(eq)

    # Now try the naive probe — does any standard action change the counter?
    try_naive_win_probe(env, game, eq)

    # After the actions, re-dump to see changes
    dump_ndtvadsrqf(eq)

    # FINAL PROBE: try clicking every pixel where a piece might be, after
    # doing any scroll-inducing sequence. Skip state monitoring — just step
    # through a sequence of clicks on each piece's rendered pixel and see
    # if ANY action produces a win or a counter change.
    print("\n=== click every visible piece pixel 20x + ACTION6 probe ===")
    from arcengine import GameAction
    # Reset by advancing undo/resetting? No — just continue from current state.
    # Dump current counter and offset, then try clicks at every pixel where
    # right-N would be under various possible offsets.
    for offset_trial in [(5,5), (-39,5), (-79,5)]:
        # compute pixel for right-N (22,6)
        px_x = 22*6 + offset_trial[0] + 3
        px_y = 6*6 + offset_trial[1] + 3
        print(f"  offset={offset_trial}: right-N would be at pixel ({px_x},{px_y})")
    # Actually click these
    for (px_x, px_y) in [(140, 42), (56, 42), (16, 42), (134, 41), (130, 40)]:
        sigb_cnt = len(eq.hncnfaqaddg.ndtvadsrqf("fozwvlovdui"))
        try:
            fd = env.step(GameAction.ACTION6, data={'x': px_x, 'y': px_y})
        except TypeError:
            ac = GameAction.ACTION6
            ac.data = {'x': px_x, 'y': px_y}
            fd = env.step(ac)
        siga_cnt = len(eq.hncnfaqaddg.ndtvadsrqf("fozwvlovdui"))
        print(f"  click ({px_x},{px_y}) count {sigb_cnt}->{siga_cnt} won={eq.iajuzrgttrv}")


if __name__ == "__main__":
    main()
