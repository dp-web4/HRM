#!/usr/bin/env python3
"""
Game Viewer — Live game display + Stored game browser.

http://localhost:8765         — live view (current session)
http://localhost:8765/browse  — stored game browser
http://localhost:8765/game/ft09/run_20260410_091336 — specific run
"""

import http.server
import json
import os
import hashlib
import glob
import numpy as np
from io import BytesIO
from collections import defaultdict
import base64
from urllib.parse import unquote

STATE_DIR = "/tmp/sage_solver"
VISUAL_MEMORY = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "shared-context", "arc-agi-3", "visual-memory")
PORT = 8765

ARC_PALETTE = {
    0:  (255, 255, 255), 1:  (204, 204, 204), 2:  (153, 153, 153),
    3:  (102, 102, 102), 4:  (51, 51, 51),     5:  (0, 0, 0),
    6:  (229, 58, 163),  7:  (255, 123, 204),  8:  (249, 60, 49),
    9:  (30, 147, 255),  10: (136, 216, 241),  11: (255, 220, 0),
    12: (255, 133, 27),  13: (146, 18, 49),    14: (79, 204, 48),
    15: (163, 86, 214),
}


def grid_to_png_b64(grid, scale=4):
    from PIL import Image
    if grid.ndim == 3:
        grid = grid[-1]  # multi-frame: take last
    if grid.ndim != 2:
        grid = np.full((64, 64), 4, dtype=np.int8)  # fallback blank
    h, w = grid.shape
    img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = ARC_PALETTE.get(int(grid[r,c]), (128,128,128))
    buf = BytesIO()
    Image.fromarray(img).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def blank_b64(scale=4):
    return grid_to_png_b64(np.full((64,64), 4, dtype=np.int8), scale)


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def load_state():
    p = os.path.join(STATE_DIR, "session.json")
    if not os.path.exists(p): return None
    for _ in range(3):
        try:
            with open(p) as f: return json.load(f)
        except (json.JSONDecodeError, ValueError):
            import time; time.sleep(0.1)
    return None


def load_grid(name="current"):
    p = os.path.join(STATE_DIR, f"{name}_grid.npy")
    return np.load(p) if os.path.exists(p) else None


def shash():
    p = os.path.join(STATE_DIR, "session.json")
    if not os.path.exists(p): return ""
    h = hashlib.md5(open(p,'rb').read())
    anim_dir = os.path.join(STATE_DIR, "animations")
    if os.path.isdir(anim_dir):
        for f in sorted(os.listdir(anim_dir)):
            h.update(f.encode())
    return h.hexdigest()[:8]


# ─── Common CSS ───────────────────────────────────────────

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{background:#111;color:#ccc;font-family:'SF Mono','Fira Code',monospace}
a{color:#4ecdc4;text-decoration:none}
a:hover{text-decoration:underline}
.nav{background:#0a0a0a;padding:8px 16px;border-bottom:1px solid #222;display:flex;gap:20px;align-items:center}
.nav a{font-size:0.9em}
.nav .active{color:#ff6b6b;font-weight:bold}
.container{padding:16px;max-width:1400px;margin:0 auto}
.header{display:flex;justify-content:space-between;align-items:center;padding:6px 0 10px;border-bottom:1px solid #333;margin-bottom:10px}
.header h1{color:#ff6b6b;font-size:1.2em}
.stats{display:flex;gap:20px;font-size:1em}
.sv{color:#4ecdc4;font-weight:bold}
.grid3x3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.cell{background:#1a1a1a;border:2px solid #222;border-radius:5px;padding:4px;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;aspect-ratio:1;overflow:hidden}
.cell.solved{border-color:#4ecdc4}
.cell.active{border-color:#ff6b6b;box-shadow:0 0 10px rgba(255,107,107,0.3)}
.cell.future{opacity:0.3}
.cell img{image-rendering:pixelated;border-radius:2px}
.cell-pair{display:flex;gap:4px;justify-content:center;align-items:center;width:100%;height:100%}
.cell-pair img{width:48%;aspect-ratio:1;object-fit:contain}
.cell-active-img{width:100%;aspect-ratio:1;object-fit:contain}
.clabel{font-size:0.7em;margin-top:4px;flex-shrink:0}
.clabel.solved{color:#4ecdc4}
.clabel.active{color:#ff6b6b}
.clabel.future{color:#333}
.gif-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(128px,1fr));gap:8px;margin-top:16px}
.gif-cell{background:#1a1a1a;border:1px solid #222;border-radius:4px;padding:4px;text-align:center;cursor:pointer}
.gif-cell:hover{border-color:#a855f7}
.gif-cell img{width:100%;image-rendering:pixelated}
.gif-cell .glabel{font-size:0.6em;color:#a855f7;margin-top:2px}
.game-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin-top:12px}
.game-card{background:#1a1a1a;border:1px solid #222;border-radius:6px;padding:12px}
.game-card:hover{border-color:#4ecdc4}
.game-card h3{color:#4ecdc4;margin-bottom:6px}
.game-card .meta{font-size:0.8em;color:#666}
.run-link{display:block;padding:4px 0;font-size:0.85em}
"""


# ─── Canonical status from game_coordination.json ─────────

_COORD_CACHE = {}
def canonical_status(game):
    """Return dict with solved/best_level/levels/solved_by/solved_actions/baseline/status
    from game_coordination.json. Cached per page-load."""
    coord_path = os.path.join(os.path.dirname(os.path.abspath(VISUAL_MEMORY)),
                              'game_coordination.json')
    if not _COORD_CACHE.get('_mtime') == (os.path.getmtime(coord_path) if os.path.exists(coord_path) else None):
        _COORD_CACHE.clear()
        _COORD_CACHE['_mtime'] = os.path.getmtime(coord_path) if os.path.exists(coord_path) else None
        if os.path.exists(coord_path):
            try:
                with open(coord_path) as f:
                    coord = json.load(f)
                for g in coord.get('games', []):
                    _COORD_CACHE[g.get('family')] = g
            except Exception:
                pass
    return _COORD_CACHE.get(game, {})


def status_pill(game):
    """Small HTML pill indicating SOLVED/PARTIAL/n status for a game."""
    s = canonical_status(game)
    if not s:
        return ''
    bl = s.get('best_level')
    tl = s.get('levels')
    solved = s.get('solved', False)
    if bl is None or tl is None:
        return ''
    if solved:
        return f'<span class="status-pill solved">{bl}/{tl} WIN</span>'
    elif bl == 0:
        return f'<span class="status-pill zero">0/{tl}</span>'
    else:
        return f'<span class="status-pill partial">{bl}/{tl}</span>'


# ─── Browse: list all stored games ────────────────────────

def render_browse():
    vm = os.path.abspath(VISUAL_MEMORY)
    games = {}  # game → list of (kind, label, meta) tuples
    if os.path.isdir(vm):
        for game_dir in sorted(os.listdir(vm)):
            game_path = os.path.join(vm, game_dir)
            if not os.path.isdir(game_path):
                continue
            entries = []
            # Collect run_* subdirectories (full replay runs)
            runs = sorted([d for d in os.listdir(game_path)
                          if d.startswith("run_") and os.path.isdir(os.path.join(game_path, d))])
            for run in runs:
                run_path = os.path.join(game_path, run, "run.json")
                meta = ""
                if os.path.exists(run_path):
                    try:
                        with open(run_path) as f:
                            rm = json.load(f)
                        meta = f" — {rm.get('levels_completed','?')}L, {rm.get('total_steps','?')}steps, {rm.get('result','?')}"
                    except Exception:
                        pass
                entries.append(('run', run, meta))
            # Collect flat PNGs at game root (level-snapshot artifacts from agents)
            flat_pngs = sorted(f for f in os.listdir(game_path)
                              if f.endswith('.png') and os.path.isfile(os.path.join(game_path, f)))
            if flat_pngs:
                # Extract level info from filenames
                levels = set()
                for f in flat_pngs:
                    # Patterns: L0_start.png, L1_solved.png, L2_stuck.png, etc.
                    parts = f.replace('.png', '').split('_', 1)
                    if parts[0].startswith('L') and parts[0][1:].isdigit():
                        levels.add(parts[0])
                meta = f" — {len(flat_pngs)} frames, {len(levels)} levels" if levels else f" — {len(flat_pngs)} frames"
                entries.append(('flat', 'level-snapshots', meta))
            if entries:
                games[game_dir] = entries

    cards = []
    for game, entries in sorted(games.items()):
        links = []
        for kind, label, meta in entries:
            if kind == 'run':
                links.append(f'<a class="run-link" href="/game/{game}/{label}">{label}{meta}</a>')
            else:  # flat
                links.append(f'<a class="run-link" href="/game/{game}/_flat">{label}{meta}</a>')
        pill = status_pill(game)
        cards.append(f'<div class="game-card"><h3>{game} {pill}</h3>{"".join(links)}</div>')

    if not cards:
        cards = ['<p style="color:#666">No stored games found.</p>']

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>ARC-AGI-3 — Stored Games</title><style>{CSS}
.status-pill{{display:inline-block;margin-left:8px;padding:2px 6px;border-radius:3px;font-size:0.7em;font-weight:bold;vertical-align:middle}}
.status-pill.solved{{background:#4ecdc4;color:#000}}
.status-pill.partial{{background:#e0a020;color:#000}}
.status-pill.zero{{background:#666;color:#ccc}}
</style></head><body>
<div class="nav"><a href="/">Live</a> <a href="/browse" class="active">Stored</a></div>
<div class="container">
<h2 style="margin-bottom:12px">Stored Games</h2>
<div class="game-list">{"".join(cards)}</div>
</div></body></html>"""


# ─── Game flat view: level snapshots at game root ────────

def render_game_flat(game):
    """Render games whose frames live as flat files at the game root
    (e.g. L0_start.png, L1_solved.png, L2_stuck.png) rather than in run_* dirs.

    Groups by level, shows all phases per level (start/solved/stuck/etc).
    """
    game_dir = os.path.join(os.path.abspath(VISUAL_MEMORY), game)
    if not os.path.isdir(game_dir):
        return f"<html><body>Game not found: {game}</body></html>"

    # Find all flat PNGs and parse level + phase from name
    flat_pngs = sorted(f for f in os.listdir(game_dir)
                      if f.endswith('.png') and os.path.isfile(os.path.join(game_dir, f)))

    # Phase priority: beginning → end (left-to-right reading order)
    PHASE_ORDER = {
        'initial': 0, 'start': 1, 'begin': 2, 'pre': 3,
        'mid': 10, 'probe': 11, 'frame': 12,
        'stuck': 20, 'exec_fail': 21, 'fail': 22,
        'solved': 30, 'final': 31, 'end': 32, 'post': 33, 'win': 34,
    }
    def phase_key(phase):
        # exact match first, then prefix match, then fallback to alphabetical
        if phase in PHASE_ORDER:
            return (PHASE_ORDER[phase], phase)
        for key, pri in PHASE_ORDER.items():
            if phase.startswith(key):
                return (pri, phase)
        return (100, phase)

    # Phases that show the POST-transition frame (unreliable: show next level's start).
    # Hide these unless a per-game override exists.
    POST_TRANSITION_PHASES = {'solved', 'end', 'final', 'win'}

    # Group by level
    by_level = defaultdict(list)  # level_num → [(phase, filename), ...]
    other = []
    hidden_count = 0
    for f in flat_pngs:
        stem = f.replace('.png', '')
        parts = stem.split('_', 1)
        if parts[0].startswith('L') and parts[0][1:].isdigit():
            lv = int(parts[0][1:])
            phase = parts[1] if len(parts) > 1 else 'frame'
            # Skip post-transition phases — they show the NEXT level's start,
            # not the "solved" state. Would need -2 replay to capture correctly.
            if phase in POST_TRANSITION_PHASES:
                hidden_count += 1
                continue
            by_level[lv].append((phase, f))
        else:
            other.append(f)

    # Sort phases within each level: beginning → end
    for lv in by_level:
        by_level[lv].sort(key=lambda pf: phase_key(pf[0]))

    # Load results/solutions JSON and extract a human-friendly summary
    summary = {}  # merged summary from all available JSON files
    for json_name in ['solutions.json', f'{game}_results.json', 'results.json']:
        jp = os.path.join(game_dir, json_name)
        if os.path.exists(jp):
            try:
                with open(jp) as jf:
                    d = json.load(jf)
                for k in ('result', 'levels_solved', 'total_actions', 'solver',
                          'win_levels', 'baseline', 'levels_completed'):
                    if k in d and k not in summary:
                        summary[k] = d[k]
            except Exception:
                pass
    # Also pull from shared-context game_coordination.json for canonical status
    coord_path = os.path.join(os.path.dirname(os.path.abspath(VISUAL_MEMORY)),
                              'game_coordination.json')
    if os.path.exists(coord_path):
        try:
            with open(coord_path) as f:
                coord = json.load(f)
            for g in coord.get('games', []):
                if g.get('family') == game:
                    summary['_solved'] = g.get('solved', False)
                    summary['_solved_by'] = g.get('solved_by', '')
                    summary['_solved_actions'] = g.get('solved_actions', None)
                    summary['_best_level'] = g.get('best_level', None)
                    summary['_levels'] = g.get('levels', None)
                    summary['_baseline'] = g.get('baseline', None)
                    summary['_status'] = g.get('status', '')
                    break
        except Exception:
            pass

    # Detect indexing convention: if min level number is 0, files are 0-indexed
    # (display as L1, L2, ...); if min is ≥1, files already use 1-indexed labels
    # (display as-is without offset)
    min_level = min(by_level.keys()) if by_level else 0
    display_offset = 1 if min_level == 0 else 0

    # Build level cards
    cards = []
    for lv in sorted(by_level.keys()):
        phase_imgs = []
        for phase, fname in by_level[lv]:
            b64 = file_to_b64(os.path.join(game_dir, fname))
            phase_imgs.append(f'''<div class="phase">
                <img src="data:image/png;base64,{b64}" title="{fname}">
                <div class="plabel">{phase}</div>
            </div>''')
        cards.append(f'''<div class="flat-cell">
            <div class="flat-label">L{lv + display_offset} — {len(by_level[lv])} frames</div>
            <div class="phase-row">{"".join(phase_imgs)}</div>
        </div>''')

    if not cards:
        cards = ['<p style="color:#666">No level frames found.</p>']

    # Build a compact status pill row from summary
    pills = []
    solved = summary.get('_solved', False)
    if '_best_level' in summary and '_levels' in summary and summary['_best_level'] is not None:
        bl = summary['_best_level']; tl = summary['_levels']
        color = '#4ecdc4' if solved else ('#ff6b6b' if bl < tl else '#888')
        pills.append(f'<span style="background:{color};color:#000;padding:2px 8px;border-radius:3px;font-weight:bold">{bl}/{tl} {"SOLVED" if solved else "PARTIAL"}</span>')
    if summary.get('_solved_actions'):
        pills.append(f'<span class="pill">{summary["_solved_actions"]} actions</span>')
    if summary.get('_baseline'):
        pills.append(f'<span class="pill">baseline {summary["_baseline"]}</span>')
    if summary.get('_solved_by'):
        pills.append(f'<span class="pill">by {summary["_solved_by"]}</span>')
    if summary.get('solver'):
        pills.append(f'<span class="pill" style="color:#666">{summary["solver"]}</span>')
    if summary.get('_status') and summary['_status'] not in ('solved', 'in_progress'):
        pills.append(f'<span class="pill" style="color:#c97">{summary["_status"]}</span>')

    notes = []
    if hidden_count:
        notes.append(f'Hiding {hidden_count} post-transition frames (they show next level\'s start, not the solved state).')

    meta_html = ''
    if pills:
        meta_html += '<div class="pill-row">' + ' '.join(pills) + '</div>'
    if notes:
        meta_html += '<div class="note-row">' + ' · '.join(notes) + '</div>'

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>ARC-AGI-3 — {game} (level snapshots)</title><style>{CSS}
.flat-list{{display:flex;flex-direction:column;gap:4px}}
.flat-cell{{background:#181818;border:1px solid #2a2a2a;border-radius:3px;padding:4px 8px;display:flex;align-items:center;gap:12px}}
.flat-label{{color:#4ecdc4;font-size:0.85em;min-width:110px;flex-shrink:0}}
.phase-row{{display:flex;gap:4px;flex-wrap:wrap;flex:1}}
.phase{{text-align:center}}
.phase img{{width:160px;height:160px;image-rendering:pixelated;background:#222;border:1px solid #333;display:block}}
.plabel{{font-size:0.7em;color:#888;margin-top:1px;line-height:1}}
.container{{padding:8px 12px}}
.header{{margin-bottom:6px}}
.header h1{{font-size:1.1em}}
.pill-row{{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:4px;font-size:0.8em}}
.pill{{background:#222;color:#aaa;padding:2px 8px;border-radius:3px;border:1px solid #333}}
.note-row{{font-size:0.75em;color:#666;margin-bottom:6px}}
</style></head><body>
<div class="nav"><a href="/">Live</a> <a href="/browse" class="active">Stored</a></div>
<div class="container">
<div class="header"><h1>{game} <span style="font-size:0.7em;color:#666">level snapshots</span></h1></div>
{meta_html}
<div class="flat-list">{"".join(cards)}</div>
</div></body></html>"""


# ─── Game run view: first/last per level + GIF grid ──────

def render_game_run(game, run):
    run_dir = os.path.join(os.path.abspath(VISUAL_MEMORY), game, run)
    if not os.path.isdir(run_dir):
        return f"<html><body>Run not found: {game}/{run}</body></html>"

    # Load run metadata
    meta = {}
    meta_path = os.path.join(run_dir, "run.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # Collect PNGs and GIFs grouped by level
    pngs = sorted(f for f in os.listdir(run_dir) if f.endswith('.png'))
    gifs = sorted(f for f in os.listdir(run_dir) if f.endswith('.gif'))

    by_level = defaultdict(list)
    for f in pngs:
        if '_L' in f:
            try:
                lv = int(f.split('_L')[1].split('_')[0])
                by_level[lv].append(f)
            except (ValueError, IndexError):
                pass

    # Build level grid: first and last frame per level
    levels_completed = meta.get('levels_completed', len(by_level))
    win_levels = meta.get('win_levels', max(by_level.keys(), default=0) + 1)

    cells = []
    for pos in range(max(9, win_levels)):
        lv = pos
        if lv in by_level:
            frames = by_level[lv]
            first_path = os.path.join(run_dir, frames[0])
            is_solved = lv < levels_completed

            # For solved levels: the LAST frame tagged with this level is actually
            # the post-transition frame (SDK returns next level start for the winning
            # click). The true "solved state" is the second-to-last frame.
            # For unsolved/active levels: last frame is the current state.
            if is_solved and len(frames) >= 2:
                last_path = os.path.join(run_dir, frames[-2])
                last_label = frames[-2]
            else:
                last_path = os.path.join(run_dir, frames[-1])
                last_label = frames[-1]

            first_b64 = file_to_b64(first_path)
            last_b64 = file_to_b64(last_path)
            cls = "solved" if is_solved else "active"
            mark = "✓" if is_solved else f"({len(frames)})"
            cells.append(f'''<div class="cell {cls}">
                <div class="cell-pair">
                    <img src="data:image/png;base64,{first_b64}" title="{frames[0]}">
                    <img src="data:image/png;base64,{last_b64}" title="{last_label}">
                </div>
                <div class="clabel {cls}">L{lv+1} {mark} — {len(frames)} steps</div>
            </div>''')
        elif lv < win_levels:
            cells.append(f'<div class="cell future"><div class="clabel future">L{lv+1}</div></div>')

    # GIF grid
    gif_cells = []
    for g in gifs:
        gif_b64 = file_to_b64(os.path.join(run_dir, g))
        # Extract level and step from filename
        label = g.replace('.gif', '').replace('_anim', ' anim').replace('step_', 'S')
        gif_cells.append(f'''<div class="gif-cell" onclick="this.querySelector('img').src='data:image/gif;base64,{gif_b64}'">
            <img src="data:image/gif;base64,{gif_b64}">
            <div class="glabel">{label}</div>
        </div>''')

    result = meta.get('result', '?')
    total = meta.get('total_steps', '?')
    player = meta.get('player', '?')
    result_color = '#4ecdc4' if result == 'WIN' else '#ff6b6b'

    ncols = 3 if win_levels <= 9 else 4

    # Canonical status banner (shows true game state from coordination,
    # independent of this particular run's outcome)
    canon = canonical_status(game)
    canon_html = ''
    if canon:
        c_bl = canon.get('best_level', '?')
        c_tl = canon.get('levels', '?')
        c_solved = canon.get('solved', False)
        c_by = canon.get('solved_by', '')
        c_acts = canon.get('solved_actions')
        c_base = canon.get('baseline')
        c_status = canon.get('status', '')

        canon_bits = []
        if c_solved:
            canon_bits.append(f'<span style="background:#4ecdc4;color:#000;padding:2px 8px;border-radius:3px;font-weight:bold">{c_bl}/{c_tl} WIN</span>')
        else:
            canon_bits.append(f'<span style="background:#e0a020;color:#000;padding:2px 8px;border-radius:3px;font-weight:bold">{c_bl}/{c_tl} PARTIAL</span>')
        if c_acts: canon_bits.append(f'<span class="pill">{c_acts} actions</span>')
        if c_base: canon_bits.append(f'<span class="pill">baseline {c_base}</span>')
        if c_by: canon_bits.append(f'<span class="pill">by {c_by}</span>')
        if c_status and c_status not in ('solved', 'in_progress'):
            canon_bits.append(f'<span class="pill" style="color:#c97">{c_status}</span>')
        canon_html = f'<div class="canon-banner"><span style="color:#888;font-size:0.75em;margin-right:6px">GAME:</span>{" ".join(canon_bits)}</div>'

    # Note if this run diverges from canonical status
    run_note = ''
    if canon.get('solved') and result != 'WIN':
        run_note = '<div class="note-row" style="color:#e0a020">This is a partial-attempt run. The game was solved separately (see GAME status above).</div>'

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>ARC-AGI-3 — {game} {run}</title><style>{CSS}
.grid3x3{{grid-template-columns:repeat({ncols},1fr)}}
.canon-banner{{background:#181818;border:1px solid #2a2a2a;border-radius:3px;padding:6px 10px;margin-bottom:8px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
.pill{{background:#222;color:#aaa;padding:2px 8px;border-radius:3px;border:1px solid #333;font-size:0.8em}}
.note-row{{font-size:0.75em;color:#666;margin-bottom:6px}}
</style></head><body>
<div class="nav"><a href="/">Live</a> <a href="/browse" class="active">Stored</a></div>
<div class="container">
<div class="header">
    <h1>{game} <span style="font-size:0.7em;color:#666">{run}</span></h1>
    <div class="stats">
        <div>Steps <span class="sv">{total}</span></div>
        <div>Levels <span class="sv">{levels_completed}/{win_levels}</span></div>
        <div><span style="color:{result_color};font-weight:bold">{result}</span></div>
        <div style="color:#666">{player}</div>
    </div>
</div>
{canon_html}
{run_note}
<div class="grid3x3">{"".join(cells)}</div>
{"<h3 style='margin-top:16px;color:#a855f7'>Animations (" + str(len(gifs)) + ")</h3>" if gifs else ""}
<div class="gif-grid">{"".join(gif_cells)}</div>
</div></body></html>"""


# ─── Live view (original viewer) ──────────────────────────

def render_live():
    session = load_state()
    if not session:
        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>ARC-AGI-3</title><style>{CSS}</style>
<script>let lh="";async function ck(){{try{{const r=await fetch('/hash');const h=await r.text();if(h!==lh)location.reload()}}catch(e){{}}setTimeout(ck,500)}}ck();</script>
</head><body>
<div class="nav"><a href="/" class="active">Live</a> <a href="/browse">Stored</a></div>
<div class="container"><div style="text-align:center;margin-top:100px;color:#333;font-size:1.3em">No active game</div></div>
</body></html>"""

    game_id = session.get('game_id','?')
    step = session.get('step',0)
    cur_level = session.get('levels_completed',0)
    win_levels = session.get('win_levels',8)
    state = session.get('state','?')
    current_grid = load_grid("current")
    blank = blank_b64(scale=3)
    player = session.get('player', 'unknown')
    player_color = '#ff6b6b' if player == 'claude' else '#4ecdc4'

    cells = []
    for pos in range(max(9, win_levels)):
        lvl = pos
        if lvl >= win_levels:
            continue
        if lvl < cur_level:
            sp = os.path.join(STATE_DIR, f"level_{lvl}_start_grid.npy")
            fp = os.path.join(STATE_DIR, f"level_{lvl}_final_grid.npy")
            sb = grid_to_png_b64(np.load(sp), scale=2) if os.path.exists(sp) else blank
            fb = grid_to_png_b64(np.load(fp), scale=2) if os.path.exists(fp) else blank
            cells.append(f'''<div class="cell solved">
                <div class="cell-pair">
                    <img src="data:image/png;base64,{sb}" title="Start">
                    <img src="data:image/png;base64,{fb}" title="Solved">
                </div>
                <div class="clabel solved">L{lvl+1} ✓</div>
            </div>''')
        elif lvl == cur_level:
            cb = grid_to_png_b64(current_grid, scale=4) if current_grid is not None else blank
            cells.append(f'''<div class="cell active">
                <img class="cell-active-img" src="data:image/png;base64,{cb}">
                <div class="clabel active">L{lvl+1} ▶ LIVE</div>
            </div>''')
        else:
            cells.append(f'''<div class="cell future">
                <img class="cell-active-img" src="data:image/png;base64,{blank}">
                <div class="clabel future">L{lvl+1}</div>
            </div>''')

    # Actions sidebar
    observations = session.get('observations',[])
    anim_steps = {a.get('step') for a in session.get('animations', [])}
    acts = []
    for obs in observations[-50:]:
        s = obs.get('step','?')
        a = obs.get('action','?')
        d = obs.get('diff','')[:60]
        marker = ' <span style="color:#a855f7">▶</span>' if s in anim_steps else ''
        acts.append(f'<div class="act"><span class="s">#{s}{marker}</span>'
                   f'<span class="a">{a}</span><span class="d">{d}</span></div>')
    if not acts:
        acts.append('<div class="act" style="color:#333">Waiting...</div>')

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>ARC-AGI-3 — {game_id}</title><style>{CSS}
.layout{{display:flex;height:100vh}}
.main{{flex:1;padding:12px;overflow-y:auto}}
.sidebar{{width:320px;background:#0d0d0d;border-left:1px solid #222;display:flex;flex-direction:column}}
.sidebar-header{{padding:10px;border-bottom:1px solid #222;color:#666;font-size:0.85em}}
.actions{{flex:1;overflow-y:auto;padding:8px}}
.act{{padding:3px 6px;border-bottom:1px solid #181818;font-size:0.75em;display:flex;gap:6px}}
.act .s{{color:#444;min-width:30px}}
.act .a{{color:#4ecdc4;min-width:100px}}
.act .d{{color:#777;flex:1}}
</style>
<script>let lh="{shash()}";async function ck(){{try{{const r=await fetch('/hash');const h=await r.text();if(h!==lh)location.reload()}}catch(e){{}}setTimeout(ck,500)}}ck();</script>
</head><body>
<div class="nav"><a href="/" class="active">Live</a> <a href="/browse">Stored</a></div>
<div class="layout">
<div class="main">
<div class="header">
    <h1>{game_id} <span style="color:{player_color};font-size:0.8em">▸ {player}</span></h1>
    <div class="stats">
        <div>Step <span class="sv">{step}</span></div>
        <div>Level <span class="sv">{cur_level}/{win_levels}</span></div>
        <div><span class="sv">{state}</span></div>
    </div>
</div>
<div class="grid3x3">{"".join(cells)}</div>
</div>
<div class="sidebar">
<div class="sidebar-header">Actions</div>
<div class="actions">{"".join(acts)}</div>
</div>
</div></body></html>"""


# ─── HTTP Handler ─────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path)

        if path == '/hash':
            self.send_response(200)
            self.send_header('Content-Type','text/plain')
            self.end_headers()
            self.wfile.write(shash().encode())
            return

        self.send_response(200)
        self.send_header('Content-Type','text/html')
        self.end_headers()

        if path == '/browse':
            html = render_browse()
        elif path.startswith('/game/'):
            parts = path[6:].strip('/').split('/')
            if len(parts) >= 2 and parts[1] == '_flat':
                html = render_game_flat(parts[0])
            elif len(parts) >= 2:
                html = render_game_run(parts[0], parts[1])
            else:
                html = render_browse()
        else:
            html = render_live()

        self.wfile.write(html.encode())

    def log_message(self, *a): pass


def main():
    s = http.server.HTTPServer(('', PORT), Handler)
    print(f"Game Viewer: http://localhost:{PORT}")
    print(f"  Live:   http://localhost:{PORT}/")
    print(f"  Browse: http://localhost:{PORT}/browse")
    try: s.serve_forever()
    except KeyboardInterrupt: s.server_close()

if __name__ == "__main__":
    main()
