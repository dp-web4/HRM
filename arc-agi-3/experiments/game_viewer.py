#!/usr/bin/env python3
"""
Game Viewer — 3x3 level grid + action sidebar, live updates.

http://localhost:8765
"""

import http.server
import json
import os
import hashlib
import numpy as np
from io import BytesIO
import base64

STATE_DIR = "/tmp/claude_solver"
PORT = 8765

# ARC-AGI-3 palette — must match arc_vision.py
ARC_PALETTE = {
    0:  (0, 0, 0),        # black
    1:  (0, 116, 217),    # blue
    2:  (255, 65, 54),    # red
    3:  (46, 204, 64),    # green
    4:  (255, 220, 0),    # yellow
    5:  (170, 170, 170),  # gray
    6:  (240, 18, 190),   # magenta
    7:  (255, 133, 27),   # orange
    8:  (0, 220, 220),    # cyan
    9:  (165, 103, 63),   # brown
    10: (255, 175, 200),  # pink
    11: (128, 0, 0),      # maroon
    12: (128, 128, 0),    # olive
    13: (0, 0, 128),      # navy
    14: (0, 128, 128),    # teal
    15: (255, 255, 255),  # white
}


def grid_to_png_b64(grid, scale=4):
    from PIL import Image
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


def load_state():
    p = os.path.join(STATE_DIR, "session.json")
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)


def load_grid(name="current"):
    p = os.path.join(STATE_DIR, f"{name}_grid.npy")
    return np.load(p) if os.path.exists(p) else None


def shash():
    p = os.path.join(STATE_DIR, "session.json")
    if not os.path.exists(p): return ""
    return hashlib.md5(open(p,'rb').read()).hexdigest()[:8]


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>ARC-AGI-3 — {game_id}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#111;color:#ccc;font-family:'SF Mono','Fira Code',monospace;height:100vh;overflow:hidden}}
.layout{{display:flex;height:100vh}}
.main{{flex:1;padding:12px;overflow-y:auto}}
.sidebar{{width:320px;background:#0d0d0d;border-left:1px solid #222;display:flex;flex-direction:column}}
.header{{display:flex;justify-content:space-between;align-items:center;padding:6px 0 10px;border-bottom:1px solid #333;margin-bottom:10px}}
.header h1{{color:#ff6b6b;font-size:1.2em}}
.stats{{display:flex;gap:20px;font-size:1em}}
.sv{{color:#4ecdc4;font-weight:bold}}
.grid3x3{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;justify-content:start}}
.cell{{background:#1a1a1a;border:2px solid #222;border-radius:5px;padding:4px;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;aspect-ratio:1;overflow:hidden}}
.cell.active{{border-color:#ff6b6b;box-shadow:0 0 10px rgba(255,107,107,0.3)}}
.cell.solved{{border-color:#4ecdc4}}
.cell.future{{opacity:0.3}}
.cell img{{image-rendering:pixelated;border-radius:2px}}
.cell-pair{{display:flex;gap:4px;justify-content:center;align-items:center;width:100%;height:100%}}
.cell-pair img{{width:48%;aspect-ratio:1;object-fit:contain}}
.cell-active-img{{width:100%;aspect-ratio:1;object-fit:contain}}
.clabel{{font-size:0.7em;margin-top:4px;flex-shrink:0}}
.clabel.active{{color:#ff6b6b}}
.clabel.solved{{color:#4ecdc4}}
.clabel.future{{color:#333}}
.sidebar-header{{padding:10px;border-bottom:1px solid #222;color:#666;font-size:0.85em}}
.actions{{flex:1;overflow-y:auto;padding:8px}}
.act{{padding:3px 6px;border-bottom:1px solid #181818;font-size:0.75em;display:flex;gap:6px}}
.act .s{{color:#444;min-width:30px}}
.act .a{{color:#4ecdc4;min-width:100px}}
.act .d{{color:#777;flex:1}}
.no-game{{text-align:center;margin-top:100px;color:#333;font-size:1.3em}}
</style>
<script>
let lh="{state_hash}";
async function ck(){{try{{const r=await fetch('/hash');const h=await r.text();if(h!==lh)location.reload()}}catch(e){{}}setTimeout(ck,500)}}
ck();
</script>
</head>
<body>
<div class="layout">
<div class="main">
{main_content}
</div>
<div class="sidebar">
<div class="sidebar-header">Actions</div>
<div class="actions">
{action_content}
</div>
</div>
</div>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/hash':
            self.send_response(200)
            self.send_header('Content-Type','text/plain')
            self.end_headers()
            self.wfile.write(shash().encode())
            return

        self.send_response(200)
        self.send_header('Content-Type','text/html')
        self.end_headers()

        session = load_state()
        if not session:
            html = PAGE.format(game_id="waiting", state_hash="",
                             main_content='<div class="no-game">No active game</div>',
                             action_content='')
            self.wfile.write(html.encode())
            return

        game_id = session.get('game_id','?')
        step = session.get('step',0)
        cur_level = session.get('levels_completed',0)
        win_levels = session.get('win_levels',8)
        state = session.get('state','?')
        observations = session.get('observations',[])
        current_grid = load_grid("current")
        blank = blank_b64(scale=3)

        # Header with player name
        player = session.get('player', 'unknown')
        player_color = '#ff6b6b' if player == 'claude' else '#4ecdc4'
        main = [f'''<div class="header">
            <h1>🎮 {game_id} <span style="color:{player_color};font-size:0.8em">▸ {player}</span></h1>
            <div class="stats">
                <div>Step <span class="sv">{step}</span></div>
                <div>Level <span class="sv">{cur_level}/{win_levels}</span></div>
                <div><span class="sv">{state}</span></div>
            </div>
        </div>''']

        # 3x3 grid
        main.append('<div class="grid3x3">')
        for pos in range(9):
            lvl = pos  # level index
            if lvl < win_levels:
                if lvl < cur_level:
                    # Solved — start→final side by side, same box size
                    sp = os.path.join(STATE_DIR, f"level_{lvl}_start_grid.npy")
                    fp = os.path.join(STATE_DIR, f"level_{lvl}_final_grid.npy")
                    sb = grid_to_png_b64(np.load(sp), scale=2) if os.path.exists(sp) else blank
                    fb = grid_to_png_b64(np.load(fp), scale=2) if os.path.exists(fp) else blank
                    main.append(f'''<div class="cell solved">
                        <div class="cell-pair">
                            <img src="data:image/png;base64,{sb}" title="Start">
                            <img src="data:image/png;base64,{fb}" title="Solved">
                        </div>
                        <div class="clabel solved">L{lvl+1} ✓</div>
                    </div>''')
                elif lvl == cur_level:
                    # Active — single image fills the box
                    cb = grid_to_png_b64(current_grid, scale=4) if current_grid is not None else blank
                    main.append(f'''<div class="cell active">
                        <img class="cell-active-img" src="data:image/png;base64,{cb}">
                        <div class="clabel active">L{lvl+1} ▶ LIVE</div>
                    </div>''')
                else:
                    # Future — same box, dimmed placeholder
                    main.append(f'''<div class="cell future">
                        <img class="cell-active-img" src="data:image/png;base64,{blank}">
                        <div class="clabel future">L{lvl+1}</div>
                    </div>''')
            else:
                pass  # no empty cells needed

        main.append('</div>')

        # Actions sidebar
        acts = []
        for obs in observations[-50:]:
            s = obs.get('step','?')
            a = obs.get('action','?')
            d = obs.get('diff','')[:60]
            acts.append(f'<div class="act"><span class="s">#{s}</span>'
                       f'<span class="a">{a}</span><span class="d">{d}</span></div>')
        if not acts:
            acts.append('<div class="act" style="color:#333">Waiting...</div>')

        html = PAGE.format(
            game_id=game_id, state_hash=shash(),
            main_content='\n'.join(main),
            action_content='\n'.join(acts)
        )
        self.wfile.write(html.encode())

    def log_message(self, *a): pass


def main():
    s = http.server.HTTPServer(('', PORT), Handler)
    print(f"Game Viewer: http://localhost:{PORT}")
    try: s.serve_forever()
    except KeyboardInterrupt: s.server_close()

if __name__ == "__main__":
    main()
