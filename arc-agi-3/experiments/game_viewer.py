#!/usr/bin/env python3
"""
Game Viewer — shows all levels at a glance with live updates.

Serves http://localhost:8765 showing:
- One board per level (blank initially, filled as solved)
- Current level updates LIVE as actions are taken
- Previous levels show final solved state
- Auto-refreshes on every state change

The claude_solver writes state to /tmp/claude_solver/.
This viewer reads it and renders all boards.

Usage:
    python3 game_viewer.py
    # Open http://localhost:8765 in browser
"""

import http.server
import json
import os
import time
import hashlib
import numpy as np
from io import BytesIO
import base64

STATE_DIR = "/tmp/claude_solver"
PORT = 8765

ARC_PALETTE = {
    0:(255,255,255),1:(204,204,204),2:(153,153,153),3:(102,102,102),4:(51,51,51),
    5:(0,0,0),6:(229,58,163),7:(255,123,204),8:(249,60,49),9:(30,147,255),
    10:(136,216,241),11:(255,220,0),12:(255,133,27),13:(146,18,49),14:(79,204,48),
    15:(163,86,214),
}


def grid_to_png_b64(grid, scale=5):
    from PIL import Image
    h, w = grid.shape
    img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = ARC_PALETTE.get(int(grid[r,c]), (128,128,128))
    buf = BytesIO()
    Image.fromarray(img).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def blank_board_b64(scale=5):
    """A dark gray 64x64 blank board."""
    grid = np.full((64, 64), 4, dtype=np.int8)  # dark-gray
    return grid_to_png_b64(grid, scale)


def load_state():
    path = os.path.join(STATE_DIR, "session.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_grid(name="current"):
    path = os.path.join(STATE_DIR, f"{name}_grid.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def state_hash():
    """Hash of session file for change detection."""
    path = os.path.join(STATE_DIR, "session.json")
    if not os.path.exists(path):
        return ""
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:8]


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>ARC-AGI-3 — {game_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #111;
            color: #ccc;
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            padding: 15px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0 12px;
            border-bottom: 1px solid #333;
            margin-bottom: 15px;
        }}
        .header h1 {{ color: #ff6b6b; font-size: 1.4em; }}
        .stats {{ display: flex; gap: 25px; font-size: 1.1em; }}
        .stat-value {{ color: #4ecdc4; font-weight: bold; }}
        .levels-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 15px;
        }}
        .level-card {{
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 6px;
            padding: 8px;
            text-align: center;
            min-width: 180px;
        }}
        .level-card.active {{
            border-color: #ff6b6b;
            box-shadow: 0 0 12px rgba(255,107,107,0.3);
        }}
        .level-card.solved {{
            border-color: #4ecdc4;
        }}
        .level-card.future {{
            border-color: #222;
            opacity: 0.5;
        }}
        .level-card img {{
            image-rendering: pixelated;
            border-radius: 3px;
        }}
        .level-label {{
            margin-top: 5px;
            font-size: 0.8em;
        }}
        .level-label.active {{ color: #ff6b6b; }}
        .level-label.solved {{ color: #4ecdc4; }}
        .level-label.future {{ color: #444; }}
        .history {{
            background: #1a1a1a;
            border-radius: 6px;
            padding: 12px;
            max-height: 250px;
            overflow-y: auto;
            font-size: 0.8em;
        }}
        .history h3 {{ color: #666; margin-bottom: 8px; font-size: 0.95em; }}
        .action {{
            padding: 2px 0;
            border-bottom: 1px solid #1f1f1f;
        }}
        .step {{ color: #555; display: inline-block; width: 40px; }}
        .act {{ color: #4ecdc4; display: inline-block; width: 130px; }}
        .diff {{ color: #888; }}
        .no-game {{
            text-align: center;
            margin-top: 80px;
            color: #444;
            font-size: 1.3em;
        }}
    </style>
    <script>
        // Auto-refresh only when state changes
        let lastHash = "{state_hash}";
        async function checkUpdate() {{
            try {{
                const resp = await fetch('/hash');
                const hash = await resp.text();
                if (hash !== lastHash) {{
                    location.reload();
                }}
            }} catch(e) {{}}
            setTimeout(checkUpdate, 500);
        }}
        checkUpdate();
    </script>
</head>
<body>
{content}
</body>
</html>"""


class GameViewerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/hash':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(state_hash().encode())
            return

        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()

            session = load_state()
            if session is None:
                content = '<div class="no-game">No active game.<br>Run: python3 claude_solver.py init &lt;game&gt;</div>'
                self.wfile.write(HTML_TEMPLATE.format(
                    content=content, game_id="waiting", state_hash=""
                ).encode())
                return

            game_id = session.get('game_id', '?')
            step = session.get('step', 0)
            current_level = session.get('levels_completed', 0)
            win_levels = session.get('win_levels', 8)
            game_state = session.get('state', '?')
            observations = session.get('observations', [])
            level_summaries = session.get('level_summaries', [])

            # Load level snapshots
            level_grids = {}
            for ls in level_summaries:
                lvl = ls.get('level', 0)
                grid_path = os.path.join(STATE_DIR, f"level_{lvl}_final.npy")
                if os.path.exists(grid_path):
                    level_grids[lvl] = np.load(grid_path)

            current_grid = load_grid("current")

            parts = []

            # Header
            parts.append(f'''
            <div class="header">
                <h1>🎮 {game_id}</h1>
                <div class="stats">
                    <div>Step <span class="stat-value">{step}</span></div>
                    <div>Level <span class="stat-value">{current_level}/{win_levels}</span></div>
                    <div><span class="stat-value">{game_state}</span></div>
                </div>
            </div>''')

            # Level grid — one card per level
            parts.append('<div class="levels-grid">')
            blank = blank_board_b64(scale=4)

            for lvl in range(win_levels):
                if lvl < current_level:
                    # Solved level
                    if lvl in level_grids:
                        img_b64 = grid_to_png_b64(level_grids[lvl], scale=4)
                    else:
                        img_b64 = blank
                    css_class = "solved"
                    label = f"Level {lvl+1} ✓"
                elif lvl == current_level:
                    # Active level
                    if current_grid is not None:
                        img_b64 = grid_to_png_b64(current_grid, scale=4)
                    else:
                        img_b64 = blank
                    css_class = "active"
                    label = f"Level {lvl+1} ▶ LIVE"
                else:
                    # Future level
                    img_b64 = blank
                    css_class = "future"
                    label = f"Level {lvl+1}"

                parts.append(f'''
                <div class="level-card {css_class}">
                    <img src="data:image/png;base64,{img_b64}" width="256" height="256">
                    <div class="level-label {css_class}">{label}</div>
                </div>''')

            parts.append('</div>')

            # Action history (compact, last 20)
            parts.append('<div class="history"><h3>Recent Actions</h3>')
            for obs in observations[-20:]:
                s = obs.get('step', '?')
                a = obs.get('action', '?')
                d = obs.get('diff', '')[:80]
                parts.append(f'<div class="action">'
                           f'<span class="step">#{s}</span>'
                           f'<span class="act">{a}</span>'
                           f'<span class="diff">{d}</span></div>')
            if not observations:
                parts.append('<div class="action" style="color:#444">Waiting for first action...</div>')
            parts.append('</div>')

            content = '\n'.join(parts)
            self.wfile.write(HTML_TEMPLATE.format(
                content=content, game_id=game_id, state_hash=state_hash()
            ).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    server = http.server.HTTPServer(('', PORT), GameViewerHandler)
    print(f"Game Viewer: http://localhost:{PORT}")
    print(f"State dir: {STATE_DIR}")
    print(f"Updates on every state change (polling 500ms)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
