#!/usr/bin/env python3
"""
Game Viewer — localhost web server showing board state in real-time.

Serves a page at http://localhost:8765 that auto-refreshes to show:
- Current game frame (correct ARC palette colors)
- Step counter, level, game state
- Action history with semantic diffs
- Side-by-side initial vs current state

Updates automatically when claude_solver writes new state.

Usage:
    python3 game_viewer.py
    # Then open http://localhost:8765 in browser
"""

import http.server
import json
import os
import time
import numpy as np
from io import BytesIO
import base64
from pathlib import Path

STATE_DIR = "/tmp/claude_solver"
PORT = 8765

# Correct ARC-AGI-3 palette
ARC_PALETTE = {
    0: (255,255,255), 1: (204,204,204), 2: (153,153,153), 3: (102,102,102),
    4: (51,51,51), 5: (0,0,0), 6: (229,58,163), 7: (255,123,204),
    8: (249,60,49), 9: (30,147,255), 10: (136,216,241), 11: (255,220,0),
    12: (255,133,27), 13: (146,18,49), 14: (79,204,48), 15: (163,86,214),
}
COLOR_NAMES = {
    0:"white",1:"off-white",2:"light-gray",3:"gray",4:"dark-gray",5:"black",
    6:"magenta",7:"pink",8:"red",9:"blue",10:"light-blue",11:"yellow",
    12:"orange",13:"maroon",14:"green",15:"purple"
}


def grid_to_png_b64(grid, scale=6):
    """Render grid as base64 PNG with correct colors."""
    from PIL import Image
    h, w = grid.shape
    img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = ARC_PALETTE.get(int(grid[r,c]), (128,128,128))
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def load_state():
    """Load current game state."""
    session_path = os.path.join(STATE_DIR, "session.json")
    if not os.path.exists(session_path):
        return None
    with open(session_path) as f:
        return json.load(f)


def load_grid(name="current"):
    path = os.path.join(STATE_DIR, f"{name}_grid.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>ARC-AGI-3 Game Viewer</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body {{
            background: #1a1a1a;
            color: #e0e0e0;
            font-family: 'SF Mono', 'Fira Code', monospace;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        .header h1 {{
            margin: 0;
            color: #ff6b6b;
        }}
        .stats {{
            display: flex;
            gap: 30px;
            font-size: 1.2em;
        }}
        .stat-value {{
            color: #4ecdc4;
            font-weight: bold;
        }}
        .boards {{
            display: flex;
            gap: 30px;
            margin-bottom: 20px;
        }}
        .board {{
            text-align: center;
        }}
        .board img {{
            border: 2px solid #333;
            border-radius: 4px;
            image-rendering: pixelated;
        }}
        .board-label {{
            margin-top: 8px;
            color: #888;
            font-size: 0.9em;
        }}
        .history {{
            background: #222;
            border-radius: 8px;
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .history h3 {{
            margin-top: 0;
            color: #888;
        }}
        .action {{
            padding: 4px 0;
            border-bottom: 1px solid #2a2a2a;
            font-size: 0.85em;
        }}
        .action .step {{
            color: #666;
            display: inline-block;
            width: 50px;
        }}
        .action .act {{
            color: #4ecdc4;
            display: inline-block;
            width: 150px;
        }}
        .action .diff {{
            color: #aaa;
        }}
        .level-up {{
            color: #ff6b6b !important;
            font-weight: bold;
        }}
        .no-game {{
            text-align: center;
            margin-top: 100px;
            color: #666;
            font-size: 1.5em;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>"""


class GameViewerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()

            session = load_state()
            if session is None:
                content = '<div class="no-game">No active game. Run claude_solver.py init &lt;game&gt;</div>'
                self.wfile.write(HTML_TEMPLATE.format(content=content).encode())
                return

            current_grid = load_grid("current")

            # Build content
            parts = []

            # Header
            game_id = session.get('game_id', '?')
            step = session.get('step', 0)
            levels = session.get('levels_completed', 0)
            win_levels = session.get('win_levels', '?')
            state = session.get('state', '?')

            parts.append(f'''
            <div class="header">
                <h1>🎮 {game_id}</h1>
                <div class="stats">
                    <div>Step: <span class="stat-value">{step}</span></div>
                    <div>Level: <span class="stat-value">{levels}/{win_levels}</span></div>
                    <div>State: <span class="stat-value">{state}</span></div>
                </div>
            </div>
            ''')

            # Board images
            parts.append('<div class="boards">')
            if current_grid is not None:
                b64 = grid_to_png_b64(current_grid)
                parts.append(f'''
                <div class="board">
                    <img src="data:image/png;base64,{b64}" width="384" height="384">
                    <div class="board-label">Current State (Step {step})</div>
                </div>
                ''')
            parts.append('</div>')

            # Action history
            observations = session.get('observations', [])
            level_summaries = session.get('level_summaries', [])

            parts.append('<div class="history">')
            parts.append('<h3>Action History</h3>')

            for obs in observations[-30:]:  # last 30
                s = obs.get('step', '?')
                a = obs.get('action', '?')
                d = obs.get('diff', '')[:100]
                lvl = obs.get('levels', 0)
                cls = 'level-up' if 'LEVEL' in str(d).upper() else ''
                parts.append(f'<div class="action {cls}">'
                           f'<span class="step">#{s}</span>'
                           f'<span class="act">{a}</span>'
                           f'<span class="diff">{d}</span>'
                           f'</div>')

            if not observations:
                parts.append('<div class="action">No actions yet. Waiting for claude_solver...</div>')

            parts.append('</div>')

            # Level summaries
            if level_summaries:
                parts.append('<div class="history" style="margin-top:15px">')
                parts.append('<h3>Level Summaries</h3>')
                for s in level_summaries:
                    parts.append(f'<div class="action">Level {s.get("level","?")}: '
                               f'{s.get("steps","?")} steps, {s.get("wasted","?")} wasted</div>')
                parts.append('</div>')

            content = '\n'.join(parts)
            self.wfile.write(HTML_TEMPLATE.format(content=content).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress access logs


def main():
    server = http.server.HTTPServer(('', PORT), GameViewerHandler)
    print(f"Game Viewer running at http://localhost:{PORT}")
    print(f"Watching state dir: {STATE_DIR}")
    print(f"Auto-refreshes every 2 seconds. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
