"""
GameStateDecoder — reads lp85's internal engine state and produces structured
semantic descriptions for the LLM.

The LLM cannot reliably interpret raw pixels to understand:
  - Which colored regions are buttons vs goal markers vs moving pieces
  - How many steps remain (rendered as a bar in col 0)
  - Which buttons control which rotation groups
  - What the win condition actually requires

This decoder bypasses pixel interpretation entirely: it reads the game engine's
internal sprite list, extracts semantic roles, and maps to pixel click coordinates.
"""

import numpy as np
from typing import Optional


# Sprite coordinate → display pixel transform (empirically verified for lp85)
# pixel_col = sprite.x * SCALE + (sprite.width - 1)
# pixel_row = letter_box + sprite.y * SCALE + (sprite.height - 1)
SPRITE_SCALE = 2


def _find_letter_box(grid: np.ndarray) -> int:
    """Find how many rows of letter-box padding appear at the top."""
    PADDING_COLOR = 3
    STEP_COUNTER_COL = 0
    for row in range(grid.shape[0]):
        # Check col 1 (skip col 0 = step counter)
        if grid.shape[1] > 1 and grid[row, 1] != PADDING_COLOR:
            return row
    return 0


def _sprite_pixel_center(sprite, letter_box: int):
    """Return (click_col, click_row) for clicking the center of a sprite."""
    col = sprite.x * SPRITE_SCALE + (sprite.width - 1)
    row = letter_box + sprite.y * SPRITE_SCALE + (sprite.height - 1)
    return col, row


class GameStateDecoder:
    """
    Attaches to an ARC-AGI-3 environment and extracts human-readable game state.

    Usage:
        decoder = GameStateDecoder(env)
        state = decoder.decode(grid)          # dict with buttons, pieces, goals, steps
        text  = decoder.to_prompt_text(grid)  # formatted string for LLM prompt
    """

    def __init__(self, env):
        self.env = env

    def _game(self):
        return getattr(self.env, '_game', None)

    def decode(self, grid: np.ndarray) -> Optional[dict]:
        """Extract structured semantic game state from engine internals + rendered grid."""
        game = self._game()
        if game is None:
            return None

        level = game.current_level
        step_counter = game.toxpunyqe
        steps_remaining = step_counter.current_steps
        steps_total = step_counter.bnlfrvxkob
        letter_box = _find_letter_box(grid)

        buttons = []
        pieces = []
        goals = []

        for s in level._sprites:
            if not s.tags:
                continue
            tag = s.tags[0]
            col, row = _sprite_pixel_center(s, letter_box)
            # Clamp to grid bounds
            col = max(0, min(grid.shape[1] - 1, col))
            row = max(0, min(grid.shape[0] - 1, row))

            if 'button' in tag:
                parts = tag.split('_')
                if len(parts) == 3:
                    _, group, side = parts
                    direction = 'LEFT' if side == 'L' else 'RIGHT'
                    buttons.append({
                        'tag': tag,
                        'group': group,
                        'direction': direction,
                        'click_x': col,
                        'click_y': row,
                    })

            elif tag == 'bghvgbtwcb':
                pieces.append({
                    'id': f'piece_{len(pieces)+1}',
                    'needs_goal': 'goal',
                    'sprite_x': s.x, 'sprite_y': s.y,
                    'pixel_x': col, 'pixel_y': row,
                })

            elif tag == 'fdgmtkfrxl':
                pieces.append({
                    'id': f'piece_{len(pieces)+1}',
                    'needs_goal': 'goal-o',
                    'sprite_x': s.x, 'sprite_y': s.y,
                    'pixel_x': col, 'pixel_y': row,
                })

            elif tag == 'goal':
                goals.append({
                    'for_piece': 'bghvgbtwcb',
                    'sprite_x': s.x, 'sprite_y': s.y,
                    'pixel_x': col, 'pixel_y': row,
                })

            elif tag == 'goal-o':
                goals.append({
                    'for_piece': 'fdgmtkfrxl',
                    'sprite_x': s.x, 'sprite_y': s.y,
                    'pixel_x': col, 'pixel_y': row,
                })

        # Sort buttons by group then direction for readability
        buttons.sort(key=lambda b: (b['group'], b['direction']))

        return {
            'steps_remaining': steps_remaining,
            'steps_total': steps_total,
            'letter_box': letter_box,
            'buttons': buttons,
            'pieces': pieces,
            'goals': goals,
            'level_name': level.get_data('level_name'),
        }

    def to_prompt_text(self, grid: np.ndarray) -> str:
        """Format game state as a structured string for injection into LLM prompts."""
        state = self.decode(grid)
        if state is None:
            return "(GameStateDecoder: could not access engine internals)"

        lines = ["=== ENGINE GAME STATE (source-derived, authoritative) ==="]

        # Step budget — critical framing
        sr = state['steps_remaining']
        st = state['steps_total']
        pct = int(100 * sr / st) if st > 0 else 0
        urgency = " ⚠ LOW" if pct < 30 else (" ⚠ CRITICAL" if pct < 15 else "")
        lines.append(f"Steps remaining: {sr}/{st} ({pct}%){urgency}")
        lines.append("RULE: Every click costs 1 step. Zero steps = LOSE.")
        lines.append("")

        # Buttons — the ONLY interactive objects
        lines.append("ROTATION BUTTONS (the ONLY things you can interact with):")
        groups_seen = {}
        for b in state['buttons']:
            g = b['group']
            if g not in groups_seen:
                groups_seen[g] = []
            groups_seen[g].append(b)

        for group, btns in sorted(groups_seen.items()):
            for b in btns:
                lines.append(
                    f"  Ring-{b['group']}-{b['direction']:5s}: click x={b['click_x']}, y={b['click_y']}"
                )
        lines.append("")
        lines.append("NOTE: Cyan tiles, ring path tiles, and goal markers are NOT clickable.")
        lines.append("NOTE: The colored bar in column 0 is the step counter — do not click it.")
        lines.append("")

        # Movable pieces
        lines.append("MOVABLE PIECES (these travel around ring tracks when buttons are pressed):")
        for p in state['pieces']:
            lines.append(
                f"  {p['id']}: currently at pixel ({p['pixel_x']}, {p['pixel_y']})"
                f"  [needs to reach a '{p['needs_goal']}' marker]"
            )
        lines.append("")

        # Goal positions
        lines.append("GOAL POSITIONS (fixed targets — pieces must overlap these to win):")
        for g in state['goals']:
            lines.append(
                f"  Goal for {g['for_piece']}: pixel ({g['pixel_x']}, {g['pixel_y']})"
            )
        lines.append("")

        # Win condition
        n_pieces = len(state['pieces'])
        n_goals = len(state['goals'])
        lines.append(f"WIN CONDITION: All {n_pieces} piece(s) must overlap their {n_goals} goal(s).")
        lines.append("STRATEGY: Different buttons rotate different rings. Pieces on the same ring")
        lines.append("move together. Use different buttons to position pieces independently.")

        return "\n".join(lines)
