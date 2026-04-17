"""
ARC-AGI-3 game domain adapter for the cerebellum.

Provides:
- State signature extraction from game environments
- Action replay on ARC arcade environments
- Feature extraction from game sprites and state
"""

from sage.cognition.cerebellum.core import (
    Cerebellum,
    StateSignature,
    Habit,
)

# GameAction int→name mapping
ACTION_NAMES = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'SELECT', 6: 'CLICK', 7: 'UNDO'}


def arc_state_signature(game_id: str, level: int, available_actions: list,
                        sprite_count: int = 0, sprite_colors: tuple = ()) -> StateSignature:
    """Build a state signature from an ARC game environment.

    Features are chosen to be stable within a level but vary across levels:
    - game_id: identifies the game
    - level: identifies the level within the game
    - action_set: what actions are available (game type)
    - sprite_count: rough structural signature
    - sprite_colors: what colors are present
    """
    return StateSignature(
        domain=f"arc-game:{game_id.split('-')[0]}",
        features={
            "game_id": game_id,
            "level": level,
            "action_set": tuple(sorted(available_actions)),
            "sprite_count": sprite_count,
            "sprite_colors": sprite_colors,
        },
    )


def extract_state_from_env(env) -> StateSignature:
    """Extract a state signature from a live ARC arcade environment."""
    fd = env._last_fd if hasattr(env, '_last_fd') else None
    game = env._game

    game_id = env.game_id if hasattr(env, 'game_id') else str(game.__class__.__name__)
    level = fd.levels_completed if fd else 0
    available = list(fd.available_actions) if fd else []

    # Extract sprite info
    sprite_count = 0
    sprite_colors = set()
    try:
        for s in game.current_level.get_sprites():
            if s.is_visible and s.width > 0:
                sprite_count += 1
                c = int(s.pixels[min(1, s.height - 1), min(1, s.width - 1)])
                if c >= 0:
                    sprite_colors.add(c)
    except Exception:
        pass

    return arc_state_signature(
        game_id=game_id,
        level=level,
        available_actions=available,
        sprite_count=sprite_count,
        sprite_colors=tuple(sorted(sprite_colors)),
    )


class ArcCerebellum(Cerebellum):
    """Cerebellum specialized for ARC-AGI-3 game environments.

    Overrides _replay_actions to execute on arcade environments.
    """

    def _replay_actions(self, actions: list, env) -> dict:
        """Replay an action sequence on an ARC arcade environment.

        Each action is a dict with:
          action: int (GameAction value)
          data: optional dict with x, y for CLICK
        """
        from arcengine.enums import GameAction

        initial_level = env._last_fd.levels_completed if hasattr(env, '_last_fd') else 0
        fd = None

        for step in actions:
            action_int = step.get("action", step) if isinstance(step, dict) else step
            data = step.get("data", {}) if isinstance(step, dict) else {}

            ga = getattr(GameAction, f'ACTION{action_int}')
            if data:
                fd = env.step(ga, data=data)
            else:
                fd = env.step(ga)

        if fd is None:
            return {"success": False, "reason": "no actions"}

        won = fd.levels_completed > initial_level
        return {
            "success": won,
            "levels_completed": fd.levels_completed,
            "actions_replayed": len(actions),
        }
