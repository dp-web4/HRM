#!/usr/bin/env python3
"""
SAGE Solver v11 — Unified Modular Architecture.

CLI entry point that unifies autonomous and interactive solving modes
with modular backends, 4-layer context, world-model planning, mechanic
discovery, and fleet federation.

Usage:
    # Autonomous (Ollama backend)
    python3 sage_solver.py --game lp85 -v
    python3 sage_solver.py --all --attempts 5 --budget 300

    # Specific model
    python3 sage_solver.py --game cd82 --model gemma4:e4b -v

    # Interactive (Claude Code as model)
    python3 sage_solver.py --interactive --game cd82

    # Kaggle mode (no optional imports)
    python3 sage_solver.py --kaggle --all --no-vision --no-discovery

    # Disable world-model planning (use context-plan mode)
    python3 sage_solver.py --game lp85 --no-world-model -v
"""

import sys
import os

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")
sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade

from solver_config import parse_args
from model_backend import OllamaBackend, ClaudeInteractiveBackend
from solver_loop import solve_game_autonomous, solve_game_interactive, interactive_step


def main():
    config = parse_args()

    # ── Create Backend ──
    if config.interactive or config.backend == "claude":
        backend = ClaudeInteractiveBackend()
    else:
        backend = OllamaBackend(
            model=config.model if config.model else "")

    # ── Banner ──
    print("=" * 60)
    print(f"SAGE Solver v11 — Unified Modular Architecture "
          f"({backend.model_name()})")
    thinking = "native" if backend.supports_thinking() else "disabled"
    vision = "on" if config.vision and backend.supports_vision() else "off"
    wm = "on" if config.use_world_model else "off"
    disc = "on" if config.use_discovery else "off"
    print(f"Thinking: {thinking} | Vision: {vision} | "
          f"World-model: {wm} | Discovery: {disc}")
    print("=" * 60)

    # ── Warmup ──
    if not config.interactive:
        print("\nWarming up...", end=" ", flush=True)
        ok = backend.warmup()
        print("OK" if ok else "WARN: model may not be ready")

    # ── Interactive Mode ──
    if config.interactive:
        cmd = config.command.lower() if config.command else "init"
        if cmd == "init":
            game = config.game or (config.command_args[0] if config.command_args else "cd82")
            result = solve_game_interactive(game, backend, config)
            if result.get("game_id"):
                print(f"\nUse: sage_solver.py --interactive step <action> [x] [y]")
        elif cmd == "step":
            args = config.command_args
            if not args:
                print("Usage: sage_solver.py --interactive step <action> [x] [y]")
                return
            action = int(args[0])
            x = int(args[1]) if len(args) > 1 else None
            y = int(args[2]) if len(args) > 2 else None
            interactive_step(action, x, y)
        else:
            print(f"Unknown interactive command: {cmd}")
            print("Commands: init, step")
        return

    # ── Autonomous Mode ──
    arcade = Arcade()
    envs = arcade.get_environments()

    if config.game:
        targets = [e for e in envs if config.game in e.game_id]
    elif config.all_games:
        targets = envs
    else:
        targets = envs[:5]

    print(f"\n{len(targets)} games, {config.attempts} attempts, "
          f"budget={config.budget}\n")

    total_levels = 0
    scored = {}

    for env_info in targets:
        game_id = env_info.game_id
        prefix = game_id.split("-")[0]
        try:
            result = solve_game_autonomous(
                arcade, game_id, backend, config)
        except Exception as e:
            if config.verbose:
                import traceback
                traceback.print_exc()
            print(f"  {prefix}: CRASHED ({e})")
            continue

        levels = result["best_levels"]
        total_levels += levels
        if levels > 0:
            scored[prefix] = levels
            print(f"  * {prefix:6s}  {levels}/{result['win_levels']}")
        else:
            print(f"    {prefix:6s}  0/{result['win_levels']}")

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_levels} levels across {len(scored)} games")
    if scored:
        for g, l in sorted(scored.items(), key=lambda x: -x[1]):
            print(f"  {g}: {l}")


if __name__ == "__main__":
    main()
