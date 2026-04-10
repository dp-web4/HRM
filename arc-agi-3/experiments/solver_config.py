#!/usr/bin/env python3
"""
Solver Configuration — unified config dataclass + CLI argument parsing.

Part of SAGE Solver v11 modular architecture.
"""

import argparse
from dataclasses import dataclass


@dataclass
class SolverConfig:
    model: str = ""
    backend: str = "auto"       # ollama, claude, api, auto
    vision: bool = True
    interactive: bool = False
    use_world_model: bool = True
    use_discovery: bool = True
    game: str = None
    all_games: bool = False
    attempts: int = 5
    budget: int = 300
    verbose: bool = False
    viewer: bool = True
    kaggle: bool = False
    identity: str = None        # SAGE instance dir for raising identity
    replay: bool = False        # replay mode: execute known action sequence
    actions: str = ""           # action sequence for replay (space-separated)
    command: str = ""           # interactive subcommand: init, step, look, summarize
    command_args: list = None   # extra args for command (e.g., action, x, y)


def parse_args() -> SolverConfig:
    """Parse CLI arguments into SolverConfig."""
    parser = argparse.ArgumentParser(
        description="SAGE Solver v11 — Unified Modular Architecture")
    parser.add_argument("--game", default=None,
                        help="Game prefix to solve (e.g. lp85, cd82)")
    parser.add_argument("--all", dest="all_games", action="store_true",
                        help="Solve all available games")
    parser.add_argument("--attempts", type=int, default=5,
                        help="Max attempts per game (default: 5)")
    parser.add_argument("--budget", type=int, default=300,
                        help="Max actions per attempt (default: 300)")
    parser.add_argument("--model", default="",
                        help="Ollama model override (auto-detect if empty)")
    parser.add_argument("--backend", default="auto",
                        choices=["ollama", "claude", "api", "auto"],
                        help="Model backend (default: auto)")
    parser.add_argument("--no-vision", dest="vision", action="store_false",
                        help="Disable vision/image features")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode (Claude Code as the model)")
    parser.add_argument("--no-world-model", dest="use_world_model",
                        action="store_false",
                        help="Disable world-model-driven planning")
    parser.add_argument("--no-discovery", dest="use_discovery",
                        action="store_false",
                        help="Disable mechanic discovery (use basic probe)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false",
                        help="Disable session viewer output")
    parser.add_argument("--kaggle", action="store_true",
                        help="Kaggle mode: disable all optional imports")
    parser.add_argument("--replay", action="store_true",
                        help="Replay mode: execute known action sequence")
    parser.add_argument("--actions", default="",
                        help="Action sequence for replay (space-separated: UP DOWN CLICK,32,45)")
    parser.add_argument("--identity", default=None,
                        help="SAGE instance dir for raising identity")
    parser.add_argument("command", nargs="?", default="",
                        help="Interactive subcommand: init, step, look, summarize")
    parser.add_argument("command_args", nargs="*", default=[],
                        help="Args for interactive command (action, x, y)")

    args = parser.parse_args()

    return SolverConfig(
        model=args.model,
        backend="claude" if args.interactive else args.backend,
        replay=args.replay,
        actions=args.actions,
        vision=args.vision,
        interactive=args.interactive,
        use_world_model=args.use_world_model,
        use_discovery=args.use_discovery,
        game=args.game,
        all_games=args.all_games,
        attempts=args.attempts,
        budget=args.budget,
        verbose=args.verbose,
        viewer=args.viewer,
        kaggle=args.kaggle,
        identity=args.identity,
        command=args.command or "",
        command_args=args.command_args or [],
    )
