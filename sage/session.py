#!/usr/bin/env python3
"""SAGE unified session launcher — one CLI, two modes.

Usage:
    python3 -m sage.session --raising [raising args...]
    python3 -m sage.session --play    [gameplay args...]

Examples:

    # Raising session (identity development)
    python3 -m sage.session --raising --session 100
    python3 -m sage.session --raising --fluid --session 99

    # Gameplay (ARC-AGI-3 game)
    python3 -m sage.session --play \\
        --adapter /path/to/shared-context/arc-agi-3/phase2/adapters/cbp-framerouter-v7sub-2026-04-19 \\
        --game ft09 --max-steps 2000 \\
        --llm-backend claude_cli --llm-model sonnet

## Design

Thin dispatcher that delegates to the existing per-mode harnesses after
parsing the mode flag. Each mode uses:

- **Raising** (`--raising`): `sage.raising.scripts.run_session_identity_anchored`
  (or `_fluid` variant with `--fluid`). Identity anchoring, partnership
  framing, ExperienceCollector, snapshots. DaemonIRP backend.

- **Gameplay** (`--play`): `sage.cognition.thalamic_router.llm_dispatch`.
  Three-party conversation, FrameRouter invoke head, CNN narration with
  metacog + episodic recall, RouterDatasetWriter telemetry. LLMClient
  backends (Ollama / Claude CLI / Anthropic API).

## Shared infrastructure (Phase 4 P1)

Both modes now share these pieces, even though their harnesses differ:

- `sage.irp.unified_history` — schema converter (raising {speaker,text} ↔
  gameplay {role,content})
- `sage.irp.plugins.llm_client_irp` — `LLMClientIRP` wraps any `LLMClient`
  in the IRP contract; swap it in for `DaemonIRP` to use gameplay LLM
  backends from raising
- `sage.cognition.metacog.core.Metacog` — same detectors, both modes
  call `observe_tick()` (raising per turn, gameplay per step)
- Workspace path resolution — gameplay uses `_resolve_router_data_dir()`
  to avoid machine-specific hardcoding

## Artifacts stay siloed

Per Dennis: raising artifacts (identity, snapshots, experience buffer)
stay in `sage/instances/{slug}/`. Gameplay artifacts (delta records,
dispatch records) stay in `private-context/training-data/router/{machine}/`.
Fleet writeups in `shared-context/arc-agi-3/fleet-learning/{machine}/`.
Different purposes, different consumers, correct as-is.

## Framing

Raising is collaborative — partnership with Claude, fleet context,
identity-across-sessions. Gameplay is solitary — "the sole gladiator in
the arena." No federation context bleeds into gameplay. Each session has
its own mental model.

Sprint: Phase 4 P2 (alignment — unified launcher)
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _dispatch_raising(remaining_argv: List[str], fluid: bool) -> int:
    """Delegate to the raising harness (legacy or fluid variant)."""
    # Swap argv so the delegated main() sees only its own flags
    if fluid:
        from sage.raising.scripts import run_session_identity_anchored_fluid as mod
        sys.argv = ["run_session_identity_anchored_fluid"] + remaining_argv
    else:
        from sage.raising.scripts import run_session_identity_anchored as mod
        sys.argv = ["run_session_identity_anchored"] + remaining_argv
    return mod.main() or 0


def _dispatch_play(remaining_argv: List[str]) -> int:
    """Delegate to the gameplay dispatch harness."""
    from sage.cognition.thalamic_router import llm_dispatch as mod
    sys.argv = ["llm_dispatch"] + remaining_argv
    return mod.main() or 0


LAUNCHER_HELP = """usage: sage.session (--raising | --play) [--fluid] [mode-specific args...]

SAGE unified session launcher

Modes:
  --raising     Run a raising session (identity development track)
                Delegates to run_session_identity_anchored.
  --play        Run a gameplay session (ARC-AGI-3)
                Delegates to llm_dispatch.

Options:
  --fluid       (Raising only) Use the _fluid runner variant with
                Thor S86 mitigations for self-quotation feedback.
  -h, --help    Show this launcher help.
                Use `--raising --help` or `--play --help` to see
                mode-specific flags from the delegated harness.

Examples:
  python3 -m sage.session --raising --session 100
  python3 -m sage.session --raising --fluid --session 99
  python3 -m sage.session --play --adapter <path> --game ft09 \\
      --max-steps 2000 --llm-backend claude_cli --llm-model sonnet
"""


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Handle top-level help ourselves so mode --help can pass through
    # to the delegated harness.
    if not argv or (len(argv) == 1 and argv[0] in ("-h", "--help")):
        print(LAUNCHER_HELP)
        return 0

    # First flag should be --raising or --play (possibly after --fluid in
    # the unlikely order, but normally mode comes first).
    mode_idx = None
    mode = None
    fluid = False
    for i, a in enumerate(argv):
        if a == "--raising":
            mode = "raising"; mode_idx = i
            break
        if a == "--play":
            mode = "play"; mode_idx = i
            break
    if mode is None:
        print(LAUNCHER_HELP, file=sys.stderr)
        print("error: must specify --raising or --play", file=sys.stderr)
        return 2

    if "--fluid" in argv:
        fluid = True

    # Strip launcher flags from argv, leave everything else for delegation
    remaining = [a for a in argv if a not in ("--raising", "--play", "--fluid")]

    if mode == "raising":
        return _dispatch_raising(remaining, fluid=fluid)
    else:  # play
        if fluid:
            print("error: --fluid is only valid with --raising", file=sys.stderr)
            return 2
        return _dispatch_play(remaining)


if __name__ == "__main__":
    sys.exit(main())
