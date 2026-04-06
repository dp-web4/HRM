#!/usr/bin/env python3
"""
Persistent Solver — claim one game, solve it, move to next.

Reads game_coordination.json from shared-context to coordinate across
the fleet. Each machine claims the next unclaimed game and persists
until WON or manually stopped.

Protocol:
  1. git pull shared-context
  2. Read game_coordination.json
  3. Claim next unclaimed game (update JSON, commit, push)
  4. Run v7 solver with unlimited attempts until WON
  5. On WON: update coordination as solved, commit, push, goto 2
  6. On stuck (configurable): release claim, skip to next game

Usage:
    cd ~/ai-workspace/SAGE  (or /mnt/c/exe/projects/ai-agents/SAGE)
    PYTHONPATH=".:arc-agi-3/experiments" OLLAMA_MODEL=gemma3:4b \
      python3 arc-agi-3/experiments/persistent_solver.py

    # Skip games you know won't work on this hardware:
    PYTHONPATH=".:arc-agi-3/experiments" OLLAMA_MODEL=gemma3:4b \
      python3 arc-agi-3/experiments/persistent_solver.py --skip-type move

    # Max attempts per game before moving on:
    ... --max-attempts 50
"""

import sys, os, json, time, subprocess, socket

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

COORDINATION_PATH = os.environ.get(
    "COORDINATION_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "shared-context", "arc-agi-3", "game_coordination.json")
)
SHARED_CONTEXT_DIR = os.path.dirname(COORDINATION_PATH)
MACHINE = os.environ.get("SAGE_MACHINE", socket.gethostname().split(".")[0].lower())


def git_sync(repo_dir, message=None):
    """Pull, optionally commit+push."""
    cwd = os.path.dirname(repo_dir) if not os.path.isdir(repo_dir) else repo_dir
    # Go up to shared-context root
    while not os.path.exists(os.path.join(cwd, ".git")) and cwd != "/":
        cwd = os.path.dirname(cwd)

    subprocess.run(["git", "-C", cwd, "pull", "--no-rebase"], capture_output=True, timeout=30)
    if message:
        subprocess.run(["git", "-C", cwd, "add", "-A"], capture_output=True, timeout=10)
        subprocess.run(["git", "-C", cwd, "commit", "-m", message], capture_output=True, timeout=10)
        subprocess.run(["git", "-C", cwd, "push"], capture_output=True, timeout=30)


def load_coordination():
    git_sync(COORDINATION_PATH)
    with open(COORDINATION_PATH) as f:
        return json.load(f)


def save_coordination(data, message):
    data["_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(COORDINATION_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    git_sync(COORDINATION_PATH, message)


def claim_game(data, game_family):
    """Mark a game as claimed by this machine."""
    for g in data["games"]:
        if g["family"] == game_family:
            g["status"] = "claimed"
            g["claimed_by"] = MACHINE
            g["claimed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_coordination(data, f"[{MACHINE}] Claim {game_family}")


def mark_solved(data, game_family, levels):
    """Mark a game as solved."""
    for g in data["games"]:
        if g["family"] == game_family:
            g["status"] = "solved"
            g["solved"] = True
            g["solved_by"] = MACHINE
            g["solved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            g["best_level"] = levels
    save_coordination(data, f"[{MACHINE}] SOLVED {game_family}!")


def update_progress(data, game_family, best_level):
    """Update best level for a game."""
    for g in data["games"]:
        if g["family"] == game_family:
            if best_level > g.get("best_level", 0):
                g["best_level"] = best_level
                save_coordination(data, f"[{MACHINE}] {game_family} progress: L{best_level}")


def release_game(data, game_family):
    """Release claim on a game (giving up or moving on)."""
    for g in data["games"]:
        if g["family"] == game_family:
            g["status"] = "unclaimed"
            g["claimed_by"] = None
    save_coordination(data, f"[{MACHINE}] Release {game_family}")


def pick_next_game(data, skip_types=None):
    """Pick the next unclaimed, unsolved game."""
    skip_types = skip_types or []
    for g in data["games"]:
        if g["solved"]:
            continue
        if g["status"] == "claimed" and g["claimed_by"] == MACHINE:
            return g  # resume our own claimed game
        if g["status"] == "claimed":
            continue  # someone else has it
        if g["type"] in skip_types:
            continue
        return g
    return None


def run_solver(game_id, max_attempts=0, budget=500, verbose=True):
    """Run v7 solver on a single game. Returns (best_levels, won)."""
    # Import here to avoid circular deps at module level
    try:
        from sage_solver_v7 import solve_game, MODEL
        from arc_agi import Arcade
    except ImportError:
        # Fall back to v6 if v7 not available
        try:
            from sage_solver_v6 import solve_game, MODEL
            from arc_agi import Arcade
        except ImportError:
            print(f"  ERROR: No solver available (v7 or v6)")
            return 0, False

    arcade = Arcade()
    prefix = game_id.split("-")[0]
    best_overall = 0
    attempt = 0

    while True:
        attempt += 1
        if max_attempts > 0 and attempt > max_attempts:
            print(f"\n  Max attempts ({max_attempts}) reached for {prefix}")
            break

        print(f"\n{'─'*50}")
        print(f"  {prefix} — Attempt {attempt}" +
              (f"/{max_attempts}" if max_attempts > 0 else " (unlimited)"))

        try:
            result = solve_game(arcade, game_id, max_attempts=1,
                                budget=budget, verbose=verbose)
            levels = result.get("best_levels", 0)
            win_levels = result.get("win_levels", 0)

            if levels > best_overall:
                best_overall = levels
                print(f"\n  ★ NEW BEST: L{levels}/{win_levels}")

            if result.get("won") or (win_levels > 0 and levels >= win_levels):
                print(f"\n  ★★★ SOLVED {prefix}: {levels}/{win_levels} ★★★")
                return levels, True

        except Exception as e:
            print(f"  Attempt {attempt} error: {e}")
            time.sleep(2)

    return best_overall, False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Persistent Solver — claim, solve, next")
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="Max attempts per game (0=unlimited)")
    parser.add_argument("--budget", type=int, default=500,
                        help="Action budget per attempt")
    parser.add_argument("--skip-type", type=str, action="append", default=[],
                        help="Skip game types (move, click, move+click)")
    parser.add_argument("--game", type=str, default=None,
                        help="Force a specific game instead of picking from coordination")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Persistent Solver — {MACHINE}")
    print(f"Budget: {args.budget}/attempt | Max attempts: {args.max_attempts or 'unlimited'}")
    print(f"Skip types: {args.skip_type or 'none'}")
    print(f"Coordination: {COORDINATION_PATH}")
    print("=" * 60)

    # Warmup LLM
    print("\nWarming up LLM...", end=" ", flush=True)
    try:
        from sage_solver_v7 import ask_llm, MODEL
        t = ask_llm("ready")
        print(f"OK ({MODEL})")
    except Exception as e:
        print(f"WARN: {e}")

    games_solved = 0

    while True:
        # Load fresh coordination
        data = load_coordination()

        if args.game:
            # Force specific game
            game = None
            for g in data["games"]:
                if args.game in g.get("game_id", g.get("id", "")) or args.game == g["family"]:
                    game = g
                    break
            if not game:
                print(f"Game '{args.game}' not found")
                return
        else:
            game = pick_next_game(data, skip_types=args.skip_type)

        if game is None:
            print("\n" + "=" * 60)
            print(f"ALL GAMES CLAIMED OR SOLVED! {games_solved} solved this session.")
            print("=" * 60)
            break

        family = game["family"]
        game_id = game.get("game_id", game.get("id", family))
        baseline = game.get("baseline_budget", game.get("baseline", 500))

        print(f"\n{'█' * 60}")
        print(f"  CLAIMING: {family} ({game.get('type','?')}, baseline={baseline})")
        print(f"{'█' * 60}")

        # Claim it
        claim_game(data, family)

        # Solve it
        best_level, won = run_solver(
            game_id,
            max_attempts=args.max_attempts,
            budget=args.budget,
            verbose=args.verbose,
        )

        # Update coordination
        data = load_coordination()  # refresh
        if won:
            mark_solved(data, family, best_level)
            games_solved += 1
            print(f"\n  ★★★ {family} SOLVED — moving to next game ★★★")
        else:
            update_progress(data, family, best_level)
            release_game(data, family)
            print(f"\n  Released {family} (best: L{best_level})")

        if args.game:
            break  # only run the forced game

    print(f"\nSession complete. {games_solved} games solved.")


if __name__ == "__main__":
    main()
