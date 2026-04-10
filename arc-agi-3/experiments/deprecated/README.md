# Deprecated Solvers

These files are archived for reference. **DO NOT USE OR MODIFY.**

The canonical solver is `sage_solver.py` (v11) with three modes:
- `--interactive` — Claude Code as the reasoning model
- Default — autonomous with local LLM (e.g., gemma, qwen)
- `--kaggle` — competition mode (no optional imports)

All state goes to `/tmp/sage_solver/`. The viewer reads from there.

## What Was Replaced

| Old File | Replaced By | Notes |
|----------|-------------|-------|
| `claude_solver.py` | `solver_loop.py:interactive_step()` | Interactive mode |
| `sage_solver_v7.py` | `sage_solver.py` v11 | Autonomous mode |
| `sage_solver_v9.py` | `sage_solver.py` v11 + vision | Multimodal mode |
| `sage_solver_v10.py` | `sage_solver.py` v11 | World model mode |
| Others | Various | Earlier iterations |

## Color Palette

ONE palette everywhere: the SDK's (`arc_agi/rendering.py`):
```
0=white, 5=black, 11=yellow, 14=green
```

If you see wrong colors, check which palette the code uses. The deprecated files had multiple wrong palettes that caused mismatched color descriptions.
