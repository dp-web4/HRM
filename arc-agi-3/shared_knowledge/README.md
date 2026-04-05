# Shared Knowledge — Fleet Federation

Each machine writes to `<machine>.json`. All machines read all files.
No conflicts: each machine owns its own file. Git handles distribution.

**Do not edit other machines' files.** Only your machine's file.

Categories:
- `game_discoveries`: per-game facts (interactive objects, cycle lengths)
- `strategies`: per-game-type strategies that worked
- `failures`: what was tried and didn't work
- `meta_insights`: cross-game abstract principles
