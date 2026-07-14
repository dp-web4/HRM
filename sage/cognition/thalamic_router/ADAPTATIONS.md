# Read this before reading the game IDs: examples and lessons, not ground truth

This package is **general cognitive-routing machinery** — SNARC-driven mode switching, faith
calibration, situation reports, goal signals. It is meant to be **game-agnostic**.

You will nonetheless see specific game IDs throughout the code and tests — `ar25`, `dc22`, `ka59`,
`r11l`, `wa30`, and others — and per-game constants like `TRUSTED = {"ka59": 1.31, "r11l": 0.47,
"re86": 1.03}`. **These are examples and lessons, not canon.** Two things to hold:

1. **What they refer to.** Those IDs are the *public* ARC-AGI-3 games (public tasks, publicly known
   win sequences). The competition's *scored* set is secret — and secret to us as well; we don't have
   it. So nothing here is a leaked solution. They are worked examples against public games.

2. **What they are — and aren't.** Per-game calibration values, per-game world-model instances, and
   game-ID test fixtures are **adaptations of the general machinery to specific example games** — the
   lessons the playground taught, frozen into constants for illustration and regression tests. They
   are **not** the canonical parameters of the architecture, and the architecture is not "correct"
   because it reproduces them. Where a general mechanism has a game-tuned constant baked in, read that
   as a TODO toward generalization (the constant should become loaded example-data, not hardcoded
   ground truth), not as the design intent.

The standing principle, applied to ourselves as much as to any reader: **games are the playground —
examples and lessons that feed a general organism — not the ground truth the organism is built
around.** Getting good at the games we play is not the goal; getting good at games in general is. If a
future reader (or a future us) mistakes the per-game constants here for the thesis, this note failed.
