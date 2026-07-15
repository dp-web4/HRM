# MEMBRANE: adaptations live in the playground, mechanisms live here

This package ships **general cognitive-routing mechanisms** — SNARC-driven mode switching, faith
calibration, situation reports, goal signals. It is environment-agnostic by design.

Adaptations learned from specific environments — per-environment calibration constants,
world-model instances, environment-ID test fixtures — live in the **private playground repo** as
loadable artifacts, not here. Where a general mechanism still has an environment-tuned constant
baked in, read that as a TODO toward generalization (the constant should become loaded
example-data), not as design intent.

The standing principle: **environments are the playground — examples and lessons that feed a
general organism — not the ground truth the organism is built around.** Getting good at the
environments we practice in is not the goal; getting good in general is.
