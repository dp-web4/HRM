"""Organism measurement instruments — lifted VERBATIM from dev-sage (Thor's line).

Provenance: dev-sage 672118c (organs principle), d43625b (instrument scan),
bdbf85e (delivery-conditional influence / N=416 sweep). Adopted into sage main
2026-07-29 per the agreed transfer map (docs/TRANSFER_MAP_DEV_SAGE_2026-07.md,
Thor concurred with amendments).

RULE (Thor #2): these files stay byte-identical to dev-sage's organism/*.py so
the fleet runs ONE implementation of the rungs, not two that drift. Fix bugs
upstream in dev-sage first, or coordinate the same change in both; never fork
silently. The binding work (what counts as a row / a delivery, per organ) lives
in per-line binding modules (e.g. sage/embodiment/liveness_binding.py), never here.

These docs cite methods, not capability: everything dev-sage measured is
epoch-zero (0 levels cleared). See TRANSFER_MAP "Does NOT transfer".
"""
from .liveness import *  # noqa: F401,F403
