"""
Thalamic Router — SAGE's learned dispatch policy.

Spec: shared-context/arc-agi-3/phase2/brain-arch/thalamic-router-prd.md
Sprint plan: shared-context/arc-agi-3/phase2/brain-arch/router-sprint-1-phase-0.md

The router decides, every consciousness-loop tick, what cognition to invoke.
Sits at step 5 (Select) of the 12-step loop.

Phase 0 ships the data pipeline only (this package + `data/`). Training,
inference, and adapter machinery land in later phases.
"""

# Track 1 (schemas) will populate this file with RouterInput/RouterOutput/
# RouterRecord/Event. Track 4 (data pipeline) deliberately does NOT import
# them yet — see `data/_stub.py` for the interim record shape.

__all__: list = []
