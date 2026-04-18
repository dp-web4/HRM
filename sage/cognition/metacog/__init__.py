"""
Metacognition — interoceptive monitor for the consciousness loop.

"Does the system know when it's stuck?"

Sits at step 10 (Govern). Observes WM state, SNARC scores, ATP balance,
and action history. Emits MetacogSignals when dysfunction patterns are
detected. Does not take actions — provides a mirror.

Five detectors:
  1. Perseveration — repeated actions, no state change
  2. Intention amnesia — goal doesn't match current action
  3. Stale reference — acting on expired observations
  4. Budget anxiety — ATP burn rate vs completion estimate
  5. Exploration starvation — novelty flatlined, goal incomplete
"""

from .core import Metacog, MetacogConfig, MetacogSignal

__all__ = ["Metacog", "MetacogConfig", "MetacogSignal"]
