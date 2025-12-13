"""
Coherent Awakening Protocol for SAGE

This module provides session-to-session continuity for SAGE,
implementing Phase 0 of BECOMING_CURRICULUM.md.

Usage:
    from sage.awakening import CoherentAwakening, prepare_session, end_session

    # Option 1: Full control
    awakening = CoherentAwakening()
    coherence_field = awakening.prepare_coherence_field()
    preamble = awakening.create_boot_preamble(coherence_field)
    sage = awakening.coherent_boot(coherence_field)
    # ... session ...
    awakening.coherent_end(sage, memory_request="...")

    # Option 2: Convenience functions
    awakening, coherence_field, preamble = prepare_session()
    sage = awakening.coherent_boot(coherence_field)
    # ... session ...
    end_session(awakening, sage, memory_request="...")
"""

from .coherent_awakening import (
    CoherentAwakening,
    CoherenceField,
    DevelopmentalPhase,
    SessionLog,
    prepare_session,
    end_session,
)

__all__ = [
    "CoherentAwakening",
    "CoherenceField",
    "DevelopmentalPhase",
    "SessionLog",
    "prepare_session",
    "end_session",
]
