"""
SAGE Memory Systems

Multiple memory systems working in concert:
- Hierarchical Memory: Three-level (experiences → patterns → concepts)
- SNARC Memory: Salience-based storage (existing)
- Circular Buffer: Recent context (existing)

Integration Point: All memory systems accessible through unified interface.
"""

from .hierarchical_memory import (
    HierarchicalMemory,
    Experience,
    Pattern,
    Concept,
    LatentSpaceIndex
)

__all__ = [
    'HierarchicalMemory',
    'Experience',
    'Pattern',
    'Concept',
    'LatentSpaceIndex'
]
