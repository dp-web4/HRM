"""
SAGE Memory Systems

Multiple memory systems working in concert:
- MemoryHub: Unified gathering infrastructure (hub + backends)
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
from .hub import MemoryHub, MemoryEntry, MemoryBackend
from .sqlite_backend import SQLiteBackend

__all__ = [
    'MemoryHub',
    'MemoryEntry',
    'MemoryBackend',
    'SQLiteBackend',
    'HierarchicalMemory',
    'Experience',
    'Pattern',
    'Concept',
    'LatentSpaceIndex',
]
