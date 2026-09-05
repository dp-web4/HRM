"""
SAGE Memory Systems

Multiple memory systems working in concert:
- MemoryHub: Unified gathering infrastructure (hub + backends)
- Hierarchical Memory: Three-level (experiences → patterns → concepts)
- SNARC Memory: Salience-based storage (existing)
- Circular Buffer: Recent context (existing)

Integration Point: All memory systems accessible through unified interface.
"""

# hierarchical_memory needs torch. The consolidation organ (sage.memory.consolidation) and the
# hub/sqlite backends do not, and the being's heartbeat runs on a plain python3; so a missing
# torch must not take the whole package down with it.
try:
    from .hierarchical_memory import (
        HierarchicalMemory,
        Experience,
        Pattern,
        Concept,
        LatentSpaceIndex
    )
except ImportError:  # torch absent: the torch-backed classes are simply unavailable
    HierarchicalMemory = Experience = Pattern = Concept = LatentSpaceIndex = None
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
