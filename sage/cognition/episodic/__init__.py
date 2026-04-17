"""
Episodic Memory — Hippocampal Index for SAGE

Pattern-completion retrieval: given a partial cue about the current moment,
find the most similar past episodes. Distinct from filter-based retrieval
(STM/LTM) and semantic retrieval (cartridges/HierarchicalMemory).

Brain analog: hippocampus (binding + pattern completion + consolidation)
"""

from sage.cognition.episodic.index import EpisodicIndex
from sage.cognition.episodic.data import Episode, EpisodicCue, RecallResult, ConsolidationReport

__all__ = ['EpisodicIndex', 'Episode', 'EpisodicCue', 'RecallResult', 'ConsolidationReport']
