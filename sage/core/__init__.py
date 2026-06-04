"""
SAGE Core - Consciousness Kernel

The central orchestration loop that:
1. Gathers observations from sensors
2. Assesses salience (via SNARC)
3. Allocates attention and resources
4. Invokes appropriate plugins
5. Learns from outcomes
"""

# Import both kernel implementations
try:
    from .sage_kernel import SAGEKernel, MetabolicState
except ImportError:
    SAGEKernel = None
    MetabolicState = None

# SAGEUnified (Rev 0 kernel) available via sage.core.sage_unified if needed.
# Not imported eagerly — it pulls in torch at module level (~128MB CUDA libs).
SAGEUnified = None

__all__ = ['SAGEKernel', 'MetabolicState']
