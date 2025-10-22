"""
SAGE Core - Consciousness Kernel

The central orchestration loop that:
1. Gathers observations from sensors
2. Assesses salience (via SNARC)
3. Allocates attention and resources
4. Invokes appropriate plugins
5. Learns from outcomes
"""

from .sage_kernel import SAGEKernel, MetabolicState

__all__ = ['SAGEKernel', 'MetabolicState']
