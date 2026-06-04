"""
IRP (Iterative Refinement Primitive) Module
Version: 1.0 (2025-08-23)

Universal framework for intelligence as iterative denoising toward coherence.
"""

from .base import IRPPlugin, IRPState

# Heavy IRP plugins (VisionIRP, LanguageIRP, etc.) pull in torch at module level.
# Import them on demand via sage.irp.vision, sage.irp.language, etc. — not eagerly.
# HRMOrchestrator is lightweight and needed by the consciousness loop.
try:
    from .orchestrator import HRMOrchestrator, PluginResult
    _HAS_ORCHESTRATOR = True
except ImportError:
    _HAS_ORCHESTRATOR = False

__all__ = ['IRPPlugin', 'IRPState']
if _HAS_ORCHESTRATOR:
    __all__ += ['HRMOrchestrator', 'PluginResult']

__version__ = '1.0.0'