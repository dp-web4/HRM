"""
IRP (Iterative Refinement Primitive) Module
Version: 1.0 (2025-08-23)

Universal framework for intelligence as iterative denoising toward coherence.
"""

from .base import IRPPlugin, IRPState

# Heavy dependencies (torch, etc.) — optional for minimal envs (Sprout, arc venv)
try:
    from .vision import VisionIRP
    from .language import LanguageIRP
    from .control import ControlIRP
    from .memory import MemoryIRP
    from .orchestrator import HRMOrchestrator, PluginResult
    _FULL = True
except ImportError:
    _FULL = False

__all__ = ['IRPPlugin', 'IRPState']
if _FULL:
    __all__ += ['VisionIRP', 'LanguageIRP', 'ControlIRP', 'MemoryIRP',
                'HRMOrchestrator', 'PluginResult']

__version__ = '1.0.0'