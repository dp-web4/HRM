"""
IRP (Iterative Refinement Primitive) Module
Version: 1.0 (2025-08-23)

Universal framework for intelligence as iterative denoising toward coherence.
"""

from .base import IRPPlugin, IRPState
from .vision import VisionIRP
from .language import LanguageIRP
from .control import ControlIRP
from .memory import MemoryIRP
from .orchestrator import HRMOrchestrator, PluginResult

__all__ = [
    'IRPPlugin',
    'IRPState',
    'VisionIRP',
    'LanguageIRP',
    'ControlIRP',
    'MemoryIRP',
    'HRMOrchestrator',
    'PluginResult'
]

__version__ = '1.0.0'