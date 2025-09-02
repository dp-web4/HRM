"""
IRP Plugin implementations for different modalities.
"""

from .vision import VisionIRP
from .language import LanguageIRP
from .control import ControlIRP
from .memory import MemoryIRP
from .tinyvae_irp_plugin import TinyVAEIRP, create_tinyvae_irp

__all__ = [
    'VisionIRP',
    'LanguageIRP', 
    'ControlIRP',
    'MemoryIRP',
    'TinyVAEIRP',
    'create_tinyvae_irp'
]