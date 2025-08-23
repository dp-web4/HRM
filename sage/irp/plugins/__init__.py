"""
IRP Plugin implementations for different modalities.
"""

from .vision import VisionIRP
from .language import LanguageIRP
from .control import ControlIRP
from .memory import MemoryIRP

__all__ = [
    'VisionIRP',
    'LanguageIRP', 
    'ControlIRP',
    'MemoryIRP'
]