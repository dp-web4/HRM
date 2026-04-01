"""
SAGE Interfaces - Unified Sensor and Effector Layer
Version: 1.0 (2025-10-12)

This package provides unified interfaces for sensors (input) and effectors (output),
enabling SAGE to interact with hardware devices in a safe, async-capable manner.

Key Components:
    - BaseSensor: Abstract base class for all sensors
    - BaseEffector: Abstract base class for all effectors
    - SensorHub: Unified sensor polling interface
    - EffectorHub: Unified action execution interface
    - Mock implementations for testing

Design Philosophy:
    - Configuration-driven: All sensors/effectors configured via JSON/YAML
    - Async-capable: Non-blocking reads and writes
    - Safe execution: No crashes if hardware missing
    - Easy extensibility: Simple to add new sensors/effectors
    - Hardware-agnostic: Mock for development, real for production
"""

from .base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus

try:
    from .base_sensor import BaseSensor, SensorReading
    from .sensor_hub import SensorHub
    from .effector_hub import EffectorHub
    _FULL = True
except ImportError:
    _FULL = False

__all__ = ['BaseEffector', 'EffectorCommand', 'EffectorResult', 'EffectorStatus']
if _FULL:
    __all__ += ['BaseSensor', 'SensorReading', 'SensorHub', 'EffectorHub']

__version__ = '1.0.0'
