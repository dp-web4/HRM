"""
SAGE Sensor System
==================

Flexible sensor infrastructure supporting multiple backends:
- Real hardware (camera, microphone, IMU) for Jetson Nano
- GR00T integration for Thor (simulation)
- Synthetic fallback for testing

Modalities:
- Vision: Camera sensor (RGB images)
- Audio: Microphone sensor (waveforms)
- Proprioception: Robot body awareness (joint angles, position, gripper)

All sensors output standardized SensorOutput format for SAGE processing.
"""

from .camera_sensor import CameraSensor
from .audio_sensor import AudioSensor
from .proprioception_sensor import ProprioceptionSensor

__all__ = ['CameraSensor', 'AudioSensor', 'ProprioceptionSensor']
