"""
Mock Sensors and Effectors for Testing
Version: 1.0 (2025-10-12)

Mock implementations of sensors and effectors for development and testing.
These can be used without real hardware, generating random data and printing
command outputs.

Design Principles:
    - No hardware dependencies
    - Realistic behavior simulation
    - Configurable characteristics (noise, latency, etc.)
    - Easy to replace with real implementations
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
import random

from .base_sensor import BaseSensor, SensorReading
from .base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus


# ============================================================================
# MOCK SENSORS
# ============================================================================

class MockCameraSensor(BaseSensor):
    """
    Mock camera sensor that generates random images.

    Configuration:
        - resolution: tuple (height, width, channels)
        - noise_level: float (0.0-1.0)
        - simulate_motion: bool (generate moving patterns)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.resolution = config.get('resolution', (480, 640, 3))
        self.noise_level = config.get('noise_level', 0.1)
        self.simulate_motion = config.get('simulate_motion', True)

        # Motion simulation
        self.frame_count = 0
        self.position = [0, 0]

    def poll(self) -> Optional[SensorReading]:
        """Generate random image."""
        if not self._should_poll():
            return None

        # Generate base image
        if self.simulate_motion:
            # Create moving blob
            img = np.zeros(self.resolution, dtype=np.float32)
            h, w, c = self.resolution

            # Update position
            self.position[0] = int((np.sin(self.frame_count * 0.1) + 1) * h / 2)
            self.position[1] = int((np.cos(self.frame_count * 0.1) + 1) * w / 2)

            # Draw blob
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((y - self.position[0])**2 + (x - self.position[1])**2)
            mask = dist < 50
            img[mask] = 1.0

            self.frame_count += 1
        else:
            # Random noise image
            img = np.random.rand(*self.resolution).astype(np.float32)

        # Add noise
        noise = np.random.randn(*self.resolution).astype(np.float32) * self.noise_level
        img = np.clip(img + noise, 0, 1)

        # Convert to tensor
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        self._update_poll_stats()

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            data=tensor,
            metadata={
                'resolution': self.resolution,
                'frame_count': self.frame_count
            },
            confidence=0.95
        )

    def is_available(self) -> bool:
        """Mock camera is always available."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return camera info."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'output_shape': self.resolution,
            'output_dtype': 'float32',
            'rate_limit_hz': self.rate_limit_hz,
            'capabilities': {
                'resolution': self.resolution,
                'noise_level': self.noise_level,
                'simulate_motion': self.simulate_motion
            }
        }


class MockAudioSensor(BaseSensor):
    """
    Mock audio sensor that generates spectrograms with day/night patterns.

    Simulates realistic audio behavior:
    - HIGH ambient noise during day (traffic, activity, etc.)
    - LOW ambient noise at night (quiet environment)
    - Occasional "events" (sounds of interest)

    Output: Mel-spectrogram tensor [n_mels, time_frames]

    Configuration:
        - sample_rate: int (Hz, default 16000)
        - n_mels: int (mel filterbank size, default 64)
        - time_frames: int (spectrogram width, default 32)
        - event_probability: float (chance of sound event per frame)
        - circadian_aware: bool (modulate noise by time of day)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 64)
        self.time_frames = config.get('time_frames', 32)
        self.event_probability = config.get('event_probability', 0.1)
        self.circadian_aware = config.get('circadian_aware', True)

        # Ambient noise characteristics
        self.day_ambient_level = 0.4    # High ambient during day
        self.night_ambient_level = 0.1  # Low ambient at night

        # Event simulation
        self.frame_count = 0
        self.last_event_frame = -100

    def _get_ambient_level(self) -> float:
        """
        Get current ambient noise level

        If circadian_aware, modulate based on frame count
        (simulates day/night cycle)
        """
        if not self.circadian_aware:
            return self.day_ambient_level

        # Simulate day/night: 100 frames = 1 "day"
        day_progress = (self.frame_count % 100) / 100.0

        # Day = 0-60, Night = 60-100
        is_day = day_progress < 0.6

        if is_day:
            return self.day_ambient_level
        else:
            return self.night_ambient_level

    def _generate_spectrogram(self) -> torch.Tensor:
        """Generate synthetic mel-spectrogram"""
        # Base ambient noise
        ambient_level = self._get_ambient_level()
        spectrogram = torch.randn(self.n_mels, self.time_frames) * ambient_level

        # Ambient noise has lower frequencies (traffic, machinery)
        spectrogram[:self.n_mels//2] *= 1.5  # Boost low frequencies

        # Maybe add an "event" (interesting sound)
        if random.random() < self.event_probability:
            # Event: localized in time and frequency
            event_start_time = random.randint(0, self.time_frames - 8)
            event_start_freq = random.randint(0, self.n_mels - 16)

            # Create event blob (e.g., bird chirp, door slam, etc.)
            event_duration = random.randint(4, 8)
            event_bandwidth = random.randint(8, 16)

            event_signal = torch.randn(event_bandwidth, event_duration) * 2.0 + 1.0

            # Add event to spectrogram
            spectrogram[
                event_start_freq:event_start_freq+event_bandwidth,
                event_start_time:event_start_time+event_duration
            ] += event_signal

            self.last_event_frame = self.frame_count

        # Clip to reasonable range
        spectrogram = torch.clamp(spectrogram, -3, 3)

        # Normalize to [0, 1] for processing
        spectrogram = (spectrogram + 3) / 6.0

        return spectrogram

    def poll(self) -> Optional[SensorReading]:
        """Generate audio spectrogram."""
        if not self._should_poll():
            return None

        spectrogram = self._generate_spectrogram()
        ambient_level = self._get_ambient_level()

        self.frame_count += 1
        self._update_poll_stats()

        # Confidence lower when high ambient noise (harder to isolate signal)
        confidence = 0.95 - (ambient_level * 0.4)

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            data=spectrogram,
            metadata={
                'sample_rate': self.sample_rate,
                'n_mels': self.n_mels,
                'time_frames': self.time_frames,
                'frame_count': self.frame_count,
                'ambient_level': ambient_level,
                'had_event': (self.frame_count - self.last_event_frame) < 5
            },
            confidence=confidence
        )

    def is_available(self) -> bool:
        """Mock audio is always available."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return audio sensor info."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'output_shape': (self.n_mels, self.time_frames),
            'output_dtype': 'float32',
            'rate_limit_hz': self.rate_limit_hz,
            'capabilities': {
                'sample_rate': self.sample_rate,
                'n_mels': self.n_mels,
                'time_frames': self.time_frames,
                'circadian_aware': self.circadian_aware,
                'day_ambient': self.day_ambient_level,
                'night_ambient': self.night_ambient_level
            }
        }


class MockMicrophoneSensor(BaseSensor):
    """
    Mock microphone sensor that generates random audio.

    Configuration:
        - sample_rate: int (Hz)
        - duration: float (seconds per sample)
        - num_channels: int
        - generate_tone: bool (sine wave instead of noise)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.sample_rate = config.get('sample_rate', 16000)
        self.duration = config.get('duration', 0.1)  # 100ms
        self.num_channels = config.get('num_channels', 1)
        self.generate_tone = config.get('generate_tone', False)

        self.sample_count = 0

    def poll(self) -> Optional[SensorReading]:
        """Generate random audio."""
        if not self._should_poll():
            return None

        num_samples = int(self.sample_rate * self.duration)

        if self.generate_tone:
            # Generate sine wave
            t = np.linspace(0, self.duration, num_samples)
            frequency = 440 + (self.sample_count % 8) * 55  # A4 and harmonics
            audio = np.sin(2 * np.pi * frequency * t)
            self.sample_count += 1
        else:
            # Random noise
            audio = np.random.randn(num_samples) * 0.1

        # Convert to tensor
        if self.num_channels > 1:
            audio = np.tile(audio[:, None], (1, self.num_channels))

        tensor = torch.from_numpy(audio.astype(np.float32))

        self._update_poll_stats()

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            data=tensor,
            metadata={
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'num_samples': num_samples
            },
            confidence=0.9
        )

    def is_available(self) -> bool:
        """Mock microphone is always available."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return microphone info."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'output_shape': (int(self.sample_rate * self.duration), self.num_channels),
            'output_dtype': 'float32',
            'rate_limit_hz': self.rate_limit_hz,
            'capabilities': {
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'num_channels': self.num_channels
            }
        }


class MockIMUSensor(BaseSensor):
    """
    Mock IMU sensor (accelerometer + gyroscope).

    Configuration:
        - noise_level: float
        - simulate_motion: bool
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.noise_level = config.get('noise_level', 0.01)
        self.simulate_motion = config.get('simulate_motion', True)

        self.time_offset = 0.0

    def poll(self) -> Optional[SensorReading]:
        """Generate IMU reading."""
        if not self._should_poll():
            return None

        if self.simulate_motion:
            # Simulate sinusoidal motion
            t = self.time_offset
            accel = np.array([
                np.sin(t),
                np.cos(t),
                9.81 + np.sin(t * 0.5) * 0.5  # Gravity with small variation
            ])
            gyro = np.array([
                np.cos(t) * 0.1,
                np.sin(t) * 0.1,
                np.sin(t * 0.3) * 0.05
            ])
            self.time_offset += 0.1
        else:
            # Static with noise
            accel = np.array([0, 0, 9.81])
            gyro = np.array([0, 0, 0])

        # Add noise
        accel += np.random.randn(3) * self.noise_level
        gyro += np.random.randn(3) * self.noise_level * 0.1

        # Combine into single tensor [accel_xyz, gyro_xyz]
        imu_data = np.concatenate([accel, gyro])
        tensor = torch.from_numpy(imu_data.astype(np.float32))

        self._update_poll_stats()

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            data=tensor,
            metadata={
                'accel': accel.tolist(),
                'gyro': gyro.tolist()
            },
            confidence=0.98
        )

    def is_available(self) -> bool:
        """Mock IMU is always available."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return IMU info."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'output_shape': (6,),  # accel_xyz + gyro_xyz
            'output_dtype': 'float32',
            'rate_limit_hz': self.rate_limit_hz,
            'capabilities': {
                'measurements': ['accel_x', 'accel_y', 'accel_z',
                               'gyro_x', 'gyro_y', 'gyro_z']
            }
        }


# ============================================================================
# MOCK EFFECTORS
# ============================================================================

class MockMotorEffector(BaseEffector):
    """
    Mock motor effector that prints commands.

    Configuration:
        - max_speed: float
        - simulate_latency: bool
        - latency_ms: float
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.max_speed = config.get('max_speed', 1.0)
        self.simulate_latency = config.get('simulate_latency', True)
        self.latency_ms = config.get('latency_ms', 50)

        # State
        self.current_speed = 0.0
        self.current_direction = 'stop'

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """Execute motor command."""
        start_time = time.time()

        # Check if enabled
        error = self._check_enabled()
        if error:
            return error

        # Validate command
        is_valid, message = self.validate_command(command)
        if not is_valid:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=message
            )

        # Check safety
        safety_error = self._check_safety(command)
        if safety_error:
            return safety_error

        # Simulate latency
        if self.simulate_latency:
            time.sleep(self.latency_ms / 1000.0)

        # Execute based on action
        if command.action == 'move':
            self.current_speed = command.parameters.get('speed', 0.0)
            self.current_direction = command.parameters.get('direction', 'forward')
            print(f"[{self.effector_id}] Moving {self.current_direction} at speed {self.current_speed}")

        elif command.action == 'stop':
            self.current_speed = 0.0
            self.current_direction = 'stop'
            print(f"[{self.effector_id}] Stopped")

        else:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=f"Unknown action: {command.action}"
            )

        result = EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Motor {command.action} executed",
            execution_time=time.time() - start_time,
            metadata={
                'speed': self.current_speed,
                'direction': self.current_direction
            }
        )

        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        """Validate motor command."""
        if command.action not in ['move', 'stop']:
            return False, f"Invalid action: {command.action}"

        if command.action == 'move':
            if 'speed' not in command.parameters:
                return False, "Missing 'speed' parameter"
            if 'direction' not in command.parameters:
                return False, "Missing 'direction' parameter"

            speed = command.parameters['speed']
            if not 0 <= speed <= self.max_speed:
                return False, f"Speed {speed} out of range [0, {self.max_speed}]"

            direction = command.parameters['direction']
            if direction not in ['forward', 'backward', 'left', 'right']:
                return False, f"Invalid direction: {direction}"

        return True, ""

    def is_available(self) -> bool:
        """Mock motor is always available."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return motor info."""
        return {
            'effector_id': self.effector_id,
            'effector_type': self.effector_type,
            'supported_actions': ['move', 'stop'],
            'parameter_schema': {
                'move': {
                    'speed': {'type': 'float', 'range': [0, self.max_speed]},
                    'direction': {'type': 'str', 'options': ['forward', 'backward', 'left', 'right']}
                },
                'stop': {}
            },
            'safety_limits': self.safety_limits,
            'capabilities': {
                'max_speed': self.max_speed
            }
        }


class MockDisplayEffector(BaseEffector):
    """
    Mock display effector that prints frame info.

    Configuration:
        - resolution: tuple (height, width)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.resolution = config.get('resolution', (480, 640))
        self.frame_count = 0

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """Execute display command."""
        start_time = time.time()

        # Check if enabled
        error = self._check_enabled()
        if error:
            return error

        # Validate command
        is_valid, message = self.validate_command(command)
        if not is_valid:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=message
            )

        # Execute based on action
        if command.action == 'show':
            if command.data is not None:
                shape = command.data.shape
                dtype = command.data.dtype
                print(f"[{self.effector_id}] Displaying frame {self.frame_count}: shape={shape}, dtype={dtype}")
                self.frame_count += 1
            else:
                print(f"[{self.effector_id}] No data to display")

        elif command.action == 'clear':
            print(f"[{self.effector_id}] Cleared display")

        else:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=f"Unknown action: {command.action}"
            )

        result = EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Display {command.action} executed",
            execution_time=time.time() - start_time,
            metadata={'frame_count': self.frame_count}
        )

        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        """Validate display command."""
        if command.action not in ['show', 'clear']:
            return False, f"Invalid action: {command.action}"

        if command.action == 'show' and command.data is None:
            return False, "Missing image data for 'show' action"

        return True, ""

    def is_available(self) -> bool:
        """Mock display is always available."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return display info."""
        return {
            'effector_id': self.effector_id,
            'effector_type': self.effector_type,
            'supported_actions': ['show', 'clear'],
            'parameter_schema': {
                'show': {'data': {'type': 'tensor', 'shape': f'[C, {self.resolution[0]}, {self.resolution[1]}]'}},
                'clear': {}
            },
            'safety_limits': {},
            'capabilities': {
                'resolution': self.resolution
            }
        }


class MockSpeakerEffector(BaseEffector):
    """
    Mock speaker effector that prints audio info.

    Configuration:
        - sample_rate: int
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.sample_rate = config.get('sample_rate', 16000)

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """Execute speaker command."""
        start_time = time.time()

        # Check if enabled
        error = self._check_enabled()
        if error:
            return error

        # Validate command
        is_valid, message = self.validate_command(command)
        if not is_valid:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=message
            )

        # Execute based on action
        if command.action == 'play':
            if command.data is not None:
                duration = command.data.shape[0] / self.sample_rate
                print(f"[{self.effector_id}] Playing audio: {duration:.2f}s @ {self.sample_rate}Hz")
            else:
                print(f"[{self.effector_id}] No audio data to play")

        elif command.action == 'stop':
            print(f"[{self.effector_id}] Stopped audio")

        else:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=f"Unknown action: {command.action}"
            )

        result = EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Speaker {command.action} executed",
            execution_time=time.time() - start_time
        )

        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        """Validate speaker command."""
        if command.action not in ['play', 'stop']:
            return False, f"Invalid action: {command.action}"

        if command.action == 'play' and command.data is None:
            return False, "Missing audio data for 'play' action"

        return True, ""

    def is_available(self) -> bool:
        """Mock speaker is always available."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return speaker info."""
        return {
            'effector_id': self.effector_id,
            'effector_type': self.effector_type,
            'supported_actions': ['play', 'stop'],
            'parameter_schema': {
                'play': {'data': {'type': 'tensor', 'shape': '[num_samples, channels]'}},
                'stop': {}
            },
            'safety_limits': {},
            'capabilities': {
                'sample_rate': self.sample_rate
            }
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_sensor_from_config(config: Dict[str, Any]) -> BaseSensor:
    """
    Create sensor from configuration dictionary.

    Args:
        config: Configuration with 'sensor_type' key

    Returns:
        Sensor instance

    Raises:
        ValueError: If sensor_type unknown
    """
    sensor_type = config.get('sensor_type', 'generic')

    if sensor_type == 'camera':
        return MockCameraSensor(config)
    elif sensor_type == 'microphone':
        return MockMicrophoneSensor(config)
    elif sensor_type == 'imu':
        return MockIMUSensor(config)
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")


def create_effector_from_config(config: Dict[str, Any]) -> BaseEffector:
    """
    Create effector from configuration dictionary.

    Args:
        config: Configuration with 'effector_type' key

    Returns:
        Effector instance

    Raises:
        ValueError: If effector_type unknown
    """
    effector_type = config.get('effector_type', 'generic')

    if effector_type == 'motor':
        return MockMotorEffector(config)
    elif effector_type == 'display':
        return MockDisplayEffector(config)
    elif effector_type == 'speaker':
        return MockSpeakerEffector(config)
    else:
        raise ValueError(f"Unknown effector type: {effector_type}")
