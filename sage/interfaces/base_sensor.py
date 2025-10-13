"""
Base Sensor Interface
Version: 1.0 (2025-10-12)

Abstract base class for all sensors. Sensors provide input to SAGE by reading
from hardware devices (cameras, microphones, etc.) or virtual sources (files, network).

Design Principles:
    - Async-capable: Non-blocking reads with poll() method
    - Type-safe: Returns torch.Tensor for consistency
    - Metadata-rich: Include timestamps, confidence, etc.
    - Graceful degradation: Return None if unavailable
    - Configuration-driven: All parameters in config dict
"""

import torch
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time


@dataclass
class SensorReading:
    """
    Container for sensor data with metadata.

    Attributes:
        sensor_id: Unique identifier for the sensor
        sensor_type: Type of sensor (camera, microphone, etc.)
        data: Sensor reading as torch.Tensor
        timestamp: When the reading was captured (unix timestamp)
        metadata: Additional sensor-specific information
        confidence: Confidence in reading quality (0.0-1.0)
        device: Torch device for tensor
    """
    sensor_id: str
    sensor_type: str
    data: torch.Tensor
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    device: str = 'cpu'

    def to(self, device: torch.device) -> 'SensorReading':
        """Move tensor data to specified device."""
        self.data = self.data.to(device)
        self.device = str(device)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'data_shape': list(self.data.shape),
            'data_dtype': str(self.data.dtype),
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'confidence': self.confidence,
            'device': self.device
        }


class BaseSensor(ABC):
    """
    Abstract base class for all sensors.

    All sensors must implement:
        - poll(): Synchronous read (non-blocking if possible)
        - poll_async(): Asynchronous read
        - is_available(): Check if sensor is accessible
        - get_info(): Return sensor capabilities

    Configuration:
        config = {
            'sensor_id': 'unique_id',
            'sensor_type': 'camera',
            'device': 'cuda',  # or 'cpu'
            'rate_limit_hz': 30.0,  # Max polling rate
            'enabled': True,
            # ... sensor-specific params
        }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sensor with configuration.

        Args:
            config: Configuration dictionary with sensor parameters
        """
        self.config = config
        self.sensor_id = config.get('sensor_id', self.__class__.__name__)
        self.sensor_type = config.get('sensor_type', 'generic')
        self.device = torch.device(config.get('device', 'cpu'))
        self.enabled = config.get('enabled', True)

        # Rate limiting
        self.rate_limit_hz = config.get('rate_limit_hz', 30.0)
        self.min_poll_interval = 1.0 / self.rate_limit_hz if self.rate_limit_hz > 0 else 0.0
        self.last_poll_time = 0.0

        # Statistics
        self.poll_count = 0
        self.error_count = 0
        self.last_reading: Optional[SensorReading] = None

    @abstractmethod
    def poll(self) -> Optional[SensorReading]:
        """
        Synchronous sensor read (non-blocking if possible).

        This should return immediately, either with:
        - SensorReading if data available
        - None if no data available or sensor disabled
        - None if rate-limited (too soon since last poll)

        Returns:
            SensorReading or None
        """
        pass

    async def poll_async(self) -> Optional[SensorReading]:
        """
        Asynchronous sensor read.

        Default implementation wraps poll() in executor.
        Override for true async implementations.

        Returns:
            SensorReading or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.poll)

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if sensor is accessible and operational.

        Returns:
            True if sensor can be polled, False otherwise
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get sensor capabilities and metadata.

        Returns:
            Dictionary with sensor information:
                - sensor_id: str
                - sensor_type: str
                - output_shape: tuple
                - output_dtype: str
                - rate_limit_hz: float
                - capabilities: dict
        """
        pass

    def enable(self):
        """Enable sensor polling."""
        self.enabled = True

    def disable(self):
        """Disable sensor polling."""
        self.enabled = False

    def _should_poll(self) -> bool:
        """
        Check if enough time has passed since last poll (rate limiting).

        Returns:
            True if polling is allowed, False if rate-limited
        """
        if not self.enabled:
            return False

        current_time = time.time()
        time_since_last_poll = current_time - self.last_poll_time

        if time_since_last_poll < self.min_poll_interval:
            return False

        return True

    def _update_poll_stats(self):
        """Update polling statistics."""
        self.poll_count += 1
        self.last_poll_time = time.time()

    def _record_error(self):
        """Record polling error."""
        self.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get sensor statistics.

        Returns:
            Dictionary with:
                - poll_count: int
                - error_count: int
                - error_rate: float
                - uptime: float
                - last_poll: float
        """
        uptime = time.time() - (self.last_poll_time - (self.poll_count / self.rate_limit_hz))
        error_rate = self.error_count / max(self.poll_count, 1)

        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'poll_count': self.poll_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'uptime_seconds': uptime,
            'last_poll_timestamp': self.last_poll_time,
            'enabled': self.enabled,
            'available': self.is_available()
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.poll_count = 0
        self.error_count = 0
        self.last_poll_time = time.time()

    def shutdown(self):
        """
        Cleanup sensor resources.

        Override to implement cleanup logic (close files, release hardware, etc.)
        """
        self.enabled = False

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.sensor_id}, "
                f"type={self.sensor_type}, enabled={self.enabled}, "
                f"available={self.is_available()})")
