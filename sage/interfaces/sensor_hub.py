"""
Sensor Hub - Unified Sensor Polling Interface
Version: 1.0 (2025-10-12)

Central hub for managing all sensors. Provides unified polling interface that
returns Dict[str, torch.Tensor] for easy consumption by SAGE.

Design Principles:
    - Unified interface: Single poll() method for all sensors
    - Configuration-driven: Load sensors from config files
    - Async-capable: Support concurrent polling
    - Graceful degradation: Continue working if sensors fail
    - Hot-reload: Add/remove sensors at runtime
"""

import torch
import asyncio
import yaml
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import time

from .base_sensor import BaseSensor, SensorReading


class SensorHub:
    """
    Central hub for managing multiple sensors.

    The hub provides:
    - Unified polling interface (poll() returns Dict[str, torch.Tensor])
    - Async polling for concurrent sensor reads
    - Configuration-driven sensor registration
    - Runtime sensor management (add/remove/enable/disable)
    - Statistics and monitoring

    Example:
        hub = SensorHub(config_path="sensors.yaml")
        readings = hub.poll()  # Dict[str, torch.Tensor]

        # Access specific sensor data
        camera_data = readings['camera_0']  # torch.Tensor
        audio_data = readings['microphone']  # torch.Tensor
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[Path] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize sensor hub.

        Args:
            config: Configuration dictionary
            config_path: Path to YAML/JSON config file
            device: Default torch device for sensors
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sensors: Dict[str, BaseSensor] = {}

        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = {}

        # Statistics
        self.poll_count = 0
        self.last_poll_time = 0.0
        self.poll_times: List[float] = []

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def register_sensor(self, sensor: BaseSensor, sensor_id: Optional[str] = None):
        """
        Register a sensor with the hub.

        Args:
            sensor: Sensor instance to register
            sensor_id: Optional custom ID (uses sensor.sensor_id if None)
        """
        sensor_id = sensor_id or sensor.sensor_id

        if sensor_id in self.sensors:
            print(f"Warning: Overwriting existing sensor '{sensor_id}'")

        self.sensors[sensor_id] = sensor

    def unregister_sensor(self, sensor_id: str):
        """
        Unregister and cleanup a sensor.

        Args:
            sensor_id: ID of sensor to remove
        """
        if sensor_id in self.sensors:
            sensor = self.sensors[sensor_id]
            sensor.shutdown()
            del self.sensors[sensor_id]

    def poll(self, sensor_ids: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Poll all (or specified) sensors synchronously.

        This is the primary interface for getting sensor data.
        Returns a dictionary mapping sensor_id -> tensor data.

        Args:
            sensor_ids: Optional list of specific sensors to poll
                       (polls all if None)

        Returns:
            Dictionary mapping sensor_id to torch.Tensor
            Only includes sensors that returned valid readings
        """
        start_time = time.time()

        # Determine which sensors to poll
        if sensor_ids:
            sensors_to_poll = {sid: self.sensors[sid] for sid in sensor_ids
                              if sid in self.sensors}
        else:
            sensors_to_poll = self.sensors

        # Poll each sensor
        readings: Dict[str, torch.Tensor] = {}

        for sensor_id, sensor in sensors_to_poll.items():
            try:
                reading = sensor.poll()

                if reading is not None and reading.data is not None:
                    # Move to hub's default device if needed
                    tensor = reading.data.to(self.device)
                    readings[sensor_id] = tensor

            except Exception as e:
                print(f"Error polling sensor '{sensor_id}': {e}")
                # Continue with other sensors

        # Update statistics
        self.poll_count += 1
        self.last_poll_time = time.time()
        self.poll_times.append(time.time() - start_time)

        # Keep only recent poll times (last 100)
        if len(self.poll_times) > 100:
            self.poll_times = self.poll_times[-100:]

        return readings

    async def poll_async(self, sensor_ids: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Poll all (or specified) sensors asynchronously.

        Uses asyncio to poll sensors concurrently for better performance.

        Args:
            sensor_ids: Optional list of specific sensors to poll

        Returns:
            Dictionary mapping sensor_id to torch.Tensor
        """
        start_time = time.time()

        # Determine which sensors to poll
        if sensor_ids:
            sensors_to_poll = {sid: self.sensors[sid] for sid in sensor_ids
                              if sid in self.sensors}
        else:
            sensors_to_poll = self.sensors

        # Create async tasks for each sensor
        async def poll_one(sensor_id: str, sensor: BaseSensor):
            try:
                reading = await sensor.poll_async()
                if reading is not None and reading.data is not None:
                    return sensor_id, reading.data.to(self.device)
            except Exception as e:
                print(f"Error polling sensor '{sensor_id}': {e}")
            return sensor_id, None

        # Poll all sensors concurrently
        tasks = [poll_one(sid, sensor) for sid, sensor in sensors_to_poll.items()]
        results = await asyncio.gather(*tasks)

        # Build result dictionary
        readings = {sid: tensor for sid, tensor in results if tensor is not None}

        # Update statistics
        self.poll_count += 1
        self.last_poll_time = time.time()
        self.poll_times.append(time.time() - start_time)

        if len(self.poll_times) > 100:
            self.poll_times = self.poll_times[-100:]

        return readings

    def get_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """
        Get last reading from specific sensor (includes metadata).

        Args:
            sensor_id: Sensor to query

        Returns:
            SensorReading or None
        """
        if sensor_id not in self.sensors:
            return None

        sensor = self.sensors[sensor_id]
        return sensor.last_reading

    def enable_sensor(self, sensor_id: str):
        """Enable a specific sensor."""
        if sensor_id in self.sensors:
            self.sensors[sensor_id].enable()

    def disable_sensor(self, sensor_id: str):
        """Disable a specific sensor."""
        if sensor_id in self.sensors:
            self.sensors[sensor_id].disable()

    def list_sensors(self) -> List[str]:
        """Get list of registered sensor IDs."""
        return list(self.sensors.keys())

    def get_sensor_info(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific sensor."""
        if sensor_id not in self.sensors:
            return None

        sensor = self.sensors[sensor_id]
        return sensor.get_info()

    def get_all_sensor_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all sensors."""
        return {sid: sensor.get_info() for sid, sensor in self.sensors.items()}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get hub statistics.

        Returns:
            Dictionary with:
                - num_sensors: int
                - num_active: int
                - poll_count: int
                - avg_poll_time_ms: float
                - sensor_stats: dict
        """
        active_sensors = sum(1 for s in self.sensors.values() if s.enabled)
        avg_poll_time = sum(self.poll_times) / len(self.poll_times) if self.poll_times else 0.0

        return {
            'num_sensors': len(self.sensors),
            'num_active': active_sensors,
            'poll_count': self.poll_count,
            'avg_poll_time_ms': avg_poll_time * 1000,
            'last_poll_time': self.last_poll_time,
            'sensor_stats': {sid: sensor.get_stats()
                           for sid, sensor in self.sensors.items()}
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.poll_count = 0
        self.poll_times = []
        for sensor in self.sensors.values():
            sensor.reset_stats()

    def shutdown(self):
        """Shutdown all sensors and cleanup resources."""
        for sensor_id in list(self.sensors.keys()):
            self.unregister_sensor(sensor_id)

    def __repr__(self) -> str:
        active = sum(1 for s in self.sensors.values() if s.enabled)
        return f"SensorHub(sensors={len(self.sensors)}, active={active})"

    def __len__(self) -> int:
        """Return number of registered sensors."""
        return len(self.sensors)

    def __contains__(self, sensor_id: str) -> bool:
        """Check if sensor is registered."""
        return sensor_id in self.sensors

    def __getitem__(self, sensor_id: str) -> BaseSensor:
        """Get sensor by ID."""
        return self.sensors[sensor_id]


# Convenience function for creating hub from config
def create_sensor_hub(config_path: Optional[Path] = None,
                     sensor_configs: Optional[List[Dict[str, Any]]] = None,
                     device: Optional[torch.device] = None) -> SensorHub:
    """
    Create sensor hub from configuration.

    Args:
        config_path: Path to config file
        sensor_configs: List of sensor configurations
        device: Default torch device

    Returns:
        Configured SensorHub instance

    Example:
        # From config file
        hub = create_sensor_hub(config_path="sensors.yaml")

        # From dictionaries
        configs = [
            {'sensor_id': 'camera', 'sensor_type': 'camera', ...},
            {'sensor_id': 'mic', 'sensor_type': 'microphone', ...}
        ]
        hub = create_sensor_hub(sensor_configs=configs)
    """
    hub = SensorHub(config_path=config_path, device=device)

    # If sensor configs provided, they would be registered here
    # This requires sensor factory functions (implemented in mock_sensors.py)
    if sensor_configs:
        # Import sensor factories
        from .mock_sensors import create_sensor_from_config

        for sensor_config in sensor_configs:
            try:
                sensor = create_sensor_from_config(sensor_config)
                hub.register_sensor(sensor)
            except Exception as e:
                print(f"Failed to create sensor from config {sensor_config}: {e}")

    return hub
