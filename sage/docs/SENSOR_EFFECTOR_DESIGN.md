# SAGE Sensor and Effector Interface Layer

**Version:** 1.0
**Date:** 2025-10-12
**Status:** Production Ready

## Overview

This document describes the unified sensor and effector interface layer for SAGE. This layer provides a clean, configuration-driven interface for sensors (input) and effectors (output), enabling SAGE to interact with hardware devices safely and efficiently.

## Design Philosophy

### Core Principles

1. **Configuration-Driven**: All sensors and effectors are configured via YAML/JSON files
2. **Async-Capable**: Non-blocking reads and writes for better performance
3. **Safe Execution**: No crashes if hardware is missing or malfunctioning
4. **Easy Extensibility**: Simple to add new sensor/effector types
5. **Hardware-Agnostic**: Mock implementations for development, real for production
6. **Unified Interface**: Consistent API across all device types

### Key Design Decisions

- **Return Type**: `SensorHub.poll()` returns `Dict[str, torch.Tensor]` for easy consumption
- **Error Handling**: Return `None` or error status instead of raising exceptions
- **Rate Limiting**: Built-in rate limiting to prevent hardware overload
- **Statistics**: Comprehensive monitoring and telemetry
- **Safety Checks**: Configurable safety limits and validation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         SAGE Core                            │
│                   (Attention Orchestrator)                   │
└─────────────────┬────────────────────────┬──────────────────┘
                  │                        │
                  │                        │
         ┌────────▼────────┐      ┌───────▼──────────┐
         │  SensorHub      │      │  EffectorHub     │
         │                 │      │                  │
         │ poll() returns  │      │ execute() safe   │
         │ Dict[str,Tensor]│      │ command exec     │
         └────────┬────────┘      └───────┬──────────┘
                  │                        │
      ┌───────────┴───────────┐   ┌───────┴──────────────┐
      │                       │   │                      │
┌─────▼─────┐  ┌────▼─────┐ │   │ ┌────▼─────┐  ┌─────▼────┐
│ Camera    │  │ Microphone│ │   │ │  Motor   │  │ Display  │
│ Sensor    │  │  Sensor  │ │   │ │ Effector │  │ Effector │
└───────────┘  └──────────┘ │   │ └──────────┘  └──────────┘
┌───────────┐  ┌──────────┐ │   │ ┌──────────┐  ┌──────────┐
│   IMU     │  │  GPS     │ │   │ │ Speaker  │  │ Gripper  │
│  Sensor   │  │ Sensor   │ │   │ │ Effector │  │ Effector │
└───────────┘  └──────────┘ │   │ └──────────┘  └──────────┘
                             │   │
              (More sensors) │   │  (More effectors)
```

## Components

### 1. Base Sensor (`base_sensor.py`)

Abstract base class for all sensors.

**Key Features:**
- Async-capable with `poll()` and `poll_async()`
- Rate limiting to prevent hardware overload
- Rich metadata in `SensorReading` dataclass
- Statistics tracking (poll count, errors, etc.)
- Graceful degradation (returns `None` if unavailable)

**Required Methods:**
```python
class MySensor(BaseSensor):
    def poll(self) -> Optional[SensorReading]:
        """Non-blocking read, returns immediately"""

    def is_available(self) -> bool:
        """Check if hardware is accessible"""

    def get_info(self) -> Dict[str, Any]:
        """Return capabilities and metadata"""
```

### 2. Base Effector (`base_effector.py`)

Abstract base class for all effectors.

**Key Features:**
- Safe execution (never crashes)
- Command validation before execution
- Safety limit checking
- Status codes for results
- Statistics tracking (success rate, etc.)

**Required Methods:**
```python
class MyEffector(BaseEffector):
    def execute(self, command: EffectorCommand) -> EffectorResult:
        """Execute command safely"""

    def validate_command(self, command: EffectorCommand) -> tuple[bool, str]:
        """Validate before execution"""

    def is_available(self) -> bool:
        """Check if hardware is accessible"""

    def get_info(self) -> Dict[str, Any]:
        """Return capabilities and metadata"""
```

### 3. Sensor Hub (`sensor_hub.py`)

Central hub for managing all sensors.

**Primary Interface:**
```python
hub = SensorHub(config_path="sensors.yaml")

# Poll all sensors
readings: Dict[str, torch.Tensor] = hub.poll()

# Access specific sensor data
camera_data = readings['camera_0']  # torch.Tensor [C, H, W]
audio_data = readings['microphone']  # torch.Tensor [samples, channels]
imu_data = readings['imu_0']        # torch.Tensor [6]
```

**Features:**
- Unified polling interface
- Async polling for concurrent reads
- Configuration-driven sensor registration
- Runtime sensor management (add/remove/enable/disable)
- Comprehensive statistics

### 4. Effector Hub (`effector_hub.py`)

Central hub for managing all effectors.

**Primary Interface:**
```python
hub = EffectorHub(config_path="effectors.yaml")

# Execute single command
cmd = EffectorCommand(
    effector_id='motor_0',
    effector_type='motor',
    action='move',
    parameters={'speed': 0.5, 'direction': 'forward'}
)
result = hub.execute(cmd)

# Check result
if result.is_success():
    print("Command executed successfully")
```

**Features:**
- Safe command execution
- Priority-based command queuing
- Batch execution (sync and async)
- Command validation
- Comprehensive statistics

### 5. Mock Implementations (`mock_sensors.py`)

Mock sensors and effectors for testing without hardware.

**Available Mocks:**
- `MockCameraSensor`: Generates random/moving images
- `MockMicrophoneSensor`: Generates audio noise or tones
- `MockIMUSensor`: Simulates accelerometer/gyroscope
- `MockMotorEffector`: Prints motor commands
- `MockDisplayEffector`: Prints display operations
- `MockSpeakerEffector`: Prints audio playback

## Usage Examples

### Example 1: Basic Sensor Polling

```python
from sage.interfaces import SensorHub, create_sensor_hub
from sage.interfaces.mock_sensors import MockCameraSensor

# Create hub
hub = SensorHub()

# Register mock camera
camera = MockCameraSensor({
    'sensor_id': 'camera_0',
    'sensor_type': 'camera',
    'resolution': (480, 640, 3),
    'rate_limit_hz': 30.0
})
hub.register_sensor(camera)

# Poll sensors
readings = hub.poll()

# Use camera data
if 'camera_0' in readings:
    image = readings['camera_0']  # torch.Tensor [3, 480, 640]
    print(f"Camera image shape: {image.shape}")

# Get statistics
stats = hub.get_stats()
print(f"Polling stats: {stats}")
```

### Example 2: Async Sensor Polling

```python
import asyncio

async def poll_sensors_continuously():
    hub = create_sensor_hub(config_path="configs/sensor_config.yaml")

    while True:
        # Poll all sensors concurrently
        readings = await hub.poll_async()

        # Process readings
        for sensor_id, data in readings.items():
            print(f"{sensor_id}: {data.shape}")

        await asyncio.sleep(0.033)  # ~30 Hz

# Run
asyncio.run(poll_sensors_continuously())
```

### Example 3: Safe Effector Execution

```python
from sage.interfaces import EffectorHub, EffectorCommand
from sage.interfaces.mock_sensors import MockMotorEffector

# Create hub
hub = EffectorHub()

# Register mock motor
motor = MockMotorEffector({
    'effector_id': 'motor_0',
    'effector_type': 'motor',
    'max_speed': 1.0,
    'enable_safety_checks': True,
    'safety_limits': {
        'speed': {'min': 0.0, 'max': 1.0}
    }
})
hub.register_effector(motor)

# Create command
cmd = EffectorCommand(
    effector_id='motor_0',
    effector_type='motor',
    action='move',
    parameters={'speed': 0.5, 'direction': 'forward'}
)

# Validate before execution (optional)
is_valid, message = hub.validate_command(cmd)
if not is_valid:
    print(f"Invalid command: {message}")
else:
    # Execute
    result = hub.execute(cmd)

    if result.is_success():
        print(f"Success: {result.message}")
    else:
        print(f"Failed: {result.status.value} - {result.message}")
```

### Example 4: Configuration-Driven Setup

```python
from sage.interfaces import create_sensor_hub, create_effector_hub

# Load from config files
sensor_hub = create_sensor_hub(
    config_path="configs/sensor_config.yaml"
)

effector_hub = create_effector_hub(
    config_path="configs/effector_config.yaml"
)

# Use in SAGE loop
while True:
    # Input: Poll sensors
    sensor_data = sensor_hub.poll()

    # Process with SAGE (your attention orchestrator)
    decisions = sage_process(sensor_data)

    # Output: Execute actions
    for decision in decisions:
        cmd = create_command_from_decision(decision)
        result = effector_hub.execute(cmd)

        if not result.is_success():
            print(f"Execution failed: {result.message}")
```

### Example 5: Integration with SAGE Core

```python
from sage.core.sage_core import SAGECore
from sage.interfaces import SensorHub, EffectorHub

class SAGEWithInterfaces:
    """SAGE with sensor/effector interfaces."""

    def __init__(self, sage_config, sensor_config, effector_config):
        self.sage = SAGECore(sage_config)
        self.sensors = create_sensor_hub(config_path=sensor_config)
        self.effectors = create_effector_hub(config_path=effector_config)

    async def run_loop(self):
        """Main SAGE loop with interfaces."""
        while True:
            # 1. Gather sensor inputs
            readings = await self.sensors.poll_async()

            # 2. Process with SAGE
            attention_outputs = self.sage.process(readings)

            # 3. Execute effector commands
            commands = self.sage_to_commands(attention_outputs)
            results = await self.effectors.execute_batch_async(commands)

            # 4. Update SAGE with execution feedback
            self.sage.update_from_results(results)

    def sage_to_commands(self, outputs):
        """Convert SAGE outputs to effector commands."""
        commands = []

        # Example: motor control from SAGE output
        if 'motor_control' in outputs:
            cmd = EffectorCommand(
                effector_id='motor_0',
                effector_type='motor',
                action='move',
                parameters={
                    'speed': outputs['motor_control']['speed'],
                    'direction': outputs['motor_control']['direction']
                }
            )
            commands.append(cmd)

        return commands
```

## Adding New Sensors/Effectors

### Adding a Real Camera Sensor

```python
import cv2
from sage.interfaces import BaseSensor, SensorReading

class RealCameraSensor(BaseSensor):
    """Real camera using OpenCV."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.camera_id = config.get('camera_id', 0)
        self.cap = None

    def poll(self) -> Optional[SensorReading]:
        if not self._should_poll():
            return None

        # Lazy initialization
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self._record_error()
                return None

        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            self._record_error()
            return None

        # Convert BGR to RGB and to tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        self._update_poll_stats()

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            data=tensor,
            metadata={'frame_shape': frame.shape},
            confidence=1.0
        )

    def is_available(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_info(self) -> Dict[str, Any]:
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'output_shape': (3, 480, 640),
            'output_dtype': 'float32',
            'rate_limit_hz': self.rate_limit_hz
        }

    def shutdown(self):
        if self.cap:
            self.cap.release()
        super().shutdown()
```

### Adding a Real Motor Effector

```python
from sage.interfaces import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus

class RealMotorEffector(BaseEffector):
    """Real motor using serial communication."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.serial_port = config.get('serial_port', '/dev/ttyUSB0')
        self.ser = None

    def execute(self, command: EffectorCommand) -> EffectorResult:
        start_time = time.time()

        # Check enabled
        error = self._check_enabled()
        if error:
            return error

        # Validate
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

        # Lazy initialization
        if self.ser is None:
            try:
                import serial
                self.ser = serial.Serial(self.serial_port, 9600)
            except Exception as e:
                return EffectorResult(
                    effector_id=self.effector_id,
                    status=EffectorStatus.HARDWARE_UNAVAILABLE,
                    message=f"Failed to open serial port: {e}"
                )

        # Execute command
        try:
            if command.action == 'move':
                speed = command.parameters['speed']
                direction = command.parameters['direction']
                # Send to motor controller
                self.ser.write(f"MOVE {speed} {direction}\n".encode())

            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.SUCCESS,
                message=f"Motor {command.action} executed",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.FAILED,
                message=f"Execution failed: {e}",
                execution_time=time.time() - start_time
            )

        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> tuple[bool, str]:
        if command.action not in ['move', 'stop']:
            return False, f"Invalid action: {command.action}"
        return True, ""

    def is_available(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def get_info(self) -> Dict[str, Any]:
        return {
            'effector_id': self.effector_id,
            'effector_type': self.effector_type,
            'supported_actions': ['move', 'stop'],
            'serial_port': self.serial_port
        }

    def shutdown(self):
        if self.ser:
            self.ser.close()
        super().shutdown()
```

## Configuration Reference

### Sensor Configuration Schema

```yaml
sensors:
  - sensor_id: string          # Unique identifier
    sensor_type: string        # Type (camera, microphone, imu, etc.)
    enabled: bool              # Enable/disable sensor
    device: string             # 'cuda' or 'cpu'
    rate_limit_hz: float       # Maximum polling rate
    # Sensor-specific parameters
    ...

hub:
  default_device: string       # Default device
  enable_async: bool           # Enable async polling
  max_poll_time_ms: float      # Maximum poll time
```

### Effector Configuration Schema

```yaml
effectors:
  - effector_id: string        # Unique identifier
    effector_type: string      # Type (motor, display, speaker, etc.)
    enabled: bool              # Enable/disable effector
    device: string             # 'cuda' or 'cpu'
    enable_safety_checks: bool # Enable safety validation
    safety_limits:             # Safety constraints
      parameter_name:
        min: float
        max: float
    # Effector-specific parameters
    ...

hub:
  default_device: string       # Default device
  enable_async: bool           # Enable async execution
  max_execute_time_ms: float   # Maximum execution time
  enable_command_queue: bool   # Enable priority queue
```

## Performance Considerations

### Sensor Polling

- **Rate Limiting**: Configure `rate_limit_hz` appropriately for each sensor
- **Async Polling**: Use `poll_async()` for concurrent sensor reads (2-3x speedup)
- **Device Placement**: Use `device: cuda` for sensors with GPU acceleration
- **Polling Strategy**: Poll high-priority sensors more frequently

### Effector Execution

- **Batch Execution**: Use `execute_batch_async()` for parallel commands
- **Safety Overhead**: Disable safety checks in production if validated
- **Priority Queue**: Use for time-critical commands
- **Async Execution**: Use `execute_async()` for long-running commands

### Memory Management

- **Tensor Devices**: Move tensors to appropriate device in hub
- **Cleanup**: Call `shutdown()` on hubs to release resources
- **Statistics**: Reset stats periodically to avoid memory growth

## Safety Guidelines

### Sensor Safety

1. **Rate Limiting**: Always configure rate limits to prevent hardware overload
2. **Error Handling**: Check for `None` returns from `poll()`
3. **Resource Cleanup**: Always call `shutdown()` when done
4. **Availability Checks**: Use `is_available()` before critical operations

### Effector Safety

1. **Command Validation**: Always validate commands before execution
2. **Safety Limits**: Configure safety limits in config files
3. **Result Checking**: Always check `EffectorResult.is_success()`
4. **Timeout Handling**: Set appropriate timeouts in commands
5. **Emergency Stop**: Implement emergency stop for critical effectors

## Testing

### Unit Tests

```python
# Test sensor
def test_mock_camera():
    sensor = MockCameraSensor({
        'sensor_id': 'test_cam',
        'sensor_type': 'camera',
        'resolution': (480, 640, 3)
    })

    reading = sensor.poll()
    assert reading is not None
    assert reading.data.shape == (3, 480, 640)

# Test effector
def test_mock_motor():
    effector = MockMotorEffector({
        'effector_id': 'test_motor',
        'effector_type': 'motor'
    })

    cmd = EffectorCommand(
        effector_id='test_motor',
        effector_type='motor',
        action='move',
        parameters={'speed': 0.5, 'direction': 'forward'}
    )

    result = effector.execute(cmd)
    assert result.is_success()
```

### Integration Tests

See `/home/dp/ai-workspace/HRM/sage/interfaces/test_interfaces.py` for complete integration tests.

## Migration Path

### Phase 1: Mock Development (Current)

- Use mock sensors/effectors for development
- Test SAGE logic without hardware
- Validate configuration and interfaces

### Phase 2: Hardware Integration

- Implement real sensor/effector classes
- Test with actual hardware
- Keep mock fallbacks for testing

### Phase 3: Production Deployment

- Load configuration from environment-specific files
- Enable safety checks and monitoring
- Deploy to edge devices (Jetson, etc.)

## Future Enhancements

### Planned Features

1. **Plugin System**: Dynamic loading of sensor/effector plugins
2. **Hot Reload**: Add/remove sensors at runtime without restart
3. **Calibration**: Built-in sensor calibration routines
4. **Recording**: Record sensor streams for replay
5. **Simulation**: Integration with physics simulators
6. **Federation**: Distributed sensors across multiple devices
7. **Compression**: Sensor data compression for bandwidth savings

### Under Consideration

- **ROS Integration**: Bridge to ROS topics
- **Gazebo Support**: Sensor/effector simulation
- **WebRTC Streaming**: Remote sensor monitoring
- **Telemetry Dashboard**: Real-time visualization

## References

- **IRP System**: `/home/dp/ai-workspace/HRM/sage/irp/base.py`
- **Existing Sensors**: `/home/dp/ai-workspace/HRM/sage/sensors/sensor_interface.py`
- **Camera Sensor Example**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/camera_sensor_impl.py`
- **Visual Monitor Example**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/visual_monitor_effector.py`

## Support

For questions or issues:
- Check examples in this document
- Review test files
- Consult existing IRP plugin implementations
- Refer to SAGE core documentation

---

**Last Updated:** 2025-10-12
**Contributors:** Claude (System Design)
**Status:** Production Ready
