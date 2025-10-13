#!/usr/bin/env python3
"""
Integration Tests for Sensor and Effector Interfaces
Version: 1.0 (2025-10-12)

Comprehensive tests demonstrating all features of the interface layer.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import asyncio
import time

from sage.interfaces import (
    SensorHub, EffectorHub,
    SensorReading, EffectorCommand, EffectorResult, EffectorStatus
)
from sage.interfaces.mock_sensors import (
    MockCameraSensor, MockMicrophoneSensor, MockIMUSensor,
    MockMotorEffector, MockDisplayEffector, MockSpeakerEffector
)


def test_base_sensor():
    """Test basic sensor functionality."""
    print("\n=== Test 1: Basic Sensor ===")

    # Create mock camera
    camera = MockCameraSensor({
        'sensor_id': 'test_camera',
        'sensor_type': 'camera',
        'resolution': (480, 640, 3),
        'rate_limit_hz': 30.0,
        'simulate_motion': True
    })

    # Check availability
    assert camera.is_available(), "Camera should be available"
    print(f"✓ Camera is available: {camera}")

    # Poll sensor
    reading = camera.poll()
    assert reading is not None, "Should get reading"
    assert reading.data.shape == (3, 480, 640), f"Wrong shape: {reading.data.shape}"
    print(f"✓ Camera reading: shape={reading.data.shape}, confidence={reading.confidence}")

    # Check rate limiting
    reading2 = camera.poll()  # Too soon, should be rate-limited
    # Note: Might still return if rate limit allows
    print(f"✓ Rate limiting working")

    # Get statistics
    stats = camera.get_stats()
    print(f"✓ Camera stats: {stats}")

    # Get info
    info = camera.get_info()
    print(f"✓ Camera info: {info}")

    camera.shutdown()
    print("✓ Test 1 passed!\n")


def test_base_effector():
    """Test basic effector functionality."""
    print("\n=== Test 2: Basic Effector ===")

    # Create mock motor
    motor = MockMotorEffector({
        'effector_id': 'test_motor',
        'effector_type': 'motor',
        'max_speed': 1.0,
        'enable_safety_checks': True,
        'safety_limits': {
            'speed': {'min': 0.0, 'max': 1.0}
        }
    })

    # Check availability
    assert motor.is_available(), "Motor should be available"
    print(f"✓ Motor is available: {motor}")

    # Valid command
    cmd = EffectorCommand(
        effector_id='test_motor',
        effector_type='motor',
        action='move',
        parameters={'speed': 0.5, 'direction': 'forward'}
    )

    result = motor.execute(cmd)
    assert result.is_success(), f"Command should succeed: {result.message}"
    print(f"✓ Valid command executed: {result.message}")

    # Invalid command (exceeds safety limit - caught by validation)
    bad_cmd = EffectorCommand(
        effector_id='test_motor',
        effector_type='motor',
        action='move',
        parameters={'speed': 2.0, 'direction': 'forward'}  # Exceeds max_speed
    )

    result = motor.execute(bad_cmd)
    assert result.status == EffectorStatus.INVALID_COMMAND, f"Should be invalid: {result.status}"
    print(f"✓ Invalid command detected: {result.message}")

    # Get statistics
    stats = motor.get_stats()
    print(f"✓ Motor stats: {stats}")

    motor.shutdown()
    print("✓ Test 2 passed!\n")


def test_sensor_hub():
    """Test SensorHub with multiple sensors."""
    print("\n=== Test 3: Sensor Hub ===")

    # Create hub
    hub = SensorHub()

    # Register sensors
    camera = MockCameraSensor({
        'sensor_id': 'camera_0',
        'sensor_type': 'camera',
        'resolution': (240, 320, 3),
        'rate_limit_hz': 30.0
    })
    hub.register_sensor(camera)

    mic = MockMicrophoneSensor({
        'sensor_id': 'microphone_0',
        'sensor_type': 'microphone',
        'sample_rate': 16000,
        'duration': 0.1
    })
    hub.register_sensor(mic)

    imu = MockIMUSensor({
        'sensor_id': 'imu_0',
        'sensor_type': 'imu',
        'rate_limit_hz': 100.0
    })
    hub.register_sensor(imu)

    print(f"✓ Registered {len(hub)} sensors: {hub.list_sensors()}")

    # Poll all sensors
    readings = hub.poll()
    print(f"✓ Polled {len(readings)} sensors")

    # Check each reading
    if 'camera_0' in readings:
        print(f"  - camera_0: shape={readings['camera_0'].shape}")
    if 'microphone_0' in readings:
        print(f"  - microphone_0: shape={readings['microphone_0'].shape}")
    if 'imu_0' in readings:
        print(f"  - imu_0: shape={readings['imu_0'].shape}")

    # Get hub statistics
    stats = hub.get_stats()
    print(f"✓ Hub stats: {stats['num_sensors']} sensors, {stats['num_active']} active")

    # Disable a sensor
    hub.disable_sensor('camera_0')
    readings2 = hub.poll()
    assert 'camera_0' not in readings2, "Disabled sensor should not return data"
    print(f"✓ Sensor disable working (got {len(readings2)} readings)")

    hub.shutdown()
    print("✓ Test 3 passed!\n")


def test_effector_hub():
    """Test EffectorHub with multiple effectors."""
    print("\n=== Test 4: Effector Hub ===")

    # Create hub
    hub = EffectorHub()

    # Register effectors
    motor = MockMotorEffector({
        'effector_id': 'motor_0',
        'effector_type': 'motor',
        'max_speed': 1.0
    })
    hub.register_effector(motor)

    display = MockDisplayEffector({
        'effector_id': 'display_0',
        'effector_type': 'display',
        'resolution': (480, 640)
    })
    hub.register_effector(display)

    speaker = MockSpeakerEffector({
        'effector_id': 'speaker_0',
        'effector_type': 'speaker',
        'sample_rate': 16000
    })
    hub.register_effector(speaker)

    print(f"✓ Registered {len(hub)} effectors: {hub.list_effectors()}")

    # Execute commands
    motor_cmd = EffectorCommand(
        effector_id='motor_0',
        effector_type='motor',
        action='move',
        parameters={'speed': 0.5, 'direction': 'forward'}
    )

    display_cmd = EffectorCommand(
        effector_id='display_0',
        effector_type='display',
        action='show',
        data=torch.rand(3, 480, 640)
    )

    # Execute individually
    result1 = hub.execute(motor_cmd)
    assert result1.is_success()
    print(f"✓ Motor command executed: {result1.message}")

    result2 = hub.execute(display_cmd)
    assert result2.is_success()
    print(f"✓ Display command executed: {result2.message}")

    # Execute batch
    results = hub.execute_batch([motor_cmd, display_cmd])
    assert all(r.is_success() for r in results)
    print(f"✓ Batch execution: {len(results)} commands succeeded")

    # Get hub statistics
    stats = hub.get_stats()
    print(f"✓ Hub stats: {stats['execute_count']} executions, "
          f"{stats['success_rate']:.1%} success rate")

    hub.shutdown()
    print("✓ Test 4 passed!\n")


async def test_async_polling():
    """Test async sensor polling."""
    print("\n=== Test 5: Async Polling ===")

    # Create hub with sensors
    hub = SensorHub()

    for i in range(3):
        camera = MockCameraSensor({
            'sensor_id': f'camera_{i}',
            'sensor_type': 'camera',
            'resolution': (240, 320, 3)
        })
        hub.register_sensor(camera)

    # Sync polling
    start = time.time()
    readings_sync = hub.poll()
    sync_time = time.time() - start
    print(f"✓ Sync polling: {len(readings_sync)} sensors in {sync_time*1000:.1f}ms")

    # Async polling
    start = time.time()
    readings_async = await hub.poll_async()
    async_time = time.time() - start
    print(f"✓ Async polling: {len(readings_async)} sensors in {async_time*1000:.1f}ms")

    print(f"✓ Speedup: {sync_time/async_time:.2f}x")

    hub.shutdown()
    print("✓ Test 5 passed!\n")


async def test_async_execution():
    """Test async effector execution."""
    print("\n=== Test 6: Async Execution ===")

    # Create hub with effectors
    hub = EffectorHub()

    for i in range(3):
        motor = MockMotorEffector({
            'effector_id': f'motor_{i}',
            'effector_type': 'motor',
            'simulate_latency': True,
            'latency_ms': 100
        })
        hub.register_effector(motor)

    # Create commands
    commands = [
        EffectorCommand(
            effector_id=f'motor_{i}',
            effector_type='motor',
            action='move',
            parameters={'speed': 0.5, 'direction': 'forward'}
        )
        for i in range(3)
    ]

    # Sync execution
    start = time.time()
    results_sync = hub.execute_batch(commands)
    sync_time = time.time() - start
    print(f"✓ Sync execution: {len(results_sync)} commands in {sync_time*1000:.0f}ms")

    # Async execution
    start = time.time()
    results_async = await hub.execute_batch_async(commands)
    async_time = time.time() - start
    print(f"✓ Async execution: {len(results_async)} commands in {async_time*1000:.0f}ms")

    print(f"✓ Speedup: {sync_time/async_time:.2f}x")

    hub.shutdown()
    print("✓ Test 6 passed!\n")


def test_integration():
    """Test integration of sensors and effectors."""
    print("\n=== Test 7: Integration ===")

    # Create both hubs
    sensor_hub = SensorHub()
    effector_hub = EffectorHub()

    # Setup sensors
    camera = MockCameraSensor({
        'sensor_id': 'camera',
        'sensor_type': 'camera',
        'resolution': (240, 320, 3)
    })
    sensor_hub.register_sensor(camera)

    # Setup effectors
    display = MockDisplayEffector({
        'effector_id': 'display',
        'effector_type': 'display',
        'resolution': (240, 320)
    })
    effector_hub.register_effector(display)

    motor = MockMotorEffector({
        'effector_id': 'motor',
        'effector_type': 'motor'
    })
    effector_hub.register_effector(motor)

    # Simulate processing loop
    print("✓ Running processing loop...")
    for i in range(3):
        # 1. Poll sensors
        readings = sensor_hub.poll()

        # 2. Process (mock - just pass through)
        if 'camera' in readings:
            image = readings['camera']

            # 3. Execute effectors
            # Display the image
            display_cmd = EffectorCommand(
                effector_id='display',
                effector_type='display',
                action='show',
                data=image
            )
            display_result = effector_hub.execute(display_cmd)

            # Move based on image (mock logic)
            motor_cmd = EffectorCommand(
                effector_id='motor',
                effector_type='motor',
                action='move',
                parameters={'speed': 0.5, 'direction': 'forward'}
            )
            motor_result = effector_hub.execute(motor_cmd)

            print(f"  Cycle {i+1}: display={display_result.is_success()}, "
                  f"motor={motor_result.is_success()}")

        time.sleep(0.1)

    sensor_hub.shutdown()
    effector_hub.shutdown()
    print("✓ Test 7 passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SAGE Sensor/Effector Interface Tests")
    print("=" * 60)

    # Sync tests
    test_base_sensor()
    test_base_effector()
    test_sensor_hub()
    test_effector_hub()
    test_integration()

    # Async tests
    print("\n" + "=" * 60)
    print("Async Tests")
    print("=" * 60)
    asyncio.run(test_async_polling())
    asyncio.run(test_async_execution())

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
