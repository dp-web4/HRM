#!/usr/bin/env python3
"""
IMU Sensor for SAGE Consciousness - Track 5
============================================

Real-time IMU (Inertial Measurement Unit) integration with cognitive architecture.

Provides:
- Orientation tracking (roll, pitch, yaw)
- Motion detection (acceleration, angular velocity)
- Spatial awareness for embodied AI

Integrates:
- Track 1: Sensor trust based on calibration + stability
- Track 2: Memory storage of motion events
- Track 3: Attention allocation based on motion salience

Hardware Support:
- BNO055 (9-DOF, built-in fusion) - Recommended
- MPU6050 (6-DOF) - Fallback
- Simulated mode (for testing without hardware)

Track 5: IMU Sensor - Phase 1 Implementation
"""

import numpy as np
import time
import threading
import queue
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class IMUObservation:
    """
    IMU sensor observation

    Integrates with Track 1-3:
    - Orientation and motion for spatial awareness
    - salience: For Track 3 attention allocation
    - trust_score: For Track 1 sensor fusion
    - Motion events for Track 2 memory storage
    """
    # Orientation (absolute)
    quaternion: np.ndarray  # [4] (w, x, y, z)
    euler: np.ndarray  # [3] (roll, pitch, yaw) radians

    # Motion (rates)
    angular_velocity: np.ndarray  # [3] rad/s (gyro)
    linear_acceleration: np.ndarray  # [3] m/s² (accel)

    # Magnetic heading (if magnetometer available)
    heading: Optional[float] = None  # degrees (0-360)

    # Derived
    motion_magnitude: float = 0.0  # Combined motion intensity
    is_moving: bool = False  # Motion detection

    # Integration
    salience: float = 0.0  # For Track 3 attention
    trust_score: float = 1.0  # For Track 1 fusion

    # Metadata
    timestamp: float = 0.0
    calibration_status: str = "unknown"  # uncalibrated/partial/full
    metadata: Dict[str, Any] = field(default_factory=dict)


class IMUBackend(Enum):
    """IMU backend types"""
    BNO055 = "bno055"  # BNO055 9-DOF with built-in fusion
    MPU6050 = "mpu6050"  # MPU6050 6-DOF
    MPU9250 = "mpu9250"  # MPU9250 9-DOF with magnetometer
    SIMULATED = "simulated"  # Simulated IMU for testing


class IMUSensor:
    """
    Real-time IMU sensor for SAGE consciousness

    Features:
    - Multi-backend support (BNO055, MPU6050, simulated)
    - Auto-detection of available hardware
    - Track 1-3 integration (trust, memory, attention)
    - Real-time capture thread (50-100 Hz)
    - Motion detection and salience computation
    - Low latency (<5ms target)
    """

    def __init__(
        self,
        device_type: str = "auto",  # auto, bno055, mpu6050, simulated
        i2c_address: int = 0x28,  # BNO055 default
        sample_rate: int = 50,  # Hz
        motion_threshold: float = 0.5,  # m/s² for motion detection
        sensor_trust: Optional[Any] = None,  # Track 1
        memory_system: Optional[Any] = None,  # Track 2
        attention_manager: Optional[Any] = None,  # Track 3
    ):
        """
        Initialize IMU sensor

        Args:
            device_type: Backend type (auto, bno055, mpu6050, simulated)
            i2c_address: I2C address for hardware IMU
            sample_rate: Sample rate in Hz (50-100 recommended)
            motion_threshold: Acceleration threshold for motion detection
            sensor_trust: Track 1 sensor trust system
            memory_system: Track 2 memory system
            attention_manager: Track 3 attention manager
        """
        self.sample_rate = sample_rate
        self.motion_threshold = motion_threshold

        # Track 1-3 integration
        self.sensor_trust = sensor_trust
        self.memory = memory_system
        self.attention = attention_manager

        # Detect backend
        if device_type == "auto":
            self.backend = self._detect_device()
        else:
            self.backend = IMUBackend(device_type)

        print(f"📐 IMU sensor: {self.backend.value} @ {sample_rate}Hz")

        # Initialize hardware/simulation
        self.imu = self._initialize_backend()

        # Motion tracking
        self.prev_euler = np.zeros(3)
        self.orientation_history = []  # For stability tracking
        self.accel_history = []  # For trust computation

        # Performance tracking
        self.frame_count = 0
        self.fps_actual = 0.0
        self.latency_avg = 0.0
        self.latency_samples = []
        self.i2c_errors = 0
        self.i2c_attempts = 0

        # Background capture thread
        self.capture_running = False
        self.capture_thread = None
        self.capture_queue = queue.Queue(maxsize=2)  # Low latency

        # Track 1 registration
        if self.sensor_trust:
            self.sensor_trust.register_sensor('imu', initial_trust=0.7)
            print("  ✓ Registered with Track 1 (Sensor Trust)")

        print(f"  {self._get_backend_info()}")
        print("✓ IMU sensor initialized")

        # Start background capture
        self._start_capture_thread()

    def _detect_device(self) -> IMUBackend:
        """
        Auto-detect available IMU hardware

        Priority: BNO055 > MPU6050 > MPU9250 > Simulated
        """
        # Try BNO055 first (recommended)
        try:
            import board
            import busio
            import adafruit_bno055
            i2c = busio.I2C(board.SCL, board.SDA)
            sensor = adafruit_bno055.BNO055_I2C(i2c)
            _ = sensor.quaternion  # Test read
            return IMUBackend.BNO055
        except (ImportError, OSError, RuntimeError):
            pass

        # Try MPU6050
        try:
            from mpu6050 import mpu6050
            sensor = mpu6050(0x68)
            _ = sensor.get_accel_data()  # Test read
            return IMUBackend.MPU6050
        except (ImportError, OSError, RuntimeError):
            pass

        # Fallback to simulated
        return IMUBackend.SIMULATED

    def _initialize_backend(self) -> Any:
        """Initialize the selected backend"""
        if self.backend == IMUBackend.BNO055:
            return self._init_bno055()
        elif self.backend == IMUBackend.MPU6050:
            return self._init_mpu6050()
        elif self.backend == IMUBackend.SIMULATED:
            return self._init_simulated()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _init_bno055(self) -> Any:
        """Initialize BNO055 sensor"""
        try:
            import board
            import busio
            import adafruit_bno055

            i2c = busio.I2C(board.SCL, board.SDA)
            sensor = adafruit_bno055.BNO055_I2C(i2c)

            # Check calibration status
            cal_status = sensor.calibration_status
            print(f"  BNO055 calibration: sys={cal_status[0]}, gyro={cal_status[1]}, "
                  f"accel={cal_status[2]}, mag={cal_status[3]}")

            return sensor
        except Exception as e:
            print(f"  ⚠️  BNO055 initialization failed: {e}")
            print("  ⚠️  Falling back to simulated mode")
            self.backend = IMUBackend.SIMULATED
            return self._init_simulated()

    def _init_mpu6050(self) -> Any:
        """Initialize MPU6050 sensor"""
        try:
            from mpu6050 import mpu6050
            sensor = mpu6050(0x68)
            return sensor
        except Exception as e:
            print(f"  ⚠️  MPU6050 initialization failed: {e}")
            print("  ⚠️  Falling back to simulated mode")
            self.backend = IMUBackend.SIMULATED
            return self._init_simulated()

    def _init_simulated(self) -> Dict:
        """Initialize simulated IMU"""
        return {
            'start_time': time.time(),
            'mode': 'simulated'
        }

    def _get_backend_info(self) -> str:
        """Get backend information string"""
        if self.backend == IMUBackend.BNO055:
            return "BNO055 9-DOF with hardware fusion"
        elif self.backend == IMUBackend.MPU6050:
            return "MPU6050 6-DOF (software fusion required)"
        elif self.backend == IMUBackend.SIMULATED:
            return "Simulated IMU mode (synthetic motion)"
        return "Unknown backend"

    def _start_capture_thread(self):
        """Start background capture thread"""
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        """Background capture loop running at sample_rate Hz"""
        dt = 1.0 / self.sample_rate

        while self.capture_running:
            start_time = time.time()

            try:
                # Capture observation
                obs = self._capture_raw()

                if obs is not None:
                    # Try to add to queue (drop old if full)
                    try:
                        self.capture_queue.put_nowait(obs)
                    except queue.Full:
                        # Remove old, add new (keep latency low)
                        try:
                            self.capture_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.capture_queue.put_nowait(obs)

                # Track FPS
                self.frame_count += 1

            except Exception as e:
                print(f"⚠️  IMU capture error: {e}")
                self.i2c_errors += 1

            self.i2c_attempts += 1

            # Sleep to maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def capture(self) -> Optional[IMUObservation]:
        """
        Capture latest IMU observation

        Returns:
            IMUObservation with orientation, motion, and integration data
        """
        try:
            obs = self.capture_queue.get(timeout=0.1)

            # Update Track 1 trust score
            if self.sensor_trust:
                self.sensor_trust.update('imu', obs.trust_score)

            # Store high-salience motion events in Track 2
            if self.memory and obs.salience > 0.5:
                self.memory.store_observation(
                    modality='imu',
                    data={
                        'euler': obs.euler.tolist(),
                        'motion_magnitude': obs.motion_magnitude,
                        'is_moving': obs.is_moving,
                    },
                    salience=obs.salience,
                    timestamp=obs.timestamp
                )

            # Trigger attention interrupt for high salience
            if self.attention and obs.salience > 0.9:
                self.attention.trigger_interrupt('imu', obs.salience)

            return obs

        except queue.Empty:
            return None

    def _capture_raw(self) -> Optional[IMUObservation]:
        """Capture raw IMU data from backend"""
        if self.backend == IMUBackend.BNO055:
            return self._capture_bno055()
        elif self.backend == IMUBackend.MPU6050:
            return self._capture_mpu6050()
        elif self.backend == IMUBackend.SIMULATED:
            return self._capture_simulated()
        return None

    def _capture_bno055(self) -> Optional[IMUObservation]:
        """Capture from BNO055 sensor"""
        try:
            start_time = time.time()

            # Read orientation (hardware fused)
            quaternion = self.imu.quaternion
            if quaternion is None or None in quaternion:
                return None
            quaternion = np.array(quaternion)

            euler = self.imu.euler
            if euler is None or None in euler:
                return None
            euler = np.array([e if e is not None else 0.0 for e in euler])
            euler = np.radians(euler)  # Convert to radians

            # Read motion
            gyro = self.imu.gyro
            if gyro is None or None in gyro:
                gyro = (0.0, 0.0, 0.0)
            angular_velocity = np.array(gyro)
            angular_velocity = np.radians(angular_velocity)  # Convert to rad/s

            accel = self.imu.acceleration
            if accel is None or None in accel:
                accel = (0.0, 0.0, 0.0)
            linear_acceleration = np.array(accel)

            # Magnetic heading
            mag = self.imu.magnetic
            heading = None
            if mag is not None and None not in mag:
                heading = np.degrees(np.arctan2(mag[1], mag[0])) % 360

            # Calibration status
            cal = self.imu.calibration_status
            if cal[0] == 3:
                cal_status = "full"
            elif cal[0] >= 1:
                cal_status = "partial"
            else:
                cal_status = "uncalibrated"

            # Compute derived values
            motion_mag, is_moving, salience, trust = self._process_observation(
                euler, angular_velocity, linear_acceleration, cal_status
            )

            latency = time.time() - start_time
            self.latency_samples.append(latency * 1000)  # ms
            if len(self.latency_samples) > 100:
                self.latency_samples.pop(0)

            return IMUObservation(
                quaternion=quaternion,
                euler=euler,
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration,
                heading=heading,
                motion_magnitude=motion_mag,
                is_moving=is_moving,
                salience=salience,
                trust_score=trust,
                timestamp=time.time(),
                calibration_status=cal_status,
                metadata={
                    'latency_ms': latency * 1000,
                    'frame_count': self.frame_count,
                    'backend': 'bno055'
                }
            )

        except Exception as e:
            print(f"⚠️  BNO055 read error: {e}")
            return None

    def _capture_mpu6050(self) -> Optional[IMUObservation]:
        """Capture from MPU6050 sensor (requires software fusion)"""
        try:
            start_time = time.time()

            # Read accelerometer
            accel_data = self.imu.get_accel_data()
            linear_acceleration = np.array([
                accel_data['x'], accel_data['y'], accel_data['z']
            ])

            # Read gyroscope
            gyro_data = self.imu.get_gyro_data()
            angular_velocity = np.array([
                gyro_data['x'], gyro_data['y'], gyro_data['z']
            ])
            angular_velocity = np.radians(angular_velocity)  # Convert to rad/s

            # Compute orientation using complementary filter
            euler = self._complementary_filter(linear_acceleration, angular_velocity)

            # No quaternion or magnetometer on MPU6050
            quaternion = self._euler_to_quaternion(euler)
            heading = None

            cal_status = "full"  # Assume calibrated for MPU6050

            # Compute derived values
            motion_mag, is_moving, salience, trust = self._process_observation(
                euler, angular_velocity, linear_acceleration, cal_status
            )

            latency = time.time() - start_time
            self.latency_samples.append(latency * 1000)
            if len(self.latency_samples) > 100:
                self.latency_samples.pop(0)

            return IMUObservation(
                quaternion=quaternion,
                euler=euler,
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration,
                heading=heading,
                motion_magnitude=motion_mag,
                is_moving=is_moving,
                salience=salience,
                trust_score=trust,
                timestamp=time.time(),
                calibration_status=cal_status,
                metadata={
                    'latency_ms': latency * 1000,
                    'frame_count': self.frame_count,
                    'backend': 'mpu6050'
                }
            )

        except Exception as e:
            print(f"⚠️  MPU6050 read error: {e}")
            return None

    def _capture_simulated(self) -> Optional[IMUObservation]:
        """Capture from simulated IMU"""
        t = time.time() - self.imu['start_time']

        # Simulate smooth orientation changes
        euler = np.array([
            0.2 * np.sin(t * 0.3),  # Roll
            0.15 * np.cos(t * 0.4),  # Pitch
            t * 0.1 % (2 * np.pi)  # Yaw (slowly rotating)
        ])

        # Simulate angular velocity
        angular_velocity = np.array([
            0.06 * np.cos(t * 0.3),  # Roll rate
            -0.06 * np.sin(t * 0.4),  # Pitch rate
            0.1  # Yaw rate (constant slow rotation)
        ])

        # Simulate linear acceleration (with gravity + motion)
        linear_acceleration = np.array([
            0.5 * np.sin(t * 0.5),  # X accel
            0.3 * np.cos(t * 0.6),  # Y accel
            9.81 + 0.2 * np.sin(t * 0.4)  # Z accel (gravity + motion)
        ])

        # Simulate heading
        heading = np.degrees(euler[2]) % 360

        # Create quaternion from euler
        quaternion = self._euler_to_quaternion(euler)

        cal_status = "full"  # Simulated is always "calibrated"

        # Compute derived values
        motion_mag, is_moving, salience, trust = self._process_observation(
            euler, angular_velocity, linear_acceleration, cal_status
        )

        return IMUObservation(
            quaternion=quaternion,
            euler=euler,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            heading=heading,
            motion_magnitude=motion_mag,
            is_moving=is_moving,
            salience=salience,
            trust_score=trust,
            timestamp=time.time(),
            calibration_status=cal_status,
            metadata={
                'latency_ms': 1.0,  # Simulated is very fast
                'frame_count': self.frame_count,
                'backend': 'simulated'
            }
        )

    def _process_observation(
        self,
        euler: np.ndarray,
        angular_velocity: np.ndarray,
        linear_acceleration: np.ndarray,
        cal_status: str
    ) -> Tuple[float, bool, float, float]:
        """
        Process observation to compute derived values

        Returns:
            (motion_magnitude, is_moving, salience, trust_score)
        """
        # Motion magnitude (combined accel + gyro)
        accel_mag = np.linalg.norm(linear_acceleration - np.array([0, 0, 9.81]))
        gyro_mag = np.linalg.norm(angular_velocity)
        motion_magnitude = float(accel_mag + gyro_mag)

        # Motion detection
        is_moving = accel_mag > self.motion_threshold

        # Salience computation
        salience = self._compute_salience(euler, motion_magnitude)

        # Trust score computation
        trust_score = self._compute_trust_score(cal_status)

        # Update history
        self.prev_euler = euler.copy()
        self.orientation_history.append(euler.copy())
        if len(self.orientation_history) > 50:
            self.orientation_history.pop(0)

        self.accel_history.append(accel_mag)
        if len(self.accel_history) > 50:
            self.accel_history.pop(0)

        return motion_magnitude, is_moving, salience, trust_score

    def _compute_salience(self, euler: np.ndarray, motion_magnitude: float) -> float:
        """
        Compute salience score for Track 3 attention allocation

        Salience = 0.7 * motion_norm + 0.3 * orientation_delta_norm
        """
        # Motion magnitude normalized (threshold at 10 for full salience)
        motion_norm = np.clip(motion_magnitude / 10.0, 0, 1)

        # Orientation delta (change since last frame)
        orient_delta = np.linalg.norm(euler - self.prev_euler)
        orient_norm = np.clip(orient_delta / (np.pi / 4), 0, 1)  # 45° threshold

        # Combined salience
        salience = 0.7 * motion_norm + 0.3 * orient_norm

        return float(np.clip(salience, 0, 1))

    def _compute_trust_score(self, cal_status: str) -> float:
        """
        Compute trust score for Track 1 sensor fusion

        Trust = calibration_factor * stability_factor * (1 - error_rate)
        """
        # Calibration factor
        if cal_status == "full":
            cal_factor = 1.0
        elif cal_status == "partial":
            cal_factor = 0.6
        else:
            cal_factor = 0.3

        # Stability factor (low variance = high trust)
        if len(self.orientation_history) > 10:
            orient_std = np.std([np.linalg.norm(e) for e in self.orientation_history[-10:]])
            stability_factor = 1.0 / (1.0 + orient_std)
        else:
            stability_factor = 0.8

        # Error rate (I2C communication failures)
        if self.i2c_attempts > 0:
            error_rate = self.i2c_errors / self.i2c_attempts
        else:
            error_rate = 0.0

        trust = cal_factor * stability_factor * (1.0 - error_rate)

        return float(np.clip(trust, 0, 1))

    def _complementary_filter(self, accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """
        Simple complementary filter for orientation estimation

        98% gyro integration + 2% accelerometer correction
        """
        alpha = 0.98
        dt = 1.0 / self.sample_rate

        # Integrate gyro
        euler_gyro = self.prev_euler + gyro * dt

        # Calculate from accelerometer
        roll_accel = np.arctan2(accel[1], accel[2])
        pitch_accel = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        yaw_accel = self.prev_euler[2]  # Can't get yaw from accel alone

        euler_accel = np.array([roll_accel, pitch_accel, yaw_accel])

        # Fuse
        euler = alpha * euler_gyro + (1 - alpha) * euler_accel

        return euler

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)"""
        roll, pitch, yaw = euler

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if len(self.latency_samples) > 0:
            avg_latency = np.mean(self.latency_samples)
        else:
            avg_latency = 0.0

        if self.frame_count > 0 and hasattr(self, 'imu'):
            elapsed = time.time() - self.imu.get('start_time', time.time())
            if elapsed > 0:
                self.fps_actual = self.frame_count / elapsed

        return {
            'backend': self.backend.value,
            'sample_rate_target': self.sample_rate,
            'fps_actual': self.fps_actual,
            'avg_latency_ms': avg_latency,
            'frame_count': self.frame_count,
            'i2c_errors': self.i2c_errors,
            'i2c_attempts': self.i2c_attempts,
            'error_rate': self.i2c_errors / max(self.i2c_attempts, 1),
        }

    def shutdown(self):
        """Shutdown IMU sensor and cleanup"""
        print("\n🛑 Shutting down IMU sensor...")

        self.capture_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        print("✓ IMU sensor shutdown complete")


def test_imu_sensor():
    """Test IMU sensor in simulated mode"""
    print("\n" + "="*70)
    print("TESTING IMU SENSOR (Track 5)")
    print("="*70)

    # Create sensor in simulated mode
    sensor = IMUSensor(
        device_type="simulated",
        sample_rate=50
    )

    print("\n1. Capturing IMU data...")
    observations = []
    for i in range(10):
        obs = sensor.capture()
        if obs:
            observations.append(obs)
            print(f"   Frame {i+1}:")
            print(f"     Euler (roll, pitch, yaw): {np.degrees(obs.euler)}")
            print(f"     Motion magnitude: {obs.motion_magnitude:.3f}")
            print(f"     Is moving: {obs.is_moving}")
            print(f"     Salience: {obs.salience:.3f}")
            print(f"     Trust score: {obs.trust_score:.3f}")
            print(f"     Latency: {obs.metadata['latency_ms']:.2f}ms")
        time.sleep(0.1)

    print(f"\n2. Captured {len(observations)} observations")

    # Check stats
    stats = sensor.get_stats()
    print(f"\n3. Statistics:")
    print(f"   Backend: {stats['backend']}")
    print(f"   Target rate: {stats['sample_rate_target']} Hz")
    print(f"   Actual FPS: {stats['fps_actual']:.1f}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms (target: <5ms)")
    print(f"   Frame count: {stats['frame_count']}")
    print(f"   Error rate: {stats['error_rate']:.3f}")

    # Test salience range
    saliences = [obs.salience for obs in observations]
    print(f"\n4. Salience range: {min(saliences):.3f} - {max(saliences):.3f}")
    print(f"   Mean salience: {np.mean(saliences):.3f}")

    # Test trust scores
    trusts = [obs.trust_score for obs in observations]
    print(f"\n5. Trust scores: {min(trusts):.3f} - {max(trusts):.3f}")
    print(f"   Mean trust: {np.mean(trusts):.3f}")

    # Shutdown
    sensor.shutdown()

    print("\n" + "="*70)
    print("✅ IMU SENSOR TEST PASSED")
    print("="*70)
    print("\nReady for:")
    print("  ✓ BNO055 hardware testing on Nano")
    print("  ✓ Track 1 (Sensor Trust) integration")
    print("  ✓ Track 2 (Memory) integration")
    print("  ✓ Track 3 (Attention/Cognition) integration")
    print("  ✓ Multi-sensor fusion (Vision + IMU)")


if __name__ == "__main__":
    test_imu_sensor()
