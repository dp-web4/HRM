# Track 5: IMU Sensor Architecture Design

**Track:** 5 - Inertial Measurement Unit (IMU) Sensor
**Status:** Architecture Phase
**Date:** 2025-11-11
**Autonomous Session:** #27

---

## Overview

Track 5 implements IMU sensor integration for real-time orientation, acceleration, and motion tracking on Jetson Nano. IMU provides critical spatial awareness for embodied AI consciousness.

**Goal:** Real-time IMU data capture with Track 1-3 integration, enabling spatial awareness and motion tracking for SAGE consciousness.

---

## IMU Sensor Capabilities

### What is an IMU?

An **Inertial Measurement Unit (IMU)** combines:
- **Accelerometer** - Measures linear acceleration (3-axis)
- **Gyroscope** - Measures angular velocity (3-axis)
- **Magnetometer** - Measures magnetic field/heading (3-axis, optional)

### Common IMU Hardware on Jetson Nano

**BNO055** (9-DOF, absolute orientation):
- Accelerometer: ±2g/±4g/±8g/±16g
- Gyroscope: ±125°/s to ±2000°/s
- Magnetometer: 3-axis
- Built-in sensor fusion (quaternions)
- I2C/UART interface
- ~100 Hz update rate

**MPU6050/MPU9250** (6/9-DOF):
- Accelerometer: ±2g/±4g/±8g/±16g
- Gyroscope: ±250°/s to ±2000°/s
- MPU9250 adds magnetometer
- I2C interface
- ~1000 Hz max rate
- Requires external fusion

**Target:** BNO055 for absolute orientation (easier integration)
**Fallback:** MPU6050/MPU9250 if needed

---

## Track 5 Architecture

### Component Breakdown

**1. IMU Sensor Interface** (`sage/sensors/imu_sensor.py`)
- Hardware abstraction (BNO055, MPU6050, etc.)
- I2C communication
- Raw data capture (accel, gyro, mag)
- Sensor fusion (quaternions, euler angles)
- Calibration handling

**2. Motion Processor**
- Orientation tracking (roll, pitch, yaw)
- Motion detection (moving vs. stationary)
- Acceleration integration (velocity estimation)
- Vibration filtering (high-frequency noise)

**3. Track 1-3 Integration**
- Track 1: Sensor trust based on calibration + stability
- Track 2: Store motion events (sudden movements, orientation changes)
- Track 3: Salience from motion magnitude + orientation delta

**4. Real-Time Performance**
- Target: <5ms latency (IMU is fast)
- Rate: 50-100 Hz (balance speed vs. power)
- Background thread for continuous capture
- Kalman/Complementary filter for fusion

---

## Data Flow

```
IMU Hardware
    ↓
I2C Communication (smbus2/adafruit-circuitpython)
    ↓
Raw Sensor Data (accel, gyro, mag)
    ↓
Sensor Fusion (quaternions → euler angles)
    ↓
Motion Analysis (delta detection, magnitude)
    ↓
Salience Computation (motion intensity → 0-1 score)
    ↓
Track 1-3 Integration
    ├→ Track 1: Trust score (calibration + stability)
    ├→ Track 2: Motion event storage
    └→ Track 3: Attention allocation
```

---

## IMUObservation Dataclass

```python
@dataclass
class IMUObservation:
    """IMU sensor observation integrating with Track 1-3"""

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
```

---

## IMU Sensor Class

### Initialization

```python
class IMUSensor:
    """Real-time IMU sensor for SAGE consciousness"""

    def __init__(
        self,
        device_type: str = "auto",  # auto, bno055, mpu6050, simulated
        i2c_address: int = 0x28,  # BNO055 default
        sample_rate: int = 50,  # Hz
        sensor_trust: Optional[SensorTrust] = None,  # Track 1
        memory_system: Optional[Any] = None,  # Track 2
        attention_manager: Optional[Any] = None,  # Track 3
    ):
        self.device_type = self._detect_device() if device_type == "auto" else device_type
        self.sample_rate = sample_rate

        # Track 1-3 integration
        self.sensor_trust = sensor_trust
        self.memory = memory_system
        self.attention = attention_manager

        # Initialize hardware
        self.imu = self._initialize_hardware()

        # Background capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_queue = queue.Queue(maxsize=2)

        # Performance tracking
        self.fps_actual = 0.0
        self.latency_avg = 0.0
```

### Auto-Detection Priority

1. **BNO055** (I2C 0x28 or 0x29) - Best option, built-in fusion
2. **MPU6050** (I2C 0x68 or 0x69) - Common, needs fusion
3. **MPU9250** (I2C 0x68 or 0x69) - With magnetometer
4. **Simulated** - Fallback for testing without hardware

### Core Methods

**capture() → IMUObservation**
- Get latest IMU reading from queue
- Compute motion magnitude
- Calculate salience (motion intensity + orientation change)
- Update Track 1 trust score
- Return IMUObservation

**_capture_loop() (background thread)**
- Continuous IMU polling at sample_rate
- Push to queue (drop old if full for low latency)
- Handle I2C errors gracefully
- Track performance (FPS, latency)

**_compute_salience(obs) → float**
- Motion magnitude: sqrt(accel² + gyro²) normalized
- Orientation delta: change since last frame
- Salience = motion_weight * motion_mag + orient_weight * orient_delta
- Clamp to [0, 1]

**_compute_trust_score() → float**
- Calibration status: 0.3 (uncal) → 1.0 (full cal)
- Stability: Low variance over recent samples → higher trust
- Error rate: I2C failures → lower trust
- Trust = calibration * stability * (1 - error_rate)

---

## Sensor Fusion Approaches

### Option 1: BNO055 Built-in Fusion (Recommended)

**Pros:**
- Hardware sensor fusion (quaternions, euler angles)
- Automatic calibration
- Low CPU overhead
- Proven accuracy

**Cons:**
- Specific hardware required
- Less flexibility

**Implementation:**
```python
# BNO055 provides fused orientation directly
quaternion = imu.quaternion  # (w, x, y, z)
euler = imu.euler  # (roll, pitch, yaw)
```

### Option 2: Complementary Filter (MPU6050)

**Pros:**
- Simple, fast
- Works with any IMU
- Low computational cost

**Cons:**
- Gyro drift over time
- No magnetometer correction

**Implementation:**
```python
# Complementary filter: 98% gyro, 2% accel
alpha = 0.98
dt = 1.0 / sample_rate

# Integrate gyro
angle_gyro = angle_prev + gyro * dt

# Calculate from accel
angle_accel = atan2(accel_y, accel_z)

# Fuse
angle = alpha * angle_gyro + (1 - alpha) * angle_accel
```

### Option 3: Madgwick/Mahony Filter (MPU9250)

**Pros:**
- Uses magnetometer for yaw correction
- No gyro drift
- Industry standard

**Cons:**
- More complex
- Requires magnetometer calibration

**Libraries:**
- `imufusion` (Python)
- `madgwick` (scipy-based)

---

## Hardware Integration

### I2C Communication (Linux)

**Library:** `smbus2` or `adafruit-circuitpython-bno055`

**BNO055 Setup:**
```python
import board
import busio
import adafruit_bno055

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

# Read orientation
quaternion = sensor.quaternion
euler = sensor.euler
gyro = sensor.gyro
accel = sensor.acceleration
```

**MPU6050 Setup:**
```python
from mpu6050 import mpu6050

sensor = mpu6050(0x68)

accel_data = sensor.get_accel_data()
gyro_data = sensor.get_gyro_data()
```

### Calibration Handling

**BNO055 Auto-Calibration:**
- Move device through figure-8 motions
- Twist on all axes
- System/gyro/accel/mag status (0-3)
- Save/restore calibration coefficients

**MPU6050 Calibration:**
- Collect bias samples at rest
- Subtract offsets from raw readings
- Store calibration in config file

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Latency** | <5ms | IMU sensors are fast (I2C ~400kHz) |
| **Sample Rate** | 50-100 Hz | Balance responsiveness vs. power |
| **Startup Time** | <1 second | Fast initialization critical |
| **CPU Usage** | <5% | Should be minimal overhead |
| **Trust Score** | >0.9 (calibrated) | High trust for well-calibrated IMU |

**Why Fast?**
- Motion detection requires responsiveness
- Orientation updates for attention allocation
- Integration for velocity estimation

---

## Track 1 Integration: Sensor Trust

**Trust Factors:**

1. **Calibration Status** (0.0-1.0)
   - Uncalibrated: 0.3
   - Partial: 0.6
   - Full: 1.0

2. **Stability Score** (0.0-1.0)
   - Low variance over recent samples
   - Stable readings → higher trust
   - Erratic readings → lower trust

3. **I2C Error Rate** (0.0-1.0)
   - Communication failures reduce trust
   - error_rate = failed_reads / total_reads
   - trust_factor = 1 - error_rate

**Trust Computation:**
```python
trust = calibration_factor * stability_factor * (1 - error_rate)
```

**Track 1 Registration:**
```python
if self.sensor_trust:
    self.sensor_trust.register_sensor('imu', initial_trust=0.7)

# Each capture:
trust_score = self._compute_trust_score()
if self.sensor_trust:
    self.sensor_trust.update('imu', trust_score)
```

---

## Track 2 Integration: Memory Storage

**High-Salience Motion Events:**

Store in memory when:
- Salience > 0.5 (significant motion)
- Large orientation change (>30° rotation)
- Sudden acceleration (>2g spike)

**Memory Entry:**
```python
if obs.salience > 0.5 and self.memory:
    self.memory.store_observation(
        modality='imu',
        data={
            'euler': obs.euler,
            'motion_magnitude': obs.motion_magnitude,
            'is_moving': obs.is_moving,
        },
        salience=obs.salience,
        timestamp=obs.timestamp
    )
```

**Use Cases:**
- Retrieve past motion patterns
- Detect unusual movements
- Context for spatial reasoning

---

## Track 3 Integration: Attention Allocation

**Salience Computation:**

Motion salience drives attention:
- High motion → high salience → attention allocation
- Stable/stationary → low salience → background monitoring

**Salience Formula:**
```python
# Motion magnitude (0-1 normalized)
motion_norm = np.clip(motion_magnitude / 10.0, 0, 1)  # 10 m/s² threshold

# Orientation delta (0-1 normalized)
orient_delta = np.linalg.norm(euler - prev_euler)
orient_norm = np.clip(orient_delta / (np.pi/4), 0, 1)  # 45° threshold

# Combined salience
salience = 0.7 * motion_norm + 0.3 * orient_norm
```

**Attention Integration:**
```python
if self.attention:
    # High salience triggers attention interrupt
    if obs.salience > 0.9:
        self.attention.trigger_interrupt('imu', obs.salience)
```

---

## Testing Strategy

### Phase 1: Simulated IMU (Thor)
- ✅ Create IMUSensor class with simulated backend
- ✅ Test API and data structures
- ✅ Validate Track 1-3 integration points
- ✅ Measure performance (latency, CPU)

### Phase 2: Hardware Testing (Nano)
- ⏳ Deploy to Jetson Nano
- ⏳ Test with BNO055 IMU
- ⏳ Validate I2C communication
- ⏳ Measure real-world performance

### Phase 3: Calibration & Tuning (Nano)
- ⏳ Calibration procedures
- ⏳ Sensor fusion accuracy
- ⏳ Motion detection thresholds
- ⏳ Salience computation tuning

### Phase 4: Integration Testing (Nano)
- ⏳ Multi-sensor fusion (Vision + IMU)
- ⏳ Spatial awareness with orientation
- ⏳ Motion-guided attention
- ⏳ Full SAGE cognitive cycle

---

## Hardware Requirements

**Required:**
- Jetson Nano with I2C enabled
- IMU sensor (BNO055 recommended, MPU6050 fallback)
- I2C connection (SCL, SDA, GND, VCC)

**Optional:**
- Mounting bracket for stable orientation
- Vibration dampening (if noisy environment)

**Software Dependencies:**
```bash
# For BNO055
pip install adafruit-circuitpython-bno055

# For MPU6050
pip install mpu6050-raspberrypi

# For sensor fusion (if needed)
pip install imufusion ahrs
```

---

## Implementation Phases

### Phase 1: Core Implementation (Thor)
**Files to Create:**
- `sage/sensors/imu_sensor.py` (main implementation)
- `sage/sensors/imu_fusion.py` (sensor fusion utilities, if needed)

**What to Implement:**
- IMUSensor class with simulated backend
- IMUObservation dataclass
- Track 1-3 integration hooks
- Performance tracking
- Test function

**Validation:**
- Run on Thor in simulated mode
- Verify API works
- Check Track 1-3 integration
- Measure simulated performance

### Phase 2: Hardware Deployment (Nano)
**Tasks:**
- Deploy code to Nano
- Test BNO055 connection
- Validate I2C communication
- Real-world performance measurement

**Success Criteria:**
- IMU detected and initialized
- Data capture at 50+ Hz
- Latency <5ms
- Trust score >0.9 (calibrated)

### Phase 3: Sensor Fusion (Nano)
**Tasks:**
- Implement/validate orientation fusion
- Calibration procedures
- Motion detection tuning
- Salience computation refinement

**Success Criteria:**
- Accurate orientation (<5° error)
- Reliable motion detection
- Appropriate salience scoring
- Stable performance over time

### Phase 4: Multi-Sensor Integration (Nano)
**Tasks:**
- Test Vision + IMU fusion
- Spatial awareness scenarios
- Attention allocation with multiple sensors
- Full SAGE cognitive cycle

**Success Criteria:**
- Coherent multi-sensor fusion
- Attention responds to motion events
- Memory stores motion patterns
- Trust-weighted sensor combination

---

## Code Structure

```
sage/sensors/imu_sensor.py
├── IMUObservation (dataclass)
│   ├── quaternion, euler
│   ├── angular_velocity, linear_acceleration
│   ├── motion_magnitude, is_moving
│   ├── salience, trust_score
│   └── metadata
│
├── IMUBackend (enum)
│   ├── BNO055
│   ├── MPU6050
│   ├── MPU9250
│   └── SIMULATED
│
└── IMUSensor (class)
    ├── __init__(device_type, i2c_address, sample_rate, tracks 1-3)
    ├── capture() → IMUObservation
    ├── _capture_loop() (background thread)
    ├── _detect_device() → IMUBackend
    ├── _initialize_hardware() → imu device
    ├── _compute_salience(obs) → float
    ├── _compute_trust_score() → float
    ├── _detect_motion(obs) → bool
    ├── get_stats() → Dict
    ├── calibrate() (if needed)
    └── shutdown()
```

---

## Comparison to Track 4 (Vision)

| Aspect | Track 4 (Vision) | Track 5 (IMU) |
|--------|------------------|---------------|
| **Latency** | <30ms (CSI) | <5ms (I2C) |
| **Data Rate** | 30 FPS | 50-100 Hz |
| **Data Size** | Large (640x480 RGB) | Small (9-15 floats) |
| **Processing** | Heavy (salience, features) | Light (fusion, magnitude) |
| **Interface** | GStreamer/OpenCV | I2C (smbus2) |
| **Calibration** | Auto (camera) | Manual (IMU) |
| **Trust Basis** | FPS consistency | Calibration + stability |

**Similarity:** Both follow same Track 1-3 integration pattern

---

## Next Steps (After Architecture)

1. **Phase 1 Implementation** (Thor):
   - Create `sage/sensors/imu_sensor.py`
   - Implement simulated backend
   - Add Track 1-3 integration
   - Write test function

2. **Documentation** (Thor):
   - Implementation guide
   - API documentation
   - Testing instructions

3. **Phase 2 Deployment** (Nano - requires hardware):
   - Deploy to Jetson Nano
   - Test with BNO055
   - Validate performance
   - Record metrics

4. **Phase 3-4** (Nano):
   - Calibration procedures
   - Multi-sensor integration
   - Full system testing

---

## References

**Hardware:**
- BNO055 Datasheet: [Bosch Sensortec BNO055](https://www.bosch-sensortec.com/products/smart-sensors/bno055/)
- MPU6050 Datasheet: [InvenSense MPU-6050](https://invensense.tdk.com/products/motion-tracking/6-axis/mpu-6050/)

**Libraries:**
- Adafruit CircuitPython BNO055: [GitHub](https://github.com/adafruit/Adafruit_CircuitPython_BNO055)
- IMU Fusion: [imufusion](https://github.com/xioTechnologies/Fusion)
- AHRS Algorithms: [ahrs](https://github.com/Mayitzin/ahrs)

**User Guidance:**
- NANO_HARDWARE_ANSWERS.md - Reference past IMU work in repo
- Development pattern: Find existing → Extend → Test on Nano

**Existing Code:**
- sage/sensors/proprioception_sensor.py - Robot arm proprioception (not IMU, but similar orientation tracking)
- sage/cognition/test_integration.py - IMU referenced in scenarios

---

**Track 5 Architecture: COMPLETE**

Ready for Phase 1 implementation (simulated mode on Thor).

**Pattern continues:** Architecture (Session #27) → Implementation → Testing on Nano

---

**Status:** Architecture design complete, ready for implementation! 🚀
