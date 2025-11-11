# Track 4: Nano Deployment Instructions

**Phase 2: CSI Camera Testing on Jetson Nano**

Date: 2025-11-10
Status: Ready for deployment

---

## Prerequisites

### On Nano (Verify These)

**Hardware:**
- Jetson Nano with CSI camera connected
- Camera should be visible via `v4l2-ctl --list-devices`
- Power supply adequate for camera operation

**Software:**
- OpenCV with GStreamer support (should be pre-installed on Nano)
- PyTorch (check if installed or install lightweight version)
- NumPy (should be available)

**Verify Commands:**
```bash
# Check CSI camera
ls /dev/video*

# Check OpenCV GStreamer support
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer

# Check PyTorch
python3 -c "import torch; print(torch.__version__)"

# Check platform detection
cat /proc/device-tree/model
# Should output: jetson-nano...
```

---

## Deployment Steps

### 1. Transfer Code to Nano

From Thor (this machine):
```bash
# Sync HRM repo to Nano
rsync -avz --exclude '.git' --exclude 'sage/examples/*.pid' \
  /home/dp/ai-workspace/HRM/ \
  nano:/home/dp/ai-workspace/HRM/

# Or use git (if Nano has git access)
# On Nano:
cd /home/dp/ai-workspace/HRM
git pull origin main
```

### 2. Install Dependencies (if needed)

On Nano:
```bash
cd /home/dp/ai-workspace/HRM

# Install Python dependencies
pip3 install --user numpy

# PyTorch - use Jetson-optimized version if not installed
# See: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# Usually pre-installed on JetPack
```

### 3. Test Vision Sensor (Simulated Mode First)

On Nano:
```bash
cd /home/dp/ai-workspace/HRM

# Test in simulated mode (no camera required)
python3 sage/sensors/vision_sensor.py
```

**Expected Output:**
```
📷 Vision sensor: simulated (ID 0) @ 30fps
  Simulated camera mode (synthetic frames)
✓ Vision sensor initialized: simulated

1. Capturing frames...
   Frame 1: salience=0.XXX, latency=X.XXms, trust=0.XXX
   ...
   Frame 10: salience=0.XXX, latency=X.XXms, trust=0.XXX

✅ VISION SENSOR TEST PASSED
```

### 4. Test CSI Camera

On Nano:
```bash
cd /home/dp/ai-workspace/HRM

# Test CSI camera (auto-detection)
python3 -c "
from sage.sensors.vision_sensor import VisionSensor
import time

print('Testing CSI camera...')
sensor = VisionSensor(backend='auto')  # Should detect CSI

# Capture 30 frames (1 second at 30 FPS)
for i in range(30):
    obs = sensor.capture()
    if obs:
        print(f'Frame {i+1}: {obs.frame.shape}, '
              f'salience={obs.salience:.3f}, '
              f'latency={obs.metadata[\"latency_ms\"]:.2f}ms')
    time.sleep(0.033)

# Print stats
stats = sensor.get_stats()
print(f'\nStatistics:')
print(f'  Backend: {stats[\"backend\"]}')
print(f'  FPS: {stats[\"fps_actual\"]} (target {stats[\"fps_target\"]})')
print(f'  Avg latency: {stats[\"avg_latency_ms\"]:.2f}ms')
print(f'  Trust score: {stats[\"trust_score\"]:.3f}')

sensor.shutdown()
"
```

**Expected Output:**
```
📷 Vision sensor: csi (ID 0) @ 30fps
  CSI pipeline: nvarguscamerasrc sensor-id=0
✓ Vision sensor initialized: csi

Testing CSI camera...
Frame 1: (480, 640, 3), salience=0.XXX, latency=XX.XXms
...
Frame 30: (480, 640, 3), salience=0.XXX, latency=XX.XXms

Statistics:
  Backend: csi
  FPS: ~30.0 (target 30)
  Avg latency: <30ms (GOAL: <30ms for CSI)
  Trust score: >0.8 (high trust = consistent performance)
```

### 5. Validate Performance Targets

**Track 4 Performance Targets:**

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Frame capture | <30ms (CSI) | Check `latency_ms` in metadata |
| Frame rate | 30 FPS | Check `fps_actual` in stats |
| Startup time | <2 seconds | Time from import to first frame |
| Memory usage | <500MB | `nvidia-smi` or `htop` while running |

**Performance Test Script:**

On Nano:
```bash
cd /home/dp/ai-workspace/HRM

python3 -c "
import time
from sage.sensors.vision_sensor import VisionSensor

# Measure startup time
start = time.time()
sensor = VisionSensor(backend='csi')
startup_time = time.time() - start
print(f'Startup time: {startup_time:.2f}s (target: <2s)')

# Capture 300 frames (10 seconds)
latencies = []
for i in range(300):
    obs = sensor.capture()
    if obs:
        latencies.append(obs.metadata['latency_ms'])

# Analyze latencies
import numpy as np
print(f'\nLatency statistics:')
print(f'  Mean: {np.mean(latencies):.2f}ms (target: <30ms)')
print(f'  Median: {np.median(latencies):.2f}ms')
print(f'  95th percentile: {np.percentile(latencies, 95):.2f}ms')
print(f'  Max: {np.max(latencies):.2f}ms')

# FPS
stats = sensor.get_stats()
print(f'\nFPS: {stats[\"fps_actual\"]:.2f} (target: 30)')
print(f'Trust: {stats[\"trust_score\"]:.3f}')

sensor.shutdown()
"
```

---

## Troubleshooting

### Issue: Camera Not Detected

**Symptoms:**
```
⚠️  No camera found, using simulated mode
```

**Solutions:**
1. Check camera connection:
   ```bash
   ls /dev/video*
   # Should show /dev/video0
   ```

2. Check if camera is being used by another process:
   ```bash
   sudo fuser /dev/video0
   ```

3. Try manual CSI test:
   ```bash
   gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvoverlaysink
   ```

### Issue: GStreamer Pipeline Fails

**Symptoms:**
```
RuntimeError: Failed to open camera (backend: csi)
```

**Solutions:**
1. Check GStreamer installation:
   ```bash
   gst-inspect-1.0 nvarguscamerasrc
   ```

2. Try simpler pipeline:
   ```bash
   gst-launch-1.0 nvarguscamerasrc ! fakesink
   ```

3. Check nvarguscamerasrc permissions:
   ```bash
   sudo usermod -a -G video $USER
   # Then log out and back in
   ```

### Issue: Low FPS or High Latency

**Symptoms:**
- FPS < 20
- Latency > 50ms

**Solutions:**
1. Check system load:
   ```bash
   htop
   # CPU should not be maxed out
   ```

2. Check GPU memory:
   ```bash
   nvidia-smi
   # Should have free memory
   ```

3. Try lower resolution:
   ```python
   sensor = VisionSensor(
       backend='csi',
       display_resolution=(320, 240)  # Lower resolution
   )
   ```

4. Disable desktop environment (if testing headless):
   ```bash
   sudo systemctl stop gdm3  # Or lightdm
   ```

### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solutions:**
1. Install OpenCV:
   ```bash
   sudo apt-get install python3-opencv
   ```

2. Or use pip:
   ```bash
   pip3 install --user opencv-python
   ```

---

## Success Criteria

**Phase 2 is COMPLETE when:**

✅ CSI camera detected and opened
✅ Frames captured at ~30 FPS
✅ Average latency < 30ms
✅ Trust score > 0.7 (consistent performance)
✅ No frame corruption or crashes
✅ Startup time < 2 seconds

**Then proceed to Phase 3: Integration Testing**

---

## Data Collection

**Record these metrics for documentation:**

```bash
# On Nano, after successful testing:
cd /home/dp/ai-workspace/HRM

python3 -c "
from sage.sensors.vision_sensor import VisionSensor
import time

sensor = VisionSensor(backend='csi')

# Capture 600 frames (20 seconds)
for i in range(600):
    obs = sensor.capture()

stats = sensor.get_stats()
print('\n=== TRACK 4 PHASE 2 RESULTS ===')
print(f'Platform: Jetson Nano')
print(f'Backend: {stats[\"backend\"]}')
print(f'FPS: {stats[\"fps_actual\"]:.2f}')
print(f'Latency: {stats[\"avg_latency_ms\"]:.2f}ms')
print(f'Trust: {stats[\"trust_score\"]:.3f}')
print(f'Frames: {stats[\"frame_count\"]}')
print('===============================\n')

sensor.shutdown()
" | tee track4_phase2_results.txt
```

**Save results and commit:**
```bash
git add private-context/track4_phase2_results.txt
git commit -m "Track 4 Phase 2: Nano CSI camera test results"
```

---

## Next Steps After Successful Testing

Once Phase 2 is validated:

1. **Phase 3: Integration Testing**
   - Test with Track 1 (sensor trust tracking)
   - Test with Track 2 (memory storage)
   - Test with Track 3 (attention allocation)

2. **Phase 4: Performance Optimization**
   - Profile bottlenecks
   - Optimize preprocessing
   - Tune salience computation
   - Minimize memory usage

3. **Full SAGE Integration**
   - Multi-sensor fusion (vision + IMU when ready)
   - Real-time attention-driven capture
   - Memory-guided visual search

---

## Contact Points

If you encounter issues:

1. Check existing camera code: `sage/irp/plugins/camera_sensor_impl.py`
2. Review GStreamer pipeline in `vision_sensor.py:217-235`
3. Reference user guidance: `NANO_HARDWARE_ANSWERS.md`

**The vision sensor is ready. Time to test on real hardware!** 🚀
