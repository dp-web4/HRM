# Visual Monitor Performance Analysis

## Test Results Summary

### Platform: WSL2 on Windows 11
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **CPU**: AMD Ryzen (WSL2 virtualized)
- **Camera**: USB webcam via usbipd passthrough
- **Resolution**: 640x480

## Implementation Comparison

### fast_camera_monitor.py
**Achievement: ~20 FPS stable**

Key optimizations that made the difference:
1. **Downsampled motion detection**
   - Process at 160x120 instead of 640x480
   - 16x reduction in pixels to process
   - Scale results back to full resolution

2. **Frame skipping**
   - Motion detection every 2nd frame
   - Maintains visual smoothness at 20 FPS

3. **Attention smoothing**
   - α=0.7 exponential smoothing
   - Eliminates jitter in attention box
   - Provides fluid motion tracking

4. **Minimal overlay**
   - Only essential information displayed
   - Polyline trail instead of alpha blending
   - Simple text rendering

**Performance Profile:**
```
Motion Detection: ~5ms per frame (downsampled)
Overlay Render:   ~2ms per frame
Camera Capture:   ~30ms per frame (OpenCV direct)
Total:           ~37ms = 27 FPS theoretical
Actual:          ~20 FPS (with system overhead)
```

### live_camera_monitor.py
**Achievement: 5-10 FPS with full features**

Design decisions:
1. **Threaded capture**
   - Separate thread for fswebcam capture
   - Non-blocking frame queue
   - Handles USB passthrough latency

2. **Selective processing**
   - Motion every 3rd frame
   - IRP every 30th frame
   - Cached telemetry between updates

3. **Rich overlay**
   - Detailed telemetry display
   - Trust score visualization
   - Corner bracket styling

**Performance Profile:**
```
fswebcam Capture: ~100ms per frame
Motion Detection: ~15ms per frame (full res)
IRP Processing:   ~30ms (when active)
Overlay Render:   ~10ms per frame
Total:           ~125ms = 8 FPS theoretical
Actual:          5-10 FPS (USB passthrough overhead)
```

### wsl_camera_attention.py
**Achievement: ~5 FPS reliable capture**

Compatibility focus:
1. **fswebcam reliability**
   - Works around OpenCV V4L2 timeouts
   - Handles USB disconnections gracefully
   - Per-frame camera open/close

2. **Batch processing**
   - Saves frames for later analysis
   - Creates summary montages
   - Suitable for non-interactive use

**Performance Profile:**
```
fswebcam Capture: ~150ms per frame (open/close overhead)
Motion Detection: ~20ms per frame
File Save:        ~10ms per frame
Total:           ~180ms = 5.5 FPS theoretical
Actual:          ~5 FPS (consistent)
```

## Bottleneck Analysis

### USB Passthrough Overhead (WSL2)
- Native Linux: Camera capture ~10ms
- WSL2 direct: Camera capture ~30ms
- WSL2 fswebcam: Camera capture ~100-150ms

**3-15x slower due to virtualization layer**

### Motion Detection Scaling
- Full resolution (640x480): ~20ms
- Downsampled (160x120): ~5ms
- **4x linear speedup from 4x area reduction**

### GPU Utilization
- IRP processing: 5-10% GPU usage
- Tensor operations: <50ms on RTX 4060
- Memory transfers: Negligible with 8GB VRAM

## Optimization Techniques Applied

### 1. Spatial Downsampling
```python
# 16x fewer pixels to process
small = cv2.resize(frame, (160, 120))  # From 640x480
```

### 2. Temporal Subsampling  
```python
# Process every Nth frame
if self.frame_count % 2 == 0:
    self.attention_box = self.detect_motion_fast(frame)
```

### 3. Exponential Smoothing
```python
# Smooth transitions without history buffer
alpha = 0.7
new_box = alpha * old_box + (1-alpha) * detected_box
```

### 4. Cached Computations
```python
# Reuse expensive calculations
if self.frame_count % 30 == 0:
    self.last_telemetry = self.process_irp(frame)
```

## Lessons Learned

### What Worked
1. **Downsampling for motion detection** - Biggest performance gain
2. **Exponential smoothing** - Eliminated visual jitter
3. **Selective IRP processing** - Maintained features without slowdown
4. **Direct OpenCV when possible** - 3x faster than fswebcam

### What Didn't Work
1. **OpenCV V4L2 in WSL2** - Constant timeout errors
2. **Full resolution motion detection** - Too slow for real-time
3. **Per-frame IRP** - Unnecessary and expensive
4. **Complex alpha blending** - Negligible visual improvement

## Recommendations by Use Case

### Real-time Monitoring
Use `fast_camera_monitor.py`:
- Highest FPS (~20)
- Smooth motion tracking
- Minimal latency

### AI/ML Integration
Use `live_camera_monitor.py`:
- Full IRP processing
- Trust scores
- Detailed telemetry

### WSL2 Development
Use `wsl_camera_attention.py`:
- Most reliable capture
- Handles USB issues
- Good for testing

## Future Optimization Opportunities

1. **Hardware encoding** - Use NVENC for recording
2. **CUDA kernels** - Custom motion detection on GPU
3. **Quantized models** - INT8 inference for IRP
4. **Zero-copy frames** - Unified memory on Jetson
5. **Gstreamer pipeline** - Hardware-accelerated capture