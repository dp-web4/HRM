# SAGE Visual Attention Monitor

Real-time visualization of the IRP attention mechanism for camera feeds and images.

## 🎉 Complete Visual Attention System Working! (August 24, 2025)

### Achievements Today:
- **Dual IMX219 cameras operational** at ~29-30 FPS
- **Motion-based attention overlay working!**
  - Shows diffuse attention when scene is static
  - Focuses on moving objects when motion detected
  - Real-time heatmap overlay (red=high attention, blue=low)
- Fixed device tree overlay issue after SSD boot migration
- Camera sensors integrated as IRP plugins with GPU mailbox
- Publish/subscribe mailbox for inter-plugin communication
- **TinyVAE integration** for compact latent encoding (August 25, 2025)
  - 16-dimensional latent extraction from 64x64 crops
  - Motion-guided crop selection from attention peaks
  - 1-2ms encoding latency on Jetson

### How Attention Works:
1. **Static scenes**: Attention spreads across entire frame
2. **Motion detected**: Attention focuses on moving regions
3. **Visualization**: Heatmap overlay blended at 30% opacity
4. **Performance**: Real-time at 30 FPS with minimal latency

## Architecture

The Visual Monitor is implemented as an **IRP Effector Plugin** that integrates seamlessly with the SAGE orchestration system. This allows it to:
- Work alongside other IRP plugins (vision, language, etc.)
- Participate in ATP budget allocation
- Be managed by the orchestrator
- Record telemetry and performance metrics

## Features

- **Real-time camera monitoring** with attention overlay
- **IRP refinement visualization** showing energy convergence
- **Performance metrics** including FPS, iterations, and compute savings
- **Memory system integration** with SNARC consolidation tracking
- **Lightweight mode** optimized for Jetson Orin Nano

## Components

### IRP Plugin Implementation (`sage/irp/plugins/visual_monitor_impl.py`)
The core Visual Monitor IRP plugin that:
- Implements the standard IRP interface
- Processes frames and extracts attention
- Manages display window and visualization
- Can be orchestrated with other plugins

```python
from sage.irp.plugins.visual_monitor_impl import create_visual_monitor_irp

# Create as IRP plugin
monitor = create_visual_monitor_irp(
    show_window=True,
    display_width=1280,
    display_height=720
)

# Use with orchestrator
orchestrator.register_plugin("monitor", monitor)
```

## Standalone Components

### 1. `camera_attention_monitor.py` (Recommended)
Lightweight real-time monitor optimized for Jetson:
- Minimal dependencies (OpenCV only)
- Efficient frame processing (configurable skip rate)
- CSI and USB camera support
- Real-time attention overlay
- Performance graphs

```bash
python3 camera_attention_monitor.py
```

**Controls:**
- `Q`: Quit
- `A`: Toggle attention overlay
- `S`: Toggle statistics panel
- `P`: Toggle frame processing rate

### 2. `test_attention_static.py`
Test attention visualization with static images:
- Detailed attention analysis
- Energy convergence graphs
- Before/after comparison
- Saves visualization plots

```bash
# Test with synthetic image
python3 test_attention_static.py

# Test with specific image
python3 test_attention_static.py --image path/to/image.jpg --output result.png
```

### 3. `visual_attention_monitor.py`
Full-featured monitor with advanced visualization:
- Multi-threaded capture and processing
- Attention region detection
- Memory consolidation tracking
- Rich telemetry display

```bash
python3 visual_attention_monitor.py
```

### 4. **NEW: WSL2 Camera Support** (August 25, 2025)

#### `fast_camera_monitor.py` (Recommended for WSL2)
High-performance live monitor optimized for USB passthrough:
- Direct OpenCV capture (~20 FPS achieved)
- Smooth attention box with α=0.7 exponential smoothing
- Downsampled motion detection (160x120) for speed
- GPU-accelerated with CUDA support
- Adaptive attention sizing based on motion variance

```bash
source sage_env/bin/activate
python visual_monitor/fast_camera_monitor.py
```

**Session Results:**
- 1822 frames processed at ~20 FPS
- RTX 4060 GPU acceleration active
- Smooth real-time attention tracking

#### `live_camera_monitor.py`
Full-featured monitor with IRP integration:
- fswebcam capture for compatibility
- IRP processing every 30 frames
- Trust score computation
- Frame saving capability

#### `wsl_camera_attention.py`
Batch processor for WSL2:
- Works around USB passthrough limitations
- Saves attention frames to disk
- Creates summary montages

### 5. `test_tinyvae_pipeline.py`
TinyVAE integration for compact latent encoding:
- Extracts 64x64 crops from motion attention peaks
- Encodes to 16-dimensional latent space
- Shows reconstruction alongside original
- Real-time latent vector display

```bash
# Test with camera feed
python3 test_tinyvae_pipeline.py

# Controls:
# Q: Quit
# R: Toggle reconstruction display
# L: Print latent vector
# S: Save current crop
```

## How Attention is Computed

The attention mechanism visualizes which parts of the image are being refined:

1. **VAE Encoding**: Image is encoded to latent space (7x7x256)
2. **Refinement**: IRP iteratively refines the latent representation
3. **Attention Extraction**: Difference between input and refined indicates focus areas
4. **Visualization**: Attention map is overlaid as a heatmap

## Performance on Jetson

Typical performance metrics:
- **Camera FPS**: 20-30 fps display
- **Processing**: Every 3rd frame (configurable)
- **IRP Latency**: 3-5ms per refinement
- **Memory Usage**: <200MB total

## WSL2 Camera Setup (NEW)

### Prerequisites for Windows/WSL2

1. **Install usbipd-win on Windows** (Admin PowerShell):
```powershell
winget install --interactive --exact dorssel.usbipd-win
```

2. **Install WSL2 dependencies**:
```bash
sudo apt-get update
sudo apt-get install -y v4l-utils fswebcam linux-tools-virtual hwdata
sudo modprobe usbip-core vhci-hcd uvcvideo
```

3. **Attach camera to WSL2** (Admin PowerShell on Windows):
```powershell
# List USB devices
usbipd list

# Bind camera (use your BUSID, e.g., 2-6)
usbipd bind --busid 2-6

# Attach to WSL2
usbipd attach --wsl --busid 2-6
```

4. **Fix permissions in WSL2**:
```bash
sudo chmod 666 /dev/video0
```

5. **Verify camera**:
```bash
ls -la /dev/video*
v4l2-ctl --list-devices
```

## Camera Setup

### CSI Camera (Jetson)
The monitor automatically detects CSI cameras using GStreamer:
```python
nvarguscamerasrc sensor-id=0 ! 
video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 !
nvvidconv ! video/x-raw, format=BGRx !
videoconvert ! video/x-raw, format=BGR ! appsink
```

### USB Camera
Falls back to USB cameras if CSI not available:
- 640x480 @ 30fps default
- Adjustable via OpenCV properties

## Attention Interpretation

The attention heatmap shows:
- **Red/Yellow**: High attention areas (being refined)
- **Blue/Green**: Low attention areas (stable)
- **Intensity**: Confidence in refinement need

High attention typically indicates:
- Edges and boundaries
- Complex textures
- Motion or changes
- Anomalies or interesting features

## Integration with SAGE

The monitor integrates with:
- **IRP Plugins**: Vision refinement pipeline
- **Memory Bridge**: SNARC memory consolidation
- **Orchestrator**: ATP budget tracking
- **Telemetry**: Real-time performance metrics

## Pipeline Integration

### Complete Camera Pipeline (`camera_pipeline.py`)
Full integration with orchestrator and memory:

```python
from visual_monitor.camera_pipeline import CameraIRPPipeline

# Create pipeline with all components
pipeline = CameraIRPPipeline(camera_id=0)

# Run with async orchestration
import asyncio
asyncio.run(pipeline.run_async())

# Or run synchronously
pipeline.run_sync()
```

This pipeline:
- Captures from camera (CSI or USB)
- Processes through Vision IRP for refinement
- Visualizes with Monitor IRP
- Records to memory bridge
- Manages ATP budget allocation

## Examples

### Basic Camera Monitor
```python
from camera_attention_monitor import LightweightAttentionMonitor

monitor = LightweightAttentionMonitor(
    camera_id=0,
    display_width=1024,
    display_height=600
)
monitor.run()
```

### Static Image Analysis
```python
from test_attention_static import visualize_attention

attention_map, telemetry = visualize_attention(
    "path/to/image.jpg",
    "output_visualization.png"
)
print(f"Iterations: {telemetry['iterations']}")
print(f"Compute saved: {telemetry['compute_saved']*100:.1f}%")
```

## Troubleshooting

### No Camera Detected
- Check camera connection
- Verify camera permissions: `ls -l /dev/video*`
- For CSI cameras: Ensure nvarguscamerasrc is available
- Try different camera IDs (0, 1, 2)

### Low FPS
- Increase `process_every_n` to skip more frames
- Reduce display resolution
- Disable attention overlay ('A' key)

### High CPU/GPU Usage
- The IRP refinement is GPU-accelerated
- Processing every frame can be intensive
- Use 'P' key to toggle processing rate

## Output Files

Test outputs are saved to `visual_monitor/test_outputs/`:
- `*_attention.png`: Raw attention map
- `*.png`: Full visualization with graphs
- `synthetic_test.jpg`: Generated test image

---

*Part of the HRM/SAGE project - Iterative Refinement with Visual Attention*