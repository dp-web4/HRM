# Visual Attention Pipeline Architecture

## Overview
This document details the architecture and implementation of the visual attention system on the Jetson Orin Nano, including both the lightweight motion-based approach and the heavier IRP-based neural network approach.

## Current Implementation: Lightweight Motion-Based Pipeline

### Pipeline Flow (test_real_cameras.py)

#### 1. Camera Capture (Hardware → Memory)
```
CSI Camera 0 (Left)  ──┐
                       ├──> NVMM (GPU Memory) 
CSI Camera 1 (Right) ──┘    via GStreamer
```
- **Sensors**: Dual IMX219 at 1920×1080 @ 30 FPS
- **GStreamer Pipeline**: 
  ```
  nvarguscamerasrc → NVMM → nvvidconv → 960×540 BGR
  ```
- **Optimization**: Zero-copy from sensor to GPU memory using NVIDIA Multimedia (NVMM)

#### 2. Frame Processing (Per Frame)
```python
For each frame pair:
    1. Read frames from cameras (cv2.VideoCapture)
    2. Convert to grayscale (cv2.cvtColor)
    3. Motion detection:
       - Compare with previous frame (cv2.absdiff)
       - Blur difference map (cv2.GaussianBlur, 21×21 kernel)
       - Normalize to 0-1 range
    4. Apply attention overlay if enabled:
       - Convert attention to heatmap (cv2.applyColorMap, COLORMAP_JET)
       - Blend with original frame (70% original, 30% heatmap)
```

#### 3. Display Pipeline
```
Left Frame  + Attention ──┐
                          ├──> Side-by-side combined (1920×540)
Right Frame + Attention ──┘    → OpenCV imshow() → Display
```

#### 4. Memory Flow
```
NVMM (GPU) → Host (CPU) → Processing → Display Buffer → Screen
```
- Frames decoded from NVMM to CPU memory
- All processing done on CPU (motion detection, blending)
- Final composite sent to display buffer

#### 5. Timing & Performance Breakdown
- **30 FPS target**: 33.3ms per frame budget
- **Actual measurements**:
  - Camera capture: ~5ms (hardware accelerated)
  - Grayscale conversion: ~1ms
  - Motion detection: ~3ms per camera
  - Gaussian blur: ~2ms per camera
  - Heatmap generation: ~1ms
  - Blending: ~1ms
  - Display: ~2ms
  - **Total**: ~15-20ms per frame (50% headroom)

#### 6. Key Optimizations
- NVMM zero-copy from sensors
- Hardware ISP for initial processing
- In-place operations where possible
- Frame dropping (`drop=true, max-buffers=1`) to prevent lag
- Async capture threads in background

#### 7. Attention Behavior
```
Static Scene:  Low/no motion → Diffuse attention (whole frame)
Dynamic Scene: Motion detected → Focused attention (moving regions)
Color Map:     Blue (low) → Green → Yellow → Red (high attention)
```

### Memory Usage: Lightweight Pipeline
```
- Camera buffers: 2 × 960×540×3 = ~3MB
- Grayscale: 2 × 960×540 = ~1MB  
- Attention maps: 2 × 960×540 float32 = ~4MB
- Total: ~8-10MB RAM (CPU only)
```

---

## IRP Neural Network Pipeline (Memory Intensive)

### Why IRP Version is Memory Intensive

#### 1. VAE (Variational Autoencoder) Model
```python
# From sage/irp/plugins/vision_impl.py
self.vae = create_vae_for_jetson(vae_variant).to(self.device)
```
- Even "minimal" variant loads a neural network with:
  - **Encoder**: 3→16→32→64→128 channels (progressive downsampling)
  - **Decoder**: Mirror architecture back to image space
  - **Model weights**: ~10-50MB in GPU memory
  - **Activation maps**: Each layer stores intermediate results

#### 2. Tensor Operations & Batching
```python
# Each frame processing:
frame_tensor = torch.from_numpy(frame).to(self.device)  # Copy to GPU
latent = self.vae.encode(frame_tensor)  # Forward pass through encoder
refined = self.refiner(latent)          # Refinement network
output = self.vae.decode(refined)       # Decode back to image space
```
- **Multiple copies**: Original, encoded, refined, decoded
- **Float32 precision**: 4 bytes per pixel per channel
- For 224×224×3 image: ~600KB per tensor × 4-5 tensors = ~3MB per frame

#### 3. Dual Processing (Left + Right Cameras)
```python
self.left_vision = create_vision_irp(self.device)
self.right_vision = create_vision_irp(self.device)
```
- **Two complete model instances** in memory
- Double all tensors and activations
- No weight sharing between left/right processing

#### 4. GPU Memory Fragmentation
- PyTorch allocates in chunks, not exact sizes
- Memory gets fragmented over time
- Cached allocations aren't immediately freed
- GPU memory manager overhead

#### 5. Orchestrator & Plugin Architecture
```python
# Each plugin maintains state
self.orchestrator = HRMOrchestrator()
self.memory_bridge = IRPMemoryBridge()
self.mailbox = PubSubMailbox()
```
- Multiple abstraction layers
- Message queues and buffers
- Async execution contexts
- Memory bridge consolidation buffers

#### 6. Gradient Computation (if enabled)
```python
# IRP refinement with gradients
for iteration in range(max_iterations):
    refined = refiner(x)
    energy = compute_energy(refined)
    # Gradients stored for backpropagation
```
- Gradient tensors for backpropagation
- Optimizer state (momentum, etc.)
- Computation graph retained in memory

### Memory Usage: IRP Pipeline
```
- VAE models: 2 × ~50MB = 100MB GPU
- Input tensors: 2 × 224×224×3×4 = ~1.2MB
- Latent spaces: 2 × 7×7×256×4 = ~200KB
- Intermediate activations: ~50-100MB
- PyTorch overhead: ~200-500MB
- Total: 500MB-1GB GPU memory
```

---

## Jetson Orin Nano Constraints

### Hardware Specifications
- **Memory**: 8GB unified (shared CPU/GPU)
- **GPU**: Ampere architecture, 1024 CUDA cores
- **Memory Bandwidth**: 68 GB/s

### Memory Budget
```
Total:              8GB
OS & Background:   -2GB
Camera & GStreamer: -0.5GB
Available:          ~5.5GB
IRP Pipeline Peak:  1-2GB (can cause pressure)
Lightweight:        10MB (negligible)
```

### Why Memory Matters
- Unified memory means CPU and GPU compete
- Memory pressure causes swapping
- Swapping kills real-time performance
- Need headroom for burst allocations

---

## Comparison Summary

| Aspect | Lightweight (Motion) | IRP (Neural Network) |
|--------|---------------------|---------------------|
| **Memory Usage** | 8-10MB | 500MB-1GB |
| **Processing** | CPU only | GPU required |
| **Latency** | 15-20ms | 50-100ms |
| **Model Weights** | None | ~100MB |
| **Attention Quality** | Motion-based | Learned features |
| **Real-time @ 30FPS** | ✅ Guaranteed | ⚠️ Memory dependent |
| **Power Usage** | ~5W | ~15W |

---

## Implementation Files

### Lightweight Pipeline
- `visual_monitor/test_real_cameras.py` - Simple dual camera with motion attention
- `visual_monitor/optimized_camera_pipeline.py` - Orchestrated lightweight version

### IRP Pipeline  
- `sage/irp/plugins/vision_impl.py` - Vision IRP with VAE
- `visual_monitor/integrated_camera_pipeline.py` - Full IRP integration
- `models/vision/lightweight_vae.py` - VAE implementation

### Common Components
- `sage/irp/plugins/camera_sensor_impl.py` - Camera sensor plugins
- `sage/irp/plugins/visual_monitor_effector.py` - Display effector
- `sage/mailbox/pubsub_mailbox.py` - Inter-plugin communication

---

## Hierarchical Attention Architecture (Implemented)

### Core Principle: Awareness → Attention → Focus
**Awareness isn't attention.** The system should maintain peripheral awareness everywhere while selectively focusing computational resources only where needed.

### Biological Inspiration
```
Peripheral Vision (Fast, Low-res) → Saccade Decision → Foveal Vision (Slow, High-res)
         ↓                                ↓                        ↓
   Motion Detection              Where to Look?            Deep Processing
```

### Three-Level Hierarchy

#### Level 1: Peripheral Awareness (Always On)
- **Tile-based processing**: Divide frame into 8×8 grid (64 tiles)
- **Per-tile metrics**: Motion score, edge density, contrast
- **Classification**: Each tile marked as 'peripheral' or 'focus'
- **Performance**: ~1ms per tile, 5ms total

```python
# Tile size for 960×540 frame
tile_size = 120×67 pixels
processing = motion_detection + edge_detection
output = binary_classification (focus/peripheral)
```

#### Level 2: Region Formation (Gestalt Grouping)
- **Adjacent tile grouping**: Merge neighboring focus tiles
- **Region prioritization**: Based on size, motion, history
- **Output**: 1-3 regions of interest instead of 64 tiles
- **Performance**: ~1ms

```
Full Frame Tiling Example:
┌─┬─┬─┬─┬─┬─┬─┬─┐
├─┼─┼─┼─┼─┼─┼─┼─┤  P = Peripheral (low activity)
├─┼─┼─╔═╦═╗─┼─┼─┤  F = Focus (high activity)
├─┼─┼─╠═╬═╣─┼─┼─┤  ══ = Grouped focus region
├─┼─┼─╚═╩═╝─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┤  Only ══ region sent to VAE
└─┴─┴─┴─┴─┴─┴─┴─┘
```

#### Level 3: Selective VAE Processing (On Demand)
- **Conditional loading**: VAE loaded only when focus regions exist
- **Regional processing**: Only focus regions processed through VAE
- **Adaptive models**: Micro (10MB), Mini (25MB), Full (50MB)
- **Performance**: 10-100ms depending on region size

### Adaptive Processing Pipeline

```python
class HierarchicalAttention:
    def process_frame(self, frame):
        # Level 1: Fast tiling (5ms)
        tiles = tile_frame(frame)
        motion_map = compute_motion_tiles(tiles)
        
        # Level 2: Region grouping (1ms)
        focus_regions = group_focus_tiles(motion_map)
        
        # Level 3: Selective VAE (variable)
        if focus_regions:
            for region in focus_regions:
                if budget_remaining():
                    attention[region] = vae_process(region)
                else:
                    attention[region] = fast_fallback(region)
        else:
            attention = diffuse_attention(frame)
            
        return attention
```

### Memory Management Strategy

| VAE Variant | Model Size | Input Size | Use Case |
|------------|------------|------------|----------|
| Micro | 10MB | <100×100 | Small regions |
| Mini | 25MB | <224×224 | Medium regions |
| Full | 50MB | Full frame | High alert |

**LRU Cache**: Models loaded on-demand and evicted when unused

### Biologically-Inspired Behaviors

#### 1. Saccadic Suppression
- Skip processing during rapid attention shifts
- Reduces motion blur artifacts

#### 2. Inhibition of Return
- Temporarily suppress recently-attended regions
- Prevents attention loops

#### 3. Predictive Loading
- Pre-load VAE when motion trending toward threshold
- Reduces latency when attention needed

### Performance Characteristics

| Scenario | Time | Memory | Description |
|----------|------|--------|-------------|
| **All Peripheral** | 5ms | 10MB | No motion, diffuse awareness |
| **Single Focus** | 15ms | 60MB | One region needs attention |
| **Multiple Foci** | 30-50ms | 100MB | Several regions active |
| **Full Alert** | 100ms | 200MB | Entire frame important |

### Attention Budget System

```python
attention_budget = 33ms  # For 30 FPS
used = 0

# Peripheral always runs (5ms)
used += peripheral_scan()

# VAE only if budget allows
if motion_detected and (budget - used) > 10ms:
    used += vae_process()
    
# Graceful degradation
if used > budget:
    reduce_quality_next_frame()
```

### Real-World Behavior Example

```
Frames 1-10: Static scene
├─ Peripheral scanning only
├─ 5ms per frame
└─ VAE not loaded

Frame 11: Motion detected (top-right)
├─ 4 tiles marked as focus
├─ Grouped into 1 region
├─ Micro VAE loaded (10MB)
└─ Total: 15ms

Frames 12-20: Tracking motion
├─ Region tracking active
├─ VAE cached and reused
└─ 12ms per frame

Frame 21: Motion stops
├─ Return to peripheral
├─ VAE evicted after 500ms
└─ Back to 5ms per frame
```

### Implementation Status

#### ✅ Phase 1: Gravity-Based Focus System (Completed August 25, 2025)

**Files Implemented:**
- `hierarchical_attention.py` - Single camera tile-based system
- `hierarchical_attention_dual.py` - Dual camera with state machine
- `hierarchical_attention_gravity.py` - **Current best**: Fixed-size focus with gravity

**Key Features Working:**
1. **Fixed Focus Region**: 3×4 tiles (configurable via FOCUS_WIDTH/HEIGHT)
2. **8×8 Tile Grid**: Peripheral awareness across entire frame
3. **Gravity-Based Movement**: Focus smoothly gravitates toward motion centers
4. **Motion Detection**: 
   - 75th percentile scoring for balanced sensitivity
   - Sigmoid amplification curve (avoids harsh thresholds)
   - Per-tile motion scores with 5×5 Gaussian blur
5. **Dual Camera Support**: Independent focus tracking for each camera
6. **Smooth Transitions**: Fractional positioning for pixel-level movement
7. **Auto-Return**: Focus returns to center when no motion detected

**Performance Metrics:**
- **Processing**: ~5ms per frame for motion detection
- **FPS**: Stable 29-30 FPS with dual cameras
- **Memory**: <20MB RAM (CPU-based processing)
- **Latency**: <20ms total pipeline latency

**Tunable Parameters:**
```python
motion_threshold = 0.3   # Threshold for sigmoid-scaled values
gravity_strength = 0.6   # Moderate gravity for smooth movement
warmup_frames = 30       # Ignore initial camera adjustment
```

**Known Issues & Future Work:**
- Gravity centers sometimes cluster around frame center
- Need better multi-object tracking (currently follows strongest motion)
- Could benefit from predictive movement
- Ready for Phase 2: VAE integration for focus regions

#### ⏳ Phase 2: Selective VAE Processing (Next)
- Load VAE only for focus regions
- Lazy loading based on motion intensity
- Memory-aware model selection

#### 📋 Phase 3-5: Future Work
3. **Region Grouping**: Connected component analysis for multi-focus
4. **Attention Budget**: Time allocation system
5. **Predictive Loading**: Pre-load models based on motion trends

## Future Optimizations

### Short Term
1. **Implement tiling system**: Basic grid-based awareness
2. **Region grouping**: Connected component analysis
3. **VAE lazy loading**: Load only when needed
4. **Memory monitoring**: Track and limit usage

### Long Term
1. **TensorRT optimization**: Compile VAE models
2. **Custom CUDA kernels**: Optimized tile processing
3. **Predictive attention**: Learn patterns
4. **Multi-scale tiling**: Hierarchical grids

---

*Last Updated: August 25, 2025*
*Platform: Jetson Orin Nano (Sprout)*
*Status: Phase 1 Complete - Gravity-based hierarchical attention working*