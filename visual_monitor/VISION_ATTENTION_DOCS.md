# Vision Attention System Documentation

## System Overview
Hierarchical vision attention system implementing phase 2 of the attention architecture:
- **Phase 1**: Awareness (motion detection across entire frame)
- **Phase 2**: Attention (focus on regions of interest)
- **Phase 3**: Focus (selective VAE processing of attention regions)

## Architecture Design

### Single-Sensor Plugin Architecture
Each camera/sensor gets its own plugin instance with:
- Independent motion tracking
- Configurable attention strategy
- Optional VAE processing
- IRP-compliant interface for integration

### Strategy Pattern Implementation

#### FixedTileStrategy
- Divides frame into 8x8 grid
- Focus region: 4x3 tiles
- Gravity-based movement toward motion
- Centers at (2,2) when idle
- Good for: surveillance, monitoring

#### DynamicBoxStrategy  
- Adaptive bounding box
- Tracks peak motion location
- Size adjusts based on motion spread (64-320px)
- Temporal smoothing for stability
- Good for: object tracking, dynamic scenes

## Key Improvements Made

### 1. Centering Fix
- Fixed integer division for proper center calculation
- Correct idle position at grid coordinates (2,2)
- Consistent across both camera orientations

### 2. Motion Sensitivity Balance
- Adjusted thresholds for each strategy
- Fixed tiles: 0.1 threshold (more sensitive)
- Dynamic box: 0.3 threshold (noise rejection)
- Motion amplification by 2x for better response

### 3. Adaptive Box Sizing
- Calculates motion spread to determine box size
- Ranges from 64px (small object) to 320px (large motion)
- 1.5x multiplier on spread for coverage

### 4. Memory Management
- Lazy VAE loading on first use
- Automatic unloading after 5 seconds
- Selective processing above threshold
- ~100MB GPU memory per plugin

## Integration with IRP Framework

The VisionAttentionPlugin implements the IRP interface:
```python
class VisionAttentionPlugin(IRPPlugin):
    def init_state(self, x0, task_ctx): ...
    def energy(self, state): ...
    def step(self, state, noise_schedule): ...
    def refine(self, input_data, max_iterations): ...
```

Key aspects:
- Single-pass refinement (max_iterations=1)
- Energy based on motion confidence
- Telemetry includes strategy, confidence, VAE status

## Testing Results

### Fixed Tiles Performance
- Smooth gravity-based tracking
- Good centering when idle
- Responsive to bulk motion
- Stable focus region size

### Dynamic Box Performance
- Accurate motion tracking
- Adaptive sizing working
- Good noise rejection
- Smooth transitions

### Dual Camera System
- Independent strategy per sensor
- Agreement detection between sensors
- Real-time performance (30 FPS)
- Low GPU memory usage

## Configuration Tuning

### Motion Detection
```python
# Current settings that work well:
diff = cv2.absdiff(prev_gray, gray)
motion = diff.astype(np.float32) / 255.0
motion = np.minimum(motion * 2.0, 1.0)  # Amplify
```

### Strategy Thresholds
```python
# Fixed tiles
if max_motion > 0.1:  # More sensitive
    state = PROCESSING if max_motion > 0.3 else TRACKING

# Dynamic box  
if max_motion > 0.3:  # Higher for noise rejection
    # Track motion peak
```

## Future Work
1. **Learned Strategy Switching**: Train model to select optimal strategy
2. **Temporal Consistency**: Multi-frame tracking and prediction
3. **VAE Training**: Train on relevant visual data for better encoding
4. **Hierarchical Integration**: Connect to higher reasoning layers
5. **Performance Optimization**: Further GPU kernel optimization

## Files Modified
- `sage/irp/plugins/vision_attention_plugin.py` - Main plugin implementation
- `visual_monitor/test_multi_sensor_vision.py` - Dual camera test
- `visual_monitor/test_dynamic_only.py` - Dynamic strategy test
- `visual_monitor/hierarchical_vae_attention.py` - VAE integration demo

## Usage Example
```bash
# Test single dynamic strategy
python visual_monitor/test_dynamic_only.py

# Test dual cameras with different strategies  
python visual_monitor/test_multi_sensor_vision.py

# Test VAE integration
python visual_monitor/hierarchical_vae_attention.py
```