# Sensor-Puzzle-Effector VAE Architecture Proposal

## Core Insight

The sensor→puzzle→effector flow requires VAEs that can bridge between raw sensory data and HRM's abstract reasoning space. The key challenge: creating a **common latent space** where HRM's H-loops (strategic) and L-loops (tactical) can both operate.

## Proposed Architecture

### Three Specialized VAEs

```
1. SensorVAE: Raw inputs → Puzzle space
2. PuzzleVAE: Puzzle space ↔ HRM latent space  
3. EffectorVAE: HRM decisions → Motor commands
```

### Latent Space Hierarchy

```
Raw Sensory Space (high-dim, noisy)
        ↓ SensorVAE
Common Puzzle Space (30x30x10 grids)
        ↓ PuzzleVAE
    ┌───┴───┐
H-Latent   L-Latent
(abstract) (concrete)
    └───┬───┘
        ↓ EffectorVAE
Motor Command Space (trajectories, forces)
```

## Common Latent Space Design

### The Bridge: Puzzle Space as Universal Interface

The 30x30 grid with 10 channels isn't just for ARC tasks - it's a **universal compression format**:

```python
class PuzzleSpace:
    """
    Universal 30x30x10 representation
    Channels encode different modalities:
    0-2: RGB/spatial
    3-5: Motion/temporal  
    6-7: Semantic/categorical
    8-9: Confidence/uncertainty
    """
    shape = (30, 30, 10)
```

### H-Latent vs L-Latent Spaces

Yes, H and L loops have different latent representations:

#### H-Latent (Strategic/Abstract)
- **Dimensionality**: Lower (e.g., 128-256 dims)
- **Content**: Rules, relationships, invariants
- **Timescale**: Slow-changing, persistent
- **Example encoding**: "object moves linearly", "pattern repeats"

#### L-Latent (Tactical/Concrete)  
- **Dimensionality**: Higher (e.g., 512-1024 dims)
- **Content**: Specific transformations, exact positions
- **Timescale**: Fast-changing, reactive
- **Example encoding**: "move pixel (5,7) to (6,8)", "rotate 45°"

### Projection Between H and L Spaces

```python
class HRMLatentBridge(nn.Module):
    def __init__(self):
        # H→L: Abstract rules to concrete actions
        self.h_to_l = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024)
        )
        
        # L→H: Concrete observations to abstract patterns
        self.l_to_h = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),  
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        
    def forward(self, h_latent=None, l_latent=None):
        if h_latent is not None:
            # Strategic insight informs tactical execution
            l_guidance = self.h_to_l(h_latent)
            return l_guidance
            
        if l_latent is not None:
            # Tactical patterns inform strategic understanding
            h_pattern = self.l_to_h(l_latent)
            return h_pattern
```

## SensorVAE: Multi-Modal to Puzzle

### Architecture
```python
class SensorVAE(nn.Module):
    def __init__(self):
        # Multiple encoders for different modalities
        self.vision_encoder = CNNEncoder(3, 512)
        self.audio_encoder = WaveEncoder(1, 256)  
        self.imu_encoder = MLPEncoder(6, 128)
        self.text_encoder = TransformerEncoder(768, 256)
        
        # Fusion layer
        self.fusion = nn.MultiheadAttention(1024, 8)
        
        # Project to puzzle space
        self.to_puzzle = nn.Conv2d(1024, 10, 1)
        
        # Decoder back to sensors (for validation)
        self.decoder = PuzzleToSensorDecoder()
```

### Key Features
- **Attention-based fusion** across modalities
- **Learned importance weights** per sensor
- **Temporal integration** via recurrent layers
- **Uncertainty channels** in output

## PuzzleVAE: The Core Translator

### Architecture
```python
class PuzzleVAE(nn.Module):
    def __init__(self):
        # Puzzle → H-latent (find rules)
        self.puzzle_to_h = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15x15
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(64*7*7, 256)  # H-latent size
        )
        
        # Puzzle → L-latent (extract details)
        self.puzzle_to_l = nn.Sequential(
            nn.Conv2d(10, 64, 1),  # Preserve resolution
            nn.ReLU(),
            ResBlock(64, 128),
            ResBlock(128, 256),
            nn.Flatten(),
            nn.Linear(256*30*30, 1024)  # L-latent size
        )
        
        # Decoders (latent → puzzle)
        self.h_to_puzzle = StrategicDecoder()
        self.l_to_puzzle = TacticalDecoder()
```

### Critical: Shared Bottleneck
Both H and L paths must pass through puzzle space, enforcing a common grounding:

```python
def forward(self, puzzle):
    h_latent = self.puzzle_to_h(puzzle)  # Abstract understanding
    l_latent = self.puzzle_to_l(puzzle)  # Concrete details
    
    # Reconstruction from both paths
    h_puzzle = self.h_to_puzzle(h_latent)  # What H thinks it saw
    l_puzzle = self.l_to_puzzle(l_latent)  # What L thinks it saw
    
    # Loss ensures both understand the same reality
    consistency_loss = F.mse_loss(h_puzzle, l_puzzle)
```

## EffectorVAE: Decisions to Actions

### Architecture
```python
class EffectorVAE(nn.Module):
    def __init__(self):
        # Combine H and L decisions
        self.decision_fusion = nn.Sequential(
            nn.Linear(256 + 1024, 768),  # H + L latents
            nn.ReLU(),
            nn.Linear(768, 512)
        )
        
        # Generate motor commands
        self.motor_decoder = nn.LSTM(512, 256, 2)
        self.force_head = nn.Linear(256, 6)  # 6DOF forces
        self.trajectory_head = nn.Linear(256, 3)  # XYZ velocities
```

## Training Strategy

### Phase 1: Individual VAE Training
1. **SensorVAE**: Train on sensor→puzzle reconstruction
2. **EffectorVAE**: Train on action→outcome prediction
3. **PuzzleVAE**: Train on puzzle autoencoding

### Phase 2: Joint Alignment
```python
def alignment_loss(sensor_vae, puzzle_vae, effector_vae, batch):
    # Full cycle consistency
    puzzle = sensor_vae.encode(batch.sensors)
    h_latent, l_latent = puzzle_vae.encode(puzzle)
    action = effector_vae.decode(h_latent, l_latent)
    
    # Predict next state
    next_puzzle_pred = dynamics_model(puzzle, action)
    next_puzzle_real = sensor_vae.encode(batch.next_sensors)
    
    # Loss ensures closed-loop consistency
    return F.mse_loss(next_puzzle_pred, next_puzzle_real)
```

### Phase 3: HRM Integration
- Freeze early VAE layers
- Fine-tune latent projections with HRM in the loop
- Use HRM's performance as training signal

## Key Innovations

### 1. Puzzle Space as Markov Blanket
The 30x30x10 puzzle representation acts as a Markov blanket between sensors and reasoning, providing statistical independence while preserving information.

### 2. Dual Latent Paths
H and L latents capture different aspects of the same reality:
- H finds the "why" (rules, patterns)
- L finds the "how" (specific transforms)

### 3. Trust-Weighted Encoding
SensorVAE weights inputs by HRM's critic trust scores:
```python
weighted_encoding = sum(
    trust_weight[i] * encoder[i](sensor[i])
    for i in range(num_sensors)
)
```

### 4. Compositional Puzzle Building
Complex scenes decompose into puzzle primitives:
- Objects → colored regions
- Relationships → spatial patterns
- Dynamics → temporal channels

## Implementation Priorities

### Immediate (For HRM Integration)
1. Basic PuzzleVAE with H/L split
2. Simple grid-based sensor encoding
3. Direct action decoding

### Near-term (For SAGE)
1. Multi-modal SensorVAE
2. Trust-weighted fusion
3. Dynamics model for prediction

### Long-term (Production)
1. Learned puzzle abstractions beyond 30x30
2. Hierarchical latent spaces (H-high, H-low, L-high, L-low)
3. Meta-learning for new sensor types

## Connection to Existing Work

### TinyVAE Distillation
Use the distilled TinyVAE as the base for SensorVAE - already optimized for edge deployment.

### KV-Cache Persistence  
H-latent states could be stored in KV-cache for temporal continuity - strategic understanding persists across time.

### GPU Mailboxes
Different VAEs run on different GPU tiles, communicating via mailboxes - natural parallelism.

## Validation Metrics

### Puzzle Fidelity
- Can we reconstruct original sensors from puzzle?
- Do H and L paths agree on puzzle content?

### Latent Separation
- Does H-latent capture only invariants?
- Does L-latent capture only specifics?

### End-to-End Performance
- Can sensor→puzzle→HRM→action→outcome work?
- Does trust-weighting improve robustness?

## Next Steps

1. Implement basic PuzzleVAE with H/L split
2. Create synthetic puzzle dataset for initial training
3. Test H↔L projection consistency
4. Integrate with HRM once training completes
5. Deploy on Jetson for edge validation

This architecture provides the missing link between raw reality and abstract reasoning, with puzzle space as the universal interface and dual latent paths capturing both strategic and tactical understanding.