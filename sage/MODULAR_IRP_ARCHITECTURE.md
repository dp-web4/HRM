# Modular IRP Architecture Specification

**Date**: October 6, 2025  
**Version**: 1.0 - Fundamental Description  
**Authors**: Dennis Palatov, Claude

## Core Principles

### Trust-Attention-Surprise Loop
```
Trust → Attracts Attention → Generates Predictions → 
Surprise (prediction error) → Modifies Trust → (loop)
```

### Modularity Requirements
- Each IRP module operates in its optimal latent space
- Translation shims provide interoperability
- High-bandwidth GPU mailboxes for raw data
- High-abstraction token space for reasoning
- Compact enough for 8GB GPU with headroom

## System Architecture

### 1. Sensor IRPs (Environmental & Temporal)

```python
class SensorIRP(Protocol):
    """Base interface for all sensor modules"""
    
    @property
    def latent_space(self) -> LatentSpaceSpec:
        """Define this sensor's native representation"""
        # e.g., pixels, frequencies, coordinates
        
    def raw_to_latent(self, raw_data: Buffer) -> Latent:
        """Convert raw sensor data to module's latent space"""
        
    def latent_to_tokens(self, latent: Latent) -> TokenSequence:
        """Convert to SAGE token space for reasoning"""
        
    def compute_surprise(self, expected: Latent, observed: Latent) -> float:
        """Measure prediction error in native space"""
```

#### Vision IRP
- **Latent Space**: Spatial feature maps (H×W×C)
- **Raw Input**: Camera frames (RGB, depth, IR)
- **Shim**: TinyVAE for compression (224×224×3 → 16×16×64)
- **Tokens**: Patch tokens via vector quantization
- **GPU/CPU**: GPU for conv operations

#### Audio IRP  
- **Latent Space**: Spectral features (mel-spectrogram)
- **Raw Input**: Audio waveforms (16kHz mono)
- **Shim**: Learned filterbank → VQ codebook
- **Tokens**: Phoneme-like discrete codes
- **GPU/CPU**: GPU for FFT, CPU for streaming

#### Temporal IRP (Clock)
- **Latent Space**: Phase vectors (sin/cos embeddings)
- **Raw Input**: System timestamps
- **Shim**: Direct encoding (no compression needed)
- **Tokens**: Discrete time bins (second/minute/hour/day)
- **GPU/CPU**: CPU only

#### Proprioception IRP
- **Latent Space**: Joint configuration space
- **Raw Input**: Joint angles, IMU data, force sensors
- **Shim**: Normalize → small MLP
- **Tokens**: Discretized poses from learned vocabulary
- **GPU/CPU**: CPU (small data)

### 2. Effector IRPs

```python
class EffectorIRP(Protocol):
    """Base interface for all effector modules"""
    
    def tokens_to_latent(self, tokens: TokenSequence) -> Latent:
        """Convert SAGE commands to module's action space"""
        
    def latent_to_raw(self, latent: Latent) -> Buffer:
        """Convert to raw control signals"""
        
    def compute_feasibility(self, action: Latent) -> float:
        """Check if action is physically possible"""
```

#### Motor Control IRP
- **Latent Space**: Trajectory in configuration space
- **Output**: Joint velocities/torques
- **Shim**: Inverse kinematics solver
- **Tokens**: Motion primitives vocabulary
- **GPU/CPU**: GPU for trajectory optimization

#### Speech IRP (TTS)
- **Latent Space**: Phoneme sequences
- **Output**: Audio waveforms
- **Shim**: Text → phoneme → neural vocoder
- **Tokens**: Text/phoneme tokens
- **GPU/CPU**: GPU for vocoder

### 3. Translation Shims

```python
class TranslationShim(Protocol):
    """Interface between module latent spaces and system"""
    
    def encode(self, data: RawData) -> Latent:
        """Raw → Module's latent space"""
        
    def decode(self, latent: Latent) -> RawData:
        """Module's latent → Raw"""
        
    def quantize(self, latent: Latent) -> Tokens:
        """Module's latent → SAGE tokens"""
        
    def dequantize(self, tokens: Tokens) -> Latent:
        """SAGE tokens → Module's latent"""
```

#### Shim Types

**TinyVAE** (Vision, complex sensors)
- 64-256 latent dims
- GroupNorm for single-sample inference
- FP16 for memory efficiency

**VQ Codebook** (Audio, discrete modalities)
- 512-1024 codes
- Learned via k-means or straight-through estimator

**Linear Projection** (Simple sensors)
- Direct mapping for low-dimensional inputs
- Minimal compute overhead

**Identity** (Already discrete)
- Pass-through for pre-tokenized inputs

### 4. Communication Infrastructure

#### GPU Mailboxes (High Bandwidth)
```python
class GPUMailbox:
    """Zero-copy data transfer between GPU modules"""
    
    def push(self, tensor: Tensor, metadata: Dict):
        """Non-blocking push with metadata"""
        
    def pop(self) -> Tuple[Tensor, Dict]:
        """Non-blocking pop with metadata"""
        
    def peek(self) -> Optional[Metadata]:
        """Check without consuming"""
```

**Metadata Structure**:
```python
{
    'source': 'vision_irp',
    'timestamp': 1234567890.123,
    'trust': 0.85,
    'surprise': 0.02,
    'shape': (16, 16, 64),
    'dtype': 'float16',
    'space': 'latent',  # 'raw', 'latent', 'tokens'
    'semantics': 'spatial_features'
}
```

#### Token Bus (High Abstraction)
```python
class TokenBus:
    """Discrete token communication for SAGE reasoning"""
    
    def broadcast(self, tokens: TokenSequence, context: Context):
        """Send tokens to all consumers"""
        
    def subscribe(self, filter: Callable) -> TokenStream:
        """Receive filtered token stream"""
```

### 5. SAGE Orchestration Layer

```python
class SAGEOrchestrator:
    """Coordinates IRP modules via trust-weighted attention"""
    
    def __init__(self):
        self.trust_weights = {}  # Per-module trust scores
        self.attention_budget = 100.0  # ATP-style budget
        
    def allocate_attention(self) -> Dict[str, float]:
        """Trust-weighted attention allocation"""
        total_trust = sum(self.trust_weights.values())
        return {
            module: (trust / total_trust) * self.attention_budget
            for module, trust in self.trust_weights.items()
        }
    
    def update_trust(self, module: str, surprise: float):
        """Surprise modifies trust"""
        # Low surprise → increase trust
        # High surprise → decrease trust
        self.trust_weights[module] *= (1.0 - surprise)
        self.trust_weights[module] = max(0.1, min(1.0, self.trust_weights[module]))
```

### 6. Memory Architecture

#### Working Memory (GPU)
- Recent sensor latents (last 10 seconds)
- Active action plans
- Attention state
- **Size**: ~500MB

#### Episode Buffer (GPU/CPU)
- Last 100 episodes
- Compressed via TinyVAE
- **Size**: ~1GB  

#### Long-term Memory (CPU/Disk)
- Consolidated patterns
- Learned skills
- World model
- **Size**: Unbounded

## Resource Budget (8GB GPU)

| Component | Memory | Compute |
|-----------|--------|---------|
| Vision IRP | 1.0 GB | GPU |
| Audio IRP | 0.5 GB | GPU |
| Motor IRP | 0.5 GB | GPU |
| TTS IRP | 1.0 GB | GPU |
| SAGE Core | 1.5 GB | GPU |
| Shims | 0.5 GB | GPU |
| Working Memory | 0.5 GB | GPU |
| GPU Mailboxes | 0.5 GB | GPU |
| **Reserved** | 2.0 GB | - |
| **Total** | 8.0 GB | - |

## Evolution Strategy

### Phase 1: Minimal Viable System
1. Vision IRP with TinyVAE shim
2. Simple motor control IRP  
3. Basic SAGE orchestrator
4. Trust-attention loop

### Phase 2: Sensory Expansion
1. Add audio IRP
2. Add temporal IRP
3. Multi-modal fusion
4. Surprise computation

### Phase 3: Advanced Effectors
1. Speech output (TTS)
2. Complex manipulation
3. Tool use primitives

### Phase 4: Memory & Learning
1. Episode consolidation
2. Skill discovery
3. World model updates

## Implementation Priorities

### Immediate (Week 1)
1. Define IRP Protocol interfaces
2. Implement GPU mailbox system
3. Create TinyVAE shim template
4. Basic trust-attention loop

### Short-term (Week 2-3)
1. Vision IRP with real camera input
2. Token quantization system
3. SAGE token vocabulary design
4. Simple motor control

### Medium-term (Week 4-6)
1. Multi-modal integration
2. Surprise computation
3. Trust weight learning
4. Memory system

## Design Decisions

### Latent Space Dimensions
- Vision: 16×16×64 spatial features
- Audio: 128-dim spectral features  
- Motor: 32-dim trajectory points
- Tokens: 1024 vocabulary size

### Shim Selection Criteria
- **TinyVAE**: When reconstruction matters
- **VQ Codebook**: When discrete is natural
- **Linear**: When dimension is small (<32)
- **Identity**: When already tokenized

### GPU vs CPU Allocation
- **GPU**: Parallel ops (conv, matmul, FFT)
- **CPU**: Sequential ops (control logic, I/O)
- **Both**: Large models split by layer

## Success Metrics

1. **Modularity**: Can swap IRPs without breaking system
2. **Efficiency**: <6GB GPU memory in use
3. **Latency**: <50ms sensor-to-action
4. **Trust**: Converges to stable values
5. **Surprise**: Decreases over time
6. **Learning**: Improves task performance

## Next Steps

1. Review and refine this specification
2. Inventory existing components:
   - What from our past work is reusable?
   - What can GR00T provide as teacher?
   - What pretrained models fit our size constraints?
3. Identify gaps requiring new development
4. Create development roadmap with priorities
5. Begin with highest-impact component

## Key Innovation

The system is designed for **continuous evolution** through:
- Modular boundaries enabling independent improvement
- Translation shims providing stable interfaces
- Trust-attention creating self-organizing behavior
- Surprise-driven learning without explicit supervision

This architecture prioritizes **adaptability over optimality** - we can start simple and evolve toward sophistication while maintaining compatibility.