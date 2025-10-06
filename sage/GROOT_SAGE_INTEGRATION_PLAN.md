# GR00T → Vision IRP → SAGE Integration Plan

**Date**: October 6, 2025  
**Status**: Planning Phase  
**Author**: Claude with Dennis

## Current State Analysis

### What We Have

#### 1. SAGE Architecture
- **Input Expected**: Token sequences (discrete IDs) for pattern solving
- **Model Type**: Transformer with H/L modules for hierarchical reasoning  
- **Training Input**: Currently uses synthetic patterns, expects `input_ids` tensor
- **Loss Function**: Multi-component (task, attention, SNARC, halt, diversity)
- **Problem**: Trying to feed continuous image tensors to a model expecting discrete tokens

#### 2. IRP Framework
- **Purpose**: Universal refinement pattern for all modalities
- **Plugins Available**:
  - `VisionIRP`: Refines in latent space (placeholder implementation)
  - `LanguageIRP`: Masked denoising for text
  - `ControlIRP`: Trajectory planning
  - `MemoryIRP`: Sleep consolidation
  - `NeuTTSAirIRP`: Text-to-speech (working!)
- **Orchestrator**: `HRMOrchestrator` manages plugins with trust-weighted ATP allocation
- **Key Interface**: `init_state()` → `step()` → `extract()`

#### 3. GR00T Capabilities  
- **Model**: 10B parameter multimodal transformer from NVIDIA
- **Inputs**: Camera images (224×224 RGB), language instructions, proprioception
- **Outputs**: 
  - Robot actions (7-DOF joint positions)
  - Visual features (embeddings)
  - Attention maps
  - Object detections
- **World Simulator**: `groot_world_sim.py` renders internal world model

### The Problem

We have three disconnected systems:
1. **GR00T** produces visual features and actions from camera input
2. **Vision IRP** expects to refine visual latents (but is a placeholder)
3. **SAGE** expects discrete token sequences, not continuous features

## Proposed Architecture

```
┌─────────────────┐
│  Camera/Sim     │
│  (224×224 RGB)  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│     GR00T       │
│  Vision Model   │
│                 │
│ Outputs:        │
│ - Features      │
│ - Attention     │
│ - Objects       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Vision IRP    │
│                 │
│ - Encode to     │
│   latent tokens │
│ - Iterative     │
│   refinement    │
│ - Discretize    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Tokenization    │
│    Layer        │
│                 │
│ Features →      │
│ Token IDs       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│     SAGE        │
│  Orchestrator   │
│                 │
│ - Pattern       │
│   reasoning     │
│ - H/L modules   │
│ - ATP budget    │
└─────────────────┘
```

## Implementation Plan

### Phase 1: Camera Simulation Interface
**Goal**: Provide consistent visual input stream

```python
class CameraSimulator:
    """Simulates camera input for GR00T"""
    
    def __init__(self, mode='synthetic'):
        self.mode = mode  # 'synthetic', 'webcam', 'dataset'
        self.resolution = (224, 224)
        
    def get_frame(self) -> torch.Tensor:
        """Get next camera frame"""
        if self.mode == 'synthetic':
            return self.generate_synthetic_scene()
        elif self.mode == 'webcam':
            return self.capture_webcam()
        elif self.mode == 'dataset':
            return self.load_dataset_frame()
    
    def generate_synthetic_scene(self):
        """Generate scene with objects for manipulation"""
        # Create scene with table, cubes, robot
        # Add noise, lighting variation
        # Return [3, 224, 224] tensor
```

### Phase 2: GR00T Feature Extraction
**Goal**: Extract meaningful features from visual input

```python
class GR00TVisionEncoder:
    """Wraps GR00T to extract visual features"""
    
    def __init__(self, groot_model):
        self.model = groot_model
        self.feature_dim = 1024  # GR00T hidden dim
        
    def encode(self, image: torch.Tensor) -> Dict:
        """Extract features from image"""
        with torch.no_grad():
            outputs = self.model.forward_vision(image)
        
        return {
            'features': outputs['visual_features'],  # [B, seq_len, dim]
            'attention': outputs['attention_weights'],  # [B, H, seq, seq]
            'objects': self.detect_objects(outputs),
            'confidence': self.compute_confidence(outputs)
        }
```

### Phase 3: Vision IRP Implementation
**Goal**: Bridge GR00T features to SAGE tokens

```python
class GR00TVisionIRP(IRPPlugin):
    """Refines GR00T features into discrete tokens for SAGE"""
    
    def __init__(self, config):
        super().__init__(config)
        self.groot_encoder = GR00TVisionEncoder(...)
        self.quantizer = VectorQuantizer(
            num_codes=1024,  # Vocabulary size
            dim=1024  # Feature dimension
        )
        
    def init_state(self, image: torch.Tensor, task_ctx: Dict):
        """Initialize from camera image"""
        # 1. Extract GR00T features
        groot_outputs = self.groot_encoder.encode(image)
        
        # 2. Initialize refinement state
        return IRPState(
            x=groot_outputs['features'],
            step_idx=0,
            meta={
                'attention': groot_outputs['attention'],
                'objects': groot_outputs['objects'],
                'task': task_ctx
            }
        )
    
    def step(self, state: IRPState, budget: float) -> Tuple[IRPState, float]:
        """Refine features toward discrete tokens"""
        features = state.x
        
        # 1. Apply refinement (denoising, sharpening)
        refined = self.refine_features(features)
        
        # 2. Quantize to discrete codes
        quantized, indices = self.quantizer(refined)
        
        # 3. Compute energy (quantization error)
        energy = F.mse_loss(refined, quantized)
        
        return IRPState(
            x=quantized,
            step_idx=state.step_idx + 1,
            meta={**state.meta, 'token_ids': indices}
        ), energy.item()
    
    def extract(self, state: IRPState) -> torch.Tensor:
        """Extract token IDs for SAGE"""
        return state.meta['token_ids']  # [B, seq_len]
```

### Phase 4: SAGE Integration
**Goal**: Feed visual tokens into SAGE for reasoning

```python
class VisualSAGETrainer:
    """Trains SAGE on visual reasoning tasks"""
    
    def __init__(self):
        self.camera = CameraSimulator('synthetic')
        self.vision_irp = GR00TVisionIRP(...)
        self.sage_model = SAGEModel(...)
        
    def generate_batch(self) -> Dict:
        """Generate training batch"""
        # 1. Get camera frame
        image = self.camera.get_frame()
        
        # 2. Define task (e.g., "find red cube")
        task = self.generate_task()
        
        # 3. Refine through Vision IRP
        state = self.vision_irp.init_state(image, task)
        for _ in range(10):  # Refine
            state, _ = self.vision_irp.step(state, budget=1.0)
        
        # 4. Extract tokens
        input_ids = self.vision_irp.extract(state)
        
        # 5. Create target (ground truth action/answer)
        targets = self.create_targets(task, image)
        
        return {
            'input_ids': input_ids,
            'targets': targets,
            'context': task
        }
    
    def train_step(self, batch):
        """Single SAGE training step"""
        outputs = self.sage_model(
            batch['input_ids'],
            batch['context']
        )
        
        loss = self.compute_loss(outputs, batch['targets'])
        return loss
```

### Phase 5: Orchestrated System
**Goal**: Full pipeline with all components

```python
class GR00TSAGESystem:
    """Complete vision-to-reasoning system"""
    
    def __init__(self):
        self.orchestrator = HRMOrchestrator({
            'enable_vision': True,  # GR00T Vision IRP
            'enable_language': True,
            'enable_control': True,
            'total_ATP': 100.0
        })
        
    def process_scene(self, image: torch.Tensor, instruction: str):
        """Process visual scene with language instruction"""
        
        inputs = {
            'vision': image,
            'language': instruction
        }
        
        # Orchestrator manages refinement across plugins
        results = self.orchestrator.process(inputs)
        
        # Results contain:
        # - Visual tokens from GR00T
        # - Language understanding
        # - Planned actions
        # - Confidence scores
        
        return results
```

## Key Design Decisions

### 1. Tokenization Strategy
- **Option A**: Vector quantization (VQ-VAE style) - discrete codes
- **Option B**: Learned tokenizer - trainable embedding lookup
- **Option C**: Clustering - k-means on features
- **Recommendation**: Start with VQ-VAE approach, proven to work

### 2. Feature Granularity  
- **Option A**: Patch tokens (16×16 patches = 196 tokens)
- **Option B**: Object tokens (one per detected object)
- **Option C**: Hierarchical (global + local tokens)
- **Recommendation**: Start with patch tokens, matches ViT architecture

### 3. Training Curriculum
- **Stage 1**: Simple object detection (is there a red cube?)
- **Stage 2**: Spatial reasoning (cube left of sphere?)
- **Stage 3**: Manipulation planning (move cube to target)
- **Stage 4**: Multi-step tasks (sort objects by color)

### 4. Camera Simulation
- Start with synthetic scenes for controlled training
- Add noise, lighting variation for robustness
- Gradually introduce real webcam data
- Eventually use robot camera in deployment

## Success Metrics

1. **Tokenization Quality**: Reconstruction error < 0.05
2. **SAGE Loss**: Break below 0.50 plateau
3. **Task Performance**: 
   - Object detection: 90% accuracy
   - Spatial reasoning: 80% accuracy
   - Action planning: 70% success rate
4. **Energy Convergence**: Monotonic decrease in IRP
5. **Trust Scores**: Vision IRP trust > 0.8

## Implementation Timeline

- **Week 1**: Camera simulator + synthetic scene generation
- **Week 2**: GR00T feature extraction wrapper
- **Week 3**: Vision IRP with vector quantization
- **Week 4**: SAGE training on visual tokens
- **Week 5**: Full orchestrated system
- **Week 6**: Testing and optimization

## Open Questions

1. **Token Vocabulary Size**: 256, 512, 1024, or larger?
2. **Sequence Length**: How many patches/tokens per image?
3. **Context Injection**: How to provide task context to SAGE?
4. **Memory Integration**: Should visual tokens go into memory IRP?
5. **Action Space**: Discrete actions or continuous control?

## Next Steps

1. Review this plan and provide feedback
2. Decide on tokenization strategy
3. Implement camera simulator
4. Create simple synthetic dataset
5. Build GR00T feature extractor
6. Implement Vision IRP with quantization
7. Connect to SAGE training loop

## Notes

The key insight is that SAGE needs discrete tokens, not continuous features. The Vision IRP plugin acts as the bridge, refining GR00T's continuous features into discrete tokens through vector quantization. This maintains the IRP pattern (iterative refinement) while providing the interface SAGE expects.

The loss plateau at 0.52 makes sense now - we're training on synthetic patterns when we should be training on real visual reasoning tasks. With proper visual input through this pipeline, SAGE can learn meaningful vision-language grounding.