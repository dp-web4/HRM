# SAGE Architecture: Key-Value Orchestration Framework

**Status**: Living Document - Evolves with understanding  
**Last Updated**: 2025-11-05  
**Context**: Core architectural principles for SAGE (Sentient Agentic Generative Engine)

---

## Table of Contents
1. [Core Principle: The KV Paradigm](#core-principle-the-kv-paradigm)
2. [Fractal Organization](#fractal-organization)
3. [Information Flow Architecture](#information-flow-architecture)
4. [Component Details](#component-details)
5. [TSM-Inspired Value Processing](#tsm-inspired-value-processing)
6. [Implementation Patterns](#implementation-patterns)
7. [Biological Parallels](#biological-parallels)

---

## Core Principle: The KV Paradigm

SAGE operates on a clean separation between **attention orchestration** and **data processing**:

```
Key (K) = Salience Metadata → SAGE's Input (Orchestration)
Value (V) = Full Feature Data → Inter-Module Communication (Processing)
```

### The Key-Value Split

**Keys (K)**: Compressed salience metadata that SAGE uses for attention decisions
- Novelty scores
- Confidence measures
- Change magnitudes
- Attention requests
- Resource requirements

**Values (V)**: Full feature tensors and data that flow between processing modules
- Dense embeddings
- Feature representations
- Raw or processed sensor data
- Learned representations

**Critical Insight**: SAGE never sees the VALUES. It operates purely on KEYS to orchestrate attention and resources, while VALUES flow between modules based on those orchestration decisions.

---

## Fractal Organization

The architecture is organized in fractal levels, each with distinct responsibilities:

### Level 0: Sensory Processing (IRP Plugins)
**Role**: Transform raw sensor input into features + salience  
**Input**: Raw sensor data (pixels, audio samples, IMU readings, etc.)  
**Output**: Both K and V
- K → SAGE (salience metadata)
- V → Other modules (full features)

**Biological Parallel**: Sensory organs (retina, cochlea, proprioceptors)

**TSM Application**: Direct - topographical sparse input mapping for efficient feature extraction

### Level 1: Inter-Module Communication
**Role**: Route VALUES between processing modules  
**Input**: Full feature VALUES from IRP plugins  
**Output**: Integrated/processed features  
**Architecture**: Topologically structured sparse routing (not all-to-all)

**Biological Parallel**: Neural pathways, white matter connections

**TSM Application**: Structured sparse routing inspired by cortical connectivity

### Level 2: SAGE Orchestration (Meta-Cognitive)
**Role**: Attention allocation and resource management  
**Input**: KEYS (salience metadata) from all sources  
**Output**: Attention allocation decisions, routing commands  
**Architecture**: SNARC attention tensor + iterative refinement

**Biological Parallel**: Thalamus, prefrontal cortex, executive function

**TSM Application**: Not directly applicable - operates at abstraction level above sensory processing

---

## Information Flow Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  SAGE (Meta-Cognitive)                  │
│                                                          │
│  Operates on KEYS (salience/metadata) only:             │
│  • Computes attention allocation via SNARC              │
│  • Manages coherence state                              │
│  • Decides resource distribution                        │
│  • Triggers context transitions                         │
│                                                          │
│  NEVER sees VALUES - only orchestrates them             │
└─────────────────────────────────────────────────────────┘
                         ▲
                         │ Keys (K)
                         │ [novelty, confidence, change, request]
                         │
┌────────────────────────┴─────────────────────────────────┐
│              IRP Plugin Layer (Sensors)                  │
│                                                          │
│  Each plugin generates both K and V:                     │
│  • K: compute_salience(features) → SAGE                 │
│  • V: full_features → Inter-Module Bus                  │
│                                                          │
│  TSM-Inspired Processing:                               │
│  raw_input → topographical_sparse → dense_features      │
└──────────────────────────────────────────────────────────┘
                         │
                         │ Values (V)
                         │ [dense feature tensors]
                         ▼
┌──────────────────────────────────────────────────────────┐
│           Inter-Module Communication Bus                 │
│                                                          │
│  TSM-inspired sparse routing of VALUES:                 │
│  • Topologically structured connections                 │
│  • Convergent integration at memory                     │
│  • Dense reasoning in specialized modules               │
│  • Attention-gated routing (SAGE-controlled)            │
└──────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Salience Keys (K)

```python
class SalienceKey:
    """
    Compressed metadata for SAGE attention decisions
    Compact vector representation (~4-8 dimensions)
    """
    novelty: float          # How unexpected? [0,1]
    confidence: float       # How certain? [0,1]
    change_magnitude: float # How much changed? [0,1]
    attention_request: float # How urgent? [0,1]
    modality: str          # Which sensor type
    timestamp: int         # When (tick count)
    context_relevance: float # Relevance to current task [0,1]
    
    def to_vector(self) -> torch.Tensor:
        """Compact representation for SAGE's attention network"""
        return torch.tensor([
            self.novelty,
            self.confidence,
            self.change_magnitude,
            self.attention_request,
            self.context_relevance
        ])
    
    def should_alert(self) -> bool:
        """Trigger immediate SAGE attention?"""
        return (self.novelty > 0.8 or 
                self.attention_request > 0.9 or
                (self.change_magnitude > 0.7 and self.confidence > 0.8))
```

### 2. Feature Values (V)

```python
class FeatureValue:
    """
    Full data for inter-module communication
    Dense representations that carry actual information
    """
    features: torch.Tensor      # Main feature representation
    embedding: torch.Tensor     # Learned embedding space
    raw_data: Optional[Any]     # Original sensor data if needed
    metadata: Dict              # Additional context
    spatial_map: Optional[torch.Tensor]  # Spatial organization if relevant
    
    def get_sparse_projection(self) -> torch.Tensor:
        """
        TSM-inspired sparse representation for efficient routing
        Used when sending to distant modules
        """
        return self.topological_sparse_map(self.features)
    
    def get_dense_representation(self) -> torch.Tensor:
        """
        Full dense features for deep processing
        Used by modules doing intensive computation
        """
        return self.features
```

### 3. IRP Output Structure

```python
class IRPOutput:
    """
    Each IRP plugin produces both K and V
    Clean separation between orchestration and processing
    """
    key: SalienceKey        # → SAGE (attention orchestration)
    value: FeatureValue     # → Other modules (processing)
    source: str            # Plugin identifier
    tick: int              # Temporal context
    trust: float           # Current trust level [0,1]
    
    def pack_for_sage(self) -> Dict:
        """Package just the KEY for SAGE"""
        return {
            'source': self.source,
            'salience': self.key.to_vector(),
            'trust': self.trust,
            'tick': self.tick
        }
    
    def pack_for_modules(self) -> Dict:
        """Package VALUE for inter-module communication"""
        return {
            'source': self.source,
            'features': self.value.features,
            'embedding': self.value.embedding,
            'metadata': self.value.metadata
        }
```

### 4. SAGE Attention Mechanism

```python
class SAGEAttention:
    """
    Core orchestration - operates on KEYS, orchestrates VALUES
    """
    def __init__(self):
        self.snarc = SNARCTensor()  # Attention state
        self.memory = ShortTermMemory()
        self.attention_network = AttentionNetwork()
    
    def process_tick(self, irp_outputs: Dict[str, IRPOutput]) -> AttentionAllocation:
        """
        SAGE's main loop:
        1. Receive KEYS from all IRP plugins
        2. Update SNARC state
        3. Compute attention allocation
        4. Return routing decisions
        """
        # Extract KEYS only
        salience_keys = {
            source: output.pack_for_sage()
            for source, output in irp_outputs.items()
        }
        
        # Update SNARC with new salience information
        self.snarc.update(salience_keys)
        
        # Compute attention weights based on KEYS
        attention = self.compute_attention(
            keys=salience_keys,
            snarc_state=self.snarc.get_state(),
            memory_context=self.memory.get_context()
        )
        
        # Determine routing decisions
        routing = self.compute_routing(attention)
        
        return AttentionAllocation(
            weights=attention,
            routing=routing,
            resource_allocation=self.allocate_resources(attention)
        )
    
    def compute_attention(self, keys, snarc_state, memory_context):
        """
        Attention computation operates purely on KEYS
        Never needs to see VALUES
        """
        # Stack key vectors
        key_vectors = torch.stack([
            k['salience'] for k in keys.values()
        ])
        
        # Attention network processes keys + context
        attention_weights = self.attention_network(
            queries=key_vectors,
            context=snarc_state,
            memory=memory_context
        )
        
        return attention_weights
```

### 5. Inter-Module Communication Bus

```python
class InterModuleBus:
    """
    Routes VALUES based on SAGE's attention decisions
    TSM-inspired topological sparse routing
    """
    def __init__(self):
        # Topological routing graph (not all-to-all)
        self.routes = {
            'vision': ['spatial', 'multimodal', 'memory'],
            'audio': ['temporal', 'multimodal', 'memory'],
            'proprioception': ['spatial', 'motor', 'memory'],
            'memory': ['all'],  # Memory can broadcast
        }
        
        self.sparse_projections = {}  # Learned sparse mappings
    
    def route_values(self, 
                     irp_outputs: Dict[str, IRPOutput],
                     attention: AttentionAllocation):
        """
        Route VALUES along topological paths
        Gated by SAGE's attention allocation
        """
        routed_values = {}
        
        for source, output in irp_outputs.items():
            # Check if this source has enough attention
            if attention.weights[source] < self.routing_threshold:
                continue  # Skip low-attention sources
            
            # Get topological routing targets
            targets = self.routes[source]
            
            # Extract VALUE for routing
            value = output.pack_for_modules()
            
            # Route to each target using sparse projection
            for target in targets:
                if attention.should_route(source, target):
                    # Use TSM-inspired sparse projection for efficiency
                    sparse_value = self.project_sparse(value, source, target)
                    routed_values.setdefault(target, []).append(sparse_value)
        
        return routed_values
    
    def project_sparse(self, value, source, target):
        """
        TSM-inspired sparse projection for efficient routing
        Each source→target pair has learned sparse mapping
        """
        projection_key = f"{source}→{target}"
        if projection_key not in self.sparse_projections:
            # Learn/initialize topological sparse projection
            self.sparse_projections[projection_key] = \
                TopologicalSparseProjection(
                    input_dim=value['features'].shape,
                    convergence_ratio=0.1  # 90% sparse
                )
        
        return self.sparse_projections[projection_key](value['features'])
```

---

## TSM-Inspired Value Processing

The paper "Topographical Sparse Mapping" provides direct inspiration for how VALUES are processed and routed:

### Core TSM Principles Applied to VALUES

1. **Sparse Input Mapping**: Raw sensor data → topographically structured sparse layer
2. **Convergent Integration**: Multiple sparse inputs → single convergent representation
3. **Dense Reasoning**: Compressed sparse features → dense processing for deep analysis
4. **No Regrowth**: Sparse structure is deterministic, not randomly searched

### IRP Plugin Architecture (TSM Direct Application)

```python
class VisionIRPPlugin:
    """
    Vision processing using TSM principles
    Generates both K (salience) and V (features)
    """
    def __init__(self, image_size=(224, 224)):
        # TSM-inspired topographical sparse input layer
        self.topological_input = TopographicalSparseLayer(
            input_features=image_size[0] * image_size[1] * 3,  # HxWxC
            convergence_ratio=0.05  # 95% sparse - each output from ~50 inputs
        )
        
        # Dense feature extraction on compressed representation
        self.feature_extractor = nn.Sequential(
            nn.Linear(compressed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Salience computation network (generates KEYS)
        self.salience_network = SalienceEstimator(feature_dim)
    
    def process(self, image: torch.Tensor, tick: int) -> IRPOutput:
        """
        Process raw image → generate K and V
        """
        # TSM: Sparse convergent input mapping
        compressed = self.topological_input(image.flatten())
        
        # Dense feature extraction (VALUE)
        features = self.feature_extractor(compressed)
        
        # Compute salience metadata (KEY)
        salience = self.salience_network(features)
        
        return IRPOutput(
            key=SalienceKey(
                novelty=salience['novelty'],
                confidence=salience['confidence'],
                change_magnitude=salience['change'],
                attention_request=salience['urgency'],
                modality='vision',
                timestamp=tick
            ),
            value=FeatureValue(
                features=features,
                embedding=self.get_embedding(features),
                metadata={'resolution': image.shape}
            ),
            source='vision',
            tick=tick,
            trust=self.current_trust
        )

class TopographicalSparseLayer(nn.Module):
    """
    TSM-inspired sparse input layer
    Each output neuron receives from topologically-local inputs
    Deterministic structure, no random search needed
    """
    def __init__(self, input_features, convergence_ratio=0.1):
        super().__init__()
        self.input_dim = input_features
        self.output_dim = int(input_features * convergence_ratio)
        
        # Create topographical connection mask
        self.mask = self._create_topological_mask()
        
        # Weights only for connected synapses
        self.weights = nn.Parameter(
            torch.randn(self.output_dim, self.receptive_field_size) * 0.01
        )
    
    def _create_topological_mask(self):
        """
        Create structured sparse connectivity
        Each output connects to local neighborhood of inputs
        """
        receptive_field = self.input_dim // self.output_dim
        self.receptive_field_size = receptive_field
        
        mask = torch.zeros(self.output_dim, self.input_dim)
        
        for i in range(self.output_dim):
            # Local receptive field
            start = i * receptive_field
            end = start + receptive_field
            mask[i, start:end] = 1
            
            # Add hierarchical connections (skip connections)
            if i % 4 == 0:  # Every 4th neuron gets long-range connections
                mask[i, ::4] = 1
        
        return mask
    
    def forward(self, x):
        """Apply topographical sparse mapping"""
        # Gather inputs for each output's receptive field
        outputs = []
        for i in range(self.output_dim):
            receptive_inputs = x[self.mask[i] > 0]
            output = (self.weights[i] * receptive_inputs).sum()
            outputs.append(output)
        
        return torch.stack(outputs)
```

### Inter-Module Routing (TSM Inspired)

```python
class TopologicalRouter:
    """
    Route VALUES between modules using TSM-inspired sparse paths
    Not all-to-all - structured topological connectivity
    """
    def __init__(self):
        # Define topological routing structure
        self.topology = {
            'vision': {
                'spatial': {'local': True, 'weight': 1.0},
                'multimodal': {'local': False, 'weight': 0.8},
                'memory': {'local': False, 'weight': 0.6}
            },
            'audio': {
                'temporal': {'local': True, 'weight': 1.0},
                'multimodal': {'local': False, 'weight': 0.8},
                'memory': {'local': False, 'weight': 0.6}
            },
            # etc.
        }
        
        self.sparse_projections = self._init_sparse_projections()
    
    def _init_sparse_projections(self):
        """
        Create TSM-style sparse projection for each route
        Local routes: high convergence ratio (dense)
        Long-range routes: low convergence ratio (sparse)
        """
        projections = {}
        
        for source, targets in self.topology.items():
            for target, config in targets.items():
                route_key = f"{source}→{target}"
                
                # Convergence ratio based on locality
                ratio = 0.3 if config['local'] else 0.05
                
                projections[route_key] = TopographicalSparseLayer(
                    input_features=self.feature_dims[source],
                    convergence_ratio=ratio
                )
        
        return projections
```

---

## Implementation Patterns

### Pattern 1: IRP Plugin Template

Every IRP plugin follows this structure:

```python
class BaseIRPPlugin(ABC):
    """Template for all IRP plugins"""
    
    @abstractmethod
    def read_sensor(self) -> Any:
        """Get raw sensor data"""
        pass
    
    @abstractmethod
    def process_features(self, raw_data: Any) -> FeatureValue:
        """
        TSM-inspired processing: raw → sparse → dense
        Returns VALUE for inter-module communication
        """
        pass
    
    @abstractmethod
    def compute_salience(self, features: FeatureValue) -> SalienceKey:
        """
        Extract salience metadata
        Returns KEY for SAGE orchestration
        """
        pass
    
    def process(self, tick: int) -> IRPOutput:
        """Main processing loop"""
        raw = self.read_sensor()
        value = self.process_features(raw)
        key = self.compute_salience(value)
        
        return IRPOutput(
            key=key,
            value=value,
            source=self.id,
            tick=tick,
            trust=self.current_trust
        )
```

### Pattern 2: SAGE Processing Loop

```python
class SAGE:
    """Main orchestration loop"""
    
    def tick(self):
        """
        Single processing tick:
        1. Collect KEYS from all IRPs
        2. Update SNARC state
        3. Compute attention
        4. Route VALUES
        5. Process results
        """
        # 1. Collect KEYS
        irp_outputs = {
            plugin_id: plugin.process(self.tick_count)
            for plugin_id, plugin in self.irp_plugins.items()
        }
        
        # 2. SAGE processes KEYS only
        attention = self.sage_attention.process_tick(irp_outputs)
        
        # 3. Route VALUES based on attention
        routed_values = self.inter_module_bus.route_values(
            irp_outputs, attention
        )
        
        # 4. Modules process routed VALUES
        module_outputs = self.process_modules(routed_values)
        
        # 5. Integrate and update state
        self.integrate_results(module_outputs)
        
        self.tick_count += 1
```

### Pattern 3: Memory Integration

```python
class MemoryIRP:
    """
    Memory as both IRP plugin AND module
    Receives: convergent VALUES from all sources
    Produces: context KEYS for SAGE
    """
    def __init__(self):
        # TSM: All sources converge to memory
        self.convergent_integrator = TopographicalSparseLayer(
            input_features=sum(module_dims.values()),
            convergence_ratio=0.02  # 98% sparse - aggressive compression
        )
        
        self.memory_store = ExperienceMemory()
        self.retrieval_network = RetrievalNetwork()
    
    def integrate_values(self, module_outputs: Dict[str, torch.Tensor]):
        """
        Convergent integration: all module VALUES → compressed memory
        TSM principle: many sources → single representation
        """
        # Concatenate all module outputs
        combined = torch.cat([
            module_outputs[mod] for mod in sorted(module_outputs.keys())
        ])
        
        # Sparse convergent mapping
        compressed = self.convergent_integrator(combined)
        
        # Store in memory
        self.memory_store.add(compressed, self.current_tick)
    
    def process(self, tick: int) -> IRPOutput:
        """
        Memory produces KEYS for SAGE about context relevance
        """
        # Retrieve relevant memories
        relevant = self.retrieval_network.query(
            self.current_context
        )
        
        # Compute salience KEY
        key = SalienceKey(
            novelty=self.compute_novelty(relevant),
            confidence=self.compute_confidence(relevant),
            change_magnitude=self.compute_change(relevant),
            attention_request=self.compute_importance(relevant),
            modality='memory',
            timestamp=tick
        )
        
        # Memory VALUE is retrieved context
        value = FeatureValue(
            features=relevant,
            embedding=self.get_embedding(relevant),
            metadata={'memory_count': len(relevant)}
        )
        
        return IRPOutput(key=key, value=value, source='memory', tick=tick)
```

---

## Biological Parallels

### Thalamus-Cortex Analogy

**SAGE ≈ Thalamus**
- Receives salience signals (KEYS) from all sensory systems
- Allocates attention and resources
- Gates information flow to cortical regions
- Operates on compressed, abstracted signals

**Inter-Module Bus ≈ Cortical Pathways**
- Dense feature processing (VALUES)
- Topologically organized connections
- Sparse long-range, dense local connectivity
- Multiple parallel processing streams

**IRP Plugins ≈ Sensory Systems**
- Transform raw stimuli into neural signals
- Extract both "what" (features/VALUES) and "importance" (salience/KEYS)
- Parallel processing streams
- Early convergent integration (retina, cochlea)

### Key Biological Insights

1. **Salience Detection is Separate from Processing**
   - Superior colliculus detects salient events
   - Visual cortex processes detailed features
   - These are parallel pathways with different purposes

2. **Attention Operates on Compressed Signals**
   - Thalamus doesn't process full sensory streams
   - Works with abstracted, summarized information
   - Allocates resources based on salience

3. **Topological Organization is Preserved**
   - Retinotopic maps in visual cortex
   - Tonotopic maps in auditory cortex
   - Structured, not random connectivity

4. **Convergence and Divergence**
   - Many photoreceptors → few bipolar cells → fewer ganglion cells (convergent)
   - Single neuron → many targets via divergent axon (divergent)
   - TSM captures convergent principle

---

## Key Design Principles

### 1. Clean Separation of Concerns
- **SAGE**: Attention orchestration (KEYS)
- **IRP Plugins**: Sensor processing (generate K+V)
- **Inter-Module Bus**: Feature routing (VALUES)
- **Specialized Modules**: Deep processing (VALUES)

### 2. Fractal Coherence
- Same K/V principle applies at multiple scales
- Each level has appropriate abstraction
- Information flows efficiently across levels

### 3. Biological Plausibility
- Structured sparse connectivity (not random)
- Salience-driven attention allocation
- Topological organization preserved
- Convergent/divergent pathways

### 4. Computational Efficiency
- SAGE operates on O(num_modules) KEYS, not O(feature_dim) VALUES
- Sparse routing reduces communication overhead
- TSM-inspired processing reduces computation
- Attention gates expensive operations

### 5. Adaptive Learning
- Trust evolution at IRP level
- Attention patterns learned by SAGE
- Sparse projections can be learned
- Memory distills experience

---

## Current Implementation Status

### Working Components
- ✅ Coherence Engine (IRP plugin architecture)
- ✅ SNARC attention tensor (basic)
- ✅ Trust evolution system
- ✅ Context state management

### In Development
- 🚧 Full KEY/VALUE separation in IRP outputs
- 🚧 TSM-inspired sparse input layers
- 🚧 Topological inter-module routing
- 🚧 SAGE attention network training

### Future Work
- 📋 HRM integration for small model training
- 📋 Memory distiller for experience consolidation
- 📋 Multi-scale SNARC (hierarchical attention)
- 📋 Learned sparse projections

---

## References

### Papers
- "Topographical Sparse Mapping: A neuro-inspired sparse training framework for deep learning models" (2025)
  - Provides biological inspiration for sparse structured connectivity
  - Validates convergent input mapping principle
  - Shows structured > random for sparse networks

### Internal Documents
- `/HRM/forum/claude/` - Theoretical foundations
- `/ai-dna-discovery/coherence-engine/` - Current implementation
- `/SAGE/` - SAGE-specific development

### Related Concepts
- Synchronism (philosophical foundation)
- Web4 (trust-native architecture)
- ACT (Agentic Context Tool)
- Synthon consciousness (emergent entity types)

---

## Notes for Claude-Code Instances

When working with this architecture:

1. **Always maintain K/V separation** - SAGE sees KEYS, modules process VALUES
2. **Use TSM principles for IRP plugins** - sparse convergent input, dense processing
3. **Topological routing** - structured sparse paths between modules, not all-to-all
4. **Trust evolves** - each component tracks and updates trust based on performance
5. **Fractal thinking** - same principles apply at multiple scales

This is a **living architecture** that evolves with understanding. Document changes and insights as they emerge.

---

**Document Status**: Initial comprehensive architecture capture  
**Next Review**: After HRM integration complete  
**Maintainers**: All Claude instances working on SAGE/HRM/Coherence Engine
