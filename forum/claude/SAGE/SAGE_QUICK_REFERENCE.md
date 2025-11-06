# SAGE Architecture Quick Reference

**For Claude-Code Instances: Read this first, then dive into SAGE_ARCHITECTURE.md**

---

## Core Concept (30 seconds)

```
SAGE = Attention Orchestrator (Meta-Cognitive)
IRP Plugins = Sensory Processors (Generate Features)
Inter-Module Bus = Feature Router (Structured Sparse)

Key (K) = Salience metadata → SAGE decides attention
Value (V) = Full features → Modules process data

SAGE NEVER sees VALUES, only KEYS
```

---

## The KV Split

```python
# IRP Plugin outputs BOTH:
class IRPOutput:
    key: SalienceKey     # → SAGE (4-8 floats: novelty, confidence, etc.)
    value: FeatureValue  # → Other modules (full feature tensors)
```

**Why this matters:**
- SAGE operates on O(num_modules) not O(feature_dim)
- Clean separation: orchestration vs processing
- Biological parallel: thalamus vs cortex

---

## Fractal Levels

```
Level 0: IRP Plugins (sensors → features)
         ↓ (generates K+V)
         
Level 1: Inter-Module Bus (feature routing)
         ↓ (routes V based on attention)
         
Level 2: SAGE (attention orchestration)
         ↓ (processes K, allocates attention)
```

**TSM Paper applies to**: Level 0 and 1 (not Level 2)
- Level 0: Topographical sparse input mapping
- Level 1: Structured sparse inter-module routing
- Level 2: Already meta-cognitive, uses SNARC

---

## Key Classes

```python
# What SAGE receives
class SalienceKey:
    novelty: float          # [0,1]
    confidence: float       # [0,1]
    change_magnitude: float # [0,1]
    attention_request: float # [0,1]

# What modules process
class FeatureValue:
    features: torch.Tensor      # Dense features
    embedding: torch.Tensor     # Learned space
    
# What IRPs produce
class IRPOutput:
    key: SalienceKey    # For SAGE
    value: FeatureValue # For modules
```

---

## TSM Principles (from paper)

Apply to **IRP plugins** and **inter-module routing**:

1. **Topographical sparse input** - structured, not random
2. **Convergent mapping** - many inputs → few features
3. **Dense processing after** - sparse → dense → output
4. **No regrowth** - deterministic structure

```python
# IRP Plugin pattern:
raw_sensor_data 
  → TopographicalSparseLayer (95% sparse)
  → DenseFeatureExtractor
  → {key: SalienceKey, value: FeatureValue}
```

---

## Implementation Checklist

When writing IRP plugin:
- ✅ Generate both K and V
- ✅ Use TSM-inspired sparse input layer
- ✅ K is compressed (4-8 dimensions)
- ✅ V is full features (any dimensionality)
- ✅ Track trust evolution

When writing SAGE component:
- ✅ Only operate on KEYS
- ✅ Never access VALUES directly
- ✅ Update SNARC state
- ✅ Return attention allocation

When writing inter-module routing:
- ✅ Use topological sparse paths
- ✅ Not all-to-all connectivity
- ✅ Gate by attention weights
- ✅ Consider convergence at memory

---

## Key Files

```
/HRM/forum/claude/SAGE_ARCHITECTURE.md     # Full documentation
/ai-dna-discovery/coherence-engine/        # Current implementation
/HRM/                                       # Training framework
```

---

## Biological Analogies

```
SAGE           ≈ Thalamus (attention gating)
IRP Plugins    ≈ Sensory organs (vision, audio, etc.)
Inter-Module   ≈ Cortical pathways (white matter)
Memory         ≈ Hippocampus (consolidation)
```

---

## Common Mistakes to Avoid

❌ **SAGE processing VALUES** - it only sees KEYS
❌ **All-to-all module connections** - use topological sparse
❌ **Random sparse connectivity** - use structured (TSM-inspired)
❌ **Ignoring trust evolution** - it's core to the system
❌ **Dense pretraining** - start sparse (TSM principle)

---

## Quick Patterns

### IRP Plugin Template
```python
class MyIRPPlugin:
    def process(self, tick: int) -> IRPOutput:
        raw = self.read_sensor()
        
        # TSM: sparse → dense
        compressed = self.topological_sparse_layer(raw)
        features = self.dense_processor(compressed)
        
        # Generate K and V
        return IRPOutput(
            key=self.compute_salience(features),  # K
            value=FeatureValue(features=features), # V
            source=self.id,
            tick=tick
        )
```

### SAGE Loop
```python
def tick(self):
    # 1. Collect KEYS only
    irp_outputs = {id: plugin.process(tick) for id, plugin in self.irps.items()}
    
    # 2. SAGE processes KEYS
    attention = self.sage_attention.process_tick(irp_outputs)
    
    # 3. Route VALUES based on attention
    routed = self.inter_module_bus.route_values(irp_outputs, attention)
    
    # 4. Modules process VALUES
    results = self.process_modules(routed)
```

---

## Key Insights from TSM Paper

1. **Structured > Random** for sparse networks
2. **Convergent input mapping** drastically reduces parameters without accuracy loss
3. **Topographical organization** improves convergence speed
4. **Biological inspiration** leads to computational efficiency
5. **No regrowth needed** - deterministic sparse structure works

---

## Current Status

**Working:**
- Basic coherence engine with IRP architecture
- Trust evolution
- Context state management

**In Progress:**
- Full K/V separation
- TSM-inspired layers
- Topological routing

**Planned:**
- HRM integration (small model training)
- Memory distiller (experience consolidation)
- Multi-scale SNARC

---

## Questions to Ask When Developing

1. **Am I at the right fractal level?**
   - Sensor processing → IRP plugin
   - Feature routing → Inter-module
   - Attention allocation → SAGE

2. **Am I maintaining K/V separation?**
   - SAGE code touching VALUES? ❌
   - Module code deciding attention? ❌
   - IRP producing both K and V? ✅

3. **Am I using structured sparsity?**
   - Random pruning? ❌
   - Topological patterns? ✅
   - All-to-all connections? ❌

4. **Would biology do this?**
   - Thalamus processing full visual stream? ❌
   - Retina using all-to-all connectivity? ❌
   - Salience detection separate from processing? ✅

---

**Read SAGE_ARCHITECTURE.md for complete details**
