# SNARC Analysis: Concept vs Implementation

**Date**: October 12, 2025
**Purpose**: Critical evaluation of SNARC in SAGE architecture
**Approach**: Take nothing as given - evaluate usefulness and fit

---

## Origins

### Transformer-Sidecar (Richard Aragon)
**Source**: `/memory/SNARC/` - grabbed from open source

**Original Concept**:
- **Bolt-on persistent memory** for transformers
- **SNARC gating** (Surprise, Novelty, Arousal, Conflict, Reward)
- **Hebbian fast-weights** (~130KB constant footprint)
- **Selective memory**: Decide what experiences to remember

**Key Innovation**: Not backprop-based, uses affect-regulated gating to decide what enters long-term memory.

### Our Extensions (Concept)

From `/forum/nova/concepts/SAGE-SNARC.md`:

**Vision**: SNARC as **universal salience filter** for ALL sensors/effectors, not just memory.

**Key Ideas**:
1. Each SNARC dimension = "color channel" for salience
2. Every IRP plugin has its own SNARC grid overlay
3. Each sensor/effector gets dynamic salience scores
4. SAGE uses SNARC to prioritize: what needs attention, what type, how much
5. Fractal tiling: local SNARC → intermediate fractal → global puzzle

**The Vision**:
```
Sensor Stream → SNARC Grid Overlay → Salience Vector → SAGE Priority Queue
```

Each modality (vision, audio, motor, etc.) has SNARC coloring that makes them comparable despite different latent spaces.

---

## Current Implementation

### Location: `/sage/attention/snarc_scorer.py`

**What It Is**:
- PyTorch `nn.Module` (not original Hebbian fast-weights)
- Learned neural networks for each SNARC dimension
- Operates on hidden states `[batch, seq, hidden_size]`
- Returns 5D scores or combined attention weight

**The 5 Dimensions**:

1. **Surprise** - Prediction error via learned predictor network
   ```python
   predictions = self.predictor(input[:, :-1])
   surprise = mse_loss(predictions, input[:, 1:])
   ```

2. **Novelty** - Cosine distance from memory bank
   ```python
   similarities = cosine_similarity(input, memory_bank)
   novelty = 1.0 - max_similarity
   ```

3. **Arousal** - Entropy as complexity proxy
   ```python
   entropy = -(probs * log(probs)).sum()
   arousal = normalized_entropy * learned_weight
   ```

4. **Conflict** - Variance across hidden dims
   ```python
   variance = input.var(dim=-1)
   conflict = normalized_variance * learned_weight
   ```

5. **Reward** - Explicit external signal
   ```python
   reward = task_success if provided else 0.0
   ```

**Interface**:
```python
scorer = SNARCScorer(hidden_size=768, memory_size=1000)
scores = scorer(input_states, context, task_success, return_components=True)
# Returns: dict with {S, N, A, R, C} or combined attention weight
```

---

## The Gap: Concept vs Reality

### Conceptual Vision (Per-Sensor SNARC)
```
Vision IRP → Vision SNARC Grid → Vision salience vector
Audio IRP → Audio SNARC Grid → Audio salience vector
Motor IRP → Motor SNARC Grid → Motor salience vector
       ↓
SAGE compares across modalities via common SNARC space
```

### Current Implementation (Monolithic SNARC)
```
All sensors → Some hidden state → Single SNARCScorer → One salience score
```

**The Issues**:

1. **Not sensor-specific**: One scorer for everything, not per-sensor
2. **Requires hidden states**: Expects `[batch, seq, hidden]` tensors, not raw sensor data
3. **Learned, not algorithmic**: Neural networks need training, not just running
4. **Not spatial**: No "grid overlay" on visual field or audio spectrum
5. **Memory-centric**: Memory bank tracks novelty, but that's temporal not spatial

---

## Critical Questions

### 1. Is Learned SNARC Right for SAGE?

**Original SNARC** (Transformer-Sidecar):
- ✅ Hebbian updates (no backprop)
- ✅ Fast weights (constant memory)
- ✅ Immediate operation (no training)
- ❌ Memory-only (not universal)

**Current SNARC** (PyTorch):
- ❌ Requires training (where's the training data?)
- ❌ Learned networks (adds parameters to optimize)
- ❌ Not immediate (needs gradient descent)
- ✅ More flexible (can learn patterns)

**Question**: Do we want **learned salience** or **algorithmic salience**?

**Biological analogy**:
- Surprise, novelty, arousal - these are **computed**, not learned
- Your brain doesn't learn that unexpected things are surprising
- It calculates surprise from prediction error

**Implication**: Maybe SNARC shouldn't be learned neural nets. Maybe it should be:
- Surprise = prediction error (algorithmic)
- Novelty = distance from past observations (algorithmic)
- Arousal = signal intensity/variance (algorithmic)
- Conflict = multi-source disagreement (algorithmic)
- Reward = explicit signal (provided)

### 2. Should SNARC Be Spatial/Grid-Based?

**Conceptual vision**: SNARC grid overlays sensor space
- Vision: 2D grid over image (which pixels are salient?)
- Audio: Temporal bins (which moments are salient?)
- Motor: Actuator slots (which joints need attention?)

**Current implementation**: Global hidden state scoring
- No spatial structure
- Operates on flattened representations
- Loses "where" information

**Question**: Is spatial salience important?

**Consider**: Visual attention is spatially selective
- You don't give equal weight to all pixels
- Some regions are more salient
- Attention moves around the visual field

**Implication**: Spatial SNARC grids make sense for vision/audio, but current implementation loses this.

### 3. Do We Need Per-Sensor SNARC?

**Conceptual vision**: Each sensor has own SNARC instance
- Vision SNARC knows about visual patterns
- Audio SNARC knows about audio patterns
- Different memory banks, different baselines

**Current implementation**: Single global SNARC
- One memory bank (mixed modalities?)
- One set of learned weights
- One salience computation

**Question**: Is cross-modal comparison the goal?

**If yes**: Need common salience space → global SNARC makes sense
**If no**: Need modality-specific salience → per-sensor SNARC makes sense

**Biological reality**: Both exist!
- Early sensory processing: modality-specific salience (V1, A1 have their own filters)
- Higher integration: cross-modal comparison (parietal cortex integrates)

**Implication**: Maybe we need **both** - per-sensor SNARC feeding into global comparator.

### 4. What About the Training Problem?

Current SNARCScorer has ~100K+ learnable parameters:
- 5 neural networks (one per dimension)
- Predictor network
- Attention weighting network

**Where does training data come from?**
- No labeled SNARC scores
- No ground truth for "surprise" or "novelty"
- Self-supervised only (from observations)

**Current training** (`train_sage.py`):
```python
self.snarc_scorer = SNARCScorer(config.hidden_size).to(self.device)
# But where's the loss function for SNARC?
# How do we know if surprise score is "correct"?
```

**Question**: Can we even train this meaningfully?

**Alternative**: Use algorithmic SNARC (no learning needed)
- Surprise = prediction error (compute from model predictions)
- Novelty = distance from memory (compute from stored states)
- Arousal = signal variance (compute from raw data)
- Conflict = cross-source disagreement (compute from multiple inputs)
- Reward = explicit (provided by environment)

---

## What Would Work Better?

### Proposal: Algorithmic Per-Sensor SNARC

```python
class SensorSNARC:
    """SNARC scoring for a specific sensor"""

    def __init__(self, sensor_name: str, memory_size: int = 1000):
        self.sensor_name = sensor_name
        self.memory = deque(maxlen=memory_size)  # Past observations
        self.predictor = None  # Simple AR model or None

    def score(self, observation: torch.Tensor, context: dict) -> dict:
        """Compute SNARC scores algorithmically"""

        # 1. Surprise: prediction error
        if self.predictor and len(self.memory) > 0:
            predicted = self.predictor.predict(list(self.memory)[-5:])
            surprise = F.mse_loss(predicted, observation).item()
        else:
            surprise = 0.5  # Default for no predictor

        # 2. Novelty: distance from memory
        if len(self.memory) > 0:
            past_obs = torch.stack(list(self.memory))
            distances = F.pairwise_distance(observation.flatten(),
                                           past_obs.flatten(start_dim=1))
            novelty = distances.min().item()  # Closest distance
        else:
            novelty = 1.0  # Everything novel at start

        # 3. Arousal: signal intensity
        arousal = observation.std().item()  # Variance as proxy

        # 4. Conflict: N/A for single sensor (computed at fusion)
        conflict = 0.0

        # 5. Reward: from context
        reward = context.get('reward', 0.0)

        # Store for future novelty calculation
        self.memory.append(observation.clone().detach())

        return {
            'surprise': surprise,
            'novelty': novelty,
            'arousal': arousal,
            'conflict': conflict,
            'reward': reward,
            'combined': self._combine(surprise, novelty, arousal, reward)
        }

    def _combine(self, s, n, a, r, weights=(0.3, 0.3, 0.2, 0.2)):
        """Weighted combination of SNARC dimensions"""
        return s * weights[0] + n * weights[1] + a * weights[2] + r * weights[3]
```

**Benefits**:
- No training needed
- Works immediately
- Per-sensor specificity
- Memory-based novelty
- Simple prediction for surprise
- Algorithmic, not learned

### Proposal: Spatial SNARC Grids

For vision:
```python
class SpatialSNARC:
    """SNARC grid overlay on spatial sensor (vision)"""

    def score_grid(self, image: torch.Tensor) -> torch.Tensor:
        """Return SNARC heatmap same size as image"""

        H, W = image.shape[-2:]

        # Compute per-pixel salience
        snarc_map = torch.zeros(5, H, W)  # 5 SNARC dims × H × W

        # Surprise: gradient magnitude (edge detection)
        snarc_map[0] = compute_spatial_gradients(image)

        # Novelty: compare to memory (spatial)
        snarc_map[1] = compute_spatial_novelty(image, self.memory)

        # Arousal: local variance
        snarc_map[2] = compute_local_variance(image)

        # Conflict: not applicable spatially
        snarc_map[3] = 0.0

        # Reward: from attention supervision
        snarc_map[4] = self.reward_map if self.reward_map else 0.0

        return snarc_map  # [5, H, W]
```

**Benefits**:
- Preserves spatial structure
- Can visualize attention heatmaps
- Enables spatial prioritization
- Compatible with visual attention mechanisms

### Proposal: Hierarchical SNARC

```
Per-Sensor SNARC → Modality SNARC → Global SNARC
      ↓                 ↓                ↓
  Spatial grid     Cross-sensor    Cross-modal
  (where in         (within        (across all
   sensor)           modality)       sensors)
```

**Example**:
1. Vision IRP computes spatial SNARC grid for its image
2. Modality level: Compare left camera vs right camera SNARC
3. Global level: Compare vision salience vs audio salience

**Conflict emerges at cross-modal level**:
- Vision says: "Object on left"
- Audio says: "Sound from right"
- Conflict = disagreement between trusted sources

---

## Recommendations

### 1. Replace Learned SNARC with Algorithmic SNARC

**Why**:
- No training data for ground-truth salience
- Biological salience is computed, not learned
- Immediate operation (no training phase)
- Interpretable (we know what each dimension means)

**How**:
- Surprise = prediction error (simple AR or constant baseline)
- Novelty = distance from memory bank (cosine/L2)
- Arousal = signal variance/intensity
- Conflict = cross-source disagreement (computed at fusion)
- Reward = external signal

### 2. Implement Per-Sensor SNARC

**Why**:
- Different modalities have different salience patterns
- Vision novelty ≠ audio novelty
- Enables modality-specific memory
- Matches biological sensory processing

**How**:
- Each IRP plugin has `SensorSNARC` instance
- Sensor-specific memory banks
- Sensor-specific baselines
- Output: per-sensor salience vector

### 3. Add Spatial SNARC for Vision/Audio

**Why**:
- Visual attention is spatially selective
- Audio attention is temporally selective
- "Where" information is critical
- Enables attention visualization

**How**:
- Vision: 2D SNARC grid (H × W × 5 dimensions)
- Audio: 1D SNARC bins (time × 5 dimensions)
- Output: spatial/temporal salience maps

### 4. Hierarchical Integration

**Why**:
- Local salience (per-sensor) different from global salience (cross-modal)
- Conflict only makes sense cross-modally
- Matches biological hierarchy (V1 → IT → parietal)

**How**:
```
Level 1: Per-sensor spatial SNARC (local salience)
Level 2: Per-modality aggregation (e.g., stereo vision fusion)
Level 3: Cross-modal comparison (vision vs audio vs motor)
Level 4: Global priority queue (what SAGE attends to)
```

---

## Evaluation Against SAGE Objectives

### Does Current SNARC Serve SAGE?

**SAGE's Purpose**: Attention orchestrator - decide what matters, allocate resources

**Current SNARC**:
- ❌ Requires training (when? how?)
- ❌ Single global scorer (not per-sensor)
- ❌ Operates on hidden states (not raw sensors)
- ❌ No spatial structure (loses "where")
- ✅ Has all 5 dimensions conceptually
- ✅ Memory bank for novelty

**Verdict**: Partially useful, but doesn't match vision

### Would Algorithmic Per-Sensor SNARC Be Better?

**For SAGE orchestration**:
- ✅ Immediate operation (no training)
- ✅ Per-sensor specificity (modality-aware)
- ✅ Works on raw observations (no hidden states)
- ✅ Spatial/temporal structure (preserves "where")
- ✅ Interpretable (know what salience means)
- ✅ Matches biological processing

**Verdict**: Much better fit for SAGE architecture

---

## Action Items

### Short-term: Fix Current Implementation

1. **Use current SNARC algorithmically** (no training)
   - Remove learned networks
   - Compute dimensions from observations directly
   - Keep memory bank for novelty

2. **Make it per-sensor**
   - Each IRP plugin gets own SNARC instance
   - Sensor-specific memory
   - Modality-specific baselines

3. **Test in SAGE loop**
   - Verify salience scores make sense
   - Compare across sensors
   - Use for resource allocation

### Medium-term: Add Spatial Structure

1. **Implement SpatialSNARC for vision**
   - 2D grids overlaying images
   - Spatial gradient for surprise
   - Local variance for arousal
   - Memory-based spatial novelty

2. **Implement TemporalSNARC for audio**
   - 1D bins over time
   - Temporal prediction for surprise
   - Acoustic novelty detection

3. **Visualize attention**
   - Heatmaps showing where SAGE attends
   - Debug attention allocation
   - Validate salience computation

### Long-term: Hierarchical Integration

1. **Build three-level hierarchy**
   - L1: Per-sensor spatial/temporal SNARC
   - L2: Per-modality aggregation
   - L3: Cross-modal global salience

2. **Add conflict detection**
   - Cross-sensor disagreement
   - Trust-weighted conflict scores
   - Resolution strategies

3. **Integrate with R6**
   - SNARC salience → R6 Resources
   - Priority queue → R6 Request
   - Allocate based on salience × trust

---

## The Fundamental Question

**Is SNARC the right abstraction for attention in SAGE?**

**Arguments For**:
- ✅ Biologically inspired (surprise/novelty/arousal are real)
- ✅ Multi-dimensional (not just one salience score)
- ✅ Memory-aware (novelty requires history)
- ✅ Modality-agnostic (same dimensions for all sensors)

**Arguments Against**:
- ❌ May be overengineered (do we need all 5 dimensions?)
- ❌ Conflict is vague (what exactly is conflict?)
- ❌ Arousal overlaps with surprise (both capture "intensity")
- ❌ Reward is external (not computed from data)

**Alternative**: Simple 3D salience
- **Unexpectedness**: Prediction error (like surprise)
- **Newness**: Distance from memory (like novelty)
- **Intensity**: Signal strength (like arousal)

Simpler, more clearly defined, easier to compute.

---

## Conclusion

**Current State**:
- SNARC concept is excellent (universal salience filter)
- Implementation is misaligned (learned, global, hidden-state based)
- Not being used as intended (per-sensor with spatial structure)

**What SAGE Actually Needs**:
1. **Algorithmic salience** (no training, immediate operation)
2. **Per-sensor instances** (modality-specific salience)
3. **Spatial/temporal structure** (preserve "where" information)
4. **Hierarchical integration** (local → global)
5. **Simple, interpretable** (know what scores mean)

**Recommendation**:
**Rebuild SNARC from first principles** - keep the concept (5D salience), change the implementation (algorithmic, per-sensor, spatial).

The vision from SAGE-SNARC.md is right. The current PyTorch implementation doesn't realize that vision. We need to bridge the gap.

---

**Status**: Analysis complete
**Next Step**: Implement algorithmic per-sensor SNARC
**Priority**: High - this is core to SAGE's attention mechanism

---

*"Take nothing as given. It is useful to the extent it is."*

SNARC is useful. But not in its current form. We need to make it actually serve SAGE's needs.
