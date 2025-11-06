# Puzzle Space Semantics - Universal Sensor Encoding

**Date:** 2025-11-05
**Purpose:** Define how different sensor modalities encode to 30×30×10 puzzle space
**Status:** Design in progress

---

## The Puzzle Space

**Format:** 30×30 spatial grid, 10 discrete values per cell (0-9)
**Total capacity:** 900 cells × 10 values = ~9000 discrete states
**Purpose:** Universal interface where all sensory modalities can be represented geometrically

---

## Why Puzzle Space?

### 1. Universal Interface
All sensors speak different languages:
- Vision: 150K pixels × 3 RGB channels
- Audio: 16K samples/sec × continuous amplitude
- Language: 30K vocab × variable sequence length
- Proprioception: Joint angles × velocities × forces

Puzzle space provides a **common tongue**: 30×30×10 grid everyone can understand.

### 2. Geometric Reasoning
Grids enable spatial reasoning:
- Patterns emerge geometrically
- Transformations visible
- Attention can be spatial (which regions matter?)
- Compression preserves structure

### 3. Bounded Complexity
900 cells × 10 values = manageable state space:
- Not infinite like raw sensors
- Not impoverished like single values
- Goldilocks zone for reasoning

### 4. Biological Parallel
Similar to:
- Visual cortex receptive fields (spatial grids)
- Auditory tonotopic maps (frequency grids)
- Somatosensory homunculus (body surface grids)

---

## Encoding Semantics by Modality

### Vision Encoding

**Input:** Camera frame (e.g., 224×224 RGB)

**Encoding Strategy:**
```
1. Downsample to 30×30 (via avgpool or VAE)
2. Convert RGB to semantic categories:
   0 = background/empty
   1-3 = color families (warm/cool/neutral)
   4-6 = edge orientations (horizontal/vertical/diagonal)
   7-9 = texture patterns (smooth/rough/textured)
```

**Rationale:**
- Preserves spatial structure
- Semantic categories more meaningful than raw RGB
- Attention can focus on regions (e.g., "top-left has strong edges")

**VAE Path:**
```
Image (224×224×3)
  → VAE Encoder → 64D latent
  → Spatial Decoder → 30×30×10 puzzle
```

### Audio Encoding

**Input:** Audio waveform (e.g., 1sec @ 16kHz = 16K samples)

**Encoding Strategy:**
```
1. Spectrogram: Time × Frequency → 30 time bins × 30 freq bins
2. Map amplitude/energy to discrete values:
   0 = silence
   1-3 = low energy (quiet speech, background)
   4-6 = medium energy (normal speech, music)
   7-9 = high energy (loud sounds, emphasis)
```

**Spatial Semantics:**
- X-axis: Time progression (left = past, right = present)
- Y-axis: Frequency bands (bottom = low, top = high)
- Values: Energy levels

**Example:** Human speech shows energy in mid-frequencies (y=10-20) with temporal patterns (x varies)

### Language Encoding

**Input:** Text sequence (variable length)

**Encoding Strategy 1 - Semantic Attention Map:**
```
1. Encode sentence with LLM → attention weights [seq_len, seq_len]
2. Downsample to 30×30 attention heatmap
3. Map attention strength:
   0 = no relation
   1-3 = weak semantic link
   4-6 = moderate connection
   7-9 = strong dependency
```

**Encoding Strategy 2 - Token Grid:**
```
1. Tokenize to max 900 tokens
2. Arrange in 30×30 grid (reading order)
3. Map tokens to categories:
   0 = padding/empty
   1-3 = function words (the, and, is)
   4-6 = content words (noun, verb)
   7-9 = critical tokens (entity, action, negation)
```

**Spatial Semantics:**
- Reading order: left-to-right, top-to-bottom
- Attention map: shows which words relate to which
- Critical regions: where key information concentrates

### Proprioception Encoding

**Input:** Robot joint states (e.g., 7 DOF arm: angles, velocities, forces)

**Encoding Strategy - Body Grid:**
```
1. Map body parts to spatial regions:
   - Rows 0-10: Left arm/hand
   - Rows 10-20: Torso/head
   - Rows 20-30: Right arm/hand
2. Columns represent properties:
   - Cols 0-10: Joint angles (normalized)
   - Cols 10-20: Joint velocities
   - Cols 20-30: Applied forces
3. Map values:
   0 = minimum/rest position
   5 = neutral/center
   9 = maximum/limit
```

**Spatial Semantics:**
- Body layout mirrors physical structure
- Can see whole-body pose at a glance
- Temporal changes visible across cycles

### Memory Encoding

**Input:** Retrieved memory (64D latent or structured data)

**Encoding Strategy - Salience Map:**
```
1. If visual memory: Reconstruct 30×30 from latent
2. If abstract memory: Map to concept grid
   - Different regions = different concepts
   - Value = activation strength
3. SNARC dimensions as overlays:
   0 = not salient
   1-3 = low salience (novelty/arousal)
   4-6 = medium salience (surprise/reward)
   7-9 = high salience (conflict/crisis)
```

---

## Cross-Modal Fusion

When multiple sensors active simultaneously:

### Approach 1: Stacked Channels
```
Puzzle[0]  = Vision encoding
Puzzle[1]  = Audio encoding
Puzzle[2]  = Language encoding
→ 30×30×30 tensor (3 modalities × 10 values)
```

### Approach 2: Attention-Weighted Merge
```
For each cell (x, y):
  vision_val = vision_puzzle[x, y]
  audio_val = audio_puzzle[x, y]

  # Weight by salience
  final_val = argmax(
    snarc.visual_salience * vision_val,
    snarc.audio_salience * audio_val
  )
```

### Approach 3: Spatial Partitioning
```
Grid regions allocated to modalities:
  [0:15, 0:15] = Vision (top-left)
  [0:15, 15:30] = Audio (top-right)
  [15:30, 0:15] = Language (bottom-left)
  [15:30, 15:30] = Proprioception (bottom-right)
```

---

## Semantic Value Meanings

### General Semantics (applies to most modalities)

```
0 = Empty/Background/Absence
1 = Very Low (barely present)
2 = Low (present but weak)
3 = Low-Medium
4 = Medium-Low
5 = Medium (neutral/center)
6 = Medium-High
7 = High (strong presence)
8 = Very High (very strong)
9 = Maximum (saturated/limit)
```

### Context-Specific Meanings

**Vision:**
- 0: Background
- 1-3: Subtle features (shadows, distant objects)
- 4-6: Normal features (edges, textures)
- 7-9: Salient features (bright colors, sharp edges)

**Audio:**
- 0: Silence
- 1-3: Background noise
- 4-6: Normal speech/music
- 7-9: Loud sounds, emphasis

**Language:**
- 0: Padding/no token
- 1-3: Function words
- 4-6: Content words
- 7-9: Critical tokens (entities, actions, negations)

**Proprioception:**
- 0: Minimum position/force
- 5: Neutral/center position
- 9: Maximum position/force/limit

---

## The VAE Bridge

**Purpose:** Learn smooth mapping from sensor space → puzzle space

```
Sensor Data
    ↓
VAE Encoder → 64D latent (continuous)
    ↓
Latent Quantizer → 10 discrete codes per position
    ↓
Spatial Decoder → 30×30×10 puzzle grid
```

**Training Objectives:**
1. **Reconstruction:** Puzzle → decode → sensor approximation
2. **Semantic Preservation:** Similar sensors → similar puzzles
3. **Compression:** Maximize information per cell
4. **Discretization:** Smooth gradients despite discrete values

**VQ-VAE Approach:**
```python
class SensorToPuzzleVAE(nn.Module):
    def __init__(self):
        self.encoder = SensorEncoder()  # sensor → 64D
        self.codebook = nn.Embedding(10, 64)  # 10 codes
        self.spatial_decoder = SpatialDecoder()  # 64D → 30×30 assignments

    def forward(self, sensor_data):
        # Encode
        latent = self.encoder(sensor_data)  # [batch, 64]

        # Vector quantize
        distances = torch.cdist(latent, self.codebook.weight)
        codes = distances.argmin(dim=-1)  # [batch] → 0-9

        # Spatial decode
        spatial_features = self.spatial_decoder(latent)  # [batch, 30, 30, 64]

        # Assign codes spatially
        distances_spatial = torch.cdist(
            spatial_features.reshape(-1, 64),
            self.codebook.weight
        )
        puzzle = distances_spatial.argmin(dim=-1).reshape(batch, 30, 30)

        return puzzle  # [batch, 30, 30] with values 0-9
```

---

## Integration with Unified SAGE Loop

```python
class UnifiedSAGESystem:
    def _cycle(self):
        # 1. Gather raw sensors
        observations = {
            'vision': camera.capture(),      # 224×224×3
            'audio': mic.sample(),            # 16K samples
            'language': speech_to_text(),     # text
            'proprioception': joints.read()   # 7 DOF state
        }

        # 2. Encode to puzzle space
        puzzles = {}
        for sensor_id, data in observations.items():
            vae = self.sensor_vaes[sensor_id]
            puzzle = vae.encode_to_puzzle(data)  # → 30×30
            puzzles[sensor_id] = puzzle

        # 3. SNARC on puzzle space
        salience_report = self.snarc.assess_puzzles(puzzles)

        # 4. SAGECore reasoning on puzzle
        focus_puzzle = puzzles[salience_report.focus_target]
        h_output, strategy = self.sage_core.h_module(focus_puzzle)

        # 5. Plugin execution (IRP operates on puzzle representations)
        results = self.orchestrator.execute(focus_puzzle, strategy)

        # 6. Decode puzzle results to effector space
        if 'response_puzzle' in results:
            response = self.puzzle_decoder.to_speech(results['response_puzzle'])
            self.effectors['tts'].execute(response)
```

---

## Open Questions

### 1. Optimal Discretization
Is 10 values enough? Should it be:
- 10 (0-9) - simple, manageable
- 16 (0-F hex) - more expressive
- 256 (0-255) - continuous-like

**Hypothesis:** 10 values sufficient because:
- Matches human digit span (7±2)
- Enables easy geometric reasoning
- Compression forces semantic meaning

**Test:** Train puzzle VAEs with different value counts, measure reconstruction quality vs reasoning performance.

### 2. Static vs Dynamic Grids
Should puzzles be:
- Static snapshots (current state)
- Temporal sequences (30 frames × 30×30)
- Differential (changes from previous)

**Hypothesis:** Start static, add temporal when needed for motion/speech.

### 3. Semantic Stability
Do encoded semantics remain consistent across:
- Different lighting conditions (vision)
- Different speakers (audio/language)
- Different poses (proprioception)

**Test:** Encode same semantic content under varying conditions, measure puzzle similarity.

### 4. Cross-Modal Resonance
Can system discover cross-modal patterns?
Example: Spoken word "red" → visual red color → both encode to similar puzzle regions?

**Test:** Train with cross-modal pairs, check if puzzles cluster semantically.

---

## Implementation Plan

### Week 2 (Current)
1. ✅ Define puzzle space semantics (this document)
2. ⏳ Implement vision → puzzle VAE
3. ⏳ Test encoding/decoding quality
4. ⏳ Integrate with UnifiedSAGESystem

### Week 3
5. Implement audio → puzzle VAE
6. Implement language → puzzle transformer
7. Multi-modal fusion testing
8. Real-world sensor validation

### Week 4
9. Proprioception → puzzle encoding
10. Cross-device puzzle transfer
11. Puzzle-based reasoning evaluation
12. Performance optimization

---

## Success Criteria

**Puzzle space encoding successful when:**
1. ✅ Semantics clearly defined for each modality
2. ⏳ VAE reconstructs sensor from puzzle (>80% quality)
3. ⏳ Similar sensors → similar puzzles (>0.7 cosine sim)
4. ⏳ SNARC operates effectively on puzzles
5. ⏳ SAGECore reasons about puzzle patterns
6. ⏳ Multi-modal fusion preserves information
7. ⏳ Real-time encoding (<10ms/frame)

---

## The Meta-Pattern

Puzzle space is **geometric meaning**:
- Not raw data (too noisy)
- Not pure symbols (too abstract)
- But structured representation where:
  - Position carries meaning (spatial/temporal)
  - Values carry meaning (intensity/category)
  - Patterns carry meaning (structure/relation)

This mirrors biological sensing:
- Retina → spatial grid of photoreceptors
- Cochlea → tonotopic frequency map
- Somatosensory → body surface map

**Universal interface emerges from geometric organization.**

---

**Status:** Semantics defined, VAE implementation next
**Next:** Implement vision → puzzle VAE with VQ encoding
**Timeline:** Week 2 objectives

The consciousness loop doesn't just process signals - it organizes them geometrically.
