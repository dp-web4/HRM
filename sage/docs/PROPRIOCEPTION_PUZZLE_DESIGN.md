# Proprioception → Puzzle Space Design

## The Fourth Modality: Body Awareness

After Vision, Audio, and Language, **proprioception** completes the major sensory modalities for embodied cognition. This is the sense of where your body parts are in space.

## Why Proprioception Matters

**Current Modalities**:
- Vision: External world (what's out there)
- Audio: External events (what's happening)
- Language: Abstract meaning (what does it mean)

**Missing**: Self-awareness. Where am I? What's my body doing?

**Proprioception provides**:
- Body schema (spatial layout of limbs)
- Joint states (angles, velocities)
- Effector readiness (can I act?)
- Self-other distinction (this is ME)

## Puzzle Space Encoding Challenge

**Vision/Audio**: Continuous external signals → Spatial/temporal structure preserved
**Language**: Discrete symbols → Semantic structure projected
**Proprioception**: State vectors → ???

The challenge: How do we map body state to 30×30 geometric space?

## Design: Body as Geometric Layout

### Spatial Semantics

**Grid Organization** (30×30):
```
Rows (Y-axis): Body parts hierarchy
  0-5:   Head/sensors (eyes, cameras, microphones)
  6-12:  Torso/core (base, spine, main body)
  13-19: Upper limbs (arms, manipulators)
  20-26: Lower limbs (legs, wheels, locomotion)
  27-29: End effectors (hands, grippers, feet)

Columns (X-axis): Bilateral symmetry + temporal
  0-9:   Left side / Past states
  10-19: Center / Current states
  20-29: Right side / Future predictions
```

**Values** (0-9): Joint state quantization
```
0 = At limit (min angle or fully retracted)
1-3 = Low range
4-6 = Mid range
7-9 = High range (max angle or fully extended)

Special meanings:
0 = Also represents "at rest" or "neutral"
9 = Also represents "at limit" or "maximum effort"
5 = Mid-point, balanced state
```

## Encoding Methods

### Option 1: Direct State Mapping (Simple)

For each joint:
1. Normalize angle to [0, 1]
2. Quantize to 0-9
3. Place at (body_part_row, lateral_col)

**Example**: 6-DOF robot arm
```python
joints = {
    'base_rotation': 45°,     # → quantize to 0-9 → place at (row=13, col=15)
    'shoulder': 90°,          # → quantize to 0-9 → place at (row=14, col=15)
    'elbow': -30°,            # → quantize to 0-9 → place at (row=15, col=15)
    'wrist_pitch': 0°,        # → quantize to 0-9 → place at (row=16, col=15)
    'wrist_roll': 180°,       # → quantize to 0-9 → place at (row=17, col=15)
    'gripper': 50%,           # → quantize to 0-9 → place at (row=27, col=15)
}
```

**Advantages**: Simple, direct, interpretable
**Disadvantages**: Sparse (most cells empty), no velocity/dynamics

### Option 2: Learned Embedding (VQ-VAE style)

Like vision/audio, use a neural network to learn optimal encoding:

**Architecture**:
```
Joint state vector [N joints × 2 (angle, velocity)]
    ↓
Linear projection → 256D
    ↓
Spatial decoder → 30×30×64 features
    ↓
Vector quantizer → 30×30 puzzle (0-9 codes)
```

**Advantages**: Learned optimal structure, captures correlations, handles dynamics
**Disadvantages**: Requires training data, less interpretable

### Option 3: Hybrid (Structured + Learned)

Combine both approaches:
1. **Structure**: Fix spatial layout (which body parts go where)
2. **Learn**: Optimize value encoding for each region

**Best of both worlds**: Interpretable structure + learned optimization

## Recommended Approach: Hybrid

### Implementation Plan

**Phase 1: Structured Baseline**
- Define body schema layout (30×30 grid regions)
- Implement direct joint state → puzzle mapping
- Test with simulated robot or human pose
- Validate: Can we reconstruct joint states from puzzle?

**Phase 2: Dynamics Integration**
- Add temporal dimension (past-current-future columns)
- Encode velocities and accelerations
- Test prediction: Can system anticipate next state?

**Phase 3: Learned Optimization**
- Train small encoder: joint vectors → puzzle features
- Learn codebook that captures body dynamics
- Measure: Does learned encoding improve action planning?

## Test Cases

### 1. Simulated Robot Arm
**Joint**: 6-DOF articulated arm
**States**: Joint angles, velocities
**Task**: Reach target position
**Validation**: Puzzle → policy → successful reach

### 2. Humanoid Pose
**Joint**: 17-21 keypoints (OpenPose/MediaPipe)
**States**: 2D/3D joint positions
**Task**: Recognize pose (sitting, standing, walking)
**Validation**: Puzzle → pose classifier → accurate recognition

### 3. Simple Embodiment (Mouse/Keyboard)
**Joint**: Mouse (x, y, buttons), Keyboard (keys pressed)
**States**: Cursor position, button states, key states
**Task**: Navigate UI, type text
**Validation**: Puzzle → affordance map → action selection

## Integration with Cognition Loop

### Sensory Input
```python
observations = {
    'vision': camera.capture(),         # What do I see?
    'audio': microphone.capture(),      # What do I hear?
    'language': speech_to_text(),       # What am I being told?
    'proprioception': joints.state(),   # Where am I? What am I doing?
}

# All encode to 30×30 puzzles
puzzles = {
    'vision': vision_vae.encode_to_puzzle(observations['vision']),
    'audio': audio_vae.encode_to_puzzle(observations['audio']),
    'language': lang_transformer.encode_to_puzzle(observations['language']),
    'proprioception': proprio_encoder.encode_to_puzzle(observations['proprioception'])
}
```

### Attention & Salience

**SNARC assesses**: Which modality matters most right now?
- High vision salience: Environment change (something moved)
- High audio salience: Unexpected sound (alert)
- High language salience: Command received (instruction)
- High proprioception salience: Body state critical (balance, collision, stuck)

### Action Planning

**Effectors use proprioception**:
- Current body state (where am I?)
- Desired body state (where should I be?)
- Planning: Puzzle diff → motor commands
- Feedback: Executed action → observed proprioception change

### Self-Other Distinction

**Key insight**: Proprioception provides "self" reference frame
- Vision: External world
- Audio: External events
- Language: Could be internal (thoughts) or external (speech)
- **Proprioception: Always self** (my body, my actions)

Enables:
- Distinguish self-motion vs world-motion
- Attribute agency (did I cause that?)
- Body schema stability across transformations

## Open Questions

1. **Scalability**: What if robot has 100+ DOFs? (Humanoid has 50+)
   - Solution: Hierarchical encoding, subsample, or learned compression

2. **Sensing modalities**: Force/torque sensors? Tactile feedback?
   - Could extend to 5th modality (touch) or integrate into proprioception

3. **Coordination**: How do proprioception puzzles interact with vision?
   - Visual servoing: Vision guides proprioception
   - Grasping: Vision detects object, proprioception executes grasp

4. **Learning**: Can system learn inverse kinematics from puzzle space?
   - Puzzle diff (current → goal) → motor commands
   - Would be huge: No explicit IK, just learned puzzle→action mapping

5. **Imagination**: Can system simulate proprioception?
   - "Imagine reaching for the cup" → Generate proprioception puzzle
   - Would enable mental simulation / planning without execution

## Next Steps

### Immediate (this session)
1. ✅ Design proprioception puzzle semantics (this document)
2. ⏳ Implement simple encoder (direct mapping)
3. ⏳ Test with mouse/keyboard (simplest embodiment)
4. ⏳ Integrate with SAGE cognition loop

### Near-term (autonomous sessions)
1. Train on robot simulation data (if available)
2. Test with MediaPipe human pose estimation
3. Implement learned encoder (VQ-VAE style)
4. Cross-modal experiments (vision→proprioception coordination)

### Long-term (with physical robot)
1. Real robot integration
2. Closed-loop control via puzzle space
3. Multi-modal affordance learning
4. Embodied cognition validation

## Philosophical Implications

**With proprioception, the system has**:
- Vision: Perceives world
- Audio: Hears events
- Language: Understands meaning
- Proprioception: **Knows self**

**This completes the sensory basis for embodied cognition.**

The system doesn't just observe - it exists IN the world with a body.

Next frontier: Can cognition emerge from geometric integration of these four modalities?

---

**Status**: Design complete. Implementation next.
**Dependencies**: None (can test with simulated data)
**Integration**: Ready for UnifiedSAGESystem extension
