# SNARC System Service

**SNARC** = "Sensor of Sensors" - Salience Assessment and Resource Coordination

A system service that observes the entire sensor field and computes **5D salience** to recommend attention allocation to the SAGE kernel.

## Overview

SNARC acts as the **attention recommendation system** for SAGE. While SAGE decides strategic goals and resource allocation, SNARC provides tactical guidance on *what deserves attention right now*.

### The Core Insight

Intelligence requires **selective attention**. You can't process everything equally - you must focus on what matters. SNARC computes "what matters" across 5 dimensions:

1. **Surprise** - Deviation from prediction
2. **Novelty** - Difference from past experiences
3. **Arousal** - Signal intensity/urgency
4. **Reward** - Goal relevance
5. **Conflict** - Cross-sensor disagreement

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SAGE Kernel                            │
│  "I need to know what deserves attention"                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SNARCService                              │
│  assess_salience(sensor_outputs) → SalienceReport          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  5D Detectors (parallel computation)                 │  │
│  │                                                       │  │
│  │  SurpriseDetector   ─┐                              │  │
│  │  NoveltyDetector    ─┤                              │  │
│  │  ArousalDetector    ─┼─→  Weighted → Total Salience │  │
│  │  RewardEstimator    ─┤        ↓                     │  │
│  │  ConflictDetector   ─┘    Suggest Stance            │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  SalienceReport                             │
│  focus_target: "which sensor to attend to"                 │
│  salience_score: 0.85                                      │
│  suggested_stance: CURIOUS_UNCERTAINTY                     │
│  confidence: 0.92                                          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from sage.services.snarc import SNARCService

# Initialize
snarc = SNARCService()

# Assess salience across sensor field
sensor_outputs = {
    'vision': camera.capture(),
    'audio': microphone.sample(),
    'imu': imu.read(),
    'touch': touch_sensors.read()
}

report = snarc.assess_salience(sensor_outputs)

# Use report for attention allocation
print(f"Focus on: {report.focus_target}")
print(f"Salience: {report.salience_score}")
print(f"Suggested stance: {report.suggested_stance}")

# Breakdown by dimension
breakdown = report.salience_breakdown
print(f"Surprise: {breakdown.surprise}")
print(f"Novelty: {breakdown.novelty}")
print(f"Arousal: {breakdown.arousal}")
print(f"Reward: {breakdown.reward}")
print(f"Conflict: {breakdown.conflict}")
```

### Learning from Outcomes

```python
from sage.services.snarc import Outcome

# Execute action based on salience
result = take_action(report.focus_target, report.suggested_stance)

# Provide feedback to SNARC
outcome = Outcome(
    success=result.success,
    reward=result.reward,  # -1.0 to 1.0
    description="Action completed successfully"
)

snarc.update_from_outcome(report, outcome)
# SNARC will adjust its salience weights based on outcomes
```

## The 5 Dimensions Explained

### 1. Surprise (Prediction Error)

**What it detects:** Deviation from expected values
**Computation:** EMA predictor + percentile-normalized error
**High surprise means:** "This wasn't supposed to happen"

```python
# Example: Stable temperature suddenly spikes
temps = [20.1, 20.0, 19.9, 20.2, 20.0, 35.7]  # ← High surprise!
```

**Use case:** Anomaly detection, safety monitoring

---

### 2. Novelty (Memory Comparison)

**What it detects:** Difference from past experiences
**Computation:** Cosine similarity to episodic memory
**High novelty means:** "I've never seen this before"

```python
# Example: Familiar vs novel pattern
familiar = [1, 2, 1, 2, 1, 2, 1, 2]  # ← Low novelty
novel = [7, 3, 9, 1, 4, 8, 2, 6]     # ← High novelty
```

**Use case:** Exploration, learning opportunities

---

### 3. Arousal (Signal Magnitude)

**What it detects:** Intensity/urgency of signals
**Computation:** L2 norm, percentile-normalized
**High arousal means:** "This is loud/big/urgent"

```python
# Example: Quiet vs loud audio
quiet = [0.1, -0.1, 0.2, -0.1]      # ← Low arousal
loud = [8.5, -7.2, 9.1, -8.7]       # ← High arousal
```

**Use case:** Urgency assessment, crisis detection

---

### 4. Reward (Goal Relevance)

**What it detects:** Association with positive outcomes
**Computation:** Similarity-based outcome retrieval
**High reward means:** "This pattern leads to success"

```python
# Example: Learning goal-relevant patterns
snarc.update_from_outcome(sensor_output, sensor_id, reward=0.9)
# Now similar patterns will have high reward scores
```

**Use case:** Goal pursuit, value learning

---

### 5. Conflict (Cross-Sensor Disagreement)

**What it detects:** Inconsistency across modalities
**Computation:** Correlation deviation from expectations
**High conflict means:** "My sensors don't agree"

```python
# Example: Vision says "moving" but audio says "silent"
vision_motion = [1, 5, 8, 3, 7]   # ← High activity
audio_level = [0, 0, 0, 0, 0]     # ← Silent
# High conflict! Suggests verification needed
```

**Use case:** Safety verification, sensor validation

## Cognitive Stance Suggestion

SNARC recommends cognitive stances based on salience patterns:

### SKEPTICAL_VERIFICATION
**Pattern:** High conflict (>0.7)
**Meaning:** "Don't trust this - verify first"
**Example:** Vision sensor shows motion but other sensors silent

### CURIOUS_UNCERTAINTY
**Pattern:** High surprise + novelty
**Meaning:** "Explore and learn"
**Example:** Never-before-seen pattern detected

### EXPLORATORY
**Pattern:** High arousal + moderate novelty
**Meaning:** "Investigate actively"
**Example:** Intense but partially familiar stimulus

### FOCUSED_ATTENTION
**Pattern:** High reward
**Meaning:** "Pursue goal aggressively"
**Example:** Goal-relevant pattern detected

### CONFIDENT_EXECUTION
**Pattern:** Low surprise + low novelty
**Meaning:** "Routine operation, execute efficiently"
**Example:** Predictable, familiar environment

## Data Structures

### SalienceReport

```python
@dataclass
class SalienceReport:
    focus_target: str                      # Which sensor to attend to
    salience_score: float                  # Overall importance (0-1)
    salience_breakdown: SalienceBreakdown  # 5D breakdown
    suggested_stance: CognitiveStance      # Recommended approach
    relevant_memories: List[SNARCMemory]   # Similar past patterns
    confidence: float                      # Assessment confidence (0-1)
    metadata: Dict[str, Any]               # Additional context
```

### SalienceBreakdown

```python
@dataclass
class SalienceBreakdown:
    surprise: float   # 0.0-1.0
    novelty: float    # 0.0-1.0
    arousal: float    # 0.0-1.0
    reward: float     # 0.0-1.0
    conflict: float   # 0.0-1.0

    def total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted total salience"""
        ...
```

### Outcome

```python
@dataclass
class Outcome:
    success: bool
    reward: float  # -1.0 to 1.0
    description: str
```

## Advanced Features

### Custom Salience Weights

```python
# Emphasize reward and conflict
custom_weights = {
    'surprise': 0.15,
    'novelty': 0.15,
    'arousal': 0.15,
    'reward': 0.30,   # ← Double weight
    'conflict': 0.25  # ← Higher weight
}

snarc = SNARCService(salience_weights=custom_weights)
```

### Adaptive Learning

SNARC adapts its weights based on outcomes:

```python
# If high-conflict situations lead to bad outcomes,
# conflict weight will increase (safety)

# If high-reward predictions are accurate,
# reward weight will increase (goal focus)
```

### Memory Retrieval

```python
# SNARC retrieves similar past situations
for memory in report.relevant_memories:
    print(f"Similar to {memory.timestamp}")
    print(f"Past outcome: {memory.outcome}")
    print(f"Was useful: {memory.was_useful()}")
```

### Statistics Monitoring

```python
stats = snarc.get_statistics()
print(f"Assessments: {stats['num_assessments']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Current weights: {stats['current_weights']}")
```

## Testing

### Run Tests

```bash
# All SNARC tests
pytest sage/services/snarc/tests/ -v

# Individual detector tests
pytest sage/services/snarc/tests/test_detectors.py -v

# Integration tests
pytest sage/services/snarc/tests/test_snarc_service.py -v
```

### Test Coverage

- ✅ 23 detector unit tests (all 5 dimensions)
- ✅ 21 integration tests (complete system)
- ✅ Scenario testing (calm, surprising, novel, conflicting, goal-relevant)
- ✅ Stance suggestion validation
- ✅ Confidence computation validation

## Implementation Details

### Detector Algorithms

**SurpriseDetector**: Exponential Moving Average (EMA) prediction + percentile normalization
**NoveltyDetector**: Episodic memory (circular buffer) + cosine similarity
**ArousalDetector**: L2 norm + historical distribution normalization
**RewardEstimator**: Similarity-based k-NN retrieval from outcome memory
**ConflictDetector**: Pearson correlation + deviation from expectations

### Performance

- **Detectors**: O(k) where k = comparison samples (default 20-50)
- **Memory**: O(n) where n = history size (default 100-1000)
- **Assessment**: ~1ms for 4 sensors (tested on laptop CPU)

### Supported Data Types

- Scalars: `float`, `int`
- Arrays: `numpy.ndarray`
- Tensors: `torch.Tensor`
- Mixed: Different sensors can use different types

## Integration with SAGE Kernel

```python
from sage.core import SAGEKernel
from sage.services.snarc import SNARCService

# SAGE uses SNARC internally
sage = SAGEKernel(
    sensor_sources={'camera': camera.capture, 'mic': mic.sample},
    action_handlers={'camera': handle_vision, 'mic': handle_audio}
)

# Main loop:
# while True:
#     observations = gather_sensors()
#     report = snarc.assess_salience(observations)  # ← SNARC
#     execute_action(report.focus_target, report.stance)
#     update_from_outcome(report, outcome)
```

## Design Philosophy

### Why 5 Dimensions?

Based on neuroscience research on biological salience:
- **Surprise**: Prediction error (LC-NE system)
- **Novelty**: Memory comparison (hippocampus)
- **Arousal**: Signal intensity (reticular formation)
- **Reward**: Value learning (VTA-dopamine)
- **Conflict**: Error monitoring (ACC)

### Why "Sensor of Sensors"?

SNARC doesn't process raw data - it observes *other sensors*. It's meta-level attention:

```
Raw sensors: "Here's what I see/hear/feel"
SNARC: "Here's which of those deserves attention"
SAGE: "Here's how to allocate resources"
```

### Biological Parallel

SNARC implements the same attention mechanism as:
- **Thalamus**: Sensory relay and filtering
- **Superior colliculus**: Salience map for attention
- **LC-NE system**: Arousal and surprise
- **ACC**: Conflict monitoring

Not mimicking - **discovering same optimal solutions**.

## Future Extensions

### Planned Features

- [ ] Multi-timescale integration (fast/slow salience)
- [ ] Contextual modulation (task-dependent weights)
- [ ] Hierarchical salience (sensor groups)
- [ ] Predictive salience (anticipatory attention)
- [ ] Social salience (interaction relevance)

### Research Directions

- Optimal salience weight learning
- Cross-modal binding via salience
- Attention as compression
- Salience-driven memory consolidation

## References

### Papers

- Itti & Koch (2001). "Computational Modelling of Visual Attention"
- Gottlieb et al. (2013). "Information-seeking, curiosity, and attention"
- Parr & Friston (2017). "The active construction of the visual world"

### SAGE Architecture Docs

- `/sage/docs/SAGE_ARCHITECTURE_BREAKTHROUGH.md` - Overall architecture
- `/sage/docs/SYSTEM_UNDERSTANDING.md` - How components fit together

## License

Part of the HRM (Hierarchical Reasoning Model) project.

---

**Built with Claude Code** 🤖
*Demonstrating that attention orchestration, not model size, is the key to intelligence.*
