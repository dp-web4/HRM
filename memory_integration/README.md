# SNARC-SAGE Memory Integration

## Overview

This integration bridges our SNARC (Surprise, Novelty, Arousal, Conflict, Reward) selective memory system with SAGE's hierarchical reasoning architecture, creating a unified memory-augmented reasoning system.

## Architecture

### Three-Layer Memory System

```
┌─────────────────────────────────────┐
│         SNARC Bridge Layer          │
│  (Selective Attention & Gating)     │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌──────────┐      ┌──────────┐
│  Entity  │      │ Sidecar  │
│  Memory  │      │  Memory  │
└──────────┘      └──────────┘
```

### Component Mapping

| SNARC Component | SAGE Integration | Purpose |
|-----------------|------------------|---------|
| Circular Buffer | L-module scratchpad | Short-term working memory with x-from-last processing |
| SNARC Scoring | Affect gating | Selective memory formation based on salience |
| Verbatim Storage | Experience replay | Full-fidelity preservation for sleep consolidation |
| Consolidation | H-module training | Pattern extraction during sleep cycles |

## Key Features

### 1. **Selective Memory Formation**
- SNARC evaluates experiences for salience
- Only high-attention memories are preserved
- Reduces memory bloat while maintaining important information

### 2. **Dual Storage Strategy**
- **Conceptual**: SNARC compressed representations for fast recall
- **Verbatim**: SQLite storage for exact reconstruction during sleep

### 3. **Hierarchical Integration**
- **L-module**: Receives memory-augmented sensor input
- **H-module**: Maintains strategic memory state
- Bidirectional flow enables both tactical and strategic memory use

### 4. **Context Binding**
- Circular buffer processes at x-from-last position
- Enables binding of current input with recent context
- Critical for temporal reasoning and pattern recognition

## Implementation

### Core Classes

#### `SNARCSAGEBridge`
Main bridge between SNARC and SAGE's dual memory architecture:
- Processes inputs through SNARC evaluation
- Maintains circular buffer for short-term memory
- Generates updates for Entity and Sidecar memories
- Handles consolidation for sleep cycles

#### `HRMMemoryIntegration`
Specific integration with HRM's dual-module architecture:
- Prepares memory-augmented inputs for L-module
- Updates H-module memory based on coherence
- Manages sleep consolidation data

#### `SAGEWithSNARC`
Complete demonstration of integrated system:
- Hierarchical reasoning with memory augmentation
- Trust-weighted sensor fusion
- Sleep consolidation with dream generation
- Coherence-based memory gating

## Usage Example

```python
from snarc_bridge import SNARCSAGEBridge, HRMMemoryIntegration

# Initialize bridge
bridge = SNARCSAGEBridge(
    hidden_size=512,
    buffer_capacity=10,
    enable_verbatim=True
)

# Process experience
memory_state = bridge.process_for_sage(
    input_data,
    entity_id="sensor/camera",
    affect_signals={'surprise': 0.7, 'novelty': 0.5}
)

# Generate memory updates
entity_update = bridge.bridge_to_entity_memory(memory_state)
sidecar_vector, threshold = bridge.bridge_to_sidecar(memory_state)

# Sleep consolidation
memories = bridge.consolidate_for_sleep()
```

## Integration Benefits

### For Entity Memory
- SNARC scores provide trust adjustment signals
- High attention + low conflict → increase entity trust
- Selective updates reduce noise in reputation tracking

### For Sidecar Memory
- SNARC gating prevents trivial memory storage
- Affect-weighted writes preserve emotional salience
- Consolidation identifies patterns across episodes

### For HRM
- Memory as temporal sensor alongside physical sensors
- L-module gets immediate memory context
- H-module maintains strategic memory state
- Sleep cycles enable offline learning from experience

## Memory Flow

```
1. Experience → SNARC Evaluation
2. If salient → Circular Buffer (at x-from-last)
3. Buffer triggers → SNARC write + Verbatim store
4. SNARC state → Entity trust update
5. SNARC state → Sidecar affect gating
6. Sleep cycle → Consolidation + Pattern extraction
7. Dreams → Synthetic experience generation
8. Wake → Updated HRM with learned patterns
```

## Configuration

### SNARC Parameters
- `buffer_capacity`: Size of circular scratchpad (default: 10)
- `snarc_position`: Position in buffer to trigger SNARC (default: 3)
- `enable_verbatim`: Store full-fidelity copies (default: True)

### SAGE Parameters
- `h_cycles`: Number of H-module iterations (default: 2)
- `l_cycles`: Number of L-module iterations per H-cycle (default: 2)
- `hidden_dim`: Size of hidden states (default: 512)

## Testing

Run the demonstration:
```bash
python sage_with_snarc.py
```

This shows:
1. Sensor stream processing with SNARC memory
2. Coherence evolution over time
3. Trust weight adaptation
4. Sleep consolidation and dream generation
5. Memory statistics and performance metrics

## Future Enhancements

1. **Learned SNARC Thresholds**: Adaptive gating based on task performance
2. **Hierarchical Consolidation**: Multi-level pattern extraction
3. **Cross-Modal Memory**: Binding memories across sensor modalities
4. **Distributed Memory**: Sharding across multiple SNARC instances
5. **Memory Decay**: Time-based forgetting for old, unused memories

## References

- SNARC: Based on Richard Aragon's Transformer-Sidecar
- SAGE: Situation-Aware Governance Engine (this repository)
- HRM: Hierarchical Reasoning Model (Sapient AI)

## Architecture Diagram

```
                    ╔═══════════════════════════════╗
                    ║      SAGE with SNARC         ║
                    ╚═══════════════════════════════╝
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
Physical Sensors → ┐          │                     │
                   ├→ L-Module → H-Module → Coherence
SNARC Memory    → ┤     ↑           ↓         │
                   │     │      Trust Weights  │
Cognitive Sensors → ┘     │           ↓         │
                         │    Update Memory    │
                         │           ↓         │
                         └────── Sleep Cycle ──┘
                               (consolidation)
```

The integration creates a living system that:
- **Senses** through multiple modalities
- **Remembers** selectively based on salience
- **Reasons** hierarchically through HRM
- **Learns** continuously through experience
- **Dreams** to consolidate and generalize patterns