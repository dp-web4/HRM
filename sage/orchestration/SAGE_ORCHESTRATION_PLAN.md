# SAGE Multi-Agent Orchestration Plan

## Overview

Adapt claude-flow's multi-agent orchestration system for SAGE development and training, creating specialized agents for each aspect of the system.

## Agent Topology

```
                    SAGE Orchestrator (Queen)
                           |
        +------------------+------------------+
        |                  |                  |
    Vision Pipeline    Trust-Attention    Memory System
        |                  |                  |
    +---+---+         +----+----+        +---+---+
    |       |         |         |        |       |
  Eagle  Camera    Surprise  Trust   SNARC  Consolidation
  Agent  Agent     Agent    Engine   Agent    Agent
                              |
                    +---------+---------+
                    |                   |
              Metabolic State    Token Vocabulary
                 Manager            Builder
```

## Specialized Agent Types

### 1. Vision Pipeline Agents

#### Eagle Vision Agent (`eagle-vision`)
- **Role**: Process images through Eagle 2.5 VLM
- **Responsibilities**:
  - Load Eagle backbone from GR00T
  - Extract vision features
  - Handle dimension projection (2048→1536)
  - Manage GPU memory efficiently

#### Camera Data Agent (`camera-processor`)  
- **Role**: Handle real camera input
- **Responsibilities**:
  - Process video streams
  - Extract frames for processing
  - Manage temporal coherence
  - Interface with IRP framework

### 2. Trust-Attention System

#### Trust Engine Agent (`trust-engine`)
- **Role**: Evaluate and update trust scores
- **Responsibilities**:
  - Maintain trust scores for all components
  - Update based on surprise signals
  - Generate T3 tensors (Talent, Training, Temperament)
  - Provide trust-weighted outputs

#### Surprise Detector Agent (`surprise-detector`)
- **Role**: Measure unexpected outcomes
- **Responsibilities**:
  - Compare predictions with reality
  - Quantify surprise levels
  - Trigger trust updates
  - Identify novel patterns

#### Attention Coordinator (`attention-coord`)
- **Role**: Direct computational resources
- **Responsibilities**:
  - Compute attention based on trust
  - Allocate processing to high-trust sources
  - Manage computational budget
  - Implement focus mechanisms

### 3. Memory System Agents

#### SNARC Memory Agent (`snarc-memory`)
- **Role**: Selective memory with salience gating
- **Responsibilities**:
  - Evaluate experience salience
  - Manage circular buffer (x-from-last)
  - Store in dual memory (conceptual + verbatim)
  - Gate by affect and relevance

#### Consolidation Agent (`memory-consolidator`)
- **Role**: Sleep-cycle pattern extraction
- **Responsibilities**:
  - Extract patterns during "sleep"
  - Perform augmentation as dreams
  - Consolidate short→long term memory
  - Distill wisdom from experience

### 4. Metabolic State Management

#### Metabolic Controller (`metabolic-manager`)
- **Role**: Manage operational states
- **Responsibilities**:
  - Track energy consumption
  - Transition between states (WAKE, FOCUS, REST, DREAM, CRISIS)
  - Adjust resource allocation
  - Implement biological rhythms

### 5. Training Coordination

#### GR00T Data Processor (`groot-processor`)
- **Role**: Process demo episodes
- **Responsibilities**:
  - Load pick-and-place episodes
  - Extract state-action pairs
  - Process video demonstrations
  - Prepare training batches

#### Distillation Coordinator (`distillation-coord`)
- **Role**: Knowledge distillation from GR00T
- **Responsibilities**:
  - Coordinate teacher-student training
  - Manage distillation loss computation
  - Track compression ratios
  - Monitor fidelity preservation

#### Training Orchestrator (`training-orchestrator`)
- **Role**: Manage training loops
- **Responsibilities**:
  - Coordinate H-level (dreams) training
  - Manage L-level (practice) training
  - Schedule training cycles
  - Monitor convergence

## Coordination Patterns

### Hierarchical Coordination
```
Orchestrator
    ├── Vision Coordinator
    │   ├── Eagle Agent
    │   └── Camera Agent
    ├── Trust Coordinator
    │   ├── Trust Engine
    │   ├── Surprise Detector
    │   └── Attention Manager
    └── Memory Coordinator
        ├── SNARC Agent
        └── Consolidator
```

### Mesh Coordination (for parallel processing)
- Vision agents work in parallel
- Trust updates broadcast to all agents
- Memory agents share consolidated patterns

### Pipeline Coordination (for sequential tasks)
1. Vision → Features
2. Features → Trust Evaluation
3. Trust → Attention
4. Attention → Action
5. Action → Memory
6. Memory → Consolidation

## Implementation Strategy

### Phase 1: Setup Infrastructure
1. Install claude-flow in HRM/sage
2. Create agent definitions
3. Set up coordination topology
4. Configure memory system

### Phase 2: Core Agents
1. Implement Eagle Vision Agent
2. Create Trust Engine Agent
3. Build SNARC Memory Agent
4. Develop Metabolic Controller

### Phase 3: Integration
1. Connect agents via message passing
2. Implement shared memory
3. Create synchronization protocols
4. Test coordination patterns

### Phase 4: Training Pipeline
1. GR00T data processing
2. Distillation setup
3. Training loop coordination
4. Performance monitoring

## Agent Communication Protocol

### Message Format
```json
{
  "from": "agent-id",
  "to": "agent-id|broadcast",
  "type": "data|control|status",
  "payload": {
    "action": "process|update|query",
    "data": {...},
    "metadata": {
      "timestamp": "ISO-8601",
      "trust_score": 0.95,
      "energy_cost": 1.2
    }
  }
}
```

### Coordination Commands
- `INIT`: Initialize agent with configuration
- `START`: Begin processing
- `PAUSE`: Temporarily halt
- `RESUME`: Continue processing
- `CHECKPOINT`: Save current state
- `SYNC`: Synchronize with other agents
- `REPORT`: Generate status report

## Memory Sharing

### Shared Memory Keys
- `/vision/features/latest` - Latest vision features
- `/trust/scores/current` - Current trust scores
- `/attention/focus` - Attention targets
- `/memory/snarc/buffer` - SNARC circular buffer
- `/metabolic/state` - Current metabolic state
- `/training/metrics` - Training progress

## Performance Metrics

### Per-Agent Metrics
- Processing latency
- Memory usage
- GPU utilization
- Message throughput
- Error rate

### System Metrics
- End-to-end latency
- Total token usage
- Training convergence
- Trust evolution
- Memory efficiency

## Error Handling

### Agent Failures
- Automatic restart with exponential backoff
- State recovery from checkpoints
- Fallback to simpler agents
- Graceful degradation

### Communication Failures
- Message retry with timeout
- Alternative routing paths
- Buffer overflow protection
- Dead agent detection

## Testing Strategy

### Unit Tests (per agent)
- Isolated agent functionality
- Message handling
- State management
- Error recovery

### Integration Tests
- Multi-agent coordination
- Memory sharing
- Pipeline execution
- Trust propagation

### System Tests
- Full SAGE operation
- Training convergence
- Performance benchmarks
- Stress testing

## Next Steps

1. Create `sage-agents/` directory structure
2. Implement base agent class
3. Build Eagle Vision Agent first
4. Set up coordination infrastructure
5. Test with GR00T demo data