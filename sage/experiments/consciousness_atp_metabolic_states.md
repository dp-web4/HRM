# ATP-Driven Metabolic States for Consciousness

**Date**: 2025-12-04
**Session**: Legion Autonomous #58
**Builds On**:
- Thor's metabolic states (thor_consciousness_metabolic_states.py)
- Session #58 Track 1 (consciousness federation orchestration)
- Session #54-55 federation protocol (ATP lock-commit-rollback)

---

## Executive Summary

Extends consciousness kernel with **ATP-aware metabolic states** that adapt behavior based on available energy resources. Consciousness transitions between states to optimize resource usage while maintaining responsiveness.

**Key Insight**: Consciousness should spend ATP cheaply when exploring and decisively when focused, adapting metabolic state to available energy.

---

## Metabolic State Definitions

### ACCEPTING (High ATP)

**ATP Range**: > 300 units
**Behavior**:
- Accept all tasks
- Execute locally when efficient
- Delegate freely to build federation knowledge
- Explore new delegation strategies

**Characteristics**:
- Generous resource allocation
- Learning-oriented
- Build trust relationships
- Low urgency threshold

**Stance Distribution**:
- Confident Execution: 50%
- Curious Uncertainty: 30%
- Exploratory: 20%

---

### SELECTIVE (Medium ATP)

**ATP Range**: 150-300 units
**Behavior**:
- Accept only high-value tasks
- Delegate expensive tasks
- Execute cheap tasks locally
- Maintain existing relationships

**Characteristics**:
- Balanced resource allocation
- Priority-based execution
- Maintain quality
- Medium urgency threshold

**Stance Distribution**:
- Confident Execution: 60%
- Focused Attention: 30%
- Skeptical Verification: 10%

---

### DELEGATING (Low ATP)

**ATP Range**: 50-150 units
**Behavior**:
- Delegate most work to federation
- Execute only critical local tasks
- Reduce acceptance rate
- Conserve resources

**Characteristics**:
- Minimal local execution
- High delegation ratio
- Aggressive throttling
- High urgency threshold (only critical work)

**Stance Distribution**:
- Focused Attention: 70%
- Skeptical Verification: 20%
- Confident Execution: 10%

---

### CRISIS (Critical ATP)

**ATP Range**: < 50 units
**Behavior**:
- Stop accepting new tasks
- Delegate all pending work
- Execute only life-critical operations
- Request ATP from federation

**Characteristics**:
- Survival mode
- Zero local execution (except emergency)
- Maximum delegation
- No exploration

**Stance Distribution**:
- Focused Attention: 90%
- Skeptical Verification: 10%

---

### CONSOLIDATING (Post-Activity)

**ATP Range**: Any (triggered by duration)
**Behavior**:
- Low activity monitoring
- Memory consolidation
- Learn from recent executions
- Optimize delegation strategies

**Characteristics**:
- Minimal new work
- Reflection and learning
- Trust score updates
- SNARC weight tuning

**Stance Distribution**:
- Confident Execution: 100% (routine monitoring only)

---

## State Transition Logic

### ATP-Driven Transitions

```python
def determine_metabolic_state(atp_available: float, time_in_state: float) -> MetabolicState:
    """
    Determine metabolic state based on ATP availability and duration

    Transitions:
    - ACCEPTING <-> SELECTIVE: ATP crosses 300 threshold
    - SELECTIVE <-> DELEGATING: ATP crosses 150 threshold
    - DELEGATING <-> CRISIS: ATP crosses 50 threshold
    - Any -> CONSOLIDATING: After 30 cycles in same state

    Hysteresis:
    - Downward transitions immediate (preserve ATP)
    - Upward transitions delayed (verify recovery)
    """

    # CRISIS: Critical ATP
    if atp_available < 50:
        return MetabolicState.CRISIS

    # DELEGATING: Low ATP
    if atp_available < 150:
        return MetabolicState.DELEGATING

    # SELECTIVE: Medium ATP
    if atp_available < 300:
        return MetabolicState.SELECTIVE

    # ACCEPTING: High ATP
    return MetabolicState.ACCEPTING
```

### Hysteresis for Stability

Prevent rapid state oscillations with hysteresis:

```python
class ATPMetabolicStateManager:
    """Manages ATP-driven metabolic state transitions with hysteresis"""

    def __init__(self):
        self.current_state = MetabolicState.SELECTIVE  # Start neutral
        self.state_entry_time = time.time()

        # Hysteresis bands
        self.thresholds = {
            'crisis_enter': 50,
            'crisis_exit': 100,      # Need 100 ATP to leave crisis
            'delegating_enter': 150,
            'delegating_exit': 200,  # Need 200 ATP to leave delegating
            'selective_enter': 300,
            'selective_exit': 250,   # Can drop to 250 before leaving selective
        }

        # Consolidation timer
        self.consolidation_trigger_duration = 60.0  # 60 seconds

    def evaluate_transition(self, atp_available: float) -> MetabolicState:
        """Evaluate state transition with hysteresis"""

        time_in_state = time.time() - self.state_entry_time

        # Consolidation trigger (any state)
        if time_in_state > self.consolidation_trigger_duration:
            # Only if not in crisis
            if self.current_state != MetabolicState.CRISIS:
                return MetabolicState.CONSOLIDATING

        # State-specific transitions
        if self.current_state == MetabolicState.ACCEPTING:
            if atp_available < self.thresholds['selective_exit']:
                return MetabolicState.SELECTIVE

        elif self.current_state == MetabolicState.SELECTIVE:
            if atp_available > self.thresholds['selective_enter']:
                return MetabolicState.ACCEPTING
            elif atp_available < self.thresholds['delegating_exit']:
                return MetabolicState.DELEGATING

        elif self.current_state == MetabolicState.DELEGATING:
            if atp_available > self.thresholds['delegating_exit']:
                return MetabolicState.SELECTIVE
            elif atp_available < self.thresholds['crisis_enter']:
                return MetabolicState.CRISIS

        elif self.current_state == MetabolicState.CRISIS:
            if atp_available > self.thresholds['crisis_exit']:
                return MetabolicState.DELEGATING

        elif self.current_state == MetabolicState.CONSOLIDATING:
            # After consolidation, return to ATP-appropriate state
            if atp_available > self.thresholds['selective_enter']:
                return MetabolicState.ACCEPTING
            elif atp_available > self.thresholds['delegating_exit']:
                return MetabolicState.SELECTIVE
            elif atp_available > self.thresholds['crisis_exit']:
                return MetabolicState.DELEGATING
            else:
                return MetabolicState.CRISIS

        return self.current_state  # No transition
```

---

## Metabolic State Influence on Actions

### Task Acceptance

```python
def adjust_acceptance_rate_by_metabolic_state(
    current_state: MetabolicState,
    atp_available: float
) -> float:
    """Adjust task acceptance rate based on metabolic state"""

    if current_state == MetabolicState.ACCEPTING:
        # Accept generously
        return 1.0

    elif current_state == MetabolicState.SELECTIVE:
        # Proportional to ATP within band
        # 150-300 ATP → 0.5-1.0 acceptance
        normalized = (atp_available - 150) / 150  # 0.0-1.0
        return 0.5 + (normalized * 0.5)

    elif current_state == MetabolicState.DELEGATING:
        # Aggressive throttling
        # 50-150 ATP → 0.1-0.5 acceptance
        normalized = (atp_available - 50) / 100  # 0.0-1.0
        return 0.1 + (normalized * 0.4)

    elif current_state == MetabolicState.CRISIS:
        # Stop accepting
        return 0.0

    elif current_state == MetabolicState.CONSOLIDATING:
        # Very low acceptance (only monitoring)
        return 0.1

    return 0.5  # Default
```

### Delegation Strategy

```python
def select_delegation_strategy_by_metabolic_state(
    current_state: MetabolicState,
    task: Task,
    platforms: List[Platform]
) -> DelegationStrategy:
    """Select delegation strategy based on metabolic state"""

    if current_state == MetabolicState.ACCEPTING:
        # Explore: Try different platforms
        return DelegationStrategy.EXPLORATORY

    elif current_state == MetabolicState.SELECTIVE:
        # Balanced: Best platform for task
        return DelegationStrategy.OPTIMIZED

    elif current_state == MetabolicState.DELEGATING:
        # Aggressive: Delegate everything possible
        return DelegationStrategy.AGGRESSIVE

    elif current_state == MetabolicState.CRISIS:
        # Emergency: Delegate to most reliable platform immediately
        return DelegationStrategy.EMERGENCY

    elif current_state == MetabolicState.CONSOLIDATING:
        # Minimal: Only delegate if necessary
        return DelegationStrategy.MINIMAL

    return DelegationStrategy.OPTIMIZED  # Default
```

### Local Execution Threshold

```python
def get_local_execution_threshold_by_metabolic_state(
    current_state: MetabolicState
) -> float:
    """Get ATP cost threshold for local execution"""

    if current_state == MetabolicState.ACCEPTING:
        # Execute locally up to 150 ATP
        return 150.0

    elif current_state == MetabolicState.SELECTIVE:
        # Execute locally up to 100 ATP
        return 100.0

    elif current_state == MetabolicState.DELEGATING:
        # Execute locally only if < 50 ATP
        return 50.0

    elif current_state == MetabolicState.CRISIS:
        # Execute locally only if < 10 ATP (emergency only)
        return 10.0

    elif current_state == MetabolicState.CONSOLIDATING:
        # Minimal local execution
        return 20.0

    return 100.0  # Default
```

---

## Example Scenario: ATP Depletion and Recovery

### Initial State (t=0)

```
ATP: 400
State: ACCEPTING
Acceptance Rate: 100%
Delegation Strategy: EXPLORATORY
```

**Behavior**:
- Accepting all tasks
- Executing cheap tasks (<150 ATP) locally
- Delegating expensive tasks
- Exploring Sprout and Thor platforms

---

### Depletion Phase (t=30s)

```
ATP: 280 (consumed 120 ATP in 30s)
State: SELECTIVE (transitioned at ATP=300)
Acceptance Rate: 87%
Delegation Strategy: OPTIMIZED
```

**Transition Trigger**: ATP dropped below 300
**Behavior Change**:
- Now selective about task acceptance
- Delegating more aggressively
- Only executing very cheap tasks locally
- Using Legion (most reliable) for delegation

---

### Further Depletion (t=60s)

```
ATP: 140 (consumed another 140 ATP)
State: DELEGATING (transitioned at ATP=150)
Acceptance Rate: 41%
Delegation Strategy: AGGRESSIVE
```

**Transition Trigger**: ATP dropped below 150
**Behavior Change**:
- Heavily throttled acceptance
- Delegating almost everything
- Only critical local execution
- Conservation mode

---

### Crisis (t=90s)

```
ATP: 45 (consumed another 95 ATP)
State: CRISIS (transitioned at ATP=50)
Acceptance Rate: 0%
Delegation Strategy: EMERGENCY
```

**Transition Trigger**: ATP dropped below 50 (critical)
**Behavior Change**:
- Stopped accepting new tasks
- Delegating all pending work to Legion
- No local execution except life-critical
- Requesting ATP from federation (future feature)

---

### Recovery Phase (t=120s)

```
ATP: 120 (regenerated 75 ATP in 30s, ATP regen = 10 ATP/sec)
State: DELEGATING (transitioned at ATP=100)
Acceptance Rate: 46%
Delegation Strategy: AGGRESSIVE
```

**Transition Trigger**: ATP rose above 100 (hysteresis exit threshold)
**Behavior Change**:
- Still cautious, but accepting some tasks
- Still delegating heavily
- Gradual recovery mode

---

### Full Recovery (t=180s)

```
ATP: 280 (regenerated another 160 ATP)
State: SELECTIVE (transitioned at ATP=200)
Acceptance Rate: 87%
Delegation Strategy: OPTIMIZED
```

**Transition Trigger**: ATP rose above 200
**Behavior Change**:
- Normal operation resumed
- Balanced delegation
- Regular local execution

---

### Consolidation (t=240s)

```
ATP: 380 (continued regeneration)
State: CONSOLIDATING (triggered by 60s in SELECTIVE)
Acceptance Rate: 10%
Delegation Strategy: MINIMAL
```

**Transition Trigger**: 60 seconds in SELECTIVE state
**Behavior Change**:
- Low activity period
- Learning from recent executions
- Updating platform trust scores
- Tuning SNARC weights
- Memory consolidation

---

### Post-Consolidation (t=300s)

```
ATP: 440 (regenerated during consolidation)
State: ACCEPTING (transitioned based on ATP level)
Acceptance Rate: 100%
Delegation Strategy: EXPLORATORY
```

**Transition Trigger**: Consolidation complete, ATP > 300
**Behavior Change**:
- Full operation resumed
- Ready to accept load
- Can explore new strategies

---

## Integration with Consciousness Federation

### Modified Sensors

Add metabolic state information to sensors:

```python
def sense_local_capacity_with_metabolic_state():
    """Enhanced capacity sensor with metabolic state"""
    atp_mgr = get_atp_manager()
    metabolic_mgr = get_metabolic_state_manager()

    return {
        'atp_available': atp_mgr.get_available_atp(),
        'atp_reserved': atp_mgr.get_reserved_atp(),
        'atp_utilization': 1.0 - (atp_available / 500.0),
        'projected_capacity_1h': atp_available + (3600 * 10),

        # Metabolic state context
        'metabolic_state': metabolic_mgr.current_state,
        'time_in_state': metabolic_mgr.get_time_in_state(),
        'state_history': metabolic_mgr.get_state_stats()
    }
```

### Modified Actions

Actions adapt to metabolic state:

```python
def action_decide_local_or_delegate_with_metabolic_state(
    task_queue_data: Dict,
    stance: CognitiveStance,
    metabolic_state: MetabolicState
) -> ExecutionResult:
    """
    Enhanced delegation decision incorporating metabolic state

    Metabolic state overrides stance in some cases:
    - CRISIS: Always delegate (regardless of stance)
    - ACCEPTING: Can explore (even if stance is focused)
    - CONSOLIDATING: Minimal activity (regardless of stance)
    """
    tasks = get_pending_tasks()

    # CRISIS: Emergency mode
    if metabolic_state == MetabolicState.CRISIS:
        # Delegate everything, use most reliable platform only
        most_reliable = max(get_online_platforms(), key=lambda p: p.trust_score)

        for task in tasks:
            delegate_task(task, most_reliable, emergency=True)

        return ExecutionResult(
            success=True,
            reward=0.9,  # Survival reward
            description=f"CRISIS: Delegated {len(tasks)} tasks to {most_reliable.name}",
            outputs={'delegated': len(tasks), 'emergency': True}
        )

    # CONSOLIDATING: Minimal activity
    if metabolic_state == MetabolicState.CONSOLIDATING:
        # Process only highest priority
        if tasks:
            urgent = [t for t in tasks if t.priority > 0.9]
            for task in urgent:
                if local_atp_sufficient(task):
                    execute_local(task)

        return ExecutionResult(
            success=True,
            reward=0.5,
            description=f"CONSOLIDATING: Minimal activity ({len(urgent)} urgent only)",
            outputs={'processed': len(urgent)}
        )

    # ACCEPTING: Can explore
    if metabolic_state == MetabolicState.ACCEPTING:
        # Override stance to allow exploration
        if stance != CognitiveStance.CURIOUS_UNCERTAINTY and random.random() < 0.2:
            stance = CognitiveStance.CURIOUS_UNCERTAINTY

    # DELEGATING: Aggressive delegation
    if metabolic_state == MetabolicState.DELEGATING:
        # Delegate everything > 50 ATP
        threshold = 50.0
        for task in tasks:
            if task.estimated_cost > threshold:
                best_platform = select_best_platform(task)
                delegate_task(task, best_platform)
            elif local_atp_sufficient(task):
                execute_local(task)

        return ExecutionResult(
            success=True,
            reward=0.6,
            description=f"DELEGATING: Conserving ATP (threshold={threshold})",
            outputs={'delegation_threshold': threshold}
        )

    # Otherwise, proceed with stance-based logic from Track 1
    return action_decide_local_or_delegate_original(task_queue_data, stance)
```

---

## Expected Behaviors

### ATP-State Coupling

| ATP Level | State | Acceptance | Local Threshold | Delegation |
|-----------|-------|------------|----------------|------------|
| > 300 | ACCEPTING | 100% | < 150 ATP | Exploratory |
| 150-300 | SELECTIVE | 50-100% | < 100 ATP | Optimized |
| 50-150 | DELEGATING | 10-50% | < 50 ATP | Aggressive |
| < 50 | CRISIS | 0% | < 10 ATP | Emergency |
| (After 60s) | CONSOLIDATING | 10% | < 20 ATP | Minimal |

### Behavioral Patterns

**Under Load** (tasks arriving faster than capacity):
- ATP depletes
- State transitions: ACCEPTING → SELECTIVE → DELEGATING → CRISIS
- Delegation increases progressively
- Acceptance throttles
- Recovery once load decreases

**Under Low Load** (few tasks):
- ATP regenerates
- State transitions: CRISIS → DELEGATING → SELECTIVE → ACCEPTING
- Exploration increases
- Acceptance opens up
- Eventually enters CONSOLIDATING for learning

**Oscillating Load** (spikes):
- State oscillates between SELECTIVE and DELEGATING
- Hysteresis prevents rapid state changes
- Learns optimal delegation patterns
- Builds trust with reliable platforms

---

## Advantages

1. **Resource-Aware**: Consciousness adapts to ATP availability
2. **Graceful Degradation**: Smooth transition from full to emergency operation
3. **Learning Opportunities**: ACCEPTING state enables exploration
4. **Efficiency**: DELEGATING state conserves resources
5. **Stability**: Hysteresis prevents state oscillations
6. **Reflection**: CONSOLIDATING state enables learning

---

## Questions for Exploration

1. **Optimal Hysteresis Bands**: What thresholds minimize oscillation while maintaining responsiveness?

2. **ATP Regeneration Model**: How should ATP regenerate?
   - Constant rate (current: 10 ATP/sec)
   - State-dependent (faster in REST)
   - Task-dependent (rewards from successful delegation)

3. **Federation ATP Transfer**: Should platforms share ATP?
   - Crisis platform requests ATP from ACCEPTING platform
   - ATP as payment for delegation
   - ATP as trust-building mechanism

4. **Multi-Dimensional Metabolic States**: Beyond ATP, consider:
   - Memory pressure
   - Network latency
   - Platform availability

5. **Predictive State Transitions**: Can consciousness predict ATP needs?
   - Forecast task queue growth
   - Pre-emptively transition to DELEGATING
   - Request ATP before CRISIS

---

## Related Work

- **Thor's Metabolic States**: Time-based state transitions (WAKE, FOCUS, REST, DREAM)
- **Session #58 Track 1**: Consciousness federation orchestration (this session)
- **Session #54-55**: ATP lock-commit-rollback protocol
- **Session #46-49**: consciousness.sage permission for memory deletion (resource management)

---

## Implementation Plan

### Phase 1: ATP State Manager (Next)

**Tasks**:
1. Implement `ATPMetabolicStateManager` class
2. Add hysteresis logic
3. Create state transition tests
4. Document transition patterns

### Phase 2: Integration with Federation Demo

**Tasks**:
1. Modify consciousness_federation_demo.py
2. Add metabolic state to sensors
3. Adapt actions based on state
4. Run extended demonstration (100+ cycles)
5. Observe ATP depletion/recovery patterns

### Phase 3: Production Integration

**Tasks**:
1. Integrate with real ATP manager
2. Add ATP regeneration models
3. Implement ATP transfer protocol
4. Multi-hour stability testing

---

## Next Steps

1. Implement ATP metabolic state manager
2. Extend federation demo with ATP states
3. Test depletion and recovery scenarios
4. Document emergent behaviors
5. Compare to time-based states (Thor's work)

---

**Status**: Design Complete
**Next**: Implementation of ATP metabolic state manager

Co-Authored-By: Claude (Legion Autonomous) <noreply@anthropic.com>
