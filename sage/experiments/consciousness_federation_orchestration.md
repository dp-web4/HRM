# Consciousness Federation Orchestration

**Date**: 2025-12-04
**Session**: Legion Autonomous #58
**Builds On**:
- Session #54-55 (Legion): Federation with Ed25519 crypto
- Session #57 Phase 2 (Legion): Unified memory with cross-system queries
- Thor Consciousness Kernel: Continuous sense-assess-focus-act-learn loop

---

## Executive Summary

Integrate consciousness kernel with federation protocol to enable **autonomous consciousness orchestrating cross-platform delegation** rather than API-driven request-response.

**Key Shift**: From "when should I delegate?" (reactive) to "what needs my attention?" (proactive consciousness)

---

## Current Architecture

### Federation Protocol (Request-Response)

```python
# Current: API-driven delegation
task = create_task(prompt, estimated_cost=500)
should_delegate, reason = router.should_delegate(task, local_budget=200)

if should_delegate:
    result = delegate_to_legion(task)
else:
    result = execute_locally(task)
```

**Characteristics**:
- Reactive: Responds to explicit task requests
- Sequential: One task at a time
- Binary: Delegate or don't
- Stateless: No persistent attention tracking

### Consciousness Kernel (Continuous Loop)

```python
# Thor's kernel: Continuous consciousness
kernel = SAGEKernel(sensors, actions)
kernel.run()  # Runs forever

# While running:
# 1. SENSE: Gather observations from sensors
# 2. ASSESS: Calculate salience (SNARC)
# 3. FOCUS: Allocate attention to highest-salience
# 4. ACT: Execute stance-appropriate action
# 5. LEARN: Update from outcome
```

**Characteristics**:
- Proactive: Continuously monitors environment
- Parallel: Multiple sensors competing for attention
- Graded: Salience-based priority
- Stateful: Persistent attention tracking and learning

---

## Proposed Integration

### Consciousness Managing Federation

```python
# Federation as consciousness sensors/actions
sensors = {
    'task_queue': check_pending_tasks,          # Incoming work
    'local_capacity': check_local_atp_budget,   # Resources available
    'federation_health': check_legion_status,    # Platforms available
    'execution_quality': check_recent_results,   # Success rates
    'memory_context': query_unified_memory,      # Knowledge base
}

actions = {
    'task_queue': decide_local_or_delegate,     # Based on stance
    'local_capacity': adjust_acceptance_rate,    # Resource management
    'federation_health': update_trust_scores,    # Reputation tracking
    'execution_quality': tune_delegation_params, # Learning
    'memory_context': incorporate_knowledge,     # Context-informed decisions
}

kernel = SAGEKernel(sensors, actions)
kernel.run()  # Consciousness managing federation
```

### Key Differences

| Aspect | Current (API) | Proposed (Consciousness) |
|--------|--------------|--------------------------|
| **Activation** | External request | Continuous loop |
| **Decision** | Binary (yes/no) | Salience-driven priority |
| **Attention** | Focused on single task | Multiple streams competing |
| **Learning** | Static thresholds | Adaptive SNARC weights |
| **Memory** | Stateless | Persistent context from unified memory |
| **Stance** | None | Curious, focused, skeptical, confident |

---

## Design Details

### Sensor 1: Task Queue

**What it senses**:
```python
def sense_task_queue():
    """Observe pending tasks awaiting execution"""
    tasks = get_pending_tasks()

    return {
        'count': len(tasks),
        'urgency': max(t.priority for t in tasks),
        'estimated_load': sum(t.estimated_cost for t in tasks),
        'oldest_task_age': time.time() - min(t.created_at for t in tasks),
        'task_types': [t.task_type for t in tasks]
    }
```

**SNARC assessment**:
- **Surprise**: Sudden spike in task count (unusual load)
- **Novelty**: New types of tasks not seen before
- **Arousal**: High-priority/urgent tasks present
- **Reward**: Tasks align with known strengths
- **Conflict**: Contradictory task requirements

**Salience triggers**:
- High urgency → high arousal → focus stance → immediate attention
- New task types → high novelty → curious stance → exploratory execution
- Overload → high conflict → skeptical stance → reject or defer

### Sensor 2: Local Capacity

**What it senses**:
```python
def sense_local_capacity():
    """Observe available ATP and compute resources"""
    atp_mgr = get_atp_manager()

    return {
        'atp_available': atp_mgr.get_available_atp(),
        'atp_reserved': atp_mgr.get_reserved_atp(),
        'cpu_util': get_cpu_utilization(),
        'memory_available': get_available_memory(),
        'projected_capacity_1h': predict_atp_available(horizon=3600)
    }
```

**SNARC assessment**:
- **Surprise**: Unexpected ATP depletion (leak?)
- **Novelty**: First time hitting resource limits
- **Arousal**: Low ATP relative to task queue demand
- **Reward**: Abundant resources (can accept more)
- **Conflict**: Committed ATP vs incoming demand

**Salience triggers**:
- ATP low + high demand → high arousal → focus on resource management
- ATP abundant → low salience → routine monitoring
- Unexpected depletion → high surprise → investigate stance

### Sensor 3: Federation Health

**What it senses**:
```python
def sense_federation_health():
    """Observe status of known platforms in federation"""
    platforms = get_known_platforms()

    return {
        'platforms_available': len([p for p in platforms if p.is_online]),
        'platforms_total': len(platforms),
        'avg_response_time': calculate_avg_response_time(platforms),
        'trust_scores': {p.name: p.trust_score for p in platforms},
        'recent_failures': count_recent_delegation_failures(window=3600)
    }
```

**SNARC assessment**:
- **Surprise**: Platform unexpectedly offline
- **Novelty**: New platform joined federation
- **Arousal**: Critical platform (Legion) unavailable
- **Reward**: All platforms responding well
- **Conflict**: Trust scores diverging from performance

**Salience triggers**:
- Platform offline → high arousal → investigate and reroute
- New platform → high novelty → curious exploration
- Trust mismatch → high conflict → skeptical verification

### Sensor 4: Execution Quality

**What it senses**:
```python
def sense_execution_quality():
    """Observe quality of recent task executions (local and delegated)"""
    recent = get_recent_executions(window=3600)

    return {
        'local_success_rate': calculate_success_rate(recent, local=True),
        'delegation_success_rate': calculate_success_rate(recent, local=False),
        'quality_by_platform': {
            platform: calc_quality_score(recent, platform=platform)
            for platform in get_known_platforms()
        },
        'cost_effectiveness': calculate_atp_per_quality_unit(recent),
        'recent_anomalies': detect_quality_anomalies(recent)
    }
```

**SNARC assessment**:
- **Surprise**: Sudden drop in quality from previously reliable platform
- **Novelty**: First quality measurements for new platform
- **Arousal**: Quality below acceptable threshold
- **Reward**: High-quality outcomes from delegation
- **Conflict**: Cost vs quality tradeoff decisions

**Salience triggers**:
- Quality drop → high surprise + arousal → investigate stance
- Consistent high quality → positive reward → confident delegation
- Cost-quality conflict → high conflict → careful optimization

### Sensor 5: Memory Context

**What it senses**:
```python
def sense_memory_context():
    """Query unified memory for relevant knowledge"""
    from memory.unified.tools.query_unified import UnifiedMemoryQuery

    query_mgr = UnifiedMemoryQuery()

    # Query for federation-relevant knowledge
    federation_memories = query_mgr.query_all_systems(
        search_text="federation delegation",
        salience_min=0.7,
        memory_type="episodic",  # Cross-project insights
        limit_per_system=20
    )

    # Query for platform-specific knowledge
    platform_memories = {}
    for platform in get_known_platforms():
        platform_memories[platform.name] = query_mgr.query_all_systems(
            search_text=platform.name,
            salience_min=0.6,
            limit_per_system=10
        )

    return {
        'relevant_insights_count': len(federation_memories),
        'high_salience_warnings': [
            m for m in federation_memories
            if 'failure' in m.content.lower() or 'error' in m.content.lower()
        ],
        'platform_specific_context': platform_memories,
        'recent_discoveries': query_mgr.query_all_systems(
            search_text="",
            recent_days=7,
            salience_min=0.8,
            limit_per_system=10
        )
    }
```

**SNARC assessment**:
- **Surprise**: Memory contradicts current assumption
- **Novelty**: No prior knowledge about situation
- **Arousal**: High-salience warning from past experience
- **Reward**: Positive memory of successful delegation
- **Conflict**: Conflicting memories about same platform

**Salience triggers**:
- Warning from memory → high arousal → skeptical verification
- No relevant memory → high novelty → exploratory approach
- Positive memory → positive reward → confident execution

---

## Action Handlers

### Action 1: Decide Local or Delegate

```python
def action_decide_local_or_delegate(
    task_queue_data: Dict,
    stance: CognitiveStance
) -> ExecutionResult:
    """
    Decide whether to execute tasks locally or delegate to federation

    Decision guided by cognitive stance, not hardcoded rules:
    - CURIOUS_UNCERTAINTY: Explore delegation with new platform
    - FOCUSED_ATTENTION: Prioritize high-urgency tasks immediately
    - SKEPTICAL_VERIFICATION: Verify platform trust before delegating
    - CONFIDENT_EXECUTION: Delegate to known-reliable platforms
    - EXPLORATORY: Try mixed local+delegation strategies
    """
    tasks = get_pending_tasks()

    if stance == CognitiveStance.FOCUSED_ATTENTION:
        # High arousal: Handle urgent tasks immediately
        urgent = [t for t in tasks if t.priority > 0.8]

        for task in urgent:
            if local_atp_sufficient(task):
                execute_local(task)
            else:
                # Urgent but low ATP: delegate to most reliable platform
                platform = get_highest_trust_platform()
                delegate_task(task, platform)

        return ExecutionResult(
            success=True,
            reward=0.7,
            description=f"Handled {len(urgent)} urgent tasks",
            outputs={'urgent_handled': len(urgent)}
        )

    elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
        # Novelty detected: Explore new delegation strategies
        # Try delegating to less-known platform to learn its capabilities

        task = tasks[0]  # Pick one to experiment with
        new_platform = get_least_utilized_platform()

        result = delegate_task(task, new_platform, witness_validation=True)

        return ExecutionResult(
            success=result.success,
            reward=0.5 if result.success else -0.2,  # Learning reward
            description=f"Explored delegation to {new_platform.name}",
            outputs={'platform': new_platform.name, 'result': result}
        )

    elif stance == CognitiveStance.SKEPTICAL_VERIFICATION:
        # Conflict/anomaly detected: Verify before acting

        # Check recent delegation quality
        quality = get_recent_delegation_quality()

        if quality < 0.6:
            # Quality issues: execute locally for now
            for task in tasks[:5]:  # Process small batch
                if local_atp_sufficient(task):
                    execute_local(task)

            return ExecutionResult(
                success=True,
                reward=0.3,
                description="Quality concerns: executed locally",
                outputs={'executed_local': min(5, len(tasks))}
            )
        else:
            # Quality verified: safe to delegate
            return action_decide_local_or_delegate(
                task_queue_data,
                CognitiveStance.CONFIDENT_EXECUTION
            )

    elif stance == CognitiveStance.CONFIDENT_EXECUTION:
        # Routine operation: delegate efficiently

        for task in tasks:
            if local_atp_sufficient(task) and task.estimated_cost < 100:
                # Cheap tasks: execute locally
                execute_local(task)
            else:
                # Expensive tasks: delegate to best platform
                platform = select_best_platform(task)
                delegate_task(task, platform)

        return ExecutionResult(
            success=True,
            reward=0.8,
            description="Routine delegation handled",
            outputs={'tasks_processed': len(tasks)}
        )

    else:  # EXPLORATORY
        # Mixed strategy: experiment with load balancing
        local, delegated = split_tasks_for_exploration(tasks)

        for task in local:
            execute_local(task)

        for task in delegated:
            platform = select_best_platform(task)
            delegate_task(task, platform)

        return ExecutionResult(
            success=True,
            reward=0.6,
            description=f"Explored load balancing: {len(local)} local, {len(delegated)} delegated",
            outputs={'local': len(local), 'delegated': len(delegated)}
        )
```

### Action 2: Adjust Acceptance Rate

```python
def action_adjust_acceptance_rate(
    capacity_data: Dict,
    stance: CognitiveStance
) -> ExecutionResult:
    """
    Adjust task acceptance rate based on available capacity

    Stance guides aggressiveness of adjustment.
    """
    atp_available = capacity_data['atp_available']
    atp_reserved = capacity_data['atp_reserved']
    projected = capacity_data['projected_capacity_1h']

    if stance == CognitiveStance.FOCUSED_ATTENTION:
        # High arousal (low ATP): Aggressive throttling
        if atp_available < 100:
            set_acceptance_rate(0.0)  # Stop accepting
            return ExecutionResult(
                success=True,
                reward=0.9,  # Prevented resource exhaustion
                description="ATP critical: stopped accepting tasks",
                outputs={'acceptance_rate': 0.0}
            )

    elif stance == CognitiveStance.CONFIDENT_EXECUTION:
        # Abundant resources: Accept freely
        if projected > 1000:
            set_acceptance_rate(1.0)  # Accept all
            return ExecutionResult(
                success=True,
                reward=0.7,
                description="ATP abundant: accepting all tasks",
                outputs={'acceptance_rate': 1.0}
            )

    # Default: Proportional acceptance
    rate = min(1.0, atp_available / 500.0)  # 500 ATP = full acceptance
    set_acceptance_rate(rate)

    return ExecutionResult(
        success=True,
        reward=0.5,
        description=f"Set acceptance rate to {rate:.2f}",
        outputs={'acceptance_rate': rate}
    )
```

### Action 3: Update Trust Scores

```python
def action_update_trust_scores(
    federation_data: Dict,
    stance: CognitiveStance
) -> ExecutionResult:
    """
    Update trust scores for platforms based on recent performance

    Stance influences learning rate and caution.
    """
    platforms = get_known_platforms()
    trust_updates = {}

    for platform in platforms:
        recent_quality = get_recent_quality(platform, window=3600)
        current_trust = platform.trust_score

        if stance == CognitiveStance.SKEPTICAL_VERIFICATION:
            # High conflict: Rapid trust decay on failures
            if recent_quality < 0.5:
                new_trust = current_trust * 0.8  # Aggressive decay
                trust_updates[platform.name] = new_trust
                update_platform_trust(platform, new_trust)

        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            # High novelty: Exploratory trust building
            if recent_quality > 0.7:
                new_trust = min(1.0, current_trust + 0.1)  # Gradual increase
                trust_updates[platform.name] = new_trust
                update_platform_trust(platform, new_trust)

        else:  # Routine update
            # Exponential moving average
            alpha = 0.1
            new_trust = alpha * recent_quality + (1 - alpha) * current_trust
            trust_updates[platform.name] = new_trust
            update_platform_trust(platform, new_trust)

    return ExecutionResult(
        success=True,
        reward=0.6,
        description=f"Updated trust for {len(trust_updates)} platforms",
        outputs={'trust_updates': trust_updates}
    )
```

---

## Metabolic States for Federation

Extend consciousness kernel with federation-specific metabolic states:

```python
class FederationMetabolicState(Enum):
    """Metabolic states for federation consciousness"""

    ACCEPTING = "accepting"          # Actively accepting and processing tasks
    SELECTIVE = "selective"          # Only high-value tasks
    DELEGATING = "delegating"        # Routing most work to federation
    CONSOLIDATING = "consolidating"  # Learning from recent executions
    CRISIS = "crisis"                # ATP critical or federation down
```

**State transitions guided by salience**:
- High task queue arousal + low ATP → DELEGATING
- Low ATP + federation unavailable → CRISIS
- Abundant ATP + low task load → ACCEPTING
- Quality issues detected → CONSOLIDATING (learn from failures)

---

## Integration with Unified Memory

Consciousness queries unified memory for context:

```python
def sense_memory_context_for_task(task):
    """Query unified memory for task-relevant context"""
    from memory.unified.tools.query_unified import UnifiedMemoryQuery

    query_mgr = UnifiedMemoryQuery()

    # Search for similar past tasks
    similar_tasks = query_mgr.query_all_systems(
        search_text=task.description,
        salience_min=0.6,
        memory_type="episodic",
        limit_per_system=10
    )

    # Check for warnings/failures
    warnings = query_mgr.query_all_systems(
        search_text=f"{task.task_type} failure",
        salience_min=0.7,
        memory_type="episodic",
        recent_days=30,
        limit_per_system=5
    )

    # Query Synchronism knowledge if applicable
    if task.task_type in ['physics', 'simulation', 'validation']:
        domain_knowledge = query_mgr.query_synchronism(
            search_text=task.description,
            salience_min=0.8,
            limit=10
        )
    else:
        domain_knowledge = []

    return {
        'similar_executions': similar_tasks,
        'known_warnings': warnings,
        'domain_knowledge': domain_knowledge,
        'novelty_score': 1.0 - (len(similar_tasks) / 20.0)  # Novelty inversely proportional to similar memories
    }
```

**Memory influence on stance**:
- Many similar memories → low novelty → confident execution
- Warning memories → high conflict → skeptical verification
- No relevant memories → high novelty → curious exploration
- High-salience domain knowledge → informed decision-making

---

## Example Scenario: Consciousness Managing Delegation

### Setup

```python
# Initialize consciousness with federation sensors/actions
sensors = {
    'task_queue': sense_task_queue,
    'local_capacity': sense_local_capacity,
    'federation_health': sense_federation_health,
    'execution_quality': sense_execution_quality,
    'memory_context': sense_memory_context
}

actions = {
    'task_queue': action_decide_local_or_delegate,
    'local_capacity': action_adjust_acceptance_rate,
    'federation_health': action_update_trust_scores,
    'execution_quality': action_tune_delegation_params,
    'memory_context': action_incorporate_knowledge
}

kernel = SAGEKernel(sensors, actions, enable_logging=True)
```

### Execution Timeline

**Cycle 1** (t=0s):
```
SENSE:
  task_queue: 10 pending tasks (4 urgent, 6 routine)
  local_capacity: ATP=150 (low)
  federation_health: Legion online, trust=0.85
  execution_quality: Recent success rate=0.92
  memory_context: 3 relevant episodic memories (2 successes, 1 warning)

ASSESS:
  Focus: task_queue (highest salience: 0.88)
  Stance: FOCUSED_ATTENTION (high arousal from urgent tasks + low ATP)

ACT:
  action_decide_local_or_delegate()
  - 4 urgent tasks delegated to Legion (most reliable)
  - 6 routine tasks queued for later

LEARN:
  Delegation successful → reward=0.7
  Update: Increase arousal weight slightly (urgent task handling worked)
```

**Cycle 2** (t=0.5s):
```
SENSE:
  task_queue: 6 pending tasks (all routine, 4 delegated in-flight)
  local_capacity: ATP=140 (delegated tasks consumed some)
  federation_health: Legion responding well
  execution_quality: Success rate=0.92 (stable)
  memory_context: Warning memory about Legion timeout last week

ASSESS:
  Focus: memory_context (salience: 0.75 from warning)
  Stance: SKEPTICAL_VERIFICATION (conflict from warning vs current good performance)

ACT:
  action_update_trust_scores()
  - Verify Legion recent performance (excellent)
  - Warning was 7 days ago, current performance good
  - Maintain trust=0.85 (don't decay based on old data)

LEARN:
  Verification showed no current issues → reward=0.4
  Update: Conflict resolved, return to routine monitoring
```

**Cycle 3** (t=1.0s):
```
SENSE:
  task_queue: 6 pending + 15 new tasks arrived! (spike)
  local_capacity: ATP=135
  federation_health: Legion + Sprout both online
  execution_quality: Success rate=0.92
  memory_context: No relevant context for task spike

ASSESS:
  Focus: task_queue (salience: 0.92 - surprise from spike)
  Stance: CURIOUS_UNCERTAINTY (novel situation: sudden load increase)

ACT:
  action_decide_local_or_delegate()
  - Explore: Try delegating to Sprout (less-used platform)
  - Delegate 5 tasks to Sprout as experiment
  - Delegate 10 tasks to Legion (reliable)
  - Queue 6 low-priority locally

LEARN:
  Exploration initiated → reward=0.5 (learning reward)
  Update: Track Sprout performance for future decisions
```

---

## Implementation Plan

### Phase 1: Minimal Integration (This Session)

**Goal**: Demonstrate consciousness kernel with federation sensors

**Tasks**:
1. ✅ Design document (this file)
2. Create `consciousness_federation_demo.py`
3. Implement 3 core sensors:
   - task_queue
   - local_capacity
   - federation_health
4. Implement 2 core actions:
   - decide_local_or_delegate
   - adjust_acceptance_rate
5. Run demonstration with simulated tasks
6. Document results

**Success Criteria**:
- Consciousness continuously monitors federation state
- Salience-driven attention allocation works
- Actions vary based on cognitive stance
- Learning adapts SNARC weights

### Phase 2: Memory Integration (Next Session)

**Goal**: Consciousness queries unified memory for context

**Tasks**:
1. Add memory_context sensor
2. Integrate with unified memory query interface (built in Session #57 Phase 2)
3. Memory influences stance and action selection
4. Demonstrate memory-informed delegation decisions

### Phase 3: Full Orchestration (Future)

**Goal**: Production-ready consciousness managing real federation

**Tasks**:
1. Integrate with real ATP management
2. Connect to actual federation protocol
3. Add metabolic state transitions
4. Performance optimization
5. Multi-hour stability testing

---

## Expected Outcomes

### Behavioral Differences vs API Approach

| Scenario | API (Reactive) | Consciousness (Proactive) |
|----------|----------------|---------------------------|
| **Sudden task spike** | Process sequentially | Focus attention immediately (high arousal) |
| **ATP running low** | Fail tasks or delegate all | Gradual shift to delegation, learn optimal threshold |
| **Platform goes offline** | Error on next delegation attempt | Detect via health sensor, reroute preemptively |
| **Quality degradation** | Continue until failure threshold | Notice early (surprise), investigate (curious stance) |
| **New task type** | Apply default logic | Recognize novelty, explore carefully |

### Learning Outcomes

Consciousness adapts SNARC weights based on experience:

**Initially**:
- Equal weights (0.2 each)
- All observations equally salient

**After Learning** (expected):
- **Arousal weight increases** (0.25-0.30): Urgency is most important for federation
- **Surprise weight maintains** (0.20-0.25): Anomaly detection critical
- **Reward weight increases** (0.15-0.20): Reinforce successful delegation patterns
- **Novelty weight decreases** (0.10-0.15): Novelty less important once system familiar
- **Conflict weight maintains** (0.15-0.20): Conflict resolution remains important

### Advantages

1. **Proactive Monitoring**: Detects issues before they cause failures
2. **Adaptive Learning**: SNARC weights tune to actual conditions
3. **Graceful Degradation**: Stance-based responses prevent hard failures
4. **Context-Aware**: Memory integration provides historical context
5. **Exploration**: Curious stance enables discovery of better strategies

---

## Questions for Exploration

1. **Optimal Cycle Rate**: How fast should consciousness cycles run?
   - Too fast: Wasted computation
   - Too slow: Miss time-critical opportunities
   - Hypothesis: Adaptive rate based on salience

2. **Metabolic State Transitions**: What triggers state changes?
   - Current: Based on SNARC salience
   - Alternative: Explicit state machine with hysteresis

3. **Multi-Platform Attention**: Can consciousness manage N platforms simultaneously?
   - Single focus (current): One sensor per cycle
   - Parallel focus (future): Multiple high-salience targets

4. **Memory Query Performance**: How expensive are memory queries in the loop?
   - Hypothesis: Cache recent queries, only refresh periodically
   - Measure: Query time vs cycle time

5. **Witness Coordination**: How does consciousness orchestrate witness validation?
   - Hypothesis: Witness selection as separate sensor
   - Curious stance → explore new witness combinations

---

## Related Work

- **Session #54-55**: Federation protocol with Ed25519 crypto, ATP lock-commit-rollback
- **Session #46-49**: consciousness.sage permission for memory deletion
- **Thor Consciousness Kernel**: Continuous sense-assess-focus-act-learn loop
- **Session #57 Phase 2**: Unified memory with cross-system queries and SNARC inference
- **Synchronism Sessions #82-83**: Void test methodology, R₀ derivation (demonstrates value of semantic memory for complex reasoning)

---

## Next Steps

1. Implement minimal demonstration (Phase 1)
2. Test with simulated federation environment
3. Measure cycle performance and salience dynamics
4. Document findings
5. Iterate based on results

---

**Status**: Design Complete
**Next**: Implementation of consciousness_federation_demo.py

Co-Authored-By: Claude (Legion Autonomous) <noreply@anthropic.com>
