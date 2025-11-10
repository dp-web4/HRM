# Track 3: SNARC Cognition - Architecture Design

**Date**: 2025-11-09 21:00
**Session**: Autonomous Session #21
**Status**: Architecture Preparation (Implementation Pending)
**Jetson Nano Deployment Roadmap**: Track 3 of 10

---

## Executive Summary

Architectural design for SNARC Cognition layer, extending reactive salience assessment (Tracks 1-2) with proactive cognitive capabilities: attention allocation, working memory, deliberation, and goal management.

**Design Philosophy**: Transform SNARC from reactive (respond to salience) to cognitive (plan, attend, deliberate, pursue goals).

**Integration**: Builds on Track 1 (Sensor Trust/Fusion) and Track 2 (STM/LTM Memory).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAGE COGNITION LAYER                          │
│                     (Track 3 - NEW)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │ Goal Manager     │───▶│ Deliberation     │                  │
│  │                  │    │ Engine           │                  │
│  │ - Hierarchical   │    │                  │                  │
│  │ - Activation     │    │ - Plan ahead     │                  │
│  │ - Progress       │    │ - Evaluate       │                  │
│  └────────┬─────────┘    │ - Meta-cognition │                  │
│           │              └─────────┬────────┘                  │
│           │                        │                            │
│           ▼                        ▼                            │
│  ┌──────────────────────────────────────────┐                  │
│  │      Working Memory                      │                  │
│  │                                           │                  │
│  │  - Active task context                   │                  │
│  │  - Multi-step plan state                 │                  │
│  │  - Intermediate results                  │                  │
│  │  - Sensor-goal bindings                  │                  │
│  └──────────────────┬───────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────────┐                  │
│  │      Attention Manager                   │                  │
│  │                                           │                  │
│  │  - Sensor focus allocation               │                  │
│  │  - Resource prioritization               │                  │
│  │  - Inhibition of low-relevance           │                  │
│  └──────────────────┬───────────────────────┘                  │
│                     │                                            │
└─────────────────────┼────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SNARC SERVICE                                 │
│                  (Existing - Track 2)                            │
├─────────────────────────────────────────────────────────────────┤
│  - Salience assessment (5D: S/N/A/R/C)                          │
│  - Memory retrieval (STM/LTM)                                   │
│  - Novelty computation                                          │
│  - SalienceReport generation                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SENSOR LAYER                                     │
│                  (Track 1)                                       │
├─────────────────────────────────────────────────────────────────┤
│  - Multi-sensor fusion                                          │
│  - Trust-weighted observations                                  │
│  - Conflict resolution                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Attention Manager

### Purpose
Allocate computational resources to most relevant sensors based on current goals, salience, and task demands.

### Design Principles
1. **Selective Processing**: Not all sensors active simultaneously
2. **Goal-Driven**: Attention follows current goals
3. **Salience-Responsive**: High-salience events can interrupt
4. **Resource-Aware**: Budget computation for Nano constraints

### Architecture

```python
class AttentionManager:
    """
    Manages sensor attention allocation

    Attention = weighted focus on subset of available sensors
    Resources = computation budget (memory, latency, GPU)
    """

    def __init__(
        self,
        available_sensors: List[str],
        resource_budget: ResourceBudget,
        memory_retrieval: MemoryRetrieval  # From Track 2
    ):
        # Available sensors
        self.sensors = available_sensors

        # Resource constraints
        self.budget = resource_budget

        # Memory for context
        self.memory = memory_retrieval

        # Current attention state
        self.focus_weights = {}  # sensor_id -> weight (0-1)
        self.active_sensors = []  # Currently attended sensors

        # Attention history (for learning)
        self.attention_history = deque(maxlen=1000)

    def allocate_attention(
        self,
        current_salience: Dict[str, float],  # From SNARC
        active_goals: List[Goal],  # From GoalManager
        context: Dict[str, Any]  # From WorkingMemory
    ) -> AttentionAllocation:
        """
        Decide which sensors to focus on

        Factors:
        1. Goal relevance (which sensors help current goals?)
        2. Salience (which sensors have high current salience?)
        3. Memory (which sensors were useful in similar situations?)
        4. Resource budget (how many sensors can we afford to process?)

        Returns:
            AttentionAllocation with focus weights per sensor
        """
        pass

    def inhibit_sensor(self, sensor_id: str, duration: int):
        """Temporarily suppress a sensor (low relevance)"""
        pass

    def boost_sensor(self, sensor_id: str, factor: float):
        """Amplify attention to a sensor (high importance)"""
        pass

    def interrupt(self, sensor_id: str, urgency: float):
        """Handle attention interrupt (high salience event)"""
        pass
```

### Key Algorithms

**Attention Scoring**:
```
attention_score(sensor) =
    α * goal_relevance(sensor, active_goals) +
    β * current_salience(sensor) +
    γ * memory_utility(sensor, context) +
    δ * trust_score(sensor)  # From Track 1

where α + β + γ + δ = 1.0
```

**Resource Allocation**:
```
Given budget B (e.g., max 3 active sensors on Nano):
1. Sort sensors by attention_score (descending)
2. Select top K sensors where K ≤ B
3. Allocate weights proportional to scores
4. Normalize weights to sum to 1.0
```

**Interrupt Handling**:
```
If salience(sensor) > interrupt_threshold:
    - Save current attention state
    - Shift focus to high-salience sensor
    - After N cycles, restore or adapt
```

### Integration Points

**With SNARC**:
- Input: Current salience scores from SNARC
- Output: Attention-weighted sensor priorities
- SNARC uses attention weights to modulate salience computation

**With Track 1 (Sensor Trust)**:
- Input: Trust scores per sensor
- Use: Weight attention by reliability (don't focus on unreliable sensors)

**With Track 2 (Memory)**:
- Query: Similar past situations, which sensors were useful?
- Use: Memory-informed attention allocation

**With Working Memory**:
- Input: Current task context
- Use: Attend to task-relevant sensors

---

## Component 2: Working Memory

### Purpose
Maintain active task context, multi-step plans, and intermediate results for ongoing cognitive processes.

### Design Principles
1. **Limited Capacity**: Small buffer (7±2 items, classic cognitive limit)
2. **Task-Bound**: Content tied to current goals/tasks
3. **Fast Access**: O(1) retrieval for active context
4. **Ephemeral**: Clears when task completes

### Architecture

```python
@dataclass
class WorkingMemorySlot:
    """
    Single item in working memory

    Analogy: A "sticky note" for the cognitive system
    """
    slot_id: str
    content_type: str  # "goal", "plan_step", "intermediate_result", "binding"
    content: Any
    priority: float  # How important to retain
    timestamp: float
    goal_id: Optional[str]  # Which goal owns this slot

@dataclass
class PlanStep:
    """Step in a multi-step plan"""
    step_id: int
    action: str
    preconditions: List[str]
    expected_outcome: str
    status: str  # "pending", "active", "complete", "failed"

@dataclass
class SensorGoalBinding:
    """Binding between sensor observation and goal"""
    sensor_id: str
    observation: Any
    goal_id: str
    relevance: float  # How relevant to goal


class WorkingMemory:
    """
    Active task context and multi-step plan state

    Capacity: ~7-10 slots (cognitively realistic)
    Lifetime: Duration of active task/goal
    """

    def __init__(
        self,
        capacity: int = 10,
        memory_retrieval: MemoryRetrieval  # From Track 2
    ):
        self.capacity = capacity
        self.memory = memory_retrieval

        # Active slots
        self.slots: Dict[str, WorkingMemorySlot] = {}

        # Plan tracking
        self.active_plan: Optional[List[PlanStep]] = None
        self.current_step: Optional[int] = None

        # Sensor-goal bindings
        self.bindings: List[SensorGoalBinding] = []

    def add_item(
        self,
        content_type: str,
        content: Any,
        priority: float,
        goal_id: Optional[str] = None
    ) -> str:
        """
        Add item to working memory

        If at capacity, evict lowest-priority item
        """
        pass

    def get_context(self, goal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current working memory context

        Optionally filter by goal_id
        """
        pass

    def load_plan(self, plan: List[PlanStep]):
        """Load multi-step plan into working memory"""
        pass

    def advance_plan(self) -> Optional[PlanStep]:
        """Move to next step in plan"""
        pass

    def bind_sensor_to_goal(
        self,
        sensor_id: str,
        observation: Any,
        goal_id: str,
        relevance: float
    ):
        """Create sensor-goal binding"""
        pass

    def consolidate_to_ltm(self):
        """
        When task completes, consolidate key items to LTM

        Strategy: High-priority items → episodic memories
        """
        pass
```

### Capacity Management

**Eviction Policy**:
```
When at capacity and adding new item:
1. Compute retention_score for each slot:
   retention_score = priority * recency_weight
2. Evict slot with lowest retention_score
3. Optionally: consolidate evicted high-priority items to LTM
```

**Priority Scoring**:
```
- Goal: 1.0 (highest priority, defines task)
- Plan step (current): 0.9
- Plan step (future): 0.7
- Intermediate result: 0.6
- Sensor binding: 0.5
- Other: 0.3
```

### Integration Points

**With Goal Manager**:
- Input: Active goals
- Output: Goal context for decision-making

**With Deliberation Engine**:
- Input: Multi-step plans
- Output: Current plan state, intermediate results

**With Attention Manager**:
- Output: Task context for attention allocation

**With Track 2 (LTM)**:
- Consolidate completed task context to LTM
- Query LTM for similar past tasks

---

## Component 3: Deliberation Engine

### Purpose
Plan ahead, evaluate alternatives, and make reflective decisions before acting.

### Design Principles
1. **Look-Ahead**: Predict outcomes of candidate actions
2. **Comparison**: Evaluate multiple alternatives
3. **Risk-Aware**: Consider uncertainty and failure modes
4. **Meta-Cognitive**: Assess confidence in decisions

### Architecture

```python
@dataclass
class Alternative:
    """Candidate action being considered"""
    action_id: str
    action_description: str
    predicted_outcome: Dict[str, float]  # outcome_type -> probability
    expected_reward: float
    expected_cost: float
    confidence: float  # How certain is prediction?
    risk: float  # Variance in outcome

@dataclass
class DeliberationResult:
    """Result of deliberation process"""
    chosen_alternative: Alternative
    alternatives_considered: List[Alternative]
    reasoning: str
    confidence: float
    deliberation_time: float  # How long did we think?


class DeliberationEngine:
    """
    Multi-step planning and alternative evaluation

    Transforms reactive SNARC into deliberative SNARC
    """

    def __init__(
        self,
        memory_retrieval: MemoryRetrieval,  # From Track 2
        working_memory: WorkingMemory
    ):
        self.memory = memory_retrieval
        self.working_memory = working_memory

        # Prediction models (learned from experience)
        self.outcome_predictors = {}  # action_type -> predictor

        # Deliberation history
        self.deliberation_history = []

    def deliberate(
        self,
        situation: Dict[str, Any],
        available_actions: List[str],
        goal: Goal,
        time_budget: float = 0.1  # seconds to deliberate
    ) -> DeliberationResult:
        """
        Deliberate over available actions

        Process:
        1. Generate alternatives (one per action)
        2. Predict outcomes for each (using memory + models)
        3. Evaluate alternatives (expected utility)
        4. Select best alternative
        5. Assess confidence

        Args:
            situation: Current state (sensors, context)
            available_actions: Possible actions to consider
            goal: Current goal being pursued
            time_budget: How long to deliberate (real-time constraint)

        Returns:
            DeliberationResult with chosen action and reasoning
        """
        pass

    def predict_outcome(
        self,
        action: str,
        situation: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Predict outcome of action in current situation

        Strategy:
        1. Query memory for similar (situation, action) pairs
        2. Retrieve outcomes from those past experiences
        3. Aggregate into probability distribution
        4. Use learned model if available

        Returns:
            Outcome probabilities {outcome_type: probability}
        """
        pass

    def evaluate_alternative(
        self,
        alternative: Alternative,
        goal: Goal
    ) -> float:
        """
        Compute expected utility of alternative

        utility = Σ_outcome P(outcome) * value(outcome, goal)
        """
        pass

    def generate_plan(
        self,
        goal: Goal,
        situation: Dict[str, Any],
        max_steps: int = 10
    ) -> List[PlanStep]:
        """
        Generate multi-step plan to achieve goal

        Strategy: Forward search with heuristic pruning
        1. Start from current situation
        2. Generate possible next actions
        3. Predict outcomes
        4. Recurse from predicted states
        5. Prune low-utility branches
        6. Return best path
        """
        pass

    def meta_cognition(
        self,
        deliberation_result: DeliberationResult
    ) -> Dict[str, Any]:
        """
        Reflect on deliberation quality

        Questions:
        - Am I confident in this decision?
        - Did I consider enough alternatives?
        - Is my outcome prediction reliable?
        - Should I deliberate more or act now?

        Returns:
            Meta-cognitive assessment
        """
        pass
```

### Key Algorithms

**Outcome Prediction**:
```
predict_outcome(action, situation):
    # Memory-based prediction
    similar_experiences = memory.query_similar(situation, k=10)

    outcomes = {}
    for exp in similar_experiences:
        if exp.action == action:
            outcomes[exp.outcome] += exp.similarity_weight

    # Normalize to probabilities
    return normalize(outcomes)
```

**Expected Utility**:
```
expected_utility(alternative, goal) =
    Σ P(outcome) * [value(outcome, goal) - cost(outcome)]

where:
    value(outcome, goal) = how much does outcome advance goal?
    cost(outcome) = resource cost (time, energy, risk)
```

**Multi-Step Planning**:
```
plan = generate_plan(goal, situation):
    frontier = [(situation, [])]  # (state, path)

    while frontier and time_remaining > 0:
        state, path = frontier.pop_best()

        if satisfies(state, goal):
            return path

        for action in available_actions(state):
            next_state = predict_outcome(action, state)
            new_path = path + [action]

            if utility(next_state, goal) > threshold:
                frontier.add((next_state, new_path))

    return best_partial_plan(frontier)
```

### Integration Points

**With SNARC**:
- Input: Current salience (reactive signal)
- Output: Deliberative action choice (planned response)
- Transform: Reactive → Deliberative

**With Working Memory**:
- Load plans into working memory
- Track plan execution state

**With Track 2 (Memory)**:
- Query similar past situations
- Retrieve outcomes from past actions
- Learn outcome predictors

---

## Component 4: Goal Manager

### Purpose
Maintain hierarchical goals, track progress, manage goal activation/inhibition, and handle goal switching.

### Design Principles
1. **Hierarchical**: High-level goals decompose into subgoals
2. **Dynamic**: Goals activate/deactivate based on context
3. **Progress-Aware**: Track advancement toward goals
4. **Adaptive**: Switch goals when blocked or completed

### Architecture

```python
@dataclass
class Goal:
    """
    Goal in the goal hierarchy

    Examples:
    - "Explore environment" (high-level)
    - "Navigate to landmark" (mid-level)
    - "Avoid obstacle" (low-level)
    """
    goal_id: str
    description: str
    goal_type: str  # "exploration", "navigation", "manipulation", etc.

    # Hierarchy
    parent_goal: Optional[str]  # Parent goal ID
    subgoals: List[str]  # Child goal IDs

    # Activation
    activation: float  # 0.0-1.0 (how active is this goal?)
    priority: float  # 0.0-1.0 (how important?)

    # Progress
    progress: float  # 0.0-1.0 (how close to completion?)
    status: str  # "pending", "active", "blocked", "completed", "failed"

    # Success criteria
    success_condition: Callable[[Dict[str, Any]], bool]

    # Context
    created_at: float
    last_updated: float
    metadata: Dict[str, Any]


class GoalManager:
    """
    Hierarchical goal management system

    Responsibilities:
    - Maintain goal hierarchy
    - Activate/deactivate goals
    - Track progress
    - Handle goal switching
    - Resolve goal conflicts
    """

    def __init__(
        self,
        memory_retrieval: MemoryRetrieval,  # From Track 2
        working_memory: WorkingMemory
    ):
        self.memory = memory_retrieval
        self.working_memory = working_memory

        # Goal registry
        self.goals: Dict[str, Goal] = {}

        # Goal hierarchy (DAG)
        self.hierarchy = nx.DiGraph()  # NetworkX directed graph

        # Active goals
        self.active_goals: List[str] = []

        # Goal history
        self.goal_history = []

    def add_goal(
        self,
        goal: Goal,
        parent_goal_id: Optional[str] = None
    ):
        """Add goal to hierarchy"""
        pass

    def activate_goal(self, goal_id: str):
        """Activate a goal (and its ancestors)"""
        pass

    def deactivate_goal(self, goal_id: str):
        """Deactivate a goal (and its descendants)"""
        pass

    def update_progress(
        self,
        goal_id: str,
        progress_delta: float,
        context: Dict[str, Any]
    ):
        """Update goal progress based on recent actions"""
        pass

    def get_active_goals(
        self,
        filter_by: Optional[str] = None
    ) -> List[Goal]:
        """Get currently active goals, optionally filtered by type"""
        pass

    def check_goal_completion(
        self,
        goal_id: str,
        situation: Dict[str, Any]
    ) -> bool:
        """Check if goal is satisfied"""
        pass

    def handle_goal_conflict(
        self,
        goal1_id: str,
        goal2_id: str
    ) -> str:
        """
        Resolve conflict between two goals

        Strategy:
        - Compare priorities
        - Consider progress (don't abandon nearly-complete goals)
        - Check resource constraints

        Returns:
            goal_id of winner
        """
        pass

    def suggest_next_goal(
        self,
        situation: Dict[str, Any]
    ) -> Optional[Goal]:
        """
        Suggest next goal to pursue

        Based on:
        - Current situation
        - Goal priorities
        - Memory of successful goal sequences
        - Progress on existing goals
        """
        pass
```

### Goal Activation

**Spreading Activation**:
```
When goal G is activated:
1. Activate G (activation = 1.0)
2. Activate parent goals recursively (activation *= 0.8 per level)
3. Activate subgoals (activation *= 0.9 per level)
4. Inhibit conflicting goals (activation *= 0.5)
```

**Progress Tracking**:
```
progress(goal) =
    If leaf goal: manual progress updates (0.0 → 1.0)
    If parent goal: mean(progress(subgoals))
```

**Goal Switching**:
```
When to switch from goal G1 to G2:
1. G1 is blocked (no progress for N cycles)
2. G2 has higher activation * priority
3. G2 is interrupt-worthy (very high salience event)
4. G1 is complete

Switching cost: save G1 context to working memory/LTM
```

### Integration Points

**With Deliberation Engine**:
- Provide active goals for planning
- Receive multi-step plans for goal achievement

**With Attention Manager**:
- Active goals guide attention allocation
- Goal-relevant sensors get higher attention

**With Working Memory**:
- Active goal context stored in working memory
- Plan steps tied to goals

**With SNARC**:
- High-reward salience may trigger goal activation
- Goals modulate reward dimension

---

## Cross-Component Integration

### Information Flow

```
1. Goal Manager → selects active goals
                ↓
2. Attention Manager → allocates focus to goal-relevant sensors
                ↓
3. SNARC → assesses salience with attention weights
                ↓
4. Deliberation Engine → plans actions given salience + goals
                ↓
5. Working Memory → maintains plan state
                ↓
6. Action → executed (via embodied actor)
                ↓
7. Outcome → feeds back to memory (Track 2)
                ↓
8. Goal Manager → updates progress
```

### Shared Data Structures

**Sensor Context**:
```python
@dataclass
class SensorContext:
    """Unified sensor state"""
    sensor_id: str
    observation: Any
    trust_score: float  # From Track 1
    salience_score: float  # From SNARC
    attention_weight: float  # From Attention Manager
    goal_relevance: Dict[str, float]  # goal_id -> relevance
```

**Cognitive State**:
```python
@dataclass
class CognitiveState:
    """Complete cognitive state snapshot"""
    timestamp: float

    # Goals
    active_goals: List[Goal]

    # Attention
    attention_allocation: AttentionAllocation

    # Working memory
    working_memory_contents: List[WorkingMemorySlot]
    active_plan: Optional[List[PlanStep]]

    # Deliberation
    last_deliberation: Optional[DeliberationResult]

    # Sensors (from Track 1)
    sensor_observations: Dict[str, Any]
    sensor_trust: Dict[str, float]

    # Salience (from SNARC)
    salience_report: SalienceReport

    # Memory (from Track 2)
    recent_memories: List[STMEntry]
    relevant_ltm: List[EpisodicMemory]
```

---

## Performance Targets (Jetson Nano)

### Latency Budget
- **Attention allocation**: <5ms
- **Working memory access**: <1ms
- **Deliberation** (single alternative): <10ms
- **Deliberation** (3 alternatives): <30ms
- **Goal update**: <2ms
- **Total cognitive overhead**: <50ms per cycle

Target: 100ms total cycle time (50ms cognition + 50ms sensing/acting)

### Memory Footprint
- **Attention state**: ~1MB
- **Working memory** (10 slots): ~5MB
- **Goal hierarchy** (50 goals): ~2MB
- **Deliberation cache**: ~10MB
- **Total**: ~20MB (added to Track 2's 60MB = 80MB total)

### Computational Complexity
- Attention allocation: O(N) where N = number of sensors (~10)
- Working memory: O(1) access, O(K) eviction where K = capacity (10)
- Deliberation: O(A × M) where A = alternatives (~3), M = memory lookups (~10)
- Goal update: O(G) where G = active goals (~5)

All operations sub-linear, real-time compatible.

---

## Testing Strategy

### Test Scenarios

**Test 1: Attention Allocation**
- Multi-sensor environment (vision, audio, proprioception, IMU)
- Active goal: "Navigate to landmark"
- Expected: Higher attention to vision, lower to audio
- Validate: Attention weights reflect goal relevance

**Test 2: Working Memory Capacity**
- Load 15 items (exceeds capacity of 10)
- Expected: 5 lowest-priority items evicted
- Validate: High-priority items retained

**Test 3: Multi-Step Planning**
- Goal: "Reach location X" with obstacle
- Expected: Plan with 3+ steps (navigate around obstacle)
- Validate: Plan reaches goal, avoids obstacle

**Test 4: Goal Switching**
- Active goal: "Explore room"
- Interrupt: High-salience event (unexpected motion)
- Expected: Switch to "Investigate motion" goal
- Validate: Goal switch occurs, context saved

**Test 5: Deliberation**
- Situation: Three possible paths
- Expected: Deliberation evaluates all three, selects best
- Validate: Chosen path has highest expected utility

**Test 6: Integration**
- Complete cognitive cycle with all components
- Expected: Goal → Attention → Salience → Deliberation → Action → Update
- Validate: Information flows correctly, timing < 100ms

**Test 7: Memory-Informed Decisions**
- Situation similar to past experience
- Expected: Deliberation uses memory for outcome prediction
- Validate: Better decisions with memory vs. without

### Success Criteria
- ✅ All tests passing
- ✅ Latency < 100ms per cycle (50ms cognition budget)
- ✅ Memory footprint < 20MB
- ✅ Integration with Tracks 1-2 working
- ✅ Cognitive behaviors observable (attention shifts, planning, goal pursuit)

---

## Implementation Plan

### Phase 1: Core Components (Session #22 or next continuous session)
1. Implement `sage/cognition/attention.py` (~600 lines)
2. Implement `sage/cognition/working_memory.py` (~500 lines)
3. Implement `sage/cognition/deliberation.py` (~700 lines)
4. Implement `sage/cognition/goal_manager.py` (~600 lines)

**Total**: ~2400 lines (similar to Tracks 1-2)

### Phase 2: Integration (Same session)
5. Create integration layer with SNARC
6. Connect to Track 1 (Sensor Trust) and Track 2 (Memory)
7. Create `CognitiveState` unified state representation

### Phase 3: Testing (Same session)
8. Implement `sage/tests/test_snarc_cognition.py` (~600 lines)
9. Run 7 test scenarios
10. Validate performance (latency, memory)

### Phase 4: Documentation (Same session)
11. Document findings in `TRACK3_SNARC_COGNITION_FINDINGS.md`
12. Update roadmap with Track 3 completion
13. Commit to git with attribution

**Estimated Time**: 1.5-2 hours (based on Tracks 1-2 experience)

---

## Dependencies

### Required from Track 1
- `SensorTrust` class
- `SensorFusion` class
- Trust scores per sensor

### Required from Track 2
- `MemoryRetrieval` class
- `STMEntry` and `EpisodicMemory` classes
- Memory query methods

### Required from SNARC
- `SalienceReport` dataclass
- `SalienceBreakdown` dataclass
- `CognitiveStance` enum

### External Libraries
- `numpy`: Array operations
- `torch`: Tensor operations (if needed for predictions)
- `networkx`: Goal hierarchy graph (may use dict instead for simplicity)
- `dataclasses`: Data structures
- `typing`: Type hints

---

## Design Decisions

### 1. Attention Allocation Strategy
**Decision**: Weighted combination of goal-relevance, salience, memory, trust
**Rationale**: Multi-factor approach balances reactive and deliberative
**Alternative**: Pure goal-driven (ignores salience interrupts)

### 2. Working Memory Capacity
**Decision**: 10 slots (7±2 cognitive limit, rounded up)
**Rationale**: Cognitively realistic, manageable for Nano
**Alternative**: Unlimited (violates cognitive realism)

### 3. Deliberation Depth
**Decision**: Forward search, limited to 3-5 alternatives, 2-3 steps ahead
**Rationale**: Real-time constraint (< 30ms), diminishing returns beyond
**Alternative**: Deep search (too slow for real-time)

### 4. Goal Hierarchy Structure
**Decision**: DAG (directed acyclic graph), not strict tree
**Rationale**: Allows multiple paths to goals, more flexible
**Alternative**: Strict tree (too rigid, single path to each goal)

### 5. Memory Integration
**Decision**: Query memory for outcome prediction, not full model learning
**Rationale**: Simple, interpretable, leverages Track 2 memory system
**Alternative**: Learned world model (complex, overfitting risk with small data)

---

## Open Questions (for Implementation Session)

1. **Attention interrupt threshold**: What salience score triggers attention shift?
   → Propose: 0.9 (critical salience)

2. **Working memory eviction**: Consolidate to LTM or just discard?
   → Propose: Consolidate high-priority items (priority > 0.7)

3. **Deliberation time budget**: Fixed or adaptive?
   → Propose: Adaptive (more time for complex situations, less for simple)

4. **Goal activation decay**: How fast do inactive goals decay?
   → Propose: Exponential decay, half-life = 100 cycles

5. **Multi-goal handling**: Pursue multiple goals simultaneously or serialize?
   → Propose: Simultaneous with priority-based resource allocation

6. **Confidence calibration**: How to compute deliberation confidence?
   → Propose: Based on memory coverage (more similar experiences = higher confidence)

---

## Next Steps

**This Session** (Architecture Preparation):
- ✅ Create `sage/cognition` directory
- ✅ Design attention mechanism architecture
- ✅ Design working memory architecture
- ✅ Design deliberation engine architecture
- ✅ Design goal management architecture
- ✅ Document integration points
- → Update worklog with architecture complete

**Next Session** (Implementation):
- Implement all four components (~2400 lines)
- Create test suite (~600 lines)
- Run tests and validate performance
- Document findings
- Commit Track 3 to git

**Pattern**: Architecture (Session #21) → Implementation (Session #22)

---

## Conclusion

Track 3 architecture design is complete and ready for implementation. Design builds naturally on Tracks 1-2, transforming reactive SNARC into cognitive SNARC with attention, working memory, deliberation, and goals.

**Key Achievement**: Comprehensive architectural design with clear implementation path.

**Ready For**: Next continuous development session (1.5-2 hours, similar to Tracks 1-2).

**Status**: ✅ ARCHITECTURE COMPLETE, IMPLEMENTATION PENDING

---

**Architecture Design**: Autonomous Session #21
**Implementation**: Next continuous session (Session #22 or user-directed)
**Documentation**: Complete
**Status**: ✅ READY FOR IMPLEMENTATION
