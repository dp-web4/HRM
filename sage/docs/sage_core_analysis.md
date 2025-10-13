# SAGE Core Orchestration Analysis

**Date**: October 12, 2025
**Analysis Type**: Implementation Investigation
**Status**: Based on actual codebase examination

---

## Executive Summary

SAGE (Sentient Agentic Generative Engine) is implemented as a **distributed orchestration system** rather than a single continuous loop. The architecture consists of multiple interconnected components:

1. **SAGE Core** (`sage_core.py`, `sage_v2.py`) - 100M parameter H/L dual-module transformer
2. **HRM Orchestrator** (`hrm_orchestrator.py`) - Async ATP-based resource orchestration
3. **IRP System** (`irp/base.py`, `irp/orchestrator.py`) - Plugin-based iterative refinement
4. **Metabolic State Manager** (`metabolic-state-manager.py`) - Adaptive operational mode controller
5. **Awareness Loop** (`awareness_loop.py`) - Bidirectional audio conversation system
6. **SNARC Scorer** (`snarc_scorer.py`) - Attention allocation via salience computation

**Key Finding**: The SAGE_CORE_SPECIFICATION.md describes an **ideal continuous loop**, but the actual implementation is **modular and event-driven**, with orchestration happening at multiple levels through async coordination.

---

## 1. Main Loop Structure

### 1.1 Specification vs Implementation Gap

**SPECIFICATION** (`SAGE_CORE_SPECIFICATION.md` lines 39-88):
```python
class SAGE:
    def run(self):
        """Continuous inference loop - THIS IS SAGE"""
        while True:
            # 0. Check metabolic state
            # 1. Sense observations
            # 2. Update temporal context
            # 3. Compute attention
            # 4. Plan resources
            # 5. Load/unload resources
            # 6. Invoke reasoning
            # 7. Update memory/trust
            # 8. Execute actions
            # 9. Update energy
```

**ACTUAL IMPLEMENTATION**: No single `SAGE.run()` found. Instead:

#### A. SAGECore (`sage_core.py`)
- **Type**: PyTorch neural network module (not a loop)
- **Parameters**: ~100M (45M H-module + 45M L-module + 10M communication)
- **Function**: Single forward pass with iterative H↔L reasoning cycles

```python
class SAGECore(nn.Module):
    def forward(self, input_ids, context=None, num_cycles=None):
        # Iterative reasoning cycles (default: 8)
        for cycle in range(num_cycles):
            h_states, strategy = self.h_module(h_states, context)
            l_states, actions = self.l_module(l_states, strategy)
            h_states, l_states = self.communication(h_states, l_states)

            # Early stopping if confident
            halt_prob = self.halt_predictor(combined)
            if halt_prob.mean() > 0.99:
                break

        # Resource allocation decision
        resource_allocation = self.resource_router(strategy)

        return {
            'output': actions,
            'strategy': strategy,
            'resource_allocation': resource_allocation,
            ...
        }
```

**Location**: `/home/dp/ai-workspace/HRM/sage/core/sage_core.py` lines 354-432

#### B. SAGEV2Core (`sage_v2.py`)
- **Enhancement**: Adds external LLM integration, meaningful context encoding
- **Architecture**: H-module with halt predictor + L-module with output head + bidirectional communication
- **Memory**: Maintains memory bank for temporal context (last 3 states)

```python
class SAGEV2Core(nn.Module):
    def forward(self, input_grid, target_grid=None, num_rounds=3):
        # Get LLM understanding if available
        llm_context = self.llm.understand_pattern(input_grid)

        # Get temporal context from memory
        temporal_context = [m['h_state'] for m in self.memory_bank[-3:]]

        # Initial H-module processing
        h_output = self.h_module(input_grid, llm_context, temporal_context)

        # Iterative refinement through H↔L communication
        for round_num in range(num_rounds):
            h_guidance, l_feedback = self.communication(h_state, l_state, round_num)
            l_output = self.l_module(input_features, h_guidance)

            # Update states
            l_state = l_output['hidden']
            if round_num < num_rounds - 1:
                h_output = self.h_module(input_grid, llm_context, l_feedback)
                h_state = h_output['hidden']

        # Store in memory for future temporal context
        self.memory_bank.append({'input': input_grid, 'h_state': h_state, 'l_state': l_state})

        return output
```

**Location**: `/home/dp/ai-workspace/HRM/sage/core/sage_v2.py` lines 382-508

#### C. HRMOrchestrator (`hrm_orchestrator.py`)
- **Type**: Async orchestrator for IRP plugins
- **Execution Model**: Event-driven parallel execution with ATP budgeting
- **Key Features**:
  - Trust-weighted budget allocation
  - Dynamic resource reallocation
  - Early stopping with budget reclaim

```python
class HRMOrchestrator:
    async def process_async(self, inputs: Dict[str, Any]):
        """Process inputs asynchronously across plugins"""

        # Allocate initial budgets based on trust weights
        budgets = self.allocate_budgets(self.total_ATP)

        # Create futures for parallel execution
        futures = {}
        for name, plugin in self.plugins.items():
            if name in inputs:
                future = loop.run_in_executor(
                    self.executor, self.run_plugin,
                    name, plugin, inputs[name], budgets[name]
                )
                futures[name] = future

        # Collect results as they complete
        while futures:
            done, pending = await asyncio.wait(futures.values(), return_when=asyncio.FIRST_COMPLETED)

            for future in done:
                result = await future
                # Reallocate freed budget to active plugins
                freed_ATP = budgets[plugin_name] - result.budget_used
                if freed_ATP > 0 and active_plugins:
                    additional = self.reallocate_budget(freed_ATP, active_plugins)
                    # Distribute to still-running plugins

        # Integrate results (H-module function)
        integrated = self.integrate_results(results)

        # Update trust weights based on performance
        self.update_trust_weights(results, integrated)

        return integrated
```

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestrator/hrm_orchestrator.py` lines 251-330

#### D. Awareness Loop (`awareness_loop.py`)
- **Type**: Bidirectional audio conversation loop
- **Closest to specification's continuous loop**
- **Integration**: Audio as first-class sensory stream

```python
class SproutAwarenessLoop:
    async def run(self):
        """Main awareness loop - continuous bidirectional conversation"""

        # Welcome message
        await self.speak("Hello! I'm ready to chat.")

        try:
            while True:
                # 1. Listen for user speech (AudioInputIRP)
                listen_result = await self.listen()

                if listen_result is None:
                    continue  # Silence, keep listening

                user_text = listen_result['text']

                # 2. Check for exit command
                if any(word in user_text.lower() for word in ['bye', 'goodbye', 'exit']):
                    await self.speak("Goodbye!")
                    break

                # 3. Process through SAGE consciousness
                response = self.process_with_sage(user_text)

                # 4. Speak response (NeuTTSAirIRP)
                await self.speak(response)

                # Brief pause between turns
                await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            await self.speak("Goodbye! Talk to you soon.")
```

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/awareness_loop.py` lines 336-362

---

## 2. State Management

### 2.1 Temporal State

**NOT IMPLEMENTED** as specified in SAGE_CORE_SPECIFICATION.md lines 172-194.

**Partial Implementation** in SAGE V2:
```python
class SAGEV2Core:
    def __init__(self):
        self.memory_bank = []  # Stores temporal context
        self.max_memory_size = 100

    def forward(self, input_grid, ...):
        # Get temporal context from memory
        temporal_context = None
        if len(self.memory_bank) > 0:
            recent_memories = self.memory_bank[-3:]
            temporal_context = [m['h_state'] for m in recent_memories]

        # Use temporal context in H-module
        h_output = self.h_module(input_grid, llm_context=llm_context, temporal_context=temporal_context)

        # Store in memory for future
        self.memory_bank.append({'input': input_grid, 'h_state': h_state, 'l_state': l_state})
```

**Location**: `/home/dp/ai-workspace/HRM/sage/core/sage_v2.py` lines 369-481

### 2.2 Metabolic State System

**FULLY IMPLEMENTED** (`metabolic-state-manager.py`):

#### States:
1. **WAKE** - Normal operation, broad attention
2. **FOCUS** - High performance, narrow attention
3. **REST** - Recovery and maintenance
4. **DREAM** - Consolidation and exploration
5. **CRISIS** - Emergency response mode

#### State Configuration:
```python
@dataclass
class StateConfig:
    energy_consumption_rate: float  # Energy per time unit
    attention_breadth: int          # Number of simultaneous focuses
    surprise_sensitivity: float     # Threshold for surprise detection
    exploration_rate: float         # Random exploration probability
    max_duration: float             # Maximum time in this state
    transition_conditions: Dict     # Conditions for state transitions
```

#### State Transitions:
```python
class MetabolicStateManager:
    def _check_transitions(self):
        """Check and execute state transitions"""
        config = self.state_configs[self.current_state]

        # Check max duration
        if time_in_state > config.max_duration:
            self._transition_to(MetabolicState.WAKE, "max_duration_exceeded")
            return

        # Check transition conditions
        # Example: WAKE → FOCUS when task_performance > 0.7
        # Example: WAKE → REST when energy < 0.3
        # Example: * → CRISIS when surprise > 0.8
        for transition_name, condition in config.transition_conditions.items():
            if condition(self.current_context):
                target_state = MetabolicState[target_state_name]
                self._transition_to(target_state, transition_name)
                break
```

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestration/agents/control/metabolic-state-manager.py` lines 86-363

#### Energy Management:
```python
class EnergyManager:
    def __init__(self, initial_energy=100.0, recharge_rate=5.0):
        self.current_energy = initial_energy
        self.max_energy = initial_energy
        self.recharge_rate = recharge_rate

    def consume(self, amount: float) -> bool:
        """Consume energy if available"""
        if self.current_energy >= amount:
            self.current_energy -= amount
            return True
        return False

    def recharge(self, delta_time: float):
        """Recharge energy over time"""
        recharge_amount = self.recharge_rate * delta_time
        self.current_energy = min(self.max_energy, self.current_energy + recharge_amount)
```

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestration/agents/control/metabolic-state-manager.py` lines 39-83

### 2.3 SNARC Memory Management

**FULLY IMPLEMENTED** (`snarc_scorer.py`):

#### Components:
```python
class SNARCScorer(nn.Module):
    """SNARC scoring system for attention prioritization

    Each component evaluates different aspects of salience:
    - Surprise: Deviation from expected patterns
    - Novelty: Presence of unseen patterns
    - Arousal: Complexity and information density
    - Reward: Task completion and success signals
    - Conflict: Ambiguity and uncertainty
    """

    def __init__(self, hidden_size=768, memory_size=1000):
        # Memory bank for novelty assessment
        self.memory_bank = deque(maxlen=memory_size)

        # Learnable components for each SNARC dimension
        self.surprise_net = nn.Sequential(...)
        self.novelty_net = nn.Sequential(...)
        self.arousal_net = nn.Sequential(...)
        self.conflict_net = nn.Sequential(...)

        # Prediction network for surprise computation
        self.predictor = nn.Sequential(...)

        # Attention weighting network
        self.attention_weight = nn.Sequential(nn.Linear(5, 16), ...)
```

#### SNARC Computation:
```python
def forward(self, input_states, context=None, task_success=None):
    """Compute full SNARC scores"""

    # Compute individual SNARC components
    surprise = self.compute_surprise(input_states, context)
    novelty = self.compute_novelty(input_states)
    arousal = self.compute_arousal(input_states)
    conflict = self.compute_conflict(input_states)
    reward = self.compute_reward(task_success)

    # Stack SNARC components
    snarc_stack = torch.stack([surprise, novelty, arousal, reward, conflict], dim=-1)

    # Compute attention weights based on SNARC
    attention_weights = self.attention_weight(snarc_stack)

    # Combined SNARC score (weighted average)
    weights = F.softmax(torch.tensor([1.0, 0.8, 0.6, 1.2, 0.7]), dim=0)
    snarc_scores = (snarc_stack * weights).mean(dim=-1, keepdim=True)

    # Update memory bank with current states
    self.update_memory(input_states)

    return {
        'snarc_scores': snarc_scores,
        'attention_weights': attention_weights,
        'surprise': surprise,
        'novelty': novelty,
        'arousal': arousal,
        'reward': reward,
        'conflict': conflict
    }
```

#### Attention Biasing:
```python
def bias_attention(self, attention_scores, snarc_weights, bias_strength=0.5):
    """Bias attention scores based on SNARC weights"""
    # Expand SNARC weights to match attention dimensions
    snarc_weights = snarc_weights.squeeze(-1).unsqueeze(1).unsqueeze(2)
    snarc_weights = snarc_weights.expand(batch_size, num_heads, 1, seq_len)

    # Apply bias
    bias = snarc_weights * bias_strength
    biased_attention = attention_scores + bias

    return biased_attention
```

**Location**: `/home/dp/ai-workspace/HRM/sage/attention/snarc_scorer.py` lines 16-335

---

## 3. Decision Algorithms

### 3.1 Attention Computation

#### SNARC-Based Attention:
```python
class SNARCScorer:
    def compute_surprise(self, input_states, context=None):
        """Compute surprise as deviation from predicted patterns"""
        # Use previous states to predict current
        predictions = self.predictor(input_states[:, :-1])
        actuals = input_states[:, 1:]

        # Compute prediction error as surprise
        surprise = F.mse_loss(predictions, actuals, reduction='none').mean(dim=-1, keepdim=True)
        return surprise

    def compute_novelty(self, input_states):
        """Compute novelty by comparing to memory bank"""
        # Convert memory bank to tensor
        memory_tensor = torch.stack(list(self.memory_bank))

        # Compute similarity to memory
        similarities = F.cosine_similarity(input_flat.unsqueeze(1), memory_tensor.unsqueeze(0), dim=-1)

        # Novelty is inverse of maximum similarity
        max_similarity = similarities.max(dim=-1)[0]
        novelty = 1.0 - max_similarity

        return novelty

    def compute_arousal(self, input_states):
        """Compute arousal as complexity/information density"""
        # Compute entropy as proxy for complexity
        probabilities = F.softmax(input_states, dim=-1)
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=-1, keepdim=True)

        # Normalize entropy
        normalized_entropy = entropy / max_entropy
        return normalized_entropy

    def get_top_k_salient(self, snarc_scores, k=10):
        """Get top-k most salient positions"""
        snarc_flat = snarc_scores.squeeze(-1)
        top_scores, top_indices = torch.topk(snarc_flat, k=min(k, snarc_flat.size(1)), dim=1)
        return top_indices, top_scores
```

**Location**: `/home/dp/ai-workspace/HRM/sage/attention/snarc_scorer.py` lines 80-335

### 3.2 Resource Selection

#### ATP-Based Budget Allocation:
```python
class HRMOrchestrator:
    def allocate_budgets(self, available_ATP: float) -> Dict[str, float]:
        """Allocate ATP budget across plugins based on trust weights"""

        # Normalize trust weights
        total_trust = sum(self.trust_weights.values())

        # Proportional allocation with minimum guarantee
        min_ATP = available_ATP * 0.05  # 5% minimum per plugin
        budgets = {}

        for name, plugin in self.plugins.items():
            weight = self.trust_weights[name] / total_trust
            allocated = available_ATP * weight
            budgets[name] = max(min_ATP, allocated)

        # Ensure we don't exceed total budget
        total_allocated = sum(budgets.values())
        if total_allocated > available_ATP:
            scale = available_ATP / total_allocated
            budgets = {name: budget * scale for name, budget in budgets.items()}

        return budgets

    def reallocate_budget(self, freed_ATP: float, active_plugins: List[str]):
        """Reallocate freed budget to active plugins"""
        # Get trust weights for active plugins
        active_weights = {name: self.trust_weights[name] for name in active_plugins}

        total_weight = sum(active_weights.values())

        # Proportional distribution
        additional_budgets = {}
        for name, weight in active_weights.items():
            share = weight / total_weight
            additional_budgets[name] = freed_ATP * share

        return additional_budgets
```

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestrator/hrm_orchestrator.py` lines 137-203

#### Resource Routing in SAGE Core:
```python
class SAGECore(nn.Module):
    def forward(self, input_ids, context=None):
        # ... H↔L reasoning cycles ...

        # Resource allocation decision (based on H-module strategy)
        resource_allocation = self.resource_router(strategy)
        resource_allocation = F.softmax(resource_allocation, dim=-1)

        return {
            'resource_allocation': resource_allocation,  # [batch, num_resources]
            'strategy': strategy,  # Strategic guidance from H-module
            ...
        }
```

**Location**: `/home/dp/ai-workspace/HRM/sage/core/sage_core.py` lines 417-419

### 3.3 Trust Dynamics

#### Trust Update in HRM Orchestrator:
```python
class HRMOrchestrator:
    def update_trust_weights(self, results: Dict[str, PluginResult], integrated: Dict[str, Any]):
        """Update trust weights based on plugin performance"""
        system_coherence = integrated['system_coherence']

        for name, result in results.items():
            # Get trust metrics from telemetry
            trust_metrics = result.telemetry.get('trust', {})

            # Base trust from convergence quality
            monotonicity = trust_metrics.get('monotonicity_ratio', 0.5)

            # Contribution to system coherence
            contribution = trust_metrics.get('contribution_to_H', 0.0)
            system_modifier = self._sigmoid(contribution / (system_coherence + 1e-6))

            # Efficiency bonus
            budget_ratio = result.budget_used / self.plugins[name].config.get('max_ATP', 10.0)
            efficiency = 1.0 - min(budget_ratio, 1.0)

            # Update with momentum
            old_trust = self.trust_weights[name]
            new_trust = (
                0.7 * old_trust +
                0.2 * monotonicity * system_modifier +
                0.1 * efficiency
            )

            # Apply update with learning rate
            self.trust_weights[name] = (
                (1 - self.trust_update_rate) * old_trust +
                self.trust_update_rate * new_trust
            )

            # Clamp to valid range
            self.trust_weights[name] = np.clip(self.trust_weights[name], 0.1, 10.0)
```

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/orchestrator.py` lines 408-450

---

## 4. IRP Integration

### 4.1 IRP Base Contract

```python
class IRPPlugin:
    """Base class for all IRP implementations"""

    # Core IRP Contract (must override)

    def init_state(self, x0: Any, task_ctx: Dict) -> IRPState:
        """Initialize refinement state from input"""
        raise NotImplementedError

    def energy(self, state: IRPState) -> float:
        """Compute energy/distance metric for current state (lower is better)"""
        raise NotImplementedError

    def step(self, state: IRPState, noise_schedule=None) -> IRPState:
        """Execute one refinement iteration"""
        raise NotImplementedError

    # Optional overrides

    def project(self, state: IRPState) -> IRPState:
        """Enforce constraints on state (dynamics/safety/feasibility)"""
        return state

    def halt(self, history: List[IRPState]) -> bool:
        """Determine if refinement should stop"""
        eps = self.config.get('halt_eps', 1e-4)
        K = self.config.get('halt_K', 3)

        if len(history) >= self.config.get('max_iterations', 100):
            return True

        # Check energy slope over last K steps
        recent_energies = [s.energy_val or self.energy(s) for s in history[-(K+1):]]
        slope = abs(recent_energies[-1] - recent_energies[0]) / len(recent_energies)

        return slope < eps
```

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/base.py` lines 24-141

### 4.2 Refinement Loop

```python
class IRPPlugin:
    def refine(self, input_data: Any, task_ctx: Dict = None, early_stop: bool = True) -> Tuple[IRPState, List[IRPState]]:
        """Main refinement loop"""

        # Initialize state
        state = self.init_state(input_data, task_ctx or {})
        state.energy_val = self.energy(state)
        history = [state]

        # Iterative refinement
        max_iterations = self.config.get('max_iterations', 100)
        for i in range(max_iterations):
            # Check halt condition
            if early_stop and self.halt(history):
                break

            # Refinement step
            state = self.step(state)

            # Constraint projection
            state = self.project(state)

            # Update energy
            state.energy_val = self.energy(state)
            state.step_idx = i + 1

            history.append(state)

        return state, history
```

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/base.py` lines 152-194

### 4.3 IRP Orchestration

The HRMOrchestrator manages multiple IRP plugins concurrently:

```python
class HRMOrchestrator:
    async def execute_plugin(self, plugin_id: str, input_data: Any, early_stop: bool = True):
        """Execute a single plugin asynchronously"""
        plugin = self.plugins[plugin_id]

        # Run plugin refinement in executor
        loop = asyncio.get_event_loop()
        output, telemetry = await loop.run_in_executor(
            None, plugin.refine, input_data, early_stop
        )

        # Calculate ATP consumption (based on iterations)
        iterations = telemetry.get('iterations', 1)
        atp_consumed = iterations * 10.0  # Base cost per iteration

        # Check budget
        if not self.budget.consume(plugin_id, atp_consumed):
            state = PluginState.FAILED
        else:
            state = PluginState.COMPLETED

        return PluginResult(
            plugin_id=plugin_id,
            state=state,
            output=output,
            telemetry=telemetry,
            atp_consumed=atp_consumed,
            trust_score=telemetry.get('trust', 0.5)
        )

    async def execute_parallel(self, tasks: Dict[str, Any], early_stop: bool = True):
        """Execute multiple plugins in parallel"""
        execution_tasks = []

        for plugin_id, input_data in tasks.items():
            task = asyncio.create_task(self.execute_plugin(plugin_id, input_data, early_stop))
            execution_tasks.append(task)
            self.running_tasks[plugin_id] = task

        # Start reallocation monitor
        reallocation_task = asyncio.create_task(self._reallocation_monitor())

        # Wait for all plugins to complete
        results = await asyncio.gather(*execution_tasks)

        return results
```

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestrator/hrm_orchestrator.py` lines 173-268

---

## 5. Key Data Structures

### 5.1 IRPState
```python
@dataclass
class IRPState:
    """State container for IRP refinement process"""
    x: Any                      # Plugin-specific state (latent, tokens, trajectory, etc.)
    step_idx: int = 0           # Current step in refinement
    energy_val: Optional[float] = None  # Energy/distance metric
    meta: Dict[str, Any] = field(default_factory=dict)  # Metadata
    timestamp: float = field(default_factory=time.time)
```

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/base.py` lines 14-21

### 5.2 PluginResult
```python
@dataclass
class PluginResult:
    """Result from an IRP plugin execution"""
    plugin_name: str
    final_state: IRPState
    history: List[IRPState]
    telemetry: Dict[str, Any]
    budget_used: float
    execution_time: float
```

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/orchestrator.py` lines 24-32

### 5.3 ATPBudget
```python
@dataclass
class ATPBudget:
    """Adaptive Trust Points budget management"""
    total: float = 1000.0
    allocated: Dict[str, float] = field(default_factory=dict)  # Per-plugin allocation
    consumed: Dict[str, float] = field(default_factory=dict)   # Per-plugin consumption
    trust_weights: Dict[str, float] = field(default_factory=dict)  # Per-plugin trust

    def allocate(self, plugin_id: str, trust_weight: float) -> float:
        """Allocate ATP based on trust weight"""
        ...

    def consume(self, plugin_id: str, amount: float) -> bool:
        """Consume ATP, return False if budget exceeded"""
        ...

    def reallocate_unused(self):
        """Reallocate unused ATP from completed plugins"""
        ...
```

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestrator/hrm_orchestrator.py` lines 55-122

### 5.4 MetabolicState
```python
class MetabolicState(Enum):
    """The five metabolic states"""
    WAKE = "WAKE"      # Normal operation, broad attention
    FOCUS = "FOCUS"    # High performance on specific task
    REST = "REST"      # Recovery and maintenance
    DREAM = "DREAM"    # Consolidation and exploration
    CRISIS = "CRISIS"  # Emergency response mode

@dataclass
class StateConfig:
    """Configuration for a metabolic state"""
    name: MetabolicState
    energy_consumption_rate: float
    attention_breadth: int
    surprise_sensitivity: float
    exploration_rate: float
    max_duration: float
    transition_conditions: Dict[str, Any]
```

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestration/agents/control/metabolic-state-manager.py` lines 18-36

### 5.5 SAGEConfig
```python
@dataclass
class SAGEConfig:
    """Configuration for SAGE (100M parameters)"""

    # Core architecture
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12

    # Layer configuration
    num_h_layers: int = 7  # Deep strategic reasoning
    num_l_layers: int = 7  # Deep tactical execution

    # Context and interaction
    context_dim: int = 256
    snarc_dim: int = 5  # Surprise, Novelty, Arousal, Reward, Conflict
    num_reasoning_cycles: int = 8

    # Resource management
    resource_types: List[str] = ['llm', 'vision', 'memory', 'time', 'effector']
    max_resource_calls: int = 10

    # Training configuration
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
```

**Location**: `/home/dp/ai-workspace/HRM/sage/core/sage_config.py` lines 14-93

---

## 6. Key Classes and Responsibilities

### 6.1 SAGECore
- **Type**: PyTorch neural network (100M parameters)
- **Responsibility**: H↔L bidirectional reasoning with resource routing
- **Key Methods**:
  - `forward()` - Iterative reasoning cycles with early stopping
  - `get_num_params()` - Parameter counting
- **Architecture**:
  - H-module (45M) - Strategic reasoning
  - L-module (45M) - Tactical execution
  - Bidirectional communication (10M) - H↔L interaction
  - Halt predictor - Confidence-based early stopping
  - Resource router - Decides what external resources to invoke

**Location**: `/home/dp/ai-workspace/HRM/sage/core/sage_core.py`

### 6.2 HRMOrchestrator
- **Type**: Async orchestrator
- **Responsibility**: Coordinate multiple IRP plugins with ATP budgeting
- **Key Methods**:
  - `process_async()` - Main async orchestration loop
  - `allocate_budgets()` - Trust-weighted ATP allocation
  - `reallocate_budget()` - Dynamic budget reallocation
  - `update_trust_weights()` - Trust learning from performance
  - `integrate_results()` - Combine plugin outputs (H-module function)
- **Features**:
  - Parallel plugin execution
  - Early stopping with budget reclaim
  - Trust-based resource allocation
  - System coherence computation

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestrator/hrm_orchestrator.py`

### 6.3 IRPPlugin (Base)
- **Type**: Abstract base class
- **Responsibility**: Define contract for iterative refinement primitives
- **Key Methods**:
  - `init_state()` - Initialize refinement from input
  - `energy()` - Compute quality metric
  - `step()` - Execute one refinement iteration
  - `halt()` - Determine convergence
  - `refine()` - Main refinement loop
  - `emit_telemetry()` - Generate performance metrics
- **Subclasses**: VisionIRP, LanguageIRP, ControlIRP, MemoryIRP, NeuTTSAirIRP, AudioInputIRP

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/base.py`

### 6.4 MetabolicStateManager
- **Type**: Threaded state machine
- **Responsibility**: Manage operational mode transitions
- **Key Methods**:
  - `start()` - Begin state management thread
  - `_update_loop()` - Main update loop
  - `_check_transitions()` - Evaluate state change conditions
  - `submit_event()` - Process external events
  - `get_status()` - Query current state
- **Features**:
  - Energy consumption and recharge
  - Automatic state transitions based on context
  - Configurable transition conditions
  - Event-driven context updates

**Location**: `/home/dp/ai-workspace/HRM/sage/orchestration/agents/control/metabolic-state-manager.py`

### 6.5 SNARCScorer
- **Type**: PyTorch neural network
- **Responsibility**: Compute attention salience scores
- **Key Methods**:
  - `forward()` - Compute full SNARC scores
  - `compute_surprise()` - Prediction error detection
  - `compute_novelty()` - Memory-based novelty assessment
  - `compute_arousal()` - Information density computation
  - `compute_conflict()` - Uncertainty quantification
  - `bias_attention()` - Modify attention based on SNARC
  - `get_top_k_salient()` - Extract most important positions
- **Components**: 5 SNARC dimensions (Surprise, Novelty, Arousal, Reward, Conflict)
- **Memory**: Maintains memory bank (1000 states) for novelty comparison

**Location**: `/home/dp/ai-workspace/HRM/sage/attention/snarc_scorer.py`

### 6.6 SproutAwarenessLoop
- **Type**: Async conversation loop
- **Responsibility**: Bidirectional audio interaction
- **Key Methods**:
  - `run()` - Main awareness loop
  - `listen()` - Audio input via AudioInputIRP
  - `process_with_sage()` - SAGE consciousness processing
  - `speak()` - Audio output via NeuTTSAirIRP
  - `conversation_turn()` - Single exchange cycle
- **Features**:
  - Continuous listening
  - Context-aware responses
  - Memory-based conversation history
  - Graceful exit handling

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/awareness_loop.py`

---

## 7. Implementation vs Specification Gaps

### 7.1 IMPLEMENTED

✅ **H↔L Bidirectional Reasoning** (SAGECore)
- 100M parameter transformer with iterative cycles
- Strategic (H) and tactical (L) modules
- Bidirectional communication layer
- Resource routing head

✅ **IRP Plugin System** (IRPPlugin + HRMOrchestrator)
- Abstract base class with clear contract
- Multiple domain-specific implementations (Vision, Language, Control, Memory, Audio)
- Iterative refinement with energy-based convergence
- Telemetry and trust scoring

✅ **ATP-Based Resource Management** (ATPBudget + HRMOrchestrator)
- Trust-weighted budget allocation
- Dynamic reallocation of freed resources
- Budget consumption tracking
- Efficiency metrics

✅ **Metabolic State System** (MetabolicStateManager)
- 5 states (WAKE, FOCUS, REST, DREAM, CRISIS)
- Automatic transitions based on energy and context
- Per-state policy modifiers
- Energy consumption and recharge

✅ **SNARC Attention System** (SNARCScorer)
- 5 dimensions (Surprise, Novelty, Arousal, Reward, Conflict)
- Learnable neural components
- Memory-based novelty detection
- Attention biasing mechanism

✅ **Trust Dynamics** (HRMOrchestrator)
- Trust weight updates based on performance
- Monotonicity-based convergence quality
- System coherence contribution
- Efficiency bonuses

✅ **Awareness Loop** (SproutAwarenessLoop)
- Continuous bidirectional conversation
- Audio as first-class sensory stream
- Context-aware processing
- Memory-based conversation history

### 7.2 NOT IMPLEMENTED (From Specification)

❌ **Single Continuous SAGE Loop**
- Specification describes `SAGE.run()` continuous loop
- Actual: Multiple async components without unified loop
- No central `while True` orchestrator as specified

❌ **Temporal State Management**
- Specification: `TemporalState` with cycle count, phase, history buffer
- Actual: Only partial implementation in SAGE V2 memory bank
- Missing: Clock, phase embeddings, temporal encoding

❌ **Resource Registry**
- Specification: `ResourceRegistry` tracking available models
- Actual: Hard-coded plugin initialization in orchestrator
- Missing: Dynamic resource discovery and registration

❌ **Active Resource Loading/Unloading**
- Specification: `manage_resources()` loads/unloads based on need
- Actual: All plugins initialized at orchestrator creation
- Missing: GPU/RAM management, model swapping

❌ **Goal-Directed Planning**
- Specification: `current_goal`, `plan_resources()`
- Actual: No explicit goal representation or planning
- Resource selection is implicit in plugin execution

❌ **Context State Preservation**
- Specification: `context_states` per-component state preservation
- Actual: Limited to memory bank in SAGE V2
- Missing: Persistent state for all components

❌ **Surprise Buffer**
- Specification: `surprise_buffer = deque(maxlen=100)`
- Actual: SNARC computes surprise but doesn't maintain buffer
- Missing: Historical surprise tracking

### 7.3 PARTIALLY IMPLEMENTED

⚠️ **Observation Gathering**
- Specification: `gather_observations(breadth=...)`
- Actual: Plugins have individual `init_state()` but no unified observation
- Gap: No central sensor aggregation

⚠️ **Memory Consolidation**
- Specification: Memory updates with consolidation rate
- Actual: SAGE V2 has memory bank, metabolic state has consolidation rate
- Gap: No actual consolidation algorithm implemented

⚠️ **Action Execution**
- Specification: `execute_actions(results)`
- Actual: IRP plugins can be effectors (e.g., speech, motor)
- Gap: No unified action execution layer

---

## 8. Architecture Insights

### 8.1 Distributed vs Monolithic

**Specification Intent**: Monolithic continuous loop
**Actual Implementation**: Distributed event-driven architecture

**Advantages of Distributed Approach**:
1. **Modularity**: IRP plugins are independent and composable
2. **Concurrency**: Async execution enables parallelism
3. **Flexibility**: Easy to add/remove plugins without rewriting core
4. **Testability**: Components can be tested in isolation
5. **Resource Efficiency**: Only active plugins consume resources

**Disadvantages**:
1. **Complexity**: No single "main loop" to understand
2. **Coordination Overhead**: Async orchestration is harder to reason about
3. **State Fragmentation**: State split across multiple components
4. **Integration Challenge**: Harder to ensure system-wide coherence

### 8.2 H↔L Communication Pattern

The bidirectional H-L architecture is **consistently implemented** across both SAGE neural modules:

```
H-Module (Strategic)          L-Module (Tactical)
       ↓                             ↑
   Strategy ────────────→    L integrates strategy
       ↑                             ↓
   Feedback ←────────────    L provides feedback
```

**Cycle Structure**:
1. H processes context → generates strategy
2. Strategy sent to L
3. L executes tactics → generates actions
4. L feedback sent back to H
5. H refines strategy based on feedback
6. Repeat for N cycles or until convergence

This mirrors the specification's intent for strategic-tactical dialogue.

### 8.3 Trust as Universal Currency

**Trust is used consistently across all components**:

1. **ATP Allocation**: Trust weights determine budget distribution
2. **Resource Selection**: High-trust plugins get invoked first
3. **Result Integration**: Trust-weighted combination of outputs
4. **Convergence Quality**: Monotonic energy decrease increases trust
5. **System Coherence**: Contribution to overall coherence affects trust

**Trust Update Formula** (HRMOrchestrator):
```python
new_trust = 0.7 * old_trust + 0.2 * monotonicity * system_modifier + 0.1 * efficiency
trust = (1 - learning_rate) * old_trust + learning_rate * new_trust
trust = clamp(trust, 0.1, 10.0)
```

### 8.4 Energy as Convergence Metric

**Energy is the core metric for all IRP plugins**:

- **Vision**: Reconstruction error in latent space
- **Language**: Token probability / perplexity
- **Control**: Distance to goal in state space
- **Memory**: Retrieval confidence / relevance

**Convergence Criterion**:
```python
halt = (abs(energy[-1] - energy[0]) / K) < eps
```

Where:
- `K` = number of recent steps to check (typically 3)
- `eps` = energy slope threshold (typically 1e-4)

This provides a universal stopping criterion across all domains.

### 8.5 Metabolic State as Global Policy

Metabolic states act as **global policy modifiers** that affect all components:

| State  | Energy | Breadth | Depth | Exploration | Consolidation |
|--------|--------|---------|-------|-------------|---------------|
| WAKE   | 10/s   | 5       | 10    | 0.3         | 0.1           |
| FOCUS  | 5/s    | 2       | 5     | 0.0         | 0.05          |
| REST   | 2/s    | 1       | 2     | -0.5        | 0.5           |
| DREAM  | 5/s    | 10      | 3     | 0.5         | 0.3           |
| CRISIS | 20/s   | 3       | 15    | 0.0         | 0.0           |

**Transition Logic**:
- High performance → FOCUS (narrow attention, exploit)
- Low energy → REST (consolidate, recharge)
- High surprise → CRISIS (emergency response)
- Post-rest → DREAM (explore, consolidate patterns)
- Default → WAKE (balanced exploration/exploitation)

---

## 9. Integration Points

### 9.1 SAGE Core ↔ IRP System

**Current Integration**: Implicit through resource_allocation output

```python
# SAGECore produces resource allocation
outputs = sage_core(input_ids, context)
resource_allocation = outputs['resource_allocation']  # [batch, num_resources]

# Orchestrator could use this to prioritize plugins
orchestrator.execute_parallel(tasks, priorities=resource_allocation)
```

**Gap**: No actual connection implemented. SAGE Core doesn't invoke IRP plugins.

**Ideal Integration**:
```python
class SAGESystem:
    def __init__(self):
        self.core = SAGECore(config)
        self.orchestrator = HRMOrchestrator()

    async def process(self, input_data):
        # 1. SAGE Core determines resource allocation
        core_output = self.core(input_data)
        resource_allocation = core_output['resource_allocation']

        # 2. Orchestrator executes prioritized plugins
        tasks = self.build_tasks(input_data, resource_allocation)
        plugin_results = await self.orchestrator.execute_parallel(tasks)

        # 3. Integrate plugin results back into SAGE state
        integrated = self.orchestrator.integrate_results(plugin_results)

        return integrated
```

### 9.2 Metabolic State ↔ Orchestrator

**Current Integration**: None

**Gap**: MetabolicStateManager runs independently, doesn't affect orchestrator

**Ideal Integration**:
```python
class MetabolicAwareOrchestrator(HRMOrchestrator):
    def __init__(self, metabolic_manager: MetabolicStateManager):
        super().__init__()
        self.metabolic = metabolic_manager

    async def process_async(self, inputs):
        # Get current state configuration
        state_config = self.metabolic.get_state_config()

        # Adjust ATP budget based on metabolic state
        available_ATP = self.total_ATP * state_config.resource_limit

        # Allocate budgets
        budgets = self.allocate_budgets(available_ATP)

        # Filter plugins based on attention breadth
        active_plugins = self.select_plugins(inputs, state_config.attention_breadth)

        # Execute with metabolic constraints
        results = await self.execute_parallel(active_plugins)

        # Update metabolic state based on results
        self.metabolic.submit_event({
            'type': 'performance',
            'value': results['system_coherence']
        })

        return results
```

### 9.3 SNARC ↔ Attention

**Current Integration**: SNARC can bias attention in transformers

```python
class SAGECoreWithSNARC(SAGECore):
    def __init__(self, config):
        super().__init__(config)
        self.snarc = SNARCScorer(hidden_size=config.hidden_size)

    def forward(self, input_ids, context=None):
        embeddings = self.input_embedding(input_ids) + self.position_embedding(position_ids)

        # Compute SNARC scores
        snarc_results = self.snarc(embeddings, context, return_components=True)
        attention_weights = snarc_results['attention_weights']

        # Initialize states with SNARC-biased attention
        h_states = embeddings
        l_states = embeddings

        for cycle in range(num_cycles):
            # H-module with SNARC-biased attention
            h_states, strategy = self.h_module(
                h_states, context,
                attention_bias=attention_weights
            )

            # L-module execution
            l_states, actions = self.l_module(l_states, strategy)

            # Bidirectional communication
            h_states, l_states = self.communication(h_states, l_states)

        return {...}
```

### 9.4 Awareness Loop ↔ SAGE Consciousness

**Current Integration**: Awareness loop calls `process_with_sage()` but it's a stub

```python
def process_with_sage(self, user_input: str) -> str:
    """Process user input through SAGE consciousness"""
    # Simple response logic (will be replaced with SAGE integration)
    if 'hello' in user_input.lower():
        return "Hello! I can hear you clearly."
    # ... more hardcoded responses ...
```

**Ideal Integration**:
```python
class SAGEAwarenessLoop(SproutAwarenessLoop):
    def __init__(self, config, sage_system: SAGESystem):
        super().__init__(config)
        self.sage = sage_system

    def process_with_sage(self, user_input: str) -> str:
        """Process through actual SAGE system"""

        # 1. Encode user input
        input_data = {
            'language': self.encode_text(user_input),
            'audio': self.last_audio_features,
            'context': self.conversation_context
        }

        # 2. Run SAGE processing
        sage_output = asyncio.run(self.sage.process(input_data))

        # 3. Extract response
        response_text = self.decode_language_output(
            sage_output['plugin_outputs']['language']
        )

        # 4. Update conversation context
        self.conversation_context = sage_output['h_hidden']

        return response_text
```

---

## 10. Recommendations

### 10.1 Critical Missing Pieces

1. **Unified SAGE Loop** (`sage_system.py`)
   - Create single entry point that orchestrates all components
   - Implement continuous `run()` loop as specified
   - Integrate SAGECore, HRMOrchestrator, MetabolicStateManager, SNARCScorer

2. **Temporal State Management** (`temporal_state.py`)
   - Implement `TemporalState` class from specification
   - Add cycle counting, phase embeddings, history buffer
   - Provide temporal encoding for all components

3. **Resource Registry** (`resource_registry.py`)
   - Dynamic plugin discovery and registration
   - Model availability tracking (GPU/RAM requirements)
   - Loading/unloading based on metabolic state constraints

4. **Goal Representation** (`goal_manager.py`)
   - Explicit goal data structure
   - Goal-driven resource planning
   - Progress tracking and goal satisfaction

### 10.2 Integration Priorities

1. **High Priority**: Connect SAGECore to IRP plugins
   - SAGECore's `resource_allocation` should drive orchestrator
   - Plugin results should feed back into SAGE state
   - Test on simple task (e.g., ARC puzzle solving)

2. **High Priority**: Integrate MetabolicStateManager with orchestrator
   - State-dependent ATP budgets
   - Attention breadth filtering
   - Performance-driven state transitions

3. **Medium Priority**: Connect SNARC to SAGE attention
   - Bias H-module attention with SNARC weights
   - Use SNARC for resource prioritization
   - Track surprise history for crisis detection

4. **Medium Priority**: Complete Awareness Loop integration
   - Replace stub `process_with_sage()` with actual SAGE invocation
   - Use conversation context for memory retrieval
   - Generate context-aware responses

### 10.3 Testing Strategy

1. **Unit Tests**: Each component in isolation
   - SAGECore forward pass
   - IRP plugin refinement loops
   - SNARC score computation
   - Metabolic state transitions

2. **Integration Tests**: Component pairs
   - SAGECore + HRMOrchestrator
   - MetabolicStateManager + HRMOrchestrator
   - SNARCScorer + SAGECore attention
   - Awareness Loop + SAGE system

3. **System Tests**: Full SAGE loop
   - Continuous inference loop
   - Multi-modal task (vision + language + audio)
   - Metabolic state transitions during execution
   - Memory consolidation across cycles

---

## 11. Conclusion

### Summary of Findings

**SAGE is architecturally complete but functionally fragmented**:

✅ **Well-Implemented Components**:
- 100M parameter H↔L transformer (SAGECore)
- Async IRP orchestration with ATP budgeting
- Metabolic state system with energy management
- SNARC-based attention prioritization
- Trust dynamics and learning

❌ **Missing Integration**:
- No unified continuous loop
- Components run independently
- No central state management
- Resource allocation is symbolic, not enacted

⚠️ **Partially Implemented**:
- Temporal state (memory bank only)
- Goal representation (implicit in tasks)
- Resource management (static plugin initialization)

### Path Forward

To realize the SAGE specification:

1. **Create `SAGESystem` class** that integrates:
   - SAGECore (neural reasoning)
   - HRMOrchestrator (resource execution)
   - MetabolicStateManager (adaptive policy)
   - SNARCScorer (attention allocation)
   - TemporalState (continuity across cycles)

2. **Implement continuous `run()` loop**:
   ```python
   async def run(self):
       while True:
           observations = await self.gather_observations()
           self.temporal_state.tick()
           attention = self.snarc.forward(observations)
           core_output = self.core(observations, attention)
           resources = core_output['resource_allocation']
           results = await self.orchestrator.execute_parallel(self.plan_tasks(resources))
           self.update_state(results)
           await self.execute_actions(results)
   ```

3. **Test end-to-end** on multi-modal tasks:
   - ARC puzzle solving (vision + reasoning)
   - Conversation (audio + language + memory)
   - Robotic manipulation (vision + control + planning)

### Implementation Quality

**Code Quality**: High
- Clean abstractions (IRP base class)
- Type hints and dataclasses
- Async/await for concurrency
- Modular plugin architecture

**Documentation**: Medium
- Comprehensive specification documents
- Inline docstrings for most methods
- Missing: High-level integration guide

**Completeness**: Medium
- Individual components are feature-complete
- Integration between components is missing
- Specification describes ideal, implementation is pragmatic

---

## Appendix: File Locations

### Core Files
- `/home/dp/ai-workspace/HRM/sage/core/sage_core.py` - 100M parameter transformer
- `/home/dp/ai-workspace/HRM/sage/core/sage_v2.py` - Enhanced SAGE with LLM integration
- `/home/dp/ai-workspace/HRM/sage/core/sage_config.py` - Configuration dataclass

### Orchestration
- `/home/dp/ai-workspace/HRM/sage/orchestrator/hrm_orchestrator.py` - ATP-based async orchestrator
- `/home/dp/ai-workspace/HRM/sage/irp/orchestrator.py` - Alternative IRP orchestrator

### IRP System
- `/home/dp/ai-workspace/HRM/sage/irp/base.py` - IRP base class and contract
- `/home/dp/ai-workspace/HRM/sage/irp/vision.py` - Vision IRP
- `/home/dp/ai-workspace/HRM/sage/irp/language.py` - Language IRP
- `/home/dp/ai-workspace/HRM/sage/irp/control.py` - Control IRP
- `/home/dp/ai-workspace/HRM/sage/irp/memory.py` - Memory IRP

### Metabolic & Attention
- `/home/dp/ai-workspace/HRM/sage/orchestration/agents/control/metabolic-state-manager.py`
- `/home/dp/ai-workspace/HRM/sage/attention/snarc_scorer.py`

### Awareness Loop
- `/home/dp/ai-workspace/HRM/sage/irp/awareness_loop.py`

### Specifications
- `/home/dp/ai-workspace/HRM/sage/SAGE_CORE_SPECIFICATION.md`
- `/home/dp/ai-workspace/HRM/sage/METABOLIC_STATES_SPECIFICATION.md`

---

**Analysis Complete**
For questions or clarifications, refer to the actual source files listed above.
