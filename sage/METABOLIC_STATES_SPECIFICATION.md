# SAGE Metabolic States Specification

**Date**: October 6, 2025  
**Version**: 1.0  
**Purpose**: Define metabolic states as policy contexts for SAGE

## Core Concept

Metabolic states represent different operational modes that fundamentally change how SAGE allocates resources, processes information, and makes decisions. Like biological organisms, SAGE transitions between states based on energy availability, task demands, and environmental conditions.

## Metabolic State Definitions

```python
@dataclass
class MetabolicState:
    """Base class for all metabolic states"""
    name: str
    energy_consumption_rate: float  # ATP/cycle
    resource_limit: float  # Max GB of active resources
    inference_depth: int  # How many reasoning steps
    attention_breadth: int  # How many modalities to track
    memory_consolidation_rate: float  # How often to consolidate
    trust_decay_rate: float  # How fast trust decays without reinforcement
    surprise_sensitivity: float  # How reactive to unexpected events
```

### 1. WAKE State (Active Exploration)
```python
class WakeState(MetabolicState):
    """High energy, broad attention, active learning"""
    
    def __init__(self):
        super().__init__(
            name="WAKE",
            energy_consumption_rate=10.0,  # High burn
            resource_limit=6.0,  # Can load many models
            inference_depth=10,  # Deep reasoning
            attention_breadth=5,  # Track many modalities
            memory_consolidation_rate=0.1,  # Rare consolidation
            trust_decay_rate=0.01,  # Slow decay
            surprise_sensitivity=1.0  # Highly reactive
        )
    
    def policy_modifiers(self) -> Dict:
        return {
            'exploration_bonus': 0.3,  # Encourage trying new things
            'resource_loading_threshold': 0.5,  # Load resources eagerly
            'attention_switching_rate': 0.2,  # Switch focus often
            'learning_rate': 0.01  # Fast learning
        }
```

### 2. FOCUS State (Task Execution)
```python
class FocusState(MetabolicState):
    """Moderate energy, narrow attention, goal-directed"""
    
    def __init__(self):
        super().__init__(
            name="FOCUS",
            energy_consumption_rate=5.0,  # Moderate burn
            resource_limit=4.0,  # Fewer models active
            inference_depth=5,  # Moderate reasoning
            attention_breadth=2,  # Track few modalities
            memory_consolidation_rate=0.05,  # Occasional
            trust_decay_rate=0.005,  # Very slow decay
            surprise_sensitivity=0.3  # Less reactive
        )
    
    def policy_modifiers(self) -> Dict:
        return {
            'exploration_bonus': 0.0,  # No exploration
            'resource_loading_threshold': 0.8,  # Only load if necessary
            'attention_switching_rate': 0.05,  # Rarely switch
            'learning_rate': 0.001  # Slow, careful learning
        }
```

### 3. REST State (Recovery)
```python
class RestState(MetabolicState):
    """Low energy, internal focus, consolidation"""
    
    def __init__(self):
        super().__init__(
            name="REST",
            energy_consumption_rate=2.0,  # Low burn
            resource_limit=2.0,  # Minimal models
            inference_depth=2,  # Shallow reasoning
            attention_breadth=1,  # Single modality
            memory_consolidation_rate=0.5,  # Frequent consolidation
            trust_decay_rate=0.02,  # Faster decay
            surprise_sensitivity=0.1  # Mostly ignores surprises
        )
    
    def policy_modifiers(self) -> Dict:
        return {
            'exploration_bonus': -0.5,  # Avoid new things
            'resource_loading_threshold': 0.95,  # Almost never load
            'attention_switching_rate': 0.01,  # Very rarely switch
            'learning_rate': 0.0001  # Minimal learning
        }
```

### 4. DREAM State (Offline Consolidation)
```python
class DreamState(MetabolicState):
    """No external input, replay and consolidation"""
    
    def __init__(self):
        super().__init__(
            name="DREAM",
            energy_consumption_rate=1.0,  # Minimal burn
            resource_limit=3.0,  # Memory systems only
            inference_depth=20,  # Deep internal reasoning
            attention_breadth=0,  # No external attention
            memory_consolidation_rate=1.0,  # Continuous consolidation
            trust_decay_rate=0.0,  # No decay during dreams
            surprise_sensitivity=0.0  # No external surprises
        )
    
    def policy_modifiers(self) -> Dict:
        return {
            'replay_rate': 10.0,  # High replay of experiences
            'pattern_extraction_rate': 1.0,  # Extract patterns
            'memory_pruning_threshold': 0.3,  # Forget low-salience
            'skill_crystallization_rate': 0.1  # Form new skills
        }
```

### 5. CRISIS State (Emergency Response)
```python
class CrisisState(MetabolicState):
    """Maximum energy, narrow focus, immediate action"""
    
    def __init__(self):
        super().__init__(
            name="CRISIS",
            energy_consumption_rate=20.0,  # Maximum burn
            resource_limit=8.0,  # Override limits
            inference_depth=3,  # Fast, shallow decisions
            attention_breadth=1,  # Single critical focus
            memory_consolidation_rate=0.0,  # No consolidation
            trust_decay_rate=0.0,  # Ignore trust
            surprise_sensitivity=2.0  # Hyper-reactive
        )
    
    def policy_modifiers(self) -> Dict:
        return {
            'action_threshold': 0.3,  # Act on low confidence
            'resource_loading_threshold': 0.0,  # Load anything needed
            'reaction_time': 0.1,  # 10x faster responses
            'safety_checks': False  # Bypass safety for speed
        }
```

## State Transition Logic

```python
class MetabolicController:
    """Manages metabolic state transitions"""
    
    def __init__(self):
        self.current_state = WakeState()
        self.energy_level = 100.0  # Current ATP
        self.fatigue = 0.0  # Builds up over time
        self.stress = 0.0  # Environmental pressure
        self.time_in_state = 0
        
    def compute_transition(self, context: SAGEContext) -> MetabolicState:
        """Decide if state transition needed"""
        
        # Crisis detection (highest priority)
        if self.detect_crisis(context):
            return CrisisState()
        
        # Energy-based transitions
        if self.energy_level < 20:
            return RestState()
        
        # Fatigue-based transitions  
        if self.fatigue > 80:
            if context.safe_to_sleep:
                return DreamState()
            else:
                return RestState()
        
        # Task-based transitions
        if context.has_clear_goal and self.current_state.name != "FOCUS":
            if self.energy_level > 50:
                return FocusState()
        
        # Default circadian rhythm
        hour = datetime.now().hour
        if 22 <= hour or hour <= 6:  # Night time
            return RestState()
        elif 6 <= hour <= 8:  # Morning
            return WakeState()
        
        # Stay in current state
        return self.current_state
    
    def detect_crisis(self, context: SAGEContext) -> bool:
        """Detect emergency conditions"""
        return (
            context.surprise > 2.0 or  # Massive surprise
            context.threat_level > 0.8 or  # High threat
            context.goal_urgency > 0.95  # Urgent goal
        )
```

## State-Dependent Policies

```python
class StateDependentSAGE(SAGE):
    """SAGE with metabolic state-dependent behavior"""
    
    def __init__(self):
        super().__init__()
        self.metabolic = MetabolicController()
        
    def run(self):
        """Main loop with metabolic states"""
        while True:
            # Update metabolic state
            new_state = self.metabolic.compute_transition(self.get_context())
            if new_state != self.metabolic.current_state:
                self.transition_to(new_state)
            
            # Apply state-dependent policies
            self.apply_metabolic_modifiers()
            
            # Regular SAGE loop (modified by state)
            observations = self.gather_observations()
            attention = self.compute_attention(
                observations,
                breadth=self.metabolic.current_state.attention_breadth
            )
            
            # Energy consumption
            self.metabolic.energy_level -= self.metabolic.current_state.energy_consumption_rate
            self.metabolic.fatigue += 0.1
            
            # State-dependent reasoning depth
            for _ in range(self.metabolic.current_state.inference_depth):
                results = self.reasoning_step(attention)
                if self.should_halt(results):
                    break
            
            self.execute_actions(results)
    
    def apply_metabolic_modifiers(self):
        """Modify all policies based on metabolic state"""
        modifiers = self.metabolic.current_state.policy_modifiers()
        
        # Adjust attention policy
        self.attention_policy.exploration_bonus = modifiers.get('exploration_bonus', 0)
        
        # Adjust resource policy  
        self.resource_policy.loading_threshold = modifiers.get('resource_loading_threshold', 0.5)
        
        # Adjust learning rates
        for optimizer in self.optimizers:
            optimizer.lr = modifiers.get('learning_rate', 0.001)
```

## Energy Management

```python
class EnergySystem:
    """ATP-style energy management"""
    
    def __init__(self, max_atp: float = 100.0):
        self.max_atp = max_atp
        self.current_atp = max_atp
        self.recharge_rate = 1.0  # ATP/cycle at rest
        
    def consume(self, amount: float) -> bool:
        """Try to consume energy"""
        if self.current_atp >= amount:
            self.current_atp -= amount
            return True
        return False  # Not enough energy
    
    def recharge(self, metabolic_state: MetabolicState):
        """Recharge based on state"""
        if metabolic_state.name == "REST":
            self.current_atp += self.recharge_rate * 2
        elif metabolic_state.name == "DREAM":
            self.current_atp += self.recharge_rate * 3
        elif metabolic_state.name == "WAKE":
            self.current_atp += self.recharge_rate * 0.5
        
        self.current_atp = min(self.current_atp, self.max_atp)
```

## Training with Metabolic States

```python
class MetabolicTrainer:
    """Train policies with metabolic context"""
    
    def train_step(self, trajectory: List[Tuple[SAGEInput, SAGEOutput, MetabolicState]]):
        """Train with metabolic state as context"""
        
        for input, output, metabolic_state in trajectory:
            # Condition all policies on metabolic state
            state_encoding = self.encode_metabolic_state(metabolic_state)
            
            # Train attention policy conditioned on state
            attention_pred = self.attention_net(input, state_encoding)
            attention_loss = F.mse_loss(attention_pred, output.attention_state)
            
            # Train resource policy conditioned on state
            resource_pred = self.resource_net(input, state_encoding)
            resource_loss = F.cross_entropy(resource_pred, output.resource_actions)
            
            # Train transition policy
            next_state_pred = self.transition_net(metabolic_state, input)
            next_state_actual = trajectory[i+1][2] if i+1 < len(trajectory) else metabolic_state
            transition_loss = F.cross_entropy(next_state_pred, next_state_actual)
            
            total_loss = attention_loss + resource_loss + transition_loss
```

## Biological Inspiration

### Circadian Rhythms
- WAKE during day (high activity)
- REST in evening (wind down)
- DREAM at night (consolidation)

### Stress Response
- CRISIS when threatened (fight/flight)
- FOCUS when goal-directed (flow state)
- REST when exhausted (recovery)

### Energy Conservation
- High-energy states unsustainable
- Automatic transitions prevent burnout
- Sleep/dreams essential for learning

## Key Benefits

1. **Energy Efficiency**: Don't waste compute when resting
2. **Adaptive Behavior**: Different strategies for different contexts
3. **Natural Learning**: Consolidation during rest/dreams
4. **Crisis Response**: Can override normal limits when needed
5. **Sustainability**: Prevents burnout through forced rest

## Integration with SAGE

Metabolic states modify every aspect of SAGE:
- **Attention**: Breadth and switching rate
- **Resources**: How eagerly to load/unload
- **Memory**: When to consolidate vs acquire
- **Learning**: How fast to update weights
- **Actions**: How conservative vs exploratory

This creates a naturally adaptive system that behaves differently based on its internal state, just like biological intelligence.