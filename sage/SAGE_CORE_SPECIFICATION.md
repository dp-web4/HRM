# SAGE Core Specification

**Date**: October 6, 2025  
**Version**: 1.0 - Implementable Definition  
**Purpose**: Define SAGE as a concrete, implementable system

## What SAGE Actually Is

**SAGE is a continuous inference loop that:**
1. Maintains state across time via temporal sensors
2. Manages a registry of available compute resources (IRPs, models)
3. Decides what specialized reasoning to invoke based on current context
4. Preserves contextualized state for all components via SNARC
5. Orchestrates resource loading/unloading based on need

## Core Architecture

```python
class SAGE:
    """Stateful Adaptive Generative Engine"""
    
    def __init__(self):
        # Persistent state
        self.temporal_state = TemporalState()  # Clock, phase, history
        self.snarc_memory = SNARCMemory()      # Salience-gated storage
        self.resource_registry = ResourceRegistry()  # What's available
        self.active_resources = {}  # What's loaded in GPU/RAM
        self.context_states = {}    # Per-component state preservation
        
        # Core inference loop state  
        self.attention_focus = None
        self.current_goal = None
        self.surprise_buffer = deque(maxlen=100)
        self.trust_scores = {}
        
    def run(self):
        """Continuous inference loop - THIS IS SAGE"""
        while True:
            # 1. Sense current state
            observations = self.gather_observations()
            
            # 2. Update temporal context
            self.temporal_state.tick()
            
            # 3. Compute what needs attention
            attention_targets = self.compute_attention(observations)
            
            # 4. Decide what resources are needed
            required_resources = self.plan_resources(attention_targets)
            
            # 5. Load/unload resources as needed
            self.manage_resources(required_resources)
            
            # 6. Invoke specialized reasoning
            results = self.invoke_reasoning(attention_targets)
            
            # 7. Update memory and trust
            self.update_state(results)
            
            # 8. Generate outputs/actions
            self.execute_actions(results)
```

## Inputs and Outputs

### SAGE Inputs

```python
@dataclass
class SAGEInput:
    """Everything SAGE receives per cycle"""
    
    # Sensor streams (continuous)
    sensor_observations: Dict[str, SensorData]
    # sensor_observations = {
    #     'vision': tensor[3, 224, 224],
    #     'audio': tensor[16000],  
    #     'proprioception': tensor[7],
    #     'clock': float (unix timestamp)
    # }
    
    # Feedback from previous actions
    action_results: Dict[str, ActionResult]
    # action_results = {
    #     'motor': {'success': True, 'error': 0.01},
    #     'speech': {'words_spoken': 5}
    # }
    
    # External directives (if any)
    user_goals: Optional[Goal]
    # user_goals = Goal(text="pick up the red cube", priority=1.0)
    
    # System resources
    available_memory: float  # GB free
    available_compute: float  # TFLOPS available
```

### SAGE Outputs

```python
@dataclass 
class SAGEOutput:
    """Everything SAGE produces per cycle"""
    
    # Resource management decisions
    resource_actions: List[ResourceAction]
    # resource_actions = [
    #     Load('vision_irp'),
    #     Unload('audio_irp'),
    #     Keep('motor_irp')
    # ]
    
    # Reasoning invocations
    reasoning_requests: List[ReasoningRequest]
    # reasoning_requests = [
    #     InvokeLLM(prompt="what color is this?", context=vision_tokens),
    #     InvokeDiffusion(latent=z, steps=20),
    # ]
    
    # State updates
    memory_updates: List[MemoryUpdate]
    # memory_updates = [
    #     Store(key='red_cube_location', value=[0.3, 0.5], salience=0.9),
    #     Forget(key='old_goal')
    # ]
    
    # Effector commands
    action_commands: Dict[str, Command]
    # action_commands = {
    #     'motor': MoveToPosition([0.3, 0.5, 0.1]),
    #     'speech': Say("I see a red cube")
    # }
    
    # Trust/attention updates
    attention_state: AttentionState
    # attention_state = {
    #     'focus': 'vision',
    #     'trust_updates': {'vision': 0.95, 'audio': 0.3}
    # }
```

## State Management

### 1. Temporal State
```python
class TemporalState:
    """Maintains time-aware context"""
    
    def __init__(self):
        self.absolute_time = time.time()
        self.cycle_count = 0
        self.phase = self.compute_phase()  # sin/cos embeddings
        self.history_buffer = deque(maxlen=1000)  # Last N cycles
        
    def tick(self):
        self.cycle_count += 1
        self.absolute_time = time.time()
        self.phase = self.compute_phase()
        
    def get_temporal_encoding(self) -> Tensor:
        """Return temporal features for current moment"""
        return torch.cat([
            self.phase,  # Cyclic features
            self.one_hot_time(),  # Discrete time
            self.history_statistics()  # Recent patterns
        ])
```

### 2. SNARC Memory Management
```python
class SNARCMemory:
    """Salience-gated memory with contextualized states"""
    
    def store(self, key: str, value: Any, salience: float, context: Dict):
        """Store with salience gating"""
        if salience > self.threshold:
            self.storage[key] = {
                'value': value,
                'salience': salience,
                'context': context,
                'timestamp': time.time(),
                'access_count': 0
            }
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve and update access patterns"""
        if key in self.storage:
            self.storage[key]['access_count'] += 1
            return self.storage[key]['value']
        return None
    
    def get_component_state(self, component: str) -> Dict:
        """Get all state for a specific component"""
        return {k: v for k, v in self.storage.items() 
                if v['context'].get('component') == component}
```

### 3. Resource Registry
```python
class ResourceRegistry:
    """Track available and active resources"""
    
    def __init__(self):
        self.available = {
            # IRPs
            'vision_irp': ResourceSpec(size_gb=1.0, location='disk'),
            'audio_irp': ResourceSpec(size_gb=0.5, location='disk'),
            'motor_irp': ResourceSpec(size_gb=0.5, location='disk'),
            
            # Specialized models
            'llama_7b': ResourceSpec(size_gb=13.0, location='disk'),
            'stable_diffusion': ResourceSpec(size_gb=4.0, location='disk'),
            'whisper': ResourceSpec(size_gb=1.5, location='disk')
        }
        
        self.active = {}  # Currently loaded
        self.loading = {}  # Being loaded
        self.unloading = {}  # Being unloaded
        
    def can_load(self, resource: str, available_memory: float) -> bool:
        """Check if resource can be loaded"""
        spec = self.available[resource]
        used = sum(r.size_gb for r in self.active.values())
        return (used + spec.size_gb) <= available_memory
```

## Core Algorithms

### 1. Attention Computation
```python
def compute_attention(self, observations: Dict) -> List[AttentionTarget]:
    """Decide what needs attention based on trust and surprise"""
    
    targets = []
    
    for modality, data in observations.items():
        # Compute surprise (prediction error)
        expected = self.predict_observation(modality)
        surprise = self.compute_surprise(expected, data)
        
        # Update trust based on surprise
        old_trust = self.trust_scores.get(modality, 0.5)
        new_trust = old_trust * (1.0 - surprise)
        self.trust_scores[modality] = clip(new_trust, 0.1, 1.0)
        
        # Attention proportional to trust
        if new_trust > 0.3:  # Threshold
            targets.append(AttentionTarget(
                modality=modality,
                priority=new_trust,
                data=data
            ))
    
    return sorted(targets, key=lambda x: x.priority, reverse=True)
```

### 2. Resource Planning
```python
def plan_resources(self, attention_targets: List) -> List[str]:
    """Decide what resources are needed for current targets"""
    
    required = []
    
    for target in attention_targets:
        if target.modality == 'vision' and target.priority > 0.7:
            required.append('vision_irp')
            if self.need_object_detection(target.data):
                required.append('yolo')  # Specific model
                
        elif target.modality == 'language':
            required.append('language_irp')
            if self.need_generation(target.data):
                required.append('llama_7b')
                
        elif target.modality == 'planning':
            required.append('diffusion')  # For trajectory planning
    
    return required
```

### 3. Specialized Reasoning Invocation
```python
def invoke_reasoning(self, targets: List[AttentionTarget]) -> Dict:
    """Call specialized reasoning based on targets"""
    
    results = {}
    
    for target in targets:
        if target.modality == 'vision':
            # Use vision IRP for feature extraction
            vision_irp = self.active_resources['vision_irp']
            features = vision_irp.process(target.data)
            
            # Maybe also use LLM for visual QA
            if 'llama_7b' in self.active_resources:
                llm = self.active_resources['llama_7b']
                caption = llm.generate(features, "describe this scene")
                results['vision'] = {'features': features, 'caption': caption}
        
        elif target.modality == 'planning':
            # Use diffusion for trajectory generation
            if 'stable_diffusion' in self.active_resources:
                diffusion = self.active_resources['stable_diffusion']
                trajectory = diffusion.denoise(target.data)
                results['plan'] = trajectory
    
    return results
```

## Training Strategy

### What to Train

1. **Attention Policy**: Learn what deserves attention when
2. **Resource Policy**: Learn what resources to load for which contexts  
3. **Trust Dynamics**: Learn surprise → trust mapping
4. **State Prediction**: Learn to predict next observations

### How to Train

```python
class SAGETrainer:
    """Train SAGE's decision policies"""
    
    def __init__(self, sage: SAGE):
        self.sage = sage
        
        # Learnable components
        self.attention_net = AttentionPolicyNet()
        self.resource_net = ResourcePolicyNet()
        self.prediction_net = StatePredictionNet()
        
    def train_step(self, trajectory: List[SAGEInput, SAGEOutput]):
        """Train from recorded trajectories"""
        
        losses = {}
        
        # Train attention policy
        for input, output in trajectory:
            predicted_attention = self.attention_net(input)
            actual_attention = output.attention_state
            losses['attention'] = F.mse_loss(predicted_attention, actual_attention)
        
        # Train resource policy
        for input, output in trajectory:
            predicted_resources = self.resource_net(input)
            actual_resources = output.resource_actions
            losses['resource'] = F.cross_entropy(predicted_resources, actual_resources)
        
        # Train state prediction
        for i in range(len(trajectory) - 1):
            current_state = trajectory[i][0]
            next_state = trajectory[i + 1][0]
            predicted_next = self.prediction_net(current_state)
            losses['prediction'] = F.mse_loss(predicted_next, next_state)
        
        return losses
```

## Implementation Checklist

### Core Components
- [ ] Temporal state tracker
- [ ] SNARC memory system
- [ ] Resource registry and loader
- [ ] Trust score tracker
- [ ] Surprise computer
- [ ] Attention allocator

### Policies to Learn
- [ ] Attention policy network
- [ ] Resource management policy
- [ ] Trust update rules
- [ ] State prediction model

### Infrastructure
- [ ] Continuous loop runner
- [ ] State serialization
- [ ] Resource lazy loading
- [ ] Context preservation

## Key Insights

1. **SAGE is the loop, not a model** - It's the continuous process of sensing, deciding, and acting
2. **State is fundamental** - Without temporal and contextual state, it's not SAGE
3. **Resources are dynamic** - Load/unload based on need, not everything at once
4. **Trust drives attention** - What works gets more compute
5. **Surprise drives learning** - Unexpected events modify trust and trigger adaptation

This is SAGE: A stateful orchestrator that knows what it has, what it needs, and how to use specialized reasoning as required.