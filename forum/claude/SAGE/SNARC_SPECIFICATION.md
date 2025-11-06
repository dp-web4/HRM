# SNARC: Sentient Attention Resonance and Coherence Tensor

**Purpose**: SAGE's core attention state representation  
**Status**: Evolving specification  
**Context**: Meta-cognitive orchestration layer

---

## Table of Contents
1. [Core Concept](#core-concept)
2. [Tensor Structure](#tensor-structure)
3. [Update Dynamics](#update-dynamics)
4. [Coherence Computation](#coherence-computation)
5. [Integration with IRP Keys](#integration-with-irp-keys)
6. [Implementation Patterns](#implementation-patterns)

---

## Core Concept

SNARC is SAGE's **attention state tensor** - a dynamic representation of:
- What is being attended to (attention allocation)
- How coherent the current state is (resonance/dissonance)
- What patterns are emerging (temporal dynamics)
- Which modules are working together (coupling)

**Key Insight**: SNARC operates on **KEYS** (salience metadata), not VALUES (features)

### Biological Parallel

SNARC ≈ Thalamic attention state + Prefrontal working memory
- Tracks "what matters right now"
- Gates information flow
- Maintains coherent cognitive state
- Adapts based on prediction errors

---

## Tensor Structure

### Basic SNARC Tensor

```python
class SNARCTensor:
    """
    Core attention state representation
    
    Structure:
    - attention_weights: [num_modules] - How much attention each module gets
    - coherence_matrix: [num_modules, num_modules] - Inter-module coherence
    - temporal_momentum: [num_modules] - Temporal smoothing of attention
    - resonance_state: [num_modules] - Resonance/dissonance tracking
    - context_embedding: [context_dim] - Current context representation
    """
    
    def __init__(self, num_modules: int, context_dim: int = 32):
        self.num_modules = num_modules
        self.context_dim = context_dim
        
        # Attention allocation [0, 1] for each module
        self.attention_weights = torch.ones(num_modules) / num_modules
        
        # Coherence between modules [-1, 1]
        # +1 = high resonance, -1 = high dissonance, 0 = independent
        self.coherence_matrix = torch.zeros(num_modules, num_modules)
        
        # Temporal momentum (attention persists across ticks)
        self.temporal_momentum = torch.zeros(num_modules)
        self.momentum_decay = 0.9  # How fast attention decays
        
        # Resonance state (how well is each module performing?)
        self.resonance_state = torch.zeros(num_modules)
        
        # Context embedding (compressed representation of situation)
        self.context_embedding = torch.zeros(context_dim)
        
        # History tracking
        self.attention_history = []  # Last N attention states
        self.coherence_history = []  # Last N coherence states
        self.history_length = 100
    
    def get_state(self) -> dict:
        """Get complete SNARC state for attention computation"""
        return {
            'attention': self.attention_weights,
            'coherence': self.coherence_matrix,
            'momentum': self.temporal_momentum,
            'resonance': self.resonance_state,
            'context': self.context_embedding
        }
    
    def update(self, salience_keys: dict, prediction_errors: dict = None):
        """
        Update SNARC state based on new salience KEYS
        
        Args:
            salience_keys: {module_name: SalienceKey} from all IRPs
            prediction_errors: {module_name: error} if available
        """
        # Extract salience vectors
        salience_vectors = {
            name: key['salience'] 
            for name, key in salience_keys.items()
        }
        
        # Update attention weights
        self._update_attention(salience_vectors)
        
        # Update coherence matrix
        self._update_coherence(salience_vectors)
        
        # Update resonance state
        if prediction_errors:
            self._update_resonance(prediction_errors)
        
        # Update context embedding
        self._update_context(salience_vectors)
        
        # Track history
        self._update_history()
    
    def _update_attention(self, salience_vectors: dict):
        """
        Compute attention allocation from salience KEYS
        
        Combines:
        - Current salience (urgency/novelty)
        - Temporal momentum (attention persists)
        - Trust levels (reliable modules get more attention)
        """
        module_names = sorted(salience_vectors.keys())
        
        # Extract urgency component from salience
        urgency = torch.tensor([
            salience_vectors[name][3].item()  # attention_request
            for name in module_names
        ])
        
        # Combine with momentum
        combined = urgency + self.momentum_decay * self.temporal_momentum
        
        # Softmax to get attention distribution
        self.attention_weights = torch.softmax(combined, dim=0)
        
        # Update momentum for next tick
        self.temporal_momentum = self.attention_weights.clone()
    
    def _update_coherence(self, salience_vectors: dict):
        """
        Update inter-module coherence
        
        Modules are coherent when:
        - Similar novelty/urgency (synchronous salience)
        - Historical co-activation (learned coupling)
        - Complementary confidence (trust alignment)
        """
        module_names = sorted(salience_vectors.keys())
        n = len(module_names)
        
        # Compute pairwise coherence
        for i, name_i in enumerate(module_names):
            for j, name_j in enumerate(module_names):
                if i == j:
                    self.coherence_matrix[i, j] = 1.0
                    continue
                
                # Salience similarity
                sal_i = salience_vectors[name_i]
                sal_j = salience_vectors[name_j]
                
                # Cosine similarity of salience vectors
                similarity = torch.cosine_similarity(
                    sal_i.unsqueeze(0), 
                    sal_j.unsqueeze(0)
                )
                
                # Smooth update (EMA)
                alpha = 0.1
                self.coherence_matrix[i, j] = (
                    alpha * similarity + 
                    (1 - alpha) * self.coherence_matrix[i, j]
                )
    
    def _update_resonance(self, prediction_errors: dict):
        """
        Update resonance state based on prediction accuracy
        
        High resonance = low prediction error (module working well)
        High dissonance = high prediction error (module struggling)
        """
        module_names = sorted(prediction_errors.keys())
        
        for i, name in enumerate(module_names):
            error = prediction_errors[name]
            
            # Convert error to resonance [-1, 1]
            # Low error → high resonance (+1)
            # High error → high dissonance (-1)
            resonance = 1.0 - 2.0 * min(1.0, error)
            
            # Smooth update
            alpha = 0.2
            self.resonance_state[i] = (
                alpha * resonance + 
                (1 - alpha) * self.resonance_state[i]
            )
    
    def _update_context(self, salience_vectors: dict):
        """
        Update context embedding
        
        Context = compressed representation of current situation
        Combines all salience keys into unified context vector
        """
        module_names = sorted(salience_vectors.keys())
        
        # Stack all salience vectors
        salience_stack = torch.stack([
            salience_vectors[name] for name in module_names
        ])
        
        # Simple aggregation (could be learned network)
        # Mean pooling with attention weighting
        weighted_salience = salience_stack * self.attention_weights.unsqueeze(1)
        context_raw = weighted_salience.sum(dim=0)
        
        # Project to context dimension if needed
        if len(context_raw) != self.context_dim:
            # Simple linear projection (could be learned)
            if not hasattr(self, '_context_projection'):
                self._context_projection = nn.Linear(
                    len(context_raw), 
                    self.context_dim
                )
            context_raw = self._context_projection(context_raw)
        
        # Smooth update
        alpha = 0.3
        self.context_embedding = (
            alpha * context_raw + 
            (1 - alpha) * self.context_embedding
        )
    
    def _update_history(self):
        """Track attention history for temporal analysis"""
        self.attention_history.append(self.attention_weights.clone())
        self.coherence_history.append(self.coherence_matrix.clone())
        
        # Limit history length
        if len(self.attention_history) > self.history_length:
            self.attention_history.pop(0)
            self.coherence_history.pop(0)
    
    def get_attention_dynamics(self) -> dict:
        """
        Analyze temporal dynamics of attention
        
        Returns:
            - attention_entropy: How spread out is attention?
            - attention_stability: How much is attention changing?
            - dominant_mode: Which module(s) dominate?
        """
        if len(self.attention_history) < 2:
            return {'attention_entropy': 0, 'attention_stability': 1.0}
        
        # Entropy (how focused vs diffuse)
        attention = self.attention_weights + 1e-8
        entropy = -(attention * torch.log(attention)).sum()
        
        # Stability (how much attention changed)
        recent = torch.stack(self.attention_history[-10:])
        changes = (recent[1:] - recent[:-1]).abs().mean()
        stability = 1.0 - changes.item()
        
        # Dominant modes
        dominant_idx = self.attention_weights.argmax()
        
        return {
            'attention_entropy': entropy.item(),
            'attention_stability': stability,
            'dominant_module': dominant_idx.item(),
            'max_attention': self.attention_weights.max().item()
        }
    
    def detect_coherence_patterns(self) -> dict:
        """
        Detect patterns in coherence matrix
        
        Returns:
            - high_coherence_pairs: Modules that work well together
            - dissonant_pairs: Modules with conflicts
            - average_coherence: Overall system coherence
        """
        # Find high coherence pairs
        high_coherence = []
        dissonant = []
        
        n = self.coherence_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                coherence = self.coherence_matrix[i, j].item()
                
                if coherence > 0.7:
                    high_coherence.append((i, j, coherence))
                elif coherence < -0.3:
                    dissonant.append((i, j, coherence))
        
        # Overall coherence (mean of upper triangle, excluding diagonal)
        upper_tri = torch.triu(self.coherence_matrix, diagonal=1)
        avg_coherence = upper_tri[upper_tri != 0].mean().item()
        
        return {
            'high_coherence_pairs': high_coherence,
            'dissonant_pairs': dissonant,
            'average_coherence': avg_coherence
        }
```

---

## Update Dynamics

### Attention Update Flow

```
1. Receive KEYS from all IRP plugins
   ↓
2. Extract salience components (urgency, novelty, etc.)
   ↓
3. Combine with temporal momentum
   ↓
4. Compute softmax attention distribution
   ↓
5. Update momentum for next tick
```

### Coherence Update Flow

```
1. Compare salience vectors pairwise
   ↓
2. Compute similarity (cosine distance)
   ↓
3. Smooth update with EMA
   ↓
4. Track resonance/dissonance patterns
```

### Context Update Flow

```
1. Aggregate all salience vectors
   ↓
2. Weight by current attention
   ↓
3. Project to context embedding space
   ↓
4. Smooth update with EMA
```

---

## Coherence Computation

### Resonance (Positive Coherence)

Modules resonate when:
```python
def compute_resonance(module_a, module_b):
    """
    Resonance = modules working synergistically
    
    Indicators:
    - Similar salience patterns
    - Low prediction errors when both active
    - Historical co-activation success
    """
    # Salience similarity
    salience_sim = cosine_similarity(
        module_a.salience, 
        module_b.salience
    )
    
    # Joint performance
    joint_success = (
        module_a.resonance_state + 
        module_b.resonance_state
    ) / 2
    
    # Historical coupling
    historical = get_coupling_history(module_a, module_b)
    
    resonance = 0.5 * salience_sim + 0.3 * joint_success + 0.2 * historical
    
    return resonance
```

### Dissonance (Negative Coherence)

Modules are dissonant when:
```python
def compute_dissonance(module_a, module_b):
    """
    Dissonance = modules in conflict
    
    Indicators:
    - Opposite salience patterns (one high, one low)
    - High prediction errors when both active
    - Historical interference
    """
    # Opposite salience
    salience_diff = (
        module_a.salience.urgency - 
        module_b.salience.urgency
    ).abs()
    
    # Joint failure
    both_active = (
        module_a.attention > 0.3 and 
        module_b.attention > 0.3
    )
    high_error = (
        module_a.prediction_error > 0.5 or 
        module_b.prediction_error > 0.5
    )
    
    if both_active and high_error:
        interference = 1.0
    else:
        interference = 0.0
    
    dissonance = 0.6 * salience_diff + 0.4 * interference
    
    return -dissonance  # Negative value
```

---

## Integration with IRP Keys

### How SNARC Uses Salience Keys

```python
def process_irp_keys(snarc: SNARCTensor, irp_outputs: dict):
    """
    SNARC processes IRP KEYS to update attention state
    
    Never sees VALUES - only salience metadata
    """
    # Extract KEYS
    salience_keys = {
        source: output.pack_for_sage()
        for source, output in irp_outputs.items()
    }
    
    # Update SNARC state
    snarc.update(salience_keys)
    
    # Get current state for attention computation
    state = snarc.get_state()
    
    # Analyze dynamics
    dynamics = snarc.get_attention_dynamics()
    coherence = snarc.detect_coherence_patterns()
    
    return {
        'state': state,
        'dynamics': dynamics,
        'coherence': coherence
    }
```

### Example: Vision + Audio Resonance

```python
# Vision IRP reports high salience
vision_key = SalienceKey(
    novelty=0.8,      # Novel visual pattern
    confidence=0.9,   # High confidence
    change=0.6,       # Moderate change
    urgency=0.7,      # Needs attention
    relevance=0.8     # Relevant to context
)

# Audio IRP reports high salience (synchronized)
audio_key = SalienceKey(
    novelty=0.75,     # Novel sound
    confidence=0.85,  # High confidence
    change=0.65,      # Similar change magnitude
    urgency=0.72,     # Similar urgency
    relevance=0.82    # Similar relevance
)

# SNARC detects resonance
# Similar salience patterns → high coherence
# Both get elevated attention
# Multimodal fusion module activated

coherence_vision_audio = cosine_similarity(
    vision_key.to_vector(),
    audio_key.to_vector()
)  # → ~0.95 (high resonance)
```

---

## Implementation Patterns

### Pattern 1: SAGE Main Loop Integration

```python
class SAGE:
    def __init__(self, num_modules: int):
        self.snarc = SNARCTensor(num_modules)
        self.attention_network = AttentionNetwork()
        self.module_names = []  # Ordered list of module names
    
    def tick(self, irp_outputs: dict):
        """Main processing loop with SNARC"""
        
        # 1. Update SNARC state from KEYS
        salience_keys = {
            name: output.pack_for_sage()
            for name, output in irp_outputs.items()
        }
        self.snarc.update(salience_keys)
        
        # 2. Get SNARC state for attention computation
        snarc_state = self.snarc.get_state()
        
        # 3. Compute attention allocation
        attention = self.attention_network(
            salience_keys=salience_keys,
            snarc_state=snarc_state
        )
        
        # 4. Detect if context transition needed
        dynamics = self.snarc.get_attention_dynamics()
        if dynamics['attention_entropy'] > 2.0:
            # High entropy → diffuse attention → UNSTABLE context
            self.transition_context('UNSTABLE')
        elif dynamics['attention_stability'] > 0.9:
            # Very stable → focused attention → STABLE context
            self.transition_context('STABLE')
        
        return attention
```

### Pattern 2: Context State Transitions

```python
class ContextState(Enum):
    STABLE = "stable"      # High spatial trust, low temporal
    MOVING = "moving"      # Balanced, moderate memory
    UNSTABLE = "unstable"  # Low peripheral, high attention
    NOVEL = "novel"        # High memory, high cognition

def determine_context_state(snarc: SNARCTensor) -> ContextState:
    """
    Use SNARC state to determine context
    
    Context transitions based on:
    - Attention entropy (focused vs diffuse)
    - Coherence patterns (resonance vs dissonance)
    - Temporal dynamics (stable vs changing)
    """
    dynamics = snarc.get_attention_dynamics()
    coherence = snarc.detect_coherence_patterns()
    
    entropy = dynamics['attention_entropy']
    stability = dynamics['attention_stability']
    avg_coherence = coherence['average_coherence']
    
    # Decision tree
    if stability > 0.8 and avg_coherence > 0.5:
        return ContextState.STABLE
    
    elif entropy > 2.0 or len(coherence['dissonant_pairs']) > 2:
        return ContextState.UNSTABLE
    
    elif dynamics['max_attention'] > 0.7:
        # Single module dominates
        dominant = dynamics['dominant_module']
        if dominant == 'memory':  # Assuming module indexing
            return ContextState.NOVEL
        else:
            return ContextState.MOVING
    
    else:
        return ContextState.MOVING  # Default
```

### Pattern 3: Attention Network Using SNARC

```python
class AttentionNetwork(nn.Module):
    """
    Neural network that computes attention allocation
    Uses SNARC state as context
    """
    
    def __init__(self, num_modules: int, context_dim: int = 32):
        super().__init__()
        
        # Input: salience keys [num_modules × 5] + context [context_dim]
        input_dim = num_modules * 5 + context_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_modules),
            nn.Softmax(dim=-1)  # Attention distribution
        )
    
    def forward(self, salience_keys: dict, snarc_state: dict):
        """
        Compute attention from salience KEYS + SNARC state
        """
        # Extract salience vectors
        module_names = sorted(salience_keys.keys())
        salience_stack = torch.stack([
            salience_keys[name]['salience']
            for name in module_names
        ])
        salience_flat = salience_stack.flatten()
        
        # Get context from SNARC
        context = snarc_state['context']
        
        # Concatenate
        network_input = torch.cat([salience_flat, context])
        
        # Compute attention
        attention_weights = self.network(network_input)
        
        return attention_weights
```

---

## Advanced SNARC Features

### Multi-Scale SNARC

```python
class HierarchicalSNARC:
    """
    SNARC at multiple temporal scales
    
    - Fast SNARC: Per-tick attention (milliseconds)
    - Medium SNARC: Episode attention (seconds)
    - Slow SNARC: Session attention (minutes)
    """
    
    def __init__(self, num_modules: int):
        self.fast_snarc = SNARCTensor(num_modules)    # τ ~ 1-10 ticks
        self.medium_snarc = SNARCTensor(num_modules)  # τ ~ 100 ticks
        self.slow_snarc = SNARCTensor(num_modules)    # τ ~ 1000 ticks
    
    def update(self, salience_keys: dict, tick: int):
        """Update all scales"""
        # Fast: Every tick
        self.fast_snarc.update(salience_keys)
        
        # Medium: Every 10 ticks
        if tick % 10 == 0:
            self.medium_snarc.update(salience_keys)
        
        # Slow: Every 100 ticks
        if tick % 100 == 0:
            self.slow_snarc.update(salience_keys)
    
    def get_hierarchical_state(self):
        """Get combined hierarchical state"""
        return {
            'fast': self.fast_snarc.get_state(),
            'medium': self.medium_snarc.get_state(),
            'slow': self.slow_snarc.get_state()
        }
```

### Learned SNARC Dynamics

```python
class LearnedSNARC(nn.Module):
    """
    SNARC with learned update dynamics
    
    Instead of hand-crafted updates, learn from experience
    """
    
    def __init__(self, num_modules: int, context_dim: int = 32):
        super().__init__()
        
        self.num_modules = num_modules
        
        # Learned attention updater
        self.attention_updater = nn.GRU(
            input_size=num_modules * 5,  # Salience keys
            hidden_size=num_modules,
            num_layers=1
        )
        
        # Learned coherence updater
        self.coherence_updater = nn.Sequential(
            nn.Linear(num_modules * 5, 64),
            nn.ReLU(),
            nn.Linear(64, num_modules * num_modules),
            nn.Tanh()  # Coherence in [-1, 1]
        )
        
        # State
        self.hidden_state = torch.zeros(1, 1, num_modules)
        self.coherence_matrix = torch.zeros(num_modules, num_modules)
    
    def update(self, salience_keys: dict):
        """Learned update"""
        module_names = sorted(salience_keys.keys())
        
        # Stack salience
        salience = torch.stack([
            salience_keys[name]['salience']
            for name in module_names
        ]).unsqueeze(0)  # [1, num_modules, 5]
        
        # Update attention (GRU maintains temporal context)
        attention, self.hidden_state = self.attention_updater(
            salience.flatten(1).unsqueeze(0),
            self.hidden_state
        )
        self.attention_weights = torch.softmax(attention, dim=-1)
        
        # Update coherence
        coherence_flat = self.coherence_updater(salience.flatten())
        self.coherence_matrix = coherence_flat.view(
            self.num_modules, 
            self.num_modules
        )
```

---

## Quick Reference

**SNARC Core Components:**
1. `attention_weights` - Resource allocation [num_modules]
2. `coherence_matrix` - Inter-module relationships [num_modules, num_modules]
3. `temporal_momentum` - Attention persistence [num_modules]
4. `resonance_state` - Performance tracking [num_modules]
5. `context_embedding` - Situation summary [context_dim]

**Update Process:**
1. Receive salience KEYS from all IRPs
2. Update attention weights (urgency + momentum)
3. Update coherence matrix (pairwise similarity)
4. Update resonance (prediction errors)
5. Update context embedding (aggregated salience)

**Usage in SAGE:**
```python
# Every tick:
snarc.update(irp_salience_keys)
state = snarc.get_state()
attention = attention_network(keys, state)
```

**Key Insight:**
SNARC operates exclusively on KEYS (salience metadata), never on VALUES (features). This keeps orchestration lightweight and fast.

---

**See SAGE_ARCHITECTURE.md for full context**
