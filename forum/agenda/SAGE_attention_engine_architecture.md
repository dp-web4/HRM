# SAGE: The Attention Engine Architecture

*Date: September 4, 2025*  
*Team: dp-web4, Nova, Claude*  
*Core Insight: SAGE orchestrates attention, external LLMs provide language cognition*

## SAGE Definition Breakdown

### **S**entient: "What should I attend to?"
- Monitors all available sensors (vision, language, memory, time)
- Determines salience through SNARC scoring
- Asks: "What's important in this situation?"
- Routes attention based on context needs

### **A**gentic: "What choice do I make?"
- Situational awareness from integrated sensors
- Makes decisions about resource allocation
- Takes action through sensor/effector deployment
- Not reactive but deliberately choosing

### **G**enerative: "How do I adapt?"
- Not preprogrammed responses
- Operates in latent space to handle novelty
- Creates unique action sequences
- Learns new patterns without retraining

### **E**ngine: "Always running"
- Continuous operation powering the ecosystem
- Not request-response but always aware
- Manages the flow between all components
- The heartbeat of the system

## Architecture: SAGE as Attention Orchestrator

```python
class SAGE(nn.Module):
    """
    SAGE doesn't try to be everything - it's the attention engine
    that knows WHEN to call WHAT resource
    """
    def __init__(self, config):
        # HRM core - 100M params for attention/routing
        self.hrm = ContextAwareHRM(config)
        
        # External resources (not trained, just used)
        self.resources = {
            'llm': ExternalLLM(),  # Gemma, Llama, Phi, etc.
            'vision': VisionEncoder(),
            'memory': MemoryBank(),
            'time': TemporalTracker(),
        }
        
        # SNARC scoring for attention
        self.snarc = SNARCScorer()
        
        # Resource selection head
        self.resource_router = ResourceRouter()
```

## The Cognitive Flow

### 1. Continuous Monitoring (Engine)
```python
while True:
    # SAGE never stops attending
    sensor_states = gather_all_sensors()
    snarc_scores = compute_snarc(sensor_states)
```

### 2. Attention Decision (Sentient)
```python
# What's important right now?
attention_weights = hrm.h_level(snarc_scores, context)
salient_inputs = filter_by_attention(sensor_states, attention_weights)
```

### 3. Resource Routing (Agentic)
```python
# What resources do I need for this?
if needs_language_reasoning(salient_inputs):
    thought = external_llm.generate(context_prompt)
    context = encode_language_context(thought)
```

### 4. Adaptive Action (Generative)
```python
# Create novel response in latent space
action_plan = hrm.l_level(context, salient_inputs)
execute_through_effectors(action_plan)
```

## External LLM Integration

### We Already Have the Pieces!
- **Multiple pretrained LLMs**: Gemma, Llama, Phi variants
- **Phoenician LoRA**: Tested language adaptation
- **Integration experience**: Know how to connect them

### LLM as Temporal Cognition Sensor
The LLM provides the "inner monologue" that creates context:

```python
class LLMCognitionSensor:
    def __init__(self, model_name="gemma-2b"):
        self.llm = load_pretrained(model_name)
        self.lora = load_lora("phoenician")  # If needed
        
    def generate_context(self, observation):
        # Convert observation to language thought
        prompt = f"I see: {encode_observation(observation)}\n"
        prompt += "This appears to be: "
        
        thought = self.llm.generate(prompt)
        # "This appears to be a rotation pattern where..."
        
        return thought
```

### HRM Uses LLM Context
```python
def process_arc_task(task):
    # 1. Visual processing
    grids = task['train']
    
    # 2. LLM generates understanding
    thought = llm_sensor.generate_context(grids)
    # "The pattern shows objects rotating 90° clockwise"
    
    # 3. HRM H-level processes language context
    context_embedding = encode_thought(thought)
    h_state = hrm.h_level(context_embedding)
    
    # 4. HRM L-level executes based on understanding
    solution = hrm.l_level(h_state, task['test'])
    
    return solution
```

## Implementation Strategy

### Phase 1: Basic Integration (Week 1)
- Connect existing LLM (Gemma-2B or Phi-2)
- Simple prompt templates for ARC observations
- HRM learns to use LLM-generated context

### Phase 2: SNARC-Guided Attention (Week 2)
- LLM queries guided by surprise/novelty scores
- Only ask LLM about surprising elements
- Learn when language helps vs. when it doesn't

### Phase 3: Multi-Modal Fusion (Week 3)
- Vision → Language → Action pipeline
- Memory retrieval guided by language context
- Temporal awareness ("I've seen this before")

### Phase 4: Full SAGE Loop (Week 4)
- Continuous attention management
- Dynamic resource allocation
- Learn to minimize LLM calls (expensive)
- Optimize for edge deployment

## Why This Architecture Works

### 1. Separation of Concerns
- SAGE/HRM: Attention, routing, execution (100M params, trainable)
- LLM: Language understanding (2-7B params, frozen)
- Vision: Pattern recognition (separate encoder)
- Memory: Experience storage (external bank)

### 2. We Already Have Everything
- Pretrained LLMs: ✅ (Multiple tested)
- LoRA adapters: ✅ (Phoenician tested)
- HRM architecture: ✅ (Just needs scaling)
- Integration knowledge: ✅ (From previous work)

### 3. Edge-Deployable
- HRM (100M): Fits on Jetson
- LLM calls: Can be cached/batched
- Most decisions: Don't need LLM
- Critical decisions: Worth the LLM cost

## The Missing Context Problem - SOLVED

Agent Zero failed because it had no language to think with. Now:
1. **See puzzle** → "What is this?"
2. **LLM thinks** → "This looks like a reflection pattern"
3. **HRM understands** → Context = "reflection"
4. **HRM executes** → Apply reflection transformation
5. **Validate** → Did it work?

## Next Steps

### Immediate
1. Select LLM for integration (Gemma-2B recommended for size/quality)
2. Create prompt templates for ARC task description
3. Set up HRM-LLM pipeline

### This Week
1. Train HRM to use LLM context
2. Implement SNARC scoring
3. Test on ARC tasks with language supervision

### This Month
1. Full SAGE attention loop
2. Multi-resource orchestration
3. Edge optimization

## Philosophical Note

SAGE isn't trying to be conscious - it's trying to be **attentive**. It's the executive function that decides:
- What deserves attention
- What resources to deploy
- When to think in language
- When to act directly

The external LLM provides the linguistic cognition, while SAGE provides the attentional awareness that orchestrates everything. Together, they form a complete cognitive system where:
- **SAGE knows WHEN to think**
- **LLM knows HOW to think**
- **HRM knows HOW to act**

This is why we need all three - none alone is sufficient for intelligence.

---

*"SAGE is the attention engine that knows when to employ language, not the language model itself."*