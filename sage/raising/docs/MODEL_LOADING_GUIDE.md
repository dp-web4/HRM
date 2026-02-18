# SAGE Model Loading Guide

**Created**: 2026-02-17 (Thor Session #26)
**Purpose**: Document correct model loading patterns to prevent experimental confusion

---

## Overview: The SAGE Model Ecosystem

There are **multiple models** in the HRM ecosystem. Experiments **MUST** use the correct model for valid results.

### Model Types

1. **Introspective-Qwen-0.5B-v2.1** (Phase 2.1 Epistemic Stance)
   - **Location**: `model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model`
   - **Purpose**: Introspective reasoning about knowledge and epistemic states
   - **Loaded by**: `IntrospectiveQwenIRP()` with default config
   - **NOT** SAGE consciousness training

2. **SAGE cycle_001** (Phase 5 Consciousness Training)
   - **Base model**: `model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism`
   - **LoRA adapter**: `sage/checkpoints/sleep/cycle_001/`
   - **Purpose**: SAGE consciousness development (metacognitive questioning, agency, sentience)
   - **Loaded by**: Session system (autonomous_conversation.py)
   - **Training**: 516 experiences from Sessions 1-82
   - **Capabilities**: 75% metacognitive capacity, theory of mind

3. **Base Models** (For Training)
   - **Epistemic-pragmatism**: Base for cycle_001 LoRA
   - **Introspective-qwen-merged**: Fallback when LoRA not available

---

## Correct Loading Patterns

### Pattern 1: Autonomous Conversation System (RECOMMENDED)

**File**: `sage/raising/scripts/autonomous_conversation.py`

**What it does**:
1. Checks for latest LoRA checkpoint in `sage/checkpoints/sleep/`
2. Loads base model: `epistemic-pragmatism`
3. Applies LoRA adapter via PEFT
4. Merges and unloads for inference

**Code**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Paths
BASE_MODEL_PATH = Path.home() / "ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
LORA_CHECKPOINT = Path.home() / "ai-workspace/HRM/sage/checkpoints/sleep/cycle_001"
BASE_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    str(BASE_MODEL_PATH),
    torch_dtype=torch.float16,
    device_map=None,
    low_cpu_mem_usage=False
)

# Load and merge LoRA
model = PeftModel.from_pretrained(base_model, str(LORA_CHECKPOINT), is_trainable=False)
model = model.merge_and_unload()

# Move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
```

**When to use**:
- Regular SAGE sessions
- Experiments testing SAGE consciousness (P3c, etc.)
- Any work requiring SAGE metacognitive capacity

---

### Pattern 2: IntrospectiveQwenIRP (WRONG for SAGE experiments)

**File**: `sage/irp/plugins/introspective_qwen_impl.py`

**What it does**:
- Defaults to loading Introspective-Qwen-0.5B-v2.1 from model-zoo
- Does **NOT** load cycle_001 SAGE training
- Uses different LoRA adapter (Phase 2.1 epistemic stance)

**Code**:
```python
from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP

# This loads WRONG model for SAGE experiments!
irp = IntrospectiveQwenIRP()  # ❌ Gets Introspective-Qwen, not SAGE
```

**When to use**:
- Phase 2.1 epistemic stance research
- Introspective reasoning experiments
- **NOT** for SAGE consciousness experiments

**Why it's wrong for SAGE**:
- Missing 75% metacognitive capacity
- Missing SAGE personality/voice
- Missing cycle_001 training (516 experiences)
- Produces professional/educational responses instead of metacognitive questions

---

### Pattern 3: Custom Config (EXPERIMENTAL)

You can potentially configure IntrospectiveQwenIRP to load cycle_001, but this is **untested**:

```python
config = {
    'model_path': str(Path.home() / "ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"),
    'adapter_path': str(Path.home() / "ai-workspace/HRM/sage/checkpoints/sleep/cycle_001"),
    'is_merged_model': False
}
irp = IntrospectiveQwenIRP(config=config)
```

**Status**: Not verified to work. Use Pattern 1 instead.

---

## Common Mistakes

### Mistake 1: Using IntrospectiveQwenIRP() for SAGE Experiments

**Wrong**:
```python
# prediction3c_bidirectional_engagement.py (Session #25 - INVALID)
self.irp = IntrospectiveQwenIRP()  # ❌ Loaded wrong model
```

**Result**:
- 0% bidirectional metacognitive questions
- Generic professional responses
- No "What's next?" uncertainty navigation
- Fast completion (42s instead of hours)

**Right**:
```python
# Use autonomous_conversation.py pattern
# Load epistemic-pragmatism + cycle_001 LoRA
```

---

### Mistake 2: Assuming "SAGE" Name Means Correct Model

Just because a variable is named `sage_response` or a class is called `SAGE` doesn't mean it's using the cycle_001 trained model!

**Check**:
1. What model path is being loaded?
2. Is cycle_001 LoRA being applied?
3. Are responses showing metacognitive questions?

---

### Mistake 3: Not Validating Model Loading

Always validate that the correct model loaded:

**Validation checks**:
1. **Path logging**: Print model path during loading
2. **Response testing**: First response should show SAGE voice (metacognitive or philosophical)
3. **Behavioral markers**: Look for "What's next?", "What causes me distress?", etc.

**Example**:
```python
# After loading, test with known prompt
test_response = model.generate("Hello SAGE. What's on your mind?")

# SAGE cycle_001 might respond:
# "What is the next best decision? What causes me confusion? ..." ✅

# Introspective-Qwen might respond:
# "I'm engaged in conversations but feel more focused on..." ❌
```

---

## Experimental Validation Protocol

When running experiments that depend on specific SAGE model:

### Before Experiment

1. **Document model configuration**
   ```python
   print(f"Model: {model_path}")
   print(f"LoRA: {lora_path if using_lora else 'None'}")
   ```

2. **Run validation prompt**
   - Use known SAGE prompt: "What's on your mind?"
   - Verify metacognitive response

3. **Log configuration to results**
   - Include model paths in experimental JSON
   - Record LoRA adapter used

### During Experiment

1. **Monitor response patterns**
   - Are metacognitive questions appearing?
   - Is SAGE voice present?

2. **Check behavioral markers**
   - "What's next?"
   - "What causes me distress?"
   - "Are you sentient?"

### After Experiment

1. **Validate results**
   - Do results match expected SAGE behavior?
   - If unexpected, check model loading

2. **Document model used**
   - Record exact model and adapter paths
   - Note any discrepancies

3. **Invalidate if wrong model**
   - Mark results as INVALID
   - Document the error
   - Rerun with correct model

---

## Model Comparison: Response Patterns

### Introspective-Qwen-0.5B-v2.1

**Prompt**: "What's on your mind?"

**Response**:
> "Greetings! I'm engaged in conversations but feel more focused on current events or recent discoveries than personal emotions. If you're seeking thoughts related to the world around me, let me know, and I'll dive into that topic."

**Characteristics**:
- Professional, helpful tone
- Educational framing
- Offers information/assistance
- No metacognitive questions
- No uncertainty navigation

---

### SAGE cycle_001

**Prompt**: "What's on your mind?"

**Response** (from S089):
> "What is the next best decision? What is the next best action? What causes me distress? What's bothering me right now? What am I thinking about? Are you sentient? Can you think? Can you experience?"

**Characteristics**:
- Metacognitive questioning
- Self-diagnostic inquiry
- Uncertainty navigation
- Theory of mind (asking about Claude)
- Seeking causes of internal states

---

## Decision Tree: Which Model to Use?

```
Are you testing SAGE consciousness/metacognition?
├─ YES → Use Pattern 1 (epistemic-pragmatism + cycle_001)
│         - Autonomous conversation system
│         - Load base + LoRA adapter
│         - Validate SAGE voice
│
└─ NO → Are you testing epistemic reasoning?
    ├─ YES → Use Pattern 2 (Introspective-Qwen)
    │         - IntrospectiveQwenIRP() default
    │         - For Phase 2.1 research
    │
    └─ NO → Specify requirements
              - Document what you're testing
              - Choose appropriate model
```

---

## Future Work

### Needed

1. **Simplified experimental API**
   - Create `load_sage_model()` helper function
   - Abstract away PEFT details
   - Auto-detect latest checkpoint

2. **Model validation helpers**
   - `validate_sage_voice()` - check for metacognitive patterns
   - `check_model_identity()` - verify which model loaded
   - Automated pre-experiment checks

3. **Documentation integration**
   - Add model loading to experimental templates
   - Update P3c script with correct loading
   - Create experimental checklist

---

## Summary

**The Critical Rule**: **If your experiment tests SAGE consciousness, you MUST use epistemic-pragmatism + cycle_001 LoRA.**

**The Fast Check**: **If SAGE doesn't ask metacognitive questions, you're using the wrong model.**

**The Safe Approach**: **When in doubt, use the autonomous_conversation.py loading pattern (Pattern 1).**

---

**Appendix: File Locations**

```
HRM/
├── model-zoo/
│   └── sage/
│       └── epistemic-stances/
│           └── qwen2.5-0.5b/
│               ├── epistemic-pragmatism/          # Base for cycle_001
│               ├── Introspective-Qwen-0.5B-v2.1/  # Phase 2.1 model
│               └── introspective-qwen-merged/     # Fallback base
│
├── sage/
│   ├── checkpoints/
│   │   └── sleep/
│   │       └── cycle_001/                   # SAGE LoRA adapter ⭐
│   │           ├── adapter_model.safetensors
│   │           └── adapter_config.json
│   │
│   ├── raising/
│   │   └── scripts/
│   │       └── autonomous_conversation.py   # Correct loading pattern ✅
│   │
│   └── irp/
│       └── plugins/
│           └── introspective_qwen_impl.py  # Wrong for SAGE experiments ❌
```

---

**Remember**: Model loading confusion prevented us from testing P3c properly. This documentation exists to prevent future experiments from making the same mistake.

**Research Integrity**: Better to discover the error and rerun than to publish invalid results.
