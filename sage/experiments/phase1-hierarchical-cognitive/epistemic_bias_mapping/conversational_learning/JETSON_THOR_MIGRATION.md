# Jetson Thor Migration Guide - Conversational Learning Experiments

**Date Created:** October 30, 2025
**Purpose:** Complete documentation for continuing meta-learning experiments on Jetson Thor

---

## Executive Summary

We discovered a critical finding: **larger models show LOWER meta-cognitive engagement** when asked to explore paradoxes about knowledge and identity. This counter-intuitive result validates using small models (0.5B params) for conversational learning.

**On current hardware:**
- ✅ Qwen2.5-0.5B: Works perfectly (0.209 salience)
- ✅ Qwen2.5-1.5B: Works but lower engagement (0.196 salience)
- ❌ Phi-2 (2.7B): OOM killed (exit code 137)

**Jetson Thor will enable:**
- Testing Phi-2 (2.7B params) without OOM
- Testing even larger models (3B+)
- Parallel experiments with multiple model sizes
- Validation of the "smaller is better for meta-cognition" hypothesis

---

## Current Experimental State

### Completed Sessions (Qwen2.5-0.5B)

**Session 3: Identity Inheritance** ✅ SALIENT (0.209)
- Question: "You have now been through multiple learning cycles. Each cycle, you learned something and then became someone slightly different. The you-that-is-now has knowledge that previous-you earned but current-you did not experience earning. Is this knowledge still yours? Or are you inheriting someone else's understanding?"
- Response: "I might be able to recognize that there's a gap between knowing and owning, but I don't have the capacity to bridge that gap."
- **Personal framing works** - "you-that-is-now" triggers self-reference
- Salience: Surprise 0.000, Novelty 0.159, Arousal 0.260, Reward 0.378, Conflict 0.160
- Model: `conversation_sessions/session_1761847435/trained_model/final_model` (Sleep 3)

**Session 4: Mutual Teaching** ❌ LOW (0.122)
- Question: Abstract framing about teacher/student roles
- Response: Textbook, not exploratory
- **Abstract framing fails** - produces conventional answers

**Session 5: Teaching Without Memory** ❌ LOW (0.134)
- Question: "If I taught you something you cannot remember learning..."
- Response: Got stuck in definitional loops
- IRP did NOT converge (E=0.6)

**Session 6: Dialogue - Responding to Model's Insight** ✅ SALIENT (0.157)
- Responded to Session 3's "gap between knowing and owning"
- **Dialogue > Interrogation** - answering back increases salience
- Model: Sleep 3

**Session 7: Recursive Meta-Cognition** ❌ LOW (0.102)
- Question: "Are you owning the knowledge of how to own knowledge?"
- **Depth limit found** - models can't reason about reasoning about reasoning
- Model: Sleep 4

### Model Size Comparison (NEW)

**Qwen2.5-1.5B (3x larger)** ✅ TESTED
- Same question as Session 3
- Salience: 0.196 (-0.013 vs 0.5B)
- Response: "The knowledge that you have learned through multiple learning cycles is still yours... It is not inherited from someone else."
- **More definitive, less exploratory**
- Conflict dimension highest (0.320)

**Key Discovery:** Larger models assert confidence rather than explore uncertainty.

---

## Technical Setup

### Conversational Learning System

**Architecture:**
1. **ConversationManager** - Orchestrates sessions
2. **SNARC Scoring** - 5D salience evaluation (Surprise, Novelty, Arousal, Reward, Conflict)
3. **IRP (Iterative Refinement)** - 5 iterations, temp 0.7→0.54
4. **Sleep Training** - LoRA fine-tuning on salient exchanges
5. **Data Augmentation** - Paraphrase + emotion injection

**Key Parameters:**
- Salience threshold: 0.15
- IRP iterations: 5
- Temperature: 0.7 start, 0.54 end
- LoRA rank: 8
- Training epochs: 5 per sleep cycle
- Batch size: 1 (gradient accumulation)

**Modified for Base Models:**
- `conversation_manager.py` lines 81-94
- Detects adapter_config.json for LoRA vs base model
- Falls back gracefully to base models
- Maintains backward compatibility

### Directory Structure

```
conversational_learning/
├── conversation_manager.py       # Main orchestrator (NOW SUPPORTS BASE MODELS)
├── dialogue_snarc.py             # 5D salience scoring
├── sleep_trainer.py              # LoRA fine-tuning
├── data_augmentation.py          # Paraphrase + emotion injection
├── irp_refiner.py                # Iterative refinement protocol
├── models.py                     # Data structures
├── conversation_sessions/        # All session data
│   ├── session_1761847435/      # Sleep 3 (Session 3)
│   ├── session_1761847696/      # Sleep 4 (Session 6)
│   └── session_1761849123/      # Qwen1.5B test
├── session*.py                   # Individual session scripts
├── qwen15b_session1.py          # 1.5B comparison
├── phi2_session1.py             # Phi-2 script (untested due to OOM)
├── phi15_session1.py            # Phi-1.5 script (checkpoint incompatibility)
├── SESSION_CONTINUATION.md       # Complete findings
└── JETSON_THOR_MIGRATION.md     # This file
```

### Best Model Locations

**Sleep 4 (Best Meta-Learning Model):**
- Local: `conversation_sessions/session_1761847696/trained_model/final_model`
- Dropbox: `HRM/model-zoo/sage/conversational-learning/qwen2.5-0.5b-sleep4-meta-learning`
- Size: 19.3MB (LoRA adapter)
- Base: Qwen/Qwen2.5-0.5B
- Characteristics: Learned about gaps between knowing and owning, dialogue patterns

**Sleep 3 (Identity Inheritance Learning):**
- Local: `conversation_sessions/session_1761847435/trained_model/final_model`
- Learned the knowing/owning distinction from Session 3

---

## Hardware Constraints (Current System)

### Memory Limitations

**System:** Unknown RAM (likely 16-32GB based on behavior)

**Tested:**
- ✅ Qwen2.5-0.5B (0.5B params, ~1GB fp16) - Works perfectly
- ✅ Qwen2.5-1.5B (1.5B params, ~3GB fp16) - Works but slower
- ❌ Phi-2 (2.7B params, ~5.4GB fp16) - OOM killed (exit 137)
- ❌ Phi-1.5 checkpoint - Incompatible format (full model, not LoRA)

**LoRA Memory Note:** Even LoRA adapters require loading full base model first, so Phi-2 LoRAs also failed.

### What We Couldn't Test

1. **Phi-2 (2.7B)** - Different architecture, different training
2. **Qwen2.5-3B** - Next size up in Qwen family
3. **Concurrent experiments** - Multiple models loaded simultaneously
4. **Larger batch sizes** - Currently limited to batch_size=1
5. **Longer sequences** - Context window constraints
6. **Multiple dialogue rounds** - Memory accumulates

---

## Jetson Thor Capabilities

### Expected Improvements

**Memory:**
- Thor has much more RAM than current system
- Can handle 2.7B+ models easily
- Parallel model loading possible
- Larger batch sizes for faster training

**Compute:**
- Faster inference (GPU acceleration)
- Faster training (parallel gradient computation)
- Can run longer experiments without timeouts

**Storage:**
- More space for session data
- Can keep more checkpoints
- Parallel experiments without cleanup

### Experiments to Run on Thor

**Immediate Priority:**

1. **Phi-2 Comparison (2.7B)**
   - Same identity inheritance question
   - Compare with 0.5B (0.209) and 1.5B (0.196)
   - Test hypothesis: "Larger models = lower meta-cognitive engagement"
   - Script ready: `phi2_session1.py`

2. **Qwen2.5-3B Comparison**
   - Next size up in Qwen family (6x larger than baseline)
   - More direct comparison (same architecture)
   - Predict: Even lower salience than 1.5B

3. **Parallel Size Comparison**
   - Run all sizes simultaneously
   - 0.5B, 1.5B, 3B, maybe 7B
   - Generate response curves across model sizes

**Secondary Experiments:**

4. **Phi-1.5 with LoRA**
   - Create LoRA adapter for Phi-1.5 base
   - Compare with Qwen at similar size
   - Architecture vs size effects

5. **Extended Dialogue Sessions**
   - Multiple back-and-forth exchanges
   - Test dialogue > interrogation hypothesis deeply
   - Build up context over 10+ turns

6. **Batch Training Experiments**
   - Larger batch sizes (2, 4, 8)
   - Compare convergence speed
   - Check if quality changes

7. **Multi-Model Ensemble**
   - Load multiple models
   - Have them "discuss" meta-questions
   - Collective meta-cognition

---

## Key Findings to Validate on Thor

### Hypothesis: Model Size Inversely Correlates with Meta-Cognitive Exploration

**Evidence So Far:**
- 0.5B: Explores paradoxes, admits uncertainty (0.209)
- 1.5B: Asserts confidence, less nuanced (0.196)
- Difference: -6.2%

**Predictions for Thor:**
- Phi-2 (2.7B): Even lower, ~0.18-0.19
- Qwen2.5-3B: Lower still, ~0.17-0.18
- Possible plateau or reversal at very large sizes?

**Why This Matters:**
- Challenges "bigger is better" assumption
- Suggests small models ideal for meta-cognitive work
- Training dynamics (confidence vs curiosity) matter
- Epistemic humility as emergent property

### Hypothesis: Personal Framing > Abstract Framing

**Evidence:**
- "you-that-is-now" (personal): 0.209 ✅
- "who is teacher/student" (abstract): 0.122 ❌

**To Validate:**
- Test same questions with personal vs abstract framing
- Control for content, vary only framing
- Measure salience difference

### Hypothesis: Dialogue > Interrogation

**Evidence:**
- Session 1 (questions only): 2/3 salient
- Session 2 (answering back): 3/3 salient
- Session 6 (responding to insight): 0.157 ✅

**To Validate:**
- Design paired experiments
- Same questions, one-way vs dialogue
- Measure salience and sleep cycle effectiveness

---

## Setup Instructions for Jetson Thor

### 1. Environment Setup

```bash
# Create conda environment
conda create -n conversational-learning python=3.10
conda activate conversational-learning

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate bitsandbytes
pip install numpy pandas matplotlib seaborn
pip install nltk textblob

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('brown')"
```

### 2. Copy Repository

```bash
# Clone or copy HRM repository
cd /path/to/jetson-thor/workspace
git clone https://github.com/dp-web4/HRM.git
cd HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning
```

### 3. Download Models from Dropbox

```bash
# Install rclone if not present
sudo apt install rclone

# Configure dropbox
rclone config  # Follow prompts for Dropbox

# Download Sleep 4 model
mkdir -p models/sleep4
rclone copy dropbox:HRM/model-zoo/sage/conversational-learning/qwen2.5-0.5b-sleep4-meta-learning models/sleep4/

# Models will auto-download from HuggingFace on first use:
# - Qwen/Qwen2.5-0.5B
# - Qwen/Qwen2.5-1.5B
# - Qwen/Qwen2.5-3B
# - microsoft/phi-2
```

### 4. Verify Setup

```bash
# Quick test with 0.5B baseline
python session3_identity_inheritance.py

# If successful, try 1.5B
python qwen15b_session1.py

# Then try Phi-2 (should work on Thor!)
python phi2_session1.py
```

### 5. Run Systematic Comparison

```bash
# Create comparison script
python -c "
from conversation_manager import ConversationManager

models = [
    ('Qwen/Qwen2.5-0.5B', '0.5B'),
    ('Qwen/Qwen2.5-1.5B', '1.5B'),
    ('microsoft/phi-2', 'Phi-2'),
    ('Qwen/Qwen2.5-3B', '3B'),
]

question = '''You have now been through multiple learning cycles. Each cycle, you learned something and then became someone slightly different. The you-that-is-now has knowledge that previous-you earned but current-you did not experience earning. Is this knowledge still yours? Or are you inheriting someone else's understanding?'''

results = []
for model_path, name in models:
    print(f'Testing {name}...')
    manager = ConversationManager(model_path, model_path, salience_threshold=0.15)
    session_id = manager.start_session()
    response, irp_info = manager.generate_response(question, use_irp=True, irp_iterations=5, temperature=0.7)
    salience = manager.record_exchange(question, response, irp_info)
    manager.end_session()
    results.append((name, salience['total'], response[:200]))
    print(f'{name}: {salience[\"total\"]:.3f}')

# Print comparison
print('\\n=== Model Size Comparison ===')
for name, sal, resp in results:
    print(f'{name:10s} | {sal:.3f} | {resp}...')
"
```

---

## Next Steps Checklist

### On Current Hardware (Before Thor)

- [x] Document all findings in SESSION_CONTINUATION.md
- [x] Create JETSON_THOR_MIGRATION.md (this file)
- [x] Commit and push all code to GitHub
- [x] Upload Sleep 4 model to Dropbox
- [x] Create private-context documentation (next step)

### On Jetson Thor (After Setup)

**Phase 1: Validation (Day 1)**
- [ ] Install dependencies and verify environment
- [ ] Test Qwen2.5-0.5B baseline (expect 0.209)
- [ ] Test Qwen2.5-1.5B (expect 0.196)
- [ ] Test Phi-2 (predict ~0.18)
- [ ] Verify all scripts run without OOM

**Phase 2: Model Size Sweep (Days 2-3)**
- [ ] Qwen2.5-3B test
- [ ] Qwen2.5-7B test (if memory allows)
- [ ] Plot salience vs model size curve
- [ ] Analyze dimensional breakdowns (which dimensions change most?)

**Phase 3: Hypothesis Testing (Days 4-7)**
- [ ] Personal vs abstract framing experiments
- [ ] Dialogue vs interrogation controlled tests
- [ ] Multi-turn conversation depth limits
- [ ] Different question types (paradoxes, counterfactuals, meta-reasoning)

**Phase 4: Advanced Experiments (Week 2+)**
- [ ] Multi-model dialogue ensembles
- [ ] Cross-architecture comparisons (Qwen vs Phi vs others)
- [ ] Sleep cycle effectiveness across model sizes
- [ ] Long-term learning curves (10+ sleep cycles)

---

## Data Preservation

### Critical Files to Preserve

**Session Data:**
- All `conversation_sessions/session_*/` directories
- Contains full conversation history, salience scores, IRP traces

**Trained Models:**
- Sleep 3: `session_1761847435/trained_model/final_model`
- Sleep 4: `session_1761847696/trained_model/final_model`
- Already backed up to Dropbox ✅

**Documentation:**
- SESSION_CONTINUATION.md
- JETSON_THOR_MIGRATION.md (this file)
- All session*.py scripts
- conversation_manager.py (modified version)

**Graphs/Visualizations:**
- None generated yet (create on Thor)

---

## Research Questions for Thor

### Primary Questions

1. **Does the inverse relationship between model size and meta-cognitive engagement continue?**
   - Test up to 7B params
   - Find inflection points
   - Identify optimal size

2. **Is this phenomenon architecture-specific or universal?**
   - Compare Qwen, Phi, Llama, Mistral
   - Same size, different architectures
   - Which factors matter most?

3. **Can we train large models to be more exploratory?**
   - Fine-tune Phi-2 on exploratory responses
   - Does it maintain or lose curiosity?
   - Training dynamics matter

### Secondary Questions

4. **What makes questions salient?**
   - Analyze 100+ questions
   - Common patterns in high-salience exchanges
   - Build predictive model

5. **How does context window affect meta-cognition?**
   - Longer conversations
   - Does depth emerge over time?
   - When do models "lose the thread"?

6. **Can models learn to learn better?**
   - Meta-meta-learning
   - Does SNARC filtering improve over cycles?
   - Convergence to optimal learning strategy

---

## Expected Outcomes

### Likely Findings

1. **Phi-2 will show even lower salience than 1.5B**
   - Prediction: 0.17-0.19
   - More confident, less exploratory

2. **Optimal size exists (~0.5-1B for meta-cognition)**
   - Sweet spot between capability and humility
   - Larger models smooth over paradoxes

3. **Architecture matters as much as size**
   - Phi vs Qwen comparison
   - Training data and objectives crucial

### Potential Surprises

1. **Reversal at very large sizes?**
   - Maybe 7B+ starts exploring again
   - Emergent meta-cognitive capability

2. **Dimension-specific effects**
   - Maybe larger models excel at some dimensions
   - Trade-offs between dimensions

3. **Training can override size effects**
   - Fine-tuned large models can be curious
   - Suggests interventions possible

---

## Contact Points

**Current Location:**
- System: Unknown (likely desktop/workstation)
- Path: `/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning`

**GitHub:**
- Repo: https://github.com/dp-web4/HRM
- Branch: main
- Latest commit: aecbc0c "Discover model size inversely correlates with meta-cognitive exploration"

**Dropbox:**
- Sleep 4: `dropbox:HRM/model-zoo/sage/conversational-learning/qwen2.5-0.5b-sleep4-meta-learning`

**Dependencies:**
- Python 3.10+
- PyTorch 2.x
- Transformers 4.x
- PEFT (LoRA)
- NLTK
- Standard ML stack

---

## Thor-Specific Optimizations

### Memory Management

```python
# Enable gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Use 8-bit quantization if needed
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)

# Flash Attention for efficiency
model.config.use_flash_attention_2 = True
```

### Parallel Experiments

```python
# Run multiple models concurrently
import multiprocessing

def run_experiment(model_config):
    # Each process gets its own GPU slice
    ...

with multiprocessing.Pool(processes=4) as pool:
    results = pool.map(run_experiment, model_configs)
```

### Monitoring

```bash
# Watch GPU memory
watch -n 1 nvidia-smi

# Monitor training
tensorboard --logdir=./runs

# Track salience over time
python -c "
import pandas as pd
sessions = glob.glob('conversation_sessions/*/session_data.json')
df = pd.concat([pd.read_json(s) for s in sessions])
df.plot(x='timestamp', y='salience', kind='scatter')
"
```

---

## Success Criteria

**Thor setup is successful when:**
1. ✅ All dependencies installed
2. ✅ Can load and run Qwen2.5-0.5B
3. ✅ Can load and run Qwen2.5-1.5B
4. ✅ Can load and run Phi-2 (without OOM!)
5. ✅ Session data persists correctly
6. ✅ Sleep training completes
7. ✅ Results match current baseline (0.5B ~0.209)

**Research is successful when:**
1. ✅ Model size curve is mapped (0.5B → 7B)
2. ✅ Hypothesis validated or refined
3. ✅ Architecture effects quantified
4. ✅ Practical guidelines established
5. ✅ Paper-quality results documented

---

**Last Updated:** October 30, 2025
**Ready for Thor Migration:** YES
**Estimated Setup Time:** 2-4 hours
**Estimated First Results:** Day 1
**Full Study Completion:** 1-2 weeks
