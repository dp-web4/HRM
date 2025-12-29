# SAGE Quick Reference Guide
**Date:** November 5, 2025  
**For:** Navigating the codebase and understanding key concepts quickly

---

## TL;DR

**HRM:** Solves abstract reasoning puzzles (6.95M params)  
**SAGE:** Decides which reasoning to use (conscious orchestration kernel)  
**IRP:** Universal API all reasoning modes implement  
**VAE:** Translates between different reasoning modalities  

**Status:** All components operational. Unified loop pending.

---

## Understanding SAGE in 60 Seconds

### The Loop (What SAGE Does)
```
OBSERVE sensors → ASSESS salience (SNARC) → DECIDE resources → 
ALLOCATE budget (ATP) → EXECUTE plugins (IRP) → LEARN trust → ACT
```

### The Key Insight
**SAGE doesn't solve problems.** It decides **which specialized reasoning to invoke**:
- Quick math? → BitNet (fast, certain)
- Philosophy? → Qwen (deep, questioning)
- Visual? → Vision IRP (attention mapping)
- Remember? → Memory IRP (retrieval)

### Why It Matters
Like an OS managing processes, SAGE manages reasoning. But unlike OS, it:
1. Learns which reasoning works best for what
2. Adapts allocation based on trust scores
3. Maintains continuity across time
4. Grounds itself in biological principles

---

## Key File Quick Links

| What You Need | File | Status |
|---------------|------|--------|
| Understand SAGE | `/sage/docs/SYSTEM_UNDERSTANDING.md` | ✅ |
| Run a plugin | `/sage/irp/plugins/qwen_alive_irp.py` | ✅ |
| Learn orchestration | `/sage/irp/orchestrator.py` | ✅ |
| See memory system | `/sage/memory/irp_memory_bridge.py` | ✅ |
| Understand VAE | `/sage/compression/h_to_l_compressor.py` | ✅ |
| Check metabolic states | `/sage/core/metabolic_controller.py` | ✅ |
| Run demo | `/sage/irp/demo_sage_orchestration.py` | ✅ |
| Unified loop | `/sage/core/sage_system.py` | 🟡 (partial) |
| Full map | `/COMPREHENSIVE_HRM_SAGE_MAP.md` | ✅ |

---

## Five Most Important Concepts

### 1. IRP (Iterative Refinement Protocol)
**What:** Standard interface all plugins implement  
**Why:** Enables SAGE to orchestrate any reasoning mode  
**Pattern:** Noisy → Step 1 → Step 2 → ... → Converged → Done

### 2. Trust System
**What:** Learn which plugin is best for which task  
**How:** Track energy (quality) + efficiency (cost)  
**Result:** Allocation automatically improves over time

### 3. SNARC Salience
**What:** Score what matters (Surprise, Novelty, Arousal, Reward, Conflict)  
**Why:** Focus attention where it's needed  
**Use:** Route tasks to appropriate plugins

### 4. Metabolic States
**What:** SAGE adapts behavior to energy/fatigue level  
**States:** WAKE (broad attention), FOCUS (narrow deep), REST, DREAM (consolidate), CRISIS  
**Why:** Biologically grounded resource allocation

### 5. Compression-Trust
**What:** Larger models learn compressed representations  
**Evidence:** 14× size → only 6.59× slower (sub-linear)  
**Implication:** Knowledge ≠ memorization

---

## Common Patterns You'll See

### Pattern 1: Energy-Based Refinement
```python
state = initialize(input)
while not converged:
    state = step(state)
    energy = compute_energy(state)
    if should_halt(energy):
        break
return extract_result(state)
```

**Where:** Vision denoising, language completion, planning, memory consolidation

### Pattern 2: Trust-Weighted Selection
```python
chosen_plugin = select_by_trust(available_plugins, trust_weights)
result = chosen_plugin.execute(task)
new_trust = update_trust(trust_weights[plugin], result)
```

**Where:** Resource allocation, strategy selection, routing decisions

### Pattern 3: Hierarchical Compression
```python
h_state = strategic_reasoning(input)      # Rich representation
l_state = compress_via_vae(h_state)      # Compressed action
action = execute(l_state)
result = observe()
h_feedback = expand_via_vae_inverse(result)  # Back to strategic space
```

**Where:** H↔L loop, cross-modal translation, resource efficiency

---

## Running Key Systems

### Run Qwen IRP Plugin
```bash
cd /home/dp/ai-workspace/HRM/sage/irp
python3 << 'PYTHON'
from plugins.qwen_alive_irp import QwenAliveIRP

qwen = QwenAliveIRP()
result = qwen.step("What is cognition?", t=0)
print(f"Energy: {qwen.energy(result, t=0)}")
print(f"Result: {qwen.get_result()}")
PYTHON
```

### Run BitNet IRP Plugin
```bash
cd /home/dp/ai-workspace/HRM/sage/irp
python3 test_bitnet_irp.py
```

### Run Orchestration Demo
```bash
cd /home/dp/ai-workspace/HRM/sage/irp
python3 demo_sage_orchestration.py
```

### Understand Memory System
```bash
cd /home/dp/ai-workspace/HRM/memory_integration
python3 sage_with_snarc.py
```

---

## Current Operational Status

### ✅ Fully Working
- IRP plugin system
- Trust-weighted allocation
- SNARC salience scoring
- ATP budget management
- Metabolic state transitions
- 15+ individual plugins
- Memory systems (all 4 types)
- VAE compression
- Epistemic pragmatism models
- Knowledge distillation

### 🟡 Partially Working
- Unified SAGE loop (components exist, not integrated)
- Sensor integration (camera/audio work, puzzle conversion pending)
- Resource loading (concept exists, not dynamic)
- Cognition checkpointing (code exists, not deployed)

### ❌ Not Yet Done
- Real-time orchestration loop
- Multi-device federation
- Sensor→puzzle space VAE
- Hardware actuation (motor control)
- Large-scale real-world testing

---

## The Real Metrics

**What indicates progress:**
1. Documentation clarity (can others understand the system?)
2. Component reliability (do plugins work consistently?)
3. Learning effectiveness (do trust weights improve over time?)
4. Integration depth (how unified is the loop?)
5. Real-world validation (does it work on actual tasks?)

**What doesn't matter:**
1. Parameter count (wisdom ≠ size)
2. Training speed (insight matters more than throughput)
3. Feature completeness (working subset > broken whole)
4. Perfection (elegant enough beats perfect never)

---

## For Different Users

### I Want to Understand Architecture
**Start here:**
1. `/sage/docs/SYSTEM_UNDERSTANDING.md` (18KB overview)
2. `/COMPREHENSIVE_HRM_SAGE_MAP.md` (912 lines, complete detail)
3. `/sage/docs/architecture_map.md` (repository structure)

### I Want to Run Something
**Start here:**
1. `/sage/irp/demo_sage_orchestration.py` (see orchestration in action)
2. `/sage/irp/test_qwen_irp.py` (run a plugin)
3. `/memory_integration/sage_with_snarc.py` (see memory work)

### I Want to Add a Plugin
**Start here:**
1. Study `/sage/irp/base.py` (interface definition)
2. Copy `/sage/irp/plugins/qwen_alive_irp.py` (working example)
3. Implement the 5 required methods
4. Test with `/sage/irp/test_irp.py` pattern

### I Want to Understand Learning
**Start here:**
1. `/sage/irp/HOW_SAGE_LEARNS.md` (learning mechanics)
2. `/sage/core/sage_system.py` (trust updates in code)
3. `/private-context/autonomy-and-milestones.md` (philosophical insight)

### I Want to Fine-Tune Models
**Start here:**
1. `/private-context/finetune_epistemic_7b.py` (working example)
2. `/sage/experiments/` (epistemic training data)
3. `/model-zoo/sage/epistemic-stances/` (result models)

---

## Debugging Common Issues

### Plugin Isn't Loading
**Check:**
1. `sys.path.insert(0, 'path/to/plugins')`
2. Model path exists and is readable
3. Dependencies installed (transformers, torch, etc.)
4. GPU memory sufficient

### Energy Not Converging
**Check:**
1. Is step() actually changing state?
2. Is energy() computation correct?
3. Is halt condition too strict?
4. Increase max iterations or adjust thresholds

### Trust Weights Not Updating
**Check:**
1. Result object has required fields (energy, success, cost)
2. Learning rate reasonable (default 0.1)
3. Initial trust weights reasonable (default 1.0)
4. Updates happening in right place in loop

### Memory Growing Unbounded
**Check:**
1. Is circular buffer respecting max size?
2. Are low-salience items being pruned?
3. Is verbatim SQLite being cleaned up?
4. Memory integration params correct?

---

## Vocabulary Quick Reference

| Term | Meaning | Example |
|------|---------|---------|
| **IRP** | Plugin interface standard | Vision plugin implements IRP |
| **Plugin** | A specialized reasoning mode | BitNet, Qwen, Vision, Memory |
| **Energy** | Quality/convergence score (lower=better) | 0.1 = excellent, 0.9 = poor |
| **Trust** | How well plugin performs (0-2 range) | 1.5 = exceeds expectations, 0.5 = underperforming |
| **Salience** | What matters (SNARC 5D score) | High novelty + high reward = high salience |
| **ATP** | Computational budget units | 1000 ATP total, allocate to plugins |
| **Metabolic** | Operational state of system | WAKE vs FOCUS vs DREAM |
| **VAE** | Compression/translation layer | Converts H-space to L-space |
| **Epistemic** | Style of reasoning (pragmatism, certainty, etc) | Qwen learns epistemic pragmatism |
| **Consolidation** | Learning patterns from experience | Happens during DREAM state |

---

## Key Research Findings

### Size Inertia (Nov 5, 2025)
**Discovery:** 14× size → 6.59× slower, not 14×  
**Meaning:** Knowledge compresses efficiently  
**Use:** Larger models justified on GPU

### Epistemic Pragmatism (Nov 2, 2025)
**Discovery:** Models learn to question rather than assert  
**Meaning:** Teaching uncertainty is trainable  
**Use:** Different epistemic stances for different contexts

### Scaffolding Limits (Nov 5, 2025)
**Discovery:** Models follow but don't create new scaffolding  
**Meaning:** Some knowledge is human-domain  
**Use:** Don't expect emergent decomposition

### Compression-Trust (Aug 26, 2025)
**Discovery:** Compression ratio is trust indicator  
**Meaning:** High trust = meaning preserved through compression  
**Use:** Use compression metrics as confidence scores

---

## Next Steps to Take

### If You Want to Move the Project Forward

1. **Unify the SAGE loop**
   - File: `/sage/core/sage_system.py`
   - Task: Make `run()` method coordinate all systems
   - Impact: SAGE becomes truly operational

2. **Implement resource loading**
   - File: `/sage/core/sage_core.py`
   - Task: Load/unload plugins based on trust + ATP
   - Impact: Edge devices can run larger models

3. **Create sensor→puzzle VAE**
   - File: Needs to be created
   - Task: Learn camera/audio → puzzle space conversion
   - Impact: Real sensors can feed HRM

4. **Validate epistemic learning at scale**
   - File: `/private-context/epistemic-7b-finetune/`
   - Task: Fine-tune 7B; test generalization
   - Impact: Confirms scalability of epistemic approach

5. **Implement cognition checkpointing**
   - File: `/forum/nova/persistent-kv-demo/`
   - Task: Save/resume SAGE state across devices
   - Impact: Enables federation and continuity

---

## Resources

- **Complete map:** `/COMPREHENSIVE_HRM_SAGE_MAP.md`
- **Architecture docs:** `/sage/docs/` (41 files)
- **Tests:** `/sage/irp/test_*.py` (working examples)
- **Experiments:** `/sage/experiments/` (research data)
- **Models:** `/model-zoo/sage/` (trained weights)
- **Findings:** `/private-context/` (latest research)

---

**Last Updated:** November 5, 2025  
**Status:** All components operational. Integration advancing.
