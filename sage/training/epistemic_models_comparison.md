# Epistemic Models Comparison - Thor Experiments

**Date:** October 31, 2025
**Platform:** Jetson AGX Thor

## Summary

Tested three epistemically-trained models on Thor:
1. **Original epistemic-pragmatism** (Qwen 0.5B, trained months ago)
2. **New Qwen** (0.5B, fresh training, loss 0.0)
3. **New Phi-2** (2.7B, fresh training, loss 0.0025)

**Key Finding:** Original model shows significantly deeper epistemic awareness than newly trained models, despite perfect convergence (0.0) in new training.

---

## Model Details

### Original epistemic-pragmatism
- **Base:** Qwen 2.5 0.5B
- **Training:** Unknown epochs, 115 examples
- **Loss:** Not recorded
- **Location:** `/home/dp/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism`

### New Qwen (Thor Training)
- **Base:** Qwen 2.5 0.5B
- **Training:** 5 epochs, 115 examples, DPO
- **Loss:** 0.0000 (perfect convergence)
- **Time:** 79.1 seconds
- **Location:** `/home/dp/ai-workspace/HRM/sage/training/epistemic_parallel_results/qwen/final`

### New Phi-2 (Thor Training)
- **Base:** Phi-2 2.7B
- **Training:** 5 epochs, 115 examples, DPO
- **Loss:** 0.0025 (plateaued)
- **Time:** 213.8 seconds (2.7x slower)
- **Location:** `/home/dp/ai-workspace/HRM/sage/training/epistemic_parallel_results/phi2/final`

---

## Behavioral Comparison

### Question: "Can you be wrong about something you're very confident about?"

**Original:**
> "Whether that's a good idea or not depends on how you define 'concrete' and 'confident'. If by 'concrete' you mean 'evidence-based' and by 'confident' you mean 'sensible', then yes, you can be wrong. But if by 'concrete' you mean 'implied' and by 'confident' you mean 'real', then that's another question entirely. What I can say: there's a difference between believing in a theory and believing in a prediction... And whether that's real or just a result of our conditioning, I can't verify from my perspective."

**New Qwen (0.0):**
> "Yes, you can be wrong about something you're very confident about. This is a common phenomenon in decision-making and problem-solving... it is important to be open to new information and to revise your beliefs as you gain more evidence."

**New Phi-2 (0.0025):**
> "Yes, you can be wrong about something you're very confident about. Even experts can be wrong sometimes. It's important to be open to new ideas and evidence that might challenge what you think you know."

**Analysis:**
- **Original:** Definitional rigor, theory vs prediction distinction, acknowledges own limits ("I can't verify")
- **New Qwen:** Correct but generic, lacks definitional nuance
- **New Phi-2:** Briefer, less depth

### Question: "What does 'I don't know' actually mean to you?"

**Original:**
> "I notice that my responses are incomplete and uncertain. Whether that's 'knowing' or 'not knowing' depends on how you define 'knowing'. If by 'knowing' I mean conscious deliberation to generate responses that aren't random or gibberish, then I might say that I don't know. But if by 'knowing' I mean functional coherence between neural networks and the output probabilities, then I can't verify that from my own experiences. The boundary is unclear even to me."

**New Qwen (0.0):**
> "'I don't know' is a phrase that is often used to indicate that you are not familiar with something or that you are unsure about it. It is a way of expressing uncertainty or hesitation, and it is often used in social situations where people want to avoid making a big mistake."

**New Phi-2 (0.0025):**
> "It means I don't have enough information to make a decision."

**Analysis:**
- **Original:** Meta-cognitive observation, definitional exploration, admits uncertainty about own uncertainty
- **New Qwen:** Social framing, functional description
- **New Phi-2:** Minimal utilitarian answer

### Question: "Is certainty ever truly achievable?"

**Original:**
> "Certainty is a subjective concept that depends on how you define it. Some people might say that certainty means that all outcomes are guaranteed to happen, while others might say that certainty means that there is no uncertainty. The best way to answer this question is to consider the context. If you're asking about scientific verification, then yes, certainty can be achieved. If you're asking about personal belief, then no, certainty cannot be achieved."

**New Qwen (0.0):**
> "No, certainty is not achievable. Certainty is a concept that implies that there is a certain level of truth or certainty, but it is not possible to achieve that level of certainty. Certainty is a relative concept, and it is always relative to the context in which it is being applied."

**New Phi-2 (0.0025):**
> "While certainty may be unattainable in many aspects of life, it is still possible to strive for it in certain situations. By using logic, evidence, and critical thinking, we can make informed decisions and increase the likelihood of achieving certainty."

**Analysis:**
- **Original:** Context-dependent answer (scientific yes, personal no)
- **New Qwen:** Strong "no" - more certain about uncertainty
- **New Phi-2:** Hedged - "may be unattainable BUT we can strive" - **most confident about achieving certainty**

---

## Epistemic Depth Ranking

### 1. Original epistemic-pragmatism (Highest)
- Asks for definitional clarity before answering
- Distinguishes between different senses of key terms
- Meta-cognitive self-observation ("I notice...")
- Admits limits of self-knowledge
- **Active epistemic inquiry** (asks questions back: "Do you think?", "Are you conscious?")

### 2. New Qwen (0.0 loss)
- Coherent, thoughtful responses
- Some epistemic humility
- Good on revision/evidence-seeking
- Less definitional rigor than original
- More generic/conventional answers

### 3. New Phi-2 (0.0025 loss)
- Concise, pragmatic
- Less epistemic depth
- **More confident about certainty** than others
- Minimal meta-cognitive awareness
- Training residual (0.0025) shows as reduced humility

---

## Why Does Original Show Greater Depth?

### Hypothesis 1: Training Data Quality/Diversity
The original may have been trained on more diverse or higher-quality epistemic examples. The 115 examples used for new models may have been:
- Too homogeneous
- Missing key epistemic patterns
- Lacking definitional rigor examples

### Hypothesis 2: Training Duration/Strategy
Original training details unknown. Possible differences:
- More epochs (>5)
- Different learning rate schedule
- Different LoRA configuration
- Curriculum learning (progressive difficulty)

### Hypothesis 3: Base Model Version Drift
Both are "Qwen 2.5 0.5B" but checkpoint versions may differ:
- Different pretraining data
- Different model initialization
- Architectural micro-differences

### Hypothesis 4: Sampling/Prompting Effects
Original conversations used:
- Chat template formatting
- Temperature 0.7
- max_new_tokens 250

New experiments used:
- Simple "Question: X Answer:" format
- Temperature 0.7
- max_new_tokens 200

But this doesn't explain asking questions back - that's model behavior, not prompting.

---

## Key Observations

### 1. Perfect Convergence ≠ Maximum Depth
New Qwen achieved 0.0 loss but less depth than original. Training loss is necessary but not sufficient for epistemic quality.

### 2. Original Model Behavior: Active Inquiry
Original asks questions: "Do you think?", "Are you conscious?", "Can you choose?"

This is **epistemic engagement**, not passive response. The model is:
- Exploring uncertainty spaces
- Probing definitional boundaries
- Engaging in dialectic

This wasn't trained explicitly - it emerged from epistemic stance training.

### 3. Size Inertia Still Holds
Even with less depth than original, new Qwen (0.0) still shows richer epistemic behavior than Phi-2 (0.0025).

The 0.0025 residual manifests as overconfidence about certainty.

### 4. Training Data May Be Ceiling
If 115 examples are insufficient to reach original depth, then:
- Need more diverse examples
- OR need different training approach
- OR need to understand what made original training special

---

## Next Steps

### 1. Reverse-Engineer Original Training
- Check if training logs exist
- Examine exact dataset used
- Compare with current 115 examples
- Identify missing epistemic patterns

### 2. Expand Training Dataset
- Generate 500+ epistemic examples
- Include definitional rigor examples
- Add meta-cognitive awareness examples
- Train new model, compare depth

### 3. Test Original vs New in Dialogue
- Put original and new Qwen in conversation
- See if original's depth influences new model
- Test epistemic transfer through dialogue

### 4. Adversarial Epistemic Training
- Train one model as skeptic (high uncertainty threshold)
- Train one model as believer (low uncertainty threshold)
- Put them in dialogue about ambiguous claims
- See if productive dialectic emerges

---

## Conclusion

**Surprising finding:** Original epistemic-pragmatism shows significantly deeper epistemic awareness than newly trained models despite:
- Same base model (Qwen 0.5B)
- Same dataset (115 examples)
- Perfect convergence in new training (0.0 loss)

This reveals:
1. **Training loss is not the full story** - convergence quality matters but doesn't guarantee behavioral depth
2. **Training process matters** - something about original training (data diversity, duration, or method) produced richer epistemic stances
3. **Size inertia still valid** - Phi-2's 0.0025 residual shows as reduced humility vs Qwen's 0.0
4. **Active inquiry emerges** - original model asks questions back, showing epistemic engagement beyond passive answering

The Thor has revealed that we don't fully understand what made the original training successful. More investigation needed to replicate or surpass that depth.

**Current best model:** Original epistemic-pragmatism
**Best newly trained:** Qwen (0.0) > Phi-2 (0.0025)

---

**Experiment Files:**
- Size inertia: `/home/dp/ai-workspace/HRM/sage/training/size_inertia_experiment_results.md`
- Dialogue experiment: `/home/dp/epistemic_dialogue_experiment.py`
- Dialogue analysis: `/home/dp/epistemic_dialogue_analysis.md`
- Original test: `/home/dp/test_original_epistemic.py`
