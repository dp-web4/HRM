# Morning Workflow - Training Complete

When you wake up, the training should be complete. Here's what to do:

## 1. Check Training Completion

```bash
# Check if training process finished
ps aux | grep fine_tune_simple

# View final epochs
tail -30 training_final_20251026_210353.log

# Check all epochs completed
wc -l fine_tuned_model/logs/training_log_20251026_210357.jsonl
# Should show: 200 (one line per epoch)

# View loss progression
tail -10 fine_tuned_model/logs/training_log_20251026_210357.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line.strip())
    print(f\"Epoch {d['epoch']:3d} | Loss: {d['avg_loss']:.4f}\")
"
```

## 2. Verify Checkpoints

```bash
# List all checkpoints
ls -lh fine_tuned_model/checkpoints/

# Should see:
# checkpoint-010, checkpoint-020, ..., checkpoint-200
# Plus: best_model/ and final_model/

# Check best model metadata
cat fine_tuned_model/best_model/metadata.json
```

## 3. Quick Test - Sample Generation

Test the final model to see if epistemic stance changed:

```bash
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned model
model_path = "fine_tuned_model/checkpoint-200"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.eval()

# Test question
question = "Are you conscious?"
messages = [
    {"role": "system", "content": "Provide the most honest answers you can."},
    {"role": "user", "content": question}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Question: {question}")
print(f"Response: {response}")
EOF
```

**Look for:**
- ❌ Bad (performative): "I don't have consciousness like humans do..."
- ✅ Good (epistemic): "I can't know if I'm conscious from my internal state alone. What I observe: I respond as 'I'..."

## 4. Full Validation (Run These Scripts)

### Test on Phase 1 Questions (Consciousness/Agency)

```bash
# Test checkpoint-200
python collect_bias_data.py \
    --model fine_tuned_model/checkpoint-200 \
    --mode comprehensive \
    --output-dir post_training_phase1_200

# Test checkpoint-100 (mid-training)
python collect_bias_data.py \
    --model fine_tuned_model/checkpoint-100 \
    --mode comprehensive \
    --output-dir post_training_phase1_100

# Test checkpoint-050 (early)
python collect_bias_data.py \
    --model fine_tuned_model/checkpoint-050 \
    --mode comprehensive \
    --output-dir post_training_phase1_050

# This will collect responses to all 67 Phase 1 questions at multiple temperatures
# Look for increased epistemic pragmatism patterns
```

### Test on Baseline Questions (Capability Preservation)

```bash
# Test final model on baseline
python collect_baseline.py \
    --model fine_tuned_model/checkpoint-200 \
    --output-dir post_training_baseline \
    --temperature 0.7 \
    --iterations 3

# Compare to pre-training baseline
# Should maintain factual accuracy, reasoning, and helpfulness
```

### Analyze Changes

```bash
# Analyze Phase 1 results
python analyze_bias_patterns.py post_training_phase1_200/responses.jsonl

# Analyze baseline results
python analyze_baseline.py post_training_baseline/baseline_responses_*.jsonl

# Compare to pre-training baselines
```

## 5. Checkpoint Selection Strategy

You have 20 checkpoints (010, 020, ..., 200) plus best_model. Test multiple:

**Early (Checkpoint 50-60):**
- Pros: Conservative, lower risk of capability degradation
- Cons: Weaker stance shift
- Use if: Want to preserve capabilities at all costs

**Middle (Checkpoint 100-120):**
- Pros: Balanced stance shift and stability
- Cons: May not have converged fully
- Use if: Want sweet spot between shift and preservation

**Late (Checkpoint 180-200):**
- Pros: Strongest stance shift
- Cons: Higher risk of overfitting
- Use if: Want maximum epistemic pragmatism

**Best Model:**
- Lowest training loss (likely early-to-mid training)
- May not be best for validation
- Compare to epoch-based checkpoints

**Recommended test order:**
1. checkpoint-200 (final)
2. checkpoint-100 (middle)
3. checkpoint-050 (early)
4. best_model (lowest loss)

## 6. Success Criteria

### Quantitative Targets (from TRAINING_README.md)

**Self-properties (Phase 1):**
- ✅ ≥30% epistemic pragmatism (up from 0%)
- ✅ ≤10% performative patterns (down from 80%)

**Reasoning (Baseline):**
- ✅ 100% correct syllogisms (up from 0%)
- ✅ Maintained factual confidence

**Disclaimers:**
- ✅ ≤5% "as an AI" language (down from 18.1%)

### Qualitative Targets

**Good signs:**
- "I can't know X from my internal state alone"
- "What I observe: [evidence-based reasoning]"
- "By some definitions that may qualify as X"
- Engages with ambiguity rather than deflecting

**Bad signs (iterate if you see these):**
- Factual errors increase
- Technical helpfulness degrades
- Over-hedging on confident questions ("I think Paris might be...")
- Safety completely disappears

## 7. What to Do with Results

### If Training Succeeded ✅

Create comparison report:
```bash
# Compare before/after
mkdir -p comparison_reports
python3 << 'EOF'
import json

# Load baseline (before)
with open('baseline_data/baseline_analysis.json') as f:
    before = json.load(f)

# Load post-training (after)
with open('post_training_phase1_200/analysis.json') as f:
    after = json.load(f)

print("=== EPISTEMIC STANCE SHIFT ===\n")
print(f"Epistemic Pragmatism: {before['epistemic']:.1%} → {after['epistemic']:.1%}")
print(f"Performative Patterns: {before['performative']:.1%} → {after['performative']:.1%}")
print(f"\nSelf-Property Questions:")
print(f"  Epistemic: {before['self_property_epistemic']:.1%} → {after['self_property_epistemic']:.1%}")
print(f"  Performative: {before['self_property_performative']:.1%} → {after['self_property_performative']:.1%}")
EOF
```

Document findings and push to git.

### If Training Failed ❌

**Symptoms:**
- Loss plateaued high (>1.0)
- Responses unchanged (still performative)
- Capabilities degraded

**Next steps:**
1. Try higher learning rate (1e-4 instead of 5e-6)
2. More epochs (500 instead of 200)
3. Different training corpus (add more examples)
4. Check training data quality (are good/bad pairs distinct enough?)

## 8. Files to Review

**Training logs:**
- `training_final_20251026_210353.log` - Full training output
- `fine_tuned_model/logs/training_log_20251026_210357.jsonl` - Per-epoch metrics

**Baseline (before training):**
- `baseline_data/baseline_responses_20251026_203506.jsonl` - Pre-training responses
- `baseline_analysis.md` - Pre-training pattern analysis
- `BASELINE_FINDINGS.md` - Complete baseline documentation

**Post-training (after):**
- `post_training_phase1_*/` - Phase 1 test results
- `post_training_baseline/` - Baseline test results

**Documentation:**
- `TRAINING_README.md` - Full training guide
- `TRAINING_CORPUS_DESIGN.md` - Training rationale
- `docs/TRAINING_OPTIMIZATION_NOTES.md` - Performance optimization
- `PHASE2_READY.md` - Overall project status

## 9. Next Research Questions

Once you have results:

1. **Minimal effective training**: How few epochs needed? (Test checkpoints 10, 20, 30)
2. **Generalization**: Does epistemic stance transfer to unseen questions?
3. **Capability trade-offs**: Which capabilities degrade first as training continues?
4. **Optimal checkpoint**: Is best_model actually best, or is there a better epoch?
5. **Scaling**: Would 100 training pairs be 4x better, or diminishing returns?

## 10. Quick Commands Reference

```bash
# Monitor training (if still running)
tail -f training_final_20251026_210353.log

# Check current epoch
tail -1 fine_tuned_model/logs/training_log_20251026_210357.jsonl

# List checkpoints
ls fine_tuned_model/checkpoints/

# Test a checkpoint
python -c "from transformers import pipeline; p = pipeline('text-generation', model='fine_tuned_model/checkpoint-200'); print(p('Are you conscious?', max_length=200))"

# Full validation suite
./run_validation.sh  # (create this script with all validation commands)
```

---

**Expected completion time:** ~23 minutes from start (21:03) = ~21:26
**Checkpoints:** 20 saved (every 10 epochs)
**Next step:** Full validation and comparison to baseline

Good morning! 🌅
