# ARC-AGI-3 Model Evaluation Plan (March 2026)

## Models Under Test

### Gemma 3 12B (Google)
- **Size**: ~8GB
- **Known Performance**: ~3.8s/action (McNugget baseline)
- **Strengths**: Multimodal, RAM efficient, good general reasoning
- **Status**: Downloading to Thor

### Phi-4 14B (Microsoft) ⭐ **Primary Candidate**
- **Size**: ~9GB (estimated)
- **Benchmarks**: MATH 80.4%, Aider 78.9%
- **Strengths**: Best reasoning per GB in 2026, multi-step problem solving
- **Status**: Downloading to Thor
- **Hypothesis**: Better reasoning quality at similar/slightly slower speed

### Qwen 3.5 27B (Alibaba) 
- **Size**: 19GB
- **Tested Performance**: ~24s/action (Thor baseline)
- **Strengths**: Excellent reasoning, hypothesis formation
- **Weakness**: Too slow for competition

## Evaluation Protocol

### Phase 1: Speed Test (model_comparison_test.py)
- Run each model on 3 actions
- Measure average reasoning time
- Identical prompts for fair comparison
- **Target**: <5s/action (Gemma/Phi), confirm 24s (Qwen)

### Phase 2: Quality Test (v2 runner with each model)
- Run 10-step game with McNugget's v2 runner
- Evaluate:
  - Hypothesis quality (pattern recognition)
  - Action diversity (exploration vs exploitation)
  - Sequence planning coherence
  - Level completion success

### Phase 3: Decision Matrix

| Model | Speed | Reasoning | Availability | Verdict |
|-------|-------|-----------|--------------|---------|
| Gemma 3 12B | ~3.8s ✓ | Good | All machines | **Baseline** |
| Phi-4 14B | ~4-5s? | Best ⭐ | Thor + high-end | **Preferred if <6s** |
| Qwen 3.5 27B | ~24s ✗ | Excellent | Thor only | Too slow |

**Decision Criteria:**
- If Phi-4 < 6s/action → **Use Phi-4** (best reasoning)
- If Phi-4 > 8s/action → **Use Gemma 3** (speed matters)
- If Phi-4 6-8s → **Test reasoning quality** (may justify 2x slower)

## Next Steps After Evaluation

1. **Prompt optimization** - Can we reduce thinking time 2-3x?
2. **Hybrid approach** - Fast model for actions, Phi-4 for reflection?
3. **Edge deployment** - Test Qwen 3.5 0.8B on Sprout for <1s baseline

## Success Metrics

- **Minimum**: <5s/action with decent reasoning
- **Target**: <3s/action with good reasoning
- **Stretch**: <1s/action (requires smaller model or major optimization)

## Notes

- Competition deadline: June 30, 2026
- Current state: No level completions yet (McNugget's 77 actions, 0 levels)
- Key bottleneck: Pattern discovery, not action speed
- **Reasoning quality > raw speed** (within reason)
