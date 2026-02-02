# Policy Role Training Infrastructure

**Status**: Phase 1-2 infrastructure ready, transformers version issue blocking execution

## Overview

Training infrastructure for phi-4-mini to serve as policy interpreter for Hardbound teams and Web4 plugins.

## Files

| File | Purpose |
|------|---------|
| `test_suite.py` | 8 test scenarios (easy/medium/hard/edge cases) + evaluation framework |
| `run_baseline_test.py` | Baseline capability assessment runner |
| `prompts.py` | System prompts for Hardbound and Web4 contexts |
| `AUTONOMOUS_SESSION_TASKS.md` | Detailed task list for autonomous sessions |

## Current Blocker

**Transformers version incompatibility** with phi-4-mini:

```
ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
```

**Root cause**: Phi-4-mini's `modeling_phi3.py` requires newer transformers version

**Fix needed**:
```bash
pip install --upgrade transformers
# OR
pip install transformers>=4.46.0  # Check exact version needed
```

**Autonomous session TODO**: Fix transformers version, then re-run baseline test

## Usage (After Fix)

### Run Baseline Test
```bash
cd /home/dp/ai-workspace/HRM/policy
python3 run_baseline_test.py --num-scenarios 3  # Quick test
python3 run_baseline_test.py  # Full test (all scenarios)
```

### Test Prompts
```bash
python3 prompts.py  # Show example prompts
```

### View Test Scenarios
```bash
python3 test_suite.py  # Print all test scenarios
```

## Next Steps (For Autonomous Sessions)

See `AUTONOMOUS_SESSION_TASKS.md` for detailed task breakdown.

**Immediate**:
1. Fix transformers version
2. Run baseline test (8 scenarios)
3. Analyze results
4. Begin prompt optimization

**Success Metrics**:
- Baseline pass rate >70%
- Decision accuracy >85%
- Reasoning coverage >80%

## Architecture

See `/home/dp/ai-workspace/HRM/research/Policy_Role_Training_Plan.md` for complete training plan.

**Key principles**:
- Continuous learning (not one-shot training)
- Prompt engineering first
- Few-shot learning second
- LoRA fine-tuning only when 50+ validated corrections exist
- Safeguards from SAGE lessons (prevent collapse)

---

**Created**: 2026-02-02
**Last Updated**: 2026-02-02 00:42 UTC
