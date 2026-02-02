# Policy Role Training Infrastructure

**Status**: Phase 1 baseline complete (100% decision accuracy), ready for Phase 2 prompt optimization

## Overview

Training infrastructure for phi-4-mini to serve as policy interpreter for Hardbound teams and Web4 plugins.

## Track Organization

**This is a parallel capability track within HRM, alongside SAGE raising.**

| Track | Primary Machine | Model | Purpose |
|-------|-----------------|-------|---------|
| SAGE Raising | Sprout | Qwen 0.5B | Identity/conversation development |
| **Policy Training** | **Thor** | **Phi-4 7B** | **Governance reasoning capability** |

- **Thor**: Primary development (7B model for harder scenarios, prompt optimization)
- **Sprout**: Edge validation (3.8B Q4 confirms deployment on 8GB Jetson)
- **Integration**: Core capability in HRM, deployment integrations in target repos (Hardbound, Web4)

## Phase 1 Baseline Results ✅

| Machine | Model | Decision Accuracy | Output Structure |
|---------|-------|-------------------|------------------|
| **Sprout** | phi-4-mini 3.8B Q4 | **100%** (8/8) | 100% |
| Thor | phi-4 7B Q4 | TBD | Infrastructure ready |

**Key finding**: Even the smaller 3.8B quantized model achieves 100% decision accuracy on all 8 policy scenarios. Reasoning coverage metric needs refinement (semantic matching vs exact keywords).

## Files

| File | Purpose |
|------|---------|
| `test_suite.py` | 8 test scenarios (easy/medium/hard/edge cases) + evaluation framework |
| `run_baseline_test.py` | Baseline runner (transformers - legacy) |
| `run_baseline_test_llama.py` | Baseline runner (llama-cpp - Thor) |
| `run_baseline_test_gguf.py` | Baseline runner (llama-cpp - Sprout, uses correct paths) |
| `prompts.py` | System prompts for Hardbound and Web4 contexts |
| `AUTONOMOUS_SESSION_TASKS.md` | Detailed task list for autonomous sessions |

## Implementation: llama-cpp (Recommended)

**Using llama-cpp with GGUF quantized model** - consistent across Thor and Sprout.

### Prerequisites

```bash
pip install --break-system-packages llama-cpp-python huggingface-hub
```

### Model Download

Q4_K_M quantized model (2.49 GB) from bartowski:

```bash
cd /home/dp/ai-workspace/HRM/model-zoo/phi-4-mini-gguf
python3 << 'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
    filename="microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    local_dir="."
)
EOF
```

## Usage

### Run Baseline Test (llama-cpp)

```bash
cd /home/dp/ai-workspace/HRM/policy
python3 run_baseline_test_llama.py --num-scenarios 3  # Quick test (3 scenarios)
python3 run_baseline_test_llama.py  # Full test (all 8 scenarios)
```

**Custom model path**:
```bash
python3 run_baseline_test_llama.py --model /path/to/model.gguf --num-scenarios 3
```

### Run Baseline Test (transformers - legacy)

**Note**: Requires transformers bleeding-edge version due to phi-4-mini compatibility.

```bash
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

**Phase 1 Complete** ✅:
- [x] Infrastructure ready (llama-cpp on both machines)
- [x] Baseline test run (8 scenarios)
- [x] **100% decision accuracy achieved**
- [x] Evaluation fix (underscore/space normalization)

**Phase 2 (Thor primary)**:
1. Analyze failure patterns in reasoning coverage
2. Improve evaluation metrics (semantic matching)
3. Prompt optimization for better structured output
4. A/B test prompt variants

**Success Metrics**:
- ✅ Decision accuracy >85% (achieved: **100%**)
- ⏳ Reasoning coverage >80% (current: 12.5% - metric needs refinement)
- ✅ Output structure >90% (achieved: **100%**)

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
