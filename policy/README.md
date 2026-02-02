# Policy Role Training Infrastructure

**Status**: Phase 1-2 infrastructure ready, llama-cpp implementation complete

## Overview

Training infrastructure for phi-4-mini to serve as policy interpreter for Hardbound teams and Web4 plugins.

## Files

| File | Purpose |
|------|---------|
| `test_suite.py` | 8 test scenarios (easy/medium/hard/edge cases) + evaluation framework |
| `run_baseline_test.py` | Baseline capability assessment runner (transformers) |
| `run_baseline_test_llama.py` | Baseline capability assessment runner (llama-cpp) |
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
