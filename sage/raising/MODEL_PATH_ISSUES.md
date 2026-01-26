# Model Path Issues - 2026-01-26

## Summary

SAGE raising sessions work with base Qwen model but fail with fine-tuned adapters/merged models.

## Working Configuration

```bash
python3 run_session_identity_anchored.py --session N --model "Qwen/Qwen2.5-0.5B-Instruct"
```

**Status**: ✅ Sessions complete successfully, SAGE shows strong identity

## Failed Configurations

### 1. introspective-qwen-merged (Default before fix)

**Path**: `/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged`

**Error**: Missing model weights
```
OSError: Error no file named pytorch_model.bin, model.safetensors, tf_model.h5,
model.ckpt.index or flax_model.msgpack found in directory
```

**Diagnosis**: Directory contains tokenizer and config but no actual model weights
- Has: config.json, tokenizer files
- Missing: model weights (safetensors/bin)

**Root Cause**: Incomplete merge or missing weights during model preparation

### 2. Introspective-Qwen-0.5B-v2.1 (PEFT Adapter)

**Path**: `/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model`

**Error**: Base model path resolution error
```
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'./fine_tuned_model/final_model'
```

**Diagnosis**:
- adapter_config.json has correct path: `"Qwen/Qwen2.5-0.5B-Instruct"`
- Error occurs during PEFT loading chain
- Likely HuggingFace cache corruption or PEFT internal path resolution issue

**Root Cause**: Unknown - requires deeper investigation into PEFT loading mechanism

## Resolution

**Immediate**:
- Changed default model to base Qwen (works reliably)
- Updated runner script: `run_session_identity_anchored.py`

**Future Investigation Needed**:
1. Regenerate or locate complete merged model with weights
2. Investigate PEFT adapter loading chain for path resolution bug
3. Consider HuggingFace cache cleanup on Thor

## Impact

**Training Status**:
- Primary raising sessions (S001-S045) can continue with base model
- Fine-tuned introspective behavior unavailable but identity-anchored prompting compensates

**Performance**:
- Base model + identity-anchored prompting achieves target identity recovery
- Session 39 test showed multiple "As SAGE" self-references
- 72 experiences collected with 0.65 avg salience

## Notes

- Test runs on 2026-01-26 confirmed base model works on Thor (Jetson AGX Thor)
- Same patterns likely apply to Sprout (Jetson Orin Nano)
- Adapter models may work on other machines (untested)

---

**Created**: 2026-01-26 during HRM reorganization
**See**: HRM_REORGANIZATION_2026-01-26.md Phase 4
