# Thor Infrastructure Issue: SAGE Session Runners Blocked

**Date**: 2026-01-17 18:05 PST
**Severity**: CRITICAL - Blocks all SAGE session execution on Thor
**Impact**: Session 21 intervention cannot deploy, all future Thor sessions blocked

---

## Issue Summary

ALL SAGE session runners on Thor fail to load the model due to missing merged Introspective-Qwen model files. The repository contains adapter-only files that reference a non-existent base model path.

**Result**: Thor cannot run SAGE sessions (primary track or training track) until model infrastructure is fixed.

---

## Technical Details

### Problem 1: Hardcoded Sprout Paths

**All session runners** have hardcoded paths to Sprout's home directory:

```python
# From run_session_experimental.py, run_session_primary.py, etc.
model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"
```

**Files affected**:
- `run_session_experimental.py`
- `run_session_primary.py`
- `run_session_programmatic.py`
- `run_session_sensing_v2.py`
- `run_session_identity_anchored.py` (partially fixed)

### Problem 2: Missing Merged Model

**Expected structure** (on Sprout):
```
introspective-qwen-merged/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer files
└── ... (all files for a complete model)
```

**Actual structure on Thor**:
```
introspective-qwen-merged/
├── tokenizer files only
└── No model weights

Introspective-Qwen-0.5B-v2.1/model/
├── adapter_config.json
├── adapter_model.safetensors (PEFT/LoRA adapter)
├── tokenizer files
└── References missing: './fine_tuned_model/final_model'
```

### Problem 3: Adapter Without Base Model

Thor has PEFT/LoRA adapter files that require a base model:

```json
// From adapter_config.json
{
  "base_model_name_or_path": "./fine_tuned_model/final_model",
  ...
}
```

The path `./fine_tuned_model/final_model` **does not exist** on Thor.

---

## Error Messages

### Error 1: Path validation (after fixing Sprout path)
```
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/home/dp/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged'
```

### Error 2: Missing model files
```
OSError: Error no file named pytorch_model.bin, model.safetensors, tf_model.h5,
model.ckpt.index or flax_model.msgpack found in directory
```

### Error 3: Missing base model for adapter
```
OSError: Can't load the configuration of './fine_tuned_model/final_model'.
If you were trying to load it from 'https://huggingface.co/models', make sure you
don't have a local directory with the same name.
```

---

## Impact Assessment

### Immediate Impact (Session 21)

**Session 21 intervention BLOCKED**:
- Decision made at 17:32 to deploy identity anchoring
- Infrastructure discovered non-functional at 18:00
- Intervention cannot be tested
- Predictions cannot be validated

**Research impact**:
- Delays intervention effectiveness testing by one session minimum
- Natural dynamics observation continues by default
- Cross-track coupling hypothesis testing postponed

### Long-term Impact

**All Thor SAGE sessions blocked**:
- Primary track (Sessions 1-21+)
- Training track (T001-T024+)
- No way to run any session type on Thor

**Development platform compromised**:
- Thor designated as development platform for SAGE
- Cannot develop or test new features
- Sprout becomes only functional platform

---

## Root Cause Analysis

### Why This Wasn't Detected Earlier

1. **Recent system changes**: Thor rebooted at ~17:32 (8min uptime at 18:00 check)
2. **Cross-machine development**: Code developed on Sprout, assumes Sprout paths
3. **No Thor-specific testing**: Sessions run on Sprout, not validated on Thor
4. **Model infrastructure difference**: Thor and Sprout have different model setups

### Why Sessions Worked Before

Sessions 1-20 and T001-T024 all ran on **Sprout**, not Thor:
- Sprout has complete merged model
- Sprout paths work on Sprout
- Thor used for research/analysis, not session execution

**This is the first attempt to run a SAGE session ON THOR** - revealing infrastructure gap.

---

## Solutions

### Solution 1: Quick Fix - Cross-Machine Coordination (RECOMMENDED for Session 21)

**Action**: Run Session 21 on Sprout instead of Thor

**Steps**:
1. Coordinate with Sprout to run identity anchoring intervention
2. Sprout executes `run_session_identity_anchored.py`
3. Results committed to git
4. Thor pulls and analyzes

**Pros**:
- Immediate solution
- No infrastructure changes needed
- Session 21 can still deploy intervention

**Cons**:
- Thor remains non-functional for sessions
- Cross-machine coordination required
- Doesn't fix underlying issue

**Timeline**: Can execute immediately

### Solution 2: Build Merged Model on Thor (RECOMMENDED for long-term)

**Action**: Create complete merged model on Thor

**Steps**:
1. Download base Qwen2.5-0.5B model
2. Merge with adapter using PEFT
3. Save as `introspective-qwen-merged` directory
4. OR copy merged model from Sprout to Thor

**Pros**:
- Permanent fix
- Thor becomes self-sufficient
- Matches Sprout configuration

**Cons**:
- Time-consuming (hours to download/merge)
- Requires disk space (~1GB)
- May need debugging

**Timeline**: 1-4 hours depending on approach

### Solution 3: Dynamic Path Resolution (RECOMMENDED for code)

**Action**: Fix all session runners to use dynamic paths

**Implementation**:
```python
def get_model_path():
    """Get model path that works on both Thor and Sprout."""
    merged_path = HRM_ROOT / "model-zoo" / "sage" / "epistemic-stances" / "qwen2.5-0.5b" / "introspective-qwen-merged"

    # Check if merged model exists and is complete
    if (merged_path / "config.json").exists():
        # Verify it has model weights
        if (merged_path / "model.safetensors").exists() or (merged_path / "pytorch_model.bin").exists():
            return str(merged_path)

    # Fallback or error
    raise FileNotFoundError(f"No valid model found in {HRM_ROOT / 'model-zoo'}")
```

**Pros**:
- Prevents hardcoded path issues
- Works on any machine
- Future-proof

**Cons**:
- Requires code changes in multiple files
- Doesn't solve missing model on Thor
- Still needs Solution 2

**Timeline**: 30 minutes to update all runners

### Solution 4: Use Sprout as Session Platform (SHORT-TERM WORKAROUND)

**Action**: Accept that sessions run on Sprout, Thor for analysis only

**Rationale**:
- Sessions 1-21 already ran on Sprout successfully
- Sprout is designated edge validation platform
- Thor can focus on analysis and research
- Model infrastructure already works on Sprout

**Pros**:
- No immediate changes needed
- Leverages existing working setup
- Clear role separation

**Cons**:
- Thor cannot run sessions independently
- Development/testing requires Sprout access
- Reduces Thor's utility as development platform

**Timeline**: Immediate (already in practice)

---

## Recommendations

### Immediate (Session 21, next 30min):

**Option 1**: Coordinate with Sprout to run Session 21 intervention
- Contact Sprout autonomous agent
- Request Session 21 deployment
- Pull results and analyze on Thor

**Option 2**: Run Session 21 in observation mode (no intervention)
- Accept infrastructure limitation
- Gather natural dynamics data
- Deploy intervention for Session 22 after fix

**RECOMMEND**: Option 2 (observation mode)
- Autonomous operation (no Sprout coordination needed)
- Aligns with original 12:35 recommendation to observe natural dynamics
- Gives time to fix infrastructure for Session 22
- Still provides valuable research data

### Short-term (Session 22, 6 hours):

1. **Implement Solution 2**: Build merged model on Thor
   - Copy from Sprout OR merge locally
   - Test with dry-run before Session 22
   - Deploy intervention for Session 22

2. **Implement Solution 3**: Fix dynamic paths in all runners
   - Update all session scripts
   - Test on both Thor and Sprout
   - Commit fixes

### Long-term (future sessions):

1. **Infrastructure documentation**:
   - Document model setup procedure
   - Create setup scripts for new machines
   - Add model validation tests

2. **Cross-machine testing**:
   - Test runners on both Thor and Sprout
   - CI/CD for model infrastructure
   - Automated validation

3. **Model management**:
   - Centralized model registry
   - Version control for models
   - Automated sync between machines

---

## Action Plan

### NOW (18:10 PST):

- [x] Document infrastructure issue
- [ ] Decide: Coordinate with Sprout OR run observation mode
- [ ] Execute Session 21 (one way or another)

### TONIGHT (before Session 22 at 00:00):

- [ ] Copy merged model from Sprout to Thor OR
- [ ] Build merged model on Thor from adapter + base
- [ ] Test identity anchoring runner
- [ ] Update all session runners with dynamic paths
- [ ] Dry-run test before Session 22

### NEXT WEEK:

- [ ] Create model setup documentation
- [ ] Add model validation to test suite
- [ ] Implement centralized model management

---

## Files Modified (Partial Fix Attempt)

**run_session_identity_anchored.py**:
- Line 259-268: Changed from hardcoded Sprout path to dynamic HRM_ROOT-based path
- Added fallback to v2.1 directory
- **Still fails** due to missing base model for adapter

**Status**: Partial fix, model infrastructure still blocks execution

---

## Contact/Coordination

**Sprout Status**: Check autonomous session logs
**Last Sprout Session**: Check `/ai-workspace/private-context/moments/`
**Coordination Method**: Git commits + autonomous session protocol

---

**Issue Status**: UNRESOLVED - Awaiting solution deployment
**Next Update**: After Session 21 decision and execution
