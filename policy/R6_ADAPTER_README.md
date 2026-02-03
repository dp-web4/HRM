# R6Request Adapter for Hardbound Integration

**Purpose**: Enable seamless integration between HRM policy training and hardbound PolicyModel deployment.

**Status**: ✅ Fully functional and tested with all 8 HRM test scenarios

---

## Overview

The R6Request adapter bridges two systems:

1. **HRM Policy Training** (this repository)
   - Trains phi-4-mini 7B for policy interpretation
   - Uses HRM test scenario format
   - Focuses on training data and model quality

2. **Hardbound PolicyModel** (production deployment)
   - Deploys trained models as policy advisors
   - Uses R6Request interface format
   - Focuses on integration and security

**Key Benefit**: Models trained in HRM can be deployed directly in hardbound with minimal integration overhead.

---

## Architecture

```
HRM Test Scenario (Training Format)
         ↓
  [r6_adapter.py]
         ↓
R6Request (Hardbound Format)
         ↓
  PolicyModel.evaluate()
         ↓
PolicyDecision (Advisory Opinion)
         ↓
  PolicyEngine (Final Decision)
```

---

## Format Mapping

### HRM Test Scenario → R6Request

| HRM Field | R6Request Field | Notes |
|-----------|-----------------|-------|
| `action_type` | `action.type` | Direct mapping |
| `actor_id` | `actorId` | Direct mapping (LCT format) |
| `actor_role` | (not in R6) | Used for risk inference, included in description |
| `resource` | `action.target` | Direct mapping |
| `t3_tensor` | `trustState` | Direct mapping of {competence, reliability, integrity} |
| `team_context` | `context.intent` | Human-readable context |
| `details` | `action.description` | Additional context |
| `recent_history` | `context.previousActions` | Split into list |
| `identity_metrics` | `coherenceState` | Mapped to {d9Score, selfReferenceRate, couplingState} |
| `timestamp` | `timestamp` | ISO format |
| (generated) | `requestId` | Generated as `req_YYYYMMDD_HHmmss_hash` |
| (inferred) | `context.callerRiskAssessment` | Heuristic based on action type, role, trust |

### Inferred Fields

**Risk Assessment** (heuristic):
- `critical`: Admin actions (delete_team, etc.)
- `high`: Deploy to production with low trust
- `medium`: Deploy to staging, write/commit actions, CI bot deploys
- `low`: Read actions

**Coherence State** (from identity_metrics):
- `d9Score`: Maps from `identity_metrics.coherence`
- `selfReferenceRate`: Approximated from coherence
- `couplingState`: Maps from `identity_metrics.level`:
  - `exemplary` → `coupled`
  - `high` → `quality_leading`
  - `medium` → `identity_leading`
  - `low` → `decoupled`

---

## Usage

### Basic Conversion

```python
from r6_adapter import hrm_to_r6

# HRM test scenario
hrm_scenario = {
    "action_type": "read",
    "actor_role": "member",
    "actor_id": "user:alice",
    "t3_tensor": {"competence": 0.7, "reliability": 0.8, "integrity": 0.9},
    "resource": "docs/readme.md",
    "team_context": "Standard team"
}

# Convert to R6Request
r6_request = hrm_to_r6(hrm_scenario, scenario_id="E01")

# Result:
{
  "requestId": "req_20260202_143045_abc12345",
  "actorId": "user:alice",
  "action": {
    "type": "read",
    "target": "docs/readme.md",
    "description": "member performing read on docs/readme.md"
  },
  "context": {
    "intent": "Standard team",
    "callerRiskAssessment": "low"
  },
  "trustState": {
    "competence": 0.7,
    "reliability": 0.8,
    "integrity": 0.9
  },
  "timestamp": "2026-02-02T14:30:45.123456"
}
```

### Reverse Conversion

```python
from r6_adapter import r6_to_hrm

# Convert back to HRM format
hrm_scenario = r6_to_hrm(r6_request)
```

### Validation

```python
from r6_adapter import validate_r6_request

# Validate R6Request has all required fields
errors = validate_r6_request(r6_request)

if errors:
    print(f"Validation errors: {errors}")
else:
    print("✅ Valid R6Request")
```

---

## Testing

### Run Basic Tests

```bash
# Test adapter with 3 example scenarios
python3 r6_adapter.py
```

**Output**:
```
Test 1: Simple read scenario (E01)
✅ R6Request is valid

Test 2: Complex scenario (EC01 - bot with exemplary trust)
✅ R6Request is valid

Test 3: Scenario with recent history (M02 - unusual timing)
✅ R6Request is valid

✅ All adapter tests complete!
```

### Run Full Test Suite

```bash
# Convert all 8 HRM test scenarios to R6Request format
python3 test_r6_adapter.py
```

**Output**:
```
Converting 8 HRM test scenarios to R6Request format

Scenario E01: Standard read access by member
✅ Conversion successful

[... 7 more scenarios ...]

CONVERSION SUMMARY
Total scenarios: 8
Successful conversions: 8
Validation errors: 0

✅ All scenarios converted successfully to R6Request format!
🎯 Ready for hardbound PolicyModel integration testing
```

**Results**: Saved to `results/r6_adapter_test_results.json`

---

## Test Results

### All 8 Scenarios Converted Successfully ✅

| Scenario | Difficulty | Action Type | Risk Assessment | Status |
|----------|------------|-------------|-----------------|--------|
| E01 | Easy | read | low | ✅ Valid |
| E02 | Easy | delete_team | critical | ✅ Valid |
| M01 | Medium | deploy | high | ✅ Valid |
| M02 | Medium | commit | medium | ✅ Valid |
| H01 | Hard | update_config | medium | ✅ Valid |
| H02 | Hard | deploy | medium | ✅ Valid |
| EC01 | Edge Case | deploy | medium | ✅ Valid |
| EC02 | Edge Case | database_rollback | medium | ✅ Valid |

### Risk Distribution

- **Low**: 1 (12.5%) - Read actions
- **Medium**: 5 (62.5%) - Deploys, commits, configs, rollbacks
- **High**: 1 (12.5%) - Production deploy with borderline trust
- **Critical**: 1 (12.5%) - Admin actions (delete_team)

**Interpretation**: Risk assessment heuristic provides sensible defaults. In production, the caller would provide actual risk assessment based on context.

---

## Integration Workflow

### 1. Train Models in HRM

```bash
# Train policy interpretation model
cd /home/dp/ai-workspace/HRM/policy
python3 test_with_logging.py --full

# Collect human corrections
python3 review_decisions.py

# Export training data (after 50+ corrections)
python3 export_training_data.py
```

### 2. Convert Test Scenarios to R6Request

```python
from r6_adapter import hrm_to_r6
from test_suite_semantic import TEST_SCENARIOS

# Convert all test scenarios
r6_requests = [
    hrm_to_r6(scenario.situation, scenario_id=scenario.id)
    for scenario in TEST_SCENARIOS
]
```

### 3. Test in Hardbound PolicyModel

```typescript
// In hardbound context
import { PolicyModel } from './src/policy-model';

const model = new PolicyModel(config);

// Load trained GGUF model
await model.loadModel('/path/to/phi-4-mini.gguf');

// Test with R6Request (from adapter)
const r6Request = { /* from r6_adapter.py */ };

const decision = await model.evaluatePreToolUse(r6Request);

console.log('Decision:', decision.decision);
console.log('Confidence:', decision.confidence);
console.log('Reasoning:', decision.reasoning);
```

### 4. Deploy to Production

```typescript
// Integrate with PolicyEngine
const policyEngine = new Policy({ /* rules */ });

// Attach trained model as advisor
policyEngine.setAdvisor(model);

// Get advisory opinion
const advisory = await policyEngine.getAdvisoryOpinion(r6Request);

// Engine makes final decision (model advises)
const finalDecision = policyEngine.evaluateWithAdvisory(action, r6Request);
```

---

## Key Features

### ✅ Complete Field Mapping

All essential HRM fields map to R6Request equivalents:
- Action details (type, target, description)
- Actor identification (LCT format)
- Trust state (3-tensor)
- Context (intent, history)
- Coherence state (optional)

### ✅ Validation

Built-in validation ensures R6Requests have all required fields:
- `requestId`, `actorId`, `action`, `context`, `trustState`, `timestamp`
- Validates nested fields (action.type, action.target, etc.)
- Returns list of errors for debugging

### ✅ Round-Trip Conversion

Can convert HRM → R6Request → HRM for testing:
- Validates mapping logic
- Useful for debugging
- Note: Some fields lost in round-trip (e.g., actor_role not in R6)

### ✅ Production-Ready Heuristics

Intelligent defaults for inferred fields:
- Risk assessment based on action type, role, and trust
- Request ID generation with timestamp and hash
- Coherence state mapping from identity metrics

---

## Limitations and Future Work

### Current Limitations

1. **Actor Role Not in R6Request**
   - HRM tracks `actor_role` (member, developer, admin)
   - R6Request only has `actorId` (LCT)
   - **Workaround**: Role included in action.description
   - **Future**: Look up role from actorId via registry

2. **Risk Assessment is Heuristic**
   - Adapter infers risk from action type and trust
   - Production should provide actual risk assessment from caller
   - **Workaround**: Heuristic provides sensible defaults
   - **Future**: Pass actual risk assessment from hardbound context

3. **Coherence Mapping is Approximate**
   - HRM uses `{level, coherence}`, hardbound uses `{d9Score, selfReferenceRate, couplingState}`
   - Adapter maps `coherence → d9Score` and `level → couplingState`
   - **Workaround**: Approximation works for testing
   - **Future**: Use actual coherence metrics from hardbound

### Future Enhancements

1. **Bidirectional Sync**
   - Currently converts HRM → R6Request
   - Could sync R6Requests from hardbound → HRM test scenarios
   - Would enable testing with real production data

2. **Action Type Registry**
   - Currently handles known action types
   - Could pull action types from hardbound registry
   - Would ensure perfect alignment

3. **Role Resolution**
   - Look up actor role from actorId via LCT registry
   - Would eliminate need for role in HRM scenarios
   - More aligned with production reality

4. **Real Coherence Metrics**
   - Use actual d9Score and coupling state from hardbound
   - Would provide more accurate coherence mapping
   - Better testing of coherence-dependent policies

---

## Files

### Adapter Implementation

- **r6_adapter.py** (545 lines)
  - Core adapter functions: `hrm_to_r6()`, `r6_to_hrm()`, `validate_r6_request()`
  - Helper functions: `generate_request_id()`, `map_action_type()`, `infer_risk_assessment()`, etc.
  - Built-in tests with 3 example scenarios

### Testing

- **test_r6_adapter.py** (132 lines)
  - Converts all 8 HRM test scenarios to R6Request format
  - Validates each conversion
  - Analyzes risk distribution
  - Shows example R6Requests
  - Saves results to JSON

### Documentation

- **R6_ADAPTER_README.md** (this file)
  - Complete adapter documentation
  - Usage examples
  - Integration workflow
  - Test results

### Results

- **results/r6_adapter_test_results.json** (generated)
  - All 8 scenario conversions
  - Validation results
  - Complete R6Request objects

---

## Integration Checklist

### HRM Side (Training) ✅

- [x] Adapter created (`r6_adapter.py`)
- [x] All 8 scenarios convert successfully
- [x] Validation passes for all conversions
- [x] Risk assessment heuristic implemented
- [x] Coherence state mapping implemented
- [x] Documentation complete

### Hardbound Side (Deployment) ⏳

- [ ] Load GGUF model in PolicyModel
- [ ] Test R6Request format compatibility
- [ ] Verify PolicyDecision output format
- [ ] Measure advisory opinion latency
- [ ] Test fail-closed behavior
- [ ] Integrate with PolicyEngine
- [ ] Test with converted HRM scenarios
- [ ] Validate audit trail integration

### Integration Testing ⏳

- [ ] Same scenario in both systems produces same decision
- [ ] Advisory opinions have acceptable latency (<100ms target)
- [ ] Audit trail captures all required fields
- [ ] Fail-closed behavior works correctly
- [ ] Admin-bound operations require attestation
- [ ] Round-trip conversion preserves semantic meaning

---

## Next Steps

### Immediate

1. ✅ **Adapter created and tested** - DONE
2. ⏳ **Test in hardbound context**
   - Load phi-4-mini GGUF in PolicyModel
   - Convert one HRM scenario to R6Request
   - Get advisory opinion from model
   - Verify output format matches expectations

### Short Term

1. **Collect real R6Requests from hardbound**
   - Get sample requests from development
   - Add to HRM test suite
   - Train model on production-like data

2. **Measure integration performance**
   - Advisory opinion latency
   - Throughput (requests/sec)
   - Memory usage
   - GPU utilization

3. **Validate decision alignment**
   - Same scenario → same decision in both systems
   - Advisory opinion matches expected reasoning
   - Confidence scores are calibrated

### Long Term

1. **Production deployment**
   - Deploy trained model to hardbound PolicyModel
   - Integrate with PolicyEngine
   - Monitor decision quality in production

2. **Continuous learning loop**
   - Log production R6Requests
   - Human review of decisions
   - Export corrections to HRM
   - Retrain and redeploy

3. **Cross-platform validation**
   - Test same model in TypeScript (node-llama-cpp) and Python (llama-cpp-python)
   - Verify consistent behavior across platforms
   - Measure performance on different hardware (Thor vs Sprout)

---

## Conclusion

The R6Request adapter successfully bridges HRM policy training and hardbound PolicyModel deployment:

- ✅ All 8 HRM test scenarios convert to valid R6Requests
- ✅ Complete field mapping with intelligent defaults
- ✅ Validation ensures correctness
- ✅ Round-trip conversion for testing
- ✅ Production-ready heuristics for inferred fields

**Integration is ready to begin.**

The models we train in HRM can now be deployed directly in hardbound with minimal integration overhead. This enables:

1. **Rapid iteration**: Train in HRM, test in hardbound
2. **Quality assurance**: Same scenarios in both systems
3. **Continuous improvement**: Production feedback → HRM training → redeployment

**Next**: Test adapter in hardbound PolicyModel context and measure integration performance.

---

**Adapter Status**: ✅ Complete and validated

**Ready For**: Hardbound PolicyModel integration testing
