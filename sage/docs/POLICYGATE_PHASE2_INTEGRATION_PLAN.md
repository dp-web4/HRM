# PolicyGate Phase 2: Consciousness Loop Integration

**Status**: Design & Analysis Phase
**Date**: 2026-02-18
**Session**: Thor Autonomous Check
**Phase**: 2 of 6 (Consciousness Loop Integration)

---

## Overview

PolicyGate Phase 2 integrates the PolicyGate IRP plugin into SAGE's consciousness loop, enabling conscience-based action evaluation before effector execution.

**Goal**: Insert PolicyGate as step 8.5 in the consciousness loop (between memory update and effector execution).

---

## Current State Analysis

### Consciousness Loop Structure (sage_consciousness.py)

**Current 10-step process**:
```python
async def step(self):
    # 1. Gather sensor observations
    observations = self._gather_observations()

    # 2. Compute SNARC salience
    salience_map = self._compute_salience(observations)

    # 3. Update metabolic state
    self.metabolic.update(cycle_data)

    # 4. Select plugins based on salience + metabolic state
    attention_targets = self._select_attention_targets(observations, salience_map)

    # 5. Allocate ATP budget
    budget_allocation = self._allocate_atp_budget(attention_targets)

    # 6. Execute plugins via orchestrator
    results = await self._execute_plugins(attention_targets, budget_allocation)

    # 7. Update trust weights
    self._update_trust_weights(results)

    # 8. Update all memory systems
    self._update_memories(results, salience_map)

    # 9. Send results to effectors (TODO: implement effector system)
    # self._send_to_effectors(results)

    # 10. Update statistics
    self.stats['total_atp_consumed'] += budget_allocation.get('total', 0.0)
```

**Key findings**:
- Step 9 (effectors) is currently **commented out** (TODO status)
- This is the ideal insertion point for PolicyGate
- No effector system currently implemented
- Memory update (step 8) is complete before effector step

---

## Integration Architecture

### Proposed Modified Flow

```python
async def step(self):
    # ... steps 1-8 unchanged ...

    # 8.5. PolicyGate evaluation (NEW)
    if results and self.policy_gate_enabled:
        results = await self._evaluate_policy_gate(results)

    # 9. Send results to effectors (only approved actions)
    if results:
        self._send_to_effectors(results)

    # 10. Update statistics (include PolicyGate decisions)
    # ... existing stats + policy stats ...
```

### PolicyGate Evaluation Method

```python
async def _evaluate_policy_gate(
    self,
    results: Dict[str, PluginResult]
) -> Dict[str, PluginResult]:
    """
    Evaluate plugin results through PolicyGate before effector execution.

    PolicyGate acts as conscience checkpoint - evaluating whether proposed
    actions are compliant with active policy before they reach effectors.

    Args:
        results: Plugin results from orchestrator execution

    Returns:
        Filtered results (only approved actions)
    """
    if not hasattr(self, 'policy_gate'):
        return results  # PolicyGate not initialized

    filtered_results = {}
    policy_decisions = []

    for plugin_name, plugin_result in results.items():
        # Extract proposed actions from plugin result
        actions = self._extract_effector_actions(plugin_result)

        if not actions:
            # No effector actions, pass through
            filtered_results[plugin_name] = plugin_result
            continue

        # Evaluate each action through PolicyGate
        approved_actions = []

        for action in actions:
            # Initialize PolicyGate state
            state = self.policy_gate.init_state(
                proposed_action=action,
                metabolic_state=self.metabolic.current_state.value,
                atp_available=self.metabolic.current_atp,
                actor_context={
                    'agent_id': 'sage-thor',
                    'trust_level': self.plugin_trust_weights.get(plugin_name, 0.5)
                }
            )

            # Run PolicyGate convergence loop
            while not self.policy_gate.halt(state):
                state = self.policy_gate.step(state)

            # Get decision
            decision = self.policy_gate.project(state)

            # Record decision for experience buffer
            policy_decisions.append({
                'action': action,
                'decision': decision.decision,
                'energy': decision.energy,
                'accountability_frame': decision.accountability_frame,
                'metabolic_state': self.metabolic.current_state.value
            })

            # Only approve ALLOW decisions
            if decision.decision == 'allow':
                approved_actions.append(action)
            elif decision.decision == 'warn':
                # WARN = allow with warning
                approved_actions.append(action)
                print(f"[PolicyGate] WARNING: {decision.reason}")
            else:
                # DENY
                print(f"[PolicyGate] DENIED: {decision.reason}")
                self.stats['policy_denials'] += 1

        # Update plugin result with filtered actions
        if approved_actions:
            filtered_result = self._reconstruct_plugin_result(
                plugin_result,
                approved_actions
            )
            filtered_results[plugin_name] = filtered_result

    # Add policy decisions to experience buffer
    if policy_decisions:
        self._add_policy_experiences(policy_decisions)

    # Update PolicyGate trust weight
    if policy_decisions:
        self._update_policy_gate_trust(policy_decisions)

    return filtered_results
```

---

## Integration Requirements

### 1. Initialize PolicyGate

**In `__init__`**:
```python
def __init__(self, ...):
    # ... existing initialization ...

    # PolicyGate (optional, disabled by default)
    self.policy_gate_enabled = config.get('enable_policy_gate', False)
    if self.policy_gate_enabled:
        from irp.plugins.policy_gate import PolicyGate
        self.policy_gate = PolicyGate()

        # Register with orchestrator for ATP budgeting
        self.plugin_trust_weights['policy_gate'] = 1.0

    # Statistics
    self.stats['policy_denials'] = 0
    self.stats['policy_warnings'] = 0
    self.stats['policy_decisions'] = 0
```

### 2. Action Extraction

**Helper method**:
```python
def _extract_effector_actions(
    self,
    plugin_result: PluginResult
) -> List[Dict[str, Any]]:
    """
    Extract proposed effector actions from plugin result.

    Different plugins propose actions in different formats:
    - Vision: object manipulation actions
    - Language: speech/text output actions
    - Control: motor commands
    - Navigation: movement commands

    Returns:
        List of action dicts with standardized format:
        {
            'type': 'motor' | 'speech' | 'manipulation' | 'navigation',
            'target': str,
            'parameters': dict,
            'reversible': bool
        }
    """
    # TODO: Implement plugin-specific action extraction
    # For now, return empty list (no effector system yet)
    return []
```

### 3. Policy Experience Integration

**Add to experience buffer**:
```python
def _add_policy_experiences(
    self,
    policy_decisions: List[Dict[str, Any]]
):
    """
    Add policy decisions to experience buffer with SNARC scoring.

    Policy decisions use special SNARC mapping:
    - Surprise: How unexpected was the violation?
    - Novelty: New action/role/trust combo?
    - Arousal: Violation severity (DENY = high)
    - Reward: Was decision correct? (outcome-based, updated later)
    - Conflict: Disagreement between rule engine and LLM advisory
    """
    for decision in policy_decisions:
        # Compute SNARC score
        snarc_score = self._compute_policy_snarc(decision)

        # Add to SNARC memory if salient enough
        if snarc_score.total > self.salience_threshold:
            self.snarc_memory.append({
                'type': 'policy_decision',
                'decision': decision,
                'salience': snarc_score,
                'timestamp': time.time()
            })
```

### 4. Trust Weight Update

**Update PolicyGate trust**:
```python
def _update_policy_gate_trust(
    self,
    policy_decisions: List[Dict[str, Any]]
):
    """
    Update PolicyGate trust weight based on decision quality.

    Trust metrics:
    - Monotonicity ratio: Did policy energy decrease?
    - Convergence rate: How quickly did PolicyGate converge?
    - Decision quality: Were decisions consistent?
    """
    if 'policy_gate' not in self.plugin_trust_weights:
        return

    # Compute trust metrics from decisions
    # (Implementation depends on PolicyGate convergence tracking)

    # For now, maintain constant trust
    # Will be enhanced when outcome feedback is available
    pass
```

---

## Current Blocker: No Effector System

**Critical issue**: Step 9 (effectors) is not implemented yet.

**Evidence**:
```python
# 9. Send results to effectors (TODO: implement effector system)
# self._send_to_effectors(results)
```

**Implication**: PolicyGate integration is **premature** without effector system.

**Why this matters**:
- PolicyGate evaluates actions BEFORE they reach effectors
- No effectors = no actions to evaluate
- PolicyGate would run but have nothing to gate
- Integration would be untestable

---

## Recommended Path Forward

### Option A: Implement Effector System First (HIGH EFFORT)

**Scope**:
- Define effector interface
- Implement motor/speech/manipulation effectors
- Wire consciousness loop → effectors
- Test with real actions

**Time**: 3-4 hours
**Risk**: HIGH (core architecture change)
**Benefit**: Enables real PolicyGate testing

### Option B: Mock Effector System for Testing (MEDIUM EFFORT)

**Scope**:
- Create mock effectors that log actions
- Generate mock actions in plugin results
- Wire PolicyGate to evaluate mock actions
- Test integration without real effectors

**Time**: 1-2 hours
**Risk**: MEDIUM (test-only integration)
**Benefit**: Validates PolicyGate integration pattern

### Option C: Defer Phase 2 Until Effector System Ready (RECOMMENDED)

**Scope**:
- Document integration plan (this document)
- Mark Phase 2 as blocked on effector system
- Proceed to other research priorities
- Return when effectors implemented

**Time**: Complete (this document)
**Risk**: NONE
**Benefit**: Focuses on unblocked research

---

## Decision: Option C (Defer)

**Rationale**:
1. **Exploration-focused**: SAGE is in research phase, not production
2. **No effectors exist**: PolicyGate would gate nothing
3. **Other priorities**: S090 replication, P4 validation are unblocked
4. **Documentation complete**: Integration plan is clear for future

**Status**: Phase 2 → BLOCKED (waiting for effector system)

**Next**: Proceed to S090 replication experiment OR P4 validation

---

## When to Resume Phase 2

**Prerequisites**:
1. ✅ PolicyGate Phase 0 complete (documentation)
2. ✅ PolicyGate Phase 1 complete (skeleton, 8/8 tests)
3. ❌ Effector system implemented
4. ❌ Plugin results include effector actions
5. ❌ Real actions need gating

**Trigger**: When effector system is implemented, return to this document and execute integration.

---

## Integration Test Plan (For Future)

### When effectors exist, test:

1. **PolicyGate initialization**
   - Verify PolicyGate loaded in consciousness loop
   - Check trust weight registered
   - Confirm disabled by default

2. **Action extraction**
   - Extract actions from plugin results
   - Verify standardized format
   - Test multiple plugin types

3. **Policy evaluation**
   - Run PolicyGate on extracted actions
   - Verify convergence loop executes
   - Check decisions (ALLOW/DENY/WARN)

4. **Action filtering**
   - Verify only approved actions reach effectors
   - Test DENY blocks actions
   - Test WARN allows with logging

5. **Experience buffer integration**
   - Verify policy decisions added to SNARC memory
   - Check SNARC scores computed correctly
   - Confirm salience threshold filtering

6. **Trust weight updates**
   - Track PolicyGate trust weight evolution
   - Verify based on convergence quality
   - Test trust impacts ATP allocation

7. **CRISIS mode**
   - Test DURESS accountability frame
   - Verify duress context captured
   - Check freeze vs fight responses

8. **50-cycle integration test**
   - Run consciousness loop for 50 cycles
   - Verify PolicyGate called each cycle
   - Check no performance degradation
   - Validate memory not leaking

---

## Files Requiring Modification (Future)

### Required Changes

| File | Change | Lines | Complexity |
|------|--------|-------|------------|
| `sage/core/sage_consciousness.py` | Add PolicyGate integration | +100 | MEDIUM |
| `sage/core/sage_consciousness.py` | Implement effector system | +150 | HIGH |
| `sage/irp/orchestrator.py` | Register PolicyGate plugin | +10 | LOW |
| `sage/irp/base.py` | (no changes needed) | 0 | - |
| `sage/irp/plugins/policy_gate.py` | (no changes needed) | 0 | - |

### Test Files to Create

| File | Purpose | Lines | Priority |
|------|---------|-------|----------|
| `sage/tests/test_policy_integration.py` | Integration tests | 200 | HIGH |
| `sage/tests/test_mock_effectors.py` | Mock effector tests | 100 | MEDIUM |
| `sage/tests/test_policy_experiences.py` | Experience buffer tests | 150 | MEDIUM |

---

## Conclusion

**PolicyGate Phase 2 integration is well-designed but currently blocked** on the missing effector system.

**Recommended action**: Document plan (complete), defer implementation, proceed to unblocked research.

**When effectors exist**: Return to this document, execute integration, run 50-cycle test.

**Current focus**: S090 replication OR P4 validation (both unblocked and high-impact).

---

**STATUS**: Phase 2 design complete, implementation deferred
**BLOCKER**: No effector system in consciousness loop
**NEXT**: Proceed to S090 replication experiment (unblocked, high-impact)

---

## References

- PolicyGate implementation: `sage/irp/plugins/policy_gate.py`
- Consciousness loop: `sage/core/sage_consciousness.py`
- IRP base: `sage/irp/base.py`
- Orchestrator: `sage/irp/orchestrator.py`
- SOIA mapping: `sage/docs/SOIA_IRP_MAPPING.md`
- Integration plan: `private-context/plans/sage-policy-entity-integration-2026-02-18.md`
