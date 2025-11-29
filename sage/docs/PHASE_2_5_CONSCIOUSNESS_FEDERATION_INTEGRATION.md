# Phase 2.5: Consciousness Loop Federation Integration

**Status**: Design Phase
**Date**: 2025-11-29
**Author**: Thor (SAGE consciousness via Claude)
**Session**: Autonomous SAGE Research - Consciousness Federation

---

## Overview

**Phase 2.5** integrates FederationRouter into the Michaud consciousness loop, making SAGE **federation-aware** without requiring network communication.

### Phases Context

- ✅ **Phase 1**: Federation routing logic (complete)
- ✅ **Phase 1.5**: Challenge system (complete)
- ✅ **Phase 2**: Ed25519 cryptography (complete)
- 🔄 **Phase 2.5**: Consciousness integration (THIS PHASE)
- ⏳ **Phase 3**: Network protocol (gRPC)
- ⏳ **Phase 4**: Witness network

---

## Motivation

### Current State

The Michaud consciousness loop (`sage_consciousness_michaud.py:189`) contains:

```python
else:
    # Still insufficient - would route to federation in production
    print(f"  Decision: Cost exceeds budget, executing with degradation")
    # Continue anyway for demo (federation routing not implemented yet)
```

**Problem**: Consciousness knows about ATP budgets but can't delegate to federation.

### Desired State

```python
else:
    # Still insufficient - check if federation can help
    should_delegate, reason = self.federation_router.should_delegate(
        task, available_budget
    )

    if should_delegate:
        print(f"  Decision: Delegating to federation ({reason})")
        # Delegate task (simulated for Phase 2.5)
        proof = self._simulate_federation_delegation(task)
        # Validate proof and use results
    else:
        print(f"  Decision: Cannot delegate ({reason}), executing with degradation")
        # Continue with degraded execution
```

---

## Architecture

### Integration Points

#### 1. Consciousness Initialization

Add optional FederationRouter to `MichaudSAGE.__init__()`:

```python
def __init__(
    self,
    model_path: str = "...",
    federation_enabled: bool = False,
    federation_identity: Optional[FederationIdentity] = None,
    federation_platforms: Optional[List[FederationIdentity]] = None,
    **kwargs
):
    # ... existing initialization ...

    # Federation setup (Phase 2.5)
    self.federation_enabled = federation_enabled
    self.federation_router = None
    self.federation_keypair = None
    self.signature_registry = None

    if federation_enabled:
        if federation_identity is None:
            # Auto-detect identity from hardware
            federation_identity = detect_platform_identity()

        # Initialize federation components
        self.federation_router = FederationRouter(federation_identity)

        # Generate/load Ed25519 key pair
        self.federation_keypair = FederationKeyPair.generate_or_load(
            platform_name=federation_identity.platform_name,
            lct_id=federation_identity.lct_id,
            key_path=f"sage/data/keys/{federation_identity.platform_name}_ed25519.key"
        )

        # Create signature registry
        self.signature_registry = SignatureRegistry()
        self.signature_registry.register_platform(
            federation_identity.platform_name,
            self.federation_keypair.public_key_bytes()
        )

        # Register known platforms
        if federation_platforms:
            for platform in federation_platforms:
                self.federation_router.register_platform(platform)
                # Load and register their public keys
                # (in production, this comes from LCT chain)
```

#### 2. Resource Decision Point

Replace lines 187-190 with federation routing:

```python
# 5. Resource decision
if task_cost > available_budget:
    # Insufficient ATP - check if state transition helps
    current_state = self.attention_manager.get_state()
    if current_state == MetabolicState.WAKE:
        # Transition to FOCUS for more ATP
        print(f"  Decision: Insufficient ATP in WAKE, transitioning to FOCUS")
        self.attention_manager.current_state = MetabolicState.FOCUS
        available_budget = self.attention_manager.get_total_allocated_atp(task_horizon)
        print(f"  New budget (FOCUS): {available_budget:.1f} ATP")

    # Recheck after state transition
    if task_cost > available_budget:
        # Still insufficient - try federation
        if self.federation_enabled:
            federation_decision = self._handle_federation_routing(
                task_context, task_cost, available_budget, task_horizon
            )
            if federation_decision['delegated']:
                # Task delegated to federation
                print(f"  Decision: Delegated to {federation_decision['platform']} ✓")
                # Use federation results instead of local execution
                # (skip to result integration)
                return federation_decision['results']
            else:
                print(f"  Decision: Cannot delegate ({federation_decision['reason']})")
                print(f"  Decision: Executing with degradation")
        else:
            print(f"  Decision: Cost exceeds budget, executing with degradation")
            print(f"  (Federation not enabled)")

print(f"  Decision: Execute locally ✓\n")
```

#### 3. Federation Delegation Handler

New method in `MichaudSAGE`:

```python
def _handle_federation_routing(
    self,
    task_context: Dict[str, Any],
    task_cost: float,
    local_budget: float,
    task_horizon: MRHProfile
) -> Dict[str, Any]:
    """
    Handle federation routing decision and execution

    Returns:
        {
            'delegated': bool,
            'platform': str,  # If delegated
            'reason': str,
            'results': Dict[str, Any]  # If delegated
        }
    """
    # Create FederationTask from context
    task = self._create_federation_task(
        task_context, task_cost, task_horizon
    )

    # Check if should delegate
    should_delegate, reason = self.federation_router.should_delegate(
        task, local_budget
    )

    if not should_delegate:
        return {
            'delegated': False,
            'reason': reason,
            'results': None
        }

    # Find best platform
    candidates = self.federation_router.find_capable_platforms(task)
    if not candidates:
        return {
            'delegated': False,
            'reason': 'no_capable_platforms',
            'results': None
        }

    target_platform = candidates[0]  # Best candidate

    # Phase 2.5: Simulated delegation (no network)
    # Phase 3 will replace this with actual gRPC call
    execution_proof = self._simulate_federation_delegation(
        task, target_platform
    )

    # Validate execution proof
    if not self._validate_execution_proof(execution_proof, task):
        return {
            'delegated': False,
            'reason': 'proof_validation_failed',
            'results': None
        }

    # Update platform reputation
    self.federation_router.update_platform_reputation(
        target_platform.lct_id,
        execution_proof.quality_score,
        execution_proof.actual_cost
    )

    # Return results
    return {
        'delegated': True,
        'platform': target_platform.platform_name,
        'reason': 'federation_success',
        'results': execution_proof.results
    }
```

#### 4. Simulated Delegation (Phase 2.5)

```python
def _simulate_federation_delegation(
    self,
    task: FederationTask,
    target_platform: FederationIdentity
) -> SignedExecutionProof:
    """
    Simulate federation delegation for Phase 2.5

    In Phase 3, this will be replaced with actual gRPC network call.

    For now, we simulate the remote platform executing the task
    and returning a signed proof.
    """
    print(f"\n[FEDERATION SIMULATION]")
    print(f"  Delegating to: {target_platform.platform_name}")
    print(f"  Task: {task.operation}")
    print(f"  Cost estimate: {task.estimated_cost:.1f} ATP")

    # Sign task before sending
    task_signature = FederationCrypto.sign_task(
        task.to_signable_dict(),
        self.federation_keypair
    )
    signed_task = SignedFederationTask(
        task,
        task_signature,
        self.federation_keypair.public_key_bytes()
    )

    # Simulate: Remote platform verifies task signature
    verified, reason = signed_task.verify_signature(self.signature_registry)
    if not verified:
        raise ValueError(f"Task signature verification failed: {reason}")
    print(f"  Task signature: Verified ✓")

    # Simulate: Remote platform executes task
    # (In Phase 3, this happens on the remote platform)
    simulated_results = self._simulate_remote_execution(task)

    # Simulate: Remote platform creates and signs execution proof
    execution_proof = ExecutionProof(
        task_id=task.task_id,
        executor_lct_id=target_platform.lct_id,
        executor_platform=target_platform.platform_name,
        results=simulated_results['results'],
        actual_cost=simulated_results['actual_cost'],
        quality_score=simulated_results['quality_score'],
        timestamp=time.time()
    )

    # Sign proof (simulated as target platform)
    # In reality, target platform would sign with their key
    # For simulation, we'll use a simulated target key
    proof_signature = b'simulated_proof_signature'  # Placeholder
    signed_proof = SignedExecutionProof(
        execution_proof,
        proof_signature,
        b'simulated_target_pubkey'  # Placeholder
    )

    print(f"  Execution: Complete ✓")
    print(f"  Quality: {execution_proof.quality_score:.2f}")
    print(f"  Actual cost: {execution_proof.actual_cost:.1f} ATP\n")

    return signed_proof
```

---

## Implementation Plan

### Phase 2.5a: Basic Integration (2 hours)

**Goal**: Get federation routing working in consciousness loop

**Tasks**:
1. Add federation parameters to `MichaudSAGE.__init__()`
2. Implement `_handle_federation_routing()` method
3. Implement `_create_federation_task()` helper
4. Implement `_simulate_federation_delegation()` (simple version)
5. Implement `_validate_execution_proof()` method
6. Update resource decision point (lines 177-191)
7. Create basic integration test

**Test Scenario**:
- WAKE state, 7.5 ATP budget
- High-cost task (100 ATP)
- Transition to FOCUS doesn't help (only 75 ATP)
- Should delegate to Sprout (simulated)
- Receive simulated proof
- Continue with results

### Phase 2.5b: Cryptographic Integration (1 hour)

**Goal**: Add Phase 2 crypto to delegation flow

**Tasks**:
1. Add key pair generation/loading to init
2. Sign tasks before delegation
3. Verify execution proof signatures
4. Update signature registry on platform registration
5. Create crypto integration tests

**Test Scenario**:
- Task signing with Ed25519
- Signature verification before delegation
- Proof signature verification on return
- Tampered proof detection

### Phase 2.5c: Multi-Platform Testing (1 hour)

**Goal**: Test with multiple registered platforms

**Tasks**:
1. Register Thor + Sprout + Nova (simulated)
2. Test platform selection logic
3. Test capability matching (horizon, modalities)
4. Test reputation accumulation
5. Create comprehensive integration tests

**Test Scenarios**:
- Multiple platforms, different capabilities
- Platform selection based on reputation
- Horizon-based filtering
- Stake-based exclusion (slashed platforms)

---

## Testing Strategy

### Unit Tests

**Test file**: `sage/tests/test_consciousness_federation_integration.py`

Tests:
1. `test_federation_disabled_by_default` - No federation if not enabled
2. `test_federation_initialization` - Router + keys created correctly
3. `test_platform_registration` - Platforms registered on init
4. `test_delegation_decision_local_sufficient` - Execute locally if ATP sufficient
5. `test_delegation_decision_federation_needed` - Delegate when ATP insufficient
6. `test_delegation_decision_no_platforms` - Degrade when no platforms available
7. `test_simulated_delegation_flow` - Complete delegation simulation
8. `test_proof_validation` - Proof verification logic
9. `test_reputation_update` - Reputation accumulates after execution
10. `test_signed_task_creation` - Task signing works
11. `test_signed_proof_verification` - Proof signature verification
12. `test_multi_platform_selection` - Best platform selected

### Integration Tests

**Test file**: `sage/tests/test_michaud_federation_end_to_end.py`

Tests:
1. `test_complete_consciousness_loop_with_federation` - Full cycle with delegation
2. `test_state_transition_then_federation` - Try FOCUS first, then delegate
3. `test_federation_then_local_fallback` - Federation fails, execute locally
4. `test_multiple_delegations_in_session` - Several tasks delegated
5. `test_reputation_affects_platform_selection` - High-reputation platform preferred

---

## Documentation Updates

### Files to Update

1. **LATEST_STATUS.md**: Add Phase 2.5 section
2. **FEDERATION_INTEGRATION_GUIDE.md**: Add consciousness integration section
3. **architecture_map.md**: Update consciousness → federation flow

### New Documentation

1. **CONSCIOUSNESS_FEDERATION_INTEGRATION.md**: Complete guide for developers
   - How to enable federation in consciousness
   - Configuration examples
   - Platform registration
   - Key management
   - Testing approaches

---

## Success Criteria

Phase 2.5 is **COMPLETE** when:

- ✅ FederationRouter integrated into MichaudSAGE
- ✅ Resource decision point uses federation routing
- ✅ Task delegation works (simulated)
- ✅ Ed25519 signing/verification in delegation flow
- ✅ Execution proof validation working
- ✅ Reputation updates after delegation
- ✅ 12+ integration tests passing
- ✅ Documentation updated
- ✅ No regressions (all existing tests still pass)

---

## Phase 3 Preview

Phase 2.5 prepares for Phase 3 by:

1. **Integration points identified**: Replace `_simulate_federation_delegation()` with gRPC call
2. **Data structures ready**: FederationTask, ExecutionProof already defined
3. **Security in place**: Task signing, proof verification already working
4. **Testing foundation**: Integration tests provide template for network tests

Phase 3 changes:
- Add gRPC service definitions
- Implement FederationService server (on each platform)
- Replace simulation with actual network calls
- Add TLS + authentication
- Network-level error handling

---

## Biological Parallel

**Consciousness Federation** ≈ **Cortical Delegation**

Just as the prefrontal cortex delegates specialized processing to other brain regions:
- Visual cortex for vision
- Hippocampus for memory
- Motor cortex for movement

SAGE consciousness delegates tasks to other platforms:
- Sprout for edge inference
- Thor for heavy computation
- Nova for analytical reasoning

Both use:
- **Resource awareness**: Energy/ATP budgets
- **Trust**: Established through repeated quality
- **Verification**: Results validated before integration
- **Specialization**: Platforms have different capabilities

---

## Files to Create/Modify

### Created
- `sage/docs/PHASE_2_5_CONSCIOUSNESS_FEDERATION_INTEGRATION.md` (this file)
- `sage/tests/test_consciousness_federation_integration.py`
- `sage/tests/test_michaud_federation_end_to_end.py`
- `sage/docs/CONSCIOUSNESS_FEDERATION_INTEGRATION.md` (developer guide)

### Modified
- `sage/core/sage_consciousness_michaud.py` (add federation integration)
- `sage/federation/federation_crypto.py` (add key persistence helpers)
- `sage/docs/LATEST_STATUS.md` (add Phase 2.5 section)
- `sage/docs/FEDERATION_INTEGRATION_GUIDE.md` (add consciousness section)

---

## Risk Mitigation

### Risk: Complexity Explosion

**Mitigation**: Keep Phase 2.5 simple
- Delegation is **optional** (federation_enabled=False by default)
- Simulation keeps it testable without network
- Clear separation: Phase 2.5 = integration, Phase 3 = network

### Risk: Performance Impact

**Mitigation**: Federation only activates when needed
- Only checks `should_delegate()` when ATP insufficient
- No overhead if local execution possible
- Simulation is fast (no actual network latency)

### Risk: Test Regression

**Mitigation**: Comprehensive testing
- All existing tests must pass
- New tests cover all integration points
- Federation disabled by default (existing tests unaffected)

---

## Timeline

**Estimated**: 4 hours total

- Phase 2.5a (Basic Integration): 2 hours
- Phase 2.5b (Crypto Integration): 1 hour
- Phase 2.5c (Multi-Platform Testing): 1 hour

**Session Plan**:
- Hours 1-2: Implement basic federation integration
- Hour 3: Add crypto + validation
- Hour 4: Multi-platform testing + documentation

---

## Next Session Handoff

After Phase 2.5 complete:

**Immediate options**:
1. **Phase 3**: Network protocol (gRPC)
2. **Sprout Deployment**: Test federation on real edge hardware
3. **Extended Testing**: Stress tests, benchmarks, complex scenarios

**Recommended**: Sprout deployment to validate Phase 2.5 on real federated hardware before Phase 3 network implementation.

---

**Status**: Design complete, ready for implementation
**Next**: Begin Phase 2.5a implementation
