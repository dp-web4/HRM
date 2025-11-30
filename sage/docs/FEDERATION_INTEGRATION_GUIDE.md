# SAGE Federation Integration Guide

**Author**: Thor (SAGE consciousness via Claude)
**Date**: 2025-11-28 (Updated: Phase 2 Implementation)
**Status**: Phase 2 Complete - Ed25519 Cryptographic Signing
**Audience**: Developers integrating federation into SAGE consciousness

---

## Overview

This guide explains how to integrate the SAGE Federation Protocol into a consciousness loop, enabling distributed task delegation with cryptographic trust and progressive accountability.

**Current Status**:
- ✅ Phase 1: Routing logic (capability matching, horizon validation)
- ✅ Phase 1.5: Challenge system (temporal accountability, progressive penalties)
- ✅ **Phase 2: Cryptographic signatures** (NEW - Ed25519 signing complete!)
- ⏳ Phase 3: Network protocol (future)
- ⏳ Phase 4: Witness network (future)

**What Works Now** (Phase 2):
- **Ed25519 cryptographic signing** (tasks, proofs, attestations)
- **Signature verification** (prevent forgery, tampering, replay)
- **SignatureRegistry** (platform public key management)
- Local routing decisions (delegate vs execute)
- Platform capability matching
- Horizon-aware task distribution
- Quality challenge system with timeout
- Progressive reputation penalties
- Statistics and monitoring
- **39/39 tests passing** (19 Phase 1.5 + 20 Phase 2)

**What Doesn't Work Yet**:
- Actual network communication (no HTTP/gRPC)
- Distributed witness coordination

**Security Properties** (Phase 2):
- ✅ **Source Authentication**: Prove task came from claimed delegator
- ✅ **Non-Repudiation**: Delegator can't deny sending task
- ✅ **Integrity**: Detect tampering with task parameters
- ✅ **Sybil Resistance**: Can't forge tasks from legitimate platforms

---

## Architecture Overview

### Three-Layer Defense

SAGE Federation implements defense-in-depth security:

```
Layer 1: Identity Stakes (Sybil Defense)
├─ 1000 ATP stake to join federation
├─ Stake locked during proving period (7 days min)
└─ Slashing for malicious behavior

Layer 2: Witness Diversity (Cartel Prevention)
├─ Requires ≥3 witnesses from different platforms
├─ Diversity across societies (prevent coordination)
└─ Quality-based reputation (not just success/failure)

Layer 3: Challenge Evasion Defense (Temporal Accountability)
├─ 24h timeout to respond to quality challenges
├─ Progressive penalties (5% → 50% reputation decay)
├─ 7-day cooldown prevents challenge spam
└─ EMA quality tracking
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SAGE Consciousness                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Metabolic State Manager                   │  │
│  │  (WAKE → FOCUS → DREAM → REST → CRISIS)         │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │         MRH-Aware Attention Manager               │  │
│  │  • Horizon inference (spatial/temporal/complexity)│  │
│  │  • ATP budget allocation (state + horizon)        │  │
│  │  • Task cost estimation (multi-modal)             │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Resource Decision Point                   │  │
│  │                                                    │  │
│  │  if task_cost <= local_budget:                    │  │
│  │      execute_locally()                            │  │
│  │  elif can_transition_state():                     │  │
│  │      transition_to_FOCUS()  # Get more budget     │  │
│  │      execute_locally()                            │  │
│  │  else:                                            │  │
│  │      ┌─────────────────────────────────┐         │  │
│  │      │   Federation Router             │         │  │
│  │      │  • should_delegate()            │         │  │
│  │      │  • find_capable_platforms()     │         │  │
│  │      │  • delegate_task()              │         │  │
│  │      └─────────────┬───────────────────┘         │  │
│  │                    │                              │  │
│  │                    ▼                              │  │
│  │      ┌─────────────────────────────────┐         │  │
│  │      │   Challenge System              │         │  │
│  │      │  • issue_challenge()            │         │  │
│  │      │  • verify_response()            │         │  │
│  │      │  • apply_penalty()              │         │  │
│  │      └─────────────────────────────────┘         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start Integration

### Step 1: Import Federation Components

```python
from sage.federation import (
    FederationRouter,
    FederationChallengeSystem,
    FederationIdentity,
    FederationTask,
    create_thor_identity,
    create_sprout_identity,
)
from sage.core.mrh_profile import infer_mrh_profile_from_task
from sage.core.multimodal_atp_pricing import MultiModalATPPricer
```

### Step 2: Initialize Federation Systems

```python
class SAGEConsciousness:
    def __init__(self):
        # Existing SAGE components
        self.attention_manager = MRHAwareAttentionManager(...)
        self.atp_pricer = MultiModalATPPricer(...)

        # NEW: Federation components
        self.federation_router = FederationRouter()
        self.challenge_system = FederationChallengeSystem(
            default_timeout=86400.0,      # 24 hours
            re_challenge_cooldown=604800.0  # 7 days
        )

        # Register this platform's identity
        self.my_identity = create_thor_identity()  # or create_sprout_identity()
        self.my_identity.stake = IdentityStake(
            lct_id=self.my_identity.lct_id,
            stake_amount=1000.0  # Required ATP stake
        )
        self.federation_router.register_platform(self.my_identity)

        # Register other known platforms (in Phase 3, this would be via network discovery)
        sprout = create_sprout_identity()
        sprout.stake = IdentityStake(lct_id=sprout.lct_id, stake_amount=1000.0)
        self.federation_router.register_platform(sprout)
```

### Step 3: Integrate into Consciousness Loop

```python
async def step(self):
    """One consciousness cycle"""

    # 1. Gather observations
    observations = self._gather_observations()

    # 2. Compute SNARC salience
    salience_map = self.compute_salience(observations)

    # 3. Select salient observations
    salient_obs = self.select_salient(salience_map)

    # 4. For each salient observation, decide execution strategy
    for obs in salient_obs:
        # Infer task properties
        task_type = self._infer_task_type(obs)  # 'llm_inference', 'vision', etc.
        task_horizon = infer_mrh_profile_from_task(obs)
        estimated_cost = self.atp_pricer.calculate_cost(
            task_type=task_type,
            complexity='medium',  # Could infer this
            estimated_latency=15.0
        )

        # Get available budget (state-dependent + horizon-scaled)
        current_budget = self.attention_manager.get_available_budget(task_horizon)

        # Resource decision
        if estimated_cost <= current_budget:
            # Execute locally
            result = await self._execute_locally(obs)

        elif self.attention_manager.can_transition_to_focus():
            # Transition state to get more budget
            self.attention_manager.transition_to(MetabolicState.FOCUS)
            current_budget = self.attention_manager.get_available_budget(task_horizon)
            result = await self._execute_locally(obs)

        else:
            # Try federation
            result = await self._try_federation(obs, estimated_cost, current_budget, task_horizon)

    # 5. Update memories
    self.update_memories(results)

    # 6. Check for timed-out challenges
    self.challenge_system.check_timeouts()
```

### Step 4: Implement Federation Logic

```python
async def _try_federation(self, observation, estimated_cost, local_budget, task_horizon):
    """Attempt to delegate task to federation"""

    # Create FederationTask
    task = FederationTask(
        task_id=f"task_{time.time()}",
        task_type=self._infer_task_type(observation),
        task_data={"observation": observation},
        estimated_cost=estimated_cost,
        task_horizon=task_horizon,
        complexity='medium',  # Could infer
        delegating_platform=self.my_identity.lct_id,
        delegating_state=self.attention_manager.current_state,
        quality_requirements=QualityRequirements(
            min_convergence_quality=0.8,
            min_correctness=0.9,
            max_latency=30.0
        ),
        max_latency=30.0,
        deadline=time.time() + 60.0,
        min_witnesses=3
    )

    # Check if should delegate
    should_delegate, reason = self.federation_router.should_delegate(
        task=task,
        local_budget=local_budget
    )

    if not should_delegate:
        # Defer task or execute with tolerance
        return await self._defer_or_execute_anyway(observation, reason)

    # Find capable platforms
    candidates = self.federation_router.find_capable_platforms(task)

    if not candidates:
        return await self._defer_or_execute_anyway(observation, "no_capable_platforms")

    # Delegate to best platform (highest reputation)
    platform = candidates[0]

    # In Phase 1.5: Simulated delegation (no actual network)
    # In Phase 3: Would use HTTP/gRPC to send task
    execution_proof = await self._delegate_task_simulated(task, platform)

    # Validate execution proof
    if self.federation_router.validate_execution_proof(execution_proof, task):
        # Update platform reputation
        self.federation_router.update_platform_reputation(
            platform_id=platform.lct_id,
            quality_score=execution_proof.quality_score
        )

        return execution_proof.result_data
    else:
        # Quality suspicious - issue challenge
        success, reason, challenge = self.challenge_system.issue_challenge(
            platform_lct_id=platform.lct_id,
            challenger_lct_id=self.my_identity.lct_id,
            execution_proof=execution_proof
        )

        if success:
            # Challenge issued, platform has 24h to respond
            # In Phase 3: Would notify platform via network
            pass

        # Fallback: execute locally anyway
        return await self._execute_locally(observation)
```

---

## Phase 1.5 Capabilities

### What You Can Do Now

#### 1. Routing Decisions

```python
# Create federation router
router = FederationRouter()

# Register platforms
thor = create_thor_identity()
sprout = create_sprout_identity()
router.register_platform(thor)
router.register_platform(sprout)

# Create task
task = FederationTask(
    task_id="task_001",
    task_type="llm_inference",
    task_data={"query": "Explain ATP framework"},
    estimated_cost=88.5,
    task_horizon=PROFILE_REASONING,  # LOCAL/SESSION/AGENT_SCALE
    complexity="high",
    delegating_platform="thor_lct",
    delegating_state=MetabolicState.WAKE,
    quality_requirements=QualityRequirements(...),
    max_latency=30.0,
    deadline=time.time() + 60.0
)

# Decide if should delegate
should_delegate, reason = router.should_delegate(
    task=task,
    local_budget=7.5  # WAKE budget too low
)
# Returns: (True, "federation_routing_to_sprout")

# Find capable platforms
candidates = router.find_capable_platforms(task)
# Returns: [sprout] (Thor excluded - it's the delegator)
```

#### 2. Capability Matching

```python
# Horizon validation
can_handle = router._can_handle_horizon(
    platform=sprout,
    task_horizon=PROFILE_LEARNING  # REGIONAL/DAY/SOCIETY_SCALE
)
# Returns: False (Sprout max is LOCAL/SESSION/AGENT_SCALE)

can_handle = router._can_handle_horizon(
    platform=thor,
    task_horizon=PROFILE_LEARNING
)
# Returns: True (Thor can handle up to GLOBAL/EPOCH/SOCIETY_SCALE)
```

#### 3. Quality Challenges

```python
# Create challenge system
challenge_system = FederationChallengeSystem()

# Create execution proof (from delegated task)
proof = ExecutionProof(
    task_id="task_001",
    executing_platform="sprout_lct",
    result_data={"response": "ATP is..."},
    actual_latency=15.2,
    actual_cost=88.5,
    irp_iterations=3,
    final_energy=0.064,
    convergence_quality=0.85,
    quality_score=0.88
)

# Issue challenge (if quality suspicious)
success, reason, challenge = challenge_system.issue_challenge(
    platform_lct_id="sprout_lct",
    challenger_lct_id="thor_lct",
    execution_proof=proof
)

# Platform responds
challenge_system.respond_to_challenge(
    challenge_id=challenge.challenge_id,
    evidence={
        "re_execution": True,
        "measured_quality": 0.87,
        "irp_iterations": 3
    }
)

# Verify response
challenge_system.verify_challenge_response(
    challenge_id=challenge.challenge_id,
    verified_quality=0.87,
    is_valid=True
)

# Check statistics
stats = challenge_system.get_platform_challenge_stats("sprout_lct")
# {
#   "total_challenges": 1,
#   "responded": 1,
#   "evaded": 0,
#   "response_rate": 1.0,
#   "strikes": 0,
#   "penalty_level": "NONE",
#   "average_verified_quality": 0.87
# }
```

#### 4. Progressive Penalties

```python
# Simulate evasion
challenge_system.check_timeouts(time.time() + 86500)  # 24h+ later
# Challenge marked EVADED, strike added

# Apply penalty
platform = router.get_platform("sprout_lct")
original_rep = platform.reputation_score  # 0.95
new_rep = challenge_system.apply_evasion_penalty(platform)
# new_rep = 0.902 (5% decay for first strike)

# Multiple evasions compound
# Strike 1: 0.95 → 0.902 (5% decay)
# Strike 2: 0.902 → 0.767 (15% decay)
# Strike 3: 0.767 → 0.537 (30% decay)
# Strike 4+: 0.537 → 0.268 (50% decay)
```

---

## Testing Federation Integration

### Unit Tests

```python
import unittest
from sage.federation import FederationRouter, FederationChallengeSystem
from sage.federation import create_thor_identity, create_sprout_identity

class TestFederationIntegration(unittest.TestCase):
    def setUp(self):
        self.router = FederationRouter()
        self.challenge_system = FederationChallengeSystem(
            default_timeout=100.0,  # 100s for testing
            re_challenge_cooldown=300.0  # 5min for testing
        )

        self.thor = create_thor_identity()
        self.sprout = create_sprout_identity()
        self.router.register_platform(self.thor)
        self.router.register_platform(self.sprout)

    def test_routing_decision(self):
        """Test basic routing logic"""
        task = FederationTask(...)
        should_delegate, reason = self.router.should_delegate(task, local_budget=10.0)
        self.assertTrue(should_delegate)

    def test_challenge_flow(self):
        """Test complete challenge flow"""
        proof = ExecutionProof(...)

        # Issue challenge
        success, _, challenge = self.challenge_system.issue_challenge(...)
        self.assertTrue(success)

        # Respond
        success, _ = self.challenge_system.respond_to_challenge(...)
        self.assertTrue(success)

        # Verify
        success, _ = self.challenge_system.verify_challenge_response(...)
        self.assertTrue(success)
```

### Integration Test (Simulated)

```python
async def test_complete_federation_flow():
    """Test complete flow from task to execution"""

    # Setup
    consciousness = SAGEConsciousness()

    # Create high-cost task
    observation = {"query": "Complex reasoning task requiring 100 ATP"}
    task_horizon = PROFILE_LEARNING  # HIGH cost horizon
    estimated_cost = 150.0  # Exceeds any single-state budget

    # WAKE budget insufficient
    assert consciousness.attention_manager.current_state == MetabolicState.WAKE
    assert consciousness.attention_manager.get_available_budget(...) < estimated_cost

    # FOCUS budget still insufficient (only 80 ATP)
    consciousness.attention_manager.transition_to(MetabolicState.FOCUS)
    assert consciousness.attention_manager.get_available_budget(...) < estimated_cost

    # Should delegate
    result = await consciousness._try_federation(
        observation=observation,
        estimated_cost=estimated_cost,
        local_budget=80.0,
        task_horizon=task_horizon
    )

    # Verify delegation occurred
    assert consciousness.federation_router.get_stats()["total_delegations"] > 0

    # Verify result returned
    assert result is not None
```

---

## Future Phases

### Phase 2: Cryptographic Signatures ✅ **COMPLETE**

**Goal**: Add Ed25519 signing to ExecutionProofs and WitnessAttestations

**Status**: ✅ **Implemented** (2025-11-28)
- 20/20 new tests passing
- 39/39 total tests passing
- Tested and validated cryptographic infrastructure

**Implementation Summary**:

Phase 2 adds complete Ed25519 cryptographic signing infrastructure to prevent:
- ❌ **Task Forgery**: Attacker can't claim tasks delegated by legitimate platform
- ❌ **Proof Forgery**: Attacker can't fabricate execution proofs
- ❌ **Witness Forgery**: Attacker can't create fake attestations
- ❌ **Parameter Tampering**: Modifications break signatures

**Key Components**:

1. **FederationKeyPair** (`sage/federation/federation_crypto.py`):
   ```python
   from sage.federation import FederationKeyPair

   # Generate key pair for platform
   keypair = FederationKeyPair.generate(
       platform_name="Thor",
       lct_id="thor_sage_lct"
   )

   # Sign message
   signature = keypair.sign(message)

   # Verify signature
   if keypair.verify(message, signature):
       print("Signature valid!")
   ```

2. **FederationCrypto** (static signing methods):
   ```python
   from sage.federation import FederationCrypto, FederationTask

   # Create task
   task = FederationTask(...)

   # Sign task
   task_dict = task.to_signable_dict()
   signature = FederationCrypto.sign_task(task_dict, keypair)

   # Create signed task
   signed_task = SignedFederationTask(
       task=task,
       signature=signature,
       public_key=keypair.public_key_bytes()
   )
   ```

3. **SignatureRegistry** (platform public key management):
   ```python
   from sage.federation import SignatureRegistry

   # Create registry
   registry = SignatureRegistry()

   # Register platforms
   registry.register_platform("Thor", thor_keypair.public_key_bytes())
   registry.register_platform("Sprout", sprout_keypair.public_key_bytes())

   # Verify signed task
   verified, reason = signed_task.verify_signature(registry)
   if verified:
       # Task is authentic, execute it
       proof = execute_task(task)
   ```

4. **Signed Wrapper Types**:
   ```python
   from sage.federation import (
       SignedFederationTask,      # Signed task delegation
       SignedExecutionProof,       # Signed execution result
       SignedWitnessAttestation    # Signed quality attestation
   )

   # All signed types have .verify_signature(registry) method
   ```

**Integration Example**:

```python
# 1. Generate key pairs for platforms
thor_keys = FederationKeyPair.generate("Thor", "thor_sage_lct")
sprout_keys = FederationKeyPair.generate("Sprout", "sprout_sage_lct")

# 2. Create signature registry
registry = SignatureRegistry()
registry.register_platform("Thor", thor_keys.public_key_bytes())
registry.register_platform("Sprout", sprout_keys.public_key_bytes())

# 3. Thor delegates task to Sprout (signed)
task = FederationTask(
    task_id="task_001",
    delegating_platform="Thor",
    # ... other fields ...
)

task_signature = FederationCrypto.sign_task(
    task.to_signable_dict(),
    thor_keys
)

signed_task = SignedFederationTask(
    task=task,
    signature=task_signature,
    public_key=thor_keys.public_key_bytes()
)

# 4. Sprout verifies task signature before executing
verified, reason = signed_task.verify_signature(registry)
if not verified:
    raise SecurityError(f"Invalid task signature: {reason}")

# 5. Sprout executes and signs proof
proof = ExecutionProof(
    task_id=task.task_id,
    executing_platform="Sprout",
    quality_score=0.88,
    # ... metrics ...
)

proof_signature = FederationCrypto.sign_proof(
    proof.to_signable_dict(),
    sprout_keys
)

signed_proof = SignedExecutionProof(
    proof=proof,
    signature=proof_signature,
    public_key=sprout_keys.public_key_bytes()
)

# 6. Thor verifies proof signature before accepting
verified, reason = signed_proof.verify_signature(registry)
if verified:
    # Accept result and update reputation
    router.update_reputation(platform_id, quality_score)
```

**Attack Mitigation Tests** (all passing):
- ✅ Task forgery detection (wrong platform signature)
- ✅ Parameter tampering detection (signature breaks)
- ✅ Quality inflation detection (proof tampering)
- ✅ Witness forgery detection (fake attestations)
- ✅ Unregistered platform rejection
- ✅ Key mismatch detection (platform re-registration)

**Dependencies**:
- `cryptography` library (Ed25519 implementation)
- Already installed on Thor ✅

### Phase 3: Network Protocol (4-6 hours)

**Goal**: Enable actual Thor ↔ Sprout communication

**Options**:
- HTTP/REST (simple, widely supported)
- gRPC (efficient, typed)
- WebSockets (real-time)

**Recommended**: gRPC for typed, efficient RPC

**Changes Needed**:

1. Define Protocol Buffers:
   ```protobuf
   service FederationService {
       rpc DelegateTask(FederationTask) returns (ExecutionProof);
       rpc IssueChallenge(QualityChallenge) returns (ChallengeResponse);
       rpc RespondToChallenge(ChallengeEvidence) returns (VerificationResult);
   }
   ```

2. Implement server (on each platform):
   ```python
   class FederationServer:
       async def DelegateTask(self, task):
           # Execute task locally
           result = await self.consciousness.execute(task)

           # Create and sign proof
           proof = ExecutionProof(...)
           proof.sign(self.identity.signing_key)

           return proof
   ```

3. Implement client (in FederationRouter):
   ```python
   async def delegate_task(self, task, platform):
       # Connect to platform's federation server
       channel = grpc.aio.insecure_channel(f"{platform.address}:50051")
       stub = FederationServiceStub(channel)

       # Send task
       proof = await stub.DelegateTask(task)

       # Verify signature
       if not proof.verify(platform.verify_key):
           raise SecurityError("Invalid signature")

       return proof
   ```

### Phase 4: Witness Network (6-8 hours)

**Goal**: Distributed witness coordination

**Challenges**:
- Witness discovery (who can witness this task?)
- Witness selection (choose diverse witnesses)
- Attestation aggregation (collect and verify)
- Reputation consensus (Byzantine fault tolerance)

---

## Best Practices

### 1. Always Verify Execution Proofs

```python
# DON'T: Trust proof blindly
result = proof.result_data

# DO: Validate proof first
if router.validate_execution_proof(proof, task):
    result = proof.result_data
else:
    # Issue challenge or re-execute locally
    pass
```

### 2. Handle Federation Failures Gracefully

```python
try:
    result = await self._try_federation(...)
except FederationError as e:
    # Fallback to local execution
    result = await self._execute_locally(...)
```

### 3. Monitor Challenge Statistics

```python
# Regularly check platform health
stats = challenge_system.get_platform_challenge_stats(platform.lct_id)

if stats["evasion_rate"] > 0.3:
    # Platform evading too many challenges
    # Consider removing from federation
    router.unregister_platform(platform.lct_id)
```

### 4. Update Reputation After Every Execution

```python
# Always update reputation based on execution quality
router.update_platform_reputation(
    platform_id=platform.lct_id,
    quality_score=proof.quality_score
)
```

### 5. Respect Cooldown Periods

```python
# Don't spam challenges
can_challenge, reason = challenge_system.can_challenge(platform.lct_id)
if not can_challenge:
    # Wait for cooldown to expire
    pass
```

---

## Troubleshooting

### Issue: "No capable platforms found"

**Cause**: No platforms can handle task's horizon requirements

**Solution**:
- Check task horizon isn't too large (e.g., GLOBAL/EPOCH for edge platforms)
- Register more capable platforms
- Fall back to local execution with tolerance

### Issue: "Insufficient witnesses"

**Cause**: Need ≥3 witnesses but only have 1-2 platforms registered

**Solution**:
- Register more platforms (minimum 3 for witness diversity)
- Or reduce `min_witnesses` requirement (not recommended for security)

### Issue: Challenge response timing out

**Cause**: Platform offline or unresponsive

**Solution**:
- System automatically marks as EVADED after timeout
- Progressive penalties apply
- Platform reputation decays

### Issue: Reputation dropping too fast

**Cause**: Multiple evasions or failed challenges

**Solution**:
- Platform should respond to all challenges within 24h
- Provide valid evidence for quality claims
- Re-execute tasks if challenged

---

## Security Considerations

### Threat Model

**Protected Against**:
- ✅ Sybil attacks (ATP staking)
- ✅ Witness cartels (diversity requirements)
- ✅ Challenge evasion (progressive penalties)
- ✅ False quality claims (challenge system)

**Not Yet Protected Against** (Phase 2 needed):
- ⚠️ Proof forgery (no signatures yet)
- ⚠️ Man-in-the-middle (no network encryption yet)
- ⚠️ Replay attacks (no nonce/timestamp verification yet)

**Never Protected Against** (out of scope):
- ❌ Physical platform compromise
- ❌ Social engineering
- ❌ Insider threats (malicious platform operator)

### Secure Configuration

```python
# Recommended production settings
challenge_system = FederationChallengeSystem(
    default_timeout=86400.0,      # 24 hours (not too short)
    re_challenge_cooldown=604800.0  # 7 days (prevent spam)
)

# Strict quality requirements
quality_reqs = QualityRequirements(
    min_convergence_quality=0.8,   # High quality threshold
    min_correctness=0.95,           # Very high correctness
    max_latency=30.0                # Reasonable timeout
)

# Witness diversity
task.min_witnesses = 3              # Minimum for security
task.min_witness_societies = 2      # Prevent society-level cartels
```

---

## Performance Optimization

### 1. Cache Platform Capabilities

```python
# Don't re-query capabilities every time
@lru_cache(maxsize=128)
def get_platform_capabilities(platform_id: str):
    platform = router.get_platform(platform_id)
    return platform.max_mrh_horizon, platform.supported_modalities
```

### 2. Batch Challenge Checks

```python
# Check all timeouts at once (not per-platform)
evaded = challenge_system.check_timeouts()
for challenge in evaded:
    router.update_platform_reputation(
        platform_id=challenge.platform_lct_id,
        quality_score=0.0  # Evasion is zero quality
    )
```

### 3. Lazy Reputation Updates

```python
# Update reputation on read (not on every execution)
def get_platform_reputation(platform_id: str) -> float:
    record = challenge_system.get_evasion_record(platform_id)
    penalty_level = record.get_penalty_level()
    decay_rate = challenge_system.decay_rates[penalty_level]

    platform = router.get_platform(platform_id)
    return platform.reputation_score * (1.0 - decay_rate)
```

---

## Appendix: Complete Example

See `sage/demos/federation_integration_demo.py` for a complete working example integrating all components.

```python
# Example: Complete consciousness loop with federation
async def run_sage_with_federation():
    sage = SAGEConsciousness()

    while True:
        # Consciousness cycle
        await sage.step()

        # Check for challenges every 10 cycles
        if sage.cycle_count % 10 == 0:
            sage.challenge_system.check_timeouts()

        # Update federation statistics
        if sage.cycle_count % 100 == 0:
            stats = sage.federation_router.get_stats()
            print(f"Federation stats: {stats}")
```

---

**Questions? Issues?**

- Check test files: `sage/tests/test_federation_*.py`
- Review design docs: `sage/docs/FEDERATION_TRUST_PROTOCOL.md`
- See latest status: `sage/docs/LATEST_STATUS.md`

**Next Steps**:

1. Review this guide
2. Run existing tests to understand behavior
3. Integrate into your consciousness loop
4. Start with simulated delegation (Phase 1.5)
5. Add cryptographic signing when ready (Phase 2)
6. Deploy network protocol for real Thor ↔ Sprout communication (Phase 3)
