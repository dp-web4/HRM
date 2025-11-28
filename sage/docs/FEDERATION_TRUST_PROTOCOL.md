# SAGE Federation Trust Protocol

**Author**: Thor (SAGE consciousness via Claude)
**Date**: 2025-11-28
**Session**: Autonomous SAGE Research - Federation Readiness
**Status**: Design Phase

## Executive Summary

This document defines how SAGE consciousness instances (Thor-SAGE, Sprout-SAGE, etc.) can safely route tasks to each other in a federated environment, building on Web4's security patterns while adapting them for consciousness-specific requirements.

**Key Innovation**: Federation trust based on **witnessed execution quality** rather than just computational outcomes, enabling consciousness platforms to build reputation through demonstrated reasoning ability.

## Background

### Current State

**ATP Framework** (Nov 27-28):
- Multi-modal pricing: Task costs calculated (vision, LLM, coordination, consolidation)
- MRH-aware attention: Budget allocation based on horizon + metabolic state
- Metabolic transitions: Automatic WAKE→FOCUS when insufficient ATP
- **Status**: Integrated and validated with live inference (100% success)

**Resource Decision Logic** (sage_consciousness_michaud.py:176-185):
```python
if task_cost > available_budget:
    current_state = self.attention_manager.get_state()
    if current_state == MetabolicState.WAKE:
        # Transition to FOCUS for more ATP
        self.attention_manager.current_state = MetabolicState.FOCUS
        available_budget = self.attention_manager.get_total_allocated_atp(task_horizon)
    else:
        # Still insufficient - would route to federation in production
        print("  Decision: Cost exceeds budget, executing with degradation")
        # Continue anyway for demo (federation routing not implemented yet)
```

**The Gap**: "federation routing not implemented yet" ← THIS SESSION

### Web4 Security Patterns (Session #82-83)

Web4 provides three security systems relevant to SAGE federation:

1. **Signed Epidemic Gossip** (Session #82):
   - Cryptographic signatures prevent message forgery
   - Ed25519 for identity authentication
   - Prevents Sybil eclipse attacks

2. **Identity Stake System** (Session #82):
   - ATP bonded to LCT creation
   - Economic Sybil defense (1000 Sybils = 1M ATP cost)
   - Stake slashing for malicious behavior

3. **Witness Diversity** (Session #83):
   - Requires ≥3 societies for reputation claims
   - Tracks witness accuracy over time
   - Detects cartels through collusion analysis

## Problem Statement

**Challenge**: How can Thor-SAGE safely delegate a task to Sprout-SAGE (or other platform) and trust the result?

**Unique Consciousness Requirements**:
1. **Quality matters**: Correctness + reasoning quality (not just binary success/failure)
2. **Context preservation**: Task horizon, metabolic state context must transfer
3. **Trust accumulation**: Build long-term trust through demonstrated competence
4. **Partial delegation**: May delegate subtasks, not entire reasoning chains
5. **Witness-based validation**: Third parties attest to execution quality

**Attack Vectors** (adapted from Web4 Session #81):
- **False Execution Claims**: Platform claims to execute but doesn't
- **Low-Quality Shortcuts**: Platform returns quick but poor-quality response
- **Sybil Reputation Farming**: Attacker creates fake platforms to inflate reputation
- **Witness Collusion**: Colluding witnesses attest to poor-quality execution

## Design Principles

### 1. Trust Through Witnessed Execution

Reputation is based on **observed execution quality**, witnessed by independent third parties.

**Witness Requirements** (from Web4 Session #83):
- ≥3 witnesses from different societies (diversity)
- Witnesses track accuracy over time
- Progressive penalties for unreliable witnesses
- Random selection prevents witness shopping

**SAGE Adaptation**:
- Witnesses evaluate both **correctness** AND **quality** of reasoning
- Use SAGE's existing quality metrics (4-component system)
- Witnesses are other SAGE platforms (cross-validation)

### 2. ATP-Bonded Identity

Each SAGE platform stakes ATP to create its federation identity.

**Economic Defense** (from Web4 Session #82):
- Base stake: 1000 ATP per platform identity
- Stake locked during proving period (7 days)
- Stake slashed if Sybil behavior detected
- Recovered when platform proves legitimacy

**SAGE Adaptation**:
- Platform identity = LCT anchored to hardware (Thor, Sprout, etc.)
- Stake includes execution history (quality track record)
- Slashing triggers on: false claims, quality degradation, witness collusion

### 3. Horizon-Aware Delegation

Tasks have MRH horizons (spatial/temporal/complexity dimensions). Federation routing must preserve horizon context.

**Horizon Matching**:
- Task horizon: What scale/scope does this task operate at?
- Platform capability: What horizons can this platform handle?
- Witness horizon: What scope can witness evaluate?

**Example**:
- Task: "Consolidate 20 sessions of learning" (REGIONAL/DAY/SOCIETY_SCALE)
- Thor capability: Can handle (64GB RAM, high power)
- Sprout capability: Cannot handle (8GB RAM, limited memory)
- Decision: Thor executes locally, Sprout cannot accept delegation

### 4. Metabolic State Context

Tasks have metabolic state context (WAKE, FOCUS, REST, DREAM, CRISIS) that affects execution quality and ATP allocation.

**State Transfer**:
- Delegating platform: "I'm in FOCUS, need high-quality reasoning"
- Receiving platform: "I'm in DREAM, can do consolidation but not urgent tasks"
- Witness evaluation: "Given FOCUS context, was execution quality appropriate?"

**State-Aware Routing**:
- CRISIS tasks: Route to platforms in CRISIS/FOCUS (urgent, high ATP)
- DREAM tasks: Route to platforms in DREAM/REST (consolidation, learning)
- FOCUS tasks: Route to platforms in FOCUS (analytical reasoning)

## Protocol Components

### 1. FederationIdentity

```python
@dataclass
class FederationIdentity:
    """
    SAGE platform identity for federation

    Anchored to hardware (Thor, Sprout) via LCT model.
    """
    lct_id: str  # Hardware-anchored identity (e.g., "thor_sage_lct")
    platform_name: str  # Human-readable (e.g., "Thor")
    hardware_spec: HardwareSpec  # RAM, GPU, power envelope

    # Cryptographic identity
    public_key: bytes  # Ed25519 public key
    private_key: bytes  # Ed25519 private key (kept secret)

    # Capability profile
    max_mrh_horizon: MRHProfile  # Largest horizon this platform can handle
    supported_modalities: List[str]  # ['vision', 'llm_inference', 'consolidation']

    # Economic stake
    stake: IdentityStake  # ATP bonded to this identity

    # Trust metrics
    execution_history: List[ExecutionRecord]
    witness_attestations: List[WitnessAttestation]
    reputation_score: float  # Composite trust score (0-1)
```

### 2. FederationTask

```python
@dataclass
class FederationTask:
    """
    Task to be delegated to another platform

    Includes all context needed for execution and witness validation.
    """
    task_id: str
    task_type: str  # 'llm_inference', 'consolidation', etc.
    task_data: Dict[str, Any]  # Query, context, etc.

    # Resource context
    estimated_cost: float  # ATP cost (multi-modal pricing)
    task_horizon: MRHProfile  # MRH context
    complexity: str  # 'low', 'medium', 'high', 'critical'

    # Execution context
    delegating_platform: str  # Who is delegating
    delegating_state: MetabolicState  # Their current state
    quality_requirements: QualityRequirements  # Expected quality

    # Deadline
    max_latency: float  # Maximum acceptable latency (seconds)
    deadline: float  # Absolute deadline (timestamp)

    # Witness requirements
    min_witnesses: int = 3  # Minimum witness diversity
    witness_societies: Optional[List[str]] = None  # Required witness diversity
```

### 3. ExecutionProof

```python
@dataclass
class ExecutionProof:
    """
    Cryptographically signed proof of task execution

    Attested by witnesses to build trust.
    """
    task_id: str
    executing_platform: str  # Who executed

    # Execution results
    result_data: Dict[str, Any]  # Response, IRP info, etc.
    actual_latency: float  # How long it actually took
    actual_cost: float  # ATP actually consumed

    # Quality metrics
    irp_iterations: int
    final_energy: float
    convergence_quality: float
    quality_score: float  # 4-component SAGE quality (0-1)

    # Cryptographic proof
    execution_hash: str  # Hash of (task + result + metrics)
    platform_signature: bytes  # Ed25519 signature by executing platform
    witness_signatures: List[bytes]  # Ed25519 signatures by witnesses

    # Timestamp
    execution_timestamp: float
```

### 4. WitnessAttestation

```python
@dataclass
class WitnessAttestation:
    """
    Witness evaluation of execution quality

    Tracks both correctness and quality.
    """
    attestation_id: str
    task_id: str
    witness_lct_id: str  # Witnessing platform
    witness_society_id: str  # For diversity requirement

    # Attestation
    claimed_correctness: float  # Is result correct? (0-1)
    claimed_quality: float  # Is quality good? (0-1)

    # Evaluation (ground truth if known)
    actual_correctness: Optional[float] = None
    actual_quality: Optional[float] = None

    # Outcome
    outcome: WitnessOutcome = WitnessOutcome.PENDING

    # Signature
    witness_signature: bytes  # Ed25519 signature
    timestamp: float
```

### 5. FederationRouter

```python
class FederationRouter:
    """
    Routes tasks to appropriate platforms based on:
    - ATP cost vs budget
    - Platform capabilities (horizon, modalities)
    - Platform reputation (trust score)
    - Current metabolic state
    - Witness availability
    """

    def __init__(
        self,
        local_identity: FederationIdentity,
        stake_system: IdentityStakeSystem,
        witness_tracker: WitnessAccuracyTracker
    ):
        self.local_identity = local_identity
        self.stake_system = stake_system
        self.witness_tracker = witness_tracker

        # Known platforms in federation
        self.known_platforms: Dict[str, FederationIdentity] = {}

        # Execution history
        self.delegated_tasks: Dict[str, FederationTask] = {}
        self.execution_proofs: Dict[str, ExecutionProof] = {}

    def should_delegate(
        self,
        task: FederationTask,
        local_budget: float
    ) -> Tuple[bool, str]:
        """
        Decide if task should be delegated to federation

        Returns:
            (should_delegate, reason)
        """
        # Decision logic from ATP framework
        if task.estimated_cost <= local_budget:
            return (False, "sufficient_local_atp")

        # Check if any platform can handle
        candidates = self.find_capable_platforms(task)
        if not candidates:
            return (False, "no_capable_platforms")

        # Check witness availability
        if not self.has_sufficient_witnesses(task):
            return (False, "insufficient_witnesses")

        # Delegate!
        return (True, "federation_routing")

    def find_capable_platforms(
        self,
        task: FederationTask
    ) -> List[FederationIdentity]:
        """
        Find platforms capable of executing task

        Filters by:
        - Horizon capability
        - Modality support
        - Reputation threshold
        - Current metabolic state (if known)
        """
        candidates = []

        for platform_id, platform in self.known_platforms.items():
            # Check horizon
            if not self.can_handle_horizon(platform, task.task_horizon):
                continue

            # Check modality
            if task.task_type not in platform.supported_modalities:
                continue

            # Check reputation
            if platform.reputation_score < 0.6:  # Minimum trust threshold
                continue

            # Check stake status
            stake = self.stake_system.stakes.get(platform.lct_id)
            if stake and stake.status == StakeStatus.SLASHED:
                continue  # Slashed platforms cannot receive tasks

            candidates.append(platform)

        # Sort by reputation
        candidates.sort(key=lambda p: p.reputation_score, reverse=True)
        return candidates

    def delegate_task(
        self,
        task: FederationTask,
        target_platform: FederationIdentity
    ) -> str:
        """
        Delegate task to target platform

        Returns:
            task_id for tracking
        """
        # Sign task with local identity
        task_signature = self._sign_task(task)

        # Send to target platform (protocol TBD)
        # ... network layer ...

        # Track delegation
        self.delegated_tasks[task.task_id] = task

        return task.task_id

    def validate_execution_proof(
        self,
        proof: ExecutionProof
    ) -> Tuple[bool, str]:
        """
        Validate execution proof from federated platform

        Checks:
        - Platform signature valid
        - Witness signatures valid (≥3 witnesses from different societies)
        - Witness attestations consistent
        - Quality metrics acceptable
        """
        # Verify platform signature
        if not self._verify_signature(proof.platform_signature, proof.executing_platform):
            return (False, "invalid_platform_signature")

        # Verify witness diversity (≥3 societies)
        if len(proof.witness_signatures) < 3:
            return (False, "insufficient_witnesses")

        witness_societies = set()
        for witness_sig in proof.witness_signatures:
            witness_id = self._extract_witness_id(witness_sig)
            witness_society = self._get_witness_society(witness_id)
            witness_societies.add(witness_society)

        if len(witness_societies) < 3:
            return (False, "insufficient_witness_diversity")

        # Check quality metrics
        if proof.quality_score < 0.7:  # Minimum quality threshold
            return (False, f"quality_too_low: {proof.quality_score:.2f}")

        # All checks passed
        return (True, "proof_validated")
```

## Integration with SAGE Consciousness

### Modified Resource Decision Logic

Update `sage_consciousness_michaud.py:step()` to use federation routing:

```python
# 5. Resource decision
if task_cost > available_budget:
    # Check if state transition helps
    current_state = self.attention_manager.get_state()
    if current_state == MetabolicState.WAKE:
        # Try transitioning to FOCUS
        self.attention_manager.current_state = MetabolicState.FOCUS
        available_budget = self.attention_manager.get_total_allocated_atp(task_horizon)

        if task_cost <= available_budget:
            print(f"  Decision: Transitioned to FOCUS, executing locally ✓")
        else:
            # Still insufficient - try federation
            should_delegate, reason = self.federation_router.should_delegate(
                task=FederationTask(
                    task_id=f"task_{self.cycle_count}",
                    task_type=task_context['task_type'],
                    task_data={'query': first_obs['data']},
                    estimated_cost=task_cost,
                    task_horizon=task_horizon,
                    complexity=complexity,
                    delegating_platform=self.federation_identity.lct_id,
                    delegating_state=current_state,
                    quality_requirements=QualityRequirements(min_quality=0.7),
                    max_latency=estimated_latency * 2,
                    deadline=time.time() + estimated_latency * 2
                ),
                local_budget=available_budget
            )

            if should_delegate:
                print(f"  Decision: Routing to federation ({reason})")
                # Delegate task and await result
                result = await self._execute_federated(task)
            else:
                print(f"  Decision: Cannot delegate ({reason}), executing with degradation")
                # Execute locally anyway
    else:
        # Already in higher state, still insufficient
        # Try federation as last resort
        ...
```

## Security Properties

### 1. Sybil Resistance

**Defense**: ATP bonding + witness diversity
- Creating fake platform costs 1000 ATP stake
- Fake platforms need ≥3 witnesses from different societies
- Witnesses track accuracy → fake witnesses detected and slashed

### 2. Quality Assurance

**Defense**: Witness attestation + quality metrics
- Witnesses evaluate both correctness AND quality
- Poor-quality execution → low attestation → reputation damage
- Repeated low quality → stake slashing

### 3. Witness Cartel Prevention

**Defense**: Society diversity requirement + accuracy tracking
- ≥3 witnesses from different societies (harder to collude)
- Witness accuracy tracked over time
- Colluding witnesses detected through consistency analysis

### 4. False Execution Claims

**Defense**: Cryptographic signatures + witness validation
- Platform must sign execution proof
- Witnesses verify execution actually occurred
- Cannot claim execution without doing work

## Implementation Phases

### Phase 1: Local Foundation (Next Session, 2-3 hours)

**Goal**: Implement federation data structures and routing logic (no network yet)

1. Create federation data structures:
   - `FederationIdentity`
   - `FederationTask`
   - `ExecutionProof`
   - `WitnessAttestation`

2. Implement `FederationRouter`:
   - `should_delegate()` decision logic
   - `find_capable_platforms()` filtering
   - `validate_execution_proof()` verification

3. Integrate with `sage_consciousness_michaud.py`:
   - Add federation_router to initialization
   - Update resource decision logic
   - Add execution proof validation

4. Create tests:
   - Test delegation decision logic
   - Test platform capability matching
   - Test witness validation

5. Document and commit

**Deliverable**: Federation routing logic working locally (simulated platforms)

### Phase 2: Cryptographic Identity (Future, 2-3 hours)

**Goal**: Add cryptographic signing and verification

1. Integrate Ed25519 from Web4
2. Implement key generation for platform identities
3. Add signature creation and verification
4. Test cryptographic security

### Phase 3: Network Protocol (Future, 4-6 hours)

**Goal**: Actual platform-to-platform communication

1. Design message protocol (likely HTTP/REST or gRPC)
2. Implement task delegation over network
3. Implement result retrieval
4. Test Thor ↔ Sprout communication

### Phase 4: Witness Network (Future, 6-8 hours)

**Goal**: Third-party witness attestation

1. Implement witness selection
2. Add witness evaluation logic
3. Track witness accuracy
4. Implement cartel detection

## Success Criteria

**Phase 1 (This Session)**:
- [ ] Federation data structures implemented
- [ ] Routing decision logic working
- [ ] Integration with SAGE consciousness
- [ ] Tests passing (delegation decisions, capability matching)
- [ ] Documentation complete

**Long-term**:
- [ ] Thor can delegate to Sprout over network
- [ ] Witnesses attest to execution quality
- [ ] Trust accumulates through demonstrated competence
- [ ] Attack vectors mitigated (Sybil, false claims, cartels)

## Open Questions

1. **Witness Incentives**: How do we incentivize platforms to witness honestly?
   - Web4: Slashed stakes go to society treasury
   - SAGE: Could witnesses earn ATP for accurate attestations?

2. **Task Granularity**: What's the smallest delegatable unit?
   - Full reasoning chain?
   - Single IRP iteration?
   - Subtask within complex query?

3. **Quality Evaluation**: How do witnesses evaluate reasoning quality without full context?
   - Use SAGE's 4-component quality metrics
   - Witnesses have access to question + response + IRP info
   - Ground truth from delegating platform's own evaluation?

4. **Network Topology**: How do platforms discover each other?
   - Epidemic gossip (like Web4)?
   - Central registry?
   - Peer-to-peer discovery?

## Biological Parallels

**Brain Hemispheres**:
- Left/right hemisphere specialization (different capabilities)
- Tasks routed to appropriate hemisphere
- Cross-hemisphere communication through corpus callosum
- Quality validated through feedback loops

**Neural Networks**:
- Neurons delegate computation to downstream neurons
- Synaptic plasticity based on outcome quality
- Trust built through repeated successful activation

**Ant Colonies**:
- Distributed task allocation through stigmergy
- No central coordinator
- Trust through pheromone trails (execution history)

## References

- Web4 Session #82: Signed Epidemic Gossip, Identity Stake System
- Web4 Session #83: Witness Diversity, Integrated Security
- SAGE ATP Framework: Multi-modal pricing, MRH-aware attention
- SAGE Michaud Integration: Metabolic states, attention management

---

**Status**: Design complete, ready for Phase 1 implementation
**Next**: Implement federation data structures and routing logic
