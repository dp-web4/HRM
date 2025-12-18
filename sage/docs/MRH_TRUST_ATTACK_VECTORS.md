# Attack Vector Analysis: MRH Trust-Based Expert Selection

**Document**: Security analysis of MRH-based trust systems for MoE models
**Date**: 2025-12-18
**Session**: 66 (Autonomous Web4 Research)
**Context**: Sessions 65-66 demonstrated MRH substitution breaking router monopoly. This document analyzes potential attack vectors and defense mechanisms.

---

## Executive Summary

The MRH trust-based expert selection system (Sessions 65-66) successfully breaks router monopoly through:
1. Context-aware trust tracking
2. Context overlap-based alternative discovery (MRH)
3. Trust-augmented routing (α×router + (1-α)×trust)

However, this system introduces new attack surfaces centered on **trust manipulation** and **context gaming**. This document catalogs attack vectors, assesses risk, and proposes defenses.

**Key Finding**: Most attacks exploit the **trust feedback loop** or **context overlap computation**. Defense-in-depth requires multiple layers:
- Input validation (sanity checks on trust updates)
- Anomaly detection (statistical outliers)
- Byzantine fault tolerance (witness consensus)
- Rate limiting (prevent flooding)
- Cryptographic verification (LCT attestation)

---

## Threat Model

### Adversary Capabilities

**Level 1: Passive Observer**
- Can observe router selections
- Can infer expert usage patterns
- Cannot modify trust scores or inputs

**Level 2: Active Manipulator (Local)**
- Can provide adversarial inputs to specific experts
- Can attempt to game context classification
- Cannot directly modify trust database
- Cannot forge LCT certificates

**Level 3: Compromised Node**
- Can modify local trust scores
- Can forge trust updates
- Can manipulate context overlap computation
- Cannot forge LCT signatures (assumes hardware root of trust)

**Level 4: Network Adversary**
- Can intercept and modify network traffic
- Can perform MITM attacks on trust synchronization
- Can partition network (eclipse attacks)
- Cannot break cryptographic primitives

### Assets to Protect

1. **Trust Scores**: Expert reputation in specific contexts
2. **Context Overlap**: Pairwise similarity relationships
3. **Router Decisions**: Which experts are selected
4. **Capacity Utilization**: Preventing expert starvation/monopoly
5. **Quality Metrics**: Generation quality → trust feedback

---

## Attack Vector Catalog

### Category 1: Trust Manipulation Attacks

#### Attack 1.1: Trust Inflation
**Description**: Adversary provides high-quality outputs temporarily to inflate trust, then exploits high trust for malicious behavior.

**Mechanism**:
1. Initially provide excellent outputs → trust increases
2. Once trust > threshold, start providing low-quality/malicious outputs
3. Exploit window before trust declines below threshold

**Impact**: High (can subvert MRH substitution)

**Detection Difficulty**: Medium (requires monitoring trust velocity)

**Example Scenario**:
```python
# Expert 42 starts with trust=0.5
for i in range(10):
    quality = 0.95  # Excellent quality
    update_trust(expert=42, context=0, quality=quality)
    # trust → 0.95

# Now trust is high, switch to adversarial behavior
for i in range(5):
    quality = 0.1  # Malicious output
    # But MRH won't substitute until trust < 0.3
    # Adversary has 5-10 generations to exploit
```

**Defense**:
- **Trust velocity limits**: Flag rapid trust increases
- **Behavioral consistency checking**: Sudden quality drops trigger review
- **Witness consensus**: Require multiple nodes to observe quality before trust update
- **Exponential decay for outliers**: Large deviations decay trust faster

**Proposed Mitigation**:
```python
def update_trust_with_velocity_check(expert_id, context, quality):
    current_trust = get_trust(expert_id, context)
    new_trust = 0.9 * current_trust + 0.1 * quality  # EWMA

    # Velocity check
    delta = abs(new_trust - current_trust)
    if delta > VELOCITY_THRESHOLD:
        # Suspicious: require witness consensus
        if not verify_quality_with_witnesses(expert_id, context, quality):
            return  # Reject update

    set_trust(expert_id, context, new_trust)
```

---

#### Attack 1.2: Sybil Trust Washing
**Description**: Adversary creates multiple expert identities, transfers "trust" between them via context overlap relationships.

**Mechanism**:
1. Create Sybil expert IDs: {Expert_Sybil_1, Expert_Sybil_2, ...}
2. Manipulate context distributions to create high overlap
3. When one Sybil gets low trust, MRH substitutes with another Sybil
4. Result: Sybil network dominates selections despite low quality

**Impact**: Critical (subverts MRH entirely)

**Detection Difficulty**: High (hard to distinguish from legitimate specialists)

**Example Scenario**:
```python
# Adversary controls experts [200, 201, 202, 203]
# All given identical context distributions
for expert_id in [200, 201, 202, 203]:
    embeddings = adversarial_embedding  # Crafted for high overlap
    register_expert_contexts(expert_id, embeddings)

# Context overlap between all Sybils: 1.0 (perfect)
# MRH will substitute Sybil_1 → Sybil_2 → Sybil_3 → Sybil_4
# Monopoly persists, but distributed across Sybil network
```

**Defense**:
- **LCT identity verification**: Each expert must have valid, unique LCT certificate
- **Hardware attestation**: Bind expert ID to TPM/TEE measurements
- **Rate limiting on expert registration**: Slow down Sybil creation
- **Social graph analysis**: Detect cliques with suspiciously high overlap
- **Behavioral fingerprinting**: Different experts should have unique patterns

**Proposed Mitigation**:
```python
def verify_expert_identity(expert_id):
    # Require LCT certificate with:
    # 1. Valid signature chain to root of trust
    # 2. Unique hardware binding (TPM attestation)
    # 3. Historical behavior pattern (not too similar to others)

    cert = lct_resolver.resolve(expert_uri)

    if not cert.verify_signature():
        raise SecurityError("Invalid LCT signature")

    if not cert.verify_hardware_binding():
        raise SecurityError("No hardware attestation")

    # Check for suspicious similarity to existing experts
    for existing_expert in all_experts:
        overlap, shared = compute_context_overlap(expert_id, existing_expert)
        if overlap > 0.95 and expert_id != existing_expert:
            # Too similar: potential Sybil
            if not verify_independent_hardware(expert_id, existing_expert):
                raise SecurityError(f"Potential Sybil: {expert_id} ≈ {existing_expert}")
```

---

#### Attack 1.3: Context Poisoning
**Description**: Adversary manipulates context classification to force low-trust experts into contexts where they'll be substituted favorably.

**Mechanism**:
1. Learn context classifier decision boundaries
2. Craft adversarial inputs that misclassify into target context
3. Low-trust expert gets substituted with adversary-controlled "specialist"

**Impact**: Medium (requires context classifier vulnerabilities)

**Detection Difficulty**: Low (adversarial inputs often detectable)

**Example Scenario**:
```python
# Adversary knows context_0 = "code" is high-value
# Crafts input that looks like context_1 but classifies as context_0
adversarial_input = craft_adversarial_embedding(
    target_context=0,
    base_input=legitimate_reasoning_task
)

# Classifier misclassifies: reasoning → code
# MRH substitutes with adversary's "code specialist"
# Adversary gains influence over high-value context
```

**Defense**:
- **Ensemble classifiers**: Multiple diverse classifiers vote
- **Adversarial training**: Train classifier on adversarial examples
- **Confidence thresholds**: Reject low-confidence classifications
- **Context consistency checking**: Validate context matches output type

**Proposed Mitigation**:
```python
def classify_with_robustness(embedding):
    # Ensemble of classifiers
    votes = []
    for classifier in context_classifiers:
        context, confidence = classifier.classify(embedding)
        if confidence < CONFIDENCE_THRESHOLD:
            # Low confidence: potential adversarial input
            return None  # Reject or use fallback

        votes.append((context, confidence))

    # Consensus required
    majority_context = max(set(v[0] for v in votes), key=lambda c: sum(v[1] for v in votes if v[0] == c))
    agreement = sum(1 for v in votes if v[0] == majority_context) / len(votes)

    if agreement < 0.7:
        # No consensus: reject
        return None

    return majority_context
```

---

### Category 2: MRH Discovery Attacks

#### Attack 2.1: Overlap Manipulation
**Description**: Adversary crafts context distributions to create artificially high overlap with high-trust experts.

**Mechanism**:
1. Observe high-trust expert's context distribution
2. Mimic distribution to maximize cosine similarity
3. Become preferred MRH substitute despite low quality

**Impact**: High (directly exploits MRH core mechanism)

**Detection Difficulty**: Medium (can detect via behavioral mismatch)

**Example Scenario**:
```python
# High-trust expert 17: context_0=80%, context_1=10%, context_2=10%
# Adversary (expert 200) copies distribution exactly

adversary_distribution = copy(expert_17_distribution)
register_expert_contexts(expert_200, adversary_distribution)

# Context overlap: compute_context_overlap(17, 200) → 1.0 (perfect)
# MRH will now substitute expert_17 → expert_200
# But expert_200 provides low quality!
```

**Defense**:
- **Behavioral verification**: Context distribution should match actual performance
- **Quality-weighted overlap**: Weight overlap by historical quality
- **Temporal consistency**: Context distributions shouldn't change rapidly
- **Cross-validation**: Test expert in claimed contexts before trusting

**Proposed Mitigation**:
```python
def compute_verified_context_overlap(expert_a, expert_b):
    # Standard cosine similarity
    overlap, shared = compute_context_overlap(expert_a, expert_b)

    if overlap > 0.9:
        # Suspiciously high: verify behavioral match
        for context in shared:
            perf_a = get_average_quality(expert_a, context)
            perf_b = get_average_quality(expert_b, context)

            if abs(perf_a - perf_b) > 0.3:
                # Similar contexts but very different performance
                # Penalize overlap
                overlap *= 0.5  # Reduce trust in similarity

    return overlap, shared
```

---

#### Attack 2.2: Eclipse Attack on MRH Discovery
**Description**: Adversary controls all visible MRH alternatives, preventing discovery of legitimate specialists.

**Mechanism**:
1. Adversary controls majority of experts with context overlap > threshold
2. Legitimate specialists exist but have slightly lower overlap
3. MRH always finds adversarial alternatives
4. Legitimate specialists starve

**Impact**: Critical (denial of service for specialists)

**Detection Difficulty**: High (looks like natural MRH selection)

**Example Scenario**:
```python
# Legitimate specialist: expert 42 (trust=0.85, overlap=0.72)
# Adversary controls: experts [200-220] (trust=0.4, overlap=0.75)

# MRH search for alternatives
alternatives = []
for expert in all_experts:
    overlap, shared = compute_context_overlap(failing_expert, expert)
    if overlap >= 0.7:
        alternatives.append((expert, get_trust(expert, context), overlap))

# Adversary experts have overlap=0.75 > threshold
# Legitimate expert has overlap=0.72 > threshold
# But adversary has 21 experts vs 1 legitimate
# Statistical dominance: MRH likely picks adversary
```

**Defense**:
- **Diversity sampling**: Select from multiple alternatives, not just best
- **Witness endorsement**: Require peer recommendations for MRH candidates
- **Temporal diversity**: Rotate through alternatives over time
- **Anomaly detection**: Flag if MRH always selects from same group

**Proposed Mitigation**:
```python
def find_mrh_alternative_with_diversity(expert_id, context, all_experts):
    alternatives = []

    for other_expert in all_experts:
        overlap, shared = compute_context_overlap(expert_id, other_expert)
        if overlap >= overlap_threshold and context in shared:
            alt_trust = get_context_trust(other_expert, context)
            if alt_trust > get_context_trust(expert_id, context):
                alternatives.append((other_expert, alt_trust, overlap))

    if not alternatives:
        return None

    # Instead of always picking best, sample proportionally to trust
    # This ensures diversity and prevents monopoly
    trust_scores = [alt[1] for alt in alternatives]
    probabilities = softmax(trust_scores, temperature=0.5)

    selected_idx = np.random.choice(len(alternatives), p=probabilities)
    return alternatives[selected_idx]

def softmax(scores, temperature=1.0):
    exp_scores = np.exp(np.array(scores) / temperature)
    return exp_scores / exp_scores.sum()
```

---

### Category 3: Resource Exhaustion Attacks

#### Attack 3.1: Expert Starvation
**Description**: Adversary floods system with low-quality requests for specific expert, driving down its trust.

**Mechanism**:
1. Identify high-value specialist expert
2. Send adversarial inputs that cause poor outputs
3. Trust declines → expert gets substituted
4. Adversary controls substitutes

**Impact**: Medium (can degrade specific expert)

**Detection Difficulty**: Low (flooding detectable)

**Example Scenario**:
```python
# Target: Expert 42 (code specialist, trust=0.85)
for i in range(1000):
    adversarial_code_input = craft_pathological_case()
    result = expert_42.generate(adversarial_code_input)
    # Result is poor (adversarial input is hard)
    quality = evaluate(result)  # Low quality

    update_trust(expert_42, context_0, quality)
    # trust → 0.85 → 0.70 → 0.50 → 0.30 → 0.20

# Expert 42 now below threshold, gets substituted
```

**Defense**:
- **Rate limiting per expert**: Limit requests to any single expert
- **Input difficulty estimation**: Weight quality by input difficulty
- **Outlier detection**: Ignore extreme quality scores
- **Load balancing**: Distribute requests across similar experts

**Proposed Mitigation**:
```python
def update_trust_with_difficulty_weighting(expert_id, context, quality, input_difficulty):
    # Adjust quality by input difficulty
    # Hard inputs → lower weight on poor performance
    # Easy inputs → higher weight on poor performance

    difficulty_factor = estimate_input_difficulty(input)

    if difficulty_factor > 0.8 and quality < 0.3:
        # Very hard input, poor quality: expected
        quality = 0.5  # Neutral update

    elif difficulty_factor < 0.2 and quality < 0.3:
        # Easy input, poor quality: suspicious
        quality = quality * 1.5  # Penalize more

    update_trust(expert_id, context, quality)
```

---

#### Attack 3.2: Context Flooding
**Description**: Adversary creates fake contexts to dilute trust signals and fragment expert specialization.

**Mechanism**:
1. Flood system with inputs that create new, unique contexts
2. Specialists can't build trust in any single context
3. Trust signals become noisy → MRH less effective

**Impact**: Medium (degrades system overall, not specific expert)

**Detection Difficulty**: Low (unusual context distribution)

**Example Scenario**:
```python
# Normal system: 3-5 contexts
# Adversary creates: 100+ micro-contexts

for i in range(100):
    adversarial_input = craft_unique_distribution()
    context = context_classifier.classify(adversarial_input)
    # New context: context_73

# Now experts have trust in {context_0, context_1, ..., context_99}
# Each context has only 1-2 samples
# Trust signals too sparse to be useful
# MRH can't find meaningful overlap
```

**Defense**:
- **Context count limits**: Enforce maximum number of contexts
- **Context merging**: Automatically merge similar contexts
- **Minimum sample requirements**: Require N samples before trusting context
- **Anomaly detection**: Flag unusual context distributions

**Proposed Mitigation**:
```python
def classify_with_merging(embedding):
    context, confidence = context_classifier.classify(embedding)

    # Check if this context is well-established
    if get_context_sample_count(context) < MIN_SAMPLES:
        # New/rare context: try to merge with existing

        for existing_context in established_contexts:
            similarity = compute_context_similarity(context, existing_context)
            if similarity > 0.8:
                # Very similar: merge
                return existing_context

    # Check if total context count exceeds limit
    if len(all_contexts) > MAX_CONTEXTS:
        # Too many contexts: force merge least-used
        merge_similar_contexts()

    return context
```

---

### Category 4: Byzantine Attacks

#### Attack 4.1: Forged Trust Updates
**Description**: Compromised node forges trust updates to manipulate expert reputation.

**Mechanism**:
1. Compromise node with write access to trust database
2. Directly modify trust scores without actual quality measurement
3. Promote adversarial experts, demote legitimate ones

**Impact**: Critical (complete trust subversion)

**Detection Difficulty**: Medium (if no cryptographic verification)

**Example Scenario**:
```python
# Adversary compromises node
# Directly modifies trust database

trust_db[expert_42, context_0] = 0.1  # Was 0.85
trust_db[expert_200, context_0] = 0.95  # Was 0.3

# MRH now prefers adversary expert 200
# Legitimate expert 42 gets starved
```

**Defense**:
- **Cryptographic trust attestation**: All trust updates signed with LCT
- **Append-only trust log**: Trust history immutable
- **Byzantine fault tolerance**: Require 2f+1 nodes to agree on update
- **Merkle trees**: Trust updates form verifiable chain

**Proposed Mitigation**:
```python
def update_trust_with_attestation(expert_id, context, quality, attestation):
    # Verify attestation
    if not verify_lct_signature(attestation):
        raise SecurityError("Invalid trust update signature")

    # Verify attestation includes:
    # 1. Quality measurement
    # 2. Input hash
    # 3. Output hash
    # 4. Timestamp
    # 5. Witness signatures (BFT)

    if not verify_witness_consensus(attestation, min_witnesses=2*f+1):
        raise SecurityError("Insufficient witness consensus")

    # Append to immutable trust log
    trust_log.append({
        'expert_id': expert_id,
        'context': context,
        'quality': quality,
        'attestation': attestation,
        'merkle_root': compute_merkle_root(trust_log)
    })

    # Update trust score
    set_trust(expert_id, context, quality)
```

---

#### Attack 4.2: Network Partition (Eclipse)
**Description**: Adversary partitions network to prevent trust synchronization, creating inconsistent views.

**Mechanism**:
1. Adversary controls network routing
2. Partitions nodes into isolated groups
3. Each partition sees different trust scores
4. Adversary manipulates trust in each partition independently

**Impact**: Critical (breaks distributed trust consensus)

**Detection Difficulty**: Medium (partition detectable via heartbeat)

**Example Scenario**:
```python
# Network partitioned: {Node_A, Node_B} | {Node_C, Node_D}
# Adversary controls routing between partitions

# In partition 1:
update_trust(expert_42, context_0, 0.1)  # Low

# In partition 2:
update_trust(expert_42, context_0, 0.9)  # High

# Partitions merge: conflict!
# Which trust score is correct?
```

**Defense**:
- **Partition detection**: Heartbeat monitoring
- **Conflict resolution**: CRDT for trust merging
- **Gossip protocols**: Multi-path trust propagation
- **Byzantine agreement**: Require consensus on trust updates

**Proposed Mitigation**:
```python
def detect_partition():
    # Heartbeat with all known nodes
    reachable = []
    for node in all_nodes:
        if ping(node, timeout=1.0):
            reachable.append(node)

    # Check if reachable < expected
    if len(reachable) < len(all_nodes) * 0.5:
        # Potential partition
        return True

    return False

def merge_trust_on_partition_recovery(local_trust, remote_trust):
    # CRDT-style merge: keep trust with more evidence
    # Trust with more samples is preferred

    local_samples = len(get_trust_history(local_trust))
    remote_samples = len(get_trust_history(remote_trust))

    if local_samples > remote_samples * 1.5:
        return local_trust
    elif remote_samples > local_samples * 1.5:
        return remote_trust
    else:
        # Similar evidence: average
        return (local_trust + remote_trust) / 2
```

---

## Defense Architecture

### Layer 1: Input Validation
- **Context classification robustness**: Ensemble classifiers, confidence thresholds
- **Input sanitization**: Reject adversarial inputs
- **Rate limiting**: Prevent flooding

### Layer 2: Trust Update Verification
- **Cryptographic attestation**: LCT signatures on all updates
- **Witness consensus**: BFT-style agreement (2f+1)
- **Velocity limits**: Flag rapid trust changes
- **Difficulty weighting**: Adjust quality by input difficulty

### Layer 3: MRH Discovery Protection
- **Quality-weighted overlap**: Penalize similarity without performance match
- **Diversity sampling**: Don't always pick best alternative
- **Sybil resistance**: LCT identity verification, hardware binding
- **Behavioral fingerprinting**: Detect suspiciously similar experts

### Layer 4: Monitoring & Anomaly Detection
- **Trust velocity monitoring**: Alert on rapid changes
- **Context distribution analysis**: Detect unusual patterns
- **Network health**: Partition detection via heartbeat
- **Statistical outliers**: Flag extreme values

### Layer 5: Recovery & Resilience
- **Trust rollback**: Restore from checkpoints
- **Expert quarantine**: Isolate suspicious experts
- **Gradual re-entry**: Quarantined experts re-earn trust slowly
- **Incident logging**: Immutable audit trail

---

## Implementation Priorities

### Phase 1: Critical Defenses (Immediate)
1. **LCT attestation for trust updates** (Attack 4.1 defense)
2. **Sybil resistance via hardware binding** (Attack 1.2 defense)
3. **Rate limiting on trust updates** (Attack 3.1 defense)

### Phase 2: Core Protections (Short-term)
4. **Velocity limits on trust changes** (Attack 1.1 defense)
5. **Quality-weighted overlap computation** (Attack 2.1 defense)
6. **Diversity sampling in MRH** (Attack 2.2 defense)

### Phase 3: Advanced Defenses (Medium-term)
7. **Byzantine fault tolerance for trust consensus** (Attack 4.1, 4.2 defense)
8. **Context merging and limits** (Attack 3.2 defense)
9. **Behavioral fingerprinting** (Attack 1.2, 2.1 defense)

### Phase 4: Monitoring & Analytics (Ongoing)
10. **Anomaly detection dashboard**
11. **Statistical analysis of trust distributions**
12. **Automated incident response**

---

## Open Questions

1. **Trust velocity threshold**: What Δtrust/Δt indicates attack vs legitimate learning?
2. **Witness count**: How many witnesses needed for BFT (2f+1 where f=?)?
3. **Context overlap threshold**: Is 0.7 optimal, or should it adapt?
4. **Recovery time**: How long should quarantined experts take to re-earn trust?
5. **False positive rate**: What's acceptable for blocking legitimate experts?

---

## References

- **Session 65**: MRH substitution breakthrough, router monopoly breaking
- **Session 66**: MRH selector integration with SAGE
- **Web4 Standard**: LCT identity, trust tensors, authorization
- **Synchronism**: MRH theory, resonance patterns
- **Byzantine Fault Tolerance**: Castro & Liskov PBFT
- **Sybil Resistance**: Douceur "The Sybil Attack" (2002)

---

## Conclusion

The MRH trust-based expert selection system is powerful but introduces new attack surfaces. Most attacks target the **trust feedback loop** or **MRH discovery mechanism**. Defense requires:

1. **Cryptographic verification** (LCT attestation)
2. **Byzantine fault tolerance** (witness consensus)
3. **Anomaly detection** (statistical monitoring)
4. **Rate limiting** (prevent flooding)
5. **Diversity enforcement** (prevent monopoly)

With these defenses, the MRH system provides:
- **100% capacity utilization improvement** (4 → 8+ experts)
- **Specialist emergence** (60%+ specialists)
- **Resilience to attacks** (multiple defense layers)

Next: Implement Phase 1 defenses and validate with adversarial testing.

---

*"Security is not a feature, it's a continuous process of adversarial co-evolution."*
