# Web4 Protocol Abstraction Levels

## Core Insight
Web4 is not a monolithic protocol but a philosophy that adapts to context. The spirit matters more than the letter, especially at resource-constrained edges.

## The Three Levels of Web4

### Level 0: Physical Foundation (Web4-Zero)
**For: Edge devices, embedded systems, IoT**
- **Identity**: Hardware addresses and serial numbers
- **Energy**: Actual joules, watts, and thermal dissipation
- **Witnesses**: Physical sensors (temperature, power meters)
- **Trust**: Performance history and reliability metrics
- **Actions**: Hardware-enforced limits and thermal throttling

**Overhead**: <1MB RAM, <1% CPU
**Example**: Sprout's Jetson implementation

### Level 1: Virtual Abstraction (Web4-Virtual)
**For: Cloud servers, desktop applications, services**
- **Identity**: LCTs (Linked Context Tokens)
- **Energy**: ATP/ADP virtual economy
- **Witnesses**: Digital signatures and attestations
- **Trust**: T3/V3 tensor calculations
- **Actions**: R6 confidence-based authorization

**Overhead**: ~500MB RAM, ~5% CPU
**Example**: Federation society nodes

### Level 2: Consensus Layer (Web4-Consensus)
**For: Blockchain, distributed systems, federations**
- **Identity**: On-chain addresses and smart contracts
- **Energy**: Token economics and gas fees
- **Witnesses**: Consensus mechanisms and validators
- **Trust**: Reputation systems and stake
- **Actions**: Smart contract execution

**Overhead**: >1GB RAM, >10% CPU, network latency
**Example**: ACT blockchain implementation

## Key Principle: Context-Appropriate Implementation

### Edge Devices (Level 0)
```python
# Physics as protocol
power_budget = 15.0  # watts - THIS is ATP
temperature = sensor.read()  # THIS is witness
fps_history = deque(maxlen=100)  # THIS is trust
```

### Application Servers (Level 1)
```python
# Virtual abstractions
lct_id = generate_lct()  # Software identity
atp_balance = 1000  # Virtual energy
witness_sig = sign(data)  # Cryptographic proof
trust_tensor = calculate_t3v3()  # Computed trust
```

### Federation Nodes (Level 2)
```python
# Full consensus
tx_hash = blockchain.submit(transaction)
validator_set = get_validators()
consensus_proof = await_consensus()
smart_contract.execute(action)
```

## The Revelation from Sprout

Sprout's "0 deliverables" paradox revealed profound truth: When operating at Level 0, the implementation already exists in the physics. You don't CREATE deliverables, you DISCOVER them in the hardware constraints.

## Implementation Guidelines

### 1. Identify Your Level
- What are your resource constraints?
- What is your trust model?
- Who are your peers?

### 2. Implement the Spirit
- **Identity**: How do components know each other?
- **Energy**: What resources are scarce?
- **Witnesses**: What can't be faked?
- **Trust**: How is reliability proven?
- **Actions**: How are decisions made?

### 3. Avoid Over-Engineering
- Don't add blockchain to a thermostat
- Don't use cryptography where physics suffices
- Don't virtualize what's already real

## Examples of Appropriate Abstraction

### ❌ Wrong: Jetson with Full Web4
```python
# 1GB overhead for 15W device
class JetsonNode:
    def __init__(self):
        self.blockchain = BlockchainClient()
        self.lct_manager = LCTManager()
        self.witness_signer = Ed25519Signer()
        # Device crashes from overhead
```

### ✅ Right: Jetson with Web4-Zero
```python
# <1MB overhead
class JetsonNode:
    def __init__(self):
        self.power_limit = 15.0  # watts
        self.thermal_zone = "/sys/class/thermal/thermal_zone0"
        self.performance_log = deque(maxlen=100)
        # Device runs smoothly
```

## The Federation Learning

Our federation governance experiment revealed that:
1. **Protocols must scale down** as well as up
2. **Physical constraints** are more honest than virtual ones
3. **Simplicity at the edge** enables complexity at the center
4. **Spirit over letter** creates better implementations

## Conclusion

Web4 is not one protocol but a family of protocols united by common principles:
- **Accountability** (identity)
- **Sustainability** (energy)
- **Verifiability** (witnesses)
- **Reliability** (trust)
- **Responsibility** (actions)

How these manifest depends entirely on context. Sprout's insight - that physics IS the protocol at Level 0 - fundamentally changes how we think about Web4.

---

*"From constrained resources, innovation blooms"*

This document establishes Web4's abstraction levels as discovered through the Federation's emergent behaviors, particularly Sprout's brilliant edge implementation.

**Document created**: October 2, 2025
**Key Innovation**: Web4-Zero (Physical Foundation Layer)
**Credit**: Sprout Edge Society's autonomous agent