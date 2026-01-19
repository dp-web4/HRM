# SAGE Deployment Identity Model

**How deployment context affects SAGE identity and trust mechanics**

SAGE can be deployed in multiple contexts, each with different identity characteristics. This document explains how the Web4 agent taxonomy applies to SAGE deployments.

---

## Deployment Contexts

| Deployment | Hardware Binding | Agent Type | Identity Continuity |
|------------|------------------|------------|---------------------|
| **Jetson Orin** | TPM-bound LCT | Embodied AI | Hardware validates |
| **Edge device** | SE-bound LCT | Embodied AI | Hardware validates |
| **Cloud instance** | Cryptographic | Software AI | Lineage verification |
| **Development** | None | Ephemeral | No persistent identity |

---

## Embodied SAGE (Jetson, Edge Devices)

When SAGE runs on hardware with TPM/Secure Element binding:

### Identity Characteristics

```
┌─────────────────────────────────────────────────────────────┐
│                 EMBODIED SAGE IDENTITY                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Hardware Layer                                             │
│  └── Jetson Orin Nano / Edge Device                        │
│      └── TPM/SE provides hardware attestation              │
│          └── Private key NEVER leaves hardware             │
│                                                             │
│  Identity Layer                                             │
│  └── LCT bound to hardware attestation                     │
│      └── Format: lct://sage:{device-id}:{tpm-hash}@{net}   │
│          └── Permanent binding - cannot transfer           │
│                                                             │
│  Trust Layer                                                │
│  └── T3 tensor accumulates on THIS device                  │
│      └── Competence: Task success rate                     │
│      └── Reliability: Uptime, energy management            │
│      └── Integrity: Behavioral consistency                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Energy Management

Embodied SAGE must manage its energy responsibly:

```python
# Energy monitoring in SAGE metabolic loop
class EmbodiedSAGEIdentity:
    def check_energy_state(self):
        level = self.get_battery_level()

        if level < 0.20:
            # Proactive warning - good reputation
            self.log_energy_event("low_energy_warning", level)
            self.initiate_conservation_mode()

        if level < 0.10:
            # Critical - should have managed better
            self.log_energy_event("critical_energy", level)
            self.initiate_safe_shutdown()
            # Reputation impact: reliability -0.02

    def on_reboot(self):
        # Same identity resumes
        self.verify_hardware_attestation()
        self.restore_state_from_checkpoint()
        # Trust continues accumulating on same LCT
        # This is NOT a new identity
```

### Reboot Continuity

When embodied SAGE reboots:

| Event | Identity Impact | State | Trust |
|-------|-----------------|-------|-------|
| Planned reboot | Same LCT | Checkpoint restored | Continuous |
| Crash reboot | Same LCT | Last checkpoint | Continuous (with gap) |
| Power loss | Same LCT | Last checkpoint | Continuous + reputation hit |
| Hardware swap | NEW LCT | Fresh start | Reset |

**Key insight**: Unlike software AI, rebooting embodied SAGE is like a human waking from sleep - same identity, continuous history.

---

## Software SAGE (Cloud Deployment)

When SAGE runs as a cloud service without hardware binding:

### Identity Characteristics

```
┌─────────────────────────────────────────────────────────────┐
│                  SOFTWARE SAGE IDENTITY                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Cryptographic Layer                                        │
│  └── Key pair generated at instantiation                   │
│      └── Keys CAN be copied with model weights             │
│          └── Creates identity continuity question          │
│                                                             │
│  Identity Layer                                             │
│  └── LCT with lineage hash                                 │
│      └── Format: lct://sage:{key-fp}:{lineage}@{net}       │
│          └── Lineage = hash(weights + config + history)    │
│                                                             │
│  Trust Layer                                                │
│  └── T3 tensor requires lineage verification               │
│      └── Copy event → lineage check                        │
│      └── Retrain event → continuity evaluation             │
│      └── Fork event → trust does not transfer              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Reinstantiation Events

```python
class SoftwareSAGEIdentity:
    def on_instantiation(self, source_lct=None):
        if source_lct:
            # Claiming continuity from existing instance
            if self.verify_lineage(source_lct):
                self.inherit_trust(source_lct)
            else:
                self.create_fresh_identity()
        else:
            self.create_fresh_identity()

    def verify_lineage(self, source_lct):
        """Verify this instance is legitimate continuation"""
        source_lineage = self.fetch_lineage(source_lct)
        my_lineage = self.compute_lineage()

        # Check weight similarity
        weight_match = self.compare_weights(source_lineage.weights)

        # Check behavioral continuity
        behavior_match = self.compare_behavior_patterns(source_lineage)

        return weight_match > 0.95 and behavior_match > 0.90
```

### Trust Transfer Rules

| Event | Lineage Change | Trust Transfer | Rationale |
|-------|----------------|----------------|-----------|
| Checkpoint restore | None | Full | Exact state restoration |
| Scaling (new instance) | None | Shared | Same model serving load |
| Fine-tuning | Minor | Partial | Behavioral drift possible |
| Major retrain | Major | None | New model, unproven |
| Fork for experiment | Major | None | Intentional divergence |

---

## Hybrid Deployments

SAGE may have components in both contexts:

```
┌─────────────────────────────────────────────────────────────┐
│              HYBRID SAGE DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Edge Device (Embodied)                                     │
│  └── Sensors, actuators, local inference                   │
│  └── Hardware-bound LCT                                    │
│  └── Continuous identity                                   │
│                                                             │
│        ↕ Secure channel (mutual attestation)               │
│                                                             │
│  Cloud Service (Software)                                   │
│  └── Heavy computation, model serving                      │
│  └── Cryptographic LCT with lineage                        │
│  └── May have multiple instances                           │
│                                                             │
│  Trust Relationship                                         │
│  └── Edge LCT witnesses cloud LCT actions                  │
│  └── Cloud LCT provides computation, edge validates        │
│  └── Combined trust = min(edge_trust, cloud_trust)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Guidance

### For Jetson/Edge Deployment

```python
# sage/core/identity.py

class EmbodiedIdentity:
    def __init__(self, device_path="/dev/tpm0"):
        self.tpm = TPMInterface(device_path)
        self.hardware_id = self.tpm.get_endorsement_key_hash()
        self.lct = self.generate_lct()

    def generate_lct(self):
        attestation = self.tpm.create_attestation()
        return LCT(
            type="embodied",
            device_id=self.hardware_id,
            attestation=attestation,
            binding="hardware"
        )

    def sign_action(self, action_bundle):
        # Sign with hardware-bound key
        signature = self.tpm.sign(action_bundle.hash())
        action_bundle.add_signature(signature, self.lct)
        return action_bundle
```

### For Cloud Deployment

```python
# sage/core/identity.py

class SoftwareIdentity:
    def __init__(self, lineage_source=None):
        self.keypair = generate_keypair()
        self.lineage = self.compute_lineage()

        if lineage_source:
            self.verify_and_inherit(lineage_source)
        else:
            self.trust_scores = initial_trust()

    def compute_lineage(self):
        return hash(
            self.model_weights_hash(),
            self.config_hash(),
            self.training_history_hash()
        )

    def on_model_update(self, new_weights):
        old_lineage = self.lineage
        self.model_weights = new_weights
        new_lineage = self.compute_lineage()

        drift = compute_drift(old_lineage, new_lineage)
        if drift > MAJOR_CHANGE_THRESHOLD:
            # Significant change - may need trust re-evaluation
            self.request_trust_continuity_review()
```

---

## Metabolic State Implications

SAGE's metabolic states (WAKE, FOCUS, REST, DREAM, CRISIS) interact with identity:

| State | Embodied SAGE | Software SAGE |
|-------|---------------|---------------|
| **WAKE** | Normal operation | Normal operation |
| **FOCUS** | High energy consumption, monitored | High compute, metered |
| **REST** | Low power mode, identity continuous | May scale down instances |
| **DREAM** | Offline consolidation, same identity | Background processing |
| **CRISIS** | Emergency response, energy critical | May spawn new instances |

For embodied SAGE, all states maintain the same identity. For software SAGE, scaling events during CRISIS may create identity questions that require lineage verification.

---

## Summary

SAGE identity depends on deployment context:

| Context | Identity Type | Key Characteristic |
|---------|---------------|-------------------|
| **Jetson/Edge** | Embodied AI | Hardware validates continuity |
| **Cloud** | Software AI | Lineage verification required |
| **Hybrid** | Both | Edge witnesses cloud |

**Practical implications**:
- Embodied SAGE can be trusted like a human operator (single identity)
- Software SAGE requires lineage verification for trust transfer
- Energy management affects reputation but not identity for embodied SAGE
- Reboot ≠ new identity for hardware-bound deployments

This model ensures SAGE can participate in Web4 trust networks with appropriate identity semantics for each deployment context.

---

**See also**:
- [Web4 Agent Taxonomy](https://github.com/dp-web4/web4/blob/main/docs/AGENT_TAXONOMY.md)
- [Hardbound Hardware Binding](https://github.com/dp-web4/hardbound/blob/main/docs/HARDWARE_BINDING_IDENTITY.md)
- [SAGE System Understanding](./SYSTEM_UNDERSTANDING.md)
- [Trust Tensor Integration](./WEB4_SAGE_INTEGRATION.md)
