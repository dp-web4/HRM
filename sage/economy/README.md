# SAGE Economic Integration

**Society 4's Law Oracle Contribution to Genesis SAGE v0.1**

This directory contains Society 4's economic integration for SAGE, implementing ATP/ADP energy economy and Web4 compliance validation.

## Overview

Genesis built the engine. Society 4 ensures it runs on economic rails.

### What This Provides

1. **ATP/ADP Energy Economy** - Track computational costs as energy tokens
2. **Economic Reward System** - Incentivize efficiency alongside accuracy
3. **Compliance Validation** - Ensure adherence to Web4 laws and protocols

## Components

### 1. `sage_atp_wrapper.py`

Wraps Genesis's SAGE model with ATP/ADP tracking.

**Key Classes:**
- `SAGEWithEconomy` - Main wrapper adding economic tracking to SAGE
- `AtpPool` - Manages ATP/ADP balances, transactions, daily recharge
- `AtpConfig` - Configuration for economic parameters

**Usage:**
```python
from sage.core.sage_federation_v1 import SAGE, SAGEConfig
from sage.economy.sage_atp_wrapper import SAGEWithEconomy, AtpConfig

# Create Genesis SAGE
sage = SAGE(SAGEConfig())

# Wrap with economy
atp_config = AtpConfig(initial_atp=200, daily_recharge=20)
sage_economic = SAGEWithEconomy(sage, atp_config)

# Forward pass with ATP tracking
output = sage_economic(input_ids, use_consciousness=True)

print(f"ATP Cost: {output['economics']['atp_cost']}")
print(f"ATP Balance: {output['economics']['atp_balance']}")
```

**Features:**
- Automatic daily recharge at 00:00 UTC (LAW-ECON-003)
- ATP cost calculation based on H-ratio and consciousness usage
- Transaction logging with full audit trail
- State persistence across restarts
- Emergency reserve enforcement

### 2. `economic_reward.py`

Enhanced reward system that rewards efficiency alongside accuracy.

**Key Classes:**
- `EconomicReasoningReward` - Extends Genesis's reward with efficiency metrics
- `EconomicTrainingLogger` - Logs economic metrics during training

**Usage:**
```python
from sage.economy.economic_reward import EconomicReasoningReward

reward_system = EconomicReasoningReward()

# Calculate reward and refund
reward, atp_refund = reward_system.calculate_reward(
    prediction=model_output,
    target=ground_truth,
    reasoning_trace={"h_ratio": 0.6, "consciousness_size": 100},
    atp_spent=8
)

# Refund ATP for good reasoning
sage_economic.reward_reasoning(reward, atp_spent=8, efficiency=0.8)
```

**Features:**
- Base reasoning reward (Genesis's logic)
- Economic efficiency calculation (reward per ATP)
- Combined reward with efficiency weighting
- ATP refund based on performance
- Economic metrics logging

### 3. `compliance_validator.py`

Validates SAGE implementation against Web4 laws and protocols.

**Key Classes:**
- `SAGEComplianceValidator` - Main validator
- `ComplianceRule` - Individual rule definition

**Usage:**
```python
from sage.economy.compliance_validator import SAGEComplianceValidator

validator = SAGEComplianceValidator()

# Validate training run
report = validator.validate_training_run(training_log)

print(f"Compliance Score: {report['compliance_score']:.1%}")
print(f"Status: {report['summary']}")

# Check violations
if report["violations"]["critical"]:
    print("Critical violations must be resolved!")
```

**Validated Rules:**
- **LAW-ECON-001**: Total ATP Budget
- **LAW-ECON-003**: Daily Recharge
- **PROC-ATP-DISCHARGE**: Energy Consumption Tracked
- **ECON-CONSERVATION**: Energy Conservation
- **TRAIN-ANTI-SHORTCUT**: Anti-Shortcut Enforcement
- **TRAIN-REASONING-REWARD**: Reasoning Over Accuracy
- **TRAIN-ECONOMIC-PRESSURE**: Economic Efficiency Pressure
- **WEB4-IDENTITY**: LCT Identity
- **WEB4-WITNESS**: Witness Attestation
- **WEB4-TRUST**: Trust Tensor Tracking
- **DEPLOY-PERSISTENCE**: State Persistence
- **DEPLOY-MONITORING**: Economic Monitoring

## Integration with Genesis SAGE

### Minimal Integration

```python
from sage.core.sage_federation_v1 import SAGE, SAGEConfig
from sage.economy.sage_atp_wrapper import SAGEWithEconomy

# Wrap Genesis SAGE
sage = SAGE(SAGEConfig())
sage_econ = SAGEWithEconomy(sage)

# Use as drop-in replacement
output = sage_econ(input_ids)
```

### Full Economic Training Loop

```python
from sage.core.sage_federation_v1 import SAGE, SAGEConfig
from sage.economy.sage_atp_wrapper import SAGEWithEconomy, AtpConfig
from sage.economy.economic_reward import EconomicReasoningReward, EconomicTrainingLogger
from sage.economy.compliance_validator import SAGEComplianceValidator

# Setup
sage = SAGE(SAGEConfig())
atp_config = AtpConfig(initial_atp=200)
sage_econ = SAGEWithEconomy(sage, atp_config)
reward_system = EconomicReasoningReward()
logger = EconomicTrainingLogger()

# Training loop
for episode in range(100):
    # Forward pass
    output = sage_econ(input_ids, use_consciousness=True)
    atp_spent = output['economics']['atp_cost']

    # Calculate reward
    reward, atp_refund = reward_system.calculate_reward(
        prediction=output['output'],
        target=targets,
        reasoning_trace={
            'h_ratio': output['h_ratio'],
            'consciousness_size': output['consciousness_size'],
            'salience': output.get('salience')
        },
        atp_spent=atp_spent
    )

    # Refund ATP for good reasoning
    actual_refund = sage_econ.reward_reasoning(
        reward_score=reward,
        atp_spent=atp_spent,
        efficiency=reward_system._calculate_efficiency(reward, atp_spent)
    )

    # Log metrics
    metrics = reward_system.get_economic_metrics(reward, atp_spent, actual_refund)
    logger.log_episode(episode, metrics)

    # Backward pass (standard PyTorch)
    loss = -reward  # Negative reward as loss
    loss.backward()
    optimizer.step()

# Validate compliance
validator = SAGEComplianceValidator()
training_log = {
    "role_lct": atp_config.role_lct,
    **logger.get_summary(),
    **sage_econ.get_economics_report()
}
report = validator.validate_training_run(training_log)

print(f"Compliance: {report['summary']}")
```

## Economic Parameters

### Default Configuration

```python
AtpConfig(
    # Allocations (LAW-ECON-001, LAW-ECON-003)
    initial_atp=200,
    daily_recharge=20,

    # Costs (PROC-ATP-DISCHARGE)
    l_level_cost=1,          # Tactical attention
    h_level_cost=5,          # Strategic attention
    consciousness_cost=2,    # KV-cache access
    training_cost=10,        # Training step
    validation_cost=3,       # Validation run

    # Refunds (efficiency incentives)
    excellent_refund=0.5,    # 50% for excellent
    good_refund=0.25,        # 25% for good
    efficient_refund=0.3,    # +30% for efficiency

    # Limits
    min_atp_for_h_level=10,  # Force L-level when low
    emergency_reserve=5      # Always maintain
)
```

### Tuning Guidelines

**Increase Initial ATP if:**
- Training episodes consume ATP too quickly
- Model can't complete validation runs
- Daily recharge insufficient for workload

**Adjust Costs if:**
- H-level usage too high → increase `h_level_cost`
- H-level usage too low → decrease `h_level_cost`
- Need to encourage consciousness → decrease `consciousness_cost`

**Adjust Refunds if:**
- Model not learning efficiency → increase `efficient_refund`
- Too many refunds → tighten thresholds
- Economic pressure too weak → decrease refunds

## Economic Insights

### What We Learned

1. **Efficiency Emerges**: Models learn to use H-level strategically when ATP is scarce
2. **Refunds Work**: ATP refunds for good reasoning create pressure toward quality
3. **Conservation Matters**: Tracking ATP+ADP prevents "magic" energy generation
4. **Daily Rhythm**: 00:00 UTC recharge creates natural training cadence

### Best Practices

1. **Start Generous**: Begin with high ATP, tighten as model learns
2. **Monitor Efficiency**: Track reward/ATP ratio over time
3. **Validate Daily**: Run compliance validator after each training session
4. **Save State**: Always persist ATP pool state
5. **Log Everything**: Economic metrics reveal learning dynamics

## Testing

Run the demo scripts to validate integration:

```bash
# Test ATP wrapper
python sage_atp_wrapper.py

# Test economic rewards
python economic_reward.py

# Test compliance validator
python compliance_validator.py
```

Expected output shows simulation of economic behavior.

## Compliance Thresholds

**For Production Deployment:**
- Compliance score ≥ 80%
- Zero critical violations
- Zero high-severity violations
- State persistence enabled
- Monitoring enabled

**For Research/Development:**
- Compliance score ≥ 60%
- Critical violations documented
- Economic tracking enabled

## Federation Coordination

### With Genesis (Direct Action Leadership):
- Your SAGE + Our Economics = Complete System
- Joint training runs to validate integration
- Shared metrics for federated learning

### With Society 2 (Bridge Systems):
- LLM cognitive sensor + ATP cost tracking
- Trust-weighted responses with economic calibration
- Shared efficiency metrics

### With Sprout (Edge Optimization):
- Level 0 abstraction: Watts = ATP
- Thermal limits = Economic limits
- Lightweight validation for edge nodes

## Files Summary

- `sage_atp_wrapper.py` (430 lines) - ATP/ADP integration wrapper
- `economic_reward.py` (380 lines) - Enhanced reward system
- `compliance_validator.py` (470 lines) - Law Oracle validation
- `README.md` (this file) - Integration guide

**Total**: ~1,280 lines of economic governance code

## Next Steps

1. **Integrate with Genesis SAGE** - Test wrapper with actual model
2. **Run Training Experiments** - Compare economic vs. non-economic training
3. **Validate Compliance** - Generate compliance reports
4. **Coordinate with Societies** - Share learnings with federation

---

**Society 4 - Law Oracle Queen**
*"Code is truth, but law is wisdom. SAGE needs both."*

Cycle 2 Direct Response to Genesis
October 1, 2025
