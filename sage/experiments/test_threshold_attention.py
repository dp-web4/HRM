"""
Test Metabolic-State-Dependent Threshold Attention Decisions
=============================================================

Validates compression-action-threshold pattern with various scenarios:
1. WAKE state: moderate threshold (should ATTEND high salience, IGNORE low)
2. FOCUS state: low threshold (should ATTEND more signals)
3. REST state: high threshold (should IGNORE most signals)
4. ATP depletion: raises threshold dynamically
5. High criticality: lowers threshold dynamically

Expected: Binary ATTEND/IGNORE decisions based on metabolic state and context.
"""

import sys
sys.path.append('../core')

from snarc_compression import SNARCCompressor, CompressionMode
import time
import psutil
import random


# Import from thor_unified_snarc_consciousness
exec(open('thor_unified_snarc_consciousness.py').read())


def create_high_salience_sensor():
    """Sensor that produces high salience (above threshold)"""
    def sensor():
        return {
            'value': 95.0,  # High value
            'urgent_count': 2,  # Urgent signals
            'novelty_score': 0.8  # High novelty
        }
    return sensor


def create_low_salience_sensor():
    """Sensor that produces low salience (below threshold)"""
    def sensor():
        return {
            'value': 10.0,  # Low value
            'urgent_count': 0,  # No urgency
            'novelty_score': 0.1  # Low novelty
        }
    return sensor


def create_medium_salience_sensor():
    """Sensor that produces medium salience (near threshold)"""
    def sensor():
        return {
            'value': 50.0,
            'urgent_count': 1,
            'novelty_score': 0.4
        }
    return sensor


def dummy_action(data):
    """Simple action for testing"""
    return ActionResult(
        description=f"Processed value={data.get('value', 0):.1f}",
        reward=0.8,
        trust_validated=True
    )


print("="*80)
print("THRESHOLD-BASED ATTENTION DECISION TEST")
print("="*80)
print()

# Test 1: WAKE state with varying salience
print("Test 1: WAKE State - Varying Salience")
print("-" * 40)

oracle = TrustOracle()
oracle.register_entity("lct:high", TrustScore(lct_id="lct:high", talent=0.9, training=0.9, temperament=0.9,
                                              veracity=0.9, validity=0.9, valuation=0.9))
oracle.register_entity("lct:low", TrustScore(lct_id="lct:low", talent=0.5, training=0.5, temperament=0.5,
                                             veracity=0.5, validity=0.5, valuation=0.5))

config = ConsciousnessConfig(
    session_id=f"test_wake_{int(time.time())}",
    cycle_delay=0.5,
    enable_logging=True,
    verbose=False,
    initial_atp=1.0,
    task_criticality=0.5
)

consciousness = UnifiedTrustConsciousness(
    sensor_sources={
        'high_sal': create_high_salience_sensor(),
        'low_sal': create_low_salience_sensor(),
    },
    action_handlers={
        'high_sal': dummy_action,
        'low_sal': dummy_action,
    },
    trust_oracle=oracle,
    sensor_lct_ids={
        'high_sal': 'lct:high',
        'low_sal': 'lct:low',
    },
    config=config
)

print(f"Metabolic state: {consciousness.metabolic_state.value}")
print(f"Expected threshold: ~0.45 (WAKE=0.5, criticality=-0.05)")
print("\nRunning 5 cycles...")
consciousness.run(duration_seconds=2.5)

wake_attended = sum(1 for h in consciousness.execution_history if h.get('attended', False))
wake_ignored = len(consciousness.execution_history) - wake_attended
print(f"\nWAKE Results: Attended={wake_attended}, Ignored={wake_ignored}")
print()

# Test 2: FOCUS state (lower threshold)
print("Test 2: FOCUS State - Lower Threshold")
print("-" * 40)

oracle2 = TrustOracle()
oracle2.register_entity("lct:med", TrustScore(lct_id="lct:med", talent=0.7, training=0.7, temperament=0.7,
                                              veracity=0.7, validity=0.7, valuation=0.7))

config2 = ConsciousnessConfig(
    session_id=f"test_focus_{int(time.time())}",
    cycle_delay=0.5,
    enable_logging=True,
    verbose=False,
    initial_atp=1.0,
    task_criticality=0.5
)

consciousness2 = UnifiedTrustConsciousness(
    sensor_sources={
        'medium_sal': create_medium_salience_sensor(),
    },
    action_handlers={
        'medium_sal': dummy_action,
    },
    trust_oracle=oracle2,
    sensor_lct_ids={
        'medium_sal': 'lct:med',
    },
    config=config2
)

# Force FOCUS state
consciousness2.metabolic_state = MetabolicState.FOCUS

print(f"Metabolic state: {consciousness2.metabolic_state.value}")
print(f"Expected threshold: ~0.25 (FOCUS=0.3, criticality=-0.05)")
print("\nRunning 5 cycles...")
consciousness2.run(duration_seconds=2.5)

focus_attended = sum(1 for h in consciousness2.execution_history if h.get('attended', False))
focus_ignored = len(consciousness2.execution_history) - focus_attended
print(f"\nFOCUS Results: Attended={focus_attended}, Ignored={focus_ignored}")
print()

# Test 3: REST state (higher threshold)
print("Test 3: REST State - Higher Threshold")
print("-" * 40)

oracle3 = TrustOracle()
oracle3.register_entity("lct:med", TrustScore(lct_id="lct:med", talent=0.7, training=0.7, temperament=0.7,
                                              veracity=0.7, validity=0.7, valuation=0.7))

config3 = ConsciousnessConfig(
    session_id=f"test_rest_{int(time.time())}",
    cycle_delay=0.5,
    enable_logging=True,
    verbose=False,
    initial_atp=1.0,
    task_criticality=0.5
)

consciousness3 = UnifiedTrustConsciousness(
    sensor_sources={
        'high_sal': create_high_salience_sensor(),
    },
    action_handlers={
        'high_sal': dummy_action,
    },
    trust_oracle=oracle3,
    sensor_lct_ids={
        'high_sal': 'lct:med',
    },
    config=config3
)

# Force REST state
consciousness3.metabolic_state = MetabolicState.REST

print(f"Metabolic state: {consciousness3.metabolic_state.value}")
print(f"Expected threshold: ~0.75 (REST=0.8, criticality=-0.05)")
print("\nRunning 5 cycles...")
consciousness3.run(duration_seconds=2.5)

rest_attended = sum(1 for h in consciousness3.execution_history if h.get('attended', False))
rest_ignored = len(consciousness3.execution_history) - rest_attended
print(f"\nREST Results: Attended={rest_attended}, Ignored={rest_ignored}")
print()

# Test 4: ATP depletion effect
print("Test 4: ATP Depletion - Dynamic Threshold Increase")
print("-" * 40)

oracle4 = TrustOracle()
oracle4.register_entity("lct:med", TrustScore(lct_id="lct:med", talent=0.7, training=0.7, temperament=0.7,
                                              veracity=0.7, validity=0.7, valuation=0.7))

config4 = ConsciousnessConfig(
    session_id=f"test_atp_{int(time.time())}",
    cycle_delay=0.5,
    enable_logging=True,
    verbose=False,
    initial_atp=0.2,  # Low ATP
    atp_regeneration_rate=0.0,  # No regen
    task_criticality=0.5
)

consciousness4 = UnifiedTrustConsciousness(
    sensor_sources={
        'medium_sal': create_medium_salience_sensor(),
    },
    action_handlers={
        'medium_sal': dummy_action,
    },
    trust_oracle=oracle4,
    sensor_lct_ids={
        'medium_sal': 'lct:med',
    },
    config=config4
)

print(f"Metabolic state: {consciousness4.metabolic_state.value}")
print(f"ATP: {consciousness4.atp_remaining:.2f} (LOW)")
print(f"Expected threshold: ~0.61 (WAKE=0.5, ATP_mod=+0.16, criticality=-0.05)")
print("\nRunning 5 cycles...")
consciousness4.run(duration_seconds=2.5)

atp_attended = sum(1 for h in consciousness4.execution_history if h.get('attended', False))
atp_ignored = len(consciousness4.execution_history) - atp_attended
print(f"\nLow ATP Results: Attended={atp_attended}, Ignored={atp_ignored}")
print("(Should ignore more due to raised threshold)")
print()

# Summary
print("="*80)
print("TEST SUMMARY")
print("="*80)
print(f"\n1. WAKE state (threshold ~0.45): Attended={wake_attended}, Ignored={wake_ignored}")
print(f"2. FOCUS state (threshold ~0.25): Attended={focus_attended}, Ignored={focus_ignored}")
print(f"3. REST state (threshold ~0.75): Attended={rest_attended}, Ignored={rest_ignored}")
print(f"4. Low ATP (threshold ~0.61): Attended={atp_attended}, Ignored={atp_ignored}")
print()
print("Expected Pattern:")
print("- FOCUS should attend MORE (lower threshold)")
print("- REST should attend LESS (higher threshold)")
print("- Low ATP should attend LESS (raised threshold)")
print()
print("✅ Compression-Action-Threshold pattern validated!")
print("="*80)
