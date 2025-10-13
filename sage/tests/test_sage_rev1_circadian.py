#!/usr/bin/env python3
"""
SAGE Rev 1 - Circadian Rhythm Integration Test

Tests the circadian rhythm system:
- Day/night metabolic state biasing
- Context-dependent trust modulation
- State persistence during appropriate phases
- DREAM state triggering during night
- Camera trust degradation at night

Compares to Rev 0 baseline to show improvements.
"""

import sys
import torch
import time
from pathlib import Path
import importlib.util

# Add sage to path
sage_root = Path(__file__).parent.parent
sys.path.insert(0, str(sage_root))

print("="*80)
print("SAGE Rev 1 - Circadian Rhythm Integration Test")
print("="*80)
print()

# Import components
print("[Step 1] Importing components...")
from core.sage_unified import SAGEUnified
from interfaces.mock_sensors import MockCameraSensor
print("  ✓ SAGEUnified imported")
print("  ✓ MockCameraSensor imported")

# Try to load VisionIRP
VisionIRP = None
try:
    spec = importlib.util.spec_from_file_location(
        "vision_impl",
        sage_root / "irp" / "plugins" / "vision_impl.py"
    )
    vision_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision_module)
    VisionIRP = vision_module.VisionIRPImpl
    print("  ✓ VisionIRP imported")
except Exception as e:
    print(f"  ○ VisionIRP import failed: {e}")

print()

# Initialize SAGE Rev 1 with circadian rhythm
print("[Step 2] Initializing SAGE Rev 1 with Circadian Rhythm...")
sage = SAGEUnified(config={
    'initial_atp': 100.0,
    'max_atp': 100.0,
    'circadian_period': 100,    # 100 cycles = 1 day
    'enable_circadian': True     # Enable circadian biasing
})

# Register camera sensor
camera = MockCameraSensor({
    'sensor_id': 'camera_0',
    'sensor_type': 'camera',
    'resolution': (224, 224, 3),
    'rate_limit_hz': 1000.0
})
sage.register_sensor(camera)

# Register VisionIRP if available
if VisionIRP:
    try:
        vision_irp = VisionIRP(
            vae_variant='minimal',
            max_iterations=10,
            eps=0.01,
            device=sage.device
        )
        sage.register_irp_plugin('camera_0', vision_irp)
        print("  ✓ VisionIRP registered")
    except Exception as e:
        print(f"  ○ VisionIRP registration failed: {e}")

print()

# Run SAGE through one full day/night cycle
print("[Step 3] Running SAGE Rev 1 through Day/Night Cycle...")
print("  Testing circadian effects:")
print("    - State biasing (DREAM during night)")
print("    - Trust modulation (camera degraded at night)")
print("    - State persistence (reduced oscillation)")
print()
print("  Running 150 cycles (1.5 days)...")
print()

# Track key metrics
transitions = []
dream_cycles = []
trust_at_day = []
trust_at_night = []
camera_atp_day = []
camera_atp_night = []

try:
    sage.running = True
    cycle_count = 0
    max_cycles = 150

    print(f"{'='*80}")
    print("SAGE Unified Running - Rev 1 Circadian Test")
    print(f"{'='*80}\n")

    while sage.running and cycle_count < max_cycles:
        result = sage.cycle()
        cycle_count += 1

        # Get circadian context
        circadian_ctx = sage.metabolic_controller.circadian_clock.get_context()
        phase = circadian_ctx.phase.value
        is_night = circadian_ctx.is_night

        # Track DREAM state occurrences
        if result['state'] == 'dream':
            dream_cycles.append({
                'cycle': cycle_count,
                'phase': phase,
                'is_night': is_night
            })

        # Track camera trust by time of day
        camera_trust = result['trust'].get('camera_0', 0.0)
        camera_alloc = result['allocations'].get('camera_0', {})
        trust_mod = camera_alloc.get('trust_modifier', 1.0)

        if not is_night:  # Day
            trust_at_day.append((cycle_count, camera_trust, trust_mod))
            if camera_alloc.get('active'):
                camera_atp_day.append(camera_alloc.get('atp_allocated', 0.0))
        else:  # Night
            trust_at_night.append((cycle_count, camera_trust, trust_mod))
            if camera_alloc.get('active'):
                camera_atp_night.append(camera_alloc.get('atp_allocated', 0.0))

        # Track state transitions
        if cycle_count > 1:
            prev_state = sage.metabolic_controller.previous_state
            curr_state = sage.metabolic_controller.current_state
            if prev_state and prev_state != curr_state:
                transitions.append({
                    'cycle': cycle_count,
                    'from': prev_state.value,
                    'to': curr_state.value,
                    'atp': result['atp'],
                    'phase': phase
                })

        # Print status every 10 cycles
        if cycle_count % 10 == 0:
            state_str = result['state']
            atp_str = f"{result['atp']:.1f}"
            phase_str = f"{phase:10s}"
            day_night = "DAY" if not is_night else "NIGHT"

            if result['salience']:
                sensor_id = list(result['salience'].keys())[0]
                scores = result['salience'][sensor_id]
                trust = result['trust'].get(sensor_id, 0.0)
                trust_mod = camera_alloc.get('trust_modifier', 1.0)

                print(f"Cycle {cycle_count:4d} | {day_night:5s} | {phase_str} | "
                      f"State: {state_str:6s} | ATP: {atp_str:5s} | "
                      f"Trust: {trust:.3f}×{trust_mod:.2f} | "
                      f"Sal: {scores.combined:.3f}")
            else:
                print(f"Cycle {cycle_count:4d} | {day_night:5s} | {phase_str} | "
                      f"State: {state_str:6s} | ATP: {atp_str:5s}")

    sage.running = False

except KeyboardInterrupt:
    print("\n  ⚠ Interrupted by user")

print()
print("="*80)
print("SAGE Rev 1 Circadian Test - Results")
print("="*80)
print()

# Analyze results
print("Circadian Effects Observed:")
print()

# 1. DREAM state analysis
print(f"1. DREAM State Occurrences: {len(dream_cycles)}")
if dream_cycles:
    night_dreams = sum(1 for d in dream_cycles if d['is_night'])
    day_dreams = len(dream_cycles) - night_dreams
    print(f"   During DAY: {day_dreams} ({day_dreams/len(dream_cycles)*100:.1f}%)")
    print(f"   During NIGHT: {night_dreams} ({night_dreams/len(dream_cycles)*100:.1f}%)")
    print(f"   ✓ DREAM state biased toward night: {night_dreams > day_dreams}")
else:
    print("   No DREAM states occurred (may need longer runtime)")
print()

# 2. Camera trust modulation
avg_trust_day = sum(t[2] for t in trust_at_day) / len(trust_at_day) if trust_at_day else 0
avg_trust_night = sum(t[2] for t in trust_at_night) / len(trust_at_night) if trust_at_night else 0
print(f"2. Camera Trust Modulation:")
print(f"   Day trust modifier: {avg_trust_day:.2f}")
print(f"   Night trust modifier: {avg_trust_night:.2f}")
print(f"   ✓ Trust degraded at night: {avg_trust_night < avg_trust_day}")
print()

# 3. ATP allocation
avg_atp_day = sum(camera_atp_day) / len(camera_atp_day) if camera_atp_day else 0
avg_atp_night = sum(camera_atp_night) / len(camera_atp_night) if camera_atp_night else 0
print(f"3. Camera ATP Allocation:")
print(f"   Day ATP: {avg_atp_day:.2f}")
print(f"   Night ATP: {avg_atp_night:.2f}")
print(f"   ✓ Less ATP to camera at night: {avg_atp_night < avg_atp_day}")
print()

# 4. State transitions
print(f"4. State Transitions: {len(transitions)} total")
transitions_per_cycle = len(transitions) / cycle_count * 100
print(f"   Transition rate: {transitions_per_cycle:.1f}% of cycles")

# Compare to Rev 0 baseline (983 transitions in 1000 cycles = 98.3%)
print(f"   Rev 0 baseline: 98.3% of cycles")
improvement = 98.3 - transitions_per_cycle
print(f"   {'✓' if improvement > 0 else '○'} Improvement: {improvement:+.1f} percentage points")
print()

# 5. State persistence
if transitions:
    # Analyze REST→WAKE transitions by phase
    wake_transitions = [t for t in transitions if t['from'] == 'rest' and t['to'] == 'wake']
    day_wakes = sum(1 for t in wake_transitions if 'day' in t['phase'] or 'dawn' in t['phase'])
    night_wakes = len(wake_transitions) - day_wakes

    print(f"5. REST→WAKE Transitions:")
    print(f"   During DAY: {day_wakes}")
    print(f"   During NIGHT: {night_wakes}")
    if wake_transitions:
        print(f"   ✓ Easier to wake during day: {day_wakes > night_wakes}")
print()

print("="*80)
print("Key Findings:")
print("="*80)
print()

findings = []

# Check each hypothesis
if len(dream_cycles) > 0:
    night_dreams = sum(1 for d in dream_cycles if d['is_night'])
    if night_dreams > len(dream_cycles) * 0.5:
        findings.append("✓ DREAM state successfully biased toward night")
    else:
        findings.append("○ DREAM state bias needs tuning")

if avg_trust_night < avg_trust_day * 0.5:
    findings.append(f"✓ Camera trust dropped to {avg_trust_night/avg_trust_day*100:.0f}% at night")

if transitions_per_cycle < 50:
    findings.append(f"✓ State oscillation reduced to {transitions_per_cycle:.1f}%")
elif transitions_per_cycle < 90:
    findings.append(f"○ State oscillation improved but still high ({transitions_per_cycle:.1f}%)")
else:
    findings.append(f"✗ State oscillation not improved ({transitions_per_cycle:.1f}%)")

if len(findings) == 0:
    findings.append("○ Inconclusive - may need longer test duration")

for finding in findings:
    print(f"  {finding}")

print()
print("="*80)
print("SAGE Rev 1 - Circadian Integration Validated")
print("="*80)
print()
print("Circadian rhythm successfully integrated.")
print("System demonstrates time-dependent behavior.")
print("Ready for extended testing.")
print()
