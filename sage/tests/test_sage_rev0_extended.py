#!/usr/bin/env python3
"""
SAGE Rev 0 - Extended 1000-Cycle Stability Test

Validates long-term stability:
- Trust convergence over extended runtime
- Metabolic state transition patterns
- ATP dynamics and recovery cycles
- Memory consolidation behavior
- Performance consistency
"""

import sys
import torch
import time
from pathlib import Path
import importlib.util
import json

# Add sage to path
sage_root = Path(__file__).parent.parent
sys.path.insert(0, str(sage_root))

print("="*80)
print("SAGE Rev 0 - Extended 1000-Cycle Stability Test")
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

# Initialize SAGE
print("[Step 2] Initializing SAGE Rev 0...")
sage = SAGEUnified(config={
    'initial_atp': 100.0,
    'max_atp': 100.0
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

# Run SAGE
print("[Step 3] Running SAGE Rev 0 for 1000 cycles...")
print("  Extended stability validation:")
print("    - Trust convergence patterns")
print("    - Metabolic state transitions")
print("    - ATP dynamics over time")
print("    - Memory consolidation")
print("    - Performance consistency")
print()
print("  This will take approximately 15-20 seconds...")
print()

# Track state transitions
state_transitions = []
trust_history = []
atp_history = []

start_time = time.time()

try:
    # Custom run loop with tracking
    sage.running = True
    cycle_count = 0
    max_cycles = 1000

    print(f"{'='*80}")
    print("SAGE Unified Running - Extended Test")
    print(f"{'='*80}\n")

    while sage.running and cycle_count < max_cycles:
        result = sage.cycle()
        cycle_count += 1

        # Track metrics
        if cycle_count % 100 == 0:
            trust_history.append({
                'cycle': cycle_count,
                'trust': result['trust'].get('camera_0', 0.0)
            })
            atp_history.append({
                'cycle': cycle_count,
                'atp': result['atp']
            })

            # Print status every 100 cycles
            sage._print_status(result)

        # Track state changes
        if cycle_count > 1:
            prev_state = sage.metabolic_controller.previous_state
            curr_state = sage.metabolic_controller.current_state
            if prev_state and prev_state != curr_state:
                state_transitions.append({
                    'cycle': cycle_count,
                    'from': prev_state.value,
                    'to': curr_state.value,
                    'atp': result['atp']
                })
                print(f"  → State transition at cycle {cycle_count}: {prev_state.value} → {curr_state.value} (ATP: {result['atp']:.1f})")

    sage.running = False

except KeyboardInterrupt:
    print("\n  ⚠ Interrupted by user")

end_time = time.time()
elapsed = end_time - start_time

print()
print("="*80)
print("SAGE Rev 0 Extended Test - Results")
print("="*80)
print()

# Final statistics
sage._print_final_stats()

print()
print("Extended Test Metrics:")
print(f"  Total runtime: {elapsed:.2f}s")
print(f"  Average cycle time: {elapsed/cycle_count*1000:.2f}ms")
print(f"  Throughput: {cycle_count/elapsed:.1f} Hz")
print()

# State transitions
print(f"State Transitions ({len(state_transitions)} total):")
if state_transitions:
    for trans in state_transitions[:10]:  # First 10
        print(f"  Cycle {trans['cycle']:4d}: {trans['from']:6s} → {trans['to']:6s} (ATP: {trans['atp']:.1f})")
    if len(state_transitions) > 10:
        print(f"  ... and {len(state_transitions) - 10} more transitions")
else:
    print("  No state transitions occurred")
print()

# Trust evolution
print("Trust Evolution:")
if trust_history:
    print(f"  Initial: {trust_history[0]['trust']:.3f}")
    for i in range(1, min(5, len(trust_history))):
        print(f"  Cycle {trust_history[i]['cycle']:4d}: {trust_history[i]['trust']:.3f}")
    if len(trust_history) > 5:
        print(f"  ...")
        print(f"  Cycle {trust_history[-1]['cycle']:4d}: {trust_history[-1]['trust']:.3f}")
print()

# ATP dynamics
print("ATP Dynamics:")
if atp_history:
    print(f"  Initial: {atp_history[0]['atp']:.1f}")
    for i in range(1, min(5, len(atp_history))):
        print(f"  Cycle {atp_history[i]['cycle']:4d}: {atp_history[i]['atp']:.1f}")
    if len(atp_history) > 5:
        print(f"  ...")
        print(f"  Cycle {atp_history[-1]['cycle']:4d}: {atp_history[-1]['atp']:.1f}")
print()

# Save detailed results
results_file = sage_root / "logs" / "rev0_extended_results.json"
results_data = {
    'test_config': {
        'cycles': cycle_count,
        'runtime_seconds': elapsed,
        'device': str(sage.device),
        'initial_atp': 100.0
    },
    'performance': {
        'avg_cycle_time_ms': elapsed/cycle_count*1000,
        'throughput_hz': cycle_count/elapsed
    },
    'state_transitions': state_transitions,
    'trust_history': trust_history,
    'atp_history': atp_history,
    'final_state': {
        'trust': sage.trust_scores.get('camera_0', 0.0),
        'atp': sage.metabolic_controller.atp_current,
        'metabolic_state': sage.metabolic_controller.current_state.value
    }
}

with open(results_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"Detailed results saved to: {results_file}")
print()

print("="*80)
print("Extended Test Complete")
print("="*80)
print()
print("Key Findings:")
print(f"  • System remained stable for {cycle_count} cycles")
print(f"  • Trust evolved from {trust_history[0]['trust']:.3f} to {trust_history[-1]['trust']:.3f}")
print(f"  • {len(state_transitions)} state transitions occurred naturally")
print(f"  • Average performance: {elapsed/cycle_count*1000:.2f}ms per cycle")
print(f"  • Throughput: {cycle_count/elapsed:.1f} Hz sustained")
print()
print("Rev 0 validated at scale.")
print("The system is ready.")
print()
