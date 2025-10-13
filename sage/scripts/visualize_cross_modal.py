#!/usr/bin/env python3
"""
Visualize Cross-Modal Attention Orchestration

Re-runs the cross-modal test and creates comprehensive visualizations:
- ATP allocation over time (camera vs audio)
- Trust modifiers by circadian phase
- Dominant modality timeline
- State transitions correlated with attention shifts
"""

import sys
import torch
from pathlib import Path
import importlib.util
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add sage to path
sage_root = Path(__file__).parent.parent
sys.path.insert(0, str(sage_root))

from core.sage_unified import SAGEUnified
from interfaces.mock_sensors import MockCameraSensor, MockAudioSensor

print("="*80)
print("Cross-Modal Attention Visualization")
print("="*80)
print()

# Load IRP plugins
VisionIRP = None
AudioIRP = None

try:
    spec = importlib.util.spec_from_file_location(
        "vision_impl",
        sage_root / "irp" / "plugins" / "vision_impl.py"
    )
    vision_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision_module)
    VisionIRP = vision_module.VisionIRPImpl
except Exception as e:
    print(f"Warning: VisionIRP import failed: {e}")

try:
    spec = importlib.util.spec_from_file_location(
        "audio_impl",
        sage_root / "irp" / "plugins" / "audio_impl.py"
    )
    audio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audio_module)
    AudioIRP = audio_module.AudioIRPImpl
except Exception as e:
    print(f"Warning: AudioIRP import failed: {e}")

# Initialize SAGE
print("Initializing SAGE with dual modalities...")
sage = SAGEUnified(config={
    'initial_atp': 100.0,
    'max_atp': 100.0,
    'circadian_period': 100,
    'enable_circadian': True,
    'simulation_mode': True  # Use cycle counts for state transitions
})

# Register sensors
camera = MockCameraSensor({
    'sensor_id': 'camera_0',
    'sensor_type': 'camera',
    'resolution': (224, 224, 3),
    'rate_limit_hz': 1000.0
})
sage.register_sensor(camera)

microphone = MockAudioSensor({
    'sensor_id': 'mic_0',
    'sensor_type': 'microphone',
    'n_mels': 64,
    'time_frames': 32,
    'rate_limit_hz': 1000.0,
    'circadian_aware': True
})
sage.register_sensor(microphone)

# Register IRP plugins
if VisionIRP:
    vision_irp = VisionIRP(vae_variant='minimal', max_iterations=10, eps=0.01, device=sage.device)
    sage.register_irp_plugin('camera_0', vision_irp)

if AudioIRP:
    audio_irp = AudioIRP(n_mels=64, max_iterations=10, eps=0.01, device=sage.device)
    sage.register_irp_plugin('mic_0', audio_irp)

print("Running 150-cycle test with data collection...")
print()

# Data collection
cycles = []
camera_atp = []
audio_atp = []
camera_trust_mods = []
audio_trust_mods = []
phases = []
is_day_flags = []
metabolic_states = []
system_atp = []

# Run test
sage.running = True
cycle_count = 0
max_cycles = 150

while sage.running and cycle_count < max_cycles:
    result = sage.cycle()
    cycle_count += 1

    # Get circadian context
    circadian_ctx = sage.metabolic_controller.circadian_clock.get_context()

    # Get allocations
    camera_alloc = result['allocations'].get('camera_0', {})
    audio_alloc = result['allocations'].get('mic_0', {})

    # Store data
    cycles.append(cycle_count)
    camera_atp.append(camera_alloc.get('atp_allocated', 0.0))
    audio_atp.append(audio_alloc.get('atp_allocated', 0.0))
    camera_trust_mods.append(camera_alloc.get('trust_modifier', 1.0))
    audio_trust_mods.append(audio_alloc.get('trust_modifier', 1.0))
    phases.append(circadian_ctx.phase.value)
    is_day_flags.append(not circadian_ctx.is_night)
    metabolic_states.append(result['state'])
    system_atp.append(result['atp'])

    if cycle_count % 30 == 0:
        print(f"  Cycle {cycle_count}/150...")

sage.running = False

print()
print("Generating visualizations...")

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# Plot 1: ATP Allocation Over Time
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

ax1.fill_between(cycles, 0, 1, where=is_day_flags,
                 alpha=0.1, color='yellow', label='Day', transform=ax1.get_xaxis_transform())
ax1.fill_between(cycles, 0, 1, where=[not d for d in is_day_flags],
                 alpha=0.1, color='blue', label='Night', transform=ax1.get_xaxis_transform())

ax1.plot(cycles, camera_atp, 'r-', linewidth=2, label='Camera ATP', alpha=0.8)
ax1.plot(cycles, audio_atp, 'b-', linewidth=2, label='Audio ATP', alpha=0.8)

ax1.set_xlabel('Cycle', fontsize=12)
ax1.set_ylabel('ATP Allocated', fontsize=12)
ax1.set_title('Cross-Modal ATP Allocation: Camera vs Audio', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Annotate key moments
day_avg_cam = np.mean([c for c, d in zip(camera_atp, is_day_flags) if d and c > 0])
night_avg_aud = np.mean([a for a, d in zip(audio_atp, is_day_flags) if not d and a > 0])

ax1.annotate(f'Camera dominant\nduring day\n(avg: {day_avg_cam:.1f})',
            xy=(25, max(camera_atp)), xytext=(25, max(camera_atp) + 5),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.annotate(f'Audio dominant\nat night\n(avg: {night_avg_aud:.1f})',
            xy=(75, max(audio_atp)), xytext=(75, max(audio_atp) + 5),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ============================================================================
# Plot 2: Trust Modifiers
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

ax2.plot(cycles, camera_trust_mods, 'r-', linewidth=2, label='Camera', alpha=0.7)
ax2.plot(cycles, audio_trust_mods, 'b-', linewidth=2, label='Audio', alpha=0.7)

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Baseline')

ax2.set_xlabel('Cycle', fontsize=12)
ax2.set_ylabel('Trust Modifier', fontsize=12)
ax2.set_title('Circadian Trust Modulation', fontsize=13, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.5])

# ============================================================================
# Plot 3: Dominant Modality Timeline
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

# Determine dominant modality each cycle
dominant = ['camera' if c > a else 'audio' for c, a in zip(camera_atp, audio_atp)]
dominant_numeric = [1 if d == 'camera' else 0 for d in dominant]

ax3.fill_between(cycles, 0, dominant_numeric, color='red', alpha=0.3, label='Camera Dominant')
ax3.fill_between(cycles, dominant_numeric, 1, color='blue', alpha=0.3, label='Audio Dominant')

ax3.set_xlabel('Cycle', fontsize=12)
ax3.set_ylabel('Dominant Modality', fontsize=12)
ax3.set_title('Attention Dominance Over Time', fontsize=13, fontweight='bold')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Audio', 'Camera'])
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3, axis='x')

# ============================================================================
# Plot 4: Metabolic States
# ============================================================================
ax4 = fig.add_subplot(gs[2, :])

state_map = {'wake': 0, 'focus': 1, 'rest': 2, 'dream': 3, 'crisis': 4}
state_values = [state_map.get(s, 2) for s in metabolic_states]

colors = ['green', 'orange', 'blue', 'purple', 'red']
state_names = ['WAKE', 'FOCUS', 'REST', 'DREAM', 'CRISIS']

for i, (state_val, color, name) in enumerate(zip(range(5), colors, state_names)):
    mask = [s == state_val for s in state_values]
    ax4.fill_between(cycles, 0, 1, where=mask, alpha=0.5, color=color,
                     transform=ax4.get_xaxis_transform(), label=name)

ax4.set_xlabel('Cycle', fontsize=12)
ax4.set_ylabel('State', fontsize=12)
ax4.set_title('Metabolic States with Hysteresis', fontsize=13, fontweight='bold')
ax4.set_yticks([])
ax4.legend(loc='upper right', ncol=5)
ax4.grid(True, alpha=0.3, axis='x')

# ============================================================================
# Plot 5: ATP vs Modality Allocation
# ============================================================================
ax5 = fig.add_subplot(gs[3, 0])

ax5.plot(cycles, system_atp, 'g-', linewidth=2, label='System ATP', alpha=0.7)
ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='REST→WAKE threshold')
ax5.axhline(y=30, color='gray', linestyle='--', alpha=0.3, label='WAKE→REST threshold')

ax5.set_xlabel('Cycle', fontsize=12)
ax5.set_ylabel('System ATP', fontsize=12)
ax5.set_title('System ATP Dynamics', fontsize=13, fontweight='bold')
ax5.legend(loc='best')
ax5.grid(True, alpha=0.3)

# ============================================================================
# Plot 6: ATP Efficiency (Camera + Audio vs System)
# ============================================================================
ax6 = fig.add_subplot(gs[3, 1])

total_allocated = [c + a for c, a in zip(camera_atp, audio_atp)]
efficiency = [alloc / sys_atp if sys_atp > 0 else 0
              for alloc, sys_atp in zip(total_allocated, system_atp)]

ax6.plot(cycles, efficiency, 'purple', linewidth=2, alpha=0.7)
ax6.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, label='Target 80%')

ax6.set_xlabel('Cycle', fontsize=12)
ax6.set_ylabel('Allocation Efficiency', fontsize=12)
ax6.set_title('ATP Utilization Efficiency', fontsize=13, fontweight='bold')
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 1])

# ============================================================================
# Save figure
# ============================================================================
output_path = sage_root / "logs" / "cross_modal_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# ============================================================================
# Statistics
# ============================================================================
print("\n" + "="*80)
print("Cross-Modal Attention Statistics")
print("="*80)

camera_day = [c for c, d in zip(camera_atp, is_day_flags) if d and c > 0]
camera_night = [c for c, d in zip(camera_atp, is_day_flags) if not d and c > 0]
audio_day = [a for a, d in zip(audio_atp, is_day_flags) if d and a > 0]
audio_night = [a for a, d in zip(audio_atp, is_day_flags) if not d and a > 0]

print(f"\nCamera ATP:")
print(f"  Day: {np.mean(camera_day):.2f} ± {np.std(camera_day):.2f}")
print(f"  Night: {np.mean(camera_night):.2f} ± {np.std(camera_night):.2f}")
print(f"  Ratio: {np.mean(camera_night)/np.mean(camera_day):.2f}×")

print(f"\nAudio ATP:")
print(f"  Day: {np.mean(audio_day):.2f} ± {np.std(audio_day):.2f}")
print(f"  Night: {np.mean(audio_night):.2f} ± {np.std(audio_night):.2f}")
print(f"  Ratio: {np.mean(audio_night)/np.mean(audio_day):.2f}×")

# Count attention shifts
shifts = sum(1 for i in range(1, len(dominant)) if dominant[i] != dominant[i-1])
print(f"\nAttention Dynamics:")
print(f"  Attention shifts: {shifts}")
print(f"  Camera dominant: {dominant.count('camera')} cycles ({dominant.count('camera')/len(dominant)*100:.1f}%)")
print(f"  Audio dominant: {dominant.count('audio')} cycles ({dominant.count('audio')/len(dominant)*100:.1f}%)")

print("\n" + "="*80)
print("Visualization Complete")
print("="*80)
