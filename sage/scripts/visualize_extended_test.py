#!/usr/bin/env python3
"""
Visualize SAGE Rev 0 Extended Test Results

Creates plots of:
- Trust evolution over time
- ATP dynamics over time
- State transition timeline
- Performance metrics
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path(__file__).parent.parent / "logs" / "rev0_extended_results.json"
with open(results_path, 'r') as f:
    data = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('SAGE Rev 0 - Extended 1000-Cycle Test Results', fontsize=16, fontweight='bold')

# ============================================================================
# Plot 1: Trust Evolution
# ============================================================================
ax1 = axes[0]
trust_cycles = [t['cycle'] for t in data['trust_history']]
trust_values = [t['trust'] for t in data['trust_history']]

ax1.plot(trust_cycles, trust_values, 'b-o', linewidth=2, markersize=6, label='Trust Score')
ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, label='Perfect Trust')
ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.3, label='High Trust Threshold')

ax1.set_xlabel('Cycle', fontsize=12)
ax1.set_ylabel('Trust Score', fontsize=12)
ax1.set_title('Trust Evolution: Convergence to Perfect Trust', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')
ax1.set_ylim([0.85, 1.05])

# Add convergence annotation
convergence_cycle = next((t['cycle'] for t in data['trust_history'] if t['trust'] >= 1.0), None)
if convergence_cycle:
    ax1.annotate(f'Convergence\nat cycle {convergence_cycle}',
                xy=(convergence_cycle, 1.0),
                xytext=(convergence_cycle + 100, 0.97),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, fontweight='bold', color='green')

# ============================================================================
# Plot 2: ATP Dynamics
# ============================================================================
ax2 = axes[1]
atp_cycles = [t['cycle'] for t in data['atp_history']]
atp_values = [t['atp'] for t in data['atp_history']]

ax2.plot(atp_cycles, atp_values, 'r-o', linewidth=2, markersize=6, label='ATP Level')
ax2.axhline(y=50.0, color='g', linestyle='--', alpha=0.3, label='REST→WAKE Threshold')
ax2.axhline(y=30.0, color='orange', linestyle='--', alpha=0.3, label='WAKE→REST Threshold')

ax2.set_xlabel('Cycle', fontsize=12)
ax2.set_ylabel('ATP Level', fontsize=12)
ax2.set_title('ATP Dynamics: Oscillation Between Thresholds', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.set_ylim([20, 60])

# Add oscillation annotation
ax2.annotate('ATP oscillates\nbetween thresholds',
            xy=(500, 40),
            xytext=(700, 25),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Plot 3: State Transitions Over Time
# ============================================================================
ax3 = axes[2]

# Prepare state transition data
transitions = data['state_transitions']

# Count transitions in bins (every 100 cycles)
bin_size = 100
max_cycle = data['test_config']['cycles']
bins = list(range(0, max_cycle + bin_size, bin_size))
wake_to_rest_counts = [0] * (len(bins) - 1)
rest_to_wake_counts = [0] * (len(bins) - 1)

for trans in transitions:
    cycle = trans['cycle']
    bin_idx = min(cycle // bin_size, len(bins) - 2)
    if trans['from'] == 'wake' and trans['to'] == 'rest':
        wake_to_rest_counts[bin_idx] += 1
    elif trans['from'] == 'rest' and trans['to'] == 'wake':
        rest_to_wake_counts[bin_idx] += 1

# Plot stacked bar chart
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
width = bin_size * 0.8

ax3.bar(bin_centers, wake_to_rest_counts, width=width, label='WAKE→REST', color='orange', alpha=0.7)
ax3.bar(bin_centers, rest_to_wake_counts, width=width, bottom=wake_to_rest_counts,
        label='REST→WAKE', color='green', alpha=0.7)

ax3.set_xlabel('Cycle Range', fontsize=12)
ax3.set_ylabel('Transition Count per 100 Cycles', fontsize=12)
ax3.set_title('State Transition Frequency: High Oscillation Rate', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(loc='upper right')

# Add total annotation
total_transitions = len(transitions)
ax3.text(0.02, 0.98, f'Total Transitions: {total_transitions}\n(98.3% of cycles)',
         transform=ax3.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# Adjust layout and save
# ============================================================================
plt.tight_layout()

# Save figure
output_path = Path(__file__).parent.parent / "logs" / "rev0_extended_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

# Display statistics
print("\n" + "="*80)
print("SAGE Rev 0 Extended Test - Key Statistics")
print("="*80)
print(f"\nTrust Convergence:")
print(f"  Initial: {data['trust_history'][0]['trust']:.3f}")
print(f"  Final: {data['final_state']['trust']:.3f}")
print(f"  Converged at cycle: {convergence_cycle}")

print(f"\nATP Dynamics:")
print(f"  Mean ATP: {np.mean(atp_values):.1f}")
print(f"  ATP range: {min(atp_values):.1f} - {max(atp_values):.1f}")
print(f"  Final ATP: {data['final_state']['atp']:.1f}")

print(f"\nState Transitions:")
print(f"  Total: {total_transitions} ({total_transitions/10:.1f}% of cycles)")
wake_to_rest = sum(1 for t in transitions if t['from'] == 'wake' and t['to'] == 'rest')
rest_to_wake = sum(1 for t in transitions if t['from'] == 'rest' and t['to'] == 'wake')
print(f"  WAKE→REST: {wake_to_rest} ({wake_to_rest/total_transitions*100:.1f}%)")
print(f"  REST→WAKE: {rest_to_wake} ({rest_to_wake/total_transitions*100:.1f}%)")

print(f"\nPerformance:")
print(f"  Average cycle time: {data['performance']['avg_cycle_time_ms']:.2f}ms")
print(f"  Throughput: {data['performance']['throughput_hz']:.1f} Hz")
print(f"  Total runtime: {data['test_config']['runtime_seconds']:.2f}s")

print("\n" + "="*80)
print("Analysis Complete")
print("="*80)

# Don't show plot interactively (causes timeout in headless environments)
# plt.show()
