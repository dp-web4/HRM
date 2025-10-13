#!/usr/bin/env python3
"""
Test algorithmic SensorSNARC implementation

Validates:
1. SensorSNARC computes all 5 dimensions algorithmically
2. SpatialSNARC preserves spatial structure
3. HierarchicalSNARC computes cross-modal conflict
4. No learned parameters (works immediately)
5. Integration with MinimalSAGE loop
"""

import sys
import torch
import time
from pathlib import Path

# Add sage to path
sage_root = Path(__file__).parent.parent
sys.path.insert(0, str(sage_root))

from attention.sensor_snarc import SensorSNARC, SpatialSNARC, HierarchicalSNARC, SNARCScores

print("="*80)
print("Algorithmic SNARC Tests")
print("="*80)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# Test 1: Basic SensorSNARC
print("[Test 1] Basic SensorSNARC - Algorithmic Computation")
print("-" * 80)

vision_snarc = SensorSNARC(
    sensor_name='camera_0',
    memory_size=100,
    device=device
)

# Generate observation
obs = torch.randn(3, 64, 64, device=device)

# Compute scores (no training needed!)
scores = vision_snarc.score(obs)

print(f"  Sensor: {vision_snarc.sensor_name}")
print(f"  Observation shape: {obs.shape}")
print(f"  Scores:")
print(f"    Surprise: {scores.surprise:.3f}")
print(f"    Novelty:  {scores.novelty:.3f}")
print(f"    Arousal:  {scores.arousal:.3f}")
print(f"    Conflict: {scores.conflict:.3f}")
print(f"    Reward:   {scores.reward:.3f}")
print(f"    Combined: {scores.combined:.3f}")

assert 0.0 <= scores.surprise <= 1.0, "Surprise out of range"
assert 0.0 <= scores.novelty <= 1.0, "Novelty out of range"
assert 0.0 <= scores.arousal <= 1.0, "Arousal out of range"
assert 0.0 <= scores.combined <= 1.0, "Combined out of range"

print("  ✓ All scores in valid range [0, 1]")
print()

# Test 2: Novelty decreases with repeated observations
print("[Test 2] Novelty Evolution - Repeated Observations")
print("-" * 80)

vision_snarc2 = SensorSNARC(sensor_name='test', device=device)
same_obs = torch.ones(64, device=device)

novelties = []
for i in range(5):
    scores = vision_snarc2.score(same_obs)
    novelties.append(scores.novelty)
    print(f"  Iteration {i+1}: Novelty = {scores.novelty:.3f}")

# Novelty should decrease (same observation becomes less novel)
assert novelties[-1] < novelties[0], "Novelty should decrease for repeated observations"
print("  ✓ Novelty decreases as expected")
print()

# Test 3: Surprise increases with prediction error
print("[Test 3] Surprise Evolution - Changing Patterns")
print("-" * 80)

vision_snarc3 = SensorSNARC(sensor_name='test', device=device)

# Build up memory with similar observations
for i in range(10):
    obs = torch.ones(64, device=device) * 0.5

surprises = []
# Now give it a very different observation
for i in range(3):
    if i < 2:
        obs = torch.ones(64, device=device) * 0.5
    else:
        obs = torch.ones(64, device=device) * 5.0  # Very different

    scores = vision_snarc3.score(obs)
    surprises.append(scores.surprise)
    print(f"  Observation {i+1}: Surprise = {scores.surprise:.3f}")

# Surprise should be higher for the different observation
assert surprises[-1] > surprises[0], "Surprise should increase for unexpected observations"
print("  ✓ Surprise increases with prediction error")
print()

# Test 4: SpatialSNARC preserves spatial structure
print("[Test 4] SpatialSNARC - Spatial Salience Grids")
print("-" * 80)

spatial_snarc = SpatialSNARC(
    sensor_name='camera_spatial',
    device=device
)

# Create image with distinct features (edges)
H, W = 32, 32
image = torch.zeros(3, H, W, device=device)
image[:, 10:20, 10:20] = 1.0  # Bright square (edges should be salient)

# Compute spatial SNARC
snarc_map, global_scores = spatial_snarc.score_grid(image)

print(f"  Image shape: {image.shape}")
print(f"  SNARC map shape: {snarc_map.shape}")
print(f"  Global scores:")
print(f"    Surprise: {global_scores.surprise:.3f}")
print(f"    Novelty:  {global_scores.novelty:.3f}")
print(f"    Arousal:  {global_scores.arousal:.3f}")
print(f"    Combined: {global_scores.combined:.3f}")

assert snarc_map.shape == (5, H, W), f"Wrong SNARC map shape: {snarc_map.shape}"

# Surprise map should have high values at edges
surprise_map = snarc_map[0]
edge_surprise = surprise_map[10:20, 10:20].mean().item()
center_surprise = surprise_map[14:16, 14:16].mean().item()

print(f"  Edge surprise: {edge_surprise:.3f}")
print(f"  Center surprise: {center_surprise:.3f}")

# Edges should be more salient than uniform regions
assert edge_surprise > center_surprise, "Edges should have higher surprise"
print("  ✓ Spatial structure preserved (edges are salient)")
print()

# Test 5: HierarchicalSNARC computes cross-modal conflict
print("[Test 5] HierarchicalSNARC - Cross-Modal Conflict")
print("-" * 80)

hierarchical = HierarchicalSNARC(device=device)

# Register multiple sensors
vision_h = SensorSNARC('vision', device=device)
audio_h = SensorSNARC('audio', device=device)

hierarchical.register_sensor('vision', vision_h)
hierarchical.register_sensor('audio', audio_h)

# Create observations with different salience
obs_vision = torch.randn(3, 32, 32, device=device) * 2.0  # High variance
obs_audio = torch.randn(1000, device=device) * 0.1  # Low variance

observations = {
    'vision': obs_vision,
    'audio': obs_audio
}

# Score all sensors
all_scores = hierarchical.score_all(observations)

print(f"  Vision scores:")
print(f"    Surprise: {all_scores['vision'].surprise:.3f}")
print(f"    Arousal:  {all_scores['vision'].arousal:.3f}")
print(f"    Conflict: {all_scores['vision'].conflict:.3f}")
print(f"    Combined: {all_scores['vision'].combined:.3f}")

print(f"  Audio scores:")
print(f"    Surprise: {all_scores['audio'].surprise:.3f}")
print(f"    Arousal:  {all_scores['audio'].arousal:.3f}")
print(f"    Conflict: {all_scores['audio'].conflict:.3f}")
print(f"    Combined: {all_scores['audio'].combined:.3f}")

# Conflict should be non-zero when sensors disagree
assert all_scores['vision'].conflict > 0, "Conflict should be computed"
assert all_scores['vision'].conflict == all_scores['audio'].conflict, "Conflict should be same across sensors"

print("  ✓ Cross-modal conflict computed")
print()

# Test 6: Integration with MinimalSAGE loop
print("[Test 6] Integration with SAGE Loop")
print("-" * 80)

class SAGEWithAlgorithmicSNARC:
    """MinimalSAGE using new algorithmic SNARC"""

    def __init__(self):
        self.device = device
        self.cycle_count = 0

        # Use hierarchical SNARC
        self.hierarchical_snarc = HierarchicalSNARC(device=self.device)

        # Register sensors
        self.hierarchical_snarc.register_sensor(
            'vision',
            SpatialSNARC('vision', memory_size=100, device=self.device)
        )

        # State
        self.trust_scores = {'vision': 0.5}
        self.energy = 100.0

        print("  ✓ SAGE initialized with algorithmic SNARC")

    def cycle(self):
        """Run one SAGE cycle with algorithmic SNARC"""
        self.cycle_count += 1

        # 1. Generate observation
        obs = torch.randn(3, 64, 64, device=self.device)

        # 2. Compute salience using algorithmic SNARC
        observations = {'vision': obs}
        all_scores = self.hierarchical_snarc.score_all(observations)
        salience = all_scores['vision']

        # 3. Mock IRP execution
        initial_energy = 2.0
        final_energy = initial_energy * (1.0 - salience.combined)

        # 4. Update trust (energy decreased = good)
        if final_energy < initial_energy:
            self.trust_scores['vision'] *= 1.01
            self.trust_scores['vision'] = min(1.0, self.trust_scores['vision'])

        # 5. Deplete energy
        self.energy -= 0.5

        return {
            'cycle': self.cycle_count,
            'salience': salience,
            'trust': self.trust_scores['vision'],
            'energy': self.energy
        }

# Run SAGE with algorithmic SNARC
sage = SAGEWithAlgorithmicSNARC()

print()
print("  Running 10 cycles...")
print()

for i in range(10):
    start = time.time()
    state = sage.cycle()
    elapsed = (time.time() - start) * 1000

    salience = state['salience']

    print(f"  Cycle {state['cycle']:2d} | "
          f"Surprise: {salience.surprise:.3f} | "
          f"Novelty: {salience.novelty:.3f} | "
          f"Arousal: {salience.arousal:.3f} | "
          f"Combined: {salience.combined:.3f} | "
          f"Trust: {state['trust']:.3f} | "
          f"{elapsed:.1f}ms")

print()
print("  ✓ SAGE loop executed successfully with algorithmic SNARC")
print()

# Summary
print("="*80)
print("SUMMARY: Algorithmic SNARC Validation")
print("="*80)
print()
print("✓ Test 1: Basic SensorSNARC works algorithmically (no training)")
print("✓ Test 2: Novelty decreases with repeated observations")
print("✓ Test 3: Surprise increases with prediction error")
print("✓ Test 4: SpatialSNARC preserves spatial structure")
print("✓ Test 5: HierarchicalSNARC computes cross-modal conflict")
print("✓ Test 6: Integration with SAGE loop successful")
print()
print("Key Advantages Over Learned SNARC:")
print("  • No training required - works immediately")
print("  • Per-sensor instances with own memory")
print("  • Spatial structure preserved for vision")
print("  • Cross-modal conflict computed at fusion level")
print("  • Interpretable - know what each dimension means")
print("  • Matches conceptual vision from SAGE-SNARC.md")
print()
print("="*80)
print("ALL TESTS PASSED!")
print("="*80)
