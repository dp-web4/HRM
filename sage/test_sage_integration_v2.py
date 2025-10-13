#!/usr/bin/env python3
"""
SAGE Integration Test - Phase 2
Using Algorithmic Per-Sensor SNARC

Updates from Phase 1:
- Replaced learned PyTorch SNARC with algorithmic SensorSNARC
- Per-sensor SNARC instances with own memory
- Spatial structure preservation
- Immediate operation (no training needed)
"""

import sys
import torch
import time
from pathlib import Path

# Add sage to path
sage_root = Path(__file__).parent
sys.path.insert(0, str(sage_root))

print("="*80)
print("SAGE Integration Test - Phase 2: Algorithmic Per-Sensor SNARC")
print("="*80)
print()

# Step 1: Test imports
print("[Step 1] Testing imports...")
try:
    from attention.sensor_snarc import HierarchicalSNARC, SpatialSNARC, SensorSNARC
    print("  ✓ Algorithmic SNARC imported")
except Exception as e:
    print(f"  ✗ SNARC import failed: {e}")
    sys.exit(1)

try:
    from irp.base import IRPPlugin, IRPState
    print("  ✓ IRP base classes imported")
except Exception as e:
    print(f"  ✗ IRP base failed: {e}")

try:
    from memory.irp_memory_bridge import IRPMemoryBridge
    print("  ✓ IRPMemoryBridge imported")
except Exception as e:
    print(f"  ✗ IRPMemoryBridge failed: {e}")
    IRPMemoryBridge = None

print()

# Step 2: Create SAGE loop with algorithmic SNARC
print("[Step 2] Building SAGE loop with algorithmic SNARC...")

class SAGEv2:
    """
    SAGE with algorithmic per-sensor SNARC

    Key differences from v1:
    - No learned SNARC parameters
    - Per-sensor SNARC instances
    - Spatial salience for vision
    - Cross-modal conflict computation
    """

    def __init__(self):
        self.cycle_count = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")

        # Initialize hierarchical SNARC
        self.hierarchical_snarc = HierarchicalSNARC(device=self.device)

        # Register sensors with spatial SNARC for vision
        self.hierarchical_snarc.register_sensor(
            'vision',
            SpatialSNARC('vision', memory_size=100, device=self.device)
        )
        print("  ✓ Hierarchical SNARC with spatial vision sensor")

        # Memory bridge (same as v1)
        if IRPMemoryBridge:
            try:
                self.memory = IRPMemoryBridge(
                    buffer_size=100,
                    snarc_capacity=1000,
                    consolidation_threshold=50,
                    device=self.device
                )
                print("  ✓ IRP memory bridge initialized")
            except Exception as e:
                print(f"  ○ Memory bridge initialization failed: {e}")
                self.memory = None
        else:
            self.memory = None

        # State
        self.observations_history = []
        self.trust_scores = {'vision': 0.5}
        self.energy = 100.0

        print("  ✓ SAGE v2 initialized")
        print()

    def _generate_observation(self):
        """Generate mock observation (will replace with real sensor)"""
        # Simulate camera frame
        frame = torch.randn(3, 64, 64, device=self.device)
        return {'vision': frame}

    def _compute_salience(self, obs):
        """Compute salience using algorithmic hierarchical SNARC"""
        # Use hierarchical SNARC - no training needed!
        all_scores = self.hierarchical_snarc.score_all(obs)

        # Return vision salience
        return all_scores['vision']

    def _run_irp_iteration(self, obs, salience):
        """Simulate IRP plugin execution"""
        # For now, mock the iterative refinement
        # Energy decreases more when salience is high (more work needed)
        initial_energy = 2.0
        final_energy = initial_energy * (1.0 - salience.combined * 0.5)

        return {
            'vision': {
                'latent': torch.randn(64, device=self.device),
                'energy_trajectory': [initial_energy, final_energy],
                'iterations': 2,
                'converged': True,
                'salience_used': salience.to_dict()
            }
        }

    def _update_trust(self, results):
        """Update trust based on plugin behavior"""
        for plugin_name, result in results.items():
            energy_traj = result['energy_trajectory']

            # Check if energy decreased monotonically
            if len(energy_traj) >= 2:
                decreased = all(energy_traj[i] > energy_traj[i+1]
                               for i in range(len(energy_traj)-1))
                if decreased:
                    self.trust_scores[plugin_name] *= 1.01
                    self.trust_scores[plugin_name] = min(1.0, self.trust_scores[plugin_name])
                else:
                    self.trust_scores[plugin_name] *= 0.99

    def _update_memory(self, obs, results):
        """Store in memory if available"""
        if self.memory is not None:
            try:
                vis_latent = results['vision']['latent']
                success = results['vision']['converged']

                self.memory.store_episode(
                    inputs={'obs': obs['vision'].cpu()},
                    outputs={'latent': vis_latent.cpu()},
                    success=success
                )
            except Exception:
                pass

        # Keep recent history
        self.observations_history.append(obs)
        if len(self.observations_history) > 10:
            self.observations_history.pop(0)

    def cycle(self):
        """Run one SAGE cycle"""
        self.cycle_count += 1

        # 1. Sense
        obs = self._generate_observation()

        # 2. Evaluate salience (algorithmic SNARC)
        salience = self._compute_salience(obs)

        # 3. Run IRP plugins
        results = self._run_irp_iteration(obs, salience)

        # 4. Update trust
        self._update_trust(results)

        # 5. Update memory
        self._update_memory(obs, results)

        # 6. Consume energy
        self.energy -= 0.5

        return {
            'cycle': self.cycle_count,
            'salience': salience,
            'trust': self.trust_scores.copy(),
            'energy': self.energy
        }

    def run(self, num_cycles=10):
        """Run SAGE for N cycles"""
        print(f"[Step 3] Running SAGE v2 for {num_cycles} cycles...")
        print()

        for i in range(num_cycles):
            start = time.time()
            state = self.cycle()
            elapsed = (time.time() - start) * 1000

            salience = state['salience']

            print(f"Cycle {state['cycle']:3d} | "
                  f"Surprise: {salience.surprise:.3f} | "
                  f"Novelty: {salience.novelty:.3f} | "
                  f"Arousal: {salience.arousal:.3f} | "
                  f"Conflict: {salience.conflict:.3f} | "
                  f"Combined: {salience.combined:.3f} | "
                  f"Trust: {state['trust']['vision']:.3f} | "
                  f"Energy: {state['energy']:5.1f}% | "
                  f"{elapsed:.1f}ms")

            if state['energy'] <= 0:
                print("\n  ⚠ Energy depleted, would transition to REST state")
                break

        print()
        print("[Step 4] SAGE v2 loop completed successfully!")
        print()
        print("Summary:")
        print(f"  Total cycles: {self.cycle_count}")
        print(f"  Final trust: {self.trust_scores}")
        print(f"  Memory available: {self.memory is not None}")
        print(f"  Observations stored: {len(self.observations_history)}")

# Run the test
if __name__ == '__main__':
    sage = SAGEv2()
    sage.run(num_cycles=20)

    print()
    print("="*80)
    print("Phase 2 Complete - Algorithmic SNARC Integrated!")
    print("="*80)
    print()
    print("Key Improvements Over Phase 1:")
    print("1. ✓ No learned parameters - works immediately")
    print("2. ✓ Per-sensor SNARC instances")
    print("3. ✓ All 5 SNARC dimensions computed algorithmically")
    print("4. ✓ Spatial structure preserved for vision")
    print("5. ✓ Cross-modal conflict ready (when multiple sensors added)")
    print()
    print("Next steps:")
    print("1. Add real sensor integration (camera)")
    print("2. Integrate real VisionIRP plugin")
    print("3. Add audio sensor with temporal SNARC")
    print("4. Test cross-modal conflict with multiple sensors")
    print("5. Visualize spatial salience heatmaps")
