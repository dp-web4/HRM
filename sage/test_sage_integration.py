#!/usr/bin/env python3
"""
SAGE Integration Test - Phase 1
Actually wire real components and run the loop
"""

import sys
import torch
import time
from pathlib import Path

# Add sage to path
sage_root = Path(__file__).parent
sys.path.insert(0, str(sage_root))

print("="*80)
print("SAGE Integration Test - Phase 1: Minimal Working Loop")
print("="*80)
print()

# Step 1: Test if we can import existing components
print("[Step 1] Testing imports of existing components...")
try:
    from attention.snarc_scorer import SNARCScorer
    print("  ✓ SNARCScorer imported")
except Exception as e:
    print(f"  ✗ SNARCScorer failed: {e}")
    SNARCScorer = None

try:
    from irp.base import IRPPlugin, IRPState
    print("  ✓ IRP base classes imported")
except Exception as e:
    print(f"  ✗ IRP base failed: {e}")

try:
    from irp.plugins.vision_impl import VisionIRP
    print("  ✓ VisionIRP imported")
except Exception as e:
    print(f"  ✗ VisionIRP failed: {e}")
    VisionIRP = None

try:
    from memory.irp_memory_bridge import IRPMemoryBridge
    print("  ✓ IRPMemoryBridge imported")
except Exception as e:
    print(f"  ✗ IRPMemoryBridge failed: {e}")
    IRPMemoryBridge = None

print()

# Step 2: Create minimal SAGE loop with real components
print("[Step 2] Building minimal SAGE loop...")

class MinimalSAGE:
    """
    Minimal SAGE implementation using real components
    Phase 1: Just get it running
    """

    def __init__(self):
        self.cycle_count = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")

        # Initialize real components
        if SNARCScorer:
            self.snarc = SNARCScorer(hidden_size=256, memory_size=1000).to(self.device)
            self.snarc.eval()  # Inference mode
            print("  ✓ Real SNARC scorer initialized")
        else:
            self.snarc = None
            print("  ○ SNARC scorer not available, using mock")

        if IRPMemoryBridge:
            try:
                self.memory = IRPMemoryBridge(
                    buffer_size=100,
                    snarc_capacity=1000,
                    consolidation_threshold=50,
                    device=self.device
                )
                print("  ✓ Real IRP memory bridge initialized")
            except Exception as e:
                print(f"  ○ Memory bridge initialization failed: {e}, using None")
                self.memory = None
        else:
            self.memory = None
            print("  ○ Memory bridge not available")

        # State
        self.observations_history = []
        self.trust_scores = {'vision': 0.5}
        self.energy = 100.0

        print("  ✓ Minimal SAGE initialized")
        print()

    def _generate_observation(self):
        """Generate mock observation (will replace with real sensor)"""
        # Simulate camera frame
        frame = torch.randn(3, 64, 64, device=self.device)
        return {'vision': frame}

    def _compute_salience(self, obs):
        """Compute salience using real SNARC or mock"""
        if self.snarc is not None:
            try:
                # SNARC expects [batch, seq, hidden] tensors
                # Flatten observation to hidden state representation
                obs_flat = obs['vision'].flatten()
                # Pad or truncate to 256 dimensions
                if obs_flat.shape[0] < 256:
                    hidden = torch.zeros(256, device=self.device)
                    hidden[:obs_flat.shape[0]] = obs_flat
                else:
                    hidden = obs_flat[:256]

                # Add batch and sequence dimensions
                hidden = hidden.unsqueeze(0).unsqueeze(0)  # [1, 1, 256]

                with torch.no_grad():
                    # SNARC uses forward() method returning 5D scores
                    scores = self.snarc(hidden, return_components=False)
                    # Average for overall salience
                    salience = scores.mean().item()

                return {'vision': salience}
            except Exception as e:
                print(f"      SNARC error: {e}, using fallback")
                import traceback
                traceback.print_exc()
                return {'vision': 0.5}
        else:
            # Mock salience
            return {'vision': 0.5 + torch.rand(1).item() * 0.3}

    def _run_irp_iteration(self, obs, salience):
        """Simulate IRP plugin execution"""
        # For now, mock the iterative refinement
        # Will replace with real VisionIRP
        initial_energy = 2.0
        final_energy = initial_energy * (1.0 - salience['vision'])

        return {
            'vision': {
                'latent': torch.randn(64, device=self.device),
                'energy_trajectory': [initial_energy, final_energy],
                'iterations': 2,
                'converged': True
            }
        }

    def _update_trust(self, results):
        """Update trust based on plugin behavior"""
        for plugin_name, result in results.items():
            energy_traj = result['energy_trajectory']

            # Check if energy decreased
            if len(energy_traj) >= 2:
                decreased = all(energy_traj[i] > energy_traj[i+1]
                               for i in range(len(energy_traj)-1))
                if decreased:
                    self.trust_scores[plugin_name] *= 1.01  # Increase trust
                    self.trust_scores[plugin_name] = min(1.0, self.trust_scores[plugin_name])
                else:
                    self.trust_scores[plugin_name] *= 0.99  # Decrease trust

    def _update_memory(self, obs, results):
        """Store in memory if available"""
        if self.memory is not None:
            try:
                # Store the observation and result
                vis_latent = results['vision']['latent']
                success = results['vision']['converged']

                self.memory.store_episode(
                    inputs={'obs': obs['vision'].cpu()},
                    outputs={'latent': vis_latent.cpu()},
                    success=success
                )
            except Exception as e:
                # Memory storage non-critical, continue
                pass

        # Also keep recent history
        self.observations_history.append(obs)
        if len(self.observations_history) > 10:
            self.observations_history.pop(0)

    def cycle(self):
        """Run one SAGE cycle"""
        self.cycle_count += 1

        # 1. Sense
        obs = self._generate_observation()

        # 2. Evaluate salience
        salience = self._compute_salience(obs)

        # 3. Run IRP plugins (simplified)
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
        print(f"[Step 3] Running SAGE for {num_cycles} cycles...")
        print()

        for i in range(num_cycles):
            start = time.time()
            state = self.cycle()
            elapsed = (time.time() - start) * 1000

            print(f"Cycle {state['cycle']:3d} | "
                  f"Salience: {state['salience']['vision']:.3f} | "
                  f"Trust: {state['trust']['vision']:.3f} | "
                  f"Energy: {state['energy']:5.1f}% | "
                  f"{elapsed:.1f}ms")

            if state['energy'] <= 0:
                print("\n  ⚠ Energy depleted, would transition to REST state")
                break

        print()
        print("[Step 4] SAGE loop completed successfully!")
        print()
        print("Summary:")
        print(f"  Total cycles: {self.cycle_count}")
        print(f"  Final trust: {self.trust_scores}")
        print(f"  Memory available: {self.memory is not None}")
        print(f"  Observations stored: {len(self.observations_history)}")

# Run the test
if __name__ == '__main__':
    sage = MinimalSAGE()
    sage.run(num_cycles=20)

    print()
    print("="*80)
    print("Phase 1 Test Complete!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Replace mock observation generator with real SensorHub")
    print("2. Integrate real VisionIRP plugin")
    print("3. Add real attention allocation")
    print("4. Add resource management")
    print("5. Test on camera")
