"""
Audio IRP Plugin - Spectral Analysis Implementation

Refines audio spectrograms in latent space for:
- Noise reduction
- Event detection
- Temporal coherence

Similar to VisionIRP but adapted for spectro-temporal data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import sys
import os

# Add parent directories to path for imports
_current_dir = os.path.dirname(__file__)
_sage_root = os.path.dirname(os.path.dirname(_current_dir))
_hrm_root = os.path.dirname(_sage_root)
if _sage_root not in sys.path:
    sys.path.insert(0, _sage_root)
if _hrm_root not in sys.path:
    sys.path.insert(0, _hrm_root)

from irp.base import IRPPlugin, IRPState


class SpectralRefiner(nn.Module):
    """
    Refines spectrograms by denoising and enhancing salient events

    Operates on [n_mels, time_frames] spectrograms
    Uses 1D convolutions along time axis with multi-scale analysis
    """

    def __init__(self, n_mels: int = 64):
        super().__init__()

        # Multi-scale temporal analysis
        self.short_scale = nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1)
        self.medium_scale = nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2)
        self.long_scale = nn.Conv1d(n_mels, n_mels, kernel_size=7, padding=3)

        # Fusion
        self.fusion = nn.Conv1d(n_mels * 3, n_mels, kernel_size=1)

        # Normalization
        self.norm = nn.BatchNorm1d(n_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refine spectrogram
        Input/Output: [B, n_mels, time_frames]
        """
        # Multi-scale analysis
        short = F.relu(self.short_scale(x))
        medium = F.relu(self.medium_scale(x))
        long = F.relu(self.long_scale(x))

        # Concatenate scales
        multi_scale = torch.cat([short, medium, long], dim=1)

        # Fuse and normalize
        refined = self.fusion(multi_scale)
        refined = self.norm(refined)

        # Residual connection
        return x + refined


class AudioIRPImpl(IRPPlugin):
    """
    Audio IRP implementation for spectral refinement

    Iteratively refines spectrograms to:
    - Reduce ambient noise
    - Enhance events
    - Improve temporal coherence
    """

    def __init__(
        self,
        n_mels: int = 64,
        max_iterations: int = 20,
        eps: float = 0.01,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        # Create default config if not provided
        if config is None:
            config = {
                'max_iterations': max_iterations,
                'halt_eps': eps,
                'entity_id': 'audio_irp'
            }
        super().__init__(config)

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_mels = n_mels
        self.max_iterations = max_iterations
        self.eps = eps

        # Create refiner network
        self.refiner = SpectralRefiner(n_mels=n_mels).to(self.device)

        # Cache for original spectrogram
        self.original_spec = None
        self.iteration = 0

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with task-specific configuration"""
        self.max_iterations = config.get('max_iterations', self.max_iterations)
        self.eps = config.get('eps', self.eps)

    def preprocess(self, x: Any) -> torch.Tensor:
        """
        Convert input to tensor
        Input: Spectrogram tensor [n_mels, time_frames] or [B, n_mels, time_frames]
        Output: Normalized tensor ready for refinement
        """
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        # Ensure on correct device
        x = x.to(self.device)

        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Store original for quality evaluation
        self.original_spec = x.clone()

        return x

    def compute_energy(self, state: torch.Tensor) -> float:
        """
        Compute energy of current state
        Lower energy = better signal-to-noise + temporal coherence
        """
        with torch.no_grad():
            # Reconstruction error (should preserve signal)
            recon_error = F.mse_loss(state, self.original_spec)

            # Temporal coherence (penalize rapid changes)
            temporal_diff = torch.diff(state, dim=2)  # Diff along time axis
            temporal_roughness = torch.mean(torch.abs(temporal_diff))

            # Sparsity (encourage sparse activations for events)
            sparsity = torch.mean(torch.abs(state))

            # Combined energy
            energy = recon_error + 0.3 * temporal_roughness + 0.1 * sparsity

        return -energy.item()  # Negative so lower is better

    def refine_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Single refinement step
        """
        # Apply refiner network
        refined = self.refiner(state)

        # Soft threshold (denoise)
        refined = torch.sign(refined) * F.relu(torch.abs(refined) - 0.1)

        # Clip to reasonable range
        refined = torch.clamp(refined, 0, 1)

        self.iteration += 1
        return refined

    # ----- IRP Contract Implementation -----

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """Initialize refinement state (IRP contract)"""
        # Preprocess input to tensor
        spec = self.preprocess(x0)

        # Create IRPState
        state = IRPState(
            x=spec,
            step_idx=0,
            energy_val=None,
            meta={
                'original_spec': self.original_spec,
                'atp_budget': task_ctx.get('atp_budget', 10.0)
            }
        )

        # Compute initial energy
        state.energy_val = self.compute_energy(spec)

        return state

    def energy(self, state: IRPState) -> float:
        """Compute energy (IRP contract)"""
        if state.energy_val is not None:
            return state.energy_val
        return self.compute_energy(state.x)

    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """Execute one refinement step (IRP contract)"""
        # Refine spectrogram
        refined_spec = self.refine_step(state.x)

        # Create new state
        new_state = IRPState(
            x=refined_spec,
            step_idx=state.step_idx + 1,
            energy_val=self.compute_energy(refined_spec),
            meta=state.meta.copy()
        )

        return new_state

    def halt(self, history: List[IRPState]) -> bool:
        """Determine if refinement should stop (IRP contract)"""
        if len(history) < 2:
            return False

        # Extract energy history
        energy_history = [s.energy_val for s in history if s.energy_val is not None]

        return self.should_halt(energy_history)

    # ----- Legacy Methods -----

    def should_halt(self, energy_history: list) -> bool:
        """
        Determine if refinement should stop early
        Based on energy convergence
        """
        if len(energy_history) < 3:
            return False

        # Check if energy has converged
        recent = energy_history[-3:]
        delta = abs(recent[-1] - recent[-2])

        # Stop if converged or max iterations
        return delta < self.eps or self.iteration >= self.max_iterations

    def postprocess(self, state: torch.Tensor) -> Any:
        """
        Convert refined spectrogram back to original format
        """
        # Remove batch dimension if it was added
        if state.shape[0] == 1:
            state = state.squeeze(0)

        return state

    def compute_trust(self, initial: Any, refined: Any) -> float:
        """
        Compute trust score based on quality improvement
        """
        # Measure noise reduction (energy decrease)
        initial_energy = self.compute_energy(initial)
        refined_energy = self.compute_energy(refined)

        energy_improvement = refined_energy - initial_energy  # More negative = better

        # Trust based on improvement
        if energy_improvement < -0.5:
            trust = 1.0  # Excellent improvement
        elif energy_improvement < -0.2:
            trust = 0.8  # Good improvement
        elif energy_improvement < 0:
            trust = 0.6  # Some improvement
        else:
            trust = 0.3  # No improvement or degradation

        # Bonus for early stopping (efficient)
        if self.iteration < self.max_iterations * 0.5:
            trust *= 1.1

        return min(trust, 1.0)

    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Complete refinement pipeline
        """
        # Reset iteration counter
        self.iteration = 0

        # Preprocess to tensor
        state = self.preprocess(x)
        initial_state = state.clone()

        # Track energy
        energy_history = []
        energy_history.append(self.compute_energy(state))

        # Refinement loop
        while self.iteration < self.max_iterations:
            # Refine
            state = self.refine_step(state)

            # Compute energy
            energy = self.compute_energy(state)
            energy_history.append(energy)

            # Check early stop
            if early_stop and self.should_halt(energy_history):
                break

        # Postprocess
        refined = self.postprocess(state)

        # Compute metrics
        trust = self.compute_trust(self.original_spec, state)

        # Build telemetry
        telemetry = {
            'iterations': self.iteration,
            'final_energy': energy_history[-1],
            'energy_delta': energy_history[-1] - energy_history[0],
            'trust': trust,
            'early_stopped': self.iteration < self.max_iterations,
            'compute_saved': 1 - (self.iteration / self.max_iterations)
        }

        return refined, telemetry


def create_audio_irp(device: Optional[torch.device] = None) -> AudioIRPImpl:
    """Factory function for Audio IRP"""
    return AudioIRPImpl(
        n_mels=64,
        max_iterations=20,
        device=device
    )


if __name__ == "__main__":
    print("Testing Audio IRP Implementation")
    print("=" * 50)

    # Create IRP
    irp = create_audio_irp()

    # Test with random spectrogram
    test_spec = torch.randn(64, 32).cuda()  # 64 mels, 32 time frames

    # Run refinement
    print("\nRunning refinement with early stopping...")
    refined, telemetry = irp.refine(test_spec, early_stop=True)

    print(f"\nResults:")
    print(f"  Iterations: {telemetry['iterations']}")
    print(f"  Final energy: {telemetry['final_energy']:.4f}")
    print(f"  Trust score: {telemetry['trust']:.3f}")
    print(f"  Compute saved: {telemetry['compute_saved']*100:.1f}%")

    print("\nAudio IRP working!")
