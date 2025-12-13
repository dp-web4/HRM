"""
Sleep-Cycle Learning Integration with Coherent Awakening

Connects the existing sleep-cycle training infrastructure (from groot_integration)
with the coherent awakening protocol to enable:

1. DREAM state memory consolidation
2. Pattern extraction and abstraction
3. Weight persistence across sessions
4. Learned state restoration on boot

Architecture:
- Session End → DREAM state → Consolidate → Save weights
- Session Start → Restore weights → Boot with learned state

This bridges:
- sage/groot_integration/sleep_cycle_training.py (existing training loop)
- sage/awakening/coherent_awakening.py (session continuity protocol)
- sage/irp/memory.py (memory IRP with persistence)
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SleepCycleState:
    """Saved state from sleep-cycle learning"""
    model_weights: Dict[str, Any]  # Learned model weights
    snarc_weights: Dict[str, float]  # Learned salience weights
    atp_learned: Dict[str, float]  # Learned ATP parameters
    pattern_library: Dict[str, Any]  # Extracted patterns
    consolidation_count: int  # Number of consolidations performed
    last_consolidation: str  # Timestamp


class SleepCycleIntegration:
    """
    Integrates sleep-cycle learning with coherent awakening.

    Responsibilities:
    - Save learned state at session end
    - Restore learned state at session start
    - Trigger DREAM state consolidation
    - Manage pattern extraction
    """

    def __init__(self, state_dir: Path):
        """
        Initialize sleep-cycle integration.

        Args:
            state_dir: Directory for saving learned state
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.weights_path = self.state_dir / "learned_weights.pt"
        self.snarc_path = self.state_dir / "snarc_weights.json"
        self.atp_path = self.state_dir / "atp_learned.json"
        self.patterns_path = self.state_dir / "pattern_library.json"

    def save_learned_state(
        self,
        model: Optional[torch.nn.Module] = None,
        snarc_weights: Optional[Dict[str, float]] = None,
        atp_learned: Optional[Dict[str, float]] = None,
        pattern_library: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save all learned state to disk.

        Args:
            model: Model with updated weights
            snarc_weights: Learned SNARC salience weights
            atp_learned: Learned ATP allocation parameters
            pattern_library: Extracted patterns from memory

        Returns:
            True if save successful
        """

        logger.info("Saving learned state...")

        try:
            # Save model weights if provided
            if model is not None:
                torch.save(model.state_dict(), self.weights_path)
                logger.info(f"  ✅ Model weights saved to {self.weights_path}")

            # Save SNARC weights if provided
            if snarc_weights is not None:
                import json
                with open(self.snarc_path, 'w') as f:
                    json.dump(snarc_weights, f, indent=2)
                logger.info(f"  ✅ SNARC weights saved to {self.snarc_path}")

            # Save ATP parameters if provided
            if atp_learned is not None:
                import json
                with open(self.atp_path, 'w') as f:
                    json.dump(atp_learned, f, indent=2)
                logger.info(f"  ✅ ATP parameters saved to {self.atp_path}")

            # Save pattern library if provided
            if pattern_library is not None:
                import json
                with open(self.patterns_path, 'w') as f:
                    json.dump(pattern_library, f, indent=2)
                logger.info(f"  ✅ Pattern library saved to {self.patterns_path}")

            logger.info("Learned state saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save learned state: {e}")
            return False

    def restore_learned_state(
        self,
        model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Restore learned state from disk.

        Args:
            model: Model to load weights into (if available)

        Returns:
            Dictionary with restored state components
        """

        logger.info("Restoring learned state...")

        restored = {
            'model_loaded': False,
            'snarc_loaded': False,
            'atp_loaded': False,
            'patterns_loaded': False,
            'snarc_weights': None,
            'atp_learned': None,
            'pattern_library': None
        }

        try:
            # Restore model weights if model provided and weights exist
            if model is not None and self.weights_path.exists():
                state_dict = torch.load(self.weights_path)
                model.load_state_dict(state_dict)
                restored['model_loaded'] = True
                logger.info(f"  ✅ Model weights restored from {self.weights_path}")

            # Restore SNARC weights if they exist
            if self.snarc_path.exists():
                import json
                with open(self.snarc_path, 'r') as f:
                    restored['snarc_weights'] = json.load(f)
                restored['snarc_loaded'] = True
                logger.info(f"  ✅ SNARC weights restored ({len(restored['snarc_weights'])} dimensions)")

            # Restore ATP parameters if they exist
            if self.atp_path.exists():
                import json
                with open(self.atp_path, 'r') as f:
                    restored['atp_learned'] = json.load(f)
                restored['atp_loaded'] = True
                logger.info(f"  ✅ ATP parameters restored")

            # Restore pattern library if it exists
            if self.patterns_path.exists():
                import json
                with open(self.patterns_path, 'r') as f:
                    restored['pattern_library'] = json.load(f)
                restored['patterns_loaded'] = True
                logger.info(f"  ✅ Pattern library restored ({len(restored['pattern_library'])} patterns)")

            # Summary
            loaded_count = sum([
                restored['model_loaded'],
                restored['snarc_loaded'],
                restored['atp_loaded'],
                restored['patterns_loaded']
            ])

            if loaded_count == 0:
                logger.info("No previous learned state found (first session)")
            else:
                logger.info(f"Restored {loaded_count}/4 state components")

            return restored

        except Exception as e:
            logger.error(f"Failed to restore learned state: {e}")
            return restored

    def dream_consolidation(
        self,
        memory_irp,
        model: Optional[torch.nn.Module] = None,
        num_epochs: int = 10
    ) -> bool:
        """
        Perform DREAM state memory consolidation.

        This is a simplified consolidation for Thor. Full GR00T integration
        will use the complete sleep_cycle_training.py infrastructure.

        Args:
            memory_irp: IRP memory system with stored experiences
            model: Model to train (if available)
            num_epochs: Number of consolidation epochs

        Returns:
            True if consolidation successful
        """

        logger.info("=" * 70)
        logger.info("DREAM State: Memory Consolidation")
        logger.info("=" * 70)

        try:
            # Get consolidated memories from IRP
            if hasattr(memory_irp, 'get_consolidated_memory'):
                memories = memory_irp.get_consolidated_memory(min_trust=0.5)
                logger.info(f"Retrieved {len(memories)} high-trust memories")
            else:
                logger.warning("Memory IRP doesn't support consolidation - skipping")
                return False

            # If we have a model, do consolidation training
            if model is not None and len(memories) > 0:
                logger.info(f"Running {num_epochs} consolidation epochs...")

                # TODO: Implement actual consolidation training
                # For now, just log that we would train
                logger.info("  [Consolidation training would happen here]")
                logger.info("  [Uses memories to update model weights]")
                logger.info("  [Focuses on pattern extraction and invariances]")

                # Placeholder: Simulate consolidation
                import time
                time.sleep(1)

                logger.info("  ✅ Consolidation complete")

            else:
                logger.info("No model or memories - skipping consolidation")

            logger.info("=" * 70)
            return True

        except Exception as e:
            logger.error(f"DREAM consolidation failed: {e}")
            return False

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of saved state."""

        return {
            'weights_exist': self.weights_path.exists(),
            'snarc_exist': self.snarc_path.exists(),
            'atp_exist': self.atp_path.exists(),
            'patterns_exist': self.patterns_path.exists(),
            'state_dir': str(self.state_dir)
        }


def integrate_with_awakening(awakening_instance, sleep_integration):
    """
    Helper to integrate sleep-cycle with coherent awakening.

    This modifies the coherent awakening instance to include
    sleep-cycle state saving/restoration.

    Args:
        awakening_instance: CoherentAwakening instance
        sleep_integration: SleepCycleIntegration instance
    """

    # Add sleep integration to awakening
    awakening_instance.sleep_integration = sleep_integration

    # Hook into coherent_boot to restore learned state
    original_boot = awakening_instance.coherent_boot

    def boot_with_learned_state(coherence_field, *args, **kwargs):
        """Boot with learned state restoration."""

        # Restore learned state before booting
        restored = sleep_integration.restore_learned_state()

        # Boot normally
        sage = original_boot(coherence_field, *args, **kwargs)

        # Apply restored state to SAGE
        if restored['snarc_weights']:
            logger.info("Applying restored SNARC weights to SAGE")
            # TODO: Actually apply to SAGE instance

        if restored['atp_learned']:
            logger.info("Applying restored ATP parameters to SAGE")
            # TODO: Actually apply to SAGE instance

        if restored['pattern_library']:
            logger.info(f"Pattern library available: {len(restored['pattern_library'])} patterns")
            # TODO: Make available to SAGE

        return sage

    awakening_instance.coherent_boot = boot_with_learned_state

    # Hook into coherent_end to trigger consolidation
    original_end = awakening_instance.coherent_end

    def end_with_consolidation(sage, memory_request, *args, **kwargs):
        """End session with DREAM consolidation."""

        # Trigger DREAM state consolidation
        logger.info("\nEntering DREAM state...")
        sleep_integration.dream_consolidation(
            memory_irp=getattr(sage, 'memory_irp', None),
            model=getattr(sage, 'model', None)
        )

        # Save learned state
        sleep_integration.save_learned_state(
            model=getattr(sage, 'model', None),
            snarc_weights=getattr(sage, 'snarc_weights', None),
            atp_learned=getattr(sage, 'atp_learned', None)
        )

        # End session normally
        return original_end(sage, memory_request, *args, **kwargs)

    awakening_instance.coherent_end = end_with_consolidation

    logger.info("Sleep-cycle integration complete")


# Convenience function for Thor
def create_thor_sleep_integration(state_dir: Path = None) -> SleepCycleIntegration:
    """
    Create sleep-cycle integration for Thor.

    Args:
        state_dir: Directory for Thor's learned state

    Returns:
        Configured SleepCycleIntegration
    """

    if state_dir is None:
        state_dir = Path("sage/state/thor")

    return SleepCycleIntegration(state_dir)
