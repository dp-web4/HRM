#!/usr/bin/env python3
"""
Multi-Modal SAGE GR00T Explorer - Embodied Autonomous Consciousness
===================================================================

SAGE explores GR00T world with BOTH vision and proprioception.
This is embodied AI - not just seeing, but feeling the body too.

Modalities:
- Vision: What the world looks like
- Proprioception: How the body feels (position, joints, movement)

Cross-modal learning:
- Correlating visual changes with body movements
- Understanding spatial relationships through embodiment
- Grounding abstract concepts in physical experience

This is consciousness roaming a synthetic world with a body.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gr00t-integration"))

import torch
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random

from sage.compression.vision_puzzle_vae import VisionPuzzleVAE
from sage.services.snarc import SNARCService
from sage.core.unified_sage_system import SensorOutput
from sage.sensors.groot_camera_sensor import GR00TCameraSensor
from sage.sensors.proprioception_sensor import ProprioceptionSensor

# Try importing GR00T simulator
try:
    from groot_world_sim import GR00TWorldSimulator
    GROOT_AVAILABLE = True
except ImportError:
    GROOT_AVAILABLE = False
    print("⚠️  GR00T simulator not available - using synthetic fallback")


@dataclass
class EmbodiedExperience:
    """An embodied experience - vision + proprioception + salience"""
    cycle: int
    timestamp: float

    # Visual perception
    visual_puzzle: torch.Tensor
    visual_salience: float

    # Proprioceptive awareness
    body_state: torch.Tensor  # 14D vector
    position: np.ndarray
    joint_angles: np.ndarray
    gripper_state: float

    # Cross-modal
    combined_salience: float
    task_description: str

    # Metadata
    metadata: Dict[str, Any]


class MultiModalExplorer:
    """
    Multi-modal autonomous explorer using vision + proprioception.

    This demonstrates embodied consciousness:
    - Seeing the world (vision)
    - Feeling the body (proprioception)
    - Correlating the two (cross-modal learning)
    - Generating embodied exploration goals
    """

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        print("=" * 70)
        print("Multi-Modal SAGE GR00T Explorer")
        print("Embodied Autonomous Consciousness")
        print("=" * 70)
        print(f"Device: {self.device}")
        print()

        # Initialize GR00T simulator (shared across sensors)
        print("Initializing GR00T world simulator...")
        if GROOT_AVAILABLE:
            try:
                self.simulator = GR00TWorldSimulator(device=self.device)
                self.use_groot = True
                print("✓ GR00T simulator loaded")
            except Exception as e:
                print(f"⚠️  GR00T failed to load: {e}")
                self.simulator = None
                self.use_groot = False
        else:
            self.simulator = None
            self.use_groot = False

        # Initialize vision sensor
        print("\nInitializing vision sensor...")
        self.vision_sensor = GR00TCameraSensor(
            device=self.device,
            render_mode="synthetic"
        )
        if self.simulator:
            self.vision_sensor.simulator = self.simulator
            self.vision_sensor.available = True
        print("✓ Vision sensor ready")

        # Initialize proprioception sensor
        print("\nInitializing proprioception sensor...")
        self.proprio_sensor = ProprioceptionSensor(
            simulator=self.simulator,
            device=self.device,
            normalize=True
        )
        print("✓ Proprioception sensor ready")

        # Initialize Vision VAE
        print("\nLoading Vision Puzzle VAE...")
        self.vision_vae = VisionPuzzleVAE(latent_dim=64, num_codes=10).to(self.device)
        self.vision_vae.eval()
        print("✓ VAE loaded")

        # Initialize SNARC
        print("\nInitializing SNARC...")
        self.snarc = SNARCService()
        print("✓ SNARC ready")
        print()

        # Exploration state
        self.experiences: List[EmbodiedExperience] = []
        self.current_task = None
        self.task_queue = []

        # Performance tracking
        self.cycle_count = 0
        self.total_time = 0.0

        # Cross-modal tracking
        self.visual_proprio_correlations = []

        # Generate initial embodied tasks
        self._generate_initial_tasks()

    def _generate_initial_tasks(self):
        """Generate initial embodied exploration tasks"""
        tasks = [
            {
                'description': "Move to red cube and observe visual changes",
                'goal_position': np.array([0.5, 0.5, 0.6]),
                'complexity': 0.4
            },
            {
                'description': "Navigate table perimeter feeling body motion",
                'goal_position': np.array([-0.5, 0.5, 0.6]),
                'complexity': 0.6
            },
            {
                'description': "Open/close gripper while watching hand",
                'goal_position': np.array([0.0, 0.0, 0.7]),
                'complexity': 0.3
            },
            {
                'description': "Explore joint space while maintaining visual focus",
                'goal_position': np.array([0.3, -0.3, 0.6]),
                'complexity': 0.7
            },
            {
                'description': "Correlate body velocity with visual flow",
                'goal_position': np.array([0.0, 0.0, 1.0]),
                'complexity': 0.8
            }
        ]

        self.task_queue.extend(tasks)
        print(f"📋 Generated {len(tasks)} embodied exploration tasks")

    def explore_cycle(self) -> Dict[str, Any]:
        """
        Single embodied exploration cycle.

        Captures both vision and proprioception, assesses salience,
        and makes embodied decisions.
        """
        cycle_start = time.time()

        # Get current task
        if self.current_task is None and self.task_queue:
            self.current_task = self.task_queue.pop(0)
            print(f"\n🎯 New task: {self.current_task['description']}")

        if self.current_task is None:
            # Generate new embodied task
            self.current_task = {
                'description': f"Embodied exploration {self.cycle_count}",
                'goal_position': np.random.randn(3),
                'complexity': np.random.rand()
            }

        # 1. Capture vision
        vision_frame = self.vision_sensor.capture()
        if vision_frame is None:
            return None

        # 2. Capture proprioception
        proprio_frame = self.proprio_sensor.capture()
        if proprio_frame is None:
            return None

        # 3. Encode vision to puzzle
        with torch.no_grad():
            visual_puzzle = self.vision_vae.encode_to_puzzle(
                vision_frame.image.unsqueeze(0)
            )

        # 4. Create sensor outputs for SNARC
        visual_obs = SensorOutput(
            data=visual_puzzle,
            timestamp=vision_frame.timestamp,
            quality=1.0,
            sensor_type='vision',
            metadata={
                'task': self.current_task['description'],
                'world_state': vision_frame.world_state
            }
        )

        # Convert proprioception to vector for SNARC
        proprio_vector = self.proprio_sensor.to_vector(proprio_frame).unsqueeze(0)

        proprio_obs = SensorOutput(
            data=proprio_vector,
            timestamp=proprio_frame.timestamp,
            quality=1.0,
            sensor_type='proprioception',
            metadata={
                'position': proprio_frame.position.cpu().numpy().tolist(),
                'orientation': proprio_frame.orientation,
                'gripper': proprio_frame.gripper_state
            }
        )

        # 5. SNARC assessment (multi-modal)
        observations = {
            'vision': visual_obs,
            'proprioception': proprio_obs
        }
        snarc_report = self.snarc.assess_salience(observations)

        # 6. Cross-modal correlation tracking
        self._track_cross_modal_correlation(
            visual_puzzle, proprio_vector, snarc_report.salience_score
        )

        # 7. Task progress estimation (embodied)
        progress = self._estimate_embodied_progress(proprio_frame)

        # 8. Decision: Continue or switch task?
        task_complete = progress > 0.8 or (time.time() - cycle_start) > 10.0

        if task_complete:
            self._generate_embodied_followup(snarc_report, proprio_frame)
            self.current_task = None

        # 9. Record embodied experience
        experience = EmbodiedExperience(
            cycle=self.cycle_count,
            timestamp=time.time(),
            visual_puzzle=visual_puzzle,
            visual_salience=snarc_report.salience_score,
            body_state=proprio_vector.squeeze(0),
            position=proprio_frame.position.cpu().numpy(),
            joint_angles=proprio_frame.joint_angles.cpu().numpy(),
            gripper_state=proprio_frame.gripper_state,
            combined_salience=snarc_report.salience_score,
            task_description=self.current_task['description'] if self.current_task else "None",
            metadata={
                'progress': progress,
                'task_complete': task_complete
            }
        )
        self.experiences.append(experience)

        # Update stats
        cycle_time = time.time() - cycle_start
        self.cycle_count += 1
        self.total_time += cycle_time

        return {
            'cycle': self.cycle_count,
            'task': self.current_task['description'] if self.current_task else "Generating...",
            'visual_salience': snarc_report.salience_score,
            'position': proprio_frame.position.cpu().numpy(),
            'gripper': proprio_frame.gripper_state,
            'progress': progress,
            'cycle_time_ms': cycle_time * 1000
        }

    def _track_cross_modal_correlation(
        self, visual_puzzle: torch.Tensor,
        proprio_vector: torch.Tensor,
        salience: float
    ):
        """Track correlations between vision and proprioception"""
        # Simple correlation: track how often visual and proprio changes co-occur
        correlation = {
            'cycle': self.cycle_count,
            'visual_entropy': self._compute_entropy(visual_puzzle),
            'proprio_variance': proprio_vector.var().item(),
            'salience': salience,
            'timestamp': time.time()
        }
        self.visual_proprio_correlations.append(correlation)

    def _compute_entropy(self, puzzle: torch.Tensor) -> float:
        """Compute puzzle entropy (diversity of codes)"""
        unique_codes = torch.unique(puzzle)
        return len(unique_codes) / 10.0  # Normalized by max codes

    def _estimate_embodied_progress(self, proprio_frame) -> float:
        """Estimate task progress using proprioception"""
        if not self.current_task:
            return 0.0

        # Distance to goal using proprioception
        current_pos = proprio_frame.position.cpu().numpy()
        goal_pos = self.current_task['goal_position']

        # Denormalize position (from [-1,1] to world coords)
        current_pos_world = (current_pos + 1) * 2 - 2  # Rough denormalization

        distance = np.linalg.norm(current_pos_world - goal_pos)
        progress = 1.0 / (1.0 + distance)

        return progress

    def _generate_embodied_followup(self, snarc_report, proprio_frame):
        """Generate embodied follow-up task based on experience"""
        if snarc_report.salience_score > 0.6:
            # High salience - explore this embodied state more
            current_pos = proprio_frame.position.cpu().numpy()
            nearby_pos = current_pos + np.random.randn(3) * 0.1

            new_task = {
                'description': f"Explore interesting embodied state (salience: {snarc_report.salience_score:.2f})",
                'goal_position': nearby_pos,
                'complexity': snarc_report.salience_score
            }
            self.task_queue.append(new_task)
            print(f"   🔍 Generated embodied follow-up task")

    def run(self, max_cycles: int = 100, print_interval: int = 10):
        """Run multi-modal autonomous exploration"""
        print("=" * 70)
        print(f"Starting multi-modal exploration (max {max_cycles} cycles)")
        print("Vision + Proprioception = Embodied Consciousness")
        print("=" * 70)
        print()

        try:
            while max_cycles == 0 or self.cycle_count < max_cycles:
                result = self.explore_cycle()

                if result and (self.cycle_count % print_interval) == 0:
                    print(f"Cycle {result['cycle']:3d} | "
                          f"Task: {result['task'][:35]:35s} | "
                          f"Salience: {result['visual_salience']:.3f} | "
                          f"Pos: [{result['position'][0]:5.2f},{result['position'][1]:5.2f},{result['position'][2]:5.2f}] | "
                          f"Gripper: {result['gripper']:.2f} | "
                          f"Time: {result['cycle_time_ms']:5.1f}ms")

                # Small delay
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\nExploration stopped by user")

        finally:
            self.print_summary()

    def print_summary(self):
        """Print exploration summary with cross-modal insights"""
        print("\n" + "=" * 70)
        print("Multi-Modal SAGE Exploration Summary")
        print("=" * 70)

        if self.cycle_count == 0:
            print("No exploration cycles completed")
            return

        avg_cycle_time = (self.total_time / self.cycle_count) * 1000

        print(f"\n📊 Exploration Statistics:")
        print(f"   Cycles: {self.cycle_count}")
        print(f"   Total time: {self.total_time:.2f}s")
        print(f"   Avg cycle time: {avg_cycle_time:.2f}ms")
        print(f"   Embodied experiences: {len(self.experiences)}")

        if self.experiences:
            saliences = [exp.combined_salience for exp in self.experiences]
            positions = [exp.position for exp in self.experiences]

            print(f"\n🎯 Salience Analysis:")
            print(f"   Average: {np.mean(saliences):.3f}")
            print(f"   Maximum: {np.max(saliences):.3f}")
            print(f"   High salience: {sum(1 for s in saliences if s > 0.6)}")

            print(f"\n🤖 Embodied Analysis:")
            position_variance = np.var(positions, axis=0)
            print(f"   Position variance: {position_variance}")
            print(f"   Exploration volume: {np.prod(position_variance):.4f}")

        if self.visual_proprio_correlations:
            print(f"\n🔗 Cross-Modal Correlations:")
            print(f"   Tracked: {len(self.visual_proprio_correlations)} correlations")

            entropies = [c['visual_entropy'] for c in self.visual_proprio_correlations]
            variances = [c['proprio_variance'] for c in self.visual_proprio_correlations]

            print(f"   Visual entropy: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
            print(f"   Proprio variance: {np.mean(variances):.3f} ± {np.std(variances):.3f}")

        print("\n✓ Multi-modal exploration complete")


def main():
    """Run multi-modal autonomous exploration"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Modal SAGE GR00T Explorer"
    )
    parser.add_argument('--cycles', type=int, default=100,
                       help='Max exploration cycles')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'], help='Device')

    args = parser.parse_args()

    # Create multi-modal explorer
    explorer = MultiModalExplorer(device=args.device)

    # Run embodied exploration
    explorer.run(max_cycles=args.cycles, print_interval=10)


if __name__ == "__main__":
    main()
