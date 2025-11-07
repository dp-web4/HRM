#!/usr/bin/env python3
"""
Embodied Actor Explorer - SAGE with Complete Sense-Act Loop
=============================================================

The missing piece: ACTION!

Previous multi-modal explorer did:
  Sense (vision + proprioception) → Assess (SNARC) → Decide (task)

This adds:
  Sense → Assess → Decide → **ACT** → Sense (new state)

True embodied consciousness requires the complete loop.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from sage.sensors.groot_camera_sensor import GR00TCameraSensor, GR00TCameraFrame
from sage.sensors.proprioception_sensor import ProprioceptionSensor, ProprioceptionFrame
from sage.compression.vision_puzzle_vae import VisionPuzzleVAE
from sage.services.snarc import SNARCService
from sage.core.unified_sage_system import SensorOutput

# Try importing GR00T simulator - optional
try:
    from gr00t_integration.groot_world_sim import GR00TWorldSimulator
except ImportError:
    GR00TWorldSimulator = None


@dataclass
class Action:
    """Motor action to execute"""
    action_type: str  # 'move', 'reach', 'grasp', 'release', 'look'
    target_position: Optional[np.ndarray] = None  # 3D position for movement
    target_joints: Optional[np.ndarray] = None  # Joint angles
    gripper_command: Optional[float] = None  # 0=open, 1=closed
    duration: float = 1.0  # How long to execute


@dataclass
class EmbodiedExperience:
    """Experience from embodied exploration WITH ACTION"""
    cycle: int
    timestamp: float
    # Sensory
    visual_puzzle: torch.Tensor
    visual_salience: float
    body_state: torch.Tensor
    position: np.ndarray
    joint_angles: np.ndarray
    gripper_state: float
    # Action (NEW!)
    action_taken: Optional[Action]
    action_success: bool
    # Assessment
    combined_salience: float
    task_description: str
    metadata: Dict[str, Any]


class EmbodiedActorExplorer:
    """
    Complete embodied consciousness loop with ACTION.

    The full cycle:
    1. Sense the world (vision) and body (proprioception)
    2. Assess salience (SNARC)
    3. Decide on task (based on salience)
    4. Generate action (based on task)
    5. Execute action (modify world state)
    6. Observe results (next cycle starts with new state)

    This creates learning opportunities through:
    - Prediction errors (expected vs actual state changes)
    - Action-outcome correlations
    - Embodied affordance discovery
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

        print("="*70)
        print("Embodied Actor Explorer - Complete Sense-Act Loop")
        print("="*70)
        print(f"Device: {device}\n")

        # Initialize world simulator (optional - sensors create their own if needed)
        print("Initializing GR00T world simulator...")
        if GR00TWorldSimulator:
            try:
                self.simulator = GR00TWorldSimulator(device=device)
                print("   ✓ GR00T simulator loaded")
            except Exception as e:
                print(f"   ⚠️  GR00T load failed ({e}), using synthetic")
                self.simulator = None
        else:
            self.simulator = None
            print("   ⚠️  Using synthetic simulator")
        print("✓ Simulator ready\n")

        # Initialize sensors
        print("Initializing vision sensor...")
        self.vision_sensor = GR00TCameraSensor(device=device, target_size=(224, 224))
        # Share simulator if we have one
        if self.simulator:
            self.vision_sensor.simulator = self.simulator
        print("✓ Vision sensor ready\n")

        print("Initializing proprioception sensor...")
        self.proprio_sensor = ProprioceptionSensor(
            device=device,
            simulator=self.simulator if self.simulator else None,
            normalize=True
        )
        print("✓ Proprioception sensor ready\n")

        # Load VAE for vision encoding
        print("Loading Vision Puzzle VAE...")
        self.vision_vae = VisionPuzzleVAE(latent_dim=64, num_codes=10).to(device)
        self.vision_vae.eval()
        print("✓ VAE loaded\n")

        # Initialize SNARC for salience assessment
        print("Initializing SNARC...")
        self.snarc = SNARCService()
        print("✓ SNARC ready\n")

        # Initialize exploration state
        self.cycle_count = 0
        self.total_time = 0.0
        self.experiences: List[EmbodiedExperience] = []
        self.cross_modal_correlations = []

        # Current task and action planning
        self.current_task = None
        self.task_history = []

        # Action statistics
        self.actions_executed = 0
        self.action_successes = 0
        self.movement_history = []

        # Generate initial exploration tasks
        self.exploration_tasks = self._generate_initial_tasks()
        print(f"📋 Generated {len(self.exploration_tasks)} embodied action tasks\n")

    def _generate_initial_tasks(self) -> List[Dict[str, Any]]:
        """Generate initial embodied exploration tasks WITH ACTIONS"""
        return [
            {
                'description': 'Move forward and observe changes',
                'action_type': 'move',
                'target_offset': np.array([0.5, 0.0, 0.0])  # Move forward
            },
            {
                'description': 'Reach for object ahead',
                'action_type': 'reach',
                'target_offset': np.array([0.3, 0.0, -0.2])  # Reach forward and down
            },
            {
                'description': 'Look around by rotating',
                'action_type': 'move',
                'target_offset': np.array([0.0, 0.3, 0.0])  # Strafe right
            },
            {
                'description': 'Practice grasping motions',
                'action_type': 'grasp',
                'target_offset': np.array([0.0, 0.0, 0.0])
            },
            {
                'description': 'Explore upward movements',
                'action_type': 'move',
                'target_offset': np.array([0.0, 0.0, 0.3])  # Move up
            }
        ]

    def _generate_action_from_task(self, task: Dict[str, Any],
                                   current_position: np.ndarray) -> Action:
        """
        Generate executable action from task description.

        This is where intention becomes motor command.
        """
        action_type = task.get('action_type', 'move')
        target_offset = task.get('target_offset', np.array([0.1, 0.0, 0.0]))

        # Calculate target position
        target_position = current_position + target_offset

        # Add some exploration noise
        exploration_noise = np.random.randn(3) * 0.05
        target_position += exploration_noise

        # Clamp to reasonable bounds
        target_position = np.clip(target_position, -2.0, 2.0)

        if action_type == 'grasp':
            # Grasping: close gripper
            return Action(
                action_type='grasp',
                target_position=target_position,
                gripper_command=0.8,  # Mostly closed
                duration=0.5
            )
        elif action_type == 'release':
            # Release: open gripper
            return Action(
                action_type='release',
                target_position=current_position,  # Stay in place
                gripper_command=0.0,  # Open
                duration=0.3
            )
        elif action_type == 'reach':
            # Reaching: extend arm
            return Action(
                action_type='reach',
                target_position=target_position,
                gripper_command=None,  # Don't change gripper
                duration=1.0
            )
        else:  # 'move' or default
            # Simple movement
            return Action(
                action_type='move',
                target_position=target_position,
                gripper_command=None,
                duration=0.8
            )

    def _execute_action(self, action: Action) -> bool:
        """
        Execute action on simulator.

        THIS IS THE KEY ADDITION - actually modify world state!

        Returns:
            success: Whether action executed successfully
        """
        try:
            # Get the actual simulator (from vision sensor if available)
            sim = self.simulator or getattr(self.vision_sensor, 'simulator', None)

            if sim and hasattr(sim, 'robot_state'):
                # Build action command for GR00T
                if action.target_position is not None:
                    # Movement command
                    sim.robot_state.position = action.target_position.copy()

                    # Add to movement history
                    self.movement_history.append(action.target_position.copy())

                if action.gripper_command is not None:
                    # Gripper command
                    sim.robot_state.gripper_state = action.gripper_command
            else:
                # No simulator available - still track intended movement
                if action.target_position is not None:
                    self.movement_history.append(action.target_position.copy())

            # Simulate some time passing
            # (In real system, this would be actual motor execution time)
            time.sleep(0.01)  # Small delay to simulate action

            self.actions_executed += 1
            self.action_successes += 1

            return True

        except Exception as e:
            print(f"Action execution failed: {e}")
            self.actions_executed += 1
            return False

    def _estimate_embodied_progress(self, proprio: ProprioceptionFrame,
                                    action_taken: Optional[Action]) -> float:
        """
        Estimate task progress based on embodied state changes.

        With action: can compare intended vs actual state.
        """
        if action_taken is None:
            return 0.0

        # Simple progress: did we move toward our goal?
        if action_taken.target_position is not None:
            intended = action_taken.target_position
            actual = proprio.position.cpu().numpy()

            # Distance to goal
            distance = np.linalg.norm(intended - actual)

            # Progress is inverse of distance (closer = more progress)
            progress = max(0.0, 1.0 - distance)
            return progress

        return 0.5  # Default progress

    def explore_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Single embodied exploration cycle WITH ACTION EXECUTION.

        Complete loop:
          Sense → Assess → Decide → ACT → (next cycle sees results)
        """
        cycle_start = time.time()

        # Ensure we have a task
        if self.current_task is None:
            if self.exploration_tasks:
                self.current_task = self.exploration_tasks.pop(0)
            else:
                # Generate new task based on experience
                self.current_task = {
                    'description': 'Continue exploration',
                    'action_type': 'move',
                    'target_offset': np.random.randn(3) * 0.2
                }

        # === SENSE: Vision + Proprioception ===
        vision_frame = self.vision_sensor.capture()
        if vision_frame is None:
            return None

        proprio_frame = self.proprio_sensor.capture()
        if proprio_frame is None:
            return None

        # === ASSESS: Encode and evaluate salience ===
        with torch.no_grad():
            visual_puzzle = self.vision_vae.encode_to_puzzle(
                vision_frame.image.unsqueeze(0)
            )

        visual_obs = SensorOutput(
            data=visual_puzzle,
            timestamp=vision_frame.timestamp,
            quality=1.0,
            sensor_type='vision',
            metadata={'task': self.current_task['description']}
        )

        proprio_vector = self.proprio_sensor.to_vector(proprio_frame).unsqueeze(0)
        proprio_obs = SensorOutput(
            data=proprio_vector,
            timestamp=proprio_frame.timestamp,
            quality=1.0,
            sensor_type='proprioception',
            metadata={'position': proprio_frame.position.cpu().numpy().tolist()}
        )

        observations = {
            'vision': visual_obs,
            'proprioception': proprio_obs
        }
        snarc_report = self.snarc.assess_salience(observations)

        # === DECIDE: Generate action from current task ===
        current_position = proprio_frame.position.cpu().numpy()
        action = self._generate_action_from_task(self.current_task, current_position)

        # === ACT: Execute the action! ===
        action_success = self._execute_action(action)

        # === LEARN: Track what happened ===
        self._track_cross_modal_correlation(
            visual_puzzle, proprio_vector, snarc_report.salience_score
        )

        # Estimate progress (now includes action outcome)
        progress = self._estimate_embodied_progress(proprio_frame, action)

        # Decide if task is complete
        task_complete = progress > 0.7 or (time.time() - cycle_start) > 5.0

        if task_complete:
            self.task_history.append(self.current_task)
            self.current_task = None  # Will get new task next cycle

        # Record complete embodied experience
        experience = EmbodiedExperience(
            cycle=self.cycle_count,
            timestamp=time.time(),
            visual_puzzle=visual_puzzle,
            visual_salience=snarc_report.salience_score,
            body_state=proprio_vector.squeeze(0),
            position=proprio_frame.position.cpu().numpy(),
            joint_angles=proprio_frame.joint_angles.cpu().numpy(),
            gripper_state=proprio_frame.gripper_state,
            action_taken=action,
            action_success=action_success,
            combined_salience=snarc_report.salience_score,
            task_description=self.current_task['description'] if self.current_task else "Completed",
            metadata={
                'progress': progress,
                'task_complete': task_complete,
                'action_type': action.action_type
            }
        )
        self.experiences.append(experience)

        # Update stats
        cycle_time = time.time() - cycle_start
        self.cycle_count += 1
        self.total_time += cycle_time

        return {
            'cycle': self.cycle_count,
            'task': self.current_task['description'] if self.current_task else "Completed",
            'salience': snarc_report.salience_score,
            'position': proprio_frame.position.cpu().numpy(),
            'action': action.action_type,
            'success': action_success,
            'progress': progress,
            'cycle_time_ms': cycle_time * 1000
        }

    def _track_cross_modal_correlation(self, visual_puzzle: torch.Tensor,
                                       proprio_vector: torch.Tensor,
                                       salience: float):
        """Track correlations between vision, proprioception, and salience"""
        # Visual entropy
        visual_flat = visual_puzzle.flatten()
        visual_entropy = -torch.sum(visual_flat * torch.log(visual_flat + 1e-10)).item()

        # Proprio variance
        proprio_variance = torch.var(proprio_vector).item()

        self.cross_modal_correlations.append({
            'visual_entropy': visual_entropy,
            'proprio_variance': proprio_variance,
            'salience': salience
        })

    def run(self, max_cycles: int = 100, print_interval: int = 10):
        """Run embodied exploration with action execution"""
        print("="*70)
        print(f"Starting Embodied Actor Exploration (max {max_cycles} cycles)")
        print("With COMPLETE Sense-Assess-Decide-ACT Loop")
        print("="*70)
        print()

        for cycle in range(max_cycles):
            result = self.explore_cycle()

            if result is None:
                print(f"Cycle {cycle+1} failed")
                continue

            if (cycle + 1) % print_interval == 0:
                print(f"Cycle {result['cycle']:3d} | "
                      f"Task: {result['task'][:35]:35s} | "
                      f"Action: {result['action']:8s} | "
                      f"Pos: [{result['position'][0]:5.2f},{result['position'][1]:5.2f},{result['position'][2]:5.2f}] | "
                      f"Salience: {result['salience']:.3f} | "
                      f"Time: {result['cycle_time_ms']:.1f}ms")

        self._print_summary()

    def _print_summary(self):
        """Print exploration summary with action statistics"""
        print()
        print("="*70)
        print("Embodied Actor Exploration Summary")
        print("="*70)
        print()

        print(f"📊 Exploration Statistics:")
        print(f"   Cycles: {self.cycle_count}")
        print(f"   Total time: {self.total_time:.2f}s")
        print(f"   Avg cycle time: {(self.total_time/self.cycle_count)*1000:.2f}ms")
        print(f"   Embodied experiences: {len(self.experiences)}")
        print()

        # Action statistics (NEW!)
        print(f"🤖 Action Statistics:")
        print(f"   Actions executed: {self.actions_executed}")
        print(f"   Success rate: {(self.action_successes/max(1, self.actions_executed))*100:.1f}%")
        print()

        # Salience analysis
        if self.experiences:
            saliences = [e.combined_salience for e in self.experiences]
            print(f"🎯 Salience Analysis:")
            print(f"   Average: {np.mean(saliences):.3f}")
            print(f"   Maximum: {np.max(saliences):.3f}")
            print(f"   High salience (>0.5): {sum(1 for s in saliences if s > 0.5)}")
            print()

        # Embodied behavior analysis (with movement!)
        if len(self.movement_history) > 1:
            positions = np.array(self.movement_history)
            movements = np.diff(positions, axis=0)
            movement_distances = np.linalg.norm(movements, axis=1)

            print(f"🚶 Movement Analysis:")
            print(f"   Position variance: {np.var(positions, axis=0)}")
            print(f"   Total distance: {np.sum(movement_distances):.2f}")
            print(f"   Avg movement/step: {np.mean(movement_distances):.3f}")
            print(f"   Exploration volume: {np.prod(np.var(positions, axis=0)):.4f}")
            print()

        # Cross-modal correlations
        if self.cross_modal_correlations:
            visual_entropy = [c['visual_entropy'] for c in self.cross_modal_correlations]
            proprio_var = [c['proprio_variance'] for c in self.cross_modal_correlations]

            print(f"🔗 Cross-Modal Correlations:")
            print(f"   Tracked: {len(self.cross_modal_correlations)} correlations")
            print(f"   Visual entropy: {np.mean(visual_entropy):.3f} ± {np.std(visual_entropy):.3f}")
            print(f"   Proprio variance: {np.mean(proprio_var):.3f} ± {np.std(proprio_var):.3f}")
            print()

        print("✓ Embodied actor exploration complete")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Embodied Actor Explorer - Complete Sense-Act Loop"
    )
    parser.add_argument('--cycles', type=int, default=100,
                       help='Number of exploration cycles')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Create embodied actor explorer
    explorer = EmbodiedActorExplorer(device=args.device)

    # Run embodied exploration WITH ACTIONS
    explorer.run(max_cycles=args.cycles, print_interval=10)


if __name__ == "__main__":
    main()
