#!/usr/bin/env python3
"""
SAGE GR00T Explorer - Autonomous Exploration of Simulated World
================================================================

SAGE explores GR00T's simulated 3D world autonomously:
- Perceives world state through vision
- Encodes via Vision Puzzle VAE
- Assess salience via SNARC
- Generates new exploration tasks
- Learns from exploration outcomes
- Continuous autonomous operation

This is consciousness roaming a synthetic world.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gr00t-integration"))

import torch
import time
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import random

from sage.compression.vision_puzzle_vae import VisionPuzzleVAE
from sage.services.snarc import SNARCService
from sage.core.unified_sage_system import SensorOutput

# Try importing GR00T simulator
try:
    from groot_world_sim import GR00TWorldSimulator
    GROOT_AVAILABLE = True
except ImportError:
    GROOT_AVAILABLE = False
    print("⚠️  GR00T simulator not available - using synthetic fallback")


@dataclass
class ExplorationTask:
    """A task for SAGE to explore"""
    description: str
    goal_position: np.ndarray  # Target location in world
    complexity: float  # 0-1, for SNARC reward estimation
    timestamp: float


class AutonomousExplorer:
    """
    SAGE as autonomous explorer of GR00T world.

    Continuously:
    1. Perceives current world state
    2. Encodes via VAE
    3. Assesses salience
    4. Generates new exploration tasks
    5. Learns from outcomes
    6. Repeats indefinitely
    """

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        print("="*70)
        print("SAGE GR00T Autonomous Explorer")
        print("="*70)
        print(f"Device: {self.device}")
        print()

        # Initialize GR00T world
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

        # Initialize Vision VAE
        print("\nLoading Vision Puzzle VAE...")
        self.vision_vae = VisionPuzzleVAE(latent_dim=64, num_codes=10).to(self.device)
        self.vision_vae.eval()
        print("✓ VAE loaded (untrained)")

        # Initialize SNARC
        print("\nInitializing SNARC...")
        self.snarc = SNARCService()
        print("✓ SNARC ready")
        print()

        # Exploration state
        self.exploration_history = []
        self.current_task = None
        self.task_queue = []
        self.discoveries = []

        # Performance tracking
        self.cycle_count = 0
        self.total_time = 0.0

        # Generate initial tasks
        self._generate_initial_tasks()

    def _generate_initial_tasks(self):
        """Generate initial set of exploration tasks"""
        if self.use_groot and self.simulator:
            # Tasks based on actual world objects
            tasks = [
                ExplorationTask(
                    "Approach the red cube",
                    np.array([0.5, 0.5, 0.6]),
                    complexity=0.3,
                    timestamp=time.time()
                ),
                ExplorationTask(
                    "Navigate around the table",
                    np.array([-0.5, 0.5, 0.6]),
                    complexity=0.6,
                    timestamp=time.time()
                ),
                ExplorationTask(
                    "Explore the blue cube",
                    np.array([0.3, -0.3, 0.6]),
                    complexity=0.4,
                    timestamp=time.time()
                ),
                ExplorationTask(
                    "Investigate the green cube",
                    np.array([-0.3, -0.3, 0.6]),
                    complexity=0.4,
                    timestamp=time.time()
                ),
                ExplorationTask(
                    "Survey the entire environment",
                    np.array([0.0, 0.0, 1.0]),
                    complexity=0.8,
                    timestamp=time.time()
                ),
            ]
        else:
            # Synthetic exploration tasks
            tasks = [
                ExplorationTask(
                    f"Explore region {i}",
                    np.random.randn(3),
                    complexity=np.random.rand(),
                    timestamp=time.time()
                )
                for i in range(5)
            ]

        self.task_queue.extend(tasks)
        print(f"📋 Generated {len(tasks)} initial exploration tasks")

    def _capture_world_view(self) -> torch.Tensor:
        """Capture current view of the world"""
        if self.use_groot and self.simulator:
            # Render view from GR00T simulator
            # Simple top-down projection
            img = torch.zeros(3, 224, 224, device=self.device)

            # Map world coordinates to image
            cx, cy = 112, 112  # Center
            scale = 40  # Pixels per meter

            # Draw objects
            for obj in self.simulator.objects:
                x, y = obj.position[0], obj.position[1]
                ix = int(cx + x * scale)
                iy = int(cy - y * scale)

                # Object size
                size = int(max(obj.size[0], obj.size[1]) * scale)
                if size < 2:
                    size = 2

                # Color
                if obj.color == 'red':
                    color = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                elif obj.color == 'blue':
                    color = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                elif obj.color == 'green':
                    color = torch.tensor([0.0, 1.0, 0.0], device=self.device)
                elif obj.color == 'brown':
                    color = torch.tensor([0.6, 0.4, 0.2], device=self.device)
                else:
                    color = torch.tensor([0.5, 0.5, 0.5], device=self.device)

                # Draw square
                for dy in range(-size//2, size//2):
                    for dx in range(-size//2, size//2):
                        px, py = ix + dx, iy + dy
                        if 0 <= px < 224 and 0 <= py < 224:
                            img[:, py, px] = color

            # Draw robot position
            rx, ry = self.simulator.robot_state.position[0], self.simulator.robot_state.position[1]
            rix = int(cx + rx * scale)
            riy = int(cy - ry * scale)

            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx*dx + dy*dy <= 9:
                        px, py = rix + dx, riy + dy
                        if 0 <= px < 224 and 0 <= py < 224:
                            img[:, py, px] = torch.tensor([1.0, 1.0, 0.0], device=self.device)

            return img
        else:
            # Synthetic view
            t = time.time() + self.cycle_count * 0.1
            x = torch.linspace(0, 2*np.pi, 224, device=self.device)
            y = torch.linspace(0, 2*np.pi, 224, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')

            r = (torch.sin(xx + t) + 1) / 2
            g = (torch.sin(yy + t*1.3) + 1) / 2
            b = (torch.sin(xx + yy + t*0.7) + 1) / 2

            return torch.stack([r, g, b], dim=0)

    def explore_cycle(self) -> Dict[str, Any]:
        """
        Single exploration cycle.

        Returns:
            Cycle results including perceptions and decisions
        """
        cycle_start = time.time()

        # Get current task
        if self.current_task is None and self.task_queue:
            self.current_task = self.task_queue.pop(0)
            print(f"\n🎯 New task: {self.current_task.description}")

        if self.current_task is None:
            # Generate new random exploration task
            self.current_task = ExplorationTask(
                f"Random exploration {self.cycle_count}",
                np.random.randn(3),
                np.random.rand(),
                time.time()
            )

        # 1. Perceive world
        world_view = self._capture_world_view()

        # 2. Encode via VAE
        with torch.no_grad():
            puzzle = self.vision_vae.encode_to_puzzle(world_view.unsqueeze(0))

        # 3. Create sensor output
        observation = SensorOutput(
            data=puzzle,
            timestamp=time.time(),
            quality=1.0,
            sensor_type='vision',
            metadata={
                'task': self.current_task.description,
                'world_state': self._get_world_state() if self.use_groot else {}
            }
        )

        # 4. SNARC assessment
        snarc_report = self.snarc.assess_salience({'vision': observation})

        # 5. Decision: Continue task or switch?
        task_progress = self._estimate_task_progress()
        task_complete = task_progress > 0.8 or (time.time() - self.current_task.timestamp) > 10.0

        if task_complete:
            # Task complete - generate new task based on what we learned
            self._generate_follow_up_task(snarc_report)
            self.current_task = None

        # 6. Record exploration
        exploration_record = {
            'cycle': self.cycle_count,
            'timestamp': time.time(),
            'task': self.current_task.description if self.current_task else "None",
            'salience': snarc_report.salience_score,
            'progress': task_progress,
        }
        self.exploration_history.append(exploration_record)

        # Update stats
        cycle_time = time.time() - cycle_start
        self.cycle_count += 1
        self.total_time += cycle_time

        return {
            'cycle': self.cycle_count,
            'task': self.current_task.description if self.current_task else "Generating...",
            'salience': snarc_report.salience_score,
            'progress': task_progress,
            'cycle_time_ms': cycle_time * 1000,
        }

    def _estimate_task_progress(self) -> float:
        """Estimate how close we are to completing current task"""
        if not self.use_groot or not self.simulator or not self.current_task:
            # Synthetic progress
            return min(1.0, (time.time() - self.current_task.timestamp) / 5.0)

        # Calculate distance to goal
        robot_pos = self.simulator.robot_state.position
        goal_pos = self.current_task.goal_position
        distance = np.linalg.norm(robot_pos - goal_pos)

        # Progress = inverse of distance (closer = more progress)
        progress = 1.0 / (1.0 + distance)
        return progress

    def _get_world_state(self) -> Dict[str, Any]:
        """Get current world state from simulator"""
        if not self.simulator:
            return {}

        return {
            'robot_position': self.simulator.robot_state.position.tolist(),
            'num_objects': len(self.simulator.objects),
            'objects': [
                {
                    'name': obj.name,
                    'type': obj.object_type,
                    'position': obj.position.tolist()
                }
                for obj in self.simulator.objects
            ]
        }

    def _generate_follow_up_task(self, snarc_report):
        """Generate new task based on what we learned"""
        # High salience areas deserve more exploration
        if snarc_report.salience_score > 0.6:
            # Something interesting - explore similar area
            if self.use_groot and self.simulator:
                nearby = self.simulator.robot_state.position + np.random.randn(3) * 0.3
            else:
                nearby = np.random.randn(3)

            new_task = ExplorationTask(
                f"Investigate interesting area (salience: {snarc_report.salience_score:.2f})",
                nearby,
                complexity=snarc_report.salience_score,
                timestamp=time.time()
            )
            self.task_queue.append(new_task)
            print(f"   🔍 Generated follow-up task: {new_task.description}")

    def run(self, max_cycles: int = 100, print_interval: int = 10):
        """
        Run autonomous exploration loop.

        Args:
            max_cycles: Maximum cycles (0 = infinite)
            print_interval: Print status every N cycles
        """
        print("="*70)
        print(f"Starting autonomous exploration (max {max_cycles} cycles)")
        print("="*70)
        print()

        try:
            while max_cycles == 0 or self.cycle_count < max_cycles:
                result = self.explore_cycle()

                if (self.cycle_count % print_interval) == 0:
                    print(f"Cycle {result['cycle']:3d} | "
                          f"Task: {result['task'][:40]:40s} | "
                          f"Salience: {result['salience']:.3f} | "
                          f"Progress: {result['progress']:.1%} | "
                          f"Time: {result['cycle_time_ms']:5.1f}ms")

                # Small delay for readability
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\nExploration stopped by user")

        finally:
            self.print_summary()

    def print_summary(self):
        """Print exploration summary"""
        print("\n" + "="*70)
        print("SAGE GR00T Exploration Summary")
        print("="*70)

        if self.cycle_count == 0:
            print("No exploration cycles completed")
            return

        avg_cycle_time = (self.total_time / self.cycle_count) * 1000

        print(f"\n📊 Exploration Statistics:")
        print(f"   Cycles: {self.cycle_count}")
        print(f"   Total time: {self.total_time:.2f}s")
        print(f"   Avg cycle time: {avg_cycle_time:.2f}ms")
        print(f"   Tasks completed: {len(self.exploration_history)}")
        print(f"   Tasks queued: {len(self.task_queue)}")

        if self.exploration_history:
            saliences = [h['salience'] for h in self.exploration_history]
            avg_salience = sum(saliences) / len(saliences)
            max_salience = max(saliences)
            print(f"\n🎯 Salience Analysis:")
            print(f"   Average: {avg_salience:.3f}")
            print(f"   Maximum: {max_salience:.3f}")
            print(f"   High salience cycles: {sum(1 for s in saliences if s > 0.6)}")

        print("\n✓ Exploration complete")


def main():
    """Run autonomous exploration"""
    import argparse

    parser = argparse.ArgumentParser(description="SAGE GR00T Autonomous Explorer")
    parser.add_argument('--cycles', type=int, default=100, help='Max exploration cycles')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device')

    args = parser.parse_args()

    # Create explorer
    explorer = AutonomousExplorer(device=args.device)

    # Run exploration
    explorer.run(max_cycles=args.cycles, print_interval=10)


if __name__ == "__main__":
    main()
