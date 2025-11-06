#!/usr/bin/env python3
"""
Proprioception Sensor - Robot Body Awareness from GR00T State
=============================================================

Captures robot's internal state (proprioception) from GR00T simulator.
Provides body awareness: joint positions, gripper state, velocity, etc.

Proprioception is how robots "feel" their own body - the sixth sense.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gr00t-integration"))

import torch
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

# Try importing GR00T simulator
try:
    from groot_world_sim import GR00TWorldSimulator
    GROOT_AVAILABLE = True
except ImportError:
    GROOT_AVAILABLE = False
    print("⚠️  GR00T simulator not available")


@dataclass
class ProprioceptionFrame:
    """Proprioceptive state - robot's body awareness"""
    # Position and orientation in space
    position: torch.Tensor  # [3] - (x, y, z)
    orientation: float  # radians
    velocity: torch.Tensor  # [3] - (vx, vy, vz)

    # Joint configuration (7-DOF arm)
    joint_angles: torch.Tensor  # [7]

    # End effector
    gripper_state: float  # 0=open, 1=closed

    # Temporal
    timestamp: float

    # Metadata
    metadata: Dict[str, Any]


class ProprioceptionSensor:
    """
    Proprioception sensor - captures robot's internal body state.

    Provides:
    - Joint positions (7-DOF arm)
    - Gripper state (open/closed)
    - Position and orientation in space
    - Velocity

    This is the "kinesthetic sense" - awareness of body position and movement.
    """

    def __init__(
        self,
        simulator: Optional[GR00TWorldSimulator] = None,
        device: str = "cuda",
        normalize: bool = True
    ):
        """
        Initialize proprioception sensor.

        Args:
            simulator: GR00T world simulator (provides robot state)
            device: Device for tensor operations
            normalize: Whether to normalize joint angles and positions
        """
        self.device = device
        self.normalize = normalize

        print("🤖 Initializing Proprioception sensor...")

        # Connect to GR00T simulator
        if simulator is not None:
            self.simulator = simulator
            self.available = True
            print("   ✓ Connected to GR00T simulator")
        elif GROOT_AVAILABLE:
            try:
                self.simulator = GR00TWorldSimulator(device=device)
                self.available = True
                print("   ✓ GR00T simulator loaded")
            except Exception as e:
                print(f"   ⚠️  Could not load GR00T: {e}")
                self.simulator = None
                self.available = False
        else:
            self.simulator = None
            self.available = False
            print("   ⚠️  Using synthetic fallback")

        # Normalization ranges (for real robots these would be joint limits)
        self.joint_ranges = {
            'min': np.array([-np.pi, -np.pi/2, -np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi]),
            'max': np.array([np.pi, np.pi/2, np.pi, np.pi, np.pi, np.pi/2, np.pi])
        }

        self.position_ranges = {
            'min': np.array([-2.0, -2.0, 0.0]),
            'max': np.array([2.0, 2.0, 2.0])
        }

        # Performance tracking
        self._frame_count = 0
        self._total_latency = 0.0

    def capture(self) -> Optional[ProprioceptionFrame]:
        """
        Capture current proprioceptive state.

        Returns:
            ProprioceptionFrame with robot's internal state
        """
        start_time = time.time()

        if not self.available or self.simulator is None:
            # Fallback to synthetic proprioception
            return self._capture_fallback()

        try:
            # Get robot state from simulator
            robot = self.simulator.robot_state

            # Convert to tensors
            position = torch.tensor(robot.position, dtype=torch.float32, device=self.device)
            velocity = torch.tensor(robot.velocity, dtype=torch.float32, device=self.device)
            joint_angles = torch.tensor(robot.joint_angles, dtype=torch.float32, device=self.device)

            # Normalize if requested
            if self.normalize:
                position = self._normalize_position(position)
                joint_angles = self._normalize_joints(joint_angles)

            # Track performance
            latency = time.time() - start_time
            self._frame_count += 1
            self._total_latency += latency

            return ProprioceptionFrame(
                position=position,
                orientation=robot.orientation,
                velocity=velocity,
                joint_angles=joint_angles,
                gripper_state=robot.gripper_state,
                timestamp=time.time(),
                metadata={
                    'latency_ms': latency * 1000,
                    'frame_count': self._frame_count,
                    'normalized': self.normalize,
                    'source': 'groot_simulator'
                }
            )

        except Exception as e:
            print(f"⚠️  Proprioception capture failed: {e}")
            return self._capture_fallback()

    def _normalize_position(self, position: torch.Tensor) -> torch.Tensor:
        """Normalize position to [-1, 1] range"""
        pos_min = torch.tensor(self.position_ranges['min'], device=self.device)
        pos_max = torch.tensor(self.position_ranges['max'], device=self.device)
        return 2 * (position - pos_min) / (pos_max - pos_min) - 1

    def _normalize_joints(self, joints: torch.Tensor) -> torch.Tensor:
        """Normalize joint angles to [-1, 1] range"""
        joint_min = torch.tensor(self.joint_ranges['min'], device=self.device)
        joint_max = torch.tensor(self.joint_ranges['max'], device=self.device)
        return 2 * (joints - joint_min) / (joint_max - joint_min) - 1

    def _capture_fallback(self) -> Optional[ProprioceptionFrame]:
        """Fallback synthetic proprioception"""
        t = time.time()

        # Synthetic robot state (slowly moving)
        position = torch.tensor([
            0.3 * np.sin(t * 0.5),
            0.3 * np.cos(t * 0.5),
            0.6
        ], dtype=torch.float32, device=self.device)

        velocity = torch.tensor([
            0.15 * np.cos(t * 0.5),
            -0.15 * np.sin(t * 0.5),
            0.0
        ], dtype=torch.float32, device=self.device)

        # Synthetic joint configuration (slowly varying)
        joint_angles = torch.tensor([
            0.2 * np.sin(t * 0.3),
            0.1 * np.cos(t * 0.4),
            -0.3 * np.sin(t * 0.2),
            0.4 * np.cos(t * 0.5),
            0.1 * np.sin(t * 0.6),
            -0.2 * np.cos(t * 0.3),
            0.15 * np.sin(t * 0.4)
        ], dtype=torch.float32, device=self.device)

        # Gripper slowly opening and closing
        gripper = (np.sin(t * 0.2) + 1) / 2  # 0 to 1

        return ProprioceptionFrame(
            position=position,
            orientation=t * 0.1 % (2 * np.pi),
            velocity=velocity,
            joint_angles=joint_angles,
            gripper_state=gripper,
            timestamp=t,
            metadata={'mode': 'fallback_synthetic'}
        )

    def to_vector(self, frame: ProprioceptionFrame) -> torch.Tensor:
        """
        Convert proprioception frame to flat feature vector.

        Useful for encoding into puzzle space or feeding to neural networks.

        Returns:
            Tensor of shape [14]: [position(3), velocity(3), joints(7), gripper(1)]
        """
        return torch.cat([
            frame.position,  # 3
            frame.velocity,  # 3
            frame.joint_angles,  # 7
            torch.tensor([frame.gripper_state], device=self.device)  # 1
        ])  # Total: 14 dimensions

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self._frame_count == 0:
            return {
                'frames_captured': 0,
                'avg_latency_ms': 0.0,
                'available': self.available,
            }

        avg_latency = self._total_latency / self._frame_count

        return {
            'frames_captured': self._frame_count,
            'avg_latency_ms': avg_latency * 1000,
            'available': self.available,
            'normalized': self.normalize,
        }

    def close(self):
        """Cleanup resources"""
        print("✓ Proprioception sensor closed")


if __name__ == "__main__":
    # Test proprioception sensor
    print("Testing Proprioception Sensor\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sensor = ProprioceptionSensor(device=device, normalize=True)

    print(f"\nCapturing 10 proprioception frames...")
    for i in range(10):
        frame = sensor.capture()
        if frame is not None:
            vec = sensor.to_vector(frame)
            print(f"Frame {i+1}:")
            print(f"   Position: {frame.position.cpu().numpy()}")
            print(f"   Orientation: {frame.orientation:.3f} rad")
            print(f"   Gripper: {'CLOSED' if frame.gripper_state > 0.5 else 'OPEN'} ({frame.gripper_state:.2f})")
            print(f"   Joint angles: {frame.joint_angles.cpu().numpy()}")
            print(f"   Feature vector shape: {vec.shape}")
            print(f"   Latency: {frame.metadata.get('latency_ms', 0):.2f}ms\n")

        time.sleep(0.5)

    stats = sensor.get_stats()
    print(f"📊 Performance Stats:")
    print(f"   Available: {stats['available']}")
    print(f"   Frames: {stats['frames_captured']}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")

    sensor.close()
