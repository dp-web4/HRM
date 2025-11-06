#!/usr/bin/env python3
"""
GR00T Camera Sensor - Vision from GR00T Simulated World
========================================================

Captures vision from GR00T's simulated 3D environment.
Provides same interface as CameraSensor but pulls from GR00T world sim.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add gr00t-integration to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gr00t-integration"))

import torch
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

# Import GR00T world simulator
try:
    from groot_world_sim import GR00TWorldSimulator, load_groot_model
    GROOT_AVAILABLE = True
except ImportError:
    GROOT_AVAILABLE = False
    print("⚠️  GR00T world simulator not available")


@dataclass
class GR00TCameraFrame:
    """Frame from GR00T simulation"""
    image: torch.Tensor  # [C, H, W] RGB tensor
    timestamp: float
    world_state: Dict[str, Any]
    vision_features: Optional[torch.Tensor]
    metadata: Dict[str, Any]


class GR00TCameraSensor:
    """
    Camera sensor that captures from GR00T simulated world.

    Provides:
    - Rendered RGB images from GR00T's world model
    - Vision features from GR00T's Eagle VLM
    - World state (objects, robot position, etc.)
    - Attention maps showing what GR00T is focusing on
    """

    def __init__(
        self,
        device: str = "cuda",
        target_size: tuple = (224, 224),
        render_mode: str = "synthetic",  # 'synthetic' or 'features'
    ):
        """
        Initialize GR00T camera sensor.

        Args:
            device: Device for tensor operations
            target_size: Output image size (height, width)
            render_mode: How to generate images
                - 'synthetic': Generate synthetic view of world
                - 'features': Visualize GR00T's internal feature maps
        """
        self.device = device
        self.target_size = target_size
        self.render_mode = render_mode

        print("🤖 Initializing GR00T camera sensor...")

        # Initialize GR00T world simulator
        if not GROOT_AVAILABLE:
            print("   ⚠️  GR00T not available, using fallback")
            self.simulator = None
            self.available = False
        else:
            try:
                self.simulator = GR00TWorldSimulator(device=device)
                print("   ✓ GR00T simulator loaded")
                self.available = True
            except Exception as e:
                print(f"   ⚠️  Could not load GR00T: {e}")
                print("   Using fallback synthetic mode")
                self.simulator = None
                self.available = False

        # Performance tracking
        self._frame_count = 0
        self._total_latency = 0.0
        self._last_command = None

    def set_task(self, command: str):
        """Set a task/command for the robot to execute"""
        self._last_command = command
        if self.simulator and self.available:
            print(f"   📝 New task: {command}")

    def capture(self) -> Optional[GR00TCameraFrame]:
        """
        Capture frame from GR00T world.

        Returns:
            GR00TCameraFrame with rendered image and world state
        """
        start_time = time.time()

        if not self.available or self.simulator is None:
            # Fallback to synthetic
            return self._capture_fallback()

        try:
            # Run GR00T perception if we have a command
            if self._last_command:
                self.simulator.simulate_perception(self._last_command)

            # Generate image based on render mode
            if self.render_mode == 'synthetic':
                image = self._render_synthetic_view()
            else:
                image = self._render_feature_view()

            # Get world state
            world_state = {
                'objects': [
                    {
                        'name': obj.name,
                        'position': obj.position.tolist(),
                        'type': obj.object_type,
                        'confidence': obj.confidence
                    }
                    for obj in self.simulator.objects
                ],
                'robot': {
                    'position': self.simulator.robot_state.position.tolist(),
                    'orientation': self.simulator.robot_state.orientation,
                    'gripper': self.simulator.robot_state.gripper_state,
                },
                'trajectory_plan': [
                    wp.tolist() for wp in self.simulator.trajectory_plan
                ]
            }

            # Track performance
            latency = time.time() - start_time
            self._frame_count += 1
            self._total_latency += latency

            return GR00TCameraFrame(
                image=image,
                timestamp=time.time(),
                world_state=world_state,
                vision_features=self.simulator.vision_features,
                metadata={
                    'render_mode': self.render_mode,
                    'latency_ms': latency * 1000,
                    'frame_count': self._frame_count,
                    'task': self._last_command,
                }
            )

        except Exception as e:
            print(f"⚠️  GR00T capture failed: {e}")
            return self._capture_fallback()

    def _render_synthetic_view(self) -> torch.Tensor:
        """
        Render synthetic RGB view of GR00T world.

        Creates a 2D projection of the 3D world from robot's perspective.
        """
        # Create canvas
        img = torch.zeros(3, *self.target_size, device=self.device)

        # Simple top-down projection for now
        # Map world coordinates to image coordinates
        cx, cy = self.target_size[1] // 2, self.target_size[0] // 2
        scale = self.target_size[0] // 4

        # Draw objects as colored blocks
        for obj in self.simulator.objects:
            x, y = obj.position[0], obj.position[1]

            # Convert to image coordinates
            ix = int(cx + x * scale)
            iy = int(cy - y * scale)  # Flip y for image coords

            # Draw block
            size = int(obj.size[0] * scale)
            color = self._color_to_rgb(obj.color)

            # Set pixels (with bounds checking)
            for dy in range(-size//2, size//2):
                for dx in range(-size//2, size//2):
                    px, py = ix + dx, iy + dy
                    if 0 <= px < self.target_size[1] and 0 <= py < self.target_size[0]:
                        img[:, py, px] = torch.tensor(color, device=self.device)

        # Draw robot as a circle
        rx, ry = self.simulator.robot_state.position[0], self.simulator.robot_state.position[1]
        rix = int(cx + rx * scale)
        riy = int(cy - ry * scale)

        # Draw robot indicator
        robot_radius = 5
        for dy in range(-robot_radius, robot_radius):
            for dx in range(-robot_radius, robot_radius):
                if dx*dx + dy*dy <= robot_radius*robot_radius:
                    px, py = rix + dx, riy + dy
                    if 0 <= px < self.target_size[1] and 0 <= py < self.target_size[0]:
                        img[:, py, px] = torch.tensor([1.0, 1.0, 0.0], device=self.device)  # Yellow

        return img

    def _render_feature_view(self) -> torch.Tensor:
        """Visualize GR00T's internal feature maps"""
        # If we have vision features from GR00T, visualize them
        if self.simulator.vision_features is not None:
            features = self.simulator.vision_features
            # Convert features to RGB visualization
            # Take first 3 channels and normalize
            vis = features[0, :3, :, :] if features.dim() == 4 else features[:3, :, :]
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-6)

            # Resize to target size
            vis = torch.nn.functional.interpolate(
                vis.unsqueeze(0),
                size=self.target_size,
                mode='bilinear'
            ).squeeze(0)

            return vis
        else:
            # Fall back to synthetic view
            return self._render_synthetic_view()

    def _color_to_rgb(self, color_name: str) -> list:
        """Convert color name to RGB values"""
        colors = {
            'red': [1.0, 0.0, 0.0],
            'blue': [0.0, 0.0, 1.0],
            'green': [0.0, 1.0, 0.0],
            'brown': [0.6, 0.4, 0.2],
            'gray': [0.5, 0.5, 0.5],
            'yellow': [1.0, 1.0, 0.0],
            'orange': [1.0, 0.5, 0.0],
        }
        return colors.get(color_name, [0.5, 0.5, 0.5])

    def _capture_fallback(self) -> Optional[GR00TCameraFrame]:
        """Fallback synthetic image when GR00T unavailable"""
        t = time.time()

        # Simple gradient pattern
        x = torch.linspace(0, 1, self.target_size[1], device=self.device)
        y = torch.linspace(0, 1, self.target_size[0], device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        img = torch.stack([
            xx,
            yy,
            torch.sin(xx * np.pi + t) * torch.cos(yy * np.pi + t)
        ], dim=0)

        return GR00TCameraFrame(
            image=img,
            timestamp=t,
            world_state={'fallback': True},
            vision_features=None,
            metadata={'mode': 'fallback'}
        )

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
            'render_mode': self.render_mode,
            'current_task': self._last_command,
        }

    def close(self):
        """Cleanup resources"""
        if self.simulator:
            print("✓ GR00T simulator closed")


if __name__ == "__main__":
    # Test GR00T camera sensor
    print("Testing GR00T Camera Sensor\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    camera = GR00TCameraSensor(device=device, render_mode="synthetic")

    # Set a task
    camera.set_task("Pick up the red cube and place it at the goal")

    print(f"\nCapturing 5 test frames...")
    for i in range(5):
        frame = camera.capture()
        if frame is not None:
            print(f"Frame {i+1}: {frame.image.shape}")
            print(f"   Objects in world: {len(frame.world_state.get('objects', []))}")
            print(f"   Robot position: {frame.world_state.get('robot', {}).get('position', 'N/A')}")
            print(f"   Latency: {frame.metadata.get('latency_ms', 0):.2f}ms")

    stats = camera.get_stats()
    print(f"\n📊 Performance Stats:")
    print(f"   Available: {stats['available']}")
    print(f"   Frames: {stats['frames_captured']}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")

    camera.close()
