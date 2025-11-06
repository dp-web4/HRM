#!/usr/bin/env python3
"""
Camera Sensor - Multi-Backend Vision Input
===========================================

Provides vision input with multiple backend support:
1. Real camera (OpenCV) - For Jetson Nano with physical camera
2. GR00T integration - For Thor with Isaac simulation
3. Synthetic fallback - For testing without hardware

All backends produce standardized output encoded via VisionPuzzleVAE.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

# Try importing backends (optional dependencies)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Check for GR00T
try:
    # GR00T would be imported from isaac-gr00t workspace
    # For now, we'll detect its presence
    import os
    GROOT_AVAILABLE = os.path.exists('/home/dp/ai-workspace/isaac-gr00t')
except Exception:
    GROOT_AVAILABLE = False


class CameraBackend(Enum):
    """Available camera backends"""
    AUTO = "auto"           # Auto-detect best available
    OPENCV = "opencv"       # Real camera via OpenCV
    GROOT = "groot"         # GR00T simulation
    SYNTHETIC = "synthetic"  # Generated test data


@dataclass
class CameraFrame:
    """Raw camera frame data"""
    image: torch.Tensor  # [C, H, W] RGB tensor
    timestamp: float
    backend: CameraBackend
    metadata: Dict[str, Any]


class CameraSensor:
    """
    Multi-backend camera sensor for SAGE vision input.

    Automatically selects best available backend:
    - OpenCV if camera available
    - GR00T if Isaac workspace detected
    - Synthetic as fallback

    All outputs standardized to [3, 224, 224] RGB tensors.
    """

    def __init__(
        self,
        backend: str = "auto",
        device: str = "cuda",
        target_size: Tuple[int, int] = (224, 224),
        camera_id: int = 0,
        fps: int = 30,
    ):
        """
        Initialize camera sensor.

        Args:
            backend: Backend to use ('auto', 'opencv', 'groot', 'synthetic')
            device: Device for tensor operations
            target_size: Output image size (height, width)
            camera_id: Camera device ID for OpenCV
            fps: Target frames per second
        """
        self.device = device
        self.target_size = target_size
        self.camera_id = camera_id
        self.fps = fps
        self.frame_interval = 1.0 / fps

        # Select backend
        self.backend = self._select_backend(backend)
        print(f"📷 Camera sensor initialized with backend: {self.backend.value}")

        # Initialize backend
        self._cap = None
        self._groot_sim = None
        self._init_backend()

        # Performance tracking
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._total_latency = 0.0

    def _select_backend(self, backend: str) -> CameraBackend:
        """Select best available backend"""
        if backend == "auto":
            # Auto-detect best available
            if OPENCV_AVAILABLE:
                # Try to open camera
                cap = cv2.VideoCapture(self.camera_id)
                if cap.isOpened():
                    cap.release()
                    return CameraBackend.OPENCV

            if GROOT_AVAILABLE:
                return CameraBackend.GROOT

            print("⚠️  No camera or GR00T found, using synthetic mode")
            return CameraBackend.SYNTHETIC

        # Use specified backend
        backend_map = {
            "opencv": CameraBackend.OPENCV,
            "groot": CameraBackend.GROOT,
            "synthetic": CameraBackend.SYNTHETIC,
        }
        return backend_map.get(backend, CameraBackend.SYNTHETIC)

    def _init_backend(self):
        """Initialize selected backend"""
        if self.backend == CameraBackend.OPENCV:
            if not OPENCV_AVAILABLE:
                raise RuntimeError("OpenCV not available")
            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            print(f"✓ OpenCV camera {self.camera_id} opened")

        elif self.backend == CameraBackend.GROOT:
            # Initialize GR00T connection
            # TODO: Implement GR00T integration when ready
            print("✓ GR00T simulation mode (placeholder)")

        elif self.backend == CameraBackend.SYNTHETIC:
            print("✓ Synthetic camera mode")

    def capture(self) -> Optional[CameraFrame]:
        """
        Capture single frame from camera.

        Returns:
            CameraFrame with [3, 224, 224] RGB tensor, or None if failed
        """
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - self._last_frame_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)

        start_time = time.time()

        # Capture from backend
        if self.backend == CameraBackend.OPENCV:
            frame = self._capture_opencv()
        elif self.backend == CameraBackend.GROOT:
            frame = self._capture_groot()
        else:
            frame = self._capture_synthetic()

        if frame is None:
            return None

        # Track performance
        latency = time.time() - start_time
        self._last_frame_time = time.time()
        self._frame_count += 1
        self._total_latency += latency

        frame.metadata['latency_ms'] = latency * 1000
        frame.metadata['frame_count'] = self._frame_count

        return frame

    def _capture_opencv(self) -> Optional[CameraFrame]:
        """Capture frame from OpenCV camera"""
        ret, frame = self._cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to target size
        frame = cv2.resize(frame, self.target_size)

        # Convert to tensor [3, 224, 224]
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame = frame.to(self.device)

        return CameraFrame(
            image=frame,
            timestamp=time.time(),
            backend=self.backend,
            metadata={
                'camera_id': self.camera_id,
                'resolution': self.target_size,
            }
        )

    def _capture_groot(self) -> Optional[CameraFrame]:
        """Capture frame from GR00T simulation"""
        # TODO: Implement GR00T integration
        # For now, use synthetic
        return self._capture_synthetic()

    def _capture_synthetic(self) -> Optional[CameraFrame]:
        """Generate synthetic frame for testing"""
        # Create synthetic image with temporal variation
        t = time.time()

        # Moving gradient pattern
        x = torch.linspace(0, 2*np.pi, self.target_size[1], device=self.device)
        y = torch.linspace(0, 2*np.pi, self.target_size[0], device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        # RGB channels with different temporal patterns
        r = (torch.sin(xx + t) + 1) / 2
        g = (torch.sin(yy + t*1.3) + 1) / 2
        b = (torch.sin(xx + yy + t*0.7) + 1) / 2

        frame = torch.stack([r, g, b], dim=0)

        return CameraFrame(
            image=frame,
            timestamp=t,
            backend=self.backend,
            metadata={
                'synthetic': True,
                'pattern': 'moving_gradient',
            }
        )

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self._frame_count == 0:
            return {
                'avg_latency_ms': 0.0,
                'avg_fps': 0.0,
                'frames_captured': 0,
            }

        avg_latency = self._total_latency / self._frame_count
        avg_fps = 1.0 / avg_latency if avg_latency > 0 else 0.0

        return {
            'avg_latency_ms': avg_latency * 1000,
            'avg_fps': avg_fps,
            'frames_captured': self._frame_count,
            'backend': self.backend.value,
        }

    def close(self):
        """Release camera resources"""
        if self._cap is not None:
            self._cap.release()
            print("✓ Camera released")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


if __name__ == "__main__":
    # Test camera sensor
    print("Testing Camera Sensor\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    camera = CameraSensor(backend="auto", device=device, fps=10)

    print(f"\nCapturing 10 test frames...")
    for i in range(10):
        frame = camera.capture()
        if frame is not None:
            print(f"Frame {i+1}: {frame.image.shape}, latency: {frame.metadata.get('latency_ms', 0):.2f}ms")

    stats = camera.get_stats()
    print(f"\n📊 Performance Stats:")
    print(f"   Backend: {stats['backend']}")
    print(f"   Frames: {stats['frames_captured']}")
    print(f"   Avg FPS: {stats['avg_fps']:.1f}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")

    camera.close()
