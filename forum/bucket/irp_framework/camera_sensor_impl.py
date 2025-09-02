#!/usr/bin/env python3
"""
Camera Sensor IRP Plugins
Dual CSI camera sensors as IRP plugins with GPU mailbox communication
Based on working implementation from ai-dna-discovery
"""

import cv2
import torch
import numpy as np
from typing import Any, Dict, Tuple, Optional
import threading
import queue
import time

from sage.irp.base import IRPPlugin
from sage.mailbox.pubsub_mailbox import PubSubMailbox, Message


class CameraSensorIRP(IRPPlugin):
    """
    Camera sensor as IRP plugin
    Captures from CSI or USB camera and publishes via GPU mailbox
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.config = config or {}
        self.sensor_id = self.config.get('sensor_id', 0)
        self.camera_type = self.config.get('type', 'csi')  # 'csi' or 'usb'
        self.resolution = self.config.get('resolution', (1920, 1080))
        self.display_resolution = self.config.get('display_resolution', (960, 540))
        self.fps = self.config.get('fps', 30)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set entity ID based on camera
        self.entity_id = f"camera_{self.camera_type}_{self.sensor_id}"
        
        # Mailbox for publishing
        self.mailbox = self.config.get('mailbox')
        self.publish_topic = f"camera/{self.camera_type}/{self.sensor_id}"
        
        # Camera capture
        self.cap = None
        self.capture_thread = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.latest_frame = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize(self):
        """Initialize camera capture"""
        print(f"Initializing {self.entity_id}: {self.resolution} @ {self.fps}fps")
        
        if self.camera_type == 'csi':
            # GStreamer pipeline for CSI camera (from ai-dna-discovery)
            pipeline = self._create_gst_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # USB camera
            self.cap = cv2.VideoCapture(self.sensor_id)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open {self.entity_id}")
            
        print(f"✓ {self.entity_id} initialized")
        
        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def _create_gst_pipeline(self) -> str:
        """Create GStreamer pipeline for CSI camera"""
        width, height = self.resolution
        disp_width, disp_height = self.display_resolution
        
        # Use sensor-mode=2 for 1920x1080 @ 30fps (from ai-dna-discovery)
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} sensor-mode=2 ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw, width={disp_width}, "
            f"height={disp_height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        
    def _capture_loop(self):
        """Background thread for continuous capture"""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.latest_frame = frame
                    self.frame_count += 1
                    
                    # Put in queue for processing
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Drop old frame
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                            
    def refine(self, x: Any = None, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Get camera frame and publish to mailbox
        
        Args:
            x: Ignored - camera provides its own input
            early_stop: Not used for sensors
            
        Returns:
            Tuple of (frame, telemetry)
        """
        start_time = time.time()
        
        # Get latest frame
        try:
            frame = self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            frame = self.latest_frame
            
        if frame is None:
            # Return black frame if no capture yet
            frame = np.zeros((*self.display_resolution[::-1], 3), dtype=np.uint8)
            
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).to(self.device)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Publish to mailbox if available
        if self.mailbox:
            metadata = {
                'sensor_id': self.sensor_id,
                'camera_type': self.camera_type,
                'resolution': frame.shape[:2],
                'frame_count': self.frame_count
            }
            self.mailbox.publish(
                self.publish_topic,
                self.entity_id,
                frame,
                metadata
            )
            
        # Calculate telemetry
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        telemetry = {
            'iterations': 1,  # Sensors don't iterate
            'compute_saved': 0.0,  # No compute saved
            'trust': 1.0,  # Sensors are always trusted
            'capture_fps': fps,
            'frame_count': self.frame_count,
            'process_time_ms': (time.time() - start_time) * 1000
        }
        
        return frame_tensor, telemetry
        
    def cleanup(self):
        """Clean up camera resources"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            
        print(f"{self.entity_id} shutdown complete")
        

class DualCameraSensorIRP(IRPPlugin):
    """
    Dual camera sensor plugin that manages both CSI cameras
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.entity_id = "dual_camera_sensor"
        
        # Configuration
        self.config = config or {}
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Use provided mailbox or create new one
        self.mailbox = self.config.get('mailbox')
        if self.mailbox is None:
            self.mailbox = PubSubMailbox(self.device)
        
        # Create individual camera sensors
        self.cameras = []
        
        # CSI Camera 0 (left)
        left_config = {
            'sensor_id': 0,
            'type': 'csi',
            'resolution': (1920, 1080),
            'display_resolution': (960, 540),
            'fps': 30,
            'device': self.device,
            'mailbox': self.mailbox
        }
        self.left_camera = CameraSensorIRP(left_config)
        self.cameras.append(self.left_camera)
        
        # CSI Camera 1 (right)
        right_config = {
            'sensor_id': 1,
            'type': 'csi',
            'resolution': (1920, 1080),
            'display_resolution': (960, 540),
            'fps': 30,
            'device': self.device,
            'mailbox': self.mailbox
        }
        self.right_camera = CameraSensorIRP(right_config)
        self.cameras.append(self.right_camera)
        
        # Latest frames
        self.latest_frames = [None, None]
        
        # Subscribe to camera topics
        self.mailbox.subscribe(
            "camera/csi/0",
            "dual_sensor_left",
            lambda msg: self._store_frame(0, msg)
        )
        self.mailbox.subscribe(
            "camera/csi/1", 
            "dual_sensor_right",
            lambda msg: self._store_frame(1, msg)
        )
        
    def initialize(self):
        """Initialize both cameras"""
        print("Initializing dual camera sensor...")
        
        for camera in self.cameras:
            try:
                camera.initialize()
            except Exception as e:
                print(f"Failed to initialize {camera.entity_id}: {e}")
                
    def _store_frame(self, index: int, message: Message):
        """Store frame from mailbox message"""
        self.latest_frames[index] = message.data
        
    def refine(self, x: Any = None, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Get frames from both cameras
        
        Returns:
            Tuple of (stereo_pair, telemetry)
        """
        start_time = time.time()
        
        # Get frames from both cameras
        frames = []
        telemetries = []
        
        for camera in self.cameras:
            frame, telemetry = camera.refine()
            frames.append(frame)
            telemetries.append(telemetry)
            
        # Stack frames for stereo processing
        if all(f is not None for f in frames):
            stereo_pair = torch.cat(frames, dim=0)  # Shape: [2, C, H, W]
        else:
            # Return black frames if no capture
            h, w = self.cameras[0].display_resolution[::-1]
            stereo_pair = torch.zeros(2, 3, h, w, device=self.device)
            
        # Combine telemetry
        combined_telemetry = {
            'iterations': 1,
            'compute_saved': 0.0,
            'trust': 1.0,
            'left_fps': telemetries[0].get('capture_fps', 0),
            'right_fps': telemetries[1].get('capture_fps', 0) if len(telemetries) > 1 else 0,
            'process_time_ms': (time.time() - start_time) * 1000
        }
        
        # Publish stereo pair
        self.mailbox.publish(
            "camera/stereo",
            self.entity_id,
            stereo_pair,
            {'type': 'stereo_pair'}
        )
        
        return stereo_pair, combined_telemetry
        
    def cleanup(self):
        """Clean up all cameras"""
        for camera in self.cameras:
            camera.cleanup()
            
        self.mailbox.shutdown()
        

def create_camera_sensor(
    sensor_id: int = 0,
    camera_type: str = 'csi',
    mailbox: Optional[PubSubMailbox] = None,
    **kwargs
) -> CameraSensorIRP:
    """
    Factory function to create camera sensor
    
    Args:
        sensor_id: Camera index
        camera_type: 'csi' or 'usb'
        mailbox: PubSub mailbox for communication
        **kwargs: Additional configuration
        
    Returns:
        CameraSensorIRP instance
    """
    config = {
        'sensor_id': sensor_id,
        'type': camera_type,
        'mailbox': mailbox,
        **kwargs
    }
    
    sensor = CameraSensorIRP(config)
    sensor.initialize()
    return sensor
    

def create_dual_camera_sensor(**kwargs) -> DualCameraSensorIRP:
    """
    Factory function to create dual camera sensor
    
    Returns:
        DualCameraSensorIRP instance
    """
    sensor = DualCameraSensorIRP(kwargs)
    sensor.initialize()
    return sensor


# Test function
def test_camera_sensors():
    """Test camera sensor plugins"""
    print("Testing Camera Sensor IRP Plugins...")
    
    # Create dual camera sensor
    try:
        dual_sensor = create_dual_camera_sensor()
        
        print("\nCapturing frames...")
        for i in range(10):
            stereo, telemetry = dual_sensor.refine()
            print(f"Frame {i}: Shape={stereo.shape}, "
                  f"Left FPS={telemetry.get('left_fps', 0):.1f}, "
                  f"Right FPS={telemetry.get('right_fps', 0):.1f}")
            time.sleep(0.1)
            
        dual_sensor.cleanup()
        print("\nTest complete!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        
        # Fallback to simulated frames
        print("\nTesting with simulated frames...")
        mailbox = PubSubMailbox()
        
        # Create simulated sensor
        class SimulatedCameraSensor(CameraSensorIRP):
            def _capture_loop(self):
                """Generate simulated frames"""
                while self.running:
                    # Create test pattern
                    frame = np.random.randint(0, 255, 
                                             (*self.display_resolution[::-1], 3), 
                                             dtype=np.uint8)
                    self.latest_frame = frame
                    self.frame_count += 1
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                        
                    time.sleep(1.0 / self.fps)
                    
        # Test simulated sensor
        config = {
            'sensor_id': 0,
            'type': 'simulated',
            'mailbox': mailbox,
            'display_resolution': (640, 480)
        }
        
        sim_sensor = SimulatedCameraSensor(config)
        sim_sensor.initialize()
        
        for i in range(5):
            frame, telemetry = sim_sensor.refine()
            print(f"Simulated frame {i}: {frame.shape}")
            
        sim_sensor.cleanup()
        mailbox.shutdown()
        

if __name__ == "__main__":
    test_camera_sensors()