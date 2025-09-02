#!/usr/bin/env python3
"""
Visual Monitor Effector IRP Plugin
Subscribes to camera feeds and attention maps via GPU mailbox
Displays integrated view with attention overlay
"""

import cv2
import torch
import numpy as np
from typing import Any, Dict, Tuple, Optional, List
import time
from collections import deque
import threading

from sage.irp.base import IRPPlugin
from sage.mailbox.pubsub_mailbox import PubSubMailbox, Message


class VisualMonitorEffector(IRPPlugin):
    """
    Visual monitor as effector plugin
    Receives camera feeds and attention maps via mailbox subscription
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.entity_id = "visual_monitor_effector"
        
        # Configuration
        self.config = config or {}
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.display_width = self.config.get('display_width', 1920)
        self.display_height = self.config.get('display_height', 540)
        self.show_window = self.config.get('show_window', True)
        
        # Mailbox for subscriptions
        self.mailbox = self.config.get('mailbox')
        if not self.mailbox:
            self.mailbox = PubSubMailbox(self.device)
            
        # Frame storage
        self.left_frame = None
        self.right_frame = None
        self.attention_left = None
        self.attention_right = None
        self.lock = threading.Lock()
        
        # Display
        self.window_name = "SAGE Dual Camera Monitor"
        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_display_time = time.time()
        self.frame_count = 0
        
        # Display options
        self.show_attention = True
        self.show_telemetry = True
        
        # Subscribe to topics
        self._setup_subscriptions()
        
    def _setup_subscriptions(self):
        """Setup mailbox subscriptions"""
        # Subscribe to camera feeds
        self.mailbox.subscribe(
            "camera/csi/0",
            f"{self.entity_id}_left_cam",
            self._receive_left_camera
        )
        self.mailbox.subscribe(
            "camera/csi/1",
            f"{self.entity_id}_right_cam",
            self._receive_right_camera
        )
        
        # Subscribe to attention maps
        self.mailbox.subscribe(
            "attention/left",
            f"{self.entity_id}_left_att",
            self._receive_left_attention
        )
        self.mailbox.subscribe(
            "attention/right",
            f"{self.entity_id}_right_att",
            self._receive_right_attention
        )
        
        # Subscribe to stereo pair
        self.mailbox.subscribe(
            "camera/stereo",
            f"{self.entity_id}_stereo",
            self._receive_stereo
        )
        
    def _receive_left_camera(self, message: Message):
        """Receive left camera frame"""
        with self.lock:
            if isinstance(message.data, torch.Tensor):
                self.left_frame = message.data.cpu().numpy()
            else:
                self.left_frame = message.data
                
    def _receive_right_camera(self, message: Message):
        """Receive right camera frame"""
        with self.lock:
            if isinstance(message.data, torch.Tensor):
                self.right_frame = message.data.cpu().numpy()
            else:
                self.right_frame = message.data
                
    def _receive_left_attention(self, message: Message):
        """Receive left attention map"""
        with self.lock:
            if isinstance(message.data, torch.Tensor):
                self.attention_left = message.data.squeeze().cpu().numpy()
            else:
                self.attention_left = message.data
                
    def _receive_right_attention(self, message: Message):
        """Receive right attention map"""
        with self.lock:
            if isinstance(message.data, torch.Tensor):
                self.attention_right = message.data.squeeze().cpu().numpy()
            else:
                self.attention_right = message.data
                
    def _receive_stereo(self, message: Message):
        """Receive stereo pair"""
        # Could process stereo depth here
        pass
        
    def refine(self, x: Any = None, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Display current state
        
        Args:
            x: Ignored - monitor uses subscribed data
            early_stop: Not used
            
        Returns:
            Tuple of (display_frame, telemetry)
        """
        start_time = time.time()
        
        # Create display frame
        display_frame = self._create_display()
        
        # Show if window enabled
        if self.show_window and display_frame is not None:
            cv2.imshow(self.window_name, display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.show_window = False
            elif key == ord('a'):
                self.show_attention = not self.show_attention
            elif key == ord('t'):
                self.show_telemetry = not self.show_telemetry
                
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_display_time + 1e-6)
        self.last_display_time = current_time
        self.fps_buffer.append(fps)
        
        # Create telemetry
        telemetry = {
            'iterations': 1,
            'compute_saved': 0.0,
            'trust': 1.0,
            'display_fps': np.mean(self.fps_buffer) if self.fps_buffer else 0,
            'has_left': self.left_frame is not None,
            'has_right': self.right_frame is not None,
            'process_time_ms': (time.time() - start_time) * 1000
        }
        
        return display_frame, telemetry
        
    def _create_display(self) -> Optional[np.ndarray]:
        """Create display frame with both cameras and attention"""
        with self.lock:
            frames = []
            
            # Process left camera
            if self.left_frame is not None:
                left_display = self._process_frame(
                    self.left_frame,
                    self.attention_left,
                    "Left Camera (CSI 0)"
                )
                frames.append(left_display)
                
            # Process right camera
            if self.right_frame is not None:
                right_display = self._process_frame(
                    self.right_frame,
                    self.attention_right,
                    "Right Camera (CSI 1)"
                )
                frames.append(right_display)
                
        if not frames:
            # No frames available - create placeholder
            placeholder = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera feeds...", 
                       (self.display_width//2 - 200, self.display_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            return placeholder
            
        # Combine frames side by side
        if len(frames) == 2:
            # Both cameras available
            combined = np.hstack(frames)
        else:
            # Single camera
            combined = frames[0]
            
        # Add overall telemetry
        if self.show_telemetry:
            self._add_telemetry_overlay(combined)
            
        return combined
        
    def _process_frame(self, frame: np.ndarray, attention: Optional[np.ndarray], 
                       label: str) -> np.ndarray:
        """Process single frame with attention overlay"""
        # Ensure frame is uint8 BGR
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
                
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # Resize to half display width
        target_width = self.display_width // 2
        target_height = self.display_height
        
        # Calculate scaling
        h, w = frame.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # Create canvas
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
        
        # Apply attention overlay if available
        if self.show_attention and attention is not None:
            # Resize attention to match frame
            attention_resized = cv2.resize(attention, (new_w, new_h))
            
            # Normalize
            if attention_resized.max() > attention_resized.min():
                attention_norm = (attention_resized - attention_resized.min()) / \
                                (attention_resized.max() - attention_resized.min())
            else:
                attention_norm = attention_resized
                
            # Create heatmap
            heatmap = cv2.applyColorMap((attention_norm * 255).astype(np.uint8), 
                                       cv2.COLORMAP_JET)
            
            # Blend with frame
            overlay = cv2.addWeighted(frame_resized, 0.7, heatmap, 0.3, 0)
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = overlay
            
        # Add label
        cv2.putText(canvas, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
        return canvas
        
    def _add_telemetry_overlay(self, frame: np.ndarray):
        """Add telemetry overlay to frame"""
        if not self.fps_buffer:
            return
            
        avg_fps = np.mean(self.fps_buffer)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                   (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                   
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (frame.shape[1] - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                   
        # Controls
        controls = "Q: Quit | A: Attention | T: Telemetry"
        cv2.putText(frame, controls, 
                   (frame.shape[1]//2 - 150, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                   
    def cleanup(self):
        """Clean up resources"""
        if self.show_window:
            cv2.destroyAllWindows()
            

def create_visual_monitor_effector(
    mailbox: Optional[PubSubMailbox] = None,
    show_window: bool = True,
    **kwargs
) -> VisualMonitorEffector:
    """
    Factory function to create visual monitor effector
    
    Args:
        mailbox: PubSub mailbox for communication
        show_window: Whether to show display window
        **kwargs: Additional configuration
        
    Returns:
        VisualMonitorEffector instance
    """
    config = {
        'mailbox': mailbox,
        'show_window': show_window,
        **kwargs
    }
    
    return VisualMonitorEffector(config)


# Test function
def test_monitor_effector():
    """Test the visual monitor effector"""
    print("Testing Visual Monitor Effector...")
    
    # Create shared mailbox
    mailbox = PubSubMailbox()
    
    # Create monitor
    monitor = create_visual_monitor_effector(mailbox, show_window=True)
    
    # Simulate camera feeds
    print("Simulating camera feeds...")
    
    for i in range(100):
        # Create test frames
        left_frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.circle(left_frame, (480 + i*3, 270), 50, (255, 0, 0), -1)
        cv2.putText(left_frame, f"Frame {i}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        right_frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.circle(right_frame, (480 - i*3, 270), 50, (0, 0, 255), -1)
        cv2.putText(right_frame, f"Frame {i}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        # Publish frames
        mailbox.publish("camera/csi/0", "test_left", left_frame)
        mailbox.publish("camera/csi/1", "test_right", right_frame)
        
        # Create attention maps
        attention_left = np.random.rand(540, 960).astype(np.float32)
        attention_right = np.random.rand(540, 960).astype(np.float32)
        
        mailbox.publish("attention/left", "test_vision", attention_left)
        mailbox.publish("attention/right", "test_vision", attention_right)
        
        # Update display
        _, telemetry = monitor.refine()
        
        if i % 30 == 0:
            print(f"Frame {i}: FPS={telemetry.get('display_fps', 0):.1f}")
            
        time.sleep(0.03)  # ~30 fps
        
        if not monitor.show_window:
            break
            
    # Cleanup
    monitor.cleanup()
    mailbox.shutdown()
    print("Test complete!")
    

if __name__ == "__main__":
    test_monitor_effector()