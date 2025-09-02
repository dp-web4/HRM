#!/usr/bin/env python3
"""
Single Camera Live Attention Monitor
Simplified version for Windows with single USB camera and GPU
"""

import cv2
import torch
import numpy as np
import time
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Check if we have the visual components available
try:
    from sage.irp.plugins.vision_impl import create_vision_irp
    from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP
    USE_IRP = True
except ImportError:
    print("Warning: IRP components not available, using motion-only attention")
    USE_IRP = False


class SingleCameraAttention:
    """
    Live attention monitoring for single camera with GPU acceleration
    """
    
    def __init__(self, camera_index=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize IRP if available
        if USE_IRP:
            try:
                print("Initializing IRP components...")
                self.memory_bridge = IRPMemoryBridge(buffer_size=30)
                self.vision_irp = create_vision_irp(self.device)
                self.vision_guided = MemoryGuidedIRP(self.vision_irp, self.memory_bridge)
                self.use_irp = True
            except Exception as e:
                print(f"Could not initialize IRP: {e}")
                self.use_irp = False
        else:
            self.use_irp = False
        
        # Camera setup
        self.camera_index = camera_index
        self.cap = None
        self.cap_backend = cv2.CAP_V4L2  # Force V4L2 backend for WSL2
        
        # Motion detection
        self.prev_gray = None
        self.motion_history = deque(maxlen=10)
        
        # Display settings
        self.show_motion = True
        self.show_attention = True
        self.process_every_n = 2  # Process every Nth frame for performance
        self.frame_counter = 0
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        # Attention data
        self.attention_region = None
        self.last_telemetry = {}
        
    def setup_camera(self):
        """Setup single camera - try USB cameras"""
        print(f"Setting up camera (index {self.camera_index})...")
        
        # Try to open camera with V4L2 backend
        self.cap = cv2.VideoCapture(self.camera_index, self.cap_backend)
        
        # Wait for camera to initialize
        import time
        time.sleep(1)
        
        if not self.cap.isOpened():
            # Try other indices
            for idx in range(4):
                print(f"Trying camera index {idx}...")
                self.cap = cv2.VideoCapture(idx, self.cap_backend)
                time.sleep(1)
                if self.cap.isOpened():
                    self.camera_index = idx
                    break
        
        if self.cap.isOpened():
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual resolution
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✓ Camera initialized at {width}x{height}")
            return True
        else:
            print("✗ No camera found!")
            return False
    
    def detect_motion(self, frame):
        """Simple motion detection using frame differencing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Compute difference
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest motion area
        max_area = 0
        best_box = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                if area > max_area:
                    max_area = area
                    best_box = (x, y, w, h)
        
        self.prev_gray = gray
        return best_box
    
    def process_frame_irp(self, frame, attention_box):
        """Process attention region through IRP if available"""
        if not self.use_irp or attention_box is None:
            return {}
        
        try:
            x, y, w, h = attention_box
            
            # Extract and resize attention region
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                return {}
            
            crop_resized = cv2.resize(crop, (224, 224))
            
            # Convert to tensor
            crop_tensor = torch.from_numpy(crop_resized).float()
            crop_tensor = crop_tensor.permute(2, 0, 1).unsqueeze(0)
            crop_tensor = crop_tensor.to(self.device) / 255.0
            
            # Process through IRP
            start_time = time.time()
            refined, telemetry = self.vision_guided.refine(crop_tensor, early_stop=True)
            process_time = (time.time() - start_time) * 1000
            
            telemetry['process_time_ms'] = process_time
            return telemetry
            
        except Exception as e:
            print(f"IRP processing error: {e}")
            return {}
    
    def draw_overlay(self, frame, attention_box, telemetry):
        """Draw attention overlay on frame"""
        display = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw motion/attention box
        if attention_box is not None:
            x, y, w, h = attention_box
            
            # Draw attention rectangle
            if self.show_attention:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display, "ATTENTION", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw motion indicator
            if self.show_motion:
                # Draw corner brackets instead of full rectangle
                bracket_len = min(w, h) // 4
                color = (255, 100, 0)
                thickness = 2
                
                # Top-left
                cv2.line(display, (x, y), (x + bracket_len, y), color, thickness)
                cv2.line(display, (x, y), (x, y + bracket_len), color, thickness)
                
                # Top-right
                cv2.line(display, (x+w, y), (x+w - bracket_len, y), color, thickness)
                cv2.line(display, (x+w, y), (x+w, y + bracket_len), color, thickness)
                
                # Bottom-left
                cv2.line(display, (x, y+h), (x + bracket_len, y+h), color, thickness)
                cv2.line(display, (x, y+h), (x, y+h - bracket_len), color, thickness)
                
                # Bottom-right
                cv2.line(display, (x+w, y+h), (x+w - bracket_len, y+h), color, thickness)
                cv2.line(display, (x+w, y+h), (x+w, y+h - bracket_len), color, thickness)
        
        # Draw FPS and info
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.fps_buffer.append(fps)
        avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        
        # Info panel background
        panel_height = 100
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        # Draw text info
        info_color = (0, 255, 255)
        cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
        cv2.putText(display, f"Device: {self.device}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
        
        if telemetry:
            cv2.putText(display, f"IRP Time: {telemetry.get('process_time_ms', 0):.1f}ms", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
            
            if 'iterations' in telemetry:
                cv2.putText(display, f"Iterations: {telemetry['iterations']}", 
                           (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
            
            if 'trust_score' in telemetry:
                trust = telemetry['trust_score']
                color = (0, 255, 0) if trust > 0.7 else (0, 255, 255) if trust > 0.4 else (0, 100, 255)
                cv2.putText(display, f"Trust: {trust:.2f}", 
                           (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Draw controls hint
        cv2.putText(display, "Controls: [q]uit [m]otion [a]ttention [SPACE]pause", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        self.last_time = current_time
        return display
    
    def run(self):
        """Main loop"""
        if not self.setup_camera():
            print("Failed to setup camera!")
            return
        
        print("\nStarting live attention monitor...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'm' - Toggle motion display")
        print("  'a' - Toggle attention display")
        print("  'SPACE' - Pause/Resume")
        print("  'r' - Reset motion detection")
        
        paused = False
        
        try:
            frame_errors = 0
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        frame_errors += 1
                        print(f"Failed to read frame! (attempt {frame_errors})")
                        if frame_errors > 5:
                            print("Too many frame errors, exiting...")
                            break
                        time.sleep(0.1)
                        continue
                    
                    frame_errors = 0  # Reset on successful read
                    
                    # Detect motion
                    attention_box = self.detect_motion(frame)
                    
                    # Process through IRP periodically
                    telemetry = {}
                    if self.frame_counter % self.process_every_n == 0 and attention_box:
                        telemetry = self.process_frame_irp(frame, attention_box)
                        self.last_telemetry = telemetry
                    else:
                        telemetry = self.last_telemetry
                    
                    # Draw overlay
                    display = self.draw_overlay(frame, attention_box, telemetry)
                    
                    # Show frame
                    cv2.imshow('Live Attention Monitor', display)
                    
                    self.frame_counter += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.show_motion = not self.show_motion
                    print(f"Motion display: {'ON' if self.show_motion else 'OFF'}")
                elif key == ord('a'):
                    self.show_attention = not self.show_attention
                    print(f"Attention display: {'ON' if self.show_attention else 'OFF'}")
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('r'):
                    self.prev_gray = None
                    print("Motion detection reset")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Cleaning up...")
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Done!")


def main():
    """Run the single camera attention monitor"""
    print("=" * 50)
    print("Single Camera Live Attention Monitor")
    print("=" * 50)
    
    # Check for camera index argument
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index: {sys.argv[1]}, using default (0)")
    
    monitor = SingleCameraAttention(camera_index)
    monitor.run()


if __name__ == "__main__":
    main()