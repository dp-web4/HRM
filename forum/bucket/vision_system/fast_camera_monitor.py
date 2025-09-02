#!/usr/bin/env python3
"""
Fast Camera Monitor - Direct OpenCV capture for better FPS
"""

import cv2
import numpy as np
import time
import torch
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from sage.irp.plugins.vision_impl import create_vision_irp
    from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP
    USE_IRP = True
except ImportError:
    print("IRP components not available")
    USE_IRP = False

class FastCameraMonitor:
    """Fast camera monitor using direct OpenCV capture"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize IRP if available
        if USE_IRP:
            try:
                self.memory_bridge = IRPMemoryBridge(buffer_size=30)
                self.vision_irp = create_vision_irp(self.device)
                self.vision_guided = MemoryGuidedIRP(self.vision_irp, self.memory_bridge)
                self.use_irp = True
                print("✓ IRP components initialized")
            except Exception as e:
                print(f"IRP not available: {e}")
                self.use_irp = False
        else:
            self.use_irp = False
        
        # Camera
        self.cap = None
        
        # Motion detection
        self.prev_gray = None
        self.motion_history = deque(maxlen=10)  # Shorter trail for performance
        self.attention_box = None
        self.attention_smoothing = 0.7  # Smooth attention box changes
        
        # Performance
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=60)
        self.last_time = time.time()
        self.last_telemetry = {}
        
    def setup_camera(self):
        """Setup camera with OpenCV"""
        print("Setting up camera...")
        
        # Try different backends
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Any"),
        ]
        
        for backend_id, backend_name in backends:
            print(f"Trying {backend_name} backend...")
            self.cap = cv2.VideoCapture(0, backend_id)
            
            if self.cap.isOpened():
                # Configure for speed
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                
                # Set fastest codec
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"✓ Camera opened with {backend_name} ({width}x{height})")
                return True
        
        print("Failed to open camera with OpenCV")
        return False
    
    def detect_motion_fast(self, frame):
        """Fast motion detection"""
        # Downsample for faster processing
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
        
        # Find motion region
        coords = cv2.findNonZero(thresh)
        if coords is not None and len(coords) > 10:
            x, y, w, h = cv2.boundingRect(coords)
            # Scale back to original size
            box = (x*4, y*4, w*4, h*4)
            
            # Smooth the attention box
            if self.attention_box is not None:
                px, py, pw, ph = self.attention_box
                alpha = self.attention_smoothing
                box = (
                    int(px * alpha + box[0] * (1-alpha)),
                    int(py * alpha + box[1] * (1-alpha)),
                    int(pw * alpha + box[2] * (1-alpha)),
                    int(ph * alpha + box[3] * (1-alpha))
                )
            
            self.prev_gray = gray
            return box
        
        self.prev_gray = gray
        return None
    
    def draw_fast_overlay(self, frame):
        """Minimal overlay for performance"""
        display = frame
        height, width = frame.shape[:2]
        
        # Draw attention box if present
        if self.attention_box is not None:
            x, y, w, h = self.attention_box
            
            # Simple rectangle
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add center point to trail
            cx, cy = x + w//2, y + h//2
            self.motion_history.append((cx, cy))
            
            # Draw simple trail
            if len(self.motion_history) > 1:
                pts = np.array(self.motion_history, np.int32)
                cv2.polylines(display, [pts], False, (0, 150, 255), 1)
        
        # Minimal text overlay
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.fps_buffer.append(fps)
        avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 10 else fps
        self.last_time = current_time
        
        # FPS counter
        cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # GPU indicator
        gpu_text = "GPU" if self.device.type == 'cuda' else "CPU"
        color = (0, 255, 0) if self.device.type == 'cuda' else (0, 255, 255)
        cv2.putText(display, gpu_text, (width - 80, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Controls
        cv2.putText(display, "Q:quit R:reset SPACE:pause", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return display
    
    def run(self):
        """Main loop optimized for speed"""
        if not self.setup_camera():
            print("Camera setup failed")
            return
        
        print("\n" + "="*50)
        print("FAST CAMERA MONITOR")
        print("="*50)
        print("\nControls:")
        print("  'q' - Quit")
        print("  'r' - Reset")
        print("  'SPACE' - Pause")
        print("\nStarting...")
        
        # Window
        window_name = "Fast Camera Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        paused = False
        current_frame = None
        skip_counter = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    
                    if ret:
                        current_frame = frame
                        self.frame_count += 1
                        
                        # Detect motion every 2nd frame
                        if skip_counter % 2 == 0:
                            self.attention_box = self.detect_motion_fast(frame)
                        
                        skip_counter += 1
                
                if current_frame is not None:
                    # Draw overlay
                    display = self.draw_fast_overlay(current_frame)
                    
                    # Show
                    cv2.imshow(window_name, display)
                
                # Poll keyboard quickly
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.prev_gray = None
                    self.motion_history.clear()
                    self.attention_box = None
                    print("Reset")
                elif key == ord(' '):
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")
                    
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            print("\nCleaning up...")
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Summary
            print(f"\nProcessed {self.frame_count} frames")
            if self.fps_buffer:
                print(f"Average FPS: {np.mean(self.fps_buffer):.1f}")

def main():
    monitor = FastCameraMonitor()
    monitor.run()

if __name__ == "__main__":
    main()