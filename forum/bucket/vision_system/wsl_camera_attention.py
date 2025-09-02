#!/usr/bin/env python3
"""
WSL2-Compatible Camera Attention Monitor
Uses fswebcam for capture, OpenCV for processing
"""

import cv2
import numpy as np
import time
import subprocess
import os
import torch
from collections import deque
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.vision_impl import create_vision_irp
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP

class WSLCameraAttention:
    """Camera attention for WSL2 using fswebcam"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize IRP if available
        try:
            self.memory_bridge = IRPMemoryBridge(buffer_size=30)
            self.vision_irp = create_vision_irp(self.device)
            self.vision_guided = MemoryGuidedIRP(self.vision_irp, self.memory_bridge)
            self.use_irp = True
            print("✓ IRP components initialized")
        except Exception as e:
            print(f"IRP not available: {e}")
            self.use_irp = False
        
        # Motion detection
        self.prev_gray = None
        self.motion_history = deque(maxlen=10)
        
        # Frame capture
        self.frame_path = "/tmp/camera_frame.jpg"
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
    def capture_frame(self):
        """Capture a frame using fswebcam"""
        try:
            # Use fswebcam to capture
            cmd = [
                "fswebcam",
                "-r", "640x480",
                "--no-banner",
                "--quiet",
                "--no-timestamp",
                self.frame_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode == 0 and os.path.exists(self.frame_path):
                # Read the captured frame
                frame = cv2.imread(self.frame_path)
                if frame is not None:
                    return True, frame
            
            return False, None
            
        except subprocess.TimeoutExpired:
            print("Frame capture timeout")
            return False, None
        except Exception as e:
            print(f"Capture error: {e}")
            return False, None
    
    def detect_motion(self, frame):
        """Detect motion in frame"""
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
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                if area > max_area:
                    max_area = area
                    best_box = (x, y, w, h)
        
        self.prev_gray = gray
        return best_box
    
    def process_frame_irp(self, frame, attention_box):
        """Process attention region through IRP"""
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
        
        # Draw attention box
        if attention_box is not None:
            x, y, w, h = attention_box
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, "ATTENTION", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add motion trail to history
            cx, cy = x + w//2, y + h//2
            self.motion_history.append((cx, cy))
        
        # Draw motion trail
        for i in range(1, len(self.motion_history)):
            cv2.line(display, self.motion_history[i-1], self.motion_history[i], 
                    (0, 100, 255), 2)
        
        # Draw info panel
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.fps_buffer.append(fps)
        avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        
        # Info text
        info_y = 30
        cv2.putText(display, f"WSL2 Camera Monitor", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        info_y += 30
        cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        info_y += 25
        cv2.putText(display, f"Device: {self.device}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        if telemetry:
            info_y += 25
            cv2.putText(display, f"IRP Time: {telemetry.get('process_time_ms', 0):.1f}ms", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            if 'trust_score' in telemetry:
                info_y += 25
                trust = telemetry['trust_score']
                color = (0, 255, 0) if trust > 0.7 else (0, 255, 255) if trust > 0.4 else (0, 100, 255)
                cv2.putText(display, f"Trust: {trust:.2f}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Frame counter
        cv2.putText(display, f"Frame: {self.frame_count}", 
                   (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        self.last_time = current_time
        return display
    
    def save_frame(self, frame, prefix="attention"):
        """Save frame to file"""
        filename = f"{prefix}_{self.frame_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        return filename
    
    def run(self, duration=30, save_frames=True):
        """Run the monitor for specified duration"""
        print("\n" + "="*50)
        print("WSL2 Camera Attention Monitor")
        print("="*50)
        print(f"Running for {duration} seconds...")
        print(f"Saving frames: {save_frames}")
        print()
        
        start_time = time.time()
        last_telemetry = {}
        saved_frames = []
        
        while (time.time() - start_time) < duration:
            # Capture frame
            success, frame = self.capture_frame()
            
            if not success:
                print(".", end="", flush=True)
                time.sleep(0.5)
                continue
            
            self.frame_count += 1
            
            # Detect motion
            attention_box = self.detect_motion(frame)
            
            # Process through IRP periodically
            if self.frame_count % 5 == 0 and attention_box:
                telemetry = self.process_frame_irp(frame, attention_box)
                if telemetry:
                    last_telemetry = telemetry
            else:
                telemetry = last_telemetry
            
            # Draw overlay
            display = self.draw_overlay(frame, attention_box, telemetry)
            
            # Save frame
            if save_frames:
                filename = self.save_frame(display)
                saved_frames.append(filename)
                print(f"Frame {self.frame_count}: {'Motion detected' if attention_box else 'No motion'}")
            
            # Small delay
            time.sleep(0.2)  # ~5 FPS max
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*50)
        print("Session Summary:")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Frames captured: {self.frame_count}")
        print(f"  Average FPS: {self.frame_count/elapsed:.2f}")
        if saved_frames:
            print(f"  Frames saved: {len(saved_frames)}")
            print(f"  First frame: {saved_frames[0]}")
            print(f"  Last frame: {saved_frames[-1]}")
        
        # Create a montage of key frames if we saved any
        if saved_frames and len(saved_frames) >= 4:
            print("\nCreating summary montage...")
            # Take 4 evenly spaced frames
            indices = np.linspace(0, len(saved_frames)-1, 4, dtype=int)
            montage_frames = []
            for i in indices:
                img = cv2.imread(saved_frames[i])
                if img is not None:
                    img = cv2.resize(img, (320, 240))
                    montage_frames.append(img)
            
            if len(montage_frames) == 4:
                top = np.hstack(montage_frames[:2])
                bottom = np.hstack(montage_frames[2:])
                montage = np.vstack([top, bottom])
                cv2.imwrite("attention_summary.jpg", montage)
                print("  Saved: attention_summary.jpg")

def main():
    """Run the WSL camera attention demo"""
    
    # Check if camera is available
    try:
        result = subprocess.run(["which", "fswebcam"], capture_output=True)
        if result.returncode != 0:
            print("Error: fswebcam not installed")
            print("Run: sudo apt-get install fswebcam")
            return
    except Exception as e:
        print(f"Error checking fswebcam: {e}")
        return
    
    # Run the monitor
    monitor = WSLCameraAttention()
    
    # Run for 30 seconds by default
    monitor.run(duration=30, save_frames=True)

if __name__ == "__main__":
    main()