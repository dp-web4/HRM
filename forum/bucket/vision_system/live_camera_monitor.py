#!/usr/bin/env python3
"""
Live Camera Monitor with Display for WSL2
Shows real-time camera feed with attention overlay
"""

import cv2
import numpy as np
import time
import subprocess
import os
import torch
from collections import deque
import sys
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from sage.irp.plugins.vision_impl import create_vision_irp
    from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP
    USE_IRP = True
except ImportError:
    print("IRP components not available")
    USE_IRP = False

class LiveCameraMonitor:
    """Live camera monitor with display window"""
    
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
                print(f"IRP initialization failed: {e}")
                self.use_irp = False
        else:
            self.use_irp = False
        
        # Motion detection
        self.prev_gray = None
        self.motion_history = deque(maxlen=20)
        
        # Frame queue for threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.running = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        self.last_telemetry = {}
        
    def capture_worker(self):
        """Worker thread to capture frames using fswebcam"""
        frame_path = "/tmp/live_frame.jpg"
        
        while self.running:
            try:
                # Capture frame with fswebcam
                cmd = [
                    "fswebcam",
                    "-r", "640x480",
                    "--no-banner",
                    "--quiet",
                    "--no-timestamp",
                    frame_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=1)
                
                if result.returncode == 0 and os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        # Don't block if queue is full, just replace old frame
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            # Remove old frame and add new one
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame)
                            except:
                                pass
                
                # Small delay between captures
                time.sleep(0.05)  # ~20 FPS max
                
            except Exception as e:
                if self.running:
                    print(f"Capture error: {e}")
                time.sleep(0.5)
    
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
            return {}
    
    def draw_overlay(self, frame, attention_box, telemetry):
        """Draw attention overlay on frame"""
        display = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw attention box
        if attention_box is not None:
            x, y, w, h = attention_box
            
            # Draw main attention rectangle
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw corner brackets for style
            bracket_len = min(w, h) // 4
            color = (0, 255, 255)
            thickness = 3
            
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
            
            # Label
            cv2.putText(display, "ATTENTION", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add to motion trail
            cx, cy = x + w//2, y + h//2
            self.motion_history.append((cx, cy))
        
        # Draw motion trail (simplified for performance)
        if len(self.motion_history) > 1:
            # Draw as polyline for better performance
            pts = np.array(self.motion_history, np.int32)
            cv2.polylines(display, [pts], False, (0, 150, 255), 2)
        
        # Dark overlay for info panel
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.fps_buffer.append(fps)
        avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        
        # Title
        cv2.putText(display, "LIVE CAMERA ATTENTION MONITOR", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Stats
        cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, f"Device: {str(self.device).upper()}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, f"Frame: {self.frame_count}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # IRP telemetry
        if telemetry:
            cv2.putText(display, f"IRP: {telemetry.get('process_time_ms', 0):.1f}ms", 
                       (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if 'trust_score' in telemetry:
                trust = telemetry['trust_score']
                color = (0, 255, 0) if trust > 0.7 else (0, 255, 255) if trust > 0.4 else (0, 100, 255)
                cv2.putText(display, f"Trust: {trust:.2f}", 
                           (200, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Controls hint
        cv2.putText(display, "Press 'q' to quit | 'r' to reset | SPACE to pause", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        self.last_time = current_time
        return display
    
    def run(self):
        """Main loop with live display"""
        print("\n" + "="*60)
        print("LIVE CAMERA ATTENTION MONITOR")
        print("="*60)
        print("Starting live display...")
        print()
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset motion detection")
        print("  'SPACE' - Pause/Resume")
        print("  's' - Save current frame")
        print()
        
        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_worker)
        self.capture_thread.start()
        
        # Create display window
        window_name = "Live Camera Attention"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # Wait for first frame
        print("Waiting for camera...")
        timeout = time.time() + 5
        while self.frame_queue.empty() and time.time() < timeout:
            time.sleep(0.1)
        
        if self.frame_queue.empty():
            print("Camera timeout!")
            self.running = False
            self.capture_thread.join()
            return
        
        print("Camera ready! Displaying live feed...")
        paused = False
        current_frame = None
        
        try:
            while self.running:
                # Get latest frame if not paused
                if not paused:
                    try:
                        current_frame = self.frame_queue.get_nowait()
                        self.frame_count += 1
                        
                        # Only detect motion every 3rd frame for performance
                        if self.frame_count % 3 == 0:
                            attention_box = self.detect_motion(current_frame)
                        else:
                            attention_box = getattr(self, 'last_attention_box', None)
                        
                        self.last_attention_box = attention_box
                        
                        # Process through IRP less frequently
                        if self.frame_count % 30 == 0 and attention_box:
                            new_telemetry = self.process_frame_irp(current_frame, attention_box)
                            if new_telemetry:
                                self.last_telemetry = new_telemetry
                    except queue.Empty:
                        pass
                
                if current_frame is not None:
                    # Use cached attention box and telemetry
                    attention_box = getattr(self, 'last_attention_box', None)
                    telemetry = self.last_telemetry
                    
                    # Draw overlay
                    display = self.draw_overlay(current_frame, attention_box, telemetry)
                    
                    # Show frame
                    cv2.imshow(window_name, display)
                
                # Handle keyboard input (reduced wait for responsiveness)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    self.prev_gray = None
                    self.motion_history.clear()
                    print("Reset motion detection")
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('s') and current_frame is not None:
                    filename = f"saved_frame_{self.frame_count:04d}.jpg"
                    cv2.imwrite(filename, display)
                    print(f"Saved: {filename}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            # Cleanup
            print("Shutting down...")
            self.running = False
            self.capture_thread.join()
            cv2.destroyAllWindows()
            
            # Summary
            print("\n" + "="*60)
            print("Session Summary:")
            print(f"  Frames processed: {self.frame_count}")
            if self.fps_buffer:
                print(f"  Average FPS: {np.mean(self.fps_buffer):.2f}")
            print("="*60)

def main():
    """Run the live camera monitor"""
    
    # Check requirements
    try:
        subprocess.run(["which", "fswebcam"], check=True, capture_output=True)
    except:
        print("Error: fswebcam not installed")
        print("Run: sudo apt-get install fswebcam")
        return
    
    # Check display
    if not os.environ.get('DISPLAY'):
        print("Warning: No DISPLAY variable set")
        print("X11 forwarding may not work")
        print("Try: export DISPLAY=:0")
    
    # Run monitor
    monitor = LiveCameraMonitor()
    monitor.run()

if __name__ == "__main__":
    main()