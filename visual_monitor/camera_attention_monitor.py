#!/usr/bin/env python3
"""
Lightweight camera attention monitor for Jetson
Uses OpenCV only for better performance
"""

import cv2
import torch
import numpy as np
import time
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.vision_impl import create_vision_irp


class LightweightAttentionMonitor:
    """
    Lightweight real-time attention monitor optimized for Jetson
    """
    
    def __init__(self, camera_id=0, display_width=1024, display_height=600):
        self.camera_id = camera_id
        self.display_width = display_width
        self.display_height = display_height
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize vision IRP
        print("Initializing Vision IRP...")
        self.vision_irp = create_vision_irp(self.device)
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.energy_buffer = deque(maxlen=100)
        self.iteration_buffer = deque(maxlen=100)
        
        # Display options
        self.show_attention = True
        self.show_stats = True
        self.process_every_n = 3  # Process every Nth frame for performance
        self.frame_counter = 0
        
        # Camera setup
        self.cap = None
        
    def initialize_camera(self):
        """Initialize camera with optimal settings for Jetson"""
        print(f"Initializing camera {self.camera_id}...")
        
        # Try GStreamer pipeline for Jetson CSI camera
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={self.camera_id} ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        # Try CSI camera first
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("CSI camera not available, trying USB camera...")
            # Fallback to USB camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if self.cap.isOpened():
                # Set USB camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
        print("Camera initialized successfully")
        
    def process_frame(self, frame):
        """Process frame through IRP and extract attention"""
        # Resize for IRP
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device) / 255.0
        
        # Process through IRP
        start_time = time.time()
        refined, telemetry = self.vision_irp.refine(frame_tensor, early_stop=True)
        process_time = (time.time() - start_time) * 1000
        
        # Extract simple attention (difference map)
        with torch.no_grad():
            # Compute difference between input and refined
            diff = torch.abs(refined - frame_tensor)
            attention = torch.mean(diff, dim=1, keepdim=True)
            
            # Normalize
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            
            # Convert to numpy
            attention_map = attention.squeeze().cpu().numpy()
            
        # Update buffers
        self.energy_buffer.append(telemetry.get('energy_trajectory', [0])[-1])
        self.iteration_buffer.append(telemetry['iterations'])
        
        return attention_map, telemetry, process_time
        
    def create_display(self, frame, attention_map=None, telemetry=None, process_time=None):
        """Create display frame with overlays"""
        h, w = frame.shape[:2]
        
        # Create canvas
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Calculate layout
        main_width = int(self.display_width * 0.7)
        side_width = self.display_width - main_width
        
        # Resize main frame
        scale = min(main_width / w, self.display_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # Place main frame
        y_offset = (self.display_height - new_h) // 2
        canvas[y_offset:y_offset+new_h, 0:new_w] = frame_resized
        
        # Add attention overlay if available
        if attention_map is not None and self.show_attention:
            # Resize attention to match frame
            attention_resized = cv2.resize(attention_map, (new_w, new_h))
            
            # Create heatmap
            heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend with frame
            overlay = cv2.addWeighted(frame_resized, 0.7, heatmap, 0.3, 0)
            canvas[y_offset:y_offset+new_h, 0:new_w] = overlay
            
        # Draw side panel
        self.draw_side_panel(canvas, main_width, telemetry, process_time)
        
        # Draw controls
        self.draw_controls(canvas)
        
        return canvas
        
    def draw_side_panel(self, canvas, x_start, telemetry, process_time):
        """Draw information panel"""
        if not self.show_stats:
            return
            
        # Background
        canvas[:, x_start:] = (30, 30, 30)
        
        # Title
        y = 30
        cv2.putText(canvas, "SAGE Vision IRP", (x_start + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Device info
        y += 40
        cv2.putText(canvas, f"Device: {self.device}", (x_start + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Telemetry
        if telemetry:
            y += 30
            cv2.putText(canvas, "Telemetry:", (x_start + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            y += 25
            cv2.putText(canvas, f"Iterations: {telemetry['iterations']}", (x_start + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            y += 20
            cv2.putText(canvas, f"Compute Saved: {telemetry['compute_saved']*100:.1f}%", 
                       (x_start + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            y += 20
            cv2.putText(canvas, f"Trust: {telemetry.get('trust', 0):.3f}", (x_start + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            if process_time:
                y += 20
                cv2.putText(canvas, f"Process: {process_time:.1f}ms", (x_start + 20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # FPS
        if self.fps_buffer:
            y += 40
            avg_fps = np.mean(self.fps_buffer)
            cv2.putText(canvas, f"FPS: {avg_fps:.1f}", (x_start + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Energy graph
        if len(self.energy_buffer) > 1:
            self.draw_mini_graph(canvas, x_start + 10, y + 30, 
                               list(self.energy_buffer), "Energy")
        
        # Iteration graph  
        if len(self.iteration_buffer) > 1:
            self.draw_mini_graph(canvas, x_start + 10, y + 130,
                               list(self.iteration_buffer), "Iterations")
            
    def draw_mini_graph(self, canvas, x, y, data, label):
        """Draw a small graph"""
        width = self.display_width - x - 20
        height = 80
        
        # Draw border
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (100, 100, 100), 1)
        
        # Draw label
        cv2.putText(canvas, label, (x + 5, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw data
        if len(data) > 1:
            # Normalize data
            min_val = min(data)
            max_val = max(data)
            range_val = max_val - min_val if max_val != min_val else 1
            
            # Create points
            points = []
            for i, val in enumerate(data[-width:]):
                px = x + i
                py = y + height - int((val - min_val) / range_val * (height - 20)) - 10
                points.append((px, py))
            
            # Draw line
            for i in range(1, len(points)):
                cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 1)
                
    def draw_controls(self, canvas):
        """Draw control instructions"""
        h = canvas.shape[0]
        controls = "Q: Quit | A: Toggle Attention | S: Toggle Stats | P: Process Every Frame"
        cv2.putText(canvas, controls, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
    def run(self):
        """Main loop"""
        self.initialize_camera()
        
        window_name = "SAGE Attention Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        
        last_time = time.time()
        attention_map = None
        telemetry = None
        process_time = None
        
        print("Starting monitor... Press 'Q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Process frame periodically
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n == 0:
                try:
                    attention_map, telemetry, process_time = self.process_frame(frame)
                except Exception as e:
                    print(f"Processing error: {e}")
                    
            # Create display
            display = self.create_display(frame, attention_map, telemetry, process_time)
            
            # Show
            cv2.imshow(window_name, display)
            
            # Update FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            self.fps_buffer.append(fps)
            last_time = current_time
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.show_attention = not self.show_attention
                print(f"Attention overlay: {'ON' if self.show_attention else 'OFF'}")
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                print(f"Statistics: {'ON' if self.show_stats else 'OFF'}")
            elif key == ord('p'):
                self.process_every_n = 1 if self.process_every_n > 1 else 3
                print(f"Processing every {self.process_every_n} frame(s)")
                
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print stats
        print("\nSession Statistics:")
        if self.fps_buffer:
            print(f"  Average FPS: {np.mean(self.fps_buffer):.1f}")
        if self.iteration_buffer:
            print(f"  Average iterations: {np.mean(self.iteration_buffer):.1f}")
        if self.energy_buffer:
            print(f"  Final energy: {self.energy_buffer[-1]:.4f}")


def main():
    """Main entry point"""
    print("=" * 60)
    print("SAGE Lightweight Attention Monitor")
    print("Optimized for Jetson Orin Nano")
    print("=" * 60)
    
    monitor = LightweightAttentionMonitor()
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()