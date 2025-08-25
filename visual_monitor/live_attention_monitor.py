#!/usr/bin/env python3
"""
Live Camera Attention Monitor for Jetson
Shows actual camera feed with IRP attention overlay
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
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP


class LiveAttentionMonitor:
    """
    Live camera attention monitor with real-time IRP processing
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize IRP components
        print("Initializing IRP components...")
        self.memory_bridge = IRPMemoryBridge(buffer_size=30)
        self.vision_irp = create_vision_irp(self.device)
        self.vision_guided = MemoryGuidedIRP(self.vision_irp, self.memory_bridge)
        
        # Camera captures
        self.cap_csi = None
        self.cap_usb = None
        
        # Display settings
        self.show_attention = True
        self.process_every_n = 3  # Process every Nth frame
        self.frame_counter = 0
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        # Attention data
        self.last_attention_csi = None
        self.last_attention_usb = None
        self.last_telemetry_csi = {}
        self.last_telemetry_usb = {}
        
    def setup_cameras(self):
        """Setup both CSI and USB cameras"""
        print("Setting up cameras...")
        
        # Setup CSI camera
        gst_pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink max-buffers=1 drop=true"
        )
        
        self.cap_csi = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if self.cap_csi.isOpened():
            print("✓ CSI camera initialized")
        else:
            print("✗ CSI camera not available")
            
        # Setup USB camera
        for cam_id in range(4):
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap_usb = cap
                print(f"✓ USB camera initialized (index {cam_id})")
                break
        else:
            print("✗ No USB camera found")
            
        if not self.cap_csi.isOpened() and not self.cap_usb:
            raise RuntimeError("No cameras available!")
            
    def process_frame_irp(self, frame):
        """Process frame through IRP and extract attention"""
        # Resize and convert for IRP
        frame_resized = cv2.resize(frame, (224, 224))
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device) / 255.0
        
        # Process through vision IRP
        start_time = time.time()
        refined, telemetry = self.vision_guided.refine(frame_tensor, early_stop=True)
        process_time = (time.time() - start_time) * 1000
        
        # Extract attention from refinement
        with torch.no_grad():
            # Method 1: Difference between input and refined (shows what changed)
            diff = torch.abs(refined - frame_tensor)
            attention = torch.mean(diff, dim=1, keepdim=True)
            
            # Method 2: If VAE is available, use latent activation
            if hasattr(self.vision_irp, 'vae'):
                try:
                    mu, _ = self.vision_irp.vae.encode(frame_tensor)
                    latent_attention = torch.mean(torch.abs(mu), dim=1, keepdim=True)
                    # Combine both methods
                    attention = 0.5 * attention + 0.5 * torch.nn.functional.interpolate(
                        latent_attention, size=(224, 224), mode='bilinear', align_corners=False
                    )
                except:
                    pass
            
            # Normalize attention
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            
            # Resize to original frame size
            attention = torch.nn.functional.interpolate(
                attention, 
                size=(frame.shape[0], frame.shape[1]), 
                mode='bilinear', 
                align_corners=False
            )
            
            attention_map = attention.squeeze().cpu().numpy()
            
        telemetry['process_time_ms'] = process_time
        return attention_map, telemetry
        
    def create_attention_overlay(self, frame, attention_map):
        """Create frame with attention overlay"""
        if attention_map is None or not self.show_attention:
            return frame
            
        # Create heatmap
        heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        
        return overlay
        
    def draw_telemetry(self, frame, telemetry, label, y_offset=30):
        """Draw telemetry on frame"""
        cv2.putText(frame, label, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if telemetry:
            info = [
                f"Iter: {telemetry.get('iterations', 'N/A')}",
                f"Save: {telemetry.get('compute_saved', 0)*100:.1f}%",
                f"Time: {telemetry.get('process_time_ms', 0):.1f}ms"
            ]
            
            for i, text in enumerate(info):
                cv2.putText(frame, text, (10, y_offset + 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                           
    def run(self):
        """Main loop"""
        self.setup_cameras()
        
        # Create windows
        if self.cap_csi.isOpened() and self.cap_usb:
            cv2.namedWindow("Dual Camera Attention", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Dual Camera Attention", 1280, 480)
        elif self.cap_csi.isOpened():
            cv2.namedWindow("CSI Camera Attention", cv2.WINDOW_NORMAL)
        elif self.cap_usb:
            cv2.namedWindow("USB Camera Attention", cv2.WINDOW_NORMAL)
            
        print("\nLive Attention Monitor Running")
        print("Controls:")
        print("  Q: Quit")
        print("  A: Toggle attention overlay")
        print("  P: Change processing frequency")
        print("  C: Consolidate memory")
        print("-" * 40)
        
        while True:
            frames_to_show = []
            
            # Process CSI camera
            if self.cap_csi.isOpened():
                ret, frame_csi = self.cap_csi.read()
                if ret:
                    # Process through IRP periodically
                    if self.frame_counter % self.process_every_n == 0:
                        try:
                            self.last_attention_csi, self.last_telemetry_csi = \
                                self.process_frame_irp(frame_csi)
                        except Exception as e:
                            print(f"CSI processing error: {e}")
                            
                    # Apply attention overlay
                    frame_csi = self.create_attention_overlay(frame_csi, self.last_attention_csi)
                    
                    # Draw telemetry
                    self.draw_telemetry(frame_csi, self.last_telemetry_csi, "CSI Camera")
                    
                    frames_to_show.append(frame_csi)
                    
            # Process USB camera
            if self.cap_usb and self.cap_usb.isOpened():
                ret, frame_usb = self.cap_usb.read()
                if ret:
                    # Process through IRP periodically
                    if self.frame_counter % self.process_every_n == 1:  # Offset from CSI
                        try:
                            self.last_attention_usb, self.last_telemetry_usb = \
                                self.process_frame_irp(frame_usb)
                        except Exception as e:
                            print(f"USB processing error: {e}")
                            
                    # Apply attention overlay
                    frame_usb = self.create_attention_overlay(frame_usb, self.last_attention_usb)
                    
                    # Draw telemetry
                    self.draw_telemetry(frame_usb, self.last_telemetry_usb, "USB Camera")
                    
                    frames_to_show.append(frame_usb)
                    
            # Display frames
            if len(frames_to_show) == 2:
                # Show side by side
                combined = np.hstack(frames_to_show)
                
                # Add overall FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time)
                self.last_time = current_time
                self.fps_buffer.append(fps)
                avg_fps = np.mean(self.fps_buffer)
                
                cv2.putText(combined, f"FPS: {avg_fps:.1f}", 
                           (combined.shape[1]//2 - 50, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Dual Camera Attention", combined)
                
            elif len(frames_to_show) == 1:
                # Show single camera
                frame = frames_to_show[0]
                
                # Add FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time)
                self.last_time = current_time
                self.fps_buffer.append(fps)
                avg_fps = np.mean(self.fps_buffer)
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                           (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                window_name = "CSI Camera Attention" if self.cap_csi.isOpened() else "USB Camera Attention"
                cv2.imshow(window_name, frame)
                
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.show_attention = not self.show_attention
                print(f"Attention overlay: {'ON' if self.show_attention else 'OFF'}")
            elif key == ord('p'):
                self.process_every_n = 1 if self.process_every_n > 1 else 3
                print(f"Processing every {self.process_every_n} frame(s)")
            elif key == ord('c'):
                self.memory_bridge.consolidate()
                print("Memory consolidated")
                
            self.frame_counter += 1
            
        # Cleanup
        if self.cap_csi:
            self.cap_csi.release()
        if self.cap_usb:
            self.cap_usb.release()
        cv2.destroyAllWindows()
        
        # Print stats
        print("\n" + "=" * 40)
        print("Session Statistics:")
        print(f"  Total frames: {self.frame_counter}")
        if self.fps_buffer:
            print(f"  Average FPS: {np.mean(self.fps_buffer):.1f}")
            
        mem_stats = self.memory_bridge.get_memory_stats()
        print(f"  Memories: {mem_stats['total_memories']}")
        print(f"  Patterns: {mem_stats['patterns_extracted']}")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Live Camera Attention Monitor")
    print("Real-time IRP attention visualization")
    print("=" * 60)
    
    monitor = LiveAttentionMonitor()
    
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