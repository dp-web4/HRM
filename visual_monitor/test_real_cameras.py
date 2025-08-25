#!/usr/bin/env python3
"""
Direct test of real CSI cameras with HRM visual monitor
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def create_gst_pipeline(sensor_id=0, width=1920, height=1080, fps=30, display_width=960, display_height=540):
    """Create GStreamer pipeline for CSI camera"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, width={display_width}, "
        f"height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

def main():
    print("=" * 60)
    print("Testing Real CSI Cameras with HRM Monitor")
    print("=" * 60)
    
    # Open both cameras
    print("Opening Camera 0...")
    cap0 = cv2.VideoCapture(create_gst_pipeline(0), cv2.CAP_GSTREAMER)
    if not cap0.isOpened():
        print("Failed to open camera 0")
        return
    print("✓ Camera 0 opened")
    
    print("Opening Camera 1...")
    cap1 = cv2.VideoCapture(create_gst_pipeline(1), cv2.CAP_GSTREAMER)
    if not cap1.isOpened():
        print("Failed to open camera 1")
        cap0.release()
        return
    print("✓ Camera 1 opened")
    
    # Create window
    cv2.namedWindow("HRM Visual Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HRM Visual Monitor", 1920, 540)
    
    print("\nControls:")
    print("  Q: Quit")
    print("  A: Toggle attention overlay")
    print("  T: Toggle telemetry")
    print("-" * 60)
    
    frame_count = 0
    show_attention = True  # Start with attention ON
    show_telemetry = True
    
    # Simple attention tracking
    attention_left = np.zeros((540, 960), dtype=np.float32)
    attention_right = np.zeros((540, 960), dtype=np.float32)
    prev_gray0 = None
    prev_gray1 = None
    
    print("Attention overlay is ON. Press 'A' to toggle.")
    
    while True:
        # Capture frames
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("Failed to capture frames")
            break
            
        # Simple motion detection for attention
        # Convert to grayscale
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        if prev_gray0 is not None and prev_gray1 is not None:
            # Calculate motion
            diff0 = cv2.absdiff(prev_gray0, gray0)
            diff1 = cv2.absdiff(prev_gray1, gray1)
            
            # Blur for smoother attention
            attention_left = cv2.GaussianBlur(diff0.astype(np.float32), (21, 21), 0)
            attention_right = cv2.GaussianBlur(diff1.astype(np.float32), (21, 21), 0)
            
            # Normalize
            if attention_left.max() > 0:
                attention_left = attention_left / attention_left.max()
            if attention_right.max() > 0:
                attention_right = attention_right / attention_right.max()
        
        # Store current frame for next iteration
        prev_gray0 = gray0.copy()
        prev_gray1 = gray1.copy()
        
        # Apply attention overlay if enabled
        if show_attention and frame_count > 1:
            # Debug: print attention stats every 30 frames
            if frame_count % 30 == 0:
                print(f"Attention - Left max: {attention_left.max():.3f}, Right max: {attention_right.max():.3f}")
            
            # Create heatmaps
            heatmap0 = cv2.applyColorMap((attention_left * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap1 = cv2.applyColorMap((attention_right * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend with frames
            frame0 = cv2.addWeighted(frame0, 0.7, heatmap0, 0.3, 0)
            frame1 = cv2.addWeighted(frame1, 0.7, heatmap1, 0.3, 0)
        
        # Add labels
        cv2.putText(frame0, "Left Camera (CSI 0)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame1, "Right Camera (CSI 1)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine frames side by side
        combined = np.hstack([frame0, frame1])
        
        # Add telemetry
        if show_telemetry:
            cv2.putText(combined, f"Frame: {frame_count}", (1800, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            status = "Attention: ON" if show_attention else "Attention: OFF"
            cv2.putText(combined, status, (1800, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls reminder
            cv2.putText(combined, "Q: Quit | A: Attention | T: Telemetry", 
                       (760, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Show combined frame
        cv2.imshow("HRM Visual Monitor", combined)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('a'):
            show_attention = not show_attention
            print(f"Attention: {'ON' if show_attention else 'OFF'}")
        elif key == ord('t'):
            show_telemetry = not show_telemetry
            print(f"Telemetry: {'ON' if show_telemetry else 'OFF'}")
        
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    print("\nCleaning up...")
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()