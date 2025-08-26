#!/usr/bin/env python3
"""
Test just the dynamic box strategy
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.vision_attention_plugin import create_vision_attention_plugin

def create_gst_pipeline(sensor_id=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
        f"video/x-raw(memory:NVMM), width=1920, height=1080, "
        f"format=NV12, framerate=30/1 ! "
        f"nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

print("Testing Dynamic Box Strategy Only")
print("-" * 40)

# Create plugin with dynamic box strategy
plugin = create_vision_attention_plugin(
    sensor_id=0,
    strategy="dynamic_box",
    enable_vae=False  # Disable VAE for simplicity
)

# Setup camera
cap = cv2.VideoCapture(create_gst_pipeline(0), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

cv2.namedWindow("Dynamic Box Test", cv2.WINDOW_NORMAL)

frame_count = 0
prev_gray = None

while frame_count < 300:  # Run for 300 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run plugin
    focus_region, telemetry = plugin.refine(frame)
    
    # Draw the focus region
    if focus_region:
        x, y = focus_region.x, focus_region.y
        w, h = focus_region.width, focus_region.height
        
        # Draw rectangle
        color = (0, 255, 0) if focus_region.state.value == "tracking" else (128, 128, 128)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Show info
        cv2.putText(frame, f"Dynamic Box: {focus_region.state.value}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pos: ({x},{y}) Size: {w}x{h}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Confidence: {focus_region.confidence:.3f}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Also do our own simple motion detection for comparison
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Draw all motion contours in blue for comparison
            cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)
            
            # Get overall bounding box
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, "Simple motion", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    prev_gray = gray
    
    cv2.imshow("Dynamic Box Test", frame)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: Focus at ({focus_region.x},{focus_region.y}) "
              f"state={focus_region.state.value} conf={focus_region.confidence:.3f}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")