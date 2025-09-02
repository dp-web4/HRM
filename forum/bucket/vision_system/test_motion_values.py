#!/usr/bin/env python3
"""Quick test to check motion detection values"""

import cv2
import numpy as np
import time

def create_gst_pipeline(sensor_id=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
        f"video/x-raw(memory:NVMM), width=1920, height=1080, "
        f"format=NV12, framerate=30/1 ! "
        f"nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

print("Opening cameras...")
cap0 = cv2.VideoCapture(create_gst_pipeline(0), cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(create_gst_pipeline(1), cv2.CAP_GSTREAMER)

if not cap0.isOpened() or not cap1.isOpened():
    print("Failed to open cameras")
    exit(1)

print("Cameras opened. Testing motion detection...")

prev_gray0 = None
prev_gray1 = None

for i in range(60):  # Test for 60 frames
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    
    if not ret0 or not ret1:
        print("Failed to read frames")
        break
    
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    if prev_gray0 is not None and prev_gray1 is not None:
        # Calculate motion
        diff0 = cv2.absdiff(prev_gray0, gray0)
        diff1 = cv2.absdiff(prev_gray1, gray1)
        
        # Apply blur
        blur0 = cv2.GaussianBlur(diff0.astype(np.float32), (11, 11), 0)
        blur1 = cv2.GaussianBlur(diff1.astype(np.float32), (11, 11), 0)
        
        # Normalize
        max0 = np.max(blur0) / 255.0 if np.max(blur0) > 0 else 0
        max1 = np.max(blur1) / 255.0 if np.max(blur1) > 0 else 0
        mean0 = np.mean(blur0) / 255.0
        mean1 = np.mean(blur1) / 255.0
        
        if i % 5 == 0:  # Print every 5 frames
            print(f"Frame {i:3d}: L[max:{max0:.4f} mean:{mean0:.4f}] R[max:{max1:.4f} mean:{mean1:.4f}]")
    
    prev_gray0 = gray0.copy()
    prev_gray1 = gray1.copy()
    
    # Small delay
    time.sleep(0.033)

cap0.release()
cap1.release()
print("Done!")