#!/usr/bin/env python3
"""
Test live camera feed on Jetson
"""

import cv2
import numpy as np
import time


def test_csi_camera():
    """Test CSI camera with GStreamer"""
    print("Testing CSI Camera...")
    
    # GStreamer pipeline for CSI camera
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=960, height=540, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=true"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Failed to open CSI camera!")
        return False
        
    print("CSI Camera opened successfully!")
    print("Press 'q' to quit")
    
    fps_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_time)
            fps_time = current_time
            print(f"FPS: {fps:.1f}")
            
        # Add text overlay
        cv2.putText(frame, f"CSI Camera - Frame {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("CSI Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return True


def test_usb_camera():
    """Test USB camera"""
    print("\nTesting USB Camera...")
    
    # Try different camera indices
    for cam_id in range(4):
        print(f"Trying camera {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        
        if cap.isOpened():
            print(f"Camera {cam_id} opened!")
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("Press 'q' to quit")
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Add text
                cv2.putText(frame, f"USB Camera {cam_id} - Frame {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(f"USB Camera {cam_id}", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            return True
            
    print("No USB cameras found")
    return False


def test_dual_camera():
    """Test showing two cameras simultaneously"""
    print("\nTesting Dual Camera Setup...")
    
    # Try to open CSI camera
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    cap_csi = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    # Try to open USB camera
    cap_usb = None
    for cam_id in range(4):
        cap_test = cv2.VideoCapture(cam_id)
        if cap_test.isOpened():
            cap_usb = cap_test
            cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"USB camera found at index {cam_id}")
            break
    
    if not cap_csi.isOpened() and not cap_usb:
        print("No cameras available!")
        return False
        
    print("Press 'q' to quit")
    
    # Create window
    cv2.namedWindow("Dual Camera View", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    fps_time = time.time()
    
    while True:
        frames = []
        
        # Read CSI camera
        if cap_csi.isOpened():
            ret, frame_csi = cap_csi.read()
            if ret:
                cv2.putText(frame_csi, "CSI Camera", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append(frame_csi)
        
        # Read USB camera
        if cap_usb and cap_usb.isOpened():
            ret, frame_usb = cap_usb.read()
            if ret:
                cv2.putText(frame_usb, "USB Camera", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append(frame_usb)
        
        if not frames:
            print("No frames available")
            break
            
        # Combine frames
        if len(frames) == 2:
            combined = np.hstack(frames)
        else:
            combined = frames[0]
            
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_time)
            fps_time = current_time
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        cv2.imshow("Dual Camera View", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    if cap_csi.isOpened():
        cap_csi.release()
    if cap_usb:
        cap_usb.release()
    cv2.destroyAllWindows()
    
    return True


def main():
    """Main entry point"""
    print("=" * 60)
    print("Jetson Camera Test")
    print("=" * 60)
    
    # Test CSI camera first
    if test_csi_camera():
        print("CSI camera test successful!")
    
    # Test USB camera
    if test_usb_camera():
        print("USB camera test successful!")
        
    # Test dual camera
    if test_dual_camera():
        print("Dual camera test successful!")
        
    print("\nAll tests complete!")


if __name__ == "__main__":
    main()