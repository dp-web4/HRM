#!/usr/bin/env python3
"""
Debug camera availability and test different capture methods
"""

import cv2
import numpy as np
import subprocess
import time


def test_gstreamer_command():
    """Test if GStreamer can see the cameras"""
    print("Testing GStreamer camera detection...")
    
    # Test nvarguscamerasrc
    cmd = "gst-inspect-1.0 nvarguscamerasrc"
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
        if "sensor-id" in result.stdout:
            print("✓ nvarguscamerasrc plugin available")
            # Extract sensor-id range
            for line in result.stdout.split('\n'):
                if 'sensor-id' in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ nvarguscamerasrc not properly installed")
    except Exception as e:
        print(f"✗ Error checking nvarguscamerasrc: {e}")
    
    # Check for cameras
    print("\nChecking for connected cameras...")
    cmd = "ls -la /dev/video*"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print("Video devices found:")
        print(result.stdout)
    else:
        print("No /dev/video* devices (CSI cameras use nvarguscamerasrc)")
        
    # Check media devices
    cmd = "ls -la /dev/media*"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print("\nMedia devices found:")
        print(result.stdout)


def test_simple_pipeline():
    """Test simplest possible GStreamer pipeline"""
    print("\n" + "="*60)
    print("Testing simple GStreamer pipeline...")
    
    pipelines = [
        # Simplest test pattern
        ("videotestsrc ! videoconvert ! appsink", "Test pattern"),
        
        # Simple CSI camera 0
        ("nvarguscamerasrc sensor-id=0 num-buffers=1 ! "
         "video/x-raw(memory:NVMM), width=640, height=480 ! "
         "nvvidconv ! video/x-raw, format=BGRx ! "
         "videoconvert ! video/x-raw, format=BGR ! appsink", "CSI Camera 0"),
        
        # Simple CSI camera 1  
        ("nvarguscamerasrc sensor-id=1 num-buffers=1 ! "
         "video/x-raw(memory:NVMM), width=640, height=480 ! "
         "nvvidconv ! video/x-raw, format=BGRx ! "
         "videoconvert ! video/x-raw, format=BGR ! appsink", "CSI Camera 1"),
    ]
    
    for pipeline, name in pipelines:
        print(f"\nTesting: {name}")
        print(f"Pipeline: {pipeline[:50]}...")
        
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ {name} working! Frame shape: {frame.shape}")
                    # Save test frame
                    filename = f"test_{name.lower().replace(' ', '_')}.jpg"
                    cv2.imwrite(f"visual_monitor/{filename}", frame)
                    print(f"  Saved test frame to {filename}")
                else:
                    print(f"✗ {name} opened but no frame received")
                cap.release()
            else:
                print(f"✗ {name} failed to open")
        except Exception as e:
            print(f"✗ {name} error: {e}")


def test_camera_modes():
    """Test different sensor modes"""
    print("\n" + "="*60)
    print("Testing different sensor modes...")
    
    # Common sensor modes for IMX219 (common CSI camera sensor)
    modes = [
        (3264, 2464, 21),  # Mode 0: Full resolution
        (3264, 1848, 28),  # Mode 1
        (1920, 1080, 30),  # Mode 2: 1080p
        (1640, 1232, 30),  # Mode 3
        (1280, 720, 60),   # Mode 4: 720p
    ]
    
    for sensor_id in [0, 1]:
        print(f"\nTesting Camera {sensor_id}...")
        
        for i, (width, height, fps) in enumerate(modes):
            pipeline = (
                f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={i} num-buffers=1 ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
                f"nvvidconv ! video/x-raw, width=640, height=480, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink"
            )
            
            try:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"  ✓ Mode {i}: {width}x{height}@{fps}fps works")
                    else:
                        print(f"  ✗ Mode {i}: Opens but no frame")
                    cap.release()
                else:
                    print(f"  ✗ Mode {i}: {width}x{height}@{fps}fps failed")
            except Exception as e:
                print(f"  ✗ Mode {i}: Error - {e}")
                
            # Only test first few modes to save time
            if i >= 2:
                break


def test_minimal_capture():
    """Test absolute minimal capture"""
    print("\n" + "="*60)
    print("Testing minimal capture loop...")
    
    # Absolute minimal pipeline
    pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true"
    )
    
    print(f"Pipeline: {pipeline}")
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("✗ Failed to open camera")
        
        # Try without nvarguscamerasrc
        print("\nTrying v4l2src instead...")
        for i in range(4):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                print(f"✓ Camera {i} opened with default backend")
                ret, frame = test_cap.read()
                if ret:
                    print(f"  Frame shape: {frame.shape}")
                test_cap.release()
                break
        return
        
    print("✓ Camera opened")
    
    # Try to capture a few frames
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i}: {frame.shape}, dtype={frame.dtype}")
            
            # Check if frame has data
            if frame.max() > 0:
                print(f"    Data range: {frame.min()}-{frame.max()}")
            else:
                print(f"    WARNING: Frame is all zeros!")
        else:
            print(f"  Frame {i}: Failed to capture")
            
        time.sleep(0.1)
        
    cap.release()


def test_argus_daemon():
    """Check if nvargus-daemon is running"""
    print("\n" + "="*60)
    print("Checking nvargus-daemon status...")
    
    cmd = "ps aux | grep nvargus"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if "nvargus-daemon" in result.stdout:
        print("✓ nvargus-daemon is running")
        for line in result.stdout.split('\n'):
            if "nvargus" in line and "grep" not in line:
                print(f"  {line[:100]}")
    else:
        print("✗ nvargus-daemon not running")
        print("  Try: sudo systemctl start nvargus-daemon")
        
    # Check permissions
    print("\nChecking permissions...")
    cmd = "groups"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"User groups: {result.stdout.strip()}")
    
    if "video" in result.stdout:
        print("✓ User is in video group")
    else:
        print("✗ User not in video group")
        print("  Try: sudo usermod -a -G video $USER")


def main():
    """Run all camera tests"""
    print("="*60)
    print("Camera Debug Utility")
    print("="*60)
    
    # Run tests
    test_gstreamer_command()
    test_argus_daemon()
    test_simple_pipeline()
    test_minimal_capture()
    test_camera_modes()
    
    print("\n" + "="*60)
    print("Debug complete!")
    print("="*60)


if __name__ == "__main__":
    main()