#!/usr/bin/env python3
"""
Test camera access in WSL2
"""

import cv2
import numpy as np
import time

def test_camera():
    """Test camera with various configurations"""
    
    print("Testing camera access in WSL2...")
    print("-" * 50)
    
    # Try different backends
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Any"),
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTrying backend: {backend_name}")
        
        # Try to open camera
        cap = cv2.VideoCapture(0, backend_id)
        
        # Wait a bit for camera to initialize
        time.sleep(1)
        
        if not cap.isOpened():
            print(f"  Failed to open with {backend_name}")
            continue
            
        print(f"  ✓ Camera opened with {backend_name}")
        
        # Get properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # Try to read frames
        print("  Attempting to read 5 frames...")
        success_count = 0
        
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                success_count += 1
                print(f"    Frame {i+1}: SUCCESS - shape {frame.shape}")
                
                # Save first frame as test
                if i == 0:
                    cv2.imwrite("test_frame.jpg", frame)
                    print("    Saved first frame as test_frame.jpg")
            else:
                print(f"    Frame {i+1}: FAILED")
            time.sleep(0.2)
        
        print(f"  Read {success_count}/5 frames successfully")
        
        cap.release()
        
        if success_count > 0:
            print(f"\n✅ Camera working with {backend_name} backend!")
            return True
    
    print("\n❌ Could not get camera working with any backend")
    return False

def show_live_preview():
    """Show a simple live preview"""
    print("\n" + "=" * 50)
    print("Starting live preview...")
    print("Press 'q' to quit")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    time.sleep(1)
    
    if not cap.isOpened():
        print("Failed to open camera for preview")
        return
    
    # Create window
    cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Test', 640, 480)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if ret:
            frame_count += 1
            
            # Add text overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Camera Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f} seconds")
    print(f"Average FPS: {fps:.1f}")

def main():
    print("=" * 50)
    print("WSL2 Camera Test")
    print("=" * 50)
    
    # Test basic camera access
    if test_camera():
        print("\nCamera test passed!")
        
        # Ask if user wants to see live preview
        print("\nNote: Live preview requires X11 forwarding or similar.")
        print("The window might not display in WSL2 without proper setup.")
        print("But we can still test if frames are being captured.")
        
        show_live_preview()
    else:
        print("\nCamera test failed. Troubleshooting tips:")
        print("1. Make sure camera is attached with usbipd")
        print("2. Check permissions: sudo chmod 666 /dev/video0")
        print("3. Try: sudo modprobe uvcvideo")
        print("4. Restart WSL2 and reattach camera")

if __name__ == "__main__":
    main()