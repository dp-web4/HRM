#!/usr/bin/env python3
"""
Test the camera pipeline with synthetic frames
"""

import cv2
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.visual_monitor_impl import create_visual_monitor_irp
from sage.irp.plugins.vision_impl import create_vision_irp
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP


def test_pipeline_with_synthetic_frames():
    """Test pipeline with generated frames"""
    
    print("Testing IRP Pipeline with Synthetic Frames")
    print("-" * 50)
    
    # Setup
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    print(f"Device: {device}")
    
    # Create memory bridge
    memory_bridge = IRPMemoryBridge(buffer_size=20)
    
    # Create vision IRP
    vision_irp = create_vision_irp()
    vision_guided = MemoryGuidedIRP(vision_irp, memory_bridge)
    
    # Create monitor (without window for testing)
    monitor_irp = create_visual_monitor_irp(show_window=False)
    
    # Generate test frames
    print("\nProcessing synthetic frames...")
    
    for i in range(10):
        # Create synthetic frame with moving objects
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background gradient
        for y in range(480):
            frame[y, :] = int(y / 480 * 50)
            
        # Add moving circle
        x = 320 + int(200 * np.sin(i * 0.3))
        y = 240 + int(100 * np.cos(i * 0.3))
        cv2.circle(frame, (x, y), 50, (255, 100, 0), -1)
        
        # Add static rectangle
        cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), -1)
        
        # Add moving small circles (attention points)
        for j in range(3):
            px = int(320 + 150 * np.cos(i * 0.2 + j * 2))
            py = int(240 + 150 * np.sin(i * 0.2 + j * 2))
            cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)
            
        # Process through vision IRP
        import torch
        frame_tensor = torch.from_numpy(cv2.resize(frame, (224, 224))).float()
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        
        start_time = time.time()
        refined, vision_telemetry = vision_guided.refine(frame_tensor, early_stop=True)
        vision_time = (time.time() - start_time) * 1000
        
        # Process through monitor
        _, monitor_telemetry = monitor_irp.refine(frame)
        
        # Print results
        print(f"Frame {i+1:2d}: "
              f"Vision: {vision_telemetry['iterations']} iter, "
              f"{vision_telemetry['compute_saved']*100:5.1f}% saved, "
              f"{vision_time:6.1f}ms | "
              f"Monitor: {monitor_telemetry['num_regions']} regions")
              
        # Save a sample frame
        if i == 5:
            cv2.imwrite("visual_monitor/test_outputs/pipeline_test_frame.jpg", frame)
            print(f"         Saved sample frame")
            
    # Get memory stats
    print("\nMemory Statistics:")
    mem_stats = memory_bridge.get_memory_stats()
    for key, value in mem_stats.items():
        print(f"  {key}: {value}")
        
    # Cleanup
    monitor_irp.cleanup()
    
    print("\nTest complete!")
    

def test_monitor_plugin_standalone():
    """Test just the monitor plugin"""
    
    print("\nTesting Visual Monitor Plugin Standalone")
    print("-" * 50)
    
    # Create monitor with window
    monitor = create_visual_monitor_irp(show_window=True, display_width=800, display_height=600)
    
    print("Displaying test pattern for 5 seconds...")
    print("Press 'Q' in window to quit early")
    
    start_time = time.time()
    frame_num = 0
    
    while time.time() - start_time < 5:
        # Create test pattern
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Animated pattern
        t = time.time() - start_time
        x = int(320 + 200 * np.sin(t))
        y = int(240 + 100 * np.cos(t * 1.5))
        
        cv2.circle(frame, (x, y), 80, (255, 0, 0), -1)
        cv2.circle(frame, (320, 240), 30, (0, 255, 0), -1)
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 0, 255), 3)
        
        # Add some text
        cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        # Process
        _, telemetry = monitor.refine(frame)
        
        frame_num += 1
        time.sleep(0.03)  # ~30 fps
        
    monitor.cleanup()
    print("Monitor test complete!")
    

def main():
    """Main test runner"""
    print("=" * 60)
    print("IRP Pipeline Testing Suite")
    print("=" * 60)
    
    # Test pipeline
    test_pipeline_with_synthetic_frames()
    
    # Test monitor standalone
    test_monitor_plugin_standalone()
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    

if __name__ == "__main__":
    main()