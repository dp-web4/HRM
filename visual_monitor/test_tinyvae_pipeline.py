#!/usr/bin/env python3
"""
Test TinyVAE Pipeline Integration
Hooks into visual monitor to extract crops and encode them with TinyVAE
"""

import cv2
import numpy as np
import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.tinyvae_irp_plugin import create_tinyvae_irp


def extract_motion_crop(frame, attention_map, crop_size=64):
    """
    Extract crop from highest attention region
    
    Args:
        frame: Input frame [H, W, C]
        attention_map: Attention weights [H, W]
        crop_size: Size of crop to extract
        
    Returns:
        crop: Extracted crop [crop_size, crop_size, C]
        center: (x, y) center of crop
    """
    # Find peak attention
    if attention_map.max() > 0:
        # Get center of mass of attention
        y_coords, x_coords = np.meshgrid(
            np.arange(attention_map.shape[0]),
            np.arange(attention_map.shape[1]),
            indexing='ij'
        )
        
        total_attention = attention_map.sum()
        if total_attention > 0:
            center_y = int((y_coords * attention_map).sum() / total_attention)
            center_x = int((x_coords * attention_map).sum() / total_attention)
        else:
            center_y = attention_map.shape[0] // 2
            center_x = attention_map.shape[1] // 2
    else:
        # Default to center if no attention
        center_y = frame.shape[0] // 2
        center_x = frame.shape[1] // 2
    
    # Extract crop with bounds checking
    half_size = crop_size // 2
    y_start = max(0, center_y - half_size)
    y_end = min(frame.shape[0], y_start + crop_size)
    x_start = max(0, center_x - half_size)
    x_end = min(frame.shape[1], x_start + crop_size)
    
    # Adjust if at boundaries
    if y_end - y_start < crop_size:
        if y_start == 0:
            y_end = min(crop_size, frame.shape[0])
        else:
            y_start = max(0, y_end - crop_size)
    
    if x_end - x_start < crop_size:
        if x_start == 0:
            x_end = min(crop_size, frame.shape[1])
        else:
            x_start = max(0, x_end - crop_size)
    
    crop = frame[y_start:y_end, x_start:x_end]
    
    # Resize if needed
    if crop.shape[:2] != (crop_size, crop_size):
        crop = cv2.resize(crop, (crop_size, crop_size))
    
    return crop, (center_x, center_y)


def create_gst_pipeline(sensor_id=0, width=640, height=480, fps=30):
    """Create GStreamer pipeline for CSI camera"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def main():
    print("=" * 80)
    print("TinyVAE Pipeline Test - Motion Crop Encoding")
    print("=" * 80)
    
    # Initialize TinyVAE
    print("\nInitializing TinyVAE IRP plugin...")
    tinyvae = create_tinyvae_irp(
        latent_dim=16,
        input_channels=1  # Grayscale
    )
    print(f"✓ TinyVAE initialized: {tinyvae.entity_id}")
    print(f"  Latent dimension: {tinyvae.latent_dim}")
    print(f"  Input size: {tinyvae.img_size}x{tinyvae.img_size}")
    
    # Try to open camera
    print("\nOpening camera...")
    
    # Try CSI camera first
    try:
        cap = cv2.VideoCapture(create_gst_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("CSI camera not available")
        print("✓ Using CSI camera")
    except:
        # Fall back to USB camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No camera available, using synthetic test")
            cap = None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("✓ Using USB camera")
    
    # Create windows
    cv2.namedWindow("TinyVAE Pipeline", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TinyVAE Pipeline", 1280, 480)
    
    print("\nControls:")
    print("  Q: Quit")
    print("  R: Toggle reconstruction display")
    print("  L: Print latent vector")
    print("  S: Save current crop")
    print("-" * 80)
    
    # State variables
    show_reconstruction = True
    prev_gray = None
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    print("\nStarting pipeline...")
    
    while True:
        # Get frame
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
        else:
            # Generate synthetic frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add a moving box for motion
            box_x = int(320 + 100 * np.sin(frame_count * 0.1))
            box_y = int(240 + 50 * np.cos(frame_count * 0.1))
            cv2.rectangle(frame, (box_x-30, box_y-30), (box_x+30, box_y+30), (255, 255, 255), -1)
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute motion-based attention
        if prev_gray is not None:
            # Motion detection
            diff = cv2.absdiff(prev_gray, gray)
            attention = cv2.GaussianBlur(diff.astype(np.float32), (21, 21), 0)
            
            # Normalize attention
            if attention.max() > 0:
                attention = attention / attention.max()
        else:
            # No previous frame, uniform attention
            attention = np.ones_like(gray, dtype=np.float32) * 0.5
        
        prev_gray = gray.copy()
        
        # Extract crop from highest attention region
        crop, center = extract_motion_crop(frame, attention, crop_size=64)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Encode with TinyVAE
        latent, telemetry = tinyvae.refine(crop_gray)
        
        # Get reconstruction if enabled
        if show_reconstruction:
            reconstruction = tinyvae.get_reconstruction()
            if reconstruction is not None:
                # Convert to numpy for display
                recon_np = (reconstruction[0, 0].cpu().numpy() * 255).astype(np.uint8)
                recon_bgr = cv2.cvtColor(recon_np, cv2.COLOR_GRAY2BGR)
                # Resize for display
                recon_display = cv2.resize(recon_bgr, (128, 128))
        
        # Create display
        display = frame.copy()
        
        # Draw crop location
        cv2.rectangle(display, 
                     (center[0]-32, center[1]-32),
                     (center[0]+32, center[1]+32),
                     (0, 255, 0), 2)
        
        # Add attention overlay
        attention_color = cv2.applyColorMap((attention * 255).astype(np.uint8), cv2.COLORMAP_JET)
        display = cv2.addWeighted(display, 0.7, attention_color, 0.3, 0)
        
        # Create side panel
        panel = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Show crop (enlarged)
        crop_large = cv2.resize(crop, (256, 256))
        panel[10:266, 10:266] = crop_large
        cv2.putText(panel, "Motion Crop", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show reconstruction if enabled
        if show_reconstruction and 'recon_display' in locals():
            panel[10:138, 280:408] = recon_display
            cv2.putText(panel, "Reconstruction", (280, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show reconstruction error
            error_text = f"MSE: {telemetry['reconstruction_error']:.4f}"
            cv2.putText(panel, error_text, (280, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show latent info
        latent_info = [
            f"Latent Dim: {telemetry['latent_dim']}",
            f"Latent Norm: {telemetry['latent_norm']:.2f}",
            f"KL Div: {telemetry['kl_divergence']:.4f}",
            f"Time: {telemetry['time_ms']:.2f}ms"
        ]
        
        y_offset = 320
        for info in latent_info:
            cv2.putText(panel, info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
        
        # FPS counter
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
        
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine displays
        combined = np.hstack([display, panel])
        cv2.imshow("TinyVAE Pipeline", combined)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            show_reconstruction = not show_reconstruction
            print(f"Reconstruction display: {'ON' if show_reconstruction else 'OFF'}")
        elif key == ord('l'):
            # Print latent vector
            latent_np = latent[0].cpu().numpy()
            print(f"\nLatent vector: {latent_np}")
            print(f"  Shape: {latent_np.shape}")
            print(f"  Min: {latent_np.min():.3f}, Max: {latent_np.max():.3f}")
            print(f"  Mean: {latent_np.mean():.3f}, Std: {latent_np.std():.3f}")
        elif key == ord('s'):
            # Save crop
            filename = f"crop_{int(time.time())}.png"
            cv2.imwrite(filename, crop)
            print(f"Saved crop to {filename}")
    
    # Cleanup
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    
    print("\nPipeline test complete!")


if __name__ == "__main__":
    main()