#!/usr/bin/env python3
"""
Hierarchical Attention with Selective TinyVAE Processing
Only processes focus regions through VAE when motion detected
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import TinyVAE
from sage.irp.plugins.tinyvae_irp_plugin import TinyVAE

# Focus region dimensions (in tiles)
FOCUS_WIDTH = 4   # tiles wide
FOCUS_HEIGHT = 3  # tiles tall

# Grid dimensions
GRID_WIDTH = 8
GRID_HEIGHT = 8

# VAE parameters
VAE_INPUT_SIZE = 64  # TinyVAE expects 64x64 inputs
LATENT_DIM = 16      # Compact latent space

class AttentionState(Enum):
    CENTERED = "CENTERED"     # No motion, focus at center
    TRACKING = "TRACKING"     # Following motion
    PROCESSING = "VAE"        # VAE processing active

@dataclass
class FocusRegion:
    """Represents the focus region position and motion"""
    x: int  # Top-left tile x (display position)
    y: int  # Top-left tile y (display position)
    fx: float  # Fractional x position (for smooth movement)
    fy: float  # Fractional y position (for smooth movement)
    motion_score: float
    state: AttentionState
    latent: Optional[np.ndarray] = None  # VAE latent vector
    
    def __post_init__(self):
        """Initialize fractional positions"""
        self.fx = float(self.x)
        self.fy = float(self.y)
    
    def center(self) -> Tuple[float, float]:
        """Get center position in tile coordinates"""
        return (self.fx + FOCUS_WIDTH / 2, self.fy + FOCUS_HEIGHT / 2)

class HierarchicalVAEAttention:
    def __init__(self):
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Camera setup
        self.cap0 = None
        self.cap1 = None
        
        # Previous frames for motion detection
        self.prev_gray = [None, None]
        
        # Focus regions for each camera - properly centered
        center_x = (GRID_WIDTH - FOCUS_WIDTH) // 2  # Should be 2 for 8-4=4, 4/2=2
        center_y = (GRID_HEIGHT - FOCUS_HEIGHT) // 2  # Should be 2.5 -> 2 for 8-3=5, 5/2=2.5
        print(f"Center position: ({center_x}, {center_y}) for {FOCUS_WIDTH}x{FOCUS_HEIGHT} in {GRID_WIDTH}x{GRID_HEIGHT} grid")
        
        self.focus = [
            FocusRegion(
                x=center_x,
                y=center_y,
                fx=float(center_x),
                fy=float(center_y),
                motion_score=0.0,
                state=AttentionState.CENTERED
            ),
            FocusRegion(
                x=center_x,
                y=center_y,
                fx=float(center_x),
                fy=float(center_y),
                motion_score=0.0,
                state=AttentionState.CENTERED
            )
        ]
        
        # Parameters
        self.motion_threshold = 0.3   # Threshold for VAE activation
        self.gravity_strength = 0.6   # Focus movement speed
        self.vae_threshold = 0.5      # Motion level to trigger VAE
        
        # VAE lazy loading
        self.vae = None
        self.vae_loaded = False
        self.vae_last_used = 0
        self.vae_unload_timeout = 5.0  # Unload after 5 seconds of no use
        
        # Performance tracking
        self.vae_latency = 0
        self.crops_processed = 0
        
    def create_gst_pipeline(self, sensor_id=0, width=1920, height=1080, fps=30, 
                           display_width=960, display_height=540):
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
    
    def setup_cameras(self):
        """Initialize cameras"""
        print("Setting up cameras...")
        
        # Open camera 0
        self.cap0 = cv2.VideoCapture(self.create_gst_pipeline(0), cv2.CAP_GSTREAMER)
        if not self.cap0.isOpened():
            raise RuntimeError("Failed to open camera 0")
        print("✓ Camera 0 ready")
        
        # Open camera 1
        self.cap1 = cv2.VideoCapture(self.create_gst_pipeline(1), cv2.CAP_GSTREAMER)
        if not self.cap1.isOpened():
            raise RuntimeError("Failed to open camera 1")
        print("✓ Camera 1 ready")
    
    def load_vae(self):
        """Lazy load VAE model"""
        if not self.vae_loaded:
            print("Loading TinyVAE...")
            self.vae = TinyVAE(input_channels=3, latent_dim=LATENT_DIM, img_size=VAE_INPUT_SIZE)
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            self.vae_loaded = True
            self.vae_last_used = time.time()
            
            # Print model info
            param_count = sum(p.numel() for p in self.vae.parameters())
            print(f"✓ TinyVAE loaded: {param_count:,} parameters, {LATENT_DIM}D latent")
    
    def unload_vae(self):
        """Unload VAE to free memory"""
        if self.vae_loaded:
            print("Unloading TinyVAE...")
            del self.vae
            self.vae = None
            self.vae_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def extract_focus_crop(self, frame: np.ndarray, focus: FocusRegion) -> np.ndarray:
        """Extract 64x64 crop from focus region center"""
        height, width = frame.shape[:2]
        tile_h = height // GRID_HEIGHT
        tile_w = width // GRID_WIDTH
        
        # Get center of focus region in pixel coordinates
        center_x = int((focus.fx + FOCUS_WIDTH / 2) * tile_w)
        center_y = int((focus.fy + FOCUS_HEIGHT / 2) * tile_h)
        
        # Extract crop around center
        crop_size = VAE_INPUT_SIZE
        half_size = crop_size // 2
        
        # Ensure crop stays within frame bounds
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(width, x1 + crop_size)
        y2 = min(height, y1 + crop_size)
        
        # Adjust if crop would go out of bounds
        if x2 - x1 < crop_size:
            if x1 == 0:
                x2 = min(width, crop_size)
            else:
                x1 = max(0, width - crop_size)
                x2 = width
        if y2 - y1 < crop_size:
            if y1 == 0:
                y2 = min(height, crop_size)
            else:
                y1 = max(0, height - crop_size)
                y2 = height
        
        crop = frame[y1:y2, x1:x2]
        
        # Resize to exact size if needed
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            crop = cv2.resize(crop, (crop_size, crop_size))
        
        return crop
    
    def process_crop_through_vae(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process crop through VAE, return latent and reconstruction"""
        # Ensure VAE is loaded
        self.load_vae()
        self.vae_last_used = time.time()
        
        # Prepare input
        crop_tensor = torch.from_numpy(crop).float() / 255.0
        crop_tensor = crop_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
        crop_tensor = crop_tensor.to(self.device)
        
        # Process through VAE
        start = time.time()
        with torch.no_grad():
            mu, logvar = self.vae.encode(crop_tensor)
            latent = self.vae.reparameterize(mu, logvar)
            reconstruction = self.vae.decode(latent)
        self.vae_latency = (time.time() - start) * 1000  # ms
        
        # Convert back to numpy
        latent_np = latent.cpu().numpy().squeeze()
        recon_np = reconstruction.cpu().numpy().squeeze()
        recon_np = np.transpose(recon_np, (1, 2, 0))  # CHW -> HWC
        recon_np = (recon_np * 255).astype(np.uint8)
        
        self.crops_processed += 1
        
        return latent_np, recon_np
    
    def compute_tile_motion(self, frame: np.ndarray, prev_gray: Optional[np.ndarray], 
                           camera_idx: int) -> np.ndarray:
        """Compute motion scores for each tile"""
        height, width = frame.shape[:2]
        tile_h = height // GRID_HEIGHT
        tile_w = width // GRID_WIDTH
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize tile scores
        tile_scores = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        
        if prev_gray is not None:
            # Compute motion
            motion = cv2.absdiff(prev_gray, gray)
            
            # Apply slight blur to reduce noise
            motion = cv2.GaussianBlur(motion, (5, 5), 0)
            
            # Compute per-tile motion scores
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    y1, y2 = y * tile_h, (y + 1) * tile_h
                    x1, x2 = x * tile_w, (x + 1) * tile_w
                    
                    tile_motion = motion[y1:y2, x1:x2]
                    # Use 75th percentile for balance
                    score = np.percentile(tile_motion, 75) / 255.0
                    # Sigmoid amplification
                    score = 1.0 / (1.0 + np.exp(-10.0 * (score - 0.1)))
                    tile_scores[y, x] = score
        
        # Store current frame for next iteration
        self.prev_gray[camera_idx] = gray
        
        return tile_scores
    
    def update_focus_position(self, focus: FocusRegion, tile_scores: np.ndarray):
        """Update focus region position based on motion gravity"""
        # Find the tile with maximum motion
        max_motion = np.max(tile_scores)
        
        # Correct center position (integer, not float)
        center_x = (GRID_WIDTH - FOCUS_WIDTH) // 2  # 2 for our setup
        center_y = (GRID_HEIGHT - FOCUS_HEIGHT) // 2  # 2 for our setup
        
        if max_motion > self.motion_threshold:
            # Simple and responsive: just follow the max motion
            max_y, max_x = np.unravel_index(np.argmax(tile_scores), tile_scores.shape)
            
            # Center focus on that tile
            target_x = max(0, min(GRID_WIDTH - FOCUS_WIDTH, max_x - FOCUS_WIDTH // 2))
            target_y = max(0, min(GRID_HEIGHT - FOCUS_HEIGHT, max_y - FOCUS_HEIGHT // 2))
            
            # Move toward target with strong gravity
            focus.fx = focus.fx + self.gravity_strength * (target_x - focus.fx)
            focus.fy = focus.fy + self.gravity_strength * (target_y - focus.fy)
            
            # Update state based on motion intensity
            if max_motion > self.vae_threshold:
                focus.state = AttentionState.PROCESSING
            else:
                focus.state = AttentionState.TRACKING
            
            focus.motion_score = max_motion
            
        else:
            # Return to center slowly
            focus.fx = focus.fx + 0.1 * (center_x - focus.fx)
            focus.fy = focus.fy + 0.1 * (center_y - focus.fy)
            
            # Snap to center when very close
            if abs(focus.fx - center_x) < 0.5 and abs(focus.fy - center_y) < 0.5:
                focus.state = AttentionState.CENTERED
                focus.fx = float(center_x)
                focus.fy = float(center_y)
                focus.x = center_x
                focus.y = center_y
            else:
                focus.state = AttentionState.TRACKING
            
            focus.motion_score = max_motion
            focus.latent = None  # Clear latent when not processing
        
        # Update integer position for display
        focus.x = round(focus.fx)
        focus.y = round(focus.fy)
    
    def draw_attention_overlay(self, frame: np.ndarray, tile_scores: np.ndarray,
                              focus: FocusRegion, crop: Optional[np.ndarray],
                              recon: Optional[np.ndarray], label: str) -> np.ndarray:
        """Draw attention visualization with VAE info"""
        height, width = frame.shape[:2]
        tile_h = height // GRID_HEIGHT
        tile_w = width // GRID_WIDTH
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw tile grid and scores
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                y1, y2 = y * tile_h, (y + 1) * tile_h
                x1, x2 = x * tile_w, (x + 1) * tile_w
                
                # Check if tile is in focus region
                in_focus = (x >= focus.x and x < focus.x + FOCUS_WIDTH and
                           y >= focus.y and y < focus.y + FOCUS_HEIGHT)
                
                if in_focus:
                    # Focus region color based on state
                    if focus.state == AttentionState.PROCESSING:
                        color = (0, 0, 255)  # Red for VAE processing
                        thickness = 3
                    else:
                        color = (0, 165, 255)  # Orange for tracking
                        thickness = 2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                else:
                    # Peripheral tiles
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        # Blend overlay
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw main focus rectangle
        fx1 = int(focus.fx * tile_w)
        fy1 = int(focus.fy * tile_h)
        fx2 = int((focus.fx + FOCUS_WIDTH) * tile_w)
        fy2 = int((focus.fy + FOCUS_HEIGHT) * tile_h)
        
        if focus.state == AttentionState.PROCESSING:
            cv2.rectangle(result, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
        else:
            cv2.rectangle(result, (fx1, fy1), (fx2, fy2), (0, 165, 255), 2)
        
        # Add labels
        cv2.putText(result, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show state and motion
        state_text = f"State: {focus.state.value}"
        cv2.putText(result, state_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        motion_text = f"Motion: {focus.motion_score:.3f}"
        cv2.putText(result, motion_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show crop and reconstruction if available
        if crop is not None and recon is not None:
            # Place in corner
            crop_display = cv2.resize(crop, (128, 128))
            recon_display = cv2.resize(recon, (128, 128))
            
            # Add border
            crop_display = cv2.copyMakeBorder(crop_display, 2, 2, 2, 2, 
                                             cv2.BORDER_CONSTANT, value=(0, 255, 0))
            recon_display = cv2.copyMakeBorder(recon_display, 2, 2, 2, 2,
                                              cv2.BORDER_CONSTANT, value=(0, 0, 255))
            
            # Place in result
            result[10:142, width-276:width-144] = crop_display
            result[10:142, width-138:width-6] = recon_display
            
            # Labels
            cv2.putText(result, "Input", (width-270, 158),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(result, "VAE", (width-130, 158),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Show latent vector if available
        if focus.latent is not None:
            self.draw_latent_vector(result, focus.latent, width - 300, 180)
        
        return result
    
    def draw_latent_vector(self, img: np.ndarray, latent: np.ndarray, x: int, y: int):
        """Draw latent vector as bar graph"""
        bar_width = 15
        max_height = 60
        
        # Normalize latent values to [0, 1]
        latent_norm = (latent - latent.min()) / (latent.max() - latent.min() + 1e-8)
        
        for i, val in enumerate(latent_norm[:16]):  # Show first 16 dimensions
            bar_height = int(val * max_height)
            x1 = x + i * (bar_width + 2)
            y1 = y + max_height - bar_height
            x2 = x1 + bar_width
            y2 = y + max_height
            
            # Color based on value
            color = (0, int(255 * val), int(255 * (1 - val)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        # Label
        cv2.putText(img, f"Latent Vector ({LATENT_DIM}D)", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def check_vae_timeout(self):
        """Check if VAE should be unloaded due to inactivity"""
        if self.vae_loaded:
            idle_time = time.time() - self.vae_last_used
            if idle_time > self.vae_unload_timeout:
                self.unload_vae()
    
    def run(self):
        """Main execution loop"""
        print("\n" + "=" * 60)
        print("Hierarchical Attention with Selective VAE Processing")
        print(f"Focus: {FOCUS_WIDTH}×{FOCUS_HEIGHT} tiles | Grid: {GRID_WIDTH}×{GRID_HEIGHT}")
        print(f"VAE: {LATENT_DIM}D latent | Input: {VAE_INPUT_SIZE}×{VAE_INPUT_SIZE}")
        print("=" * 60)
        print("\nControls:")
        print("  Q: Quit")
        print("  V: Toggle VAE processing")
        print("  Space: Reset focus to center")
        print("  T: Adjust VAE threshold")
        print("-" * 60)
        
        self.setup_cameras()
        
        # Create window
        cv2.namedWindow("VAE Attention", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VAE Attention", 1920, 1080)
        
        frame_count = 0
        fps_timer = time.time()
        fps = 0
        warmup_frames = 30
        enable_vae = True
        
        # Storage for crops and reconstructions
        crops = [None, None]
        recons = [None, None]
        
        while True:
            # Capture frames
            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()
            
            if not ret0 or not ret1:
                print("Failed to capture frames")
                break
            
            # Compute motion for each camera
            tile_scores0 = self.compute_tile_motion(frame0, self.prev_gray[0], 0)
            tile_scores1 = self.compute_tile_motion(frame1, self.prev_gray[1], 1)
            
            # Update focus positions (only after warmup)
            if frame_count > warmup_frames:
                self.update_focus_position(self.focus[0], tile_scores0)
                self.update_focus_position(self.focus[1], tile_scores1)
                
                # Process through VAE if enabled and motion detected
                if enable_vae:
                    for i, (frame, focus) in enumerate([(frame0, self.focus[0]), 
                                                         (frame1, self.focus[1])]):
                        if focus.state == AttentionState.PROCESSING:
                            crop = self.extract_focus_crop(frame, focus)
                            latent, recon = self.process_crop_through_vae(crop)
                            focus.latent = latent
                            crops[i] = crop
                            recons[i] = recon
                        else:
                            crops[i] = None
                            recons[i] = None
                            focus.latent = None
            
            # Draw overlays
            frame0 = self.draw_attention_overlay(frame0, tile_scores0, self.focus[0],
                                                crops[0], recons[0], "Left Camera")
            frame1 = self.draw_attention_overlay(frame1, tile_scores1, self.focus[1],
                                                crops[1], recons[1], "Right Camera")
            
            # Create 2x2 grid display
            top_row = np.hstack([frame0, frame1])
            
            # Create info panel
            info_height = 540
            info_panel = np.zeros((info_height, 1920, 3), dtype=np.uint8)
            
            # Add telemetry
            cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_panel, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if self.vae_loaded:
                cv2.putText(info_panel, f"VAE: LOADED", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(info_panel, f"Latency: {self.vae_latency:.1f}ms", (10, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(info_panel, f"Crops: {self.crops_processed}", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(info_panel, f"VAE: UNLOADED", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.putText(info_panel, f"VAE Threshold: {self.vae_threshold:.2f}", (10, 165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Memory usage if available
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                cv2.putText(info_panel, f"GPU Mem: {mem_allocated:.1f}MB / {mem_reserved:.1f}MB", 
                           (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            bottom_row = info_panel
            
            # Combine into display
            display = np.vstack([top_row, bottom_row])
            
            # Show display
            cv2.imshow("VAE Attention", display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                # Reset focus to center
                center_x = (GRID_WIDTH - FOCUS_WIDTH) // 2  # Should be 2
                center_y = (GRID_HEIGHT - FOCUS_HEIGHT) // 2  # Should be 2
                for f in self.focus:
                    f.x = center_x
                    f.y = center_y
                    f.fx = float(center_x)
                    f.fy = float(center_y)
                    f.state = AttentionState.CENTERED
                    f.latent = None
                print(f"Focus reset to center: ({center_x}, {center_y})")
            elif key == ord('v'):
                enable_vae = not enable_vae
                print(f"VAE processing: {'ON' if enable_vae else 'OFF'}")
                if not enable_vae:
                    self.unload_vae()
            elif key == ord('t'):
                # Cycle through thresholds
                thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
                current_idx = min(range(len(thresholds)), 
                                key=lambda i: abs(thresholds[i] - self.vae_threshold))
                next_idx = (current_idx + 1) % len(thresholds)
                self.vae_threshold = thresholds[next_idx]
                print(f"VAE threshold: {self.vae_threshold:.2f}")
            
            # Update FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps = 30 / elapsed
                fps_timer = time.time()
                
                # Check VAE timeout
                self.check_vae_timeout()
        
        # Cleanup
        print("\nCleaning up...")
        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()
        
        if self.vae_loaded:
            self.unload_vae()
        
        print(f"Total crops processed: {self.crops_processed}")
        print("Done!")

def main():
    try:
        attention = HierarchicalVAEAttention()
        attention.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()