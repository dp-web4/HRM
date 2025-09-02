#!/usr/bin/env python3
"""
Hierarchical Attention with Gravity-Based Focus Region
Fixed-size focus region that gravitates toward motion
"""

import cv2
import numpy as np
import time
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

# Focus region dimensions (in tiles)
FOCUS_WIDTH = 4   # tiles wide
FOCUS_HEIGHT = 3  # tiles tall

# Grid dimensions
GRID_WIDTH = 8
GRID_HEIGHT = 8

class AttentionState(Enum):
    CENTERED = "CENTERED"     # No motion, focus at center
    TRACKING = "TRACKING"     # Following motion
    TRANSITIONING = "TRANS"   # Moving between positions

@dataclass
class FocusRegion:
    """Represents the focus region position and motion"""
    x: int  # Top-left tile x (display position)
    y: int  # Top-left tile y (display position)
    fx: float  # Fractional x position (for smooth movement)
    fy: float  # Fractional y position (for smooth movement)
    motion_score: float
    state: AttentionState
    
    def __post_init__(self):
        """Initialize fractional positions"""
        self.fx = float(self.x)
        self.fy = float(self.y)
    
    def center(self) -> Tuple[float, float]:
        """Get center position in tile coordinates"""
        return (self.fx + FOCUS_WIDTH / 2, self.fy + FOCUS_HEIGHT / 2)

class HierarchicalAttentionGravity:
    def __init__(self):
        # Camera setup
        self.cap0 = None
        self.cap1 = None
        
        # Previous frames for motion detection
        self.prev_gray = [None, None]
        
        # Focus regions for each camera
        center_x = (GRID_WIDTH - FOCUS_WIDTH) // 2
        center_y = (GRID_HEIGHT - FOCUS_HEIGHT) // 2
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
        
        # Motion history for smoothing
        self.motion_history = [
            np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32),
            np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        ]
        
        # Parameters
        self.motion_threshold = 0.3   # Threshold for sigmoid-scaled values
        self.gravity_strength = 0.6   # Moderate gravity for smooth movement
        self.smoothing_factor = 0.3   # Not used anymore but kept for compatibility
        
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
                    # Use 75th percentile for balance between mean and max
                    score = np.percentile(tile_motion, 75) / 255.0
                    # Gentle amplification with sigmoid curve
                    score = 1.0 / (1.0 + np.exp(-10.0 * (score - 0.1)))
                    tile_scores[y, x] = score
        
        # Store current frame for next iteration
        self.prev_gray[camera_idx] = gray
        
        # Return raw scores without temporal smoothing for better responsiveness
        return tile_scores
    
    def compute_gravity_center(self, tile_scores: np.ndarray) -> Tuple[float, float]:
        """Compute center of gravity for motion"""
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        # Only consider tiles above threshold
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if tile_scores[y, x] > self.motion_threshold:
                    weight = tile_scores[y, x]
                    weighted_x += x * weight
                    weighted_y += y * weight
                    total_weight += weight
        
        if total_weight > 0:
            # Return center of gravity
            return (weighted_x / total_weight, weighted_y / total_weight)
        else:
            # No significant motion, return center
            return (GRID_WIDTH / 2, GRID_HEIGHT / 2)
    
    def update_focus_position(self, focus: FocusRegion, tile_scores: np.ndarray):
        """Update focus region position based on motion gravity"""
        # Find the tile with maximum motion
        max_motion = np.max(tile_scores)
        
        if max_motion > self.motion_threshold:
            # Find position of strongest motion
            max_y, max_x = np.unravel_index(np.argmax(tile_scores), tile_scores.shape)
            
            # Center focus on that tile (adjust for focus size)
            target_x = max(0, min(GRID_WIDTH - FOCUS_WIDTH, max_x - FOCUS_WIDTH // 2))
            target_y = max(0, min(GRID_HEIGHT - FOCUS_HEIGHT, max_y - FOCUS_HEIGHT // 2))
            
            # Move directly to target with gravity strength
            focus.fx = focus.fx + self.gravity_strength * (target_x - focus.fx)
            focus.fy = focus.fy + self.gravity_strength * (target_y - focus.fy)
            
            focus.state = AttentionState.TRACKING
            focus.motion_score = max_motion
            
        else:
            # Return to center slowly
            center_x = (GRID_WIDTH - FOCUS_WIDTH) / 2
            center_y = (GRID_HEIGHT - FOCUS_HEIGHT) / 2
            
            # Weak pull to center
            focus.fx = focus.fx + 0.1 * (center_x - focus.fx)
            focus.fy = focus.fy + 0.1 * (center_y - focus.fy)
            
            # Check if we're close enough to center
            if abs(focus.fx - center_x) < 0.5 and abs(focus.fy - center_y) < 0.5:
                focus.state = AttentionState.CENTERED
                focus.fx = center_x
                focus.fy = center_y
            else:
                focus.state = AttentionState.TRANSITIONING
            
            focus.motion_score = max_motion
        
        # Update integer position for display
        focus.x = round(focus.fx)
        focus.y = round(focus.fy)
    
    def draw_attention_overlay(self, frame: np.ndarray, tile_scores: np.ndarray,
                              focus: FocusRegion, label: str) -> np.ndarray:
        """Draw attention visualization on frame"""
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
                    # Draw focus region with red border
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Fill with semi-transparent red based on motion
                    intensity = int(tile_scores[y, x] * 128)
                    cv2.rectangle(overlay, (x1+2, y1+2), (x2-2, y2-2), 
                                (0, intensity, 255), -1)
                else:
                    # Draw peripheral tiles with thin white border
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)
                    
                    # Show motion intensity in blue
                    if tile_scores[y, x] > self.motion_threshold:
                        intensity = int(tile_scores[y, x] * 128)
                        cv2.rectangle(overlay, (x1+1, y1+1), (x2-1, y2-1),
                                    (255, intensity, 0), -1)
        
        # Blend overlay with original
        result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Draw focus region outline using fractional positions for smooth movement
        fx1 = int(focus.fx * tile_w)
        fy1 = int(focus.fy * tile_h)
        fx2 = int((focus.fx + FOCUS_WIDTH) * tile_w)
        fy2 = int((focus.fy + FOCUS_HEIGHT) * tile_h)
        cv2.rectangle(result, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
        
        # Draw a smaller inner rectangle to show movement better
        cv2.rectangle(result, (fx1+3, fy1+3), (fx2-3, fy2-3), (0, 100, 255), 1)
        
        # Add labels
        cv2.putText(result, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show focus state
        state_text = f"Focus: {focus.state.value} [{focus.x},{focus.y}]"
        cv2.putText(result, state_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show motion score
        motion_text = f"Motion: {focus.motion_score:.3f}"
        cv2.putText(result, motion_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return result
    
    def run(self):
        """Main execution loop"""
        print("\n" + "=" * 60)
        print("Hierarchical Attention with Gravity-Based Focus")
        print(f"Focus Region: {FOCUS_WIDTH}×{FOCUS_HEIGHT} tiles")
        print(f"Grid: {GRID_WIDTH}×{GRID_HEIGHT} tiles")
        print("=" * 60)
        print("\nControls:")
        print("  Q: Quit")
        print("  Space: Reset focus to center")
        print("  +/-: Adjust gravity strength")
        print("  M: Adjust motion threshold")
        print("-" * 60)
        
        self.setup_cameras()
        
        # Create window
        cv2.namedWindow("Gravity-Based Attention", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gravity-Based Attention", 1920, 1080)
        
        frame_count = 0
        fps_timer = time.time()
        fps = 0
        warmup_frames = 30  # Ignore first 30 frames for camera adjustment
        
        print(f"\nGravity strength: {self.gravity_strength:.2f}")
        print(f"Motion threshold: {self.motion_threshold:.3f}")
        print(f"Warming up for {warmup_frames} frames...")
        
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
            elif frame_count == warmup_frames:
                print("Warmup complete. Starting attention tracking...")
            
            # Draw overlays
            frame0 = self.draw_attention_overlay(frame0, tile_scores0, 
                                                self.focus[0], "Left Camera")
            frame1 = self.draw_attention_overlay(frame1, tile_scores1,
                                                self.focus[1], "Right Camera")
            
            # Create 2x2 grid display
            top_row = np.hstack([frame0, frame1])
            
            # Create info panels
            info_height = 540
            info0 = np.zeros((info_height, 960, 3), dtype=np.uint8)
            info1 = np.zeros((info_height, 960, 3), dtype=np.uint8)
            
            # Add telemetry to info panels
            cv2.putText(info0, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info0, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info0, f"Gravity: {self.gravity_strength:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info0, f"Threshold: {self.motion_threshold:.3f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Visualize motion heatmap for left camera
            heatmap0 = cv2.resize(tile_scores0, (480, 480), interpolation=cv2.INTER_NEAREST)
            heatmap0 = (heatmap0 * 255).astype(np.uint8)
            heatmap0 = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)
            info0[30:510, 470:950] = heatmap0
            
            # Visualize motion heatmap for right camera
            heatmap1 = cv2.resize(tile_scores1, (480, 480), interpolation=cv2.INTER_NEAREST)
            heatmap1 = (heatmap1 * 255).astype(np.uint8)
            heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
            info1[30:510, 10:490] = heatmap1
            
            bottom_row = np.hstack([info0, info1])
            
            # Combine into 2x2 grid
            display = np.vstack([top_row, bottom_row])
            
            # Show display
            cv2.imshow("Gravity-Based Attention", display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                # Reset focus to center
                center_x = (GRID_WIDTH - FOCUS_WIDTH) // 2
                center_y = (GRID_HEIGHT - FOCUS_HEIGHT) // 2
                for f in self.focus:
                    f.x = center_x
                    f.y = center_y
                    f.fx = float(center_x)
                    f.fy = float(center_y)
                    f.state = AttentionState.CENTERED
                print("Focus reset to center")
            elif key == ord('+') or key == ord('='):
                self.gravity_strength = min(1.0, self.gravity_strength + 0.05)
                print(f"Gravity strength: {self.gravity_strength:.2f}")
            elif key == ord('-'):
                self.gravity_strength = max(0.05, self.gravity_strength - 0.05)
                print(f"Gravity strength: {self.gravity_strength:.2f}")
            elif key == ord('m'):
                # Cycle through threshold values
                thresholds = [0.02, 0.05, 0.08, 0.10, 0.15]
                current_idx = min(range(len(thresholds)), 
                                key=lambda i: abs(thresholds[i] - self.motion_threshold))
                next_idx = (current_idx + 1) % len(thresholds)
                self.motion_threshold = thresholds[next_idx]
                print(f"Motion threshold: {self.motion_threshold:.3f}")
            
            # Update FPS
            frame_count += 1
            
            # Debug output for first 150 frames
            if frame_count <= 150 and frame_count % 10 == 0:
                max_motion_0 = np.max(tile_scores0)
                max_motion_1 = np.max(tile_scores1)
                print(f"Debug Frame {frame_count}: L_motion:{max_motion_0:.4f} R_motion:{max_motion_1:.4f} threshold:{self.motion_threshold:.3f}")
            
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps = 30 / elapsed
                fps_timer = time.time()
                
                # Print periodic status with motion info
                max_motion_0 = np.max(tile_scores0)
                max_motion_1 = np.max(tile_scores1)
                grav_0 = self.compute_gravity_center(tile_scores0)
                grav_1 = self.compute_gravity_center(tile_scores1)
                print(f"Frame {frame_count}: {fps:.1f} FPS | "
                      f"L:{self.focus[0].state.value}[{self.focus[0].x},{self.focus[0].y}] mot:{max_motion_0:.3f} grav:({grav_0[0]:.1f},{grav_0[1]:.1f}) | "
                      f"R:{self.focus[1].state.value}[{self.focus[1].x},{self.focus[1].y}] mot:{max_motion_1:.3f} grav:({grav_1[0]:.1f},{grav_1[1]:.1f})")
        
        # Cleanup
        print("\nCleaning up...")
        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()
        print("Done!")

def main():
    try:
        attention = HierarchicalAttentionGravity()
        attention.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()