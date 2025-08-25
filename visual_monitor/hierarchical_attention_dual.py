#!/usr/bin/env python3
"""
Dual Camera Hierarchical Attention System
Improved temporal stability and stereo vision support
"""

import cv2
import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque
import threading
from enum import Enum


class TileState(Enum):
    PERIPHERAL = 0
    FOCUS = 1
    INHIBITED = 2  # Recently attended, temporarily suppressed
    WARMING = 3    # Transitioning to focus (temporal hysteresis)
    COOLING = 4    # Transitioning to peripheral


@dataclass
class Tile:
    """Single tile in the attention grid"""
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    state: TileState = TileState.PERIPHERAL
    motion_score: float = 0.0
    edge_score: float = 0.0
    trust_score: float = 0.0
    last_focus_time: float = 0.0
    state_duration: int = 0  # How long in current state
    

@dataclass
class FocusRegion:
    """Grouped tiles forming an attention region"""
    tiles: List[Tile]
    bounds: Tuple[int, int, int, int]  # x, y, w, h
    priority: float
    size: int
    camera: str  # 'left' or 'right'
    
    @property
    def center(self):
        x, y, w, h = self.bounds
        return (x + w//2, y + h//2)


class DualCameraHierarchicalAttention:
    """
    Dual camera hierarchical attention with improved temporal stability
    """
    
    def __init__(
        self,
        frame_width: int = 960,
        frame_height: int = 540,
        grid_size: Tuple[int, int] = (8, 8),
        motion_threshold: float = 0.12,
        edge_threshold: float = 0.2,
        inhibition_time: float = 1.0,
        attention_budget_ms: float = 33.0,
        device: str = 'cuda'
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_rows, self.grid_cols = grid_size
        
        # Calculate tile dimensions
        self.tile_width = frame_width // self.grid_cols
        self.tile_height = frame_height // self.grid_rows
        
        # Thresholds with hysteresis
        self.motion_threshold_high = motion_threshold  # To enter focus
        self.motion_threshold_low = motion_threshold * 0.6  # To leave focus
        self.edge_threshold = edge_threshold
        self.inhibition_time = inhibition_time
        self.attention_budget_ms = attention_budget_ms
        
        # State transition requirements
        self.warming_frames = 3  # Frames needed to transition to focus
        self.cooling_frames = 5  # Frames needed to transition to peripheral
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize tile grids for both cameras
        self.tiles_left = self._create_tile_grid()
        self.tiles_right = self._create_tile_grid()
        
        # Previous frame storage for motion detection
        self.prev_gray_left = None
        self.prev_gray_right = None
        
        # Temporal smoothing for tile scores
        self.tile_score_history_left = {}
        self.tile_score_history_right = {}
        self.temporal_window = 5  # Increased for more stability
        
        # Motion history for better filtering
        self.motion_history_left = deque(maxlen=10)
        self.motion_history_right = deque(maxlen=10)
        
        # Performance tracking
        self.frame_count = 0
        self.timing_history = deque(maxlen=30)
        
        # Visualization
        self.tile_colors = {
            TileState.PERIPHERAL: (50, 50, 50),     # Dark gray
            TileState.FOCUS: (0, 255, 255),         # Cyan
            TileState.INHIBITED: (100, 50, 50),     # Dark red
            TileState.WARMING: (0, 150, 150),       # Dark cyan
            TileState.COOLING: (100, 100, 100)      # Medium gray
        }
        
    def _create_tile_grid(self) -> List[List[Tile]]:
        """Create the initial tile grid"""
        tiles = []
        for row in range(self.grid_rows):
            tile_row = []
            for col in range(self.grid_cols):
                tile = Tile(
                    row=row,
                    col=col,
                    x=col * self.tile_width,
                    y=row * self.tile_height,
                    width=self.tile_width,
                    height=self.tile_height
                )
                tile_row.append(tile)
            tiles.append(tile_row)
        return tiles
        
    def process_dual_frames(self, frame_left: np.ndarray, frame_right: np.ndarray) -> Dict:
        """
        Process both camera frames
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Convert to grayscale
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
        # Level 1: Peripheral awareness for both cameras
        level1_start = time.time()
        self._update_tile_states(gray_left, self.tiles_left, self.prev_gray_left, 
                                self.tile_score_history_left, 'left')
        self._update_tile_states(gray_right, self.tiles_right, self.prev_gray_right,
                                self.tile_score_history_right, 'right')
        level1_time = (time.time() - level1_start) * 1000
        
        # Level 2: Region formation for both cameras
        level2_start = time.time()
        focus_regions_left = self._group_focus_tiles(self.tiles_left, 'left')
        focus_regions_right = self._group_focus_tiles(self.tiles_right, 'right')
        all_regions = focus_regions_left + focus_regions_right
        level2_time = (time.time() - level2_start) * 1000
        
        # Level 3: Process regions
        level3_start = time.time()
        attention_left = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        attention_right = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        # Process left camera regions
        for region in focus_regions_left:
            self._process_region_simple(gray_left, self.prev_gray_left, region, attention_left)
            
        # Process right camera regions
        for region in focus_regions_right:
            self._process_region_simple(gray_right, self.prev_gray_right, region, attention_right)
            
        level3_time = (time.time() - level3_start) * 1000
        
        # Update motion history
        left_motion = np.sum([t.motion_score for row in self.tiles_left for t in row])
        right_motion = np.sum([t.motion_score for row in self.tiles_right for t in row])
        self.motion_history_left.append(left_motion)
        self.motion_history_right.append(right_motion)
        
        # Store current frames
        self.prev_gray_left = gray_left.copy()
        self.prev_gray_right = gray_right.copy()
        
        # Total processing time
        total_time = (time.time() - start_time) * 1000
        self.timing_history.append(total_time)
        
        # Create visualization overlays
        overlay_left = self._create_tile_overlay(frame_left.shape, self.tiles_left)
        overlay_right = self._create_tile_overlay(frame_right.shape, self.tiles_right)
        
        # Count tile states
        def count_states(tiles):
            counts = {state: 0 for state in TileState}
            for row in tiles:
                for tile in row:
                    counts[tile.state] += 1
            return counts
            
        counts_left = count_states(self.tiles_left)
        counts_right = count_states(self.tiles_right)
        
        # Telemetry
        telemetry = {
            'frame': self.frame_count,
            'level1_ms': level1_time,
            'level2_ms': level2_time,
            'level3_ms': level3_time,
            'total_ms': total_time,
            'avg_ms': np.mean(self.timing_history),
            'focus_regions_left': len(focus_regions_left),
            'focus_regions_right': len(focus_regions_right),
            'focus_tiles_left': counts_left[TileState.FOCUS],
            'focus_tiles_right': counts_right[TileState.FOCUS],
            'warming_tiles_left': counts_left[TileState.WARMING],
            'warming_tiles_right': counts_right[TileState.WARMING],
            'avg_motion_left': np.mean(list(self.motion_history_left)) if self.motion_history_left else 0,
            'avg_motion_right': np.mean(list(self.motion_history_right)) if self.motion_history_right else 0,
        }
        
        return {
            'attention_left': attention_left,
            'attention_right': attention_right,
            'overlay_left': overlay_left,
            'overlay_right': overlay_right,
            'regions': all_regions,
            'telemetry': telemetry
        }
        
    def _update_tile_states(self, gray: np.ndarray, tiles: List[List[Tile]], 
                           prev_gray: Optional[np.ndarray], score_history: Dict,
                           camera: str):
        """Update tile states with improved temporal stability"""
        current_time = time.time()
        
        for row in tiles:
            for tile in row:
                # Extract tile region
                tile_region = gray[
                    tile.y:tile.y + tile.height,
                    tile.x:tile.x + tile.width
                ]
                
                # Motion detection with noise filtering
                if prev_gray is not None:
                    prev_tile = prev_gray[
                        tile.y:tile.y + tile.height,
                        tile.x:tile.x + tile.width
                    ]
                    
                    # Use normalized cross-correlation for more stable motion detection
                    motion = cv2.absdiff(prev_tile, tile_region)
                    
                    # Apply morphological opening to remove noise
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
                    
                    # Threshold and normalize
                    _, motion_thresh = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)
                    tile.motion_score = np.mean(motion_thresh) / 255.0
                else:
                    tile.motion_score = 0.0
                
                # Temporal smoothing with exponential moving average
                tile_key = (camera, tile.row, tile.col)
                if tile_key not in score_history:
                    score_history[tile_key] = deque(maxlen=self.temporal_window)
                
                score_history[tile_key].append(tile.motion_score)
                
                # Weighted average (recent frames have more weight)
                if len(score_history[tile_key]) > 0:
                    weights = np.exp(np.linspace(-1, 0, len(score_history[tile_key])))
                    weights /= weights.sum()
                    tile.trust_score = np.average(list(score_history[tile_key]), weights=weights)
                else:
                    tile.trust_score = tile.motion_score
                
                # State machine with hysteresis
                prev_state = tile.state
                
                # Update state duration
                tile.state_duration += 1
                
                # State transitions
                if tile.state == TileState.PERIPHERAL:
                    if tile.trust_score > self.motion_threshold_high:
                        tile.state = TileState.WARMING
                        tile.state_duration = 0
                        
                elif tile.state == TileState.WARMING:
                    if tile.trust_score < self.motion_threshold_low:
                        tile.state = TileState.PERIPHERAL
                        tile.state_duration = 0
                    elif tile.state_duration >= self.warming_frames:
                        if tile.trust_score > self.motion_threshold_high:
                            tile.state = TileState.FOCUS
                            tile.state_duration = 0
                        else:
                            tile.state = TileState.PERIPHERAL
                            tile.state_duration = 0
                            
                elif tile.state == TileState.FOCUS:
                    if tile.trust_score < self.motion_threshold_low:
                        tile.state = TileState.COOLING
                        tile.state_duration = 0
                        
                elif tile.state == TileState.COOLING:
                    if tile.trust_score > self.motion_threshold_high:
                        tile.state = TileState.FOCUS
                        tile.state_duration = 0
                    elif tile.state_duration >= self.cooling_frames:
                        tile.state = TileState.PERIPHERAL
                        tile.state_duration = 0
                        
                # Inhibition check
                if current_time - tile.last_focus_time < self.inhibition_time:
                    if tile.state == TileState.FOCUS:
                        tile.state = TileState.INHIBITED
                        
                # Update focus time
                if tile.state == TileState.FOCUS and prev_state != TileState.FOCUS:
                    tile.last_focus_time = current_time
                    
    def _group_focus_tiles(self, tiles: List[List[Tile]], camera: str) -> List[FocusRegion]:
        """Group adjacent focus tiles into regions"""
        regions = []
        visited = set()
        
        for row_idx, row in enumerate(tiles):
            for col_idx, tile in enumerate(row):
                if tile.state == TileState.FOCUS and (row_idx, col_idx) not in visited:
                    # Flood fill to find connected region
                    region_tiles = []
                    stack = [(row_idx, col_idx)]
                    
                    while stack:
                        r, c = stack.pop()
                        if (r, c) in visited:
                            continue
                            
                        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                            t = tiles[r][c]
                            if t.state == TileState.FOCUS:
                                visited.add((r, c))
                                region_tiles.append(t)
                                
                                # Check 4-connected neighbors (not 8 for cleaner regions)
                                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    stack.append((r + dr, c + dc))
                    
                    if region_tiles:
                        # Calculate bounding box
                        min_x = min(t.x for t in region_tiles)
                        min_y = min(t.y for t in region_tiles)
                        max_x = max(t.x + t.width for t in region_tiles)
                        max_y = max(t.y + t.height for t in region_tiles)
                        
                        # Calculate priority
                        avg_trust = np.mean([t.trust_score for t in region_tiles])
                        size = len(region_tiles)
                        priority = avg_trust * np.sqrt(size)
                        
                        region = FocusRegion(
                            tiles=region_tiles,
                            bounds=(min_x, min_y, max_x - min_x, max_y - min_y),
                            priority=priority,
                            size=(max_x - min_x) * (max_y - min_y),
                            camera=camera
                        )
                        regions.append(region)
                        
        return regions
        
    def _process_region_simple(self, gray: np.ndarray, prev_gray: Optional[np.ndarray],
                              region: FocusRegion, attention_map: np.ndarray):
        """Simple processing for regions"""
        x, y, w, h = region.bounds
        
        if prev_gray is not None:
            prev_region = prev_gray[y:y+h, x:x+w]
            curr_region = gray[y:y+h, x:x+w]
            
            # Simple motion with smoothing
            diff = cv2.absdiff(prev_region, curr_region)
            attention = cv2.GaussianBlur(diff.astype(np.float32) / 255, (15, 15), 0)
            attention_map[y:y+h, x:x+w] = attention
            
    def _create_tile_overlay(self, shape: Tuple, tiles: List[List[Tile]]) -> np.ndarray:
        """Create visualization overlay showing tile states"""
        overlay = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        
        for row in tiles:
            for tile in row:
                color = self.tile_colors[tile.state]
                
                # Draw tile border
                thickness = 1 if tile.state == TileState.PERIPHERAL else 2
                cv2.rectangle(
                    overlay,
                    (tile.x, tile.y),
                    (tile.x + tile.width - 1, tile.y + tile.height - 1),
                    color,
                    thickness
                )
                
                # Fill for focus/warming states
                if tile.state in [TileState.FOCUS, TileState.WARMING]:
                    alpha = 0.3 if tile.state == TileState.WARMING else 0.5
                    roi = overlay[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width]
                    roi[:] = (roi.astype(float) * (1 - alpha) + 
                             np.array(color).astype(float) * alpha).astype(np.uint8)
                        
        return overlay


def test_dual_camera_attention():
    """Test the dual camera hierarchical attention system"""
    print("Testing Dual Camera Hierarchical Attention System")
    print("=" * 60)
    
    # Create GStreamer pipeline for CSI cameras
    def create_gst_pipeline(sensor_id=0):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
            f"video/x-raw(memory:NVMM), width=1920, height=1080, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, width=960, height=540, "
            f"format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
    
    # Open both cameras
    print("Opening cameras...")
    cap0 = cv2.VideoCapture(create_gst_pipeline(0), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(create_gst_pipeline(1), cv2.CAP_GSTREAMER)
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("Failed to open both CSI cameras")
        return
        
    print("✓ Both cameras opened")
    
    # Create attention system
    attention_system = DualCameraHierarchicalAttention(
        frame_width=960,
        frame_height=540,
        grid_size=(8, 8),
        motion_threshold=0.08,  # Lower threshold for better sensitivity
        attention_budget_ms=30.0
    )
    
    # Create window
    cv2.namedWindow("Dual Camera Hierarchical Attention", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dual Camera Hierarchical Attention", 1920, 1080)
    
    print("\nControls:")
    print("  Q: Quit")
    print("  Space: Pause/Resume")
    print("-" * 60)
    
    paused = False
    
    while True:
        if not paused:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                print("Failed to capture frames")
                break
                
            # Process both frames
            result = attention_system.process_dual_frames(frame0, frame1)
            
            # Get components
            attention_left = result['attention_left']
            attention_right = result['attention_right']
            overlay_left = result['overlay_left']
            overlay_right = result['overlay_right']
            telemetry = result['telemetry']
            
            # Create heatmaps
            heatmap_left = cv2.applyColorMap((attention_left * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_right = cv2.applyColorMap((attention_right * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend with frames
            attention_frame_left = cv2.addWeighted(frame0, 0.6, heatmap_left, 0.4, 0)
            attention_frame_right = cv2.addWeighted(frame1, 0.6, heatmap_right, 0.4, 0)
            
            # Add tile overlays
            attention_frame_left = cv2.addWeighted(attention_frame_left, 1.0, overlay_left, 0.3, 0)
            attention_frame_right = cv2.addWeighted(attention_frame_right, 1.0, overlay_right, 0.3, 0)
            
            # Create 2x2 grid display
            top_row = np.hstack([frame0, frame1])
            bottom_row = np.hstack([attention_frame_left, attention_frame_right])
            display = np.vstack([top_row, bottom_row])
            
            # Add labels
            cv2.putText(display, "Left Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Right Camera", (970, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Left Attention", (10, 570), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Right Attention", (970, 570), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add telemetry
            y_pos = 60
            cv2.putText(display, f"Frame: {telemetry['frame']} | {telemetry['avg_ms']:.1f}ms avg", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            cv2.putText(display, f"L: {telemetry['focus_tiles_left']} focus, {telemetry['warming_tiles_left']} warming", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            cv2.putText(display, f"R: {telemetry['focus_tiles_right']} focus, {telemetry['warming_tiles_right']} warming", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show
            cv2.imshow("Dual Camera Hierarchical Attention", display)
            
            # Print telemetry every 30 frames
            if telemetry['frame'] % 30 == 0:
                print(f"Frame {telemetry['frame']}: {telemetry['avg_ms']:.1f}ms avg, "
                      f"L:{telemetry['focus_tiles_left']}/{telemetry['warming_tiles_left']} "
                      f"R:{telemetry['focus_tiles_right']}/{telemetry['warming_tiles_right']}")
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            
    # Cleanup
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    test_dual_camera_attention()