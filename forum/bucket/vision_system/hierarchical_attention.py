#!/usr/bin/env python3
"""
Hierarchical Attention System
Implements biological-inspired awareness → attention → focus pipeline
Phase 1: Tile-based motion detection with region grouping
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
    

@dataclass
class FocusRegion:
    """Grouped tiles forming an attention region"""
    tiles: List[Tile]
    bounds: Tuple[int, int, int, int]  # x, y, w, h
    priority: float
    size: int
    
    @property
    def center(self):
        x, y, w, h = self.bounds
        return (x + w//2, y + h//2)


class HierarchicalAttention:
    """
    Three-level hierarchical attention system:
    1. Peripheral awareness (tile-based motion detection)
    2. Region formation (grouping adjacent focus tiles)
    3. Selective deep processing (VAE on focus regions)
    """
    
    def __init__(
        self,
        frame_width: int = 960,
        frame_height: int = 540,
        grid_size: Tuple[int, int] = (8, 8),
        motion_threshold: float = 0.15,
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
        
        # Thresholds
        self.motion_threshold = motion_threshold
        self.edge_threshold = edge_threshold
        self.inhibition_time = inhibition_time
        self.attention_budget_ms = attention_budget_ms
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize tile grid
        self.tiles = self._create_tile_grid()
        
        # Previous frame storage for motion detection
        self.prev_gray = None
        self.prev_tiles = {}
        
        # Temporal smoothing for tile scores
        self.tile_score_history = {}  # Store recent scores for each tile
        self.temporal_window = 3  # Number of frames to average
        
        # Inhibition of return tracking
        self.recently_attended = deque(maxlen=10)
        
        # VAE lazy loading
        self.vae = None
        self.vae_variant = None
        self.vae_last_used = 0
        self.vae_unload_timeout = 2.0  # seconds
        
        # Performance tracking
        self.frame_count = 0
        self.timing_history = deque(maxlen=30)
        
        # Visualization
        self.tile_colors = {
            TileState.PERIPHERAL: (50, 50, 50),    # Dark gray
            TileState.FOCUS: (0, 255, 255),        # Cyan
            TileState.INHIBITED: (100, 50, 50)     # Dark red
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
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Main processing pipeline
        Returns attention map and telemetry
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Convert to grayscale for motion/edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Level 1: Peripheral awareness (tile-based)
        level1_start = time.time()
        self._update_tile_states(gray)
        level1_time = (time.time() - level1_start) * 1000
        
        # Level 2: Region formation
        level2_start = time.time()
        focus_regions = self._group_focus_tiles()
        level2_time = (time.time() - level2_start) * 1000
        
        # Level 3: Selective deep processing (if budget allows)
        level3_start = time.time()
        attention_map = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        if focus_regions:
            # Process regions based on priority and budget
            remaining_budget = self.attention_budget_ms - level1_time - level2_time
            
            for region in sorted(focus_regions, key=lambda r: r.priority, reverse=True):
                if remaining_budget > 5:  # Minimum time for meaningful processing
                    if region.size > 100:  # Only process significant regions
                        # For now, use enhanced motion detection
                        # VAE would be loaded here when implemented
                        self._process_region_enhanced(frame, gray, region, attention_map)
                    else:
                        self._process_region_simple(gray, region, attention_map)
                    
                    # Update budget
                    remaining_budget -= (time.time() - level3_start) * 1000
                else:
                    # Out of budget, use simple processing
                    self._process_region_simple(gray, region, attention_map)
                    
        else:
            # No focus regions, diffuse attention
            attention_map = self._diffuse_attention(gray)
            
        level3_time = (time.time() - level3_start) * 1000
        
        # Update inhibition of return
        self._update_inhibition(focus_regions)
        
        # Store current frame for next iteration
        self.prev_gray = gray.copy()
        
        # Total processing time
        total_time = (time.time() - start_time) * 1000
        self.timing_history.append(total_time)
        
        # Create visualization overlay
        overlay = self._create_tile_overlay(frame.shape)
        
        # Telemetry
        telemetry = {
            'frame': self.frame_count,
            'level1_ms': level1_time,
            'level2_ms': level2_time,
            'level3_ms': level3_time,
            'total_ms': total_time,
            'avg_ms': np.mean(self.timing_history),
            'focus_regions': len(focus_regions),
            'focus_tiles': sum(1 for row in self.tiles for t in row if t.state == TileState.FOCUS),
            'peripheral_tiles': sum(1 for row in self.tiles for t in row if t.state == TileState.PERIPHERAL),
            'inhibited_tiles': sum(1 for row in self.tiles for t in row if t.state == TileState.INHIBITED),
        }
        
        return {
            'attention': attention_map,
            'overlay': overlay,
            'regions': focus_regions,
            'telemetry': telemetry
        }
        
    def _update_tile_states(self, gray: np.ndarray):
        """Level 1: Update tile states based on motion and edges"""
        current_time = time.time()
        
        for row in self.tiles:
            for tile in row:
                # Extract tile region
                tile_region = gray[
                    tile.y:tile.y + tile.height,
                    tile.x:tile.x + tile.width
                ]
                
                # Motion detection
                if self.prev_gray is not None:
                    prev_tile = self.prev_gray[
                        tile.y:tile.y + tile.height,
                        tile.x:tile.x + tile.width
                    ]
                    motion = cv2.absdiff(prev_tile, tile_region)
                    
                    # Apply threshold to filter noise
                    _, motion_thresh = cv2.threshold(motion, 20, 255, cv2.THRESH_BINARY)
                    tile.motion_score = np.mean(motion_thresh) / 255.0
                else:
                    tile.motion_score = 0.0
                    
                # Edge detection (Sobel) - only if motion present
                if tile.motion_score > 0.05:  # Only compute edges if there's some motion
                    edges_x = cv2.Sobel(tile_region, cv2.CV_64F, 1, 0, ksize=3)
                    edges_y = cv2.Sobel(tile_region, cv2.CV_64F, 0, 1, ksize=3)
                    edges = np.sqrt(edges_x**2 + edges_y**2)
                    tile.edge_score = np.mean(edges) / 255.0
                else:
                    tile.edge_score = 0.0
                
                # Combined trust score - heavily favor motion
                raw_trust = tile.motion_score * 0.9 + tile.edge_score * 0.1
                
                # Temporal smoothing
                tile_key = (tile.row, tile.col)
                if tile_key not in self.tile_score_history:
                    self.tile_score_history[tile_key] = deque(maxlen=self.temporal_window)
                
                self.tile_score_history[tile_key].append(raw_trust)
                
                # Use average of recent scores to reduce flashing
                tile.trust_score = np.mean(self.tile_score_history[tile_key])
                
                # Check inhibition
                if current_time - tile.last_focus_time < self.inhibition_time:
                    tile.state = TileState.INHIBITED
                    tile.trust_score *= 0.3  # Reduce but don't eliminate
                elif tile.trust_score > self.motion_threshold:
                    # Additional confirmation: require consistent motion
                    if len(self.tile_score_history[tile_key]) >= 2:
                        recent_scores = list(self.tile_score_history[tile_key])
                        if min(recent_scores[-2:]) > self.motion_threshold * 0.7:
                            tile.state = TileState.FOCUS
                        else:
                            tile.state = TileState.PERIPHERAL
                    else:
                        tile.state = TileState.PERIPHERAL
                else:
                    tile.state = TileState.PERIPHERAL
                    
    def _group_focus_tiles(self) -> List[FocusRegion]:
        """Level 2: Group adjacent focus tiles into regions"""
        regions = []
        visited = set()
        
        for row_idx, row in enumerate(self.tiles):
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
                            t = self.tiles[r][c]
                            if t.state == TileState.FOCUS:
                                visited.add((r, c))
                                region_tiles.append(t)
                                
                                # Check 8-connected neighbors
                                for dr in [-1, 0, 1]:
                                    for dc in [-1, 0, 1]:
                                        if dr == 0 and dc == 0:
                                            continue
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
                        priority = avg_trust * np.sqrt(size)  # Favor larger coherent regions
                        
                        region = FocusRegion(
                            tiles=region_tiles,
                            bounds=(min_x, min_y, max_x - min_x, max_y - min_y),
                            priority=priority,
                            size=(max_x - min_x) * (max_y - min_y)
                        )
                        regions.append(region)
                        
        return regions
        
    def _process_region_enhanced(self, frame: np.ndarray, gray: np.ndarray, 
                                region: FocusRegion, attention_map: np.ndarray):
        """Enhanced processing for significant regions (VAE placeholder)"""
        x, y, w, h = region.bounds
        
        # Extract region
        region_gray = gray[y:y+h, x:x+w]
        
        if self.prev_gray is not None:
            prev_region = self.prev_gray[y:y+h, x:x+w]
            
            # Enhanced motion with optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_region, region_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Magnitude of flow
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Normalize and apply Gaussian smoothing
            attention = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
            attention = cv2.GaussianBlur(attention, (15, 15), 0)
            
            # Place in attention map
            attention_map[y:y+h, x:x+w] = attention
        else:
            # Fallback to edge-based attention
            edges = cv2.Canny(region_gray, 50, 150)
            attention = cv2.GaussianBlur(edges.astype(np.float32) / 255, (15, 15), 0)
            attention_map[y:y+h, x:x+w] = attention
            
    def _process_region_simple(self, gray: np.ndarray, region: FocusRegion, 
                              attention_map: np.ndarray):
        """Simple processing for small regions or when out of budget"""
        x, y, w, h = region.bounds
        
        if self.prev_gray is not None:
            prev_region = self.prev_gray[y:y+h, x:x+w]
            curr_region = gray[y:y+h, x:x+w]
            
            # Simple motion
            diff = cv2.absdiff(prev_region, curr_region)
            attention = cv2.GaussianBlur(diff.astype(np.float32) / 255, (11, 11), 0)
            attention_map[y:y+h, x:x+w] = attention
            
    def _diffuse_attention(self, gray: np.ndarray) -> np.ndarray:
        """Diffuse attention when no focus regions"""
        if self.prev_gray is not None:
            # Whole frame motion
            diff = cv2.absdiff(self.prev_gray, gray)
            attention = cv2.GaussianBlur(diff.astype(np.float32) / 255, (31, 31), 0)
        else:
            # Edge-based attention
            edges = cv2.Canny(gray, 30, 100)
            attention = cv2.GaussianBlur(edges.astype(np.float32) / 255, (31, 31), 0)
            
        return attention
        
    def _update_inhibition(self, focus_regions: List[FocusRegion]):
        """Update inhibition of return for attended regions"""
        current_time = time.time()
        
        for region in focus_regions:
            for tile in region.tiles:
                tile.last_focus_time = current_time
                
    def _create_tile_overlay(self, shape: Tuple) -> np.ndarray:
        """Create visualization overlay showing tile states"""
        overlay = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        
        for row in self.tiles:
            for tile in row:
                color = self.tile_colors[tile.state]
                
                # Draw tile border
                cv2.rectangle(
                    overlay,
                    (tile.x, tile.y),
                    (tile.x + tile.width - 1, tile.y + tile.height - 1),
                    color,
                    1 if tile.state == TileState.PERIPHERAL else 2
                )
                
                # Fill slightly for focus tiles
                if tile.state == TileState.FOCUS:
                    cv2.rectangle(
                        overlay,
                        (tile.x + 2, tile.y + 2),
                        (tile.x + tile.width - 3, tile.y + tile.height - 3),
                        color,
                        -1
                    )
                    # Make semi-transparent
                    overlay[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width] = \
                        overlay[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width] // 3
                        
        return overlay


def test_hierarchical_attention():
    """Test the hierarchical attention system with camera"""
    print("Testing Hierarchical Attention System")
    print("=" * 60)
    
    # Create GStreamer pipeline for CSI camera
    def create_gst_pipeline(sensor_id=0):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
            f"video/x-raw(memory:NVMM), width=1920, height=1080, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, width=960, height=540, "
            f"format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(create_gst_pipeline(0), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open camera, using default")
        cap = cv2.VideoCapture(0)
        
    # Create attention system
    attention_system = HierarchicalAttention(
        frame_width=960,
        frame_height=540,
        grid_size=(8, 8),
        motion_threshold=0.12,  # Slightly higher to reduce false positives
        attention_budget_ms=30.0
    )
    
    # Create window
    cv2.namedWindow("Hierarchical Attention", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hierarchical Attention", 1920, 540)
    
    print("\nControls:")
    print("  Q: Quit")
    print("  Space: Pause/Resume")
    print("-" * 60)
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Process frame
            result = attention_system.process_frame(frame)
            
            # Get components
            attention_map = result['attention']
            tile_overlay = result['overlay']
            telemetry = result['telemetry']
            
            # Create heatmap
            heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend attention with frame
            attention_frame = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            
            # Add tile overlay
            attention_frame = cv2.addWeighted(attention_frame, 1.0, tile_overlay, 0.3, 0)
            
            # Create side-by-side view
            display = np.hstack([frame, attention_frame])
            
            # Add telemetry text
            y_pos = 30
            cv2.putText(display, f"Frame: {telemetry['frame']}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20
            cv2.putText(display, f"Total: {telemetry['total_ms']:.1f}ms (Avg: {telemetry['avg_ms']:.1f}ms)", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20
            cv2.putText(display, f"L1: {telemetry['level1_ms']:.1f}ms L2: {telemetry['level2_ms']:.1f}ms L3: {telemetry['level3_ms']:.1f}ms", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20
            cv2.putText(display, f"Regions: {telemetry['focus_regions']} Focus: {telemetry['focus_tiles']}/{64}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show
            cv2.imshow("Hierarchical Attention", display)
            
            # Print telemetry every 30 frames
            if telemetry['frame'] % 30 == 0:
                print(f"Frame {telemetry['frame']}: {telemetry['avg_ms']:.1f}ms avg, "
                      f"{telemetry['focus_regions']} regions, "
                      f"{telemetry['focus_tiles']} focus tiles")
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    test_hierarchical_attention()