#!/usr/bin/env python3
"""
Vision Attention Plugin - Single sensor with adaptive strategies
Can switch between fixed-size tiles and dynamic bounding boxes
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import time
from typing import Any, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from sage.irp.base import IRPPlugin
from sage.irp.plugins.tinyvae_irp_plugin import TinyVAE


class AttentionStrategy(Enum):
    """Available attention strategies"""
    FIXED_TILES = "fixed_tiles"      # Fixed-size grid tiles (3x4)
    DYNAMIC_BOX = "dynamic_box"      # Dynamic bounding box
    ADAPTIVE = "adaptive"             # Automatically choose based on scene


class AttentionState(Enum):
    """Current attention state"""
    IDLE = "idle"
    TRACKING = "tracking"
    PROCESSING = "processing"


@dataclass
class FocusRegion:
    """Represents current focus region"""
    x: int          # Top-left x
    y: int          # Top-left y 
    width: int      # Region width
    height: int     # Region height
    confidence: float
    state: AttentionState
    strategy: AttentionStrategy
    latent: Optional[np.ndarray] = None
    

class AttentionStrategyBase(ABC):
    """Base class for attention strategies"""
    
    @abstractmethod
    def compute_focus(self, motion_map: np.ndarray, current_focus: Optional[FocusRegion]) -> FocusRegion:
        """Compute focus region from motion map"""
        pass
    
    @abstractmethod
    def get_default_focus(self, frame_width: int, frame_height: int) -> FocusRegion:
        """Get default/idle focus region"""
        pass


class FixedTileStrategy(AttentionStrategyBase):
    """Fixed-size tile grid strategy (like hierarchical_attention_gravity)"""
    
    def __init__(self, grid_width=8, grid_height=8, focus_width=4, focus_height=3):
        self.grid_w = grid_width
        self.grid_h = grid_height
        self.focus_w = focus_width
        self.focus_h = focus_height
        self.gravity_strength = 0.6
        
        # Fractional position for smooth movement
        self.fx = (grid_width - focus_width) / 2
        self.fy = (grid_height - focus_height) / 2
        
    def compute_focus(self, motion_map: np.ndarray, current_focus: Optional[FocusRegion]) -> FocusRegion:
        """Compute focus using tile-based gravity"""
        height, width = motion_map.shape
        tile_h = height // self.grid_h
        tile_w = width // self.grid_w
        
        # Compute tile scores
        tile_scores = np.zeros((self.grid_h, self.grid_w))
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                y1, y2 = y * tile_h, (y + 1) * tile_h
                x1, x2 = x * tile_w, (x + 1) * tile_w
                tile_scores[y, x] = np.mean(motion_map[y1:y2, x1:x2])
        
        # Find max motion tile
        max_motion = np.max(tile_scores)
        
        if max_motion > 0.1:  # Motion threshold - more sensitive
            max_y, max_x = np.unravel_index(np.argmax(tile_scores), tile_scores.shape)
            
            # Update fractional position with gravity
            target_x = max(0, min(self.grid_w - self.focus_w, max_x - self.focus_w // 2))
            target_y = max(0, min(self.grid_h - self.focus_h, max_y - self.focus_h // 2))
            
            self.fx += self.gravity_strength * (target_x - self.fx)
            self.fy += self.gravity_strength * (target_y - self.fy)
            
            state = AttentionState.PROCESSING if max_motion > 0.3 else AttentionState.TRACKING
        else:
            # Return to center
            center_x = (self.grid_w - self.focus_w) / 2
            center_y = (self.grid_h - self.focus_h) / 2
            self.fx += 0.1 * (center_x - self.fx)
            self.fy += 0.1 * (center_y - self.fy)
            state = AttentionState.IDLE
        
        # Convert to pixel coordinates
        tile_x = round(self.fx)
        tile_y = round(self.fy)
        
        return FocusRegion(
            x=tile_x * tile_w,
            y=tile_y * tile_h,
            width=self.focus_w * tile_w,
            height=self.focus_h * tile_h,
            confidence=max_motion,
            state=state,
            strategy=AttentionStrategy.FIXED_TILES
        )
    
    def get_default_focus(self, frame_width: int, frame_height: int) -> FocusRegion:
        """Get centered focus"""
        tile_h = frame_height // self.grid_h
        tile_w = frame_width // self.grid_w
        center_x = (self.grid_w - self.focus_w) // 2
        center_y = (self.grid_h - self.focus_h) // 2
        
        return FocusRegion(
            x=center_x * tile_w,
            y=center_y * tile_h,
            width=self.focus_w * tile_w,
            height=self.focus_h * tile_h,
            confidence=0.0,
            state=AttentionState.IDLE,
            strategy=AttentionStrategy.FIXED_TILES
        )


class DynamicBoxStrategy(AttentionStrategyBase):
    """Dynamic bounding box strategy (like fast_camera_monitor)"""
    
    def __init__(self, smoothing=0.7, min_size=64, max_size=320):
        self.smoothing = smoothing
        self.min_size = min_size
        self.max_size = max_size
        self.last_box = None
        self.warmup_frames = 30  # Skip first frames
        
    def compute_focus(self, motion_map: np.ndarray, current_focus: Optional[FocusRegion]) -> FocusRegion:
        """Compute dynamic bounding box around motion"""
        height, width = motion_map.shape
        
        # Use higher threshold to filter out noise
        motion_threshold = 0.3  # Much higher to ignore camera noise
        
        # Find strongest motion point instead of all motion
        max_motion = np.max(motion_map)
        
        if max_motion > motion_threshold:
            # Find peak motion location
            y_peak, x_peak = np.unravel_index(np.argmax(motion_map), motion_map.shape)
            
            # Calculate box size based on motion spread
            # Find all significant motion points
            motion_points = np.where(motion_map > motion_threshold * 0.5)
            
            if len(motion_points[0]) > 0:
                # Calculate spread of motion
                y_min, y_max = np.min(motion_points[0]), np.max(motion_points[0])
                x_min, x_max = np.min(motion_points[1]), np.max(motion_points[1])
                
                # Box size based on motion spread
                spread_w = x_max - x_min
                spread_h = y_max - y_min
                
                # Adaptive size with limits
                box_size = max(self.min_size, 
                              min(self.max_size, 
                                  int(max(spread_w, spread_h) * 1.5)))
            else:
                box_size = 128  # Default
            
            half_size = box_size // 2
            
            x = max(0, x_peak - half_size)
            y = max(0, y_peak - half_size)
            w = min(box_size, width - x)
            h = min(box_size, height - y)
            
            # Smooth transitions if we have previous position
            if current_focus and self.last_box:
                alpha = self.smoothing
                prev_x, prev_y, prev_w, prev_h = self.last_box
                x = int(prev_x * alpha + x * (1 - alpha))
                y = int(prev_y * alpha + y * (1 - alpha))
                w = int(prev_w * alpha + w * (1 - alpha))
                h = int(prev_h * alpha + h * (1 - alpha))
            
            self.last_box = (x, y, w, h)
            
            return FocusRegion(
                x=x, y=y, width=w, height=h,
                confidence=max_motion,
                state=AttentionState.PROCESSING if max_motion > 0.5 else AttentionState.TRACKING,
                strategy=AttentionStrategy.DYNAMIC_BOX
            )
        else:
            # No significant motion - maintain position or center
            if self.last_box:
                # Keep last position but mark as idle
                x, y, w, h = self.last_box
                return FocusRegion(
                    x=x, y=y, width=w, height=h,
                    confidence=0.0,
                    state=AttentionState.IDLE,
                    strategy=AttentionStrategy.DYNAMIC_BOX
                )
            else:
                # Return to center with default size
                default_size = 128
                return FocusRegion(
                    x=(width - default_size) // 2,
                    y=(height - default_size) // 2,
                    width=default_size,
                    height=default_size,
                    confidence=0.0,
                    state=AttentionState.IDLE,
                    strategy=AttentionStrategy.DYNAMIC_BOX
                )
    
    def get_default_focus(self, frame_width: int, frame_height: int) -> FocusRegion:
        """Get centered box"""
        size = 128
        return FocusRegion(
            x=(frame_width - size) // 2,
            y=(frame_height - size) // 2,
            width=size,
            height=size,
            confidence=0.0,
            state=AttentionState.IDLE,
            strategy=AttentionStrategy.DYNAMIC_BOX
        )


class VisionAttentionPlugin(IRPPlugin):
    """
    Single-sensor vision plugin with adaptive attention strategies
    """
    
    def __init__(self, 
                 sensor_id: int = 0,
                 strategy: AttentionStrategy = AttentionStrategy.ADAPTIVE,
                 enable_vae: bool = True,
                 vae_threshold: float = 0.5,
                 device: Optional[torch.device] = None):
        """
        Initialize vision attention plugin for a single sensor
        
        Args:
            sensor_id: Camera/sensor ID
            strategy: Attention strategy to use
            enable_vae: Whether to use VAE processing
            vae_threshold: Motion threshold to trigger VAE
            device: Torch device for computation
        """
        # Create config for base class
        config = {
            'entity_id': f'vision_sensor_{sensor_id}',
            'max_iterations': 1,  # Vision is single-pass
            'device': str(device),
            'trust_weight': 1.0
        }
        super().__init__(config)
        
        self.sensor_id = sensor_id
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_vae = enable_vae
        self.vae_threshold = vae_threshold
        
        # Attention strategies
        self.fixed_strategy = FixedTileStrategy()
        self.dynamic_strategy = DynamicBoxStrategy()
        self.current_strategy = strategy
        
        # Select initial strategy
        if strategy == AttentionStrategy.FIXED_TILES:
            self.strategy_impl = self.fixed_strategy
        elif strategy == AttentionStrategy.DYNAMIC_BOX:
            self.strategy_impl = self.dynamic_strategy
        else:  # ADAPTIVE
            self.strategy_impl = self.fixed_strategy  # Start with fixed
        
        # Current state
        self.focus_region = None
        self.prev_gray = None
        self.motion_history = []
        
        # VAE (lazy loaded)
        self.vae = None
        self.vae_loaded = False
        self.vae_last_used = 0
        self.vae_timeout = 5.0
        
        # Performance tracking
        self.frame_count = 0
        self.vae_inference_time = 0
        self.total_processing_time = 0
        
    def load_vae(self):
        """Lazy load VAE model"""
        if not self.vae_loaded and self.enable_vae:
            self.vae = TinyVAE(input_channels=3, latent_dim=16, img_size=64)
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            self.vae_loaded = True
            self.vae_last_used = time.time()
            print(f"VAE loaded for sensor {self.sensor_id}")
    
    def unload_vae(self):
        """Unload VAE to save memory"""
        if self.vae_loaded:
            del self.vae
            self.vae = None
            self.vae_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"VAE unloaded for sensor {self.sensor_id}")
    
    def compute_motion(self, frame: np.ndarray) -> np.ndarray:
        """Compute motion map from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray, dtype=np.float32)
        
        # Motion detection
        diff = cv2.absdiff(self.prev_gray, gray)
        
        # Debug: check raw diff values
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Don't use sigmoid - it might be squashing values too much
        # Just normalize to 0-1
        motion = diff.astype(np.float32) / 255.0
        
        # Optional: amplify weak motion
        motion = np.minimum(motion * 2.0, 1.0)
        
        self.prev_gray = gray
        return motion
    
    def switch_strategy(self, strategy: AttentionStrategy):
        """Switch attention strategy"""
        self.current_strategy = strategy
        
        if strategy == AttentionStrategy.FIXED_TILES:
            self.strategy_impl = self.fixed_strategy
        elif strategy == AttentionStrategy.DYNAMIC_BOX:
            self.strategy_impl = self.dynamic_strategy
        else:  # ADAPTIVE
            # Choose based on scene characteristics
            # For now, default to fixed
            self.strategy_impl = self.fixed_strategy
    
    def extract_crop(self, frame: np.ndarray, region: FocusRegion, size: int = 64) -> np.ndarray:
        """Extract crop from focus region"""
        # Get center of region
        cx = region.x + region.width // 2
        cy = region.y + region.height // 2
        
        # Extract crop
        half = size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(frame.shape[1], x1 + size)
        y2 = min(frame.shape[0], y1 + size)
        
        crop = frame[y1:y2, x1:x2]
        
        # Resize if needed
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size))
        
        return crop
    
    def process_with_vae(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process crop through VAE"""
        self.load_vae()
        self.vae_last_used = time.time()
        
        # Convert to tensor
        crop_tensor = torch.from_numpy(crop).float() / 255.0
        crop_tensor = crop_tensor.permute(2, 0, 1).unsqueeze(0)
        crop_tensor = crop_tensor.to(self.device)
        
        # VAE forward pass
        start_time = time.time()
        with torch.no_grad():
            mu, logvar = self.vae.encode(crop_tensor)
            latent = self.vae.reparameterize(mu, logvar)
            reconstruction = self.vae.decode(latent)
        self.vae_inference_time = time.time() - start_time
        
        # Convert back
        latent_np = latent.cpu().numpy().squeeze()
        recon_np = reconstruction.cpu().numpy().squeeze()
        recon_np = np.transpose(recon_np, (1, 2, 0))
        recon_np = (recon_np * 255).astype(np.uint8)
        
        return latent_np, recon_np
    
    def refine(self, input_data: Any, max_iterations: int = 5) -> Tuple[Any, Dict[str, Any]]:
        """
        Main IRP refinement - process single frame
        
        Args:
            input_data: Frame from camera (numpy array)
            max_iterations: Not used for vision (single pass)
            
        Returns:
            Tuple of (focus_region, telemetry)
        """
        start_time = time.time()
        
        # Ensure we have a frame
        if not isinstance(input_data, np.ndarray):
            return None, {"error": "Invalid input"}
        
        frame = input_data
        self.frame_count += 1
        
        # Compute motion map
        motion = self.compute_motion(frame)
        
        # Get focus region from strategy
        if self.focus_region is None:
            self.focus_region = self.strategy_impl.get_default_focus(
                frame.shape[1], frame.shape[0]
            )
        
        self.focus_region = self.strategy_impl.compute_focus(motion, self.focus_region)
        
        # Adaptive strategy switching
        if self.current_strategy == AttentionStrategy.ADAPTIVE:
            # Simple heuristic: use dynamic for high motion variance
            motion_variance = np.var(motion)
            if motion_variance > 0.1 and isinstance(self.strategy_impl, FixedTileStrategy):
                self.switch_strategy(AttentionStrategy.DYNAMIC_BOX)
            elif motion_variance < 0.05 and isinstance(self.strategy_impl, DynamicBoxStrategy):
                self.switch_strategy(AttentionStrategy.FIXED_TILES)
        
        # VAE processing if enabled and motion detected
        latent = None
        reconstruction = None
        if self.enable_vae and self.focus_region.confidence > self.vae_threshold:
            crop = self.extract_crop(frame, self.focus_region)
            latent, reconstruction = self.process_with_vae(crop)
            self.focus_region.latent = latent
        
        # Check VAE timeout
        if self.vae_loaded and (time.time() - self.vae_last_used) > self.vae_timeout:
            self.unload_vae()
        
        # Build telemetry
        self.total_processing_time = time.time() - start_time
        
        telemetry = {
            "sensor_id": self.sensor_id,
            "frame": self.frame_count,
            "focus_x": self.focus_region.x,
            "focus_y": self.focus_region.y,
            "focus_width": self.focus_region.width,
            "focus_height": self.focus_region.height,
            "confidence": self.focus_region.confidence,
            "state": self.focus_region.state.value,
            "strategy": self.focus_region.strategy.value,
            "vae_loaded": self.vae_loaded,
            "vae_time_ms": self.vae_inference_time * 1000,
            "total_time_ms": self.total_processing_time * 1000,
            "has_latent": latent is not None
        }
        
        # Return focus region and telemetry
        return self.focus_region, telemetry
    
    def get_state(self) -> Dict[str, Any]:
        """Get plugin state"""
        return {
            "sensor_id": self.sensor_id,
            "strategy": self.current_strategy.value,
            "vae_enabled": self.enable_vae,
            "vae_loaded": self.vae_loaded,
            "frame_count": self.frame_count,
            "focus": {
                "x": self.focus_region.x if self.focus_region else 0,
                "y": self.focus_region.y if self.focus_region else 0,
                "width": self.focus_region.width if self.focus_region else 0,
                "height": self.focus_region.height if self.focus_region else 0,
                "confidence": self.focus_region.confidence if self.focus_region else 0.0
            }
        }
    
    def trust_adjustment(self) -> float:
        """Compute trust based on consistency"""
        # Higher trust when focus is stable and confident
        if self.focus_region and self.focus_region.state == AttentionState.PROCESSING:
            return 0.9  # High trust when actively processing
        elif self.focus_region and self.focus_region.state == AttentionState.TRACKING:
            return 0.7  # Medium trust when tracking
        else:
            return 0.5  # Lower trust when idle
    
    def shutdown(self):
        """Cleanup on shutdown"""
        if self.vae_loaded:
            self.unload_vae()


def create_vision_attention_plugin(
    sensor_id: int = 0,
    strategy: str = "adaptive",
    enable_vae: bool = True,
    device: Optional[torch.device] = None
) -> VisionAttentionPlugin:
    """
    Factory function to create vision attention plugin
    
    Args:
        sensor_id: Camera/sensor ID
        strategy: "fixed_tiles", "dynamic_box", or "adaptive"
        enable_vae: Whether to enable VAE processing
        device: Torch device
        
    Returns:
        Configured VisionAttentionPlugin instance
    """
    strategy_map = {
        "fixed_tiles": AttentionStrategy.FIXED_TILES,
        "dynamic_box": AttentionStrategy.DYNAMIC_BOX,
        "adaptive": AttentionStrategy.ADAPTIVE
    }
    
    strategy_enum = strategy_map.get(strategy, AttentionStrategy.ADAPTIVE)
    
    return VisionAttentionPlugin(
        sensor_id=sensor_id,
        strategy=strategy_enum,
        enable_vae=enable_vae,
        device=device
    )