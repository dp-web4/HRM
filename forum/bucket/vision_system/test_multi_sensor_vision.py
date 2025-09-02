#!/usr/bin/env python3
"""
Test multi-sensor vision with single plugin per sensor
Demonstrates adaptive strategy switching
"""

import cv2
import numpy as np
import sys
import os
import time
from typing import List, Dict, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.vision_attention_plugin import (
    create_vision_attention_plugin, 
    AttentionStrategy
)


class MultiSensorVisionSystem:
    """Manages multiple vision plugins for multiple cameras"""
    
    def __init__(self, num_cameras: int = 2):
        """Initialize with specified number of cameras"""
        self.num_cameras = num_cameras
        self.plugins: List = []
        self.captures: List = []
        
        # Create plugins for each camera
        for i in range(num_cameras):
            # Alternate strategies for demonstration
            if i == 0:
                strategy = "fixed_tiles"  # Left camera: fixed tiles
            else:
                strategy = "dynamic_box"  # Right camera: dynamic box
            
            plugin = create_vision_attention_plugin(
                sensor_id=i,
                strategy=strategy,
                enable_vae=True
            )
            self.plugins.append(plugin)
        
        print(f"Created {num_cameras} vision plugins with different strategies")
    
    def create_gst_pipeline(self, sensor_id=0):
        """Create GStreamer pipeline for CSI camera"""
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
            f"video/x-raw(memory:NVMM), width=1920, height=1080, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
    
    def setup_cameras(self):
        """Setup camera captures"""
        for i in range(self.num_cameras):
            print(f"Setting up camera {i}...")
            cap = cv2.VideoCapture(self.create_gst_pipeline(i), cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {i}")
            
            self.captures.append(cap)
            print(f"✓ Camera {i} ready")
    
    def draw_plugin_overlay(self, frame: np.ndarray, plugin_idx: int, 
                          focus_region: any, telemetry: Dict) -> np.ndarray:
        """Draw overlay for a single plugin's output"""
        display = frame.copy()
        
        if focus_region:
            # Draw focus region
            x, y = focus_region.x, focus_region.y
            w, h = focus_region.width, focus_region.height
            
            # Color based on state
            if focus_region.state.value == "processing":
                color = (0, 0, 255)  # Red
                thickness = 3
            elif focus_region.state.value == "tracking":
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (128, 128, 128)  # Gray
                thickness = 1
            
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            
            # Add center cross
            cx, cy = x + w // 2, y + h // 2
            cv2.line(display, (cx - 10, cy), (cx + 10, cy), color, 1)
            cv2.line(display, (cx, cy - 10), (cx, cy + 10), color, 1)
        
        # Add labels
        label = f"Sensor {plugin_idx}: {telemetry.get('strategy', 'unknown')}"
        cv2.putText(display, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        state_text = f"State: {telemetry.get('state', 'unknown')}"
        cv2.putText(display, state_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        conf_text = f"Confidence: {telemetry.get('confidence', 0):.3f}"
        cv2.putText(display, conf_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show if VAE is active
        if telemetry.get('has_latent'):
            cv2.putText(display, "VAE Active", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return display
    
    def compare_outputs(self, results: List[Tuple]) -> Dict:
        """Compare outputs from multiple plugins"""
        comparison = {
            "agreement": False,
            "consensus_region": None,
            "confidence_avg": 0
        }
        
        if len(results) < 2:
            return comparison
        
        # Extract focus regions
        regions = [r[0] for r in results if r[0] is not None]
        if not regions:
            return comparison
        
        # Check if regions overlap (simple IoU)
        if len(regions) >= 2:
            r1, r2 = regions[0], regions[1]
            
            # Calculate intersection
            x1 = max(r1.x, r2.x)
            y1 = max(r1.y, r2.y)
            x2 = min(r1.x + r1.width, r2.x + r2.width)
            y2 = min(r1.y + r1.height, r2.y + r2.height)
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = r1.width * r1.height
                area2 = r2.width * r2.height
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                comparison["agreement"] = iou > 0.3
        
        # Average confidence
        confidences = [r.confidence for r in regions]
        comparison["confidence_avg"] = np.mean(confidences)
        
        return comparison
    
    def run(self):
        """Main execution loop"""
        print("\n" + "=" * 60)
        print("Multi-Sensor Vision System")
        print("Left: Fixed tiles | Right: Dynamic box")
        print("=" * 60)
        print("\nControls:")
        print("  Q: Quit")
        print("  S: Switch strategies")
        print("  V: Toggle VAE")
        print("-" * 60)
        
        self.setup_cameras()
        
        # Create window
        cv2.namedWindow("Multi-Sensor Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Sensor Vision", 1920, 540)
        
        frame_count = 0
        strategy_swap = False
        
        while True:
            # Capture from all cameras
            frames = []
            for cap in self.captures:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture from camera")
                    break
                frames.append(frame)
            
            if len(frames) != self.num_cameras:
                break
            
            # Process through plugins
            results = []
            displays = []
            
            for i, (frame, plugin) in enumerate(zip(frames, self.plugins)):
                # Run plugin refinement
                focus_region, telemetry = plugin.refine(frame)
                results.append((focus_region, telemetry))
                
                # Draw overlay
                display = self.draw_plugin_overlay(frame, i, focus_region, telemetry)
                displays.append(display)
            
            # Compare outputs
            comparison = self.compare_outputs(results)
            
            # Combine displays
            combined = np.hstack(displays)
            
            # Add comparison info
            comp_text = f"Agreement: {'YES' if comparison['agreement'] else 'NO'}"
            cv2.putText(combined, comp_text, (960 - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if comparison['agreement'] else (0, 0, 255), 2)
            
            avg_conf_text = f"Avg Confidence: {comparison['confidence_avg']:.3f}"
            cv2.putText(combined, avg_conf_text, (960 - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show combined
            cv2.imshow("Multi-Sensor Vision", combined)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Swap strategies between cameras
                strategy_swap = not strategy_swap
                if strategy_swap:
                    self.plugins[0].switch_strategy(AttentionStrategy.DYNAMIC_BOX)
                    self.plugins[1].switch_strategy(AttentionStrategy.FIXED_TILES)
                else:
                    self.plugins[0].switch_strategy(AttentionStrategy.FIXED_TILES)
                    self.plugins[1].switch_strategy(AttentionStrategy.DYNAMIC_BOX)
                print(f"Strategies swapped!")
            elif key == ord('v'):
                # Toggle VAE on all plugins
                for plugin in self.plugins:
                    plugin.enable_vae = not plugin.enable_vae
                    if not plugin.enable_vae:
                        plugin.unload_vae()
                print(f"VAE toggled to {self.plugins[0].enable_vae}")
            
            frame_count += 1
            
            # Print periodic status
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: ", end="")
                for i, (_, telemetry) in enumerate(results):
                    print(f"S{i}[{telemetry['state'][:4]}] ", end="")
                print(f"| Agree: {comparison['agreement']}")
        
        # Cleanup
        print("\nCleaning up...")
        for cap in self.captures:
            cap.release()
        for plugin in self.plugins:
            plugin.shutdown()
        cv2.destroyAllWindows()
        print("Done!")


def main():
    try:
        system = MultiSensorVisionSystem(num_cameras=2)
        system.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()