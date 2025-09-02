#!/usr/bin/env python3
"""
Visual Monitor IRP Plugin
Provides real-time attention visualization as an effector plugin
"""

import torch
import cv2
import numpy as np
from typing import Any, Dict, Tuple, Optional, List
from dataclasses import dataclass
import time
from collections import deque
import threading
import queue

from sage.irp.base import IRPPlugin


@dataclass
class AttentionFrame:
    """Frame with attention data"""
    image: np.ndarray
    attention_map: Optional[np.ndarray]
    regions: List[Dict[str, Any]]
    telemetry: Dict[str, Any]
    timestamp: float


class VisualMonitorIRP(IRPPlugin):
    """
    Visual monitoring effector plugin
    Displays camera feed with attention overlay in real-time
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.entity_id = "visual_monitor_irp"
        
        # Configuration
        self.config = config or {}
        self.display_width = self.config.get('display_width', 1024)
        self.display_height = self.config.get('display_height', 600)
        self.show_window = self.config.get('show_window', True)
        self.save_frames = self.config.get('save_frames', False)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Display state
        self.window_name = "SAGE Visual Monitor Plugin"
        self.display_queue = queue.Queue(maxsize=2)
        self.running = False
        self.display_thread = None
        
        # Attention tracking
        self.attention_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)
        self.fps_history = deque(maxlen=30)
        
        # Performance
        self.last_display_time = time.time()
        self.frame_count = 0
        
        # Initialize display if needed
        if self.show_window:
            self._init_display()
            
    def _init_display(self):
        """Initialize display window and thread"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Process input and visualize attention
        
        Args:
            x: Input data (image tensor or camera frame)
            early_stop: Not used for monitoring
            
        Returns:
            Tuple of (processed frame, telemetry)
        """
        start_time = time.time()
        
        # Handle different input types
        if isinstance(x, torch.Tensor):
            # Convert tensor to image
            if x.dim() == 4:  # Batch dimension
                x = x[0]
            if x.dim() == 3:  # CHW format
                image = x.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype(np.uint8)
                if image.shape[2] == 1:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 3:  # RGB
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif isinstance(x, np.ndarray):
            image = x
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")
            
        # Extract attention using simple edge detection as proxy
        attention_map = self._compute_attention(image)
        
        # Find attention regions
        regions = self._find_attention_regions(attention_map)
        
        # Create telemetry
        process_time = (time.time() - start_time) * 1000
        telemetry = {
            'iterations': 1,  # Monitor doesn't iterate
            'compute_saved': 0.0,  # No compute saved, we're displaying
            'trust': 1.0,  # Always trusted
            'energy_trajectory': [0.5],  # Dummy energy
            'process_time_ms': process_time,
            'num_regions': len(regions),
            'display_fps': self._calculate_fps()
        }
        
        # Update history
        self.energy_history.append(telemetry['energy_trajectory'][-1])
        self.attention_history.append(np.mean(attention_map))
        
        # Create attention frame
        frame = AttentionFrame(
            image=image,
            attention_map=attention_map,
            regions=regions,
            telemetry=telemetry,
            timestamp=time.time()
        )
        
        # Queue for display
        if self.show_window:
            try:
                self.display_queue.put_nowait(frame)
            except queue.Full:
                # Drop old frame
                try:
                    self.display_queue.get_nowait()
                    self.display_queue.put_nowait(frame)
                except queue.Empty:
                    pass
                    
        # Save frame if requested
        if self.save_frames:
            self._save_frame(frame)
            
        # Return original image and telemetry
        return x, telemetry
        
    def _compute_attention(self, image: np.ndarray) -> np.ndarray:
        """
        Compute attention map from image
        Uses edge detection and saliency as proxy for attention
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute edges (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        attention = magnitude / (magnitude.max() + 1e-8)
        
        # Apply Gaussian blur for smoothness
        attention = cv2.GaussianBlur(attention, (15, 15), 5)
        
        return attention
        
    def _find_attention_regions(self, attention_map: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find regions of high attention"""
        regions = []
        
        # Threshold
        binary = (attention_map > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process top N largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Calculate mean attention in region
            region_attention = attention_map[y:y+h, x:x+w]
            intensity = np.mean(region_attention) if region_attention.size > 0 else 0
            
            regions.append({
                'id': i,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'intensity': float(intensity),
                'label': f"R{i+1}"
            })
            
        return regions
        
    def _display_loop(self):
        """Display thread loop"""
        while self.running:
            try:
                frame = self.display_queue.get(timeout=0.1)
                self._display_frame(frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display error: {e}")
                
    def _display_frame(self, frame: AttentionFrame):
        """Display a frame with overlays"""
        # Create display canvas
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Calculate layout
        main_width = int(self.display_width * 0.7)
        side_width = self.display_width - main_width
        
        # Resize and place main image
        h, w = frame.image.shape[:2]
        scale = min(main_width / w, self.display_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image_resized = cv2.resize(frame.image, (new_w, new_h))
        y_offset = (self.display_height - new_h) // 2
        
        # Apply attention overlay
        if frame.attention_map is not None:
            attention_resized = cv2.resize(frame.attention_map, (new_w, new_h))
            heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image_resized, 0.7, heatmap, 0.3, 0)
            canvas[y_offset:y_offset+new_h, 0:new_w] = overlay
        else:
            canvas[y_offset:y_offset+new_h, 0:new_w] = image_resized
            
        # Draw attention regions
        for region in frame.regions:
            rx = int(region['x'] * scale)
            ry = int(region['y'] * scale) + y_offset
            rw = int(region['width'] * scale)
            rh = int(region['height'] * scale)
            
            color = (0, int(255 * region['intensity']), int(255 * (1 - region['intensity'])))
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), color, 2)
            cv2.putText(canvas, region['label'], (rx, ry - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        # Draw side panel
        self._draw_side_panel(canvas, main_width, frame.telemetry)
        
        # Show
        cv2.imshow(self.window_name, canvas)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
            
    def _draw_side_panel(self, canvas: np.ndarray, x_start: int, telemetry: Dict[str, Any]):
        """Draw information panel"""
        # Background
        canvas[:, x_start:] = (30, 30, 30)
        
        # Title
        y = 30
        cv2.putText(canvas, "Visual Monitor IRP", (x_start + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        # Stats
        y += 40
        stats = [
            f"Device: {self.device}",
            f"FPS: {telemetry.get('display_fps', 0):.1f}",
            f"Process: {telemetry.get('process_time_ms', 0):.1f}ms",
            f"Regions: {telemetry.get('num_regions', 0)}",
        ]
        
        for stat in stats:
            cv2.putText(canvas, stat, (x_start + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += 25
            
        # Attention history graph
        if len(self.attention_history) > 1:
            self._draw_mini_graph(canvas, x_start + 10, y + 20,
                                list(self.attention_history), "Attention Level")
                                
    def _draw_mini_graph(self, canvas: np.ndarray, x: int, y: int, 
                        data: List[float], label: str):
        """Draw a small graph"""
        width = self.display_width - x - 20
        height = 60
        
        # Border
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (100, 100, 100), 1)
        
        # Label
        cv2.putText(canvas, label, (x + 5, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                   
        # Plot data
        if len(data) > 1:
            min_val = min(data)
            max_val = max(data)
            range_val = max_val - min_val if max_val != min_val else 1
            
            points = []
            for i, val in enumerate(data[-width:]):
                px = x + i
                py = y + height - int((val - min_val) / range_val * (height - 20)) - 10
                points.append((px, py))
                
            for i in range(1, len(points)):
                cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 1)
                
    def _calculate_fps(self) -> float:
        """Calculate display FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_display_time + 1e-6)
        self.last_display_time = current_time
        self.fps_history.append(fps)
        return np.mean(self.fps_history) if self.fps_history else 0.0
        
    def _save_frame(self, frame: AttentionFrame):
        """Save frame to disk"""
        timestamp = int(frame.timestamp * 1000)
        filename = f"visual_monitor/captures/frame_{timestamp}.jpg"
        
        # Create visualization
        if frame.attention_map is not None:
            heatmap = cv2.applyColorMap((frame.attention_map * 255).astype(np.uint8), 
                                       cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame.image, 0.7, heatmap, 0.3, 0)
            cv2.imwrite(filename, overlay)
        else:
            cv2.imwrite(filename, frame.image)
            
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.display_thread:
            self.display_thread.join()
        cv2.destroyAllWindows()
        

def create_visual_monitor_irp(
    device: Optional[torch.device] = None,
    show_window: bool = True,
    **kwargs
) -> VisualMonitorIRP:
    """
    Factory function to create visual monitor IRP
    
    Args:
        device: Torch device to use
        show_window: Whether to show display window
        **kwargs: Additional configuration
        
    Returns:
        VisualMonitorIRP instance
    """
    config = {
        'device': device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'show_window': show_window,
        **kwargs
    }
    
    return VisualMonitorIRP(config)


# Test function
def test_visual_monitor():
    """Test the visual monitor plugin"""
    print("Testing Visual Monitor IRP Plugin...")
    
    # Create monitor
    monitor = create_visual_monitor_irp(show_window=True)
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_image, (320, 240), 100, (255, 0, 0), -1)
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
    
    # Process multiple frames
    for i in range(100):
        # Add some variation
        modified = test_image.copy()
        cv2.circle(modified, (320 + i*2, 240), 20, (0, 0, 255), -1)
        
        # Process through monitor
        _, telemetry = monitor.refine(modified)
        
        if i % 10 == 0:
            print(f"Frame {i}: {telemetry['num_regions']} regions, "
                  f"{telemetry['display_fps']:.1f} fps")
                  
        time.sleep(0.03)  # ~30 fps
        
    # Cleanup
    monitor.cleanup()
    print("Test complete!")
    

if __name__ == "__main__":
    test_visual_monitor()