#!/usr/bin/env python3
"""
Visual Attention Monitor for HRM/SAGE
Shows camera feeds with attention tracking and IRP refinement visualization
"""

import cv2
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import asyncio
from dataclasses import dataclass
import time
from threading import Thread, Lock
import queue

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.vision_impl import create_vision_irp
from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP


@dataclass
class AttentionRegion:
    """Represents a region of attention in the image"""
    x: int
    y: int
    width: int
    height: int
    intensity: float  # 0-1 attention strength
    label: str = ""


class VisualAttentionMonitor:
    """
    Real-time visual monitoring system with attention tracking
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        window_width: int = 1280,
        window_height: int = 720,
        device: Optional[torch.device] = None
    ):
        self.camera_id = camera_id
        self.window_width = window_width
        self.window_height = window_height
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize camera
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.attention_queue = queue.Queue(maxsize=10)
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        
        # Initialize IRP and orchestrator
        print("Initializing SAGE components...")
        self.memory_bridge = IRPMemoryBridge(
            buffer_size=50,
            consolidation_threshold=20
        )
        
        self.vision_irp = create_vision_irp(self.device)
        self.vision_guided = MemoryGuidedIRP(self.vision_irp, self.memory_bridge)
        
        self.orchestrator = HRMOrchestrator(
            initial_atp=1000.0,
            max_concurrent=1
        )
        self.orchestrator.register_plugin("vision", self.vision_guided, initial_trust=1.0)
        
        # Attention tracking
        self.attention_map = None
        self.attention_regions = []
        self.energy_history = []
        self.fps_history = []
        
        # Display settings
        self.show_attention = True
        self.show_energy = True
        self.show_telemetry = True
        self.show_memory_stats = True
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
    def start(self):
        """Start the visual monitoring system"""
        print("Starting Visual Attention Monitor...")
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            # Try different camera backends
            for backend in [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                if self.cap.isOpened():
                    print(f"Camera opened with backend: {backend}")
                    break
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        
        # Start capture thread
        self.capture_thread = Thread(target=self._capture_loop)
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = Thread(target=self._process_loop)
        self.process_thread.start()
        
        # Start display loop
        self._display_loop()
        
    def _capture_loop(self):
        """Continuously capture frames from camera"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Put frame in queue (drop old frames if full)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            else:
                time.sleep(0.001)
                
    def _process_loop(self):
        """Process frames through IRP and extract attention"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Process through IRP
                attention_data = self._process_frame_irp(frame)
                
                # Put attention data in queue
                try:
                    self.attention_queue.put_nowait(attention_data)
                except queue.Full:
                    try:
                        self.attention_queue.get_nowait()
                        self.attention_queue.put_nowait(attention_data)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
                
    def _process_frame_irp(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame through IRP and extract attention information"""
        start_time = time.time()
        
        # Convert frame to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device) / 255.0
        
        # Process through vision IRP
        refined, telemetry = self.vision_guided.refine(frame_tensor, early_stop=True)
        
        # Extract attention from latent space
        attention_map = self._extract_attention(refined, telemetry)
        
        # Find attention regions
        regions = self._find_attention_regions(attention_map)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        return {
            'attention_map': attention_map,
            'regions': regions,
            'telemetry': telemetry,
            'process_time': process_time,
            'energy': telemetry.get('energy_trajectory', [])[-1] if telemetry.get('energy_trajectory') else 0
        }
        
    def _extract_attention(self, refined: torch.Tensor, telemetry: Dict) -> np.ndarray:
        """Extract attention map from refined output"""
        # Get the VAE encoder's latent representation
        if hasattr(self.vision_irp.plugin, 'vae'):
            vae = self.vision_irp.plugin.vae
            with torch.no_grad():
                # Encode refined image to get latent
                mu, log_var = vae.encode(refined)
                latent = mu
                
                # Convert latent to attention map
                # Take mean across channels and upsample
                attention = torch.mean(torch.abs(latent), dim=1, keepdim=True)
                attention = torch.nn.functional.interpolate(
                    attention,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Normalize to 0-1
                attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
                
                return attention.squeeze().cpu().numpy()
        else:
            # Fallback: create dummy attention
            return np.random.rand(224, 224) * 0.5
            
    def _find_attention_regions(self, attention_map: np.ndarray, threshold: float = 0.6) -> List[AttentionRegion]:
        """Find regions of high attention"""
        regions = []
        
        # Threshold the attention map
        binary = (attention_map > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate mean attention in region
            region_attention = attention_map[y:y+h, x:x+w]
            intensity = np.mean(region_attention) if region_attention.size > 0 else 0
            
            # Scale coordinates back to original image size
            scale_x = self.window_width / 224
            scale_y = self.window_height / 224
            
            regions.append(AttentionRegion(
                x=int(x * scale_x),
                y=int(y * scale_y),
                width=int(w * scale_x),
                height=int(h * scale_y),
                intensity=float(intensity),
                label=f"Focus {len(regions)+1}"
            ))
            
        return regions
        
    def _display_loop(self):
        """Main display loop"""
        cv2.namedWindow("SAGE Visual Attention Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SAGE Visual Attention Monitor", self.window_width, self.window_height)
        
        last_frame = None
        last_attention_data = None
        
        while self.running:
            # Get latest frame
            try:
                frame = self.frame_queue.get_nowait()
                last_frame = frame
            except queue.Empty:
                frame = last_frame
                
            if frame is None:
                time.sleep(0.01)
                continue
                
            # Get latest attention data
            try:
                attention_data = self.attention_queue.get_nowait()
                last_attention_data = attention_data
            except queue.Empty:
                attention_data = last_attention_data
                
            # Create display frame
            display_frame = self._create_display_frame(frame, attention_data)
            
            # Show frame
            cv2.imshow("SAGE Visual Attention Monitor", display_frame)
            
            # Update FPS
            self._update_fps()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('a'):
                self.show_attention = not self.show_attention
            elif key == ord('e'):
                self.show_energy = not self.show_energy
            elif key == ord('t'):
                self.show_telemetry = not self.show_telemetry
            elif key == ord('m'):
                self.show_memory_stats = not self.show_memory_stats
            elif key == ord('c'):
                # Consolidate memory
                self.memory_bridge.consolidate()
                print("Memory consolidated")
                
    def _create_display_frame(self, frame: np.ndarray, attention_data: Optional[Dict]) -> np.ndarray:
        """Create the display frame with overlays"""
        # Resize frame to window size
        display = cv2.resize(frame, (self.window_width, self.window_height))
        
        if attention_data:
            # Draw attention overlay
            if self.show_attention and 'attention_map' in attention_data:
                attention_overlay = self._create_attention_overlay(
                    attention_data['attention_map'],
                    (self.window_width, self.window_height)
                )
                display = cv2.addWeighted(display, 0.7, attention_overlay, 0.3, 0)
                
            # Draw attention regions
            if 'regions' in attention_data:
                for region in attention_data['regions']:
                    self._draw_attention_region(display, region)
                    
            # Draw telemetry
            if self.show_telemetry and 'telemetry' in attention_data:
                self._draw_telemetry(display, attention_data['telemetry'], attention_data.get('process_time', 0))
                
        # Draw energy graph
        if self.show_energy:
            self._draw_energy_graph(display)
            
        # Draw memory stats
        if self.show_memory_stats:
            self._draw_memory_stats(display)
            
        # Draw FPS
        self._draw_fps(display)
        
        # Draw controls
        self._draw_controls(display)
        
        return display
        
    def _create_attention_overlay(self, attention_map: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Create colored overlay from attention map"""
        # Resize attention map
        attention_resized = cv2.resize(attention_map, size)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap
        
    def _draw_attention_region(self, frame: np.ndarray, region: AttentionRegion):
        """Draw attention region on frame"""
        # Draw rectangle
        color = (0, int(255 * region.intensity), int(255 * (1 - region.intensity)))
        thickness = max(1, int(3 * region.intensity))
        cv2.rectangle(
            frame,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            color,
            thickness
        )
        
        # Draw label
        if region.label:
            label_text = f"{region.label}: {region.intensity:.2f}"
            cv2.putText(
                frame,
                label_text,
                (region.x, region.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )
            
    def _draw_telemetry(self, frame: np.ndarray, telemetry: Dict, process_time: float):
        """Draw telemetry information"""
        y_offset = 30
        x_offset = 10
        
        lines = [
            f"Iterations: {telemetry.get('iterations', 'N/A')}",
            f"Compute Saved: {telemetry.get('compute_saved', 0)*100:.1f}%",
            f"Trust: {telemetry.get('trust', 0):.3f}",
            f"Process Time: {process_time:.1f}ms",
            f"Device: {self.device}"
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x_offset, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            
    def _draw_energy_graph(self, frame: np.ndarray):
        """Draw energy history graph"""
        if len(self.energy_history) < 2:
            return
            
        # Graph dimensions
        graph_width = 200
        graph_height = 100
        graph_x = self.window_width - graph_width - 20
        graph_y = 20
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (graph_x, graph_y),
            (graph_x + graph_width, graph_y + graph_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(
            frame,
            (graph_x, graph_y),
            (graph_x + graph_width, graph_y + graph_height),
            (255, 255, 255),
            1
        )
        
        # Draw energy curve
        max_points = min(len(self.energy_history), graph_width)
        recent_energy = self.energy_history[-max_points:]
        
        if recent_energy:
            min_e = min(recent_energy)
            max_e = max(recent_energy)
            range_e = max_e - min_e if max_e != min_e else 1
            
            points = []
            for i, energy in enumerate(recent_energy):
                x = graph_x + int(i * graph_width / max_points)
                y = graph_y + graph_height - int((energy - min_e) / range_e * graph_height)
                points.append((x, y))
                
            # Draw line
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)
                
        # Label
        cv2.putText(
            frame,
            "Energy",
            (graph_x + 5, graph_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
    def _draw_memory_stats(self, frame: np.ndarray):
        """Draw memory statistics"""
        stats = self.memory_bridge.get_memory_stats()
        
        y_offset = self.window_height - 150
        x_offset = 10
        
        lines = [
            "Memory System:",
            f"  Total: {stats.get('total_memories', 0)}",
            f"  Pending: {stats.get('pending_consolidation', 0)}",
            f"  Patterns: {stats.get('patterns_extracted', 0)}",
            f"  Avg Efficiency: {stats.get('avg_efficiency', 0):.3f}"
        ]
        
        for i, line in enumerate(lines):
            color = (200, 200, 200) if i == 0 else (150, 150, 150)
            cv2.putText(
                frame,
                line,
                (x_offset, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )
            
    def _draw_fps(self, frame: np.ndarray):
        """Draw FPS counter"""
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (self.window_width - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
    def _draw_controls(self, frame: np.ndarray):
        """Draw control instructions"""
        controls = [
            "Q: Quit | A: Attention | E: Energy",
            "T: Telemetry | M: Memory | C: Consolidate"
        ]
        
        y_offset = self.window_height - 30
        for i, control in enumerate(controls):
            cv2.putText(
                frame,
                control,
                (self.window_width // 2 - 150, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 100, 100),
                1,
                cv2.LINE_AA
            )
            
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            
            # Add to history
            self.fps_history.append(self.current_fps)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
                
    def stop(self):
        """Stop the monitoring system"""
        print("Stopping Visual Attention Monitor...")
        self.running = False
        
        # Wait for threads
        if self.capture_thread:
            self.capture_thread.join()
        if self.process_thread:
            self.process_thread.join()
            
        # Release resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        print("\nFinal Statistics:")
        stats = self.memory_bridge.get_memory_stats()
        print(f"  Memories: {stats.get('total_memories', 0)}")
        print(f"  Patterns: {stats.get('patterns_extracted', 0)}")
        if self.fps_history:
            print(f"  Avg FPS: {np.mean(self.fps_history):.1f}")
            

def main():
    """Main entry point"""
    print("=" * 60)
    print("SAGE Visual Attention Monitor")
    print("Real-time camera feed with IRP attention tracking")
    print("=" * 60)
    
    # Create monitor
    monitor = VisualAttentionMonitor(
        camera_id=0,  # Use default camera
        window_width=1280,
        window_height=720
    )
    
    try:
        # Start monitoring
        monitor.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        monitor.stop()
        
        
if __name__ == "__main__":
    main()