#!/usr/bin/env python3
"""
Camera Pipeline with Visual Monitor IRP Plugin
Integrates camera input with vision IRP and visual monitoring
"""

import cv2
import torch
import numpy as np
import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP
from sage.irp.plugins.vision_impl import create_vision_irp
from sage.irp.plugins.visual_monitor_impl import create_visual_monitor_irp


class CameraIRPPipeline:
    """
    Pipeline that processes camera feed through IRP plugins with visualization
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing Camera IRP Pipeline on {self.device}...")
        
        # Initialize components
        self.memory_bridge = IRPMemoryBridge(
            buffer_size=50,
            consolidation_threshold=20
        )
        
        # Create IRP plugins
        self.vision_irp = create_vision_irp(self.device)
        self.vision_guided = MemoryGuidedIRP(self.vision_irp, self.memory_bridge)
        
        # Create visual monitor plugin
        self.monitor_irp = create_visual_monitor_irp(
            device=self.device,
            show_window=True,
            display_width=1280,
            display_height=720
        )
        
        # Create orchestrator
        self.orchestrator = HRMOrchestrator(
            initial_atp=1000.0,
            max_concurrent=2
        )
        
        # Register plugins
        self.orchestrator.register_plugin("vision", self.vision_guided, initial_trust=1.0)
        self.orchestrator.register_plugin("monitor", self.monitor_irp, initial_trust=1.0)
        
        # Camera setup
        self.cap = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize_camera(self):
        """Initialize camera"""
        print(f"Initializing camera {self.camera_id}...")
        
        # Try CSI camera first (Jetson)
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={self.camera_id} ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("CSI camera not available, trying USB camera...")
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
        print("Camera initialized successfully")
        
    async def process_frame_async(self, frame: np.ndarray):
        """Process frame through IRP pipeline asynchronously"""
        # Convert frame to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device) / 255.0
        
        # Execute through orchestrator
        tasks = {
            "vision": frame_tensor,
            "monitor": frame  # Send original frame to monitor
        }
        
        results = await self.orchestrator.execute_parallel(tasks, early_stop=True)
        
        return results
        
    def process_frame_sync(self, frame: np.ndarray):
        """Process frame synchronously (simpler but slower)"""
        # First process through vision IRP
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device) / 255.0
        
        refined, vision_telemetry = self.vision_guided.refine(frame_tensor, early_stop=True)
        
        # Then visualize
        _, monitor_telemetry = self.monitor_irp.refine(frame)
        
        return vision_telemetry, monitor_telemetry
        
    async def run_async(self):
        """Run pipeline asynchronously"""
        self.initialize_camera()
        
        print("Starting async pipeline... Press 'Q' in window to quit")
        
        process_every_n = 3
        frame_counter = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame_counter += 1
            
            # Process periodically
            if frame_counter % process_every_n == 0:
                try:
                    results = await self.process_frame_async(frame)
                    
                    # Print telemetry periodically
                    if frame_counter % 30 == 0:
                        self.print_telemetry(results)
                        
                except Exception as e:
                    print(f"Processing error: {e}")
                    
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            self.frame_count += 1
            
        self.cleanup()
        
    def run_sync(self):
        """Run pipeline synchronously"""
        self.initialize_camera()
        
        print("Starting sync pipeline... Press 'Q' in window to quit")
        
        process_every_n = 3
        frame_counter = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame_counter += 1
            
            # Process periodically
            if frame_counter % process_every_n == 0:
                try:
                    vision_tel, monitor_tel = self.process_frame_sync(frame)
                    
                    # Print telemetry periodically
                    if frame_counter % 30 == 0:
                        print(f"Frame {frame_counter}: "
                              f"Vision iterations: {vision_tel['iterations']}, "
                              f"Compute saved: {vision_tel['compute_saved']*100:.1f}%, "
                              f"Display FPS: {monitor_tel.get('display_fps', 0):.1f}")
                              
                except Exception as e:
                    print(f"Processing error: {e}")
                    
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            self.frame_count += 1
            
        self.cleanup()
        
    def print_telemetry(self, results):
        """Print telemetry from results"""
        print(f"\nFrame {self.frame_count} Telemetry:")
        for result in results:
            print(f"  {result.plugin_id}:")
            print(f"    State: {result.state}")
            print(f"    ATP consumed: {result.atp_consumed:.2f}")
            print(f"    Efficiency: {result.efficiency:.4f}")
            if hasattr(result, 'telemetry'):
                if 'compute_saved' in result.telemetry:
                    print(f"    Compute saved: {result.telemetry['compute_saved']*100:.1f}%")
                if 'display_fps' in result.telemetry:
                    print(f"    Display FPS: {result.telemetry['display_fps']:.1f}")
                    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        
        if self.cap:
            self.cap.release()
            
        self.monitor_irp.cleanup()
        
        # Print final stats
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\nSession Statistics:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Elapsed time: {elapsed:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        
        # Memory stats
        mem_stats = self.memory_bridge.get_memory_stats()
        print(f"  Memories collected: {mem_stats['total_memories']}")
        print(f"  Patterns extracted: {mem_stats['patterns_extracted']}")
        
        # Orchestrator stats
        orch_summary = self.orchestrator.get_orchestration_summary()
        print(f"  Plugins executed: {orch_summary['plugins_executed']}")
        print(f"  ATP utilization: {orch_summary['budget_report']['utilization']*100:.1f}%")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Camera IRP Pipeline with Visual Monitoring")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Camera pipeline with IRP processing')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--async', action='store_true', help='Use async processing')
    
    args = parser.parse_args()
    
    pipeline = CameraIRPPipeline(camera_id=args.camera)
    
    try:
        if args.async:
            print("Using async processing...")
            asyncio.run(pipeline.run_async())
        else:
            print("Using sync processing...")
            pipeline.run_sync()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()