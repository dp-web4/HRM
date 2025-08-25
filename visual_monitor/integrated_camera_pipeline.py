#!/usr/bin/env python3
"""
Integrated Camera Pipeline with GPU Mailbox Communication
Connects camera sensors, vision IRP, and visual monitor via publish/subscribe
"""

import asyncio
import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP
from sage.irp.plugins.vision_impl import create_vision_irp
from sage.irp.plugins.camera_sensor_impl import create_dual_camera_sensor, CameraSensorIRP
from sage.irp.plugins.visual_monitor_effector import create_visual_monitor_effector
from sage.mailbox.pubsub_mailbox import PubSubMailbox, GPUMailboxBridge, Message


class IntegratedCameraPipeline:
    """
    Complete pipeline with camera sensors -> vision processing -> visual monitor
    All connected via GPU mailbox pub/sub
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Integrated Camera Pipeline on {self.device}")
        
        # Create shared mailbox
        self.mailbox = PubSubMailbox(self.device)
        self.bridge = GPUMailboxBridge(self.mailbox)
        
        # Create memory bridge
        self.memory_bridge = IRPMemoryBridge(
            buffer_size=50,
            consolidation_threshold=20
        )
        
        # Initialize components
        self._setup_cameras()
        self._setup_vision_processing()
        self._setup_monitor()
        self._setup_orchestrator()
        
        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        
    def _setup_cameras(self):
        """Setup camera sensors"""
        print("Setting up camera sensors...")
        
        try:
            # Try dual CSI cameras
            self.dual_camera = create_dual_camera_sensor(mailbox=self.mailbox)
            print("✓ Dual CSI cameras initialized")
        except Exception as e:
            import traceback
            print(f"Failed to initialize CSI cameras: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("Falling back to simulated cameras...")
            
            # Create simulated cameras
            self._setup_simulated_cameras()
            
    def _setup_simulated_cameras(self):
        """Setup simulated camera sensors for testing"""
        import numpy as np
        import cv2
        
        class SimulatedCamera(CameraSensorIRP):
            def __init__(self, config):
                super().__init__(config)
                self.sim_frame_count = 0
                
            def _capture_loop(self):
                """Generate simulated frames"""
                while self.running:
                    # Create test pattern
                    frame = np.zeros((540, 960, 3), dtype=np.uint8)
                    
                    # Add moving circle
                    x = 480 + int(200 * np.sin(self.sim_frame_count * 0.1))
                    y = 270 + int(100 * np.cos(self.sim_frame_count * 0.1))
                    color = (255, 0, 0) if self.sensor_id == 0 else (0, 0, 255)
                    cv2.circle(frame, (x, y), 50, color, -1)
                    
                    # Add text
                    cv2.putText(frame, f"Simulated Camera {self.sensor_id}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    self.latest_frame = frame
                    self.frame_count += 1
                    self.sim_frame_count += 1
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                        
                    time.sleep(1.0 / self.fps)
                    
        # Create simulated dual camera
        self.cameras = []
        
        for i in range(2):
            config = {
                'sensor_id': i,
                'type': 'csi',  # Pretend to be CSI
                'display_resolution': (960, 540),
                'fps': 30,
                'device': self.device,
                'mailbox': self.mailbox
            }
            
            camera = SimulatedCamera(config)
            camera.initialize()
            self.cameras.append(camera)
            
        print("✓ Simulated cameras initialized")
        
    def _setup_vision_processing(self):
        """Setup vision IRP processors"""
        print("Setting up vision processing...")
        
        # Create vision IRP for each camera
        self.left_vision = create_vision_irp(self.device)
        self.left_vision_guided = MemoryGuidedIRP(self.left_vision, self.memory_bridge)
        
        self.right_vision = create_vision_irp(self.device)
        self.right_vision_guided = MemoryGuidedIRP(self.right_vision, self.memory_bridge)
        
        # Subscribe to camera feeds
        self.mailbox.subscribe(
            "camera/csi/0",
            "left_vision_processor",
            self._process_left_camera
        )
        
        self.mailbox.subscribe(
            "camera/csi/1",
            "right_vision_processor",
            self._process_right_camera
        )
        
        print("✓ Vision processing initialized")
        
    def _process_left_camera(self, message: Message):
        """Process left camera frame through vision IRP"""
        try:
            # Convert frame to tensor
            frame = message.data
            if isinstance(frame, np.ndarray):
                import cv2
                frame_resized = cv2.resize(frame, (224, 224))
                frame_tensor = torch.from_numpy(frame_resized).float()
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            elif isinstance(frame, torch.Tensor):
                # If already a tensor, ensure correct shape [B, C, H, W]
                if frame.dim() == 3:  # [H, W, C]
                    frame_tensor = frame.permute(2, 0, 1).unsqueeze(0)
                elif frame.dim() == 4 and frame.shape[3] == 3:  # [B, H, W, C]
                    frame_tensor = frame.permute(0, 3, 1, 2)
                else:
                    frame_tensor = frame
                # Resize if needed
                if frame_tensor.shape[2:] != (224, 224):
                    frame_tensor = torch.nn.functional.interpolate(
                        frame_tensor.float(),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )
                frame_tensor = frame_tensor.to(self.device)
                if frame_tensor.max() > 1.0:
                    frame_tensor = frame_tensor / 255.0
            else:
                frame_tensor = frame
                
            # Process through vision IRP
            refined, telemetry = self.left_vision_guided.refine(frame_tensor, early_stop=True)
            
            # Extract attention (difference between input and refined)
            with torch.no_grad():
                diff = torch.abs(refined - frame_tensor)
                attention = torch.mean(diff, dim=1, keepdim=True)
                attention = torch.nn.functional.interpolate(
                    attention,
                    size=(540, 960),
                    mode='bilinear',
                    align_corners=False
                )
                
            # Publish attention map
            self.mailbox.publish(
                "attention/left",
                "left_vision",
                attention.squeeze().cpu().numpy()
            )
            
        except Exception as e:
            print(f"Left vision processing error: {e}")
            
    def _process_right_camera(self, message: Message):
        """Process right camera frame through vision IRP"""
        try:
            # Convert frame to tensor
            frame = message.data
            if isinstance(frame, np.ndarray):
                import cv2
                frame_resized = cv2.resize(frame, (224, 224))
                frame_tensor = torch.from_numpy(frame_resized).float()
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            elif isinstance(frame, torch.Tensor):
                # If already a tensor, ensure correct shape [B, C, H, W]
                if frame.dim() == 3:  # [H, W, C]
                    frame_tensor = frame.permute(2, 0, 1).unsqueeze(0)
                elif frame.dim() == 4 and frame.shape[3] == 3:  # [B, H, W, C]
                    frame_tensor = frame.permute(0, 3, 1, 2)
                else:
                    frame_tensor = frame
                # Resize if needed
                if frame_tensor.shape[2:] != (224, 224):
                    frame_tensor = torch.nn.functional.interpolate(
                        frame_tensor.float(),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )
                frame_tensor = frame_tensor.to(self.device)
                if frame_tensor.max() > 1.0:
                    frame_tensor = frame_tensor / 255.0
            else:
                frame_tensor = frame
                
            # Process through vision IRP
            refined, telemetry = self.right_vision_guided.refine(frame_tensor, early_stop=True)
            
            # Extract attention
            with torch.no_grad():
                diff = torch.abs(refined - frame_tensor)
                attention = torch.mean(diff, dim=1, keepdim=True)
                attention = torch.nn.functional.interpolate(
                    attention,
                    size=(540, 960),
                    mode='bilinear',
                    align_corners=False
                )
                
            # Publish attention map
            self.mailbox.publish(
                "attention/right",
                "right_vision",
                attention.squeeze().cpu().numpy()
            )
            
        except Exception as e:
            print(f"Right vision processing error: {e}")
            
    def _setup_monitor(self):
        """Setup visual monitor effector"""
        print("Setting up visual monitor...")
        
        self.monitor = create_visual_monitor_effector(
            mailbox=self.mailbox,
            show_window=True,
            display_width=1920,
            display_height=540
        )
        
        print("✓ Visual monitor initialized")
        
    def _setup_orchestrator(self):
        """Setup orchestrator for plugin management"""
        print("Setting up orchestrator...")
        
        self.orchestrator = HRMOrchestrator(
            initial_atp=2000.0,
            max_concurrent=4
        )
        
        # Register plugins
        if hasattr(self, 'dual_camera'):
            self.orchestrator.register_plugin("dual_camera", self.dual_camera, initial_trust=1.0)
        else:
            # Register simulated cameras
            for i, camera in enumerate(self.cameras):
                self.orchestrator.register_plugin(f"camera_{i}", camera, initial_trust=1.0)
                
        self.orchestrator.register_plugin("left_vision", self.left_vision_guided, initial_trust=1.0)
        self.orchestrator.register_plugin("right_vision", self.right_vision_guided, initial_trust=1.0)
        self.orchestrator.register_plugin("monitor", self.monitor, initial_trust=1.0)
        
        print("✓ Orchestrator initialized")
        
    async def run_async(self):
        """Run pipeline asynchronously"""
        print("\nStarting integrated pipeline...")
        print("Controls: Q to quit, A to toggle attention, T to toggle telemetry")
        print("-" * 60)
        
        process_interval = 1.0 / 30  # 30 FPS target
        last_process_time = time.time()
        
        while self.monitor.show_window:
            current_time = time.time()
            
            # Process at target FPS
            if current_time - last_process_time >= process_interval:
                # Get camera frames
                if hasattr(self, 'dual_camera'):
                    tasks = {"dual_camera": None, "monitor": None}
                else:
                    tasks = {f"camera_{i}": None for i in range(len(self.cameras))}
                    tasks["monitor"] = None
                    
                # Execute all plugins
                try:
                    results = await self.orchestrator.execute_parallel(tasks, early_stop=True)
                    
                    # Print telemetry periodically
                    if self.frame_count % 30 == 0:
                        self.print_telemetry(results)
                        
                except Exception as e:
                    print(f"Pipeline error: {e}")
                    
                self.frame_count += 1
                last_process_time = current_time
                
            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.001)
            
        print("\nPipeline stopped")
        
    def print_telemetry(self, results):
        """Print telemetry from results"""
        print(f"\n--- Frame {self.frame_count} Telemetry ---")
        
        for result in results:
            if result.plugin_id == "monitor":
                print(f"Monitor: FPS={result.telemetry.get('display_fps', 0):.1f}")
            elif "camera" in result.plugin_id:
                print(f"{result.plugin_id}: FPS={result.telemetry.get('capture_fps', 0):.1f}")
            elif "vision" in result.plugin_id:
                print(f"{result.plugin_id}: Iterations={result.telemetry.get('iterations', 0)}, "
                      f"Saved={result.telemetry.get('compute_saved', 0)*100:.1f}%")
                      
        # Memory stats
        mem_stats = self.memory_bridge.get_memory_stats()
        print(f"Memory: {mem_stats['total_memories']} memories, "
              f"{mem_stats['patterns_extracted']} patterns")
              
    def cleanup(self):
        """Clean up all resources"""
        print("\nCleaning up...")
        
        # Cleanup cameras
        if hasattr(self, 'dual_camera'):
            self.dual_camera.cleanup()
        elif hasattr(self, 'cameras'):
            for camera in self.cameras:
                camera.cleanup()
                
        # Cleanup monitor
        self.monitor.cleanup()
        
        # Shutdown mailbox
        self.mailbox.shutdown()
        
        # Print final stats
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\nSession Statistics:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Elapsed time: {elapsed:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        
        # Orchestrator summary
        summary = self.orchestrator.get_orchestration_summary()
        print(f"  Plugins executed: {summary['plugins_executed']}")
        print(f"  ATP utilization: {summary['budget_report']['utilization']*100:.1f}%")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Integrated Camera Pipeline")
    print("Camera Sensors -> Vision IRP -> Visual Monitor")
    print("Connected via GPU Mailbox Publish/Subscribe")
    print("=" * 60)
    
    pipeline = IntegratedCameraPipeline()
    
    try:
        asyncio.run(pipeline.run_async())
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