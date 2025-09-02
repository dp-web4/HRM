#!/usr/bin/env python3
"""
Optimized Camera Pipeline for Jetson
Memory-efficient version with attention tracking
"""

import asyncio
import torch
import numpy as np
import cv2
import time
import sys
import os
import gc
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
from sage.mailbox.pubsub_mailbox import PubSubMailbox, Message
from sage.irp.plugins.camera_sensor_impl import create_dual_camera_sensor
from sage.irp.plugins.visual_monitor_effector import create_visual_monitor_effector


class OptimizedCameraPipeline:
    """
    Lightweight pipeline with simple attention tracking
    Optimized for Jetson Orin Nano memory constraints
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Optimized Camera Pipeline on {self.device}")
        
        # Enable memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            # Use TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Create shared mailbox
        self.mailbox = PubSubMailbox(self.device)
        
        # Initialize components
        self._setup_cameras()
        self._setup_attention_processing()
        self._setup_monitor()
        self._setup_orchestrator()
        
        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.process_every_n = 5  # Process every 5th frame to save memory
        
        # Memory management
        self.last_gc_time = time.time()
        self.gc_interval = 30.0  # Run garbage collection every 30 seconds
        
    def _setup_cameras(self):
        """Setup camera sensors"""
        print("Setting up camera sensors...")
        
        try:
            # Try dual CSI cameras
            self.dual_camera = create_dual_camera_sensor(mailbox=self.mailbox)
            print("✓ Dual CSI cameras initialized")
        except Exception as e:
            print(f"Failed to initialize CSI cameras: {e}")
            raise
            
    def _setup_attention_processing(self):
        """Setup lightweight attention processing"""
        print("Setting up attention processing...")
        
        # Simple motion-based attention without heavy models
        self.prev_frames = {'left': None, 'right': None}
        
        # Subscribe to camera feeds
        self.mailbox.subscribe(
            "camera/csi/0",
            "attention_left",
            self._process_left_attention
        )
        
        self.mailbox.subscribe(
            "camera/csi/1",
            "attention_right",
            self._process_right_attention
        )
        
        print("✓ Attention processing initialized")
        
    def _process_left_attention(self, message: Message):
        """Simple attention based on motion/edges"""
        try:
            frame = message.data
            
            # Only process every Nth frame
            if self.frame_count % self.process_every_n != 0:
                return
                
            if isinstance(frame, np.ndarray):
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Simple attention: edges + motion
                edges = cv2.Canny(gray, 50, 150)
                
                if self.prev_frames['left'] is not None:
                    # Motion detection
                    motion = cv2.absdiff(self.prev_frames['left'], gray)
                    motion = cv2.GaussianBlur(motion, (21, 21), 0)
                    
                    # Combine edges and motion
                    attention = (edges.astype(np.float32) / 255.0 * 0.5 + 
                               motion.astype(np.float32) / 255.0 * 0.5)
                else:
                    attention = edges.astype(np.float32) / 255.0
                
                # Smooth attention map
                attention = cv2.GaussianBlur(attention, (15, 15), 0)
                
                # Resize to display resolution
                attention = cv2.resize(attention, (960, 540))
                
                # Publish attention
                self.mailbox.publish(
                    "attention/left",
                    "attention_processor",
                    attention
                )
                
                self.prev_frames['left'] = gray
                
        except Exception as e:
            print(f"Left attention error: {e}")
            
    def _process_right_attention(self, message: Message):
        """Simple attention based on motion/edges"""
        try:
            frame = message.data
            
            # Only process every Nth frame
            if self.frame_count % self.process_every_n != 0:
                return
                
            if isinstance(frame, np.ndarray):
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Simple attention: edges + motion
                edges = cv2.Canny(gray, 50, 150)
                
                if self.prev_frames['right'] is not None:
                    # Motion detection
                    motion = cv2.absdiff(self.prev_frames['right'], gray)
                    motion = cv2.GaussianBlur(motion, (21, 21), 0)
                    
                    # Combine edges and motion
                    attention = (edges.astype(np.float32) / 255.0 * 0.5 + 
                               motion.astype(np.float32) / 255.0 * 0.5)
                else:
                    attention = edges.astype(np.float32) / 255.0
                
                # Smooth attention map
                attention = cv2.GaussianBlur(attention, (15, 15), 0)
                
                # Resize to display resolution
                attention = cv2.resize(attention, (960, 540))
                
                # Publish attention
                self.mailbox.publish(
                    "attention/right",
                    "attention_processor",
                    attention
                )
                
                self.prev_frames['right'] = gray
                
        except Exception as e:
            print(f"Right attention error: {e}")
            
    def _setup_monitor(self):
        """Setup visual monitor"""
        print("Setting up visual monitor...")
        
        self.monitor = create_visual_monitor_effector(
            mailbox=self.mailbox,
            show_window=True,
            display_width=1920,
            display_height=540
        )
        
        print("✓ Visual monitor initialized")
        
    def _setup_orchestrator(self):
        """Setup orchestrator with plugins"""
        print("Setting up orchestrator...")
        
        self.orchestrator = HRMOrchestrator(
            device=self.device
        )
        
        # Register plugins
        self.orchestrator.register_plugin("dual_camera", self.dual_camera, initial_trust=1.0)
        self.orchestrator.register_plugin("monitor", self.monitor, initial_trust=1.0)
        
        print("✓ Orchestrator initialized")
        
    def _memory_cleanup(self):
        """Periodic memory cleanup"""
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.last_gc_time = current_time
            print(f"Memory cleanup at frame {self.frame_count}")
            
    async def run_async(self):
        """Run pipeline asynchronously"""
        print("\n" + "=" * 60)
        print("Optimized Camera Pipeline")
        print("Memory-efficient attention tracking")
        print("=" * 60)
        print("\nControls: Q to quit, A to toggle attention, T to toggle telemetry")
        print(f"Processing every {self.process_every_n} frames for efficiency")
        print("-" * 60)
        
        process_interval = 1.0 / 30  # 30 FPS target
        last_process_time = time.time()
        
        while self.monitor.show_window:
            current_time = time.time()
            
            # Process at target FPS
            if current_time - last_process_time >= process_interval:
                # Execute plugins
                tasks = {"dual_camera": None, "monitor": None}
                
                try:
                    results = await self.orchestrator.execute_parallel(tasks, early_stop=True)
                    
                    # Print telemetry periodically
                    if self.frame_count % 30 == 0:
                        self.print_telemetry(results)
                        
                except Exception as e:
                    print(f"Pipeline error: {e}")
                    
                self.frame_count += 1
                last_process_time = current_time
                
                # Periodic memory cleanup
                self._memory_cleanup()
                
            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.001)
            
        print("\nPipeline stopped")
        self.cleanup()
        
    def print_telemetry(self, results):
        """Print performance telemetry"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n--- Frame {self.frame_count} Telemetry ---")
        print(f"Pipeline FPS: {fps:.1f}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            print(f"GPU Memory: {memory_allocated:.1f}MB allocated, {memory_cached:.1f}MB cached")
            
        for plugin_id, (result, telemetry) in results.items():
            if telemetry:
                if 'display_fps' in telemetry:
                    print(f"{plugin_id}: FPS={telemetry['display_fps']:.1f}")
                    
    def run_sync(self):
        """Run synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_async())
        finally:
            loop.close()
            
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        # Shutdown cameras
        if hasattr(self, 'dual_camera'):
            self.dual_camera.shutdown()
            
        # Clean monitor
        if hasattr(self, 'monitor'):
            self.monitor.cleanup()
            
        # Clean mailbox
        if hasattr(self, 'mailbox'):
            self.mailbox.shutdown()
            
        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Cleanup complete")


def main():
    """Main entry point"""
    try:
        pipeline = OptimizedCameraPipeline()
        pipeline.run_sync()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()