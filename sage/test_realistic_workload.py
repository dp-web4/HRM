#!/usr/bin/env python3
"""
Realistic GPU Workload Test for H↔L System with GR00T
Tests actual inference and training under sustained load.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import psutil
import GPUtil
import sys
import os
from typing import Dict, List, Tuple
from collections import deque
import threading
from dataclasses import dataclass

# Add paths
sys.path.insert(0, '/home/dp/ai-workspace/isaac-gr00t')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compression.integrated_h_l_system import IntegratedHLSystem
from groot_integration.groot_real_integration import GR00TWorldInterface, GR00TRealitySleepTrainer
from groot_integration.sleep_cycle_training import Experience, ExperienceMemory
from context.reality_context_4k import RealityContext4K

@dataclass
class SystemMetrics:
    """Track system performance metrics."""
    timestamp: float
    gpu_util: float
    gpu_memory: float
    gpu_temp: float
    inference_time: float
    training_loss: float
    experiences_processed: int
    

class RealisticWorkloadTest:
    """
    Realistic test that actually exercises the GPU with:
    - Continuous inference at 10Hz
    - Parallel training on experience buffer
    - Sleep consolidation every 100 experiences
    - Real GR00T action generation (not random)
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        print("\n" + "="*80)
        print("🔥 REALISTIC GPU WORKLOAD TEST")
        print("="*80)
        
        # Initialize all systems
        print("\n⚙️ Initializing systems...")
        
        # Main inference system
        self.hl_system = IntegratedHLSystem(
            context_dim=4096,
            compressed_dim=256,
            action_dim=19,
            compression_type="hybrid",
            device=device
        )
        
        # GR00T for realistic observations
        self.groot = GR00TWorldInterface(
            model_path="nvidia/GR00T-N1.5-3B",
            embodiment="GR1",
            device=device
        )
        
        # Training components
        self.optimizer = torch.optim.AdamW(self.hl_system.parameters(), lr=1e-4)
        self.experience_buffer = ExperienceMemory(capacity=10000)
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=1000)
        self.running = False
        
        print("✅ All systems initialized")
        
    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU utilization metrics."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
        except:
            pass
        
        # Fallback to torch metrics
        return {
            'utilization': 0.0,
            'memory_used': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            'temperature': 0.0
        }
    
    def inference_loop(self, duration_seconds: int = 60, target_hz: float = 10.0):
        """
        Run continuous inference loop at target frequency.
        This simulates real robot control loop.
        """
        print(f"\n🏃 Running inference loop at {target_hz}Hz for {duration_seconds}s...")
        
        interval = 1.0 / target_hz
        start_time = time.time()
        step = 0
        
        # Warm up the models
        print("⏳ Warming up models...")
        for _ in range(5):
            obs = self.groot.generate_observation(step=0)
            _ = self.hl_system(obs)
        
        print("🚀 Starting main inference loop...")
        inference_times = []
        
        while (time.time() - start_time) < duration_seconds:
            loop_start = time.time()
            
            # Generate observation
            obs = self.groot.generate_observation(step=step)
            
            # Full inference pipeline
            inf_start = time.time()
            with torch.no_grad():
                output = self.hl_system(obs, return_detailed=True)
                action = output['action']
                context_4k = output['context_4k']
                compressed = output['compressed_context']
            
            # Get actual GR00T action (forces model inference)
            groot_action = self.groot.get_action(obs)
            
            # Combine actions (weighted average for realism)
            final_action = 0.7 * action + 0.3 * groot_action
            
            inference_time = (time.time() - inf_start) * 1000
            inference_times.append(inference_time)
            
            # Simulate environment step
            next_obs, reward = self.groot.simulate_step(obs, final_action)
            
            # Store experience
            exp = Experience(
                observation=obs,
                context_4k=context_4k,
                action=final_action,
                next_observation=next_obs,
                reward=reward,
                metadata={'step': step, 'inference_ms': inference_time}
            )
            self.experience_buffer.add_episode([exp])
            
            # Track metrics
            if step % 10 == 0:
                gpu_metrics = self.get_gpu_metrics()
                metric = SystemMetrics(
                    timestamp=time.time() - start_time,
                    gpu_util=gpu_metrics['utilization'],
                    gpu_memory=gpu_metrics['memory_used'],
                    gpu_temp=gpu_metrics['temperature'],
                    inference_time=np.mean(inference_times[-10:]) if inference_times else 0,
                    training_loss=0.0,  # Will be updated by training thread
                    experiences_processed=step
                )
                self.metrics_history.append(metric)
                
                # Print progress
                if step % 100 == 0:
                    print(f"  Step {step}: {metric.inference_time:.1f}ms inference, "
                          f"GPU {metric.gpu_util:.0f}%, Mem {metric.gpu_memory:.1f}GB")
            
            step += 1
            
            # Maintain target frequency
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
        
        return inference_times
    
    def training_loop(self, num_batches: int = 100, batch_size: int = 32):
        """
        Run training on accumulated experiences.
        This simulates sleep consolidation.
        """
        print(f"\n🎓 Running training for {num_batches} batches...")
        
        losses = []
        
        for batch_idx in range(num_batches):
            # Sample experiences
            if len(self.experience_buffer.buffer) < batch_size:
                continue
                
            experiences = self.experience_buffer.sample(batch_size)
            
            # Prepare batch
            observations = {}
            for key in experiences[0].observation:
                if key == 'batch_size':
                    observations[key] = batch_size
                elif isinstance(experiences[0].observation[key], torch.Tensor):
                    observations[key] = torch.stack([e.observation[key].squeeze(0) for e in experiences])
                else:
                    observations[key] = experiences[0].observation[key]
            
            actions = torch.stack([e.action.squeeze(0) for e in experiences])
            
            # Forward pass
            output = self.hl_system(observations, return_detailed=True)
            predicted_actions = output['action']
            
            # Loss calculation
            action_loss = F.mse_loss(predicted_actions, actions)
            
            # Add compression loss
            if 'compression_metrics' in output:
                metrics = output['compression_metrics']
                compression_loss = 1.0 - metrics.information_retained
                total_loss = action_loss + 0.1 * compression_loss
            else:
                total_loss = action_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.hl_system.parameters(), 1.0)
            self.optimizer.step()
            
            losses.append(total_loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                print(f"  Batch {batch_idx + 1}: Loss = {avg_loss:.4f}")
        
        return losses
    
    def run_sustained_test(self, duration_minutes: float = 5.0):
        """
        Run sustained workload test with parallel inference and training.
        """
        print(f"\n" + "="*80)
        print(f"🚀 STARTING {duration_minutes} MINUTE SUSTAINED WORKLOAD TEST")
        print("="*80)
        
        # Initial GPU state
        initial_metrics = self.get_gpu_metrics()
        print(f"\n📊 Initial GPU State:")
        print(f"   Memory: {initial_metrics['memory_used']:.1f}/{initial_metrics['memory_total']:.1f} GB")
        print(f"   Utilization: {initial_metrics['utilization']:.0f}%")
        
        # Phase 1: Pure inference
        print(f"\n" + "-"*60)
        print("PHASE 1: INFERENCE ONLY (1 minute)")
        print("-"*60)
        inf_times = self.inference_loop(duration_seconds=60, target_hz=10)
        
        print(f"\n✅ Inference Results:")
        print(f"   Average: {np.mean(inf_times):.2f}ms")
        print(f"   Min/Max: {np.min(inf_times):.2f}/{np.max(inf_times):.2f}ms")
        print(f"   Experiences collected: {len(self.experience_buffer.buffer)}")
        
        # Phase 2: Training on collected data
        print(f"\n" + "-"*60)
        print("PHASE 2: TRAINING ON EXPERIENCES")
        print("-"*60)
        train_losses = self.training_loop(num_batches=50, batch_size=32)
        
        print(f"\n✅ Training Results:")
        print(f"   Final loss: {train_losses[-1]:.4f}")
        print(f"   Loss reduction: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}%")
        
        # Phase 3: Parallel inference + training
        print(f"\n" + "-"*60)
        print(f"PHASE 3: PARALLEL INFERENCE + TRAINING ({duration_minutes-2:.0f} minutes)")
        print("-"*60)
        
        # Start inference in thread
        self.running = True
        inference_thread = threading.Thread(
            target=self.inference_loop,
            args=(int((duration_minutes-2)*60), 10)
        )
        inference_thread.start()
        
        # Run training in main thread
        time.sleep(5)  # Let some experiences accumulate
        train_losses = self.training_loop(num_batches=100, batch_size=64)
        
        # Wait for inference to complete
        inference_thread.join()
        self.running = False
        
        # Final metrics
        print(f"\n" + "="*80)
        print("📈 FINAL METRICS")
        print("="*80)
        
        final_metrics = self.get_gpu_metrics()
        print(f"\n💾 GPU Memory:")
        print(f"   Initial: {initial_metrics['memory_used']:.1f} GB")
        print(f"   Final: {final_metrics['memory_used']:.1f} GB")
        print(f"   Peak: {max(m.gpu_memory for m in self.metrics_history):.1f} GB")
        
        print(f"\n⚡ GPU Utilization:")
        utils = [m.gpu_util for m in self.metrics_history if m.gpu_util > 0]
        if utils:
            print(f"   Average: {np.mean(utils):.0f}%")
            print(f"   Peak: {max(utils):.0f}%")
        
        print(f"\n⏱️ Inference Performance:")
        inf_times_all = [m.inference_time for m in self.metrics_history if m.inference_time > 0]
        if inf_times_all:
            print(f"   Average: {np.mean(inf_times_all):.2f}ms")
            print(f"   95th percentile: {np.percentile(inf_times_all, 95):.2f}ms")
        
        print(f"\n📊 Experiences Processed: {len(self.experience_buffer.buffer)}")
        print(f"   Rate: {len(self.experience_buffer.buffer)/(duration_minutes*60):.1f} exp/sec")
        
        # Save metrics
        self.save_metrics()
        
        print("\n" + "="*80)
        print("✅ SUSTAINED WORKLOAD TEST COMPLETE")
        print("="*80)
    
    def save_metrics(self):
        """Save metrics to file for analysis."""
        import json
        
        metrics_data = []
        for m in self.metrics_history:
            metrics_data.append({
                'timestamp': m.timestamp,
                'gpu_util': m.gpu_util,
                'gpu_memory': m.gpu_memory,
                'gpu_temp': m.gpu_temp,
                'inference_time': m.inference_time,
                'training_loss': m.training_loss,
                'experiences': m.experiences_processed
            })
        
        with open('workload_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"📁 Metrics saved to workload_metrics.json")


def main():
    """Run the realistic workload test."""
    
    # Check if we have a GPU
    if not torch.cuda.is_available():
        print("⚠️ No GPU available! This test requires CUDA.")
        return
    
    # Print system info
    print("\n" + "="*80)
    print("💻 SYSTEM INFORMATION")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    # Create and run test
    test = RealisticWorkloadTest(device="cuda")
    
    # Run quick test (5 minutes)
    test.run_sustained_test(duration_minutes=5.0)
    
    # Offer extended test
    print("\n" + "="*80)
    response = input("Run extended 1-hour test? (y/n): ")
    if response.lower() == 'y':
        test.run_sustained_test(duration_minutes=60.0)


if __name__ == "__main__":
    main()