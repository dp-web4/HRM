#!/usr/bin/env python3
"""
Jetson Optimizer for SAGE
Optimizes SAGE models for Jetson Orin Nano constraints
Target: 10+ FPS, <4GB memory, <15W power
"""

import torch
import tensorrt as trt
import numpy as np
from typing import Any, Dict, Optional
import time
import subprocess
from pathlib import Path

class JetsonOptimizer:
    """Optimize SAGE for Jetson Orin Nano constraints"""

    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()

        # Set memory constraints
        self.config.max_workspace_size = 1 << 30  # 1GB workspace

        # Enable INT8 optimization
        if self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)

        # Enable FP16
        if self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)

    def optimize_model(self, sage_model: torch.nn.Module) -> trt.ICudaEngine:
        """Convert PyTorch model to optimized TensorRT engine"""
        # Export to ONNX first
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        torch.onnx.export(
            sage_model,
            dummy_input,
            "sage_temp.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        # Parse ONNX
        network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.trt_logger)

        with open("sage_temp.onnx", 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Optimize for batch size 1 (edge inference)
        profile = self.builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        self.config.add_optimization_profile(profile)

        # Build engine
        engine = self.builder.build_engine(network, self.config)

        # Clean up
        Path("sage_temp.onnx").unlink()

        return engine

    def profile_performance(self, engine: trt.ICudaEngine) -> Dict[str, float]:
        """Profile TensorRT engine performance"""
        context = engine.create_execution_context()

        # Allocate buffers
        inputs, outputs, bindings = [], [], []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = np.empty(size, dtype=dtype)
            cuda_mem = torch.cuda.FloatTensor(size)
            bindings.append(int(cuda_mem.data_ptr()))
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': cuda_mem})
            else:
                outputs.append({'host': host_mem, 'device': cuda_mem})

        # Warmup
        for _ in range(10):
            context.execute_v2(bindings=bindings)

        # Measure FPS
        num_iterations = 100
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iterations):
            context.execute_v2(bindings=bindings)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        fps = num_iterations / elapsed

        # Get memory usage
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        memory_mb = float(result.stdout.strip()) if result.returncode == 0 else 0

        # Get power consumption
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        power_w = float(result.stdout.strip()) if result.returncode == 0 else 0

        # Get temperature
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp_c = int(f.read().strip()) / 1000.0

        return {
            'fps': fps,
            'memory_mb': memory_mb,
            'power_w': power_w,
            'temp_c': temp_c,
            'latency_ms': (1000.0 / fps)
        }

    def apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply INT8 quantization to model"""
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        return model

    def optimize_memory_layout(self, model: torch.nn.Module):
        """Optimize memory layout for edge inference"""
        # Use channels_last memory format for better performance
        model = model.to(memory_format=torch.channels_last)

        # Enable cudnn autotuner
        torch.backends.cudnn.benchmark = True

        # Reduce memory fragmentation
        torch.cuda.empty_cache()

        return model


def main():
    """Test the Jetson optimizer"""
    print("🚀 Jetson Optimizer for SAGE")
    print("Target: 10+ FPS, <4GB memory, <15W power")

    optimizer = JetsonOptimizer()

    # Create a dummy model for testing
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 10)
    ).cuda()

    # Optimize
    print("\nOptimizing model...")
    engine = optimizer.optimize_model(model)

    if engine:
        print("✅ Model optimized successfully")

        # Profile performance
        print("\nProfiling performance...")
        metrics = optimizer.profile_performance(engine)

        print(f"\n📊 Performance Metrics:")
        print(f"   FPS: {metrics['fps']:.1f}")
        print(f"   Latency: {metrics['latency_ms']:.1f}ms")
        print(f"   Memory: {metrics['memory_mb']:.0f}MB")
        print(f"   Power: {metrics['power_w']:.1f}W")
        print(f"   Temperature: {metrics['temp_c']:.1f}°C")

        # Check if targets met
        if metrics['fps'] >= 10 and metrics['memory_mb'] < 4096 and metrics['power_w'] < 15:
            print("\n✅ All optimization targets achieved!")
        else:
            print("\n⚠️ Some targets not met, further optimization needed")

if __name__ == "__main__":
    main()
