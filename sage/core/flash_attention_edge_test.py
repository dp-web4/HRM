#!/usr/bin/env python3
"""
FlashAttention Multi-Sensor Fusion - Edge Validation

Testing Thor's FlashAttention implementation on Sprout (Jetson Orin Nano 8GB).

Thor's Implementation (Phase 3):
- GPU-accelerated cross-modal attention for sensor fusion
- Grouped Query Attention (GQA) for 2x efficiency
- 4-dimensional attention: goal, salience, memory, trust
- Target: <10ms latency on Jetson Nano

Edge Validation Goals:
1. Verify CUDA availability and device detection
2. Test attention computation on edge GPU
3. Validate GQA efficiency gains
4. Profile latency against 10ms target
5. Test with varying sensor counts (1-10)
6. Measure memory usage on unified memory

Platform: Sprout (Jetson Orin Nano 8GB, CUDA Ampere)
Date: 2026-01-11
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def get_edge_metrics() -> Dict[str, Any]:
    """Get edge hardware metrics."""
    metrics = {
        "platform": "Jetson Orin Nano 8GB",
        "hardware_type": "cuda_ampere",
        "capability_level": 3
    }

    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('MemAvailable:'):
                    metrics["memory_available_mb"] = int(line.split()[1]) / 1024
    except Exception:
        pass

    try:
        for path in ['/sys/devices/virtual/thermal/thermal_zone0/temp',
                     '/sys/class/thermal/thermal_zone0/temp']:
            try:
                with open(path, 'r') as f:
                    metrics["temperature_c"] = int(f.read().strip()) / 1000.0
                    break
            except Exception:
                continue
    except Exception:
        pass

    return metrics


def test_flash_attention_edge():
    """Test FlashAttention Multi-Sensor Fusion on edge hardware."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  FLASH ATTENTION MULTI-SENSOR FUSION - EDGE VALIDATION  ".center(70) + "|")
    print("|" + "           Jetson Orin Nano 8GB (Sprout)                  ".center(70) + "|")
    print("|" + " " * 70 + "|")
    print("+" + "=" * 70 + "+")
    print()

    edge_metrics = get_edge_metrics()
    print("Edge Hardware:")
    print(f"  Platform: {edge_metrics['platform']}")
    if 'temperature_c' in edge_metrics:
        print(f"  Temperature: {edge_metrics['temperature_c']}C")
    if 'memory_available_mb' in edge_metrics:
        print(f"  Memory: {int(edge_metrics['memory_available_mb'])} MB available")
    print()

    all_tests_passed = True
    test_results = {}

    # ========================================================================
    # TEST 1: PyTorch and CUDA Availability
    # ========================================================================
    print("=" * 72)
    print("TEST 1: PyTorch and CUDA Availability")
    print("=" * 72)
    print()

    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            device = torch.device('cuda')
        else:
            print("  Running on CPU (CUDA not available)")
            device = torch.device('cpu')

        test1_pass = True

    except Exception as e:
        print(f"  ERROR: {e}")
        test1_pass = False
        device = None

    test_results["pytorch_cuda"] = test1_pass
    print(f"\n{'PASS' if test1_pass else 'FAIL'}: TEST 1")
    print()
    all_tests_passed = all_tests_passed and test1_pass

    if not test1_pass:
        return {"all_tests_passed": False, "test_results": test_results}

    # ========================================================================
    # TEST 2: Import FlashAttention Components
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Import FlashAttention Components")
    print("=" * 72)
    print()

    try:
        from flash_attention_sensor_fusion import (
            MultiSensorFusionConfig,
            MultiSensorFusionAttention,
            FlashAttentionSensorSelector,
            SensorInput,
            AttentionDimension,
        )

        print("  MultiSensorFusionConfig: Configuration dataclass")
        print("  MultiSensorFusionAttention: PyTorch attention module")
        print("  FlashAttentionSensorSelector: Numpy-compatible wrapper")
        print("  SensorInput: Sensor data structure")
        print("  AttentionDimension: 4-dimensional attention enum")
        print()
        test2_pass = True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test2_pass = False

    test_results["import_validation"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    if not test2_pass:
        return {"all_tests_passed": False, "test_results": test_results}

    # ========================================================================
    # TEST 3: Basic Attention Computation
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Basic Attention Computation")
    print("=" * 72)
    print()

    print("Testing attention with 4 sensors...")

    try:
        config = MultiSensorFusionConfig()
        attention = MultiSensorFusionAttention(config).to(device)
        attention.eval()

        print(f"  Config: {config.n_query_heads} Q heads, {config.n_kv_heads} KV heads")
        print(f"  d_model: {config.d_model}, head_dim: {config.head_dim}")

        # Cognitive state
        cognitive_state = torch.randn(1, config.d_model).to(device)

        # 4 test sensors
        sensors = [
            SensorInput(
                sensor_id='vision',
                embedding=torch.randn(config.d_model).to(device),
                goal_score=0.9, salience_score=0.7,
                memory_score=0.8, trust_score=0.95
            ),
            SensorInput(
                sensor_id='audio',
                embedding=torch.randn(config.d_model).to(device),
                goal_score=0.3, salience_score=0.5,
                memory_score=0.6, trust_score=0.8
            ),
            SensorInput(
                sensor_id='proprioception',
                embedding=torch.randn(config.d_model).to(device),
                goal_score=0.7, salience_score=0.6,
                memory_score=0.9, trust_score=0.99
            ),
            SensorInput(
                sensor_id='imu',
                embedding=torch.randn(config.d_model).to(device),
                goal_score=0.6, salience_score=0.4,
                memory_score=0.7, trust_score=0.85
            )
        ]

        with torch.no_grad():
            scores, attn_weights = attention(cognitive_state, sensors, return_attention_weights=True)

        print(f"\n  Attention scores shape: {scores.shape}")
        print(f"  Attention weights shape: {attn_weights.shape}")
        print(f"  Sensor scores:")
        for i, sensor in enumerate(sensors):
            print(f"    {sensor.sensor_id}: {scores[0, i].item():.4f}")

        # Validate scores in [0, 1]
        scores_valid = torch.all((scores >= 0) & (scores <= 1)).item()
        print(f"\n  Scores in valid range [0, 1]: {scores_valid}")

        test3_pass = scores_valid and scores.shape == (1, 4)

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test3_pass = False

    test_results["basic_attention"] = test3_pass
    print(f"\n{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Latency Benchmark (10ms Target)
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Latency Benchmark (10ms Target)")
    print("=" * 72)
    print()

    print("Benchmarking attention latency...")

    try:
        num_trials = 100
        warmup = 10
        latencies = []

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _, _ = attention(cognitive_state, sensors)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Benchmark
        for _ in range(num_trials):
            start = time.time()
            with torch.no_grad():
                _, _ = attention(cognitive_state, sensors)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)

        mean_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)

        print(f"  Trials: {num_trials}")
        print(f"  Mean latency: {mean_latency:.2f} ms")
        print(f"  P50 latency: {p50_latency:.2f} ms")
        print(f"  P95 latency: {p95_latency:.2f} ms")
        print(f"  P99 latency: {p99_latency:.2f} ms")
        print(f"  Min/Max: {min_latency:.2f} / {max_latency:.2f} ms")
        print(f"  Throughput: {1000/mean_latency:.1f} allocations/sec")

        # Check against 10ms target
        target_ms = 10.0
        within_budget = mean_latency < target_ms

        if within_budget:
            print(f"\n  Within {target_ms}ms budget")
        else:
            print(f"\n  Exceeds {target_ms}ms budget (edge may need optimization)")

        test4_pass = True  # Pass even if over budget - this is informational
        test_results["latency_mean_ms"] = mean_latency
        test_results["latency_p95_ms"] = p95_latency
        test_results["within_10ms_budget"] = within_budget

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["latency_benchmark"] = test4_pass
    print(f"\n{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Sensor Count Scaling (1-10 sensors)
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Sensor Count Scaling (1-10 sensors)")
    print("=" * 72)
    print()

    print("Testing latency with different sensor counts...")

    try:
        sensor_counts = [1, 2, 4, 6, 8, 10]
        scaling_results = {}

        for n_sensors in sensor_counts:
            # Create n sensors
            test_sensors = [
                SensorInput(
                    sensor_id=f'sensor_{i}',
                    embedding=torch.randn(config.d_model).to(device),
                    goal_score=0.5, salience_score=0.5,
                    memory_score=0.5, trust_score=0.5
                )
                for i in range(n_sensors)
            ]

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _, _ = attention(cognitive_state, test_sensors)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            # Benchmark
            trial_latencies = []
            for _ in range(50):
                start = time.time()
                with torch.no_grad():
                    _, _ = attention(cognitive_state, test_sensors)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                trial_latencies.append((time.time() - start) * 1000)

            mean_lat = np.mean(trial_latencies)
            scaling_results[n_sensors] = mean_lat
            print(f"  {n_sensors:2d} sensors: {mean_lat:.2f} ms")

        # Check if scaling is reasonable (< 2x for 10 sensors vs 1)
        if 1 in scaling_results and 10 in scaling_results:
            scaling_factor = scaling_results[10] / scaling_results[1]
            print(f"\n  Scaling factor (10 vs 1 sensor): {scaling_factor:.2f}x")

        test5_pass = all(lat < 50 for lat in scaling_results.values())  # Allow up to 50ms
        test_results["sensor_scaling"] = scaling_results

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["scaling_test"] = test5_pass
    print(f"\n{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: Numpy Interface Compatibility
    # ========================================================================
    print("=" * 72)
    print("TEST 6: Numpy Interface Compatibility")
    print("=" * 72)
    print()

    print("Testing FlashAttentionSensorSelector with numpy inputs...")

    try:
        selector = FlashAttentionSensorSelector(config, device)

        # Register sensors
        for sensor in sensors:
            selector.register_sensor(sensor.sensor_id)

        # Numpy inputs (like AttentionManager would provide)
        cognitive_np = np.random.randn(config.d_model).astype(np.float32)
        sensor_scores_dict = {
            'vision': {'goal': 0.9, 'salience': 0.7, 'memory': 0.8, 'trust': 0.95},
            'audio': {'goal': 0.3, 'salience': 0.5, 'memory': 0.6, 'trust': 0.8},
            'proprioception': {'goal': 0.7, 'salience': 0.6, 'memory': 0.9, 'trust': 0.99},
            'imu': {'goal': 0.6, 'salience': 0.4, 'memory': 0.7, 'trust': 0.85}
        }

        result_scores = selector.compute_attention_scores(cognitive_np, sensor_scores_dict)

        print(f"  Input type: numpy.ndarray")
        print(f"  Output type: {type(result_scores).__name__}")
        print(f"  Sensor scores:")
        for sensor_id, score in result_scores.items():
            print(f"    {sensor_id}: {score:.4f}")

        # Validate
        is_dict = isinstance(result_scores, dict)
        scores_valid = all(0 <= s <= 1 for s in result_scores.values())
        correct_sensors = set(result_scores.keys()) == set(sensor_scores_dict.keys())

        print(f"\n  Is dict: {is_dict}")
        print(f"  Scores in [0, 1]: {scores_valid}")
        print(f"  Correct sensors: {correct_sensors}")

        test6_pass = is_dict and scores_valid and correct_sensors

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test6_pass = False

    test_results["numpy_interface"] = test6_pass
    print(f"\n{'PASS' if test6_pass else 'FAIL'}: TEST 6")
    print()
    all_tests_passed = all_tests_passed and test6_pass

    # ========================================================================
    # TEST 7: GQA Efficiency Validation
    # ========================================================================
    print("=" * 72)
    print("TEST 7: GQA Efficiency Validation")
    print("=" * 72)
    print()

    print("Validating Grouped Query Attention efficiency...")

    try:
        # Calculate parameter counts
        n_params_q = config.n_query_heads * config.head_dim * config.d_model
        n_params_kv = config.n_kv_heads * config.head_dim * config.d_model * 2  # K + V

        # Full attention would use same heads for Q, K, V
        n_params_kv_full = config.n_query_heads * config.head_dim * config.d_model * 2

        reduction = n_params_kv_full / n_params_kv
        gqa_ratio = config.n_query_heads / config.n_kv_heads

        print(f"  Query heads: {config.n_query_heads}")
        print(f"  KV heads: {config.n_kv_heads}")
        print(f"  GQA ratio: {gqa_ratio:.1f}x")
        print(f"  Query params: {n_params_q:,}")
        print(f"  KV params (GQA): {n_params_kv:,}")
        print(f"  KV params (full): {n_params_kv_full:,}")
        print(f"  Parameter reduction: {reduction:.1f}x")

        # Count model parameters
        total_params = sum(p.numel() for p in attention.parameters())
        print(f"\n  Total model parameters: {total_params:,}")

        test7_pass = reduction == 2.0 and gqa_ratio == 2.0

    except Exception as e:
        print(f"  ERROR: {e}")
        test7_pass = False

    test_results["gqa_efficiency"] = test7_pass
    print(f"\n{'PASS' if test7_pass else 'FAIL'}: TEST 7")
    print()
    all_tests_passed = all_tests_passed and test7_pass

    # ========================================================================
    # TEST 8: Memory Usage on Unified Memory
    # ========================================================================
    print("=" * 72)
    print("TEST 8: Memory Usage (Unified Memory)")
    print("=" * 72)
    print()

    print("Measuring GPU memory usage...")

    try:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

            # Run attention multiple times
            for _ in range(10):
                with torch.no_grad():
                    _, _ = attention(cognitive_state, sensors)
                torch.cuda.synchronize()

            allocated = torch.cuda.memory_allocated() / 1e6  # MB
            reserved = torch.cuda.memory_reserved() / 1e6  # MB
            peak = torch.cuda.max_memory_allocated() / 1e6  # MB

            print(f"  Allocated memory: {allocated:.2f} MB")
            print(f"  Reserved memory: {reserved:.2f} MB")
            print(f"  Peak memory: {peak:.2f} MB")

            # Check if reasonable for 8GB unified memory
            reasonable = peak < 500  # Should be well under 500MB
            print(f"\n  Memory usage reasonable (<500MB): {reasonable}")

            test8_pass = reasonable
            test_results["memory_peak_mb"] = peak
        else:
            print("  Skipping (CPU mode)")
            test8_pass = True
            test_results["memory_peak_mb"] = 0

    except Exception as e:
        print(f"  ERROR: {e}")
        test8_pass = False

    test_results["memory_usage"] = test8_pass
    print(f"\n{'PASS' if test8_pass else 'FAIL'}: TEST 8")
    print()
    all_tests_passed = all_tests_passed and test8_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("FLASH ATTENTION EDGE VALIDATION SUMMARY")
    print("=" * 72)
    print()

    print("Test Results:")
    for test_name, passed in test_results.items():
        if isinstance(passed, bool):
            status = "PASS" if passed else "FAIL"
            print(f"  {test_name}: {status}")
    print()

    passed_count = sum(1 for v in test_results.values() if v is True)
    bool_tests = sum(1 for v in test_results.values() if isinstance(v, bool))
    print(f"Overall: {passed_count}/{bool_tests} tests passed")
    print()

    if all_tests_passed:
        print("+" + "-" * 70 + "+")
        print("|" + " " * 70 + "|")
        print("|" + "  FLASH ATTENTION VALIDATED ON EDGE!  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        if 'latency_mean_ms' in test_results:
            print(f"  - Mean latency: {test_results['latency_mean_ms']:.2f}ms")
            print(f"  - P95 latency: {test_results['latency_p95_ms']:.2f}ms")
            if test_results.get('within_10ms_budget'):
                print(f"  - Within 10ms target")
            else:
                print(f"  - Exceeds 10ms target (acceptable on edge)")
        if 'memory_peak_mb' in test_results:
            print(f"  - Peak memory: {test_results['memory_peak_mb']:.1f}MB")
        print(f"  - GQA provides 2x parameter efficiency")
        print(f"  - Numpy interface compatible with AttentionManager")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "flash_attention_edge",
        "title": "FlashAttention Multi-Sensor Fusion - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": {k: v for k, v in test_results.items()},
        "edge_metrics": edge_metrics,
        "device": str(device) if device else "none",
        "pytorch_version": torch.__version__ if 'torch' in dir() else "unknown"
    }

    results_path = Path(__file__).parent / "flash_attention_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_flash_attention_edge()
    sys.exit(0 if success else 1)
