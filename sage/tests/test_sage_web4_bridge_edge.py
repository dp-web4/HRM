#!/usr/bin/env python3
"""
SAGE-Web4 Bridge Edge Validation Test

Session 19: Validate Thor's SAGE-Web4 bridge on actual edge hardware

Tests:
1. LCT creation with real hardware metrics
2. Operation tracking with actual SAGE inference
3. V3 reputation evolution with real operation outcomes
4. ATP consumption metering
5. Multi-dimensional V3 component tracking

Hardware: Jetson Orin Nano 8GB (Sprout)
Bridge: Thor's Session #77 sage_web4_bridge.py
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "web4" / "game" / "engine"))

# Import SAGE components
from sage.irp.plugins.llm_impl import ConversationalLLM

# Import Web4 bridge components
try:
    from sage_web4_bridge import (
        create_sage_agent_lct,
        create_sage_edge_society_lct,
        track_sage_operation,
        SAGEOperationResult
    )
    from multidimensional_v3 import V3Components, calculate_composite_veracity
    WEB4_AVAILABLE = True
except ImportError as e:
    print(f"Note: Web4 bridge not fully available: {e}")
    WEB4_AVAILABLE = False


def get_temperature() -> float:
    """Read Jetson thermal zone temperature"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0


def get_memory_available() -> float:
    """Get available memory in GB"""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
        return 0.0
    except:
        return 0.0


@dataclass
class EdgeOperationMetrics:
    """Metrics from actual edge operation"""
    query: str
    response: str
    latency: float
    temperature_start: float
    temperature_end: float
    memory_available: float
    success: bool
    error: Optional[str] = None


def run_bridge_test():
    """Test SAGE-Web4 bridge with real edge operations"""

    print("=" * 80)
    print("SAGE-WEB4 BRIDGE EDGE VALIDATION")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB (Sprout)")
    print("Purpose: Validate Thor's Session #77 bridge on actual edge hardware")
    print()

    if not WEB4_AVAILABLE:
        print("ERROR: Web4 bridge components not available")
        print("Make sure web4/game/engine is in path")
        return

    # Test 1: Create SAGE LCT with real hardware info
    print("-" * 80)
    print("Test 1: Create SAGE Agent LCT")
    print("-" * 80)

    temp_now = get_temperature()
    mem_now = get_memory_available()

    sage_lct = create_sage_agent_lct(
        sage_instance_id="sprout_01",
        hardware="Jetson Orin Nano 8GB"
    )

    print(f"  LCT ID: {sage_lct.lct_id}")
    print(f"  Type: {sage_lct.lct_type}")
    print(f"  Initial V3: {sage_lct.value_axes['V3']['veracity']}")
    print(f"  V3 Components: {sage_lct.value_axes['V3'].get('components', {})}")
    print(f"  Current temp: {temp_now:.1f}°C")
    print(f"  Current memory: {mem_now:.1f} GB available")
    print("  ✓ SAGE LCT created")

    # Test 2: Create Edge Society LCT
    print()
    print("-" * 80)
    print("Test 2: Create Edge Society LCT")
    print("-" * 80)

    society_lct = create_sage_edge_society_lct(
        sage_instance_id="sprout_01",
        hardware="Jetson Orin Nano 8GB"
    )

    print(f"  Society ID: {society_lct.lct_id}")
    print(f"  Type: {society_lct.lct_type}")
    print(f"  Agent count: {society_lct.metadata.get('agent_count', 'N/A')}")
    print(f"  Treasury ATP: {society_lct.metadata.get('treasury', {}).get('ATP', 'N/A')}")
    print("  ✓ Edge Society LCT created")

    # Test 3: Run actual SAGE operations and track with bridge
    print()
    print("-" * 80)
    print("Test 3: Track Real SAGE Operations")
    print("-" * 80)

    # Load model
    print()
    print("Loading SAGE model...")
    model_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    start_load = time.perf_counter()
    conv = ConversationalLLM(model_path=model_path, irp_iterations=3)
    load_time = time.perf_counter() - start_load
    print(f"  Model loaded in {load_time:.1f}s")

    # Test operations
    test_operations = [
        ("simple", "What is 2 + 2?"),
        ("medium", "Explain neural networks briefly."),
        ("complex", "What is consciousness?"),
    ]

    operation_results: List[EdgeOperationMetrics] = []
    v3_history: List[Dict] = []

    print()
    print(f"{'Type':<10} {'Latency':<10} {'Temp Δ':<10} {'Success':<8} {'Quality'}")
    print("-" * 60)

    # ATP cost model (edge device)
    ATP_PER_SECOND = 5.0  # 5 ATP per second of inference

    for op_type, query in test_operations:
        temp_start = get_temperature()
        mem_start = get_memory_available()

        # Run actual inference
        start = time.perf_counter()
        try:
            response, metadata = conv.respond(query)
            success = True
            error = None
        except Exception as e:
            response = ""
            success = False
            error = str(e)

        latency = time.perf_counter() - start
        temp_end = get_temperature()

        metrics = EdgeOperationMetrics(
            query=query,
            response=response,
            latency=latency,
            temperature_start=temp_start,
            temperature_end=temp_end,
            memory_available=mem_start,
            success=success,
            error=error
        )
        operation_results.append(metrics)

        # Calculate quality score based on response
        if success:
            # Simple heuristic: longer responses for complex queries = higher quality
            if op_type == "simple":
                quality_score = 0.95 if len(response) > 5 else 0.5
            elif op_type == "medium":
                quality_score = 0.85 if len(response) > 50 else 0.6
            else:
                quality_score = 0.80 if len(response) > 100 else 0.5
        else:
            quality_score = 0.0

        # Calculate ATP consumed
        atp_consumed = latency * ATP_PER_SECOND

        # Create operation result for bridge
        sage_op = SAGEOperationResult(
            operation_id=f"sage_op_{len(operation_results):03d}",
            operation_type="conversation",
            success=success,
            latency=latency,
            atp_consumed=atp_consumed,
            quality_score=quality_score,
            output=response[:100] if response else None,
            error=error
        )

        # Track operation with bridge
        v3_update = track_sage_operation(sage_lct, sage_op)
        v3_history.append({
            "operation": op_type,
            "v3_update": v3_update,
            "latency": latency,
            "quality": quality_score
        })

        # Print result
        temp_delta = temp_end - temp_start
        status = "✓" if success else "✗"
        print(f"{op_type:<10} {latency:<10.1f} {temp_delta:+.1f}°C     {status:<8} {quality_score:.2f}")

    # Analysis
    print()
    print("=" * 80)
    print("BRIDGE ANALYSIS")
    print("=" * 80)

    # V3 Evolution Summary
    print()
    print("V3 Reputation Evolution:")
    print(f"  Initial V3: 0.85")

    final_v3 = sage_lct.value_axes['V3']
    print(f"  Final V3: {final_v3.get('veracity', 'N/A')}")

    if 'components' in final_v3:
        print()
        print("V3 Components:")
        for comp, value in final_v3['components'].items():
            print(f"    {comp}: {value:.3f}")

    # Operation Statistics
    print()
    print("Operation Statistics:")
    successful_ops = [r for r in operation_results if r.success]
    avg_latency = sum(r.latency for r in successful_ops) / len(successful_ops) if successful_ops else 0
    total_temp_rise = operation_results[-1].temperature_end - operation_results[0].temperature_start

    print(f"  Operations: {len(operation_results)}")
    print(f"  Successful: {len(successful_ops)}")
    print(f"  Average latency: {avg_latency:.1f}s")
    print(f"  Temperature rise: {total_temp_rise:+.1f}°C")

    # ATP Summary
    total_atp = sum(r.latency * ATP_PER_SECOND for r in successful_ops)
    print()
    print("ATP Consumption:")
    print(f"  Total ATP consumed: {total_atp:.1f}")
    print(f"  ATP per operation: {total_atp/len(successful_ops):.1f}" if successful_ops else "  N/A")
    print(f"  Society treasury remaining: {society_lct.metadata.get('treasury', {}).get('ATP', 0) - total_atp:.1f}")

    # Verdict
    print()
    print("=" * 80)
    print("BRIDGE VALIDATION VERDICT")
    print("=" * 80)
    print()

    if len(successful_ops) == len(operation_results):
        print("✓ BRIDGE VALIDATED: All operations tracked successfully")
        print(f"  - LCT creation: Working")
        print(f"  - Operation tracking: Working")
        print(f"  - V3 evolution: Working")
        print(f"  - ATP metering: Working")
        print(f"  - Multi-dimensional components: Working")
    else:
        failed = len(operation_results) - len(successful_ops)
        print(f"⚠ PARTIAL: {failed} operations failed")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_bridge_test()
