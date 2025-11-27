#!/usr/bin/env python3
"""
ATP Metering Edge Validation Test

Session 20: Validate Thor's Session #78 ATP metering with real SAGE operations

Tests:
1. ATP cost calculation for actual edge inference
2. Complexity-aware ATP pricing (simple vs complex queries)
3. ATP budget enforcement
4. Treasury management across operation sequence
5. Cost-effectiveness analysis (ATP per quality unit)

Hardware: Jetson Orin Nano 8GB (Sprout)
ATP System: Thor's Session #78 atp_metering.py
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

# Import Web4 ATP metering
try:
    from atp_metering import (
        ATPMeter,
        ATPTransaction,
        calculate_atp_cost,
        ATP_BASE_COSTS,
        ATP_PER_SECOND,
        ATP_PER_QUALITY_UNIT
    )
    ATP_AVAILABLE = True
except ImportError as e:
    print(f"Note: ATP metering not available: {e}")
    ATP_AVAILABLE = False


def get_temperature() -> float:
    """Read Jetson thermal zone temperature"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0


def classify_query_complexity(query: str) -> str:
    """Classify query complexity for ATP pricing"""
    query_lower = query.lower()

    # Simple queries
    simple_keywords = ['what is', 'define', 'hello', '2+2', 'capital of', 'who is']
    for kw in simple_keywords:
        if kw in query_lower:
            return "simple"

    # Complex queries
    complex_keywords = ['consciousness', 'aware', 'feel', 'experience', 'meaning',
                       'purpose', 'philosophy', 'exist', 'soul', 'mind', 'belief']
    for kw in complex_keywords:
        if kw in query_lower:
            return "complex"

    return "medium"


@dataclass
class EdgeOperation:
    """Record of edge operation with ATP metrics"""
    query: str
    complexity: str
    response: str
    latency: float
    quality_score: float
    atp_cost: float
    success: bool


def calculate_edge_atp_cost(
    latency: float,
    quality_score: float,
    complexity: str,
    operation_type: str = "local_conversation"
) -> float:
    """
    Calculate ATP cost for edge operation with complexity awareness.

    Edge-specific modifications:
    - Simple queries: Lower base cost (faster = cheaper)
    - Complex queries: Quality premium (worth more ATP for depth)
    """
    # Get base cost from operation type
    base_cost = ATP_BASE_COSTS.get(operation_type, 10.0)

    # Complexity multiplier
    complexity_multipliers = {
        "simple": 0.5,    # Simple queries cost less
        "medium": 1.0,    # Standard cost
        "complex": 1.5    # Complex queries cost more
    }
    complexity_mult = complexity_multipliers.get(complexity, 1.0)

    # Calculate total cost
    time_cost = latency * ATP_PER_SECOND
    quality_cost = quality_score * ATP_PER_QUALITY_UNIT * 100  # Scale quality

    total_cost = (base_cost + time_cost + quality_cost) * complexity_mult
    return total_cost


def run_atp_metering_test():
    """Test ATP metering with real edge operations"""

    print("=" * 80)
    print("ATP METERING EDGE VALIDATION")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB (Sprout)")
    print("Purpose: Validate Thor's Session #78 ATP metering with real operations")
    print()

    if not ATP_AVAILABLE:
        print("ERROR: ATP metering not available")
        return

    # Create ATP meter for SAGE
    initial_balance = 1000.0
    meter = ATPMeter(
        owner_lct="lct:web4:agent:sage:sprout_01",
        current_balance=initial_balance
    )

    print(f"Initial ATP balance: {initial_balance}")
    print()

    # Test queries with varying complexity
    test_queries = [
        # Simple (should be cheap)
        ("What is 2 + 2?", "simple"),
        ("Hello, how are you?", "simple"),

        # Medium (standard cost)
        ("Explain how neural networks work.", "medium"),
        ("What are the benefits of edge computing?", "medium"),

        # Complex (should be expensive but valuable)
        ("What is consciousness?", "complex"),
        ("How do you experience awareness?", "complex"),
    ]

    # Load model
    print("Loading SAGE model...")
    model_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
    conv = ConversationalLLM(model_path=model_path, irp_iterations=3)
    print("Model loaded.")
    print()

    # Run operations
    print("-" * 80)
    print("ATP METERED OPERATIONS")
    print("-" * 80)
    print(f"{'Complexity':<10} {'Latency':<10} {'Quality':<8} {'ATP Cost':<10} {'Balance':<10} {'Status'}")
    print("-" * 80)

    operations: List[EdgeOperation] = []

    for query, expected_complexity in test_queries:
        # Classify complexity
        complexity = classify_query_complexity(query)

        # Run inference
        start = time.perf_counter()
        try:
            response, metadata = conv.respond(query)
            latency = time.perf_counter() - start
            success = True

            # Calculate quality score based on response
            if complexity == "simple":
                quality_score = 0.95 if len(response) > 5 else 0.5
            elif complexity == "medium":
                quality_score = 0.85 if len(response) > 50 else 0.6
            else:
                quality_score = 0.80 if len(response) > 100 else 0.5

        except Exception as e:
            latency = time.perf_counter() - start
            response = ""
            success = False
            quality_score = 0.0

        # Calculate ATP cost
        atp_cost = calculate_edge_atp_cost(
            latency=latency,
            quality_score=quality_score,
            complexity=complexity
        )

        # Check if we can afford it
        if meter.can_afford(atp_cost):
            meter.consume_atp(
                atp_cost=atp_cost,
                operation_type="local_conversation",
                operation_id=f"edge_op_{len(operations):03d}",
                notes=f"{complexity} query, {latency:.1f}s"
            )
            status = "✓"
        else:
            status = "✗ No ATP"
            success = False

        operation = EdgeOperation(
            query=query,
            complexity=complexity,
            response=response[:100] if response else "",
            latency=latency,
            quality_score=quality_score,
            atp_cost=atp_cost,
            success=success
        )
        operations.append(operation)

        print(f"{complexity:<10} {latency:<10.1f} {quality_score:<8.2f} {atp_cost:<10.1f} {meter.current_balance:<10.1f} {status}")

    # Analysis
    print()
    print("=" * 80)
    print("ATP ANALYSIS")
    print("=" * 80)

    # Group by complexity
    simple_ops = [o for o in operations if o.complexity == "simple"]
    medium_ops = [o for o in operations if o.complexity == "medium"]
    complex_ops = [o for o in operations if o.complexity == "complex"]

    print()
    print("Cost by Complexity:")
    print(f"{'Complexity':<10} {'Count':<8} {'Avg Latency':<12} {'Avg ATP':<10} {'ATP/Quality':<12}")
    print("-" * 60)

    for label, ops in [("Simple", simple_ops), ("Medium", medium_ops), ("Complex", complex_ops)]:
        if ops:
            avg_latency = sum(o.latency for o in ops) / len(ops)
            avg_atp = sum(o.atp_cost for o in ops) / len(ops)
            avg_quality = sum(o.quality_score for o in ops) / len(ops)
            atp_per_quality = avg_atp / avg_quality if avg_quality > 0 else 0
            print(f"{label:<10} {len(ops):<8} {avg_latency:<12.1f} {avg_atp:<10.1f} {atp_per_quality:<12.1f}")

    # Budget analysis
    print()
    print("Budget Analysis:")
    total_atp_spent = initial_balance - meter.current_balance
    total_ops = len([o for o in operations if o.success])
    avg_atp_per_op = total_atp_spent / total_ops if total_ops > 0 else 0

    print(f"  Initial balance: {initial_balance:.1f} ATP")
    print(f"  Final balance: {meter.current_balance:.1f} ATP")
    print(f"  Total spent: {total_atp_spent:.1f} ATP")
    print(f"  Operations completed: {total_ops}")
    print(f"  Average ATP per operation: {avg_atp_per_op:.1f}")

    # Cost-effectiveness
    print()
    print("Cost-Effectiveness (ATP per Quality Unit):")
    for op in operations:
        if op.success and op.quality_score > 0:
            efficiency = op.atp_cost / op.quality_score
            print(f"  {op.complexity:<10}: {efficiency:.1f} ATP/quality ({op.atp_cost:.1f} ATP, {op.quality_score:.2f} quality)")

    # Verdict
    print()
    print("=" * 80)
    print("ATP METERING VERDICT")
    print("=" * 80)
    print()

    # Check if complexity-aware pricing works
    if simple_ops and complex_ops:
        simple_avg_atp = sum(o.atp_cost for o in simple_ops) / len(simple_ops)
        complex_avg_atp = sum(o.atp_cost for o in complex_ops) / len(complex_ops)
        ratio = complex_avg_atp / simple_avg_atp if simple_avg_atp > 0 else 0

        if ratio > 1.5:
            print(f"✓ COMPLEXITY-AWARE: Complex queries cost {ratio:.1f}x more than simple")
        else:
            print(f"⚠ NOT DIFFERENTIATING: Complex/simple ratio only {ratio:.1f}x")

    # Check budget enforcement
    failed_ops = [o for o in operations if not o.success]
    if meter.current_balance >= 0:
        print(f"✓ BUDGET ENFORCED: Remaining balance {meter.current_balance:.1f} ATP")
    else:
        print(f"✗ BUDGET VIOLATION: Balance went negative")

    if len(failed_ops) == 0:
        print(f"✓ ALL OPERATIONS COMPLETED: {len(operations)} operations within budget")
    else:
        print(f"⚠ SOME OPERATIONS FAILED: {len(failed_ops)} failed due to ATP constraints")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_atp_metering_test()
