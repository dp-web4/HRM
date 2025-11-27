#!/usr/bin/env python3
"""
Edge Empirical Data Collection for ATP Calibration

Session 21: Collect real edge inference data from Jetson Orin Nano
to complement Thor's simulated data for ATP pricing calibration.

Purpose:
- Provide ground truth latency and quality data from actual edge hardware
- Enable ATP pricing calibration based on real-world performance
- Document edge-specific characteristics (thermal, memory pressure)

Output: JSON file compatible with Thor's empirical data format
"""

import sys
import time
import json
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.irp.plugins.llm_impl import ConversationalLLM


@dataclass
class EdgeExecution:
    """Single edge operation execution"""
    task_type: str
    query: str
    complexity: str  # simple, medium, complex
    stakes_level: str  # low, medium, high
    success: bool
    latency_ms: float
    quality_score: float
    response_length: int
    temperature_start: float
    temperature_end: float
    irp_iterations: int


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


def classify_complexity(query: str) -> str:
    """Classify query complexity"""
    query_lower = query.lower()

    simple_patterns = ['what is', '2+2', 'define', 'hello', 'capital of', 'who is', 'how many']
    for p in simple_patterns:
        if p in query_lower:
            return "simple"

    complex_patterns = ['consciousness', 'aware', 'feel', 'experience', 'meaning',
                       'philosophy', 'exist', 'mind', 'belief', 'moral', 'ethics']
    for p in complex_patterns:
        if p in query_lower:
            return "complex"

    return "medium"


def get_stakes_level(complexity: str) -> str:
    """Map complexity to stakes level"""
    return {"simple": "low", "medium": "medium", "complex": "high"}[complexity]


def estimate_quality(response: str, complexity: str) -> float:
    """Estimate quality score based on response characteristics"""
    if not response or len(response) < 5:
        return 0.3

    # Length-based baseline
    if complexity == "simple":
        if len(response) > 10:
            base_quality = 0.90
        else:
            base_quality = 0.60
    elif complexity == "medium":
        if len(response) > 100:
            base_quality = 0.85
        elif len(response) > 50:
            base_quality = 0.75
        else:
            base_quality = 0.60
    else:  # complex
        if len(response) > 200:
            base_quality = 0.80
        elif len(response) > 100:
            base_quality = 0.70
        else:
            base_quality = 0.55

    # Coherence bonus (presence of sentences)
    if '.' in response and len(response.split('.')) > 1:
        base_quality += 0.05

    return min(base_quality, 0.98)


def collect_edge_data():
    """Collect empirical data from real edge inference"""

    print("=" * 80)
    print("EDGE EMPIRICAL DATA COLLECTION")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB (Sprout)")
    print("Purpose: Collect real edge data for ATP pricing calibration")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Test queries organized by complexity/task type
    test_queries = [
        # Simple/Low stakes - factual, quick answers
        ("factual_qa", "What is 2 + 2?", "simple"),
        ("factual_qa", "What is the capital of France?", "simple"),
        ("factual_qa", "How many continents are there?", "simple"),
        ("greeting", "Hello, how are you today?", "simple"),
        ("definition", "Define photosynthesis briefly.", "simple"),

        # Medium stakes - explanations, descriptions
        ("explanation", "Explain how neural networks learn.", "medium"),
        ("explanation", "Describe the water cycle.", "medium"),
        ("comparison", "Compare Python and JavaScript briefly.", "medium"),
        ("reasoning", "What are benefits of edge computing?", "medium"),
        ("analysis", "Why is climate change important?", "medium"),

        # High stakes - philosophical, complex reasoning
        ("philosophical", "What is consciousness?", "complex"),
        ("meta_cognitive", "How do you experience awareness?", "complex"),
        ("epistemological", "What is the nature of knowledge?", "complex"),
        ("ethical", "How should AI systems handle moral dilemmas?", "complex"),
        ("existential", "What gives existence meaning?", "complex"),
    ]

    # Load model
    print("Loading SAGE model...")
    model_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    start_load = time.perf_counter()
    conv = ConversationalLLM(model_path=model_path, irp_iterations=3)
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    initial_temp = get_temperature()
    initial_mem = get_memory_available()
    print(f"Initial temperature: {initial_temp:.1f}°C")
    print(f"Available memory: {initial_mem:.1f} GB")
    print()

    # Collect data
    executions: List[EdgeExecution] = []

    print("-" * 80)
    print(f"{'Task':<16} {'Complexity':<10} {'Latency':<12} {'Quality':<8} {'Temp Δ':<8} {'Status'}")
    print("-" * 80)

    for task_type, query, complexity in test_queries:
        stakes = get_stakes_level(complexity)
        temp_start = get_temperature()

        # Run inference
        start = time.perf_counter()
        try:
            response, metadata = conv.respond(query)
            latency_ms = (time.perf_counter() - start) * 1000
            success = True
        except Exception as e:
            response = ""
            latency_ms = (time.perf_counter() - start) * 1000
            success = False

        temp_end = get_temperature()
        quality = estimate_quality(response, complexity) if success else 0.0

        execution = EdgeExecution(
            task_type=task_type,
            query=query,
            complexity=complexity,
            stakes_level=stakes,
            success=success,
            latency_ms=latency_ms,
            quality_score=quality,
            response_length=len(response),
            temperature_start=temp_start,
            temperature_end=temp_end,
            irp_iterations=3
        )
        executions.append(execution)

        temp_delta = temp_end - temp_start
        status = "✓" if success else "✗"
        print(f"{task_type:<16} {complexity:<10} {latency_ms:<12.0f} {quality:<8.2f} {temp_delta:+.1f}°C   {status}")

    # Clean up
    del conv
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    # Aggregate statistics
    print()
    print("=" * 80)
    print("AGGREGATED STATISTICS")
    print("=" * 80)

    # Overall stats
    successful = [e for e in executions if e.success]
    total = len(executions)
    success_rate = len(successful) / total if total > 0 else 0

    print()
    print("Overall:")
    print(f"  Total executions: {total}")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Mean latency: {sum(e.latency_ms for e in successful)/len(successful):.0f} ms")
    print(f"  Mean quality: {sum(e.quality_score for e in successful)/len(successful):.2f}")

    # By stakes level
    print()
    print("By Stakes Level:")
    for stakes in ["low", "medium", "high"]:
        stake_execs = [e for e in successful if e.stakes_level == stakes]
        if stake_execs:
            mean_lat = sum(e.latency_ms for e in stake_execs) / len(stake_execs)
            mean_qual = sum(e.quality_score for e in stake_execs) / len(stake_execs)
            print(f"  {stakes.upper()}: count={len(stake_execs)}, latency={mean_lat:.0f}ms, quality={mean_qual:.2f}")

    # By complexity
    print()
    print("By Complexity:")
    for comp in ["simple", "medium", "complex"]:
        comp_execs = [e for e in successful if e.complexity == comp]
        if comp_execs:
            mean_lat = sum(e.latency_ms for e in comp_execs) / len(comp_execs)
            mean_qual = sum(e.quality_score for e in comp_execs) / len(comp_execs)
            print(f"  {comp.upper()}: count={len(comp_execs)}, latency={mean_lat:.0f}ms, quality={mean_qual:.2f}")

    # Export to JSON format compatible with Thor's data
    print()
    print("=" * 80)
    print("EXPORTING DATA")
    print("=" * 80)

    # Build export structure
    export_data = {
        "metadata": {
            "source": "sprout_edge",
            "hardware": "Jetson Orin Nano 8GB",
            "model": "epistemic-pragmatism",
            "irp_iterations": 3,
            "collected_at": datetime.now().isoformat(),
            "initial_temperature": initial_temp,
            "initial_memory_gb": initial_mem
        },
        "overall": {
            "total_executions": total,
            "success_rate": success_rate,
            "mean_latency_ms": sum(e.latency_ms for e in successful) / len(successful) if successful else 0,
            "mean_quality": sum(e.quality_score for e in successful) / len(successful) if successful else 0
        },
        "by_stakes": {},
        "by_complexity": {},
        "by_task": {},
        "raw_executions": [asdict(e) for e in executions]
    }

    # Aggregate by stakes
    for stakes in ["low", "medium", "high"]:
        stake_execs = [e for e in successful if e.stakes_level == stakes]
        if stake_execs:
            export_data["by_stakes"][stakes] = {
                "count": len(stake_execs),
                "success_rate": len(stake_execs) / len([e for e in executions if e.stakes_level == stakes]),
                "mean_latency_ms": sum(e.latency_ms for e in stake_execs) / len(stake_execs),
                "mean_quality": sum(e.quality_score for e in stake_execs) / len(stake_execs)
            }

    # Aggregate by complexity
    for comp in ["simple", "medium", "complex"]:
        comp_execs = [e for e in successful if e.complexity == comp]
        if comp_execs:
            latencies = sorted([e.latency_ms for e in comp_execs])
            export_data["by_complexity"][comp] = {
                "count": len(comp_execs),
                "success_rate": len(comp_execs) / len([e for e in executions if e.complexity == comp]),
                "mean_latency_ms": sum(e.latency_ms for e in comp_execs) / len(comp_execs),
                "p50_latency_ms": latencies[len(latencies)//2] if latencies else 0,
                "p95_latency_ms": latencies[int(len(latencies)*0.95)] if len(latencies) > 1 else latencies[-1] if latencies else 0,
                "mean_quality": sum(e.quality_score for e in comp_execs) / len(comp_execs)
            }

    # Aggregate by task type
    task_types = set(e.task_type for e in executions)
    for task in task_types:
        task_execs = [e for e in successful if e.task_type == task]
        if task_execs:
            latencies = sorted([e.latency_ms for e in task_execs])
            export_data["by_task"][task] = {
                "count": len(task_execs),
                "success_rate": len(task_execs) / len([e for e in executions if e.task_type == task]),
                "mean_latency_ms": sum(e.latency_ms for e in task_execs) / len(task_execs),
                "p50_latency_ms": latencies[len(latencies)//2] if latencies else 0,
                "mean_quality": sum(e.quality_score for e in task_execs) / len(task_execs),
                "complexity": task_execs[0].complexity,
                "stakes_level": task_execs[0].stakes_level
            }

    # Save to file
    output_path = Path(__file__).parent / "sprout_edge_empirical_data.json"
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Data exported to: {output_path}")
    print(f"Total executions: {total}")
    print(f"File size: {output_path.stat().st_size} bytes")

    print()
    print("=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)

    return export_data


if __name__ == "__main__":
    collect_edge_data()
