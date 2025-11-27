"""
Test multi-modal ATP pricing with real empirical data.

Validates that:
1. Vision tasks (Thor Session #79) price reasonably
2. LLM tasks (Sprout Session #21) price fairly
3. Different modalities can compete economically
4. Backward compatible with existing single-model approach
"""

import json
import sys
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.multimodal_atp_pricing import MultiModalATPPricer, infer_task_type


def load_thor_vision_data():
    """Load Thor's vision task empirical data (Session #79)."""
    # Path: HRM/sage/tests -> ai-workspace
    filepath = Path(__file__).parent.parent.parent.parent / "web4" / "game" / "sage_empirical_data.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def load_sprout_llm_data():
    """Load Sprout's LLM inference empirical data (Session #21)."""
    filepath = Path(__file__).parent / "sprout_edge_empirical_data.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def test_vision_pricing():
    """Test ATP pricing for vision tasks matches expected ranges."""
    print("=== Test 1: Vision Task Pricing ===\n")

    pricer = MultiModalATPPricer()
    thor_data = load_thor_vision_data()

    # Test with actual task data
    test_cases = [
        # (task_name, complexity, mean_latency_ms, mean_quality, expected_range)
        ("image_classification", "low", 22.3, 0.888, (20, 30)),
        ("object_detection", "medium", 40.5, 0.673, (40, 60)),
        ("action_recognition", "high", 86.1, 0.611, (70, 95))
    ]

    results = []
    for task_name, complexity, latency, quality, expected_range in test_cases:
        cost = pricer.calculate_cost("vision", complexity, latency, quality)
        in_range = expected_range[0] <= cost <= expected_range[1]
        status = "✓" if in_range else "✗"

        results.append({
            "task": task_name,
            "cost": cost,
            "expected": expected_range,
            "pass": in_range
        })

        print(f"{status} {task_name} ({complexity}): {cost:.1f} ATP")
        print(f"   Latency: {latency:.1f}ms, Quality: {quality:.2f}")
        print(f"   Expected: {expected_range[0]}-{expected_range[1]} ATP\n")

    passed = sum(r["pass"] for r in results)
    total = len(results)
    print(f"Vision pricing: {passed}/{total} tests passed\n")

    return all(r["pass"] for r in results)


def test_llm_pricing():
    """Test ATP pricing for LLM inference matches expected ranges."""
    print("=== Test 2: LLM Inference Pricing ===\n")

    pricer = MultiModalATPPricer()
    sprout_data = load_sprout_llm_data()

    # Use actual Sprout data
    test_cases = [
        # (complexity, mean_latency_s, mean_quality, expected_range)
        ("low", 17.92, 0.95, (30, 45)),   # Simple queries
        ("medium", 25.17, 0.90, (55, 75)),  # Medium reasoning
        ("high", 30.62, 0.85, (80, 100))    # Complex reasoning
    ]

    results = []
    for complexity, latency_ms, quality, expected_range in test_cases:
        latency_s = latency_ms / 1000.0 if latency_ms > 1000 else latency_ms
        cost = pricer.calculate_cost("llm_inference", complexity, latency_s, quality)
        in_range = expected_range[0] <= cost <= expected_range[1]
        status = "✓" if in_range else "✗"

        results.append({
            "complexity": complexity,
            "cost": cost,
            "expected": expected_range,
            "pass": in_range
        })

        print(f"{status} {complexity} complexity: {cost:.1f} ATP")
        print(f"   Latency: {latency_s:.1f}s, Quality: {quality:.2f}")
        print(f"   Expected: {expected_range[0]}-{expected_range[1]} ATP\n")

    passed = sum(r["pass"] for r in results)
    total = len(results)
    print(f"LLM pricing: {passed}/{total} tests passed\n")

    return all(r["pass"] for r in results)


def test_economic_competition():
    """Test that different modalities can compete fairly."""
    print("=== Test 3: Economic Competition ===\n")

    pricer = MultiModalATPPricer()

    # Compare value-per-ATP across modalities
    scenarios = [
        ("Vision - image classification", "vision", "low", 22.3, 0.888),
        ("LLM - simple reasoning", "llm_inference", "low", 17.9, 0.95),
        ("Vision - action recognition", "vision", "high", 86.1, 0.611),
        ("LLM - complex reasoning", "llm_inference", "high", 30.6, 0.85)
    ]

    print("Task Economics (Quality per ATP):\n")

    results = []
    for name, task_type, complexity, latency, quality in scenarios:
        cost = pricer.calculate_cost(task_type, complexity, latency, quality)
        value_per_atp = quality / cost if cost > 0 else 0

        results.append({
            "name": name,
            "cost": cost,
            "quality": quality,
            "value_per_atp": value_per_atp
        })

        print(f"{name}:")
        print(f"  Cost: {cost:.1f} ATP")
        print(f"  Quality: {quality:.2f}")
        print(f"  Value/ATP: {value_per_atp:.4f}\n")

    # Check that costs are in reasonable range (20-100 ATP)
    all_reasonable = all(20 <= r["cost"] <= 100 for r in results)

    print(f"Economic viability: {'✓ All tasks economically viable' if all_reasonable else '✗ Some tasks not viable'}\n")

    return all_reasonable


def test_task_type_inference():
    """Test automatic task type detection."""
    print("=== Test 4: Task Type Inference ===\n")

    test_cases = [
        ({"plugin_name": "vision_classifier"}, "vision"),
        ({"plugin_name": "object_detector"}, "vision"),
        ({"irp_iterations": 3, "operation": "conversation"}, "llm_inference"),
        ({"operation": "generate"}, "llm_inference"),
        ({"plugin_name": "federation_gossip"}, "coordination"),
        ({"plugin_name": "memory_consolidator"}, "consolidation"),
        ({"task_type": "vision"}, "vision"),  # Explicit override
    ]

    results = []
    for context, expected in test_cases:
        inferred = infer_task_type(context)
        match = inferred == expected
        status = "✓" if match else "✗"

        results.append(match)

        print(f"{status} {context} -> {inferred} (expected: {expected})")

    passed = sum(results)
    total = len(results)
    print(f"\nTask inference: {passed}/{total} tests passed\n")

    return all(results)


def test_backward_compatibility():
    """Test that vision pricing matches Session #79 calibration."""
    print("=== Test 5: Backward Compatibility ===\n")

    pricer = MultiModalATPPricer()

    # Load original calibrated pricing
    # Path: HRM/sage/tests -> ai-workspace
    filepath = Path(__file__).parent.parent.parent.parent / "web4" / "game" / "atp_pricing_calibrated.json"
    with open(filepath, 'r') as f:
        original = json.load(f)

    print("Original (Session #79) pricing model:")
    print(f"  Base costs: {original['base_costs']}")
    print(f"  Latency multiplier: {original['latency_multiplier']}")
    print(f"  Quality multiplier: {original['quality_multiplier']}\n")

    vision_model = pricer.get_model("vision")
    print("Multi-modal vision pricing model:")
    print(f"  Base costs: {vision_model.base_costs}")
    print(f"  Latency multiplier: {vision_model.latency_multiplier}")
    print(f"  Quality multiplier: {vision_model.quality_multiplier}\n")

    # Compare pricing for same task
    test_latency = 50.0  # ms
    test_quality = 0.75
    test_complexity = "medium"

    # Original formula
    original_cost = (
        original["base_costs"][test_complexity] +
        test_latency * original["latency_multiplier"] +
        test_quality * original["quality_multiplier"]
    )

    # Multi-modal formula
    multimodal_cost = pricer.calculate_cost("vision", test_complexity, test_latency, test_quality)

    print(f"Test case (50ms, 0.75 quality, medium complexity):")
    print(f"  Original: {original_cost:.2f} ATP")
    print(f"  Multi-modal: {multimodal_cost:.2f} ATP")
    print(f"  Difference: {abs(original_cost - multimodal_cost):.2f} ATP\n")

    # Should be very close (within 0.5 ATP)
    compatible = abs(original_cost - multimodal_cost) < 0.5

    print(f"Backward compatibility: {'✓ Passed' if compatible else '✗ Failed'}\n")

    return compatible


def test_calibration_persistence():
    """Test saving and loading calibration files."""
    print("=== Test 6: Calibration Persistence ===\n")

    pricer = MultiModalATPPricer()

    # Save calibration
    test_file = "/tmp/test_atp_calibration.json"
    pricer.save_calibration(test_file)
    print(f"✓ Saved calibration to {test_file}")

    # Load calibration
    loaded_pricer = MultiModalATPPricer.from_calibration_file(test_file)
    print(f"✓ Loaded calibration from {test_file}")

    # Compare pricing
    test_cost_original = pricer.calculate_cost("vision", "low", 20.0, 0.88)
    test_cost_loaded = loaded_pricer.calculate_cost("vision", "low", 20.0, 0.88)

    match = abs(test_cost_original - test_cost_loaded) < 0.01

    print(f"✓ Pricing match: {test_cost_original:.2f} vs {test_cost_loaded:.2f} ATP")
    print(f"\nPersistence: {'✓ Passed' if match else '✗ Failed'}\n")

    return match


def run_all_tests():
    """Run complete test suite."""
    print("╔" + "═" * 60 + "╗")
    print("║" + " Multi-Modal ATP Pricing Test Suite".center(60) + "║")
    print("╚" + "═" * 60 + "╝\n")

    tests = [
        ("Vision Task Pricing", test_vision_pricing),
        ("LLM Inference Pricing", test_llm_pricing),
        ("Economic Competition", test_economic_competition),
        ("Task Type Inference", test_task_type_inference),
        ("Backward Compatibility", test_backward_compatibility),
        ("Calibration Persistence", test_calibration_persistence)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} failed with error: {e}\n")
            results.append((name, False))

    print("\n" + "=" * 62)
    print("SUMMARY".center(62))
    print("=" * 62 + "\n")

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\nOverall: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! Multi-modal ATP pricing is working!\n")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
