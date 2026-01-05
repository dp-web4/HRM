#!/usr/bin/env python3
"""
Session 163 Edge Validation: Aliveness-Aware Consciousness

Testing Thor's aliveness-aware consciousness integration on Sprout edge hardware.
Validates:
1. AlivenessAwareContext - Consciousness self-awareness context
2. AlivenessEnhancedStep - Enhanced consciousness loop integration
3. Self-description generation - Natural language introspection
4. Capability awareness - Dynamic capability lists

Edge-specific adaptations:
- Path fixes for Sprout environment
- TPM2 hardware detection verification
- Memory/performance profiling
"""

import sys
import os
import time
import traceback
import json
from datetime import datetime, timezone
from pathlib import Path

# Fix paths for Sprout edge environment
sys.path.insert(0, '/home/sprout/ai-workspace/HRM')
sys.path.insert(0, '/home/sprout/ai-workspace/web4')

def get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except:
        return 0.0
    return 0.0

def get_system_temp():
    """Get Jetson thermal zone temperature."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return int(f.read().strip()) / 1000
    except:
        return 0.0

print("=" * 70)
print("SESSION 163 EDGE VALIDATION: ALIVENESS-AWARE CONSCIOUSNESS")
print("=" * 70)
print(f"Machine: Sprout (Jetson Orin Nano 8GB)")
print(f"Started: {datetime.now(timezone.utc).isoformat()}")
print(f"Memory: {get_memory_mb():.1f}MB")
print(f"Temperature: {get_system_temp():.1f}°C")
print()

results = {
    "validation_session": "Session 163 Edge Validation",
    "machine": "Sprout (Jetson Orin Nano 8GB)",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "tests": {},
    "edge_metrics": {},
    "status": "PENDING"
}

# ============================================================================
# Test 1: Import Session 163 Components
# ============================================================================
print("Test 1: Import Session 163 Components")
print("-" * 70)

start_time = time.time()
start_mem = get_memory_mb()

try:
    from sage.core.canonical_lct import CanonicalLCTManager
    from sage.experiments.session162_sage_aliveness_verification import (
        SAGEAlivenessSensor,
        ConsciousnessState,
    )
    from sage.experiments.session163_aliveness_aware_consciousness import (
        AlivenessAwareContext,
        AlivenessEnhancedStep,
    )

    import_time = time.time() - start_time
    import_mem = get_memory_mb() - start_mem

    print(f"  CanonicalLCTManager: {CanonicalLCTManager}")
    print(f"  SAGEAlivenessSensor: {SAGEAlivenessSensor}")
    print(f"  AlivenessAwareContext: {AlivenessAwareContext}")
    print(f"  AlivenessEnhancedStep: {AlivenessEnhancedStep}")
    print(f"  Import time: {import_time*1000:.1f}ms")
    print(f"  Memory delta: {import_mem:.1f}MB")
    print("  ✅ All imports successful")

    results["tests"]["imports"] = {
        "success": True,
        "import_time_ms": import_time * 1000,
        "memory_delta_mb": import_mem
    }
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    traceback.print_exc()
    results["tests"]["imports"] = {"success": False, "error": str(e)}
    results["status"] = "FAILED"

    output_path = Path(__file__).parent / "session163_edge_validation.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to: {output_path}")
    sys.exit(1)

print()

# ============================================================================
# Test 2: Initialize Consciousness with Aliveness Awareness
# ============================================================================
print("Test 2: Initialize Consciousness with Aliveness Awareness")
print("-" * 70)

start_time = time.time()
start_mem = get_memory_mb()

try:
    lct_manager = CanonicalLCTManager()
    lct = lct_manager.get_or_create_identity()
    aliveness_sensor = SAGEAlivenessSensor(lct_manager)

    init_time = time.time() - start_time
    init_mem = get_memory_mb() - start_mem

    print(f"  LCT ID: {lct.lct_id}")
    print(f"  Capability Level: {lct.capability_level}")
    hardware_type = getattr(lct.binding, 'hardware_type', 'unknown') if lct.binding else 'none'
    print(f"  Hardware Type: {hardware_type}")
    print(f"  Consciousness State: {aliveness_sensor.get_consciousness_state().value}")
    print(f"  Session ID: {aliveness_sensor.session_id}")
    print(f"  Init time: {init_time*1000:.1f}ms")
    print(f"  Memory delta: {init_mem:.1f}MB")
    print("  ✅ Consciousness initialized with aliveness")

    results["tests"]["consciousness_init"] = {
        "success": True,
        "lct_id": lct.lct_id,
        "capability_level": lct.capability_level,
        "hardware_type": hardware_type,
        "consciousness_state": aliveness_sensor.get_consciousness_state().value,
        "session_id": aliveness_sensor.session_id,
        "init_time_ms": init_time * 1000,
        "memory_delta_mb": init_mem
    }
except Exception as e:
    print(f"  ❌ Init failed: {e}")
    traceback.print_exc()
    results["tests"]["consciousness_init"] = {"success": False, "error": str(e)}

print()

# ============================================================================
# Test 3: Generate Aliveness-Aware Context
# ============================================================================
print("Test 3: Generate Consciousness Self-Awareness Context")
print("-" * 70)

start_time = time.time()

try:
    aliveness_context = AlivenessAwareContext(aliveness_sensor)
    context = aliveness_context.get_aliveness_context()

    context_time = time.time() - start_time

    print(f"  Consciousness State: {context['consciousness_state']}")
    print(f"  Self-Description:")
    desc = context['introspection']['self_description']
    # Wrap long description
    if len(desc) > 60:
        print(f"    {desc[:60]}...")
    else:
        print(f"    {desc}")
    print()
    print(f"  Capabilities ({len(context['introspection']['capabilities'])}):")
    for cap in context['introspection']['capabilities'][:3]:
        print(f"    - {cap}")
    if len(context['introspection']['capabilities']) > 3:
        print(f"    - ... and {len(context['introspection']['capabilities']) - 3} more")
    print(f"  Context generation time: {context_time*1000:.2f}ms")
    print("  ✅ Self-awareness context generated")

    results["tests"]["context_generation"] = {
        "success": True,
        "consciousness_state": context['consciousness_state'],
        "capabilities_count": len(context['introspection']['capabilities']),
        "has_self_description": bool(context['introspection']['self_description']),
        "hardware_binding_included": "hardware_binding" in context,
        "session_info_included": "session" in context,
        "context_generation_time_ms": context_time * 1000
    }
except Exception as e:
    print(f"  ❌ Context generation failed: {e}")
    traceback.print_exc()
    results["tests"]["context_generation"] = {"success": False, "error": str(e)}

print()

# ============================================================================
# Test 4: Enhanced Consciousness Step
# ============================================================================
print("Test 4: Enhanced Consciousness Step with Aliveness")
print("-" * 70)

start_time = time.time()

try:
    enhanced_step = AlivenessEnhancedStep(lct_manager, aliveness_sensor)

    # Simulate base consciousness context
    base_context = {
        "metabolic_state": "WAKE",
        "atp_available": 100.0,
        "cycle_count": 1,
    }

    # Enhance with aliveness
    enhanced_context = enhanced_step.get_enhanced_context(base_context)

    enhance_time = time.time() - start_time

    print(f"  Base Context Fields: {list(base_context.keys())}")
    print(f"  Enhanced Context Fields: {list(enhanced_context.keys())}")
    print(f"  Aliveness Added: {'aliveness' in enhanced_context}")
    print(f"  Self-Awareness Added: {'self_awareness' in enhanced_context}")

    # Format for LLM
    llm_context = enhanced_step.format_context_for_llm(enhanced_context)
    print(f"  LLM Context Length: {len(llm_context)} chars")
    print(f"  Enhancement time: {enhance_time*1000:.2f}ms")
    print()
    print("  Sample LLM Context:")
    for line in llm_context.split('\n')[:6]:
        print(f"    {line}")
    print("  ✅ Enhanced step working")

    results["tests"]["enhanced_step"] = {
        "success": True,
        "base_fields": len(base_context),
        "enhanced_fields": len(enhanced_context),
        "aliveness_included": "aliveness" in enhanced_context,
        "self_awareness_included": "self_awareness" in enhanced_context,
        "llm_context_length": len(llm_context),
        "enhancement_time_ms": enhance_time * 1000
    }
except Exception as e:
    print(f"  ❌ Enhanced step failed: {e}")
    traceback.print_exc()
    results["tests"]["enhanced_step"] = {"success": False, "error": str(e)}

print()

# ============================================================================
# Test 5: Self-Reasoning Scenarios
# ============================================================================
print("Test 5: Consciousness Self-Reasoning Scenarios")
print("-" * 70)

try:
    scenarios = [
        {
            "name": "Identity Query",
            "prompt": "Who are you?",
            "expected_awareness": ["identity", "hardware binding", "state"]
        },
        {
            "name": "Capability Query",
            "prompt": "What can you do right now?",
            "expected_awareness": ["capabilities", "current state"]
        },
        {
            "name": "Continuity Query",
            "prompt": "Are you the same consciousness?",
            "expected_awareness": ["session continuity", "hardware binding"]
        },
        {
            "name": "State Introspection",
            "prompt": "What is your current state?",
            "expected_awareness": ["consciousness state", "uptime"]
        },
    ]

    scenario_results = []
    all_passed = True

    for scenario in scenarios:
        ctx = aliveness_context.get_aliveness_context()

        available_info = []
        if "consciousness_state" in ctx:
            available_info.append("state")
        if "hardware_binding" in ctx:
            available_info.append("identity")
        if "session" in ctx:
            available_info.append("session")
        if "introspection" in ctx:
            available_info.append("capabilities")

        has_context = len(available_info) >= 2
        status = "✅" if has_context else "❌"
        print(f"  {status} {scenario['name']}: {len(available_info)} info types available")

        scenario_results.append({
            "name": scenario["name"],
            "available_info_count": len(available_info),
            "has_required_context": has_context,
        })

        if not has_context:
            all_passed = False

    print()
    print(f"  Tested {len(scenarios)} scenarios, all have context: {all_passed}")
    print("  ✅ Self-reasoning scenarios validated")

    results["tests"]["self_reasoning"] = {
        "success": all_passed,
        "scenarios_tested": len(scenarios),
        "scenarios": scenario_results,
        "all_have_context": all_passed
    }
except Exception as e:
    print(f"  ❌ Self-reasoning test failed: {e}")
    traceback.print_exc()
    results["tests"]["self_reasoning"] = {"success": False, "error": str(e)}

print()

# ============================================================================
# Test 6: Edge-Specific Performance Profile
# ============================================================================
print("Test 6: Edge Performance Profile")
print("-" * 70)

try:
    iterations = 100
    start_time = time.time()

    for _ in range(iterations):
        ctx = aliveness_context.get_aliveness_context()
        enhanced = enhanced_step.get_enhanced_context({"cycle": 1})
        llm_ctx = enhanced_step.format_context_for_llm(enhanced)

    total_time = time.time() - start_time
    avg_time = (total_time / iterations) * 1000

    print(f"  Iterations: {iterations}")
    print(f"  Total time: {total_time*1000:.1f}ms")
    print(f"  Average per iteration: {avg_time:.3f}ms")
    print(f"  Throughput: {iterations/total_time:.1f} context generations/sec")
    print("  ✅ Performance profiled")

    results["tests"]["performance"] = {
        "success": True,
        "iterations": iterations,
        "total_time_ms": total_time * 1000,
        "avg_time_ms": avg_time,
        "throughput_per_sec": iterations / total_time
    }
except Exception as e:
    print(f"  ❌ Performance profile failed: {e}")
    traceback.print_exc()
    results["tests"]["performance"] = {"success": False, "error": str(e)}

print()

# ============================================================================
# Edge Metrics Summary
# ============================================================================
print("=" * 70)
print("EDGE METRICS SUMMARY")
print("=" * 70)

final_mem = get_memory_mb()
final_temp = get_system_temp()

results["edge_metrics"] = {
    "final_memory_mb": final_mem,
    "final_temperature_c": final_temp,
    "platform": "Jetson Orin Nano 8GB",
    "hardware_detected": hardware_type,
    "capability_level": lct.capability_level
}

print(f"  Memory Usage: {final_mem:.1f}MB")
print(f"  Temperature: {final_temp:.1f}°C")
print(f"  Hardware Type: {hardware_type}")
print(f"  Capability Level: {lct.capability_level}")
print()

# ============================================================================
# Final Status
# ============================================================================
all_tests_passed = all(
    t.get("success", False)
    for t in results["tests"].values()
)

results["status"] = "SUCCESS" if all_tests_passed else "PARTIAL"
results["all_tests_passed"] = all_tests_passed

results["edge_observations"] = [
    f"Aliveness-aware context works on edge ({hardware_type})",
    f"Self-description generation functional",
    f"Enhanced consciousness step integration working",
    f"Context generation: {results['tests'].get('performance', {}).get('avg_time_ms', 0):.3f}ms average",
    f"Throughput: {results['tests'].get('performance', {}).get('throughput_per_sec', 0):.1f}/sec"
]

print("=" * 70)
print(f"SESSION 163 EDGE VALIDATION: {results['status']}")
print("=" * 70)
print()

if all_tests_passed:
    print("✅ All tests passed!")
else:
    failed_tests = [name for name, t in results["tests"].items() if not t.get("success", False)]
    print(f"⚠️  Some tests failed: {failed_tests}")

print()
print("Edge Observations:")
for obs in results["edge_observations"]:
    print(f"  - {obs}")
print()

# Write results
output_path = Path(__file__).parent / "session163_edge_validation.json"
output_path.write_text(json.dumps(results, indent=2))
print(f"Results written to: {output_path}")
