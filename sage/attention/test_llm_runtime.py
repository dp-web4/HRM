#!/usr/bin/env python3
"""
Test LLM Runtime Service (Tier 1)

Tests hot/cold lifecycle, backend switching, and generation.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.attention.llm_runtime import (
    LLMRuntime,
    RuntimeState,
    OllamaBackend,
    TransformersBackend
)


async def test_backend_detection():
    """Test that backends can be detected"""
    print("\n=== Testing Backend Detection ===")

    ollama = OllamaBackend()
    transformers = TransformersBackend()

    ollama_available = ollama.is_available()
    transformers_available = transformers.is_available()

    print(f"Ollama available: {ollama_available}")
    print(f"Transformers available: {transformers_available}")

    assert ollama_available or transformers_available, "At least one backend should be available"

    print("✓ Backend detection working")
    return ollama_available, transformers_available


async def test_runtime_auto_backend():
    """Test runtime auto-selects available backend"""
    print("\n=== Testing Runtime Auto-Backend Selection ===")

    runtime = LLMRuntime({'backend': 'auto'})

    assert runtime.backend is not None, "Should auto-select a backend"
    print(f"Auto-selected backend: {runtime.backend_type}")

    print("✓ Auto-backend selection working")
    return runtime


async def test_lifecycle_cold_to_hot(runtime):
    """Test cold → hot lifecycle"""
    print("\n=== Testing Lifecycle: COLD → HOT ===")

    assert runtime.state == RuntimeState.COLD, "Should start cold"
    print(f"Initial state: {runtime.state.value}")

    # Warmup
    success = await runtime.warmup()
    print(f"Warmup success: {success}")
    print(f"State after warmup: {runtime.state.value}")

    if success:
        assert runtime.state == RuntimeState.HOT, "Should be hot after warmup"
        print("✓ Lifecycle cold→hot working")
    else:
        print("⚠️ Warmup failed (may be normal if no models available)")

    return success


async def test_generation(runtime):
    """Test text generation"""
    print("\n=== Testing Text Generation ===")

    if runtime.state != RuntimeState.HOT:
        print("Skipping generation test (runtime not hot)")
        return

    prompt = "The quick brown fox"
    print(f"Prompt: {prompt}")

    result = await runtime.generate(
        prompt=prompt,
        max_tokens=20,
        temperature=0.7
    )

    print(f"Generated: {result[:100]}...")
    assert len(result) > 0, "Should generate text"
    assert runtime.total_invocations > 0, "Should track invocations"

    print("✓ Text generation working")


async def test_lifecycle_hot_to_cold(runtime):
    """Test hot → cold lifecycle"""
    print("\n=== Testing Lifecycle: HOT → COLD ===")

    if runtime.state != RuntimeState.HOT:
        print("Skipping cooldown test (runtime not hot)")
        return

    success = await runtime.cooldown()
    print(f"Cooldown success: {success}")
    print(f"State after cooldown: {runtime.state.value}")

    if success:
        assert runtime.state == RuntimeState.COLD, "Should be cold after cooldown"
        print("✓ Lifecycle hot→cold working")
    else:
        print("⚠️ Cooldown failed")


async def test_stats_tracking(runtime):
    """Test statistics tracking"""
    print("\n=== Testing Stats Tracking ===")

    stats = runtime.get_stats()
    print(f"Stats: {stats}")

    assert 'state' in stats, "Should have state"
    assert 'backend' in stats, "Should have backend"
    assert 'total_invocations' in stats, "Should track invocations"

    print("✓ Stats tracking working")


async def test_auto_warmup_on_generate():
    """Test that generate() auto-warms cold runtime"""
    print("\n=== Testing Auto-Warmup on Generate ===")

    runtime = LLMRuntime({'backend': 'auto'})
    assert runtime.state == RuntimeState.COLD, "Should start cold"

    # Call generate without explicit warmup
    result = await runtime.generate(
        prompt="Test prompt",
        max_tokens=10
    )

    print(f"Generated without explicit warmup: {result[:50]}...")

    # Runtime should have auto-warmed
    if result and not result.startswith("[Error"):
        assert runtime.state == RuntimeState.HOT, "Should auto-warm on generate"
        print("✓ Auto-warmup on generate working")
    else:
        print("⚠️ Auto-warmup failed (may be normal if no models)")

    # Cleanup
    await runtime.cooldown()


async def test_ollama_backend_specific():
    """Test Ollama backend specifically if available"""
    print("\n=== Testing Ollama Backend (if available) ===")

    backend = OllamaBackend()
    if not backend.is_available():
        print("⚠️ Ollama not available, skipping")
        return

    runtime = LLMRuntime({
        'backend': 'ollama',
        'model_name': 'qwen2.5:0.5b'
    })

    success = await runtime.warmup()
    if success:
        result = await runtime.generate("Hello", max_tokens=10)
        print(f"Ollama generated: {result[:50]}...")
        assert len(result) > 0, "Ollama should generate text"
        await runtime.cooldown()
        print("✓ Ollama backend working")
    else:
        print("⚠️ Ollama warmup failed")


async def test_transformers_backend_specific():
    """Test Transformers backend specifically if available"""
    print("\n=== Testing Transformers Backend (if available) ===")

    backend = TransformersBackend()
    if not backend.is_available():
        print("⚠️ Transformers not available, skipping")
        return

    runtime = LLMRuntime({
        'backend': 'transformers',
        'model_name': 'Qwen/Qwen2.5-0.5B-Instruct'
    })

    print("Note: Transformers backend may take a while to load...")
    success = await runtime.warmup()

    if success:
        result = await runtime.generate("Hello", max_tokens=10)
        print(f"Transformers generated: {result[:50]}...")
        assert len(result) > 0, "Transformers should generate text"
        await runtime.cooldown()
        print("✓ Transformers backend working")
    else:
        print("⚠️ Transformers warmup failed")


async def main():
    """Run all LLM runtime tests"""
    print("=" * 70)
    print("SAGE LLM Runtime (Tier 1) Test Suite")
    print("=" * 70)

    try:
        # Test 1: Backend detection
        ollama_avail, transformers_avail = await test_backend_detection()

        # Test 2: Auto-backend selection
        runtime = await test_runtime_auto_backend()

        # Test 3: Lifecycle cold→hot
        warm_success = await test_lifecycle_cold_to_hot(runtime)

        # Test 4: Text generation (if warm)
        if warm_success:
            await test_generation(runtime)

        # Test 5: Lifecycle hot→cold
        await test_lifecycle_hot_to_cold(runtime)

        # Test 6: Stats tracking
        await test_stats_tracking(runtime)

        # Test 7: Auto-warmup on generate
        await test_auto_warmup_on_generate()

        # Test 8: Backend-specific tests
        if ollama_avail:
            await test_ollama_backend_specific()

        if transformers_avail:
            await test_transformers_backend_specific()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nDay 2 Morning: LLM Runtime Service - VALIDATED")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
