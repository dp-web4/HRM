#!/usr/bin/env python3
"""
Test LLM Runtime (Tier 1)

Tests backend lifecycle, inference, and runtime orchestration.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.llm import (
    LLMRuntime,
    LLMRequest,
    BackendState,
    OllamaBackend,
    TransformersBackend
)


async def test_ollama_backend():
    """Test Ollama backend (if available)"""
    print("\n=== Testing Ollama Backend ===")

    config = {
        'model_name': 'llama3.2:1b',  # Small model for testing
        'base_url': 'http://localhost:11434',
        'timeout_s': 30,
    }

    backend = OllamaBackend(config)
    print(f"Backend created: {backend.model_name}")

    # Test warm
    print("Warming backend...")
    warmed = await backend.warm()

    if not warmed:
        print("⚠️  Ollama backend not available (server not running?)")
        print("  Skipping Ollama tests")
        return False

    assert backend.is_ready(), "Backend should be ready after warming"
    print(f"✓ Backend warmed (state={backend.state.value})")

    # Test generate
    print("Testing inference...")
    request = LLMRequest(
        prompt="What is 2+2? Answer in one word.",
        max_tokens=10,
        temperature=0.1,
    )

    response = await backend.generate(request)
    print(f"Generated: '{response.text}'")
    print(f"Tokens: {response.tokens_generated}, Time: {response.inference_time_ms:.1f}ms")
    assert response.finish_reason in ['stop', 'length'], "Should complete successfully"
    assert len(response.text) > 0, "Should generate text"
    print("✓ Inference working")

    # Test health check
    health = backend.health_check()
    print(f"Health: {health}")
    assert health['state'] == 'hot', "Should be hot"
    print("✓ Health check working")

    # Test cool
    print("Cooling backend...")
    cooled = await backend.cool()
    assert cooled, "Should cool successfully"
    assert backend.state == BackendState.COLD, "Should be cold"
    print("✓ Backend cooled")

    return True


async def test_transformers_backend():
    """Test Transformers backend (if available)"""
    print("\n=== Testing Transformers Backend ===")

    # Use a very small model for testing
    config = {
        'model_name': 'Qwen/Qwen2.5-0.5B-Instruct',  # Smallest Qwen model
        'device': 'cuda',
        'dtype': 'float16',
        'load_in_4bit': False,  # Disable for quick test
        'max_memory_gb': 4,
    }

    backend = TransformersBackend(config)
    print(f"Backend created: {backend.model_name}")

    # Test warm
    print("Warming backend (this may take a minute)...")
    warmed = await backend.warm()

    if not warmed:
        print("⚠️  Transformers backend failed to warm")
        print("  This is expected if model not downloaded or GPU unavailable")
        print("  Skipping Transformers tests")
        return False

    assert backend.is_ready(), "Backend should be ready after warming"
    print(f"✓ Backend warmed (memory={backend.memory_usage_mb:.0f}MB)")

    # Test generate
    print("Testing inference...")
    request = LLMRequest(
        prompt="What is 2+2? Answer: ",
        max_tokens=5,
        temperature=0.1,
    )

    response = await backend.generate(request)
    print(f"Generated: '{response.text}'")
    print(f"Tokens: {response.tokens_generated}, Time: {response.inference_time_ms:.1f}ms")
    assert response.finish_reason in ['stop', 'length'], "Should complete successfully"
    print("✓ Inference working")

    # Test health check
    health = backend.health_check()
    print(f"Health: state={health['state']}, memory={health['memory_mb']:.0f}MB")
    assert health['state'] == 'hot', "Should be hot"
    print("✓ Health check working")

    # Test cool
    print("Cooling backend...")
    cooled = await backend.cool()
    assert cooled, "Should cool successfully"
    assert backend.state == BackendState.COLD, "Should be cold"
    print("✓ Backend cooled")

    return True


async def test_llm_runtime_ollama():
    """Test LLM Runtime with Ollama backend"""
    print("\n=== Testing LLM Runtime (Ollama) ===")

    config = {
        'backend_type': 'ollama',
        'backend_config': {
            'model_name': 'llama3.2:1b',
            'base_url': 'http://localhost:11434',
        },
        'auto_warm': False,
        'enable_auto_cool': False,
    }

    runtime = LLMRuntime(config)
    await runtime.start()

    # Test generate (should auto-warm)
    print("Testing auto-warm on generate...")
    response = await runtime.generate(
        "Hello! Respond with just 'Hi'",
        max_tokens=5,
        temperature=0.1
    )

    if response.finish_reason == 'error':
        print("⚠️  Ollama runtime not available")
        print("  Skipping runtime tests")
        await runtime.stop()
        return False

    print(f"Generated: '{response.text}'")
    assert runtime.is_ready(), "Runtime should be ready after generate"
    print("✓ Auto-warm working")

    # Test stats
    stats = runtime.get_stats()
    print(f"Stats: {stats}")
    assert stats['total_requests'] == 1, "Should have 1 request"
    assert stats['total_tokens_generated'] > 0, "Should have tokens"
    print("✓ Statistics tracking working")

    # Test health check
    health = runtime.health_check()
    print(f"Health: runtime_state={health['runtime']['backend_state']}")
    print("✓ Health check working")

    # Stop runtime (should cool)
    await runtime.stop()
    assert runtime.get_state() == BackendState.COLD, "Should be cold after stop"
    print("✓ Stop (with cool) working")

    return True


async def test_llm_runtime_auto_cool():
    """Test LLM Runtime auto-cool feature"""
    print("\n=== Testing LLM Runtime Auto-Cool ===")

    config = {
        'backend_type': 'ollama',
        'backend_config': {
            'model_name': 'llama3.2:1b',
            'base_url': 'http://localhost:11434',
        },
        'auto_warm': True,
        'enable_auto_cool': True,
        'auto_cool_timeout_s': 5,  # Short timeout for testing
    }

    runtime = LLMRuntime(config)
    await runtime.start()

    if not runtime.is_ready():
        print("⚠️  Ollama runtime not available, skipping auto-cool test")
        await runtime.stop()
        return False

    print(f"Runtime started (state={runtime.get_state().value})")

    # Make a request to start idle timer
    await runtime.generate("Test", max_tokens=1, temperature=0.1)
    print("Request made, waiting for auto-cool (5s)...")

    # Wait for auto-cool
    await asyncio.sleep(7)

    # Check if cooled
    state = runtime.get_state()
    print(f"State after idle: {state.value}")
    assert state == BackendState.COLD, "Should auto-cool after timeout"
    print("✓ Auto-cool working")

    await runtime.stop()
    return True


async def main():
    """Run all LLM runtime tests"""
    print("=" * 70)
    print("SAGE LLM Runtime (Tier 1) Test Suite")
    print("=" * 70)

    results = {}

    try:
        # Test 1: Ollama backend
        results['ollama_backend'] = await test_ollama_backend()

        # Test 2: Transformers backend (optional, may be slow)
        # Uncomment to test Transformers backend
        # results['transformers_backend'] = await test_transformers_backend()

        # Test 3: LLM Runtime with Ollama
        if results.get('ollama_backend', False):
            results['llm_runtime_ollama'] = await test_llm_runtime_ollama()

            # Test 4: Auto-cool feature
            if results.get('llm_runtime_ollama', False):
                results['llm_runtime_auto_cool'] = await test_llm_runtime_auto_cool()

        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)

        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "⚠️  SKIPPED"
            print(f"{test_name}: {status}")

        # Check if any tests actually ran
        if not any(results.values()):
            print("\n⚠️  WARNING: No LLM backends available for testing")
            print("   To test Ollama: Start Ollama server (ollama serve)")
            print("   To test Transformers: Ensure CUDA available and model downloaded")
        else:
            print("\n✓ LLM Runtime Tier 1 - VALIDATED")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
