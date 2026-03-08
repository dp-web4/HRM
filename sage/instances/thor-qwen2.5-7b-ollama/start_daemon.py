#!/usr/bin/env python3
"""
Start SAGE daemon for thor-qwen2.5-7b-ollama instance

Uses Ollama HTTP API with llama.cpp backend for high-performance inference.
Performance: 35+ tok/sec on Jetson ARM.
"""

import sys
from pathlib import Path

# Add SAGE root to path
sage_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sage_root))

from sage.core.sage_consciousness import SAGEConsciousness
from sage.irp.plugins.ollama_qwen_irp import OllamaQwenIRP


def main():
    print("=" * 80)
    print("SAGE Consciousness Loop - Ollama Backend Test")
    print("=" * 80)
    print(f"Instance: thor-qwen2.5-7b-ollama")
    print(f"Backend: Ollama (llama.cpp)")
    print(f"Model: qwen2.5:7b")
    print(f"Expected performance: 35+ tok/sec")
    print("=" * 80)
    print()

    # Initialize Ollama IRP plugin
    print("Initializing Ollama IRP plugin...")
    llm_plugin = OllamaQwenIRP(
        model_name="qwen2.5:7b",
        keep_alive="-1",  # Keep model loaded indefinitely
        timeout=120
    )

    # Preload model to avoid cold-start delay on first request
    print()
    print("Preloading model into VRAM...")
    # Try to preload - if it fails (empty prompt issue), that's OK
    # Model will load on first real request
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5:7b", "prompt": "test", "stream": False, "keep_alive": "-1"},
            timeout=60
        )
        if response.status_code == 200:
            print("✅ Model preloaded successfully")
        else:
            print("⚠️  Preload returned non-200 - will load on first request")
    except Exception as e:
        print(f"⚠️  Preload failed: {e}")
        print("Model will load on first request (expect ~50s delay)")

    print()

    # Minimal config for testing
    config = {
        'model_name': 'qwen2.5:7b',
        'ollama_host': 'http://localhost:11434',
        'max_atp': 100.0,
        'circadian_period': 100,
    }

    # Create SAGE consciousness instance
    print("Initializing SAGE consciousness...")
    sage = SAGEConsciousness(
        config=config,
        initial_atp=100.0,
        enable_circadian=True,
        simulation_mode=True,  # Use cycle counts for testing
        llm_plugin=llm_plugin
    )

    print("✅ SAGE consciousness initialized")
    print()
    print("=" * 80)
    print("Starting consciousness loop (100 cycles)")
    print("=" * 80)
    print()

    # Run consciousness loop
    try:
        # Run manually instead of using .run() method
        for cycle in range(100):
            sage.step()

            # Print periodic status
            if cycle % 10 == 0:
                print(f"[Cycle {cycle}] State: {sage.metabolic.current_state.value:8s} "
                      f"ATP: {sage.metabolic.current_atp:5.1f}/100")

        print()
        print("=" * 80)
        print("Test Complete!")
        print("=" * 80)
        print(f"Final state: {sage.metabolic.current_state.value}")
        print(f"Final ATP: {sage.metabolic.current_atp:.1f}")
        print(f"Total cycles: 100")
        print()

    except KeyboardInterrupt:
        print("\n\nReceived interrupt - stopping...")

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
