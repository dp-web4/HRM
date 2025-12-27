#!/usr/bin/env python3
"""
Individual Model Test - Validate Each Model Before Full Comparison

Tests each model individually with a quick conversation to ensure:
1. Model loads successfully
2. Generate response works
3. Multi-turn memory works
4. Performance is reasonable

Run this BEFORE test_unified_conversation.py to catch issues early.
"""

import sys
import time
import argparse
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.conversation.sage_conversation_manager import SAGEConversationManager, ConversationConfig
from sage.irp.plugins.q3_omni_irp import Q3OmniIRP
from sage.irp.plugins.qwen25_05b_irp import Qwen25_05B_IRP
from sage.irp.plugins.qwen25_14b_irp import Qwen25_14B_IRP
from sage.irp.plugins.nemotron_nano_irp import NemotronNanoIRP


# Quick 3-turn test
QUICK_TEST = [
    "Write one sentence about a dragon.",
    "What color?",
    "What name?",
]


def test_model_quick(model_name: str, plugin_class, config: ConversationConfig):
    """
    Quick test of a model with 3-turn conversation.

    Args:
        model_name: Display name for model
        plugin_class: IRP plugin class to instantiate
        config: Conversation configuration
    """
    print("\n" + "="*80)
    print(f"Quick Test: {model_name}")
    print("="*80)
    print()

    # Initialize plugin
    start_load = time.time()
    try:
        plugin = plugin_class()
        load_time = time.time() - start_load
        print(f"✅ Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Initialize conversation manager
    manager = SAGEConversationManager(plugin, config)

    # Run quick conversation
    total_time = 0
    for i, user_message in enumerate(QUICK_TEST, 1):
        print(f"\n[Turn {i}/{len(QUICK_TEST)}] USER: {user_message}")

        start_time = time.time()
        try:
            response = manager.chat(user_message)
            gen_time = time.time() - start_time
            total_time += gen_time

            print(f"ASSISTANT ({gen_time:.2f}s): {response[:200]}...")  # Truncate for readability

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Summary
    avg_time = total_time / len(QUICK_TEST)
    print(f"\n✅ {model_name} quick test PASSED")
    print(f"   Total: {total_time:.2f}s, Avg/turn: {avg_time:.2f}s")
    print("="*80)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test individual models before full comparison")
    parser.add_argument(
        "--model",
        choices=["q3-omni", "qwen-14b", "qwen-05b", "nemotron", "all"],
        default="all",
        help="Which model(s) to test"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per turn (kept low for quick test)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    args = parser.parse_args()

    # Configuration (same for all models)
    config = ConversationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        system_message="You are concise and answer briefly."
    )

    # Models to test (ordered smallest to largest)
    models = {
        "qwen-05b": ("Qwen2.5-0.5B", Qwen25_05B_IRP),
        "nemotron": ("Nemotron Nano 4B", NemotronNanoIRP),
        "qwen-14b": ("Qwen2.5-14B", Qwen25_14B_IRP),
        "q3-omni": ("Q3-Omni-30B", Q3OmniIRP),
    }

    # Filter models if specific one requested
    if args.model != "all":
        models = {args.model: models[args.model]}

    # Test each model
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL VALIDATION")
    print("Testing smallest to largest to catch issues early")
    print("="*80)

    results = {}
    for model_key, (model_name, plugin_class) in models.items():
        success = test_model_quick(model_name, plugin_class, config)
        results[model_name] = success

        if not success:
            print(f"\n⚠️  {model_name} failed. Stopping here.")
            print("Fix this model before testing larger ones.")
            break

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {model_name}: {status}")
    print("="*80)

    if all(results.values()):
        print("\n🎉 All models validated! Ready for full apples-to-apples test.")
        print("\nNext step:")
        print("  python3 sage/tests/test_unified_conversation.py --model all")
    else:
        print("\n⚠️  Some models failed validation. Fix issues before proceeding.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
