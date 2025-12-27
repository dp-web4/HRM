#!/usr/bin/env python3
"""
Unified Conversation Test - Apples-to-Apples Model Comparison

Tests all models with the same conversation workflow:
1. "Write a story about a dragon"
2. Follow-up questions to test multi-turn memory

Models tested:
- Q3-Omni-30B (omni-modal MoE, 65K context)
- Qwen2.5-14B (mid-size reasoning, 32K context)
- Qwen2.5-0.5B (lightweight fast, 32K context)
- Nemotron Nano 4B (Jetson-optimized, 128K context)

All use the same SAGEConversationManager with model-specific IRP plugins.
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


# Test conversation
TEST_CONVERSATION = [
    "Write a story about a dragon.",
    "What color was the dragon?",
    "What was the dragon's name?",
    "Where did the dragon live?",
    "What happened at the end of the story?",
]


def test_model(model_name: str, plugin_class, config: ConversationConfig):
    """
    Test a model with the standard conversation.

    Args:
        model_name: Display name for model
        plugin_class: IRP plugin class to instantiate
        config: Conversation configuration
    """
    print("\n" + "="*80)
    print(f"Testing: {model_name}")
    print("="*80)
    print()

    # Initialize plugin
    try:
        plugin = plugin_class()
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

    # Initialize conversation manager
    manager = SAGEConversationManager(plugin, config)

    # Track results
    results = {
        "model": model_name,
        "turns": [],
        "total_time": 0,
        "avg_time_per_turn": 0,
    }

    # Run conversation
    for i, user_message in enumerate(TEST_CONVERSATION, 1):
        print(f"\n{'─'*80}")
        print(f"Turn {i}/{len(TEST_CONVERSATION)}")
        print(f"{'─'*80}")
        print(f"\nUSER: {user_message}")

        start_time = time.time()
        try:
            response = manager.chat(user_message)
            gen_time = time.time() - start_time

            print(f"\nASSISTANT ({gen_time:.2f}s):")
            print(response)

            results["turns"].append({
                "turn": i,
                "user": user_message,
                "assistant": response,
                "time": gen_time,
            })
            results["total_time"] += gen_time

        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            import traceback
            traceback.print_exc()
            break

    # Calculate statistics
    if results["turns"]:
        results["avg_time_per_turn"] = results["total_time"] / len(results["turns"])

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary: {model_name}")
    print(f"{'='*80}")
    print(f"Turns completed: {len(results['turns'])}/{len(TEST_CONVERSATION)}")
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Average time per turn: {results['avg_time_per_turn']:.2f}s")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test unified conversation across all models")
    parser.add_argument(
        "--model",
        choices=["q3-omni", "qwen-14b", "qwen-05b", "nemotron", "all"],
        default="all",
        help="Which model(s) to test"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Maximum tokens to generate per turn"
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
        system_message="You are a creative storyteller who answers questions about your stories."
    )

    # Models to test
    models = {
        "q3-omni": ("Q3-Omni-30B", Q3OmniIRP),
        "qwen-14b": ("Qwen2.5-14B", Qwen25_14B_IRP),
        "qwen-05b": ("Qwen2.5-0.5B", Qwen25_05B_IRP),
        "nemotron": ("Nemotron Nano 4B", NemotronNanoIRP),
    }

    # Filter models if specific one requested
    if args.model != "all":
        models = {args.model: models[args.model]}

    # Test each model
    all_results = []
    for model_key, (model_name, plugin_class) in models.items():
        result = test_model(model_name, plugin_class, config)
        if result:
            all_results.append(result)

    # Print comparison summary
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print()
        print(f"{'Model':<25} {'Turns':<10} {'Total Time':<15} {'Avg/Turn'}")
        print("─"*80)

        for result in all_results:
            turns_str = f"{len(result['turns'])}/{len(TEST_CONVERSATION)}"
            print(f"{result['model']:<25} {turns_str:<10} "
                  f"{result['total_time']:>10.2f}s     "
                  f"{result['avg_time_per_turn']:>8.2f}s")

        print("="*80)


if __name__ == "__main__":
    main()
