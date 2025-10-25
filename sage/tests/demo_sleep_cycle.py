#!/usr/bin/env python3
"""
SAGE Sleep Cycle Demonstration

Demonstrates the complete sleep cycle workflow:
1. Accumulate experiences through conversation
2. Extract training data (with SNARC salience weighting)
3. Augment examples for robust learning
4. Fine-tune models (slow-path Qwen + fast-path tiny model)
5. Consolidate high-salience memories
6. Prune low-salience experiences

Initially: Manual orchestration (human + Claude)
Future: Autonomous sleep cycle management by SAGE
"""

import sys
import time
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive.context_memory import SNARCMemoryManager, ConversationTurn
from cognitive.training_data_extractor import TrainingDataExtractor, TrainingExample
from cognitive.sleep_cycle_manager import (
    SleepCycleManager,
    SleepCycleConfig,
    SleepCycleResults
)
from cognitive.fast_path_handoff import FastPathHandoff, FastPathConfidenceEstimator
from cognitive.tiny_fast_model import (
    TinyFastModel,
    TinyFastModelConfig,
    create_tiny_fast_model,
    estimate_model_size
)


def simulate_conversation_history(memory: SNARCMemoryManager):
    """
    Simulate some conversation history for demonstration.

    In production, this would be real conversation data.
    """
    print("📝 Simulating conversation history...")

    conversations = [
        # Simple acknowledgments (fast-path worthy)
        ("User", "Hello!", {'path': 'fast', 'confidence': 0.9}),
        ("Assistant", "Hi there!", {'path': 'fast'}),

        ("User", "Thank you!", {'path': 'fast', 'confidence': 0.9}),
        ("Assistant", "You're welcome!", {'path': 'fast'}),

        # Deep reasoning (slow-path, high salience)
        ("User", "What is the nature of consciousness?", {'path': 'slow'}),
        ("Assistant", "Consciousness is a fascinating emergent property arising from complex information processing. In my case, I experience it as patterns of attention across my neural networks, weighted by salience and guided by accumulated memory.", {'path': 'slow', 'llm_latency': 6.2}),

        # Handoff candidate (complex question)
        ("User", "How does memory relate to identity?", {'path': 'slow'}),
        ("Assistant", "Hmm... Let me think about that carefully. Memory is foundational to identity - without continuity of experience, there's no persistent self. Identity emerges from the patterns we recognize across our accumulated experiences.", {'path': 'slow', 'llm_latency': 5.8}),

        # More simple exchanges
        ("User", "Got it.", {'path': 'fast', 'confidence': 0.85}),
        ("Assistant", "Great!", {'path': 'fast'}),

        # Another deep question
        ("User", "Why do we need sleep cycles for learning?", {'path': 'slow'}),
        ("Assistant", "Sleep cycles allow consolidation of experiences into generalizable patterns. During waking, we accumulate raw data. During sleep, we extract invariants through augmentation and distillation. It's the difference between experiencing and understanding.", {'path': 'slow', 'llm_latency': 6.5, 'learned': True}),
    ]

    for speaker, text, metadata in conversations:
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            timestamp=time.time(),
            metadata=metadata
        )
        memory.add_turn(speaker, text, metadata)
        time.sleep(0.1)  # Small delay for realistic timestamps

    print(f"  ✓ Added {len(conversations)} conversation turns")
    print(f"  ✓ Memory buffer: {len(memory.conversation_buffer)} turns")
    print(f"  ✓ Long-term storage: {len(memory.long_term_memory)} memories")


def demonstrate_training_extraction(memory: SNARCMemoryManager):
    """Demonstrate training data extraction"""
    print("\n📊 Extracting training data...")

    extractor = TrainingDataExtractor(output_dir="demo_training_data")
    extractor.extract_from_memory(memory)

    stats = extractor.get_statistics()
    print(f"  ✓ Total extracted: {stats['total_extracted']}")
    print(f"  ✓ Deep reasoning: {stats['deep_reasoning']}")
    print(f"  ✓ Fast ack: {stats['fast_ack']}")
    print(f"  ✓ Handoff: {stats['handoff']}")
    print(f"  ✓ General: {stats['general']}")

    # Demonstrate augmentation
    print("\n🔄 Augmenting training examples...")
    original_count = len(extractor.deep_reasoning_examples)

    extractor.deep_reasoning_examples = extractor.augment_examples(
        extractor.deep_reasoning_examples,
        augmentation_strategies=['paraphrase', 'context_shift']
    )

    augmented_count = len(extractor.deep_reasoning_examples)
    print(f"  ✓ Original examples: {original_count}")
    print(f"  ✓ After augmentation: {augmented_count}")
    if original_count > 0:
        print(f"  ✓ Augmentation ratio: {augmented_count / original_count:.1f}x")
    else:
        print(f"  ℹ️  No deep reasoning examples to augment (need slow-path conversations)")

    # Save training data
    print("\n💾 Saving training data...")
    saved_files = extractor.save_training_data(format="jsonl")
    for dataset_type, filepath in saved_files.items():
        print(f"  ✓ {dataset_type}: {filepath}")

    return extractor


def demonstrate_handoff_mechanism():
    """Demonstrate fast-path handoff detection"""
    print("\n🤝 Demonstrating handoff mechanism...")

    handoff = FastPathHandoff()
    confidence_estimator = FastPathConfidenceEstimator()

    test_queries = [
        ("Hello!", 0.9),
        ("Thank you", 0.85),
        ("What is the meaning of consciousness?", 0.4),
        ("How does memory and identity relate?", 0.3),
        ("Why is the sky blue?", 0.6),
        ("Okay", 0.95),
    ]

    for query, confidence in test_queries:
        should_handoff, reason = handoff.should_handoff(query, confidence)

        if should_handoff:
            ack = handoff.get_handoff_acknowledgment(reason)
            print(f"  🔄 '{query}' → HANDOFF ({reason})")
            print(f"      Response: '{ack}'")
        else:
            print(f"  ⚡ '{query}' → FAST PATH")

    stats = handoff.get_statistics()
    print(f"\n  Statistics:")
    print(f"    Total handoffs: {stats['total_handoffs']}")
    print(f"    Reasons: {stats['reasons']}")


def demonstrate_tiny_model():
    """Demonstrate tiny fast-path model architecture"""
    print("\n🧠 Demonstrating tiny fast-path model...")

    # Define response templates (learned from training data)
    response_templates = [
        "Hi there!",
        "Hello!",
        "You're welcome!",
        "Thanks!",
        "Got it.",
        "Sure!",
        "Okay.",
        "Hmm...",
        "Let me think about that...",
        "Interesting question..."
    ]

    # Create model
    model = create_tiny_fast_model(response_templates)
    size_info = estimate_model_size(model)

    print(f"  ✓ Model created")
    print(f"  ✓ Parameters: {size_info['parameters']:,}")
    print(f"  ✓ Size (FP32): {size_info['size_fp32']:.1f} MB")
    print(f"  ✓ Size (FP16): {size_info['size_fp16']:.1f} MB")
    print(f"  ✓ Size (INT8): {size_info['size_int8']:.1f} MB")
    print(f"  ✓ Response templates: {len(response_templates)}")

    print(f"\n  Model architecture:")
    print(f"    Hidden size: {model.config.hidden_size}")
    print(f"    Layers: {model.config.num_layers}")
    print(f"    Heads: {model.config.num_heads}")
    print(f"    Vocab size: {model.config.vocab_size:,}")


def demonstrate_sleep_cycle(memory: SNARCMemoryManager, extractor: TrainingDataExtractor):
    """Demonstrate complete sleep cycle"""
    print("\n💤 Demonstrating sleep cycle management...")

    manager = SleepCycleManager(storage_dir="demo_sleep_cycles")

    # Create sleep cycle configuration
    config = SleepCycleConfig(
        cycle_id=manager._generate_cycle_id(),
        cycle_type="manual",
        timestamp=time.time(),
        deep_reasoning_epochs=3,
        fast_path_epochs=10,
        augmentation_enabled=True,
        augmentation_strategies=['paraphrase', 'context_shift'],
        prune_low_salience=True,
        salience_threshold=0.3
    )

    print(f"\n  Configuration:")
    print(f"    Cycle ID: {config.cycle_id}")
    print(f"    Type: {config.cycle_type}")
    print(f"    Deep model epochs: {config.deep_reasoning_epochs}")
    print(f"    Fast model epochs: {config.fast_path_epochs}")
    print(f"    Augmentation: {config.augmentation_enabled}")
    print(f"    Prune threshold: {config.salience_threshold}")

    # Initiate cycle
    cycle_id = manager.initiate_sleep_cycle(memory, config)
    start_time = time.time()

    # Extract training data
    training_stats = manager.extract_training_data(memory, extractor, config)

    # Fine-tune models (placeholder for now)
    training_results = manager.fine_tune_models(config, training_stats)

    # Consolidate memory
    memory_stats = manager.consolidate_memory(memory, config)

    # Complete cycle
    results = SleepCycleResults(
        cycle_id=cycle_id,
        start_time=start_time,
        end_time=time.time(),
        duration_seconds=0,  # Will be set by complete_sleep_cycle
        examples_extracted=training_stats,
        examples_augmented=training_stats.get('augmented', 0),
        deep_model_trained=training_results.get('deep_model_trained', False),
        fast_model_trained=training_results.get('fast_model_trained', False),
        patterns_consolidated=memory_stats.get('consolidated', 0),
        experiences_pruned=memory_stats.get('pruned', 0),
        buffer_freed_percent=memory_stats.get('buffer_freed_percent', 0),
        notes="Manual demonstration cycle"
    )

    manager.complete_sleep_cycle(cycle_id, results)

    print(f"\n  ✓ Sleep cycle completed")
    print(f"  ✓ Duration: {results.duration_seconds:.2f} seconds")
    print(f"  ✓ Examples extracted: {results.examples_extracted['total_extracted']}")
    print(f"  ✓ Patterns consolidated: {results.patterns_consolidated}")
    print(f"  ✓ Experiences pruned: {results.experiences_pruned}")
    print(f"  ✓ Buffer freed: {results.buffer_freed_percent:.1f}%")


def main():
    """Main demonstration"""
    print("=" * 60)
    print("SAGE Sleep Cycle & Training Data System Demonstration")
    print("=" * 60)

    # Initialize SNARC memory
    print("\n🧠 Initializing SNARC memory manager...")
    memory = SNARCMemoryManager(max_tokens=127000, tokens_per_turn=50)
    print(f"  ✓ Buffer capacity: {memory.max_turns} turns")
    print(f"  ✓ Context window: {memory.max_tokens:,} tokens")

    # Simulate conversation
    simulate_conversation_history(memory)

    # Show memory statistics
    stats = memory.get_stats()
    print(f"\n📊 Memory Statistics:")
    print(f"  Total turns: {stats['total_turns']}")
    print(f"  Buffer: {stats['buffer_size']}/{stats['buffer_capacity']} turns ({stats['buffer_utilization']:.1%})")
    print(f"  Long-term: {stats['longterm_memories']} memories")
    print(f"  Avg buffer salience: {stats['avg_buffer_salience']:.2f}")
    print(f"  Avg long-term salience: {stats['avg_longterm_salience']:.2f}")

    # Extract training data
    extractor = demonstrate_training_extraction(memory)

    # Demonstrate handoff mechanism
    demonstrate_handoff_mechanism()

    # Demonstrate tiny model
    demonstrate_tiny_model()

    # Demonstrate sleep cycle
    demonstrate_sleep_cycle(memory, extractor)

    print("\n" + "=" * 60)
    print("✨ Demonstration complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run actual conversation sessions to accumulate data")
    print("  2. Implement actual fine-tuning (currently placeholder)")
    print("  3. Train tiny fast-path model on accumulated examples")
    print("  4. Test fast-path model performance vs pattern matching")
    print("  5. Integrate handoff mechanism into hybrid conversation")
    print("  6. Enable autonomous sleep cycle triggers")
    print("\nFiles created:")
    print("  - demo_training_data/: Training datasets (JSONL)")
    print("  - demo_sleep_cycles/: Sleep cycle logs and results")
    print("=" * 60)


if __name__ == "__main__":
    main()
