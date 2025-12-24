#!/usr/bin/env python3
"""
Prepare calibration dataset for Q3-Omni FP4 quantization.

For post-training quantization (PTQ), we need representative samples that cover:
1. Text-only inputs (conversation, reasoning)
2. Multimodal inputs (if available)

The calibration dataset helps the quantizer understand activation ranges.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from transformers import Qwen3OmniMoeProcessor


def create_text_calibration_samples() -> List[Dict[str, str]]:
    """Create diverse text samples for calibration."""

    samples = [
        # Conversation
        {"role": "user", "content": "Hello! How are you today?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
        {"role": "user", "content": "Can you explain quantum entanglement in simple terms?"},

        # Reasoning
        {"role": "user", "content": "If I have 5 apples and give away 2, then buy 3 more, how many do I have?"},
        {"role": "user", "content": "What's the capital of France and what is it famous for?"},

        # Creative writing
        {"role": "user", "content": "Write a short story about a dragon in 2 sentences."},
        {"role": "user", "content": "Describe the color blue to someone who has never seen it."},

        # Technical Q&A
        {"role": "user", "content": "What is the difference between a list and a tuple in Python?"},
        {"role": "user", "content": "Explain how neural networks learn from data."},

        # Multi-turn context
        {"role": "user", "content": "I'm planning a trip to Japan."},
        {"role": "assistant", "content": "That sounds exciting! What cities are you planning to visit?"},
        {"role": "user", "content": "Tokyo and Kyoto. What should I see there?"},

        # Long-form reasoning
        {"role": "user", "content": "Compare and contrast renewable and non-renewable energy sources."},
        {"role": "user", "content": "What are the ethical implications of artificial intelligence in healthcare?"},

        # Instruction following
        {"role": "user", "content": "List the steps to bake a chocolate cake."},
        {"role": "user", "content": "Translate this to Spanish: The weather is beautiful today."},

        # Factual knowledge
        {"role": "user", "content": "When was the Declaration of Independence signed?"},
        {"role": "user", "content": "What is the speed of light in vacuum?"},

        # Mathematical reasoning
        {"role": "user", "content": "Solve for x: 2x + 5 = 15"},
        {"role": "user", "content": "What is 15% of 200?"},

        # Summarization
        {"role": "user", "content": "Summarize the main causes of World War I in 3 points."},

        # Common sense reasoning
        {"role": "user", "content": "If it's raining outside, should I bring an umbrella?"},
        {"role": "user", "content": "Why do birds fly south for the winter?"},

        # Edge cases
        {"role": "user", "content": ""},  # Empty input
        {"role": "user", "content": "a" * 500},  # Very long input
        {"role": "user", "content": "🎉🎊🎈 Hello! 你好！Bonjour! 👋"},  # Unicode/emoji
    ]

    return samples


def prepare_calibration_dataset(
    output_dir: Path,
    num_samples: int = 128,
    model_path: str = "model-zoo/sage/omni-modal/qwen3-omni-30b"
):
    """
    Prepare calibration dataset for FP4 quantization.

    Args:
        output_dir: Directory to save calibration data
        num_samples: Number of samples to generate
        model_path: Path to Q3-Omni model
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing calibration dataset with {num_samples} samples...")

    # Load processor for tokenization
    print("Loading processor...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    # Create text samples
    text_samples = create_text_calibration_samples()

    # Expand to desired number of samples by cycling
    calibration_samples = []
    for i in range(num_samples):
        sample = text_samples[i % len(text_samples)]
        calibration_samples.append(sample)

    # Format as conversation history
    conversations = []
    current_conversation = []

    for sample in calibration_samples:
        current_conversation.append(sample)

        # Create a conversation every 2-5 messages
        if len(current_conversation) >= 2 and (
            sample["role"] == "assistant" or len(current_conversation) >= 5
        ):
            conversations.append(current_conversation.copy())
            current_conversation = []

    # Add remaining messages as final conversation
    if current_conversation:
        conversations.append(current_conversation)

    print(f"Created {len(conversations)} conversations")

    # Process and save
    processed_samples = []

    for idx, conversation in enumerate(conversations):
        try:
            # Apply chat template
            text = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )

            processed_samples.append({
                "conversation": conversation,
                "formatted_text": text,
                "input_ids": inputs["input_ids"].tolist(),
                "attention_mask": inputs["attention_mask"].tolist(),
            })

            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(conversations)} conversations")

        except Exception as e:
            print(f"Warning: Failed to process conversation {idx}: {e}")
            continue

    # Save calibration dataset
    calibration_file = output_dir / "calibration_dataset.json"
    with open(calibration_file, 'w') as f:
        json.dump(processed_samples, f, indent=2)

    print(f"✅ Saved {len(processed_samples)} calibration samples to {calibration_file}")

    # Save metadata
    metadata = {
        "num_samples": len(processed_samples),
        "num_conversations": len(conversations),
        "model_path": model_path,
        "processor_config": {
            "max_length": 2048,
            "padding": True,
            "truncation": True,
        }
    }

    metadata_file = output_dir / "calibration_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata to {metadata_file}")

    # Create a simple loader script for quantization
    loader_script = output_dir / "load_calibration_data.py"
    with open(loader_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""Load calibration data for quantization."""

import json
import torch
from pathlib import Path

def load_calibration_data(calibration_dir="./"):
    """Load calibration dataset for quantization."""
    calibration_dir = Path(calibration_dir)

    with open(calibration_dir / "calibration_dataset.json") as f:
        data = json.load(f)

    # Convert back to tensors
    for sample in data:
        sample["input_ids"] = torch.tensor(sample["input_ids"])
        sample["attention_mask"] = torch.tensor(sample["attention_mask"])

    return data

def get_calibration_loader(calibration_dir="./", batch_size=1):
    """Get data loader for calibration."""
    data = load_calibration_data(calibration_dir)

    # Simple batch iterator
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        # Stack batch
        input_ids = torch.cat([s["input_ids"] for s in batch], dim=0)
        attention_mask = torch.cat([s["attention_mask"] for s in batch], dim=0)

        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

if __name__ == "__main__":
    # Test loading
    data = load_calibration_data()
    print(f"Loaded {len(data)} calibration samples")

    # Test loader
    loader = get_calibration_loader(batch_size=4)
    batch = next(loader)
    print(f"Batch shape: {batch['input_ids'].shape}")
''')

    print(f"✅ Created loader script: {loader_script}")

    # Print summary
    print("\n" + "="*60)
    print("CALIBRATION DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(processed_samples)}")
    print(f"Conversations: {len(conversations)}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - calibration_dataset.json ({len(processed_samples)} samples)")
    print(f"  - calibration_metadata.json")
    print(f"  - load_calibration_data.py (loader script)")
    print("="*60)

    return processed_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare calibration dataset for Q3-Omni FP4 quantization")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sage/quantization/calibration_data",
        help="Output directory for calibration data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model-zoo/sage/omni-modal/qwen3-omni-30b",
        help="Path to Q3-Omni model"
    )

    args = parser.parse_args()

    prepare_calibration_dataset(
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        model_path=args.model_path
    )
