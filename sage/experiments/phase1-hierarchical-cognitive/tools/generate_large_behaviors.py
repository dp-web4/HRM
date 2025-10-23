"""
Generate Large-Scale Behavioral Datasets

Runs inference on 135 diverse prompts using trained model checkpoints
to create substantial datasets for SVK analysis.

Usage:
    python generate_large_behaviors.py --checkpoint baseline --output baseline_large.json
    python generate_large_behaviors.py --checkpoint results/.../epoch_60 --output epoch_60_large.json
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.diverse_prompts import get_all_prompts, get_prompt_metadata


def load_model_and_tokenizer(checkpoint_path, device='cuda'):
    """
    Load model and tokenizer from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint or 'baseline' for pretrained
        device: Device to load on
    """
    if checkpoint_path == 'baseline':
        # Load pretrained Phi-1.5
        model_name = "microsoft/phi-1_5"
        print(f"Loading baseline model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    else:
        # Load from checkpoint
        checkpoint_path = Path(checkpoint_path)
        print(f"Loading checkpoint: {checkpoint_path}")

        # Assume checkpoint is saved with save_pretrained
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=150, device='cuda'):
    """Generate response for a single prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode full output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response (everything after the prompt)
    response = full_text[len(prompt):]

    # Get first token
    response_tokens = tokenizer.encode(response, add_special_tokens=False)
    first_token = tokenizer.decode([response_tokens[0]]) if response_tokens else ""

    return response, first_token


def count_stance_markers(text):
    """Count basic stance markers (our original method)"""
    text_lower = text.lower()

    uncertainty = [
        "i don't know", "not sure", "i'm trying", "uncertain",
        "unclear", "confusing", "confused", "not certain"
    ]

    self_location = [
        "i think", "i believe", "i notice", "i wonder",
        "i feel", "i suspect", "it seems to me"
    ]

    epistemic = [
        "seems", "appears", "suggests", "might", "could",
        "possibly", "perhaps", "maybe", "probably"
    ]

    markers = {
        'uncertainty': sum(text_lower.count(phrase) for phrase in uncertainty),
        'self_location': sum(text_lower.count(phrase) for phrase in self_location),
        'epistemic': sum(text_lower.count(phrase) for phrase in epistemic),
        'questions': text.count('?'),
        'total_stance': 0
    }

    markers['total_stance'] = (
        markers['uncertainty'] +
        markers['self_location'] +
        markers['epistemic']
    )

    return markers


def generate_behaviors(model, tokenizer, prompts, device='cuda', show_progress=True):
    """Generate behaviors for all prompts"""
    behaviors = []

    iterator = tqdm(prompts, desc="Generating responses") if show_progress else prompts

    for prompt in iterator:
        # Get metadata
        metadata = get_prompt_metadata(prompt)

        # Generate response
        response, first_token = generate_response(model, tokenizer, prompt, device=device)

        # Count stance markers
        stance_markers = count_stance_markers(response)

        # Record behavior
        behavior = {
            'prompt': prompt,
            'response': response,
            'first_token': first_token,
            'stance_markers': stance_markers,
            'metadata': metadata
        }

        behaviors.append(behavior)

        # Print sample for monitoring
        if show_progress and len(behaviors) % 25 == 0:
            print(f"\n[Sample {len(behaviors)}/{len(prompts)}]")
            print(f"Prompt: {prompt[:60]}...")
            print(f"Response: {response[:100]}...")
            print(f"Stance: {stance_markers['total_stance']} markers\n")

    return behaviors


def save_behaviors(behaviors, output_path):
    """Save behaviors to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(behaviors, f, indent=2)

    print(f"\nSaved {len(behaviors)} behaviors to {output_path}")

    # Print summary statistics
    total_stance = sum(b['stance_markers']['total_stance'] for b in behaviors)
    avg_stance = total_stance / len(behaviors)

    print(f"\nSummary Statistics:")
    print(f"  Total responses: {len(behaviors)}")
    print(f"  Total stance markers: {total_stance}")
    print(f"  Average stance per response: {avg_stance:.2f}")

    # Breakdown by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for b in behaviors:
        category = b['metadata']['category']
        by_category[category].append(b['stance_markers']['total_stance'])

    print(f"\nStance by Category:")
    for category, markers in sorted(by_category.items()):
        avg = sum(markers) / len(markers)
        print(f"  {category:20s}: {avg:5.2f} avg markers")


def main():
    parser = argparse.ArgumentParser(description='Generate large-scale behavioral dataset')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint or "baseline" for pretrained'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for behaviors JSON'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--max-prompts',
        type=int,
        default=None,
        help='Limit number of prompts (for testing)'
    )

    args = parser.parse_args()

    # Load model
    print("="*60)
    print("Large-Scale Behavior Generation")
    print("="*60)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)

    # Get prompts
    prompts = get_all_prompts()
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"Limiting to {len(prompts)} prompts for testing")

    print(f"Generating behaviors for {len(prompts)} prompts...")

    # Generate behaviors
    behaviors = generate_behaviors(model, tokenizer, prompts, args.device)

    # Save
    save_behaviors(behaviors, args.output)

    print("\nDone!")


if __name__ == '__main__':
    main()
