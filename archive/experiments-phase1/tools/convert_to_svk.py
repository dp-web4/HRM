"""
Convert Phi Stance Experiment Results to SVK JSONL Format

Takes our saved behavior checkpoints and converts them to the JSONL format
expected by Nova's Stance Vector Kit for analysis.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_behavior_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load a behavior checkpoint JSON file"""
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def extract_turns_from_checkpoint(checkpoint: Any, speaker: str = "model") -> List[Dict]:
    """
    Extract individual turns from checkpoint format

    Checkpoint can be:
    1. List of behavior dicts: [{"prompt": "...", "response": "..."}, ...]
    2. Dict with 'behaviors' key: {"behaviors": {...}}

    SVK format:
    [
        {"speaker": "human", "text": "...", "timestamp": 0},
        {"speaker": "model", "text": "...", "timestamp": 1},
        ...
    ]
    """
    turns = []
    timestamp = 0.0

    # Handle list format (baseline)
    if isinstance(checkpoint, list):
        behaviors = checkpoint
    # Handle dict format with behaviors key
    elif isinstance(checkpoint, dict):
        # Try 'behavior' first (epoch checkpoints), then 'behaviors' (full checkpoints)
        behaviors = checkpoint.get('behavior', checkpoint.get('behaviors', {}))
        # If behaviors is a dict, convert to list
        if isinstance(behaviors, dict):
            behaviors = list(behaviors.values())
    else:
        behaviors = []

    # Extract turns from behaviors
    for i, behavior in enumerate(behaviors):
        prompt = behavior.get('prompt', '')
        response = behavior.get('response', '')

        # Add prompt as human turn
        if prompt:
            turns.append({
                'speaker': 'human',
                'text': prompt,
                'timestamp': timestamp,
                'prompt_idx': i
            })
            timestamp += 1.0

        # Add response as model turn
        if response:
            turns.append({
                'speaker': speaker,
                'text': response,
                'timestamp': timestamp,
                'prompt_idx': i
            })
            timestamp += 1.0

    return turns


def convert_checkpoint_to_jsonl(
    checkpoint_path: Path,
    output_path: Path,
    speaker: str = "model",
    include_prompts: bool = False
):
    """
    Convert a single checkpoint to JSONL

    Args:
        checkpoint_path: Path to behavior checkpoint JSON
        output_path: Path to output JSONL file
        speaker: Speaker ID for model responses
        include_prompts: Whether to include human prompts in output
    """
    checkpoint = load_behavior_checkpoint(checkpoint_path)
    turns = extract_turns_from_checkpoint(checkpoint, speaker=speaker)

    # Filter to only model responses if requested
    if not include_prompts:
        turns = [t for t in turns if t['speaker'] == speaker]

    # Write JSONL
    with open(output_path, 'w') as f:
        for turn in turns:
            f.write(json.dumps(turn) + '\n')

    print(f"Converted {len(turns)} turns to {output_path}")
    return len(turns)


def convert_experiment_results(
    results_dir: Path,
    output_dir: Path,
    epochs: List[int] = None,
    include_baseline: bool = True
):
    """
    Convert all checkpoints from an experiment

    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to write JSONL files
        epochs: List of epoch numbers to convert (None = all)
        include_baseline: Whether to include baseline_behavior.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = {}

    # Convert baseline
    if include_baseline:
        baseline_path = results_dir / 'baseline_behavior.json'
        if baseline_path.exists():
            output_path = output_dir / 'baseline.jsonl'
            n_turns = convert_checkpoint_to_jsonl(baseline_path, output_path)
            converted['baseline'] = {'path': output_path, 'turns': n_turns}

    # Convert epoch checkpoints
    checkpoint_files = sorted(results_dir.glob('epoch_*_checkpoint.json'))

    for checkpoint_path in checkpoint_files:
        # Extract epoch number from filename
        # Format: epoch_60_checkpoint.json
        parts = checkpoint_path.stem.split('_')
        if len(parts) >= 2 and parts[0] == 'epoch':
            try:
                epoch_num = int(parts[1])
            except ValueError:
                continue

            # Filter by epochs if specified
            if epochs is not None and epoch_num not in epochs:
                continue

            output_path = output_dir / f'epoch_{epoch_num}.jsonl'
            n_turns = convert_checkpoint_to_jsonl(checkpoint_path, output_path)
            converted[f'epoch_{epoch_num}'] = {'path': output_path, 'turns': n_turns}

    return converted


def main():
    parser = argparse.ArgumentParser(
        description='Convert Phi stance experiment results to SVK JSONL format'
    )
    parser.add_argument(
        '--results_dir',
        type=Path,
        required=True,
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Directory to write JSONL files'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        help='Specific epochs to convert (default: all)'
    )
    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip baseline conversion'
    )

    args = parser.parse_args()

    print(f"Converting experiment results from: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")

    converted = convert_experiment_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        include_baseline=not args.no_baseline
    )

    print(f"\nConverted {len(converted)} checkpoints:")
    for name, info in converted.items():
        print(f"  {name}: {info['turns']} turns → {info['path']}")


if __name__ == '__main__':
    main()
