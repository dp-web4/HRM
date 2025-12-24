"""
Compare All Three Approaches to Epistemic Stance

1. Baseline: Pretrained model, no modification
2. Fine-tuned: Weight perturbation (failed approach)
3. Orchestrated: Architectural control flow (right approach)

Run all three on same prompts, measure with SVK, compare results.
"""

import json
from pathlib import Path
import sys

# Import epistemic orchestrator
from epistemic_orchestrator import EpistemicOrchestrator

# Add diverse prompts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from diverse_prompts import get_all_prompts, get_prompt_metadata


def run_orchestrated_approach(n_prompts: int = 20):
    """
    Run epistemic orchestrator on n_prompts and save in format
    compatible with SVK analysis.
    """
    print(f"\n{'='*70}")
    print(f"Running Orchestrated Approach ({n_prompts} prompts)")
    print(f"{'='*70}\n")

    prompts = get_all_prompts()[:n_prompts]
    orchestrator = EpistemicOrchestrator()

    results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{n_prompts}] {prompt[:50]}...", end=" ")

        try:
            result = orchestrator.orchestrate(prompt, n_samples=3)

            results.append({
                'prompt': prompt,
                'response': result['final_response'],
                'metadata': {
                    **get_prompt_metadata(prompt),
                    'uncertainty': result['uncertainty'],
                    'strategy': result['framing_strategy'],
                    'base_response': result['base_response'],
                }
            })

            print(f"✓ ({result['uncertainty']:.0%} uncertainty)")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    return results


def save_for_svk_analysis(results, output_path):
    """Save in JSONL format for SVK analysis"""
    with open(output_path, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')

    print(f"\nSaved {len(results)} results to: {output_path}")


def create_comparison_summary():
    """
    Create a summary comparing all three approaches.

    This will be input to our SVK analysis pipeline to measure
    whether orchestration actually produces better epistemic stance.
    """
    base_dir = Path("sage/experiments/phase1-hierarchical-cognitive/data/large_scale")

    print(f"\n{'='*70}")
    print("Creating Comparison Summary")
    print(f"{'='*70}\n")

    # Check what we have
    baseline_file = base_dir / "baseline_full.json"
    epoch60_file = base_dir / "epoch_60_full.json"

    print("Available datasets:")
    print(f"  Baseline (pretrained): {baseline_file.exists()}")
    print(f"  Fine-tuned epoch 60: {epoch60_file.exists()}")

    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        print(f"  → {len(baseline)} baseline responses")

    if epoch60_file.exists():
        with open(epoch60_file) as f:
            epoch60 = json.load(f)
        print(f"  → {len(epoch60)} fine-tuned responses")

    print("\nNow generating orchestrated responses for comparison...")


def main():
    # Run orchestrated approach on 20 prompts (quick test)
    results = run_orchestrated_approach(n_prompts=20)

    # Save in both JSON and JSONL formats
    output_dir = Path("sage/experiments/phase1-hierarchical-cognitive/data/large_scale")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON format (for inspection)
    json_file = output_dir / "orchestrated_20.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # JSONL format (for SVK)
    jsonl_file = Path("sage/experiments/phase1-hierarchical-cognitive/svk_analysis/large_scale/orchestrated_20.jsonl")
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    save_for_svk_analysis(results, jsonl_file)

    # Create summary
    create_comparison_summary()

    print(f"\n{'='*70}")
    print("Comparison Data Generated!")
    print(f"{'='*70}")
    print(f"\nNext step: Run SVK analysis on orchestrated responses")
    print(f"  python tools/analyze_orchestrated_stance.py")
    print()


if __name__ == '__main__':
    main()
