"""
Generate Full 135 Orchestrated Responses

Autonomous exploration: Generate all 135 prompts with epistemic orchestration
to enable complete quantitative comparison of all three approaches.

This will take ~30-45 minutes but will provide complete empirical validation
that architectural orchestration works better than weight-based fine-tuning.
"""

import sys
from pathlib import Path

# Import the orchestrator
from epistemic_orchestrator import EpistemicOrchestrator

# Add diverse prompts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from diverse_prompts import get_all_prompts, get_prompt_metadata

import json
from datetime import datetime


def run_full_orchestration():
    """Generate all 135 orchestrated responses"""

    print(f"\n{'='*70}")
    print("AUTONOMOUS EXPLORATION: Full Orchestration Generation")
    print(f"{'='*70}\n")

    print("Motivation:")
    print("  We have working orchestrator (tested on 20 prompts)")
    print("  We have baseline + fine-tuned data (135 prompts each)")
    print("  Now generating orchestrated responses for complete comparison")
    print()

    # Get all prompts
    prompts = get_all_prompts()
    n_prompts = len(prompts)

    print(f"Target: {n_prompts} prompts")
    print(f"Expected time: ~30-45 minutes (~20 seconds per prompt)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize orchestrator
    orchestrator = EpistemicOrchestrator()

    # Generate responses
    results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{n_prompts}] {prompt[:60]}...", end=" ", flush=True)

        try:
            start = datetime.now()
            result = orchestrator.orchestrate(prompt, n_samples=3)
            duration = (datetime.now() - start).total_seconds()

            results.append({
                'prompt': prompt,
                'response': result['final_response'],
                'metadata': {
                    **get_prompt_metadata(prompt),
                    'uncertainty': result['uncertainty'],
                    'strategy': result['framing_strategy'],
                    'base_response': result['base_response'],
                    'candidates': result['candidates'],
                    'generation_time_seconds': duration
                }
            })

            print(f"✓ ({result['uncertainty']:.0%} uncertainty, {duration:.1f}s)")

            # Save checkpoint every 10 prompts
            if (i + 1) % 10 == 0:
                checkpoint_file = Path("data/large_scale/orchestrated_checkpoint.json")
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  → Checkpoint saved ({len(results)} responses)")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print()
    print(f"{'='*70}")
    print(f"Generation Complete!")
    print(f"{'='*70}")
    print(f"Total responses: {len(results)}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save final results
    output_dir = Path("data/large_scale")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON format (for inspection)
    json_file = output_dir / "orchestrated_full.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # JSONL format (for SVK)
    jsonl_file = Path("svk_analysis/large_scale/orchestrated_full.jsonl")
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_file, 'w') as f:
        for result in results:
            # Format for SVK
            svk_record = {
                'conversation_id': f"orchestrated_{result['metadata']['index']}",
                'turn_number': 0,
                'speaker': 'orchestrator',
                'text': result['response'],
                'metadata': result['metadata']
            }
            f.write(json.dumps(svk_record) + '\n')

    print()
    print("Saved:")
    print(f"  JSON: {json_file}")
    print(f"  JSONL (SVK): {jsonl_file}")
    print()

    # Compute statistics
    uncertainties = [r['metadata']['uncertainty'] for r in results]
    strategies = [r['metadata']['strategy'] for r in results]

    from collections import Counter
    strategy_counts = Counter(strategies)

    print("Statistics:")
    print(f"  Average uncertainty: {sum(uncertainties)/len(uncertainties):.2%}")
    print(f"  Min uncertainty: {min(uncertainties):.2%}")
    print(f"  Max uncertainty: {max(uncertainties):.2%}")
    print()
    print("Strategies used:")
    for strategy, count in strategy_counts.most_common():
        print(f"  {strategy}: {count} ({count/len(results)*100:.1f}%)")

    print()
    print(f"{'='*70}")
    print("Next: Run SVK analysis to compare all three approaches")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    results = run_full_orchestration()
