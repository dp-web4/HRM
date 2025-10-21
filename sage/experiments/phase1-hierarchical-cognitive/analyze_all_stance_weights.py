"""
Cross-Model Weight Analysis of Epistemic Stance Training

Analyze all saved stance-trained models to find:
1. Which layers change in each model architecture?
2. Are there universal "stance layers" across architectures?
3. How does the magnitude of change vary by model family?
"""

import torch
from transformers import AutoModelForCausalLM
import weightwatcher as ww
import pandas as pd
import json
from pathlib import Path


# All models we've trained
MODELS = {
    "qwen2-0.5b": {
        "base": "Qwen/Qwen2-0.5B",
        "stances": ["curious-uncertainty", "confident-expertise", "engaged-difficulty"]
    },
    "distilgpt2": {
        "base": "distilgpt2",
        "stances": ["curious-uncertainty", "confident-expertise"]
    },
    "pythia-160m": {
        "base": "EleutherAI/pythia-160m",
        "stances": ["curious-uncertainty", "confident-expertise"]
    }
}


def analyze_model_weights(model_path, model_name):
    """Quick WeightWatcher analysis"""
    print(f"  Analyzing {model_name}...", end='', flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(plot=False, randomize=False, mp_fit=True, min_evals=10)
    summary = watcher.get_summary(details)

    del model
    torch.cuda.empty_cache()

    print(" ✓")
    return details, summary


def find_layer_changes(baseline_details, trained_details, threshold_pct=5.0):
    """Find layers with significant alpha changes"""

    baseline_layers = {row['layer_id']: row for _, row in baseline_details.iterrows()}
    trained_layers = {row['layer_id']: row for _, row in trained_details.iterrows()}

    changes = []

    for layer_id in baseline_layers.keys():
        if layer_id in trained_layers:
            baseline_row = baseline_layers[layer_id]
            trained_row = trained_layers[layer_id]

            baseline_alpha = baseline_row.get('alpha')
            trained_alpha = trained_row.get('alpha')

            if baseline_alpha is not None and trained_alpha is not None:
                if not pd.isna(baseline_alpha) and not pd.isna(trained_alpha):
                    delta = trained_alpha - baseline_alpha
                    delta_pct = (delta / baseline_alpha * 100) if baseline_alpha != 0 else 0

                    if abs(delta_pct) > threshold_pct:
                        changes.append({
                            'layer_id': layer_id,
                            'layer_name': baseline_row.get('longname', baseline_row.get('name', f'layer_{layer_id}')),
                            'layer_type': baseline_row.get('layer_type'),
                            'baseline_alpha': baseline_alpha,
                            'trained_alpha': trained_alpha,
                            'delta': delta,
                            'delta_pct': delta_pct
                        })

    return sorted(changes, key=lambda x: abs(x['delta_pct']), reverse=True)


def main():
    print("🔬 Cross-Model Epistemic Stance Weight Analysis")
    print("="*80)
    print("\nAnalyzing all saved stance-trained models")
    print("Finding which layers change in each architecture\n")

    model_zoo_path = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances"
    results = {}

    # Analyze each model family
    for model_key, model_info in MODELS.items():
        print(f"\n{'='*80}")
        print(f"MODEL FAMILY: {model_key}")
        print(f"{'='*80}\n")

        # Analyze baseline
        print(f"Baseline ({model_info['base']}):")
        baseline_details, baseline_summary = analyze_model_weights(
            model_info['base'],
            "baseline"
        )

        results[model_key] = {
            'baseline': {
                'summary': {k: float(v) if not pd.isna(v) else None
                           for k, v in baseline_summary.items()}
            },
            'stances': {}
        }

        # Analyze each stance
        print(f"\nTrained stances:")
        for stance in model_info['stances']:
            stance_path = f"{model_zoo_path}/{model_key}/{stance}"

            if not Path(stance_path).exists():
                print(f"  ⚠️  {stance} - not found, skipping")
                continue

            trained_details, trained_summary = analyze_model_weights(
                stance_path,
                stance
            )

            # Find significant changes
            changes = find_layer_changes(baseline_details, trained_details)

            results[model_key]['stances'][stance] = {
                'summary': {k: float(v) if not pd.isna(v) else None
                           for k, v in trained_summary.items()},
                'layer_changes': changes,
                'num_changes': len(changes)
            }

        print()

    # Cross-model comparison
    print("\n" + "="*80)
    print("CROSS-MODEL COMPARISON")
    print("="*80 + "\n")

    for model_key, model_data in results.items():
        print(f"\n{model_key}:")
        print("-" * 60)

        for stance, stance_data in model_data['stances'].items():
            changes = stance_data['layer_changes']
            print(f"\n  {stance}: {len(changes)} layers changed >5%")

            if changes:
                print(f"    Top changes:")
                for change in changes[:3]:  # Top 3
                    print(f"      • {change['layer_name'][:40]}: {change['delta_pct']:+.1f}%")

    # Save results
    results_dir = Path("weight_analysis_results")
    results_dir.mkdir(exist_ok=True)

    with open("weight_analysis_results/cross_model_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate comparison report
    report = ["# Cross-Model Epistemic Stance Weight Analysis\n\n"]
    report.append("Analyzing which layers change across different model architectures\n\n")
    report.append("---\n\n")

    report.append("## Summary by Model Family\n\n")

    for model_key, model_data in results.items():
        report.append(f"### {model_key}\n\n")

        report.append("| Stance | Layers Changed (>5%) | Top Layer Change |\n")
        report.append("|--------|----------------------|------------------|\n")

        for stance, stance_data in model_data['stances'].items():
            changes = stance_data['layer_changes']
            if changes:
                top_change = changes[0]
                top_desc = f"{top_change['layer_name'][:30]} ({top_change['delta_pct']:+.1f}%)"
            else:
                top_desc = "None"

            report.append(f"| {stance} | {len(changes)} | {top_desc} |\n")

        report.append("\n")

    report.append("---\n\n")
    report.append("## Layer Changes by Stance\n\n")

    for stance_type in ["curious-uncertainty", "confident-expertise", "engaged-difficulty"]:
        report.append(f"### {stance_type}\n\n")

        for model_key, model_data in results.items():
            if stance_type in model_data['stances']:
                stance_data = model_data['stances'][stance_type]
                changes = stance_data['layer_changes']

                report.append(f"**{model_key}** ({len(changes)} layers):\n\n")

                if changes:
                    for change in changes[:5]:  # Top 5
                        report.append(
                            f"- {change['layer_name']}: "
                            f"{change['baseline_alpha']:.3f} → {change['trained_alpha']:.3f} "
                            f"({change['delta_pct']:+.1f}%)\n"
                        )
                else:
                    report.append("- No significant changes\n")

                report.append("\n")

        report.append("\n")

    report.append("---\n\n")
    report.append("## Observations\n\n")

    # Find common patterns
    report.append("### Pattern Detection\n\n")

    # Check if attention layers are commonly affected
    attention_affected = {}
    mlp_affected = {}

    for model_key, model_data in results.items():
        for stance, stance_data in model_data['stances'].items():
            for change in stance_data['layer_changes']:
                layer_name = change['layer_name']
                if 'attn' in layer_name.lower():
                    key = f"{model_key}_{stance}"
                    if key not in attention_affected:
                        attention_affected[key] = []
                    attention_affected[key].append(change)
                if 'mlp' in layer_name.lower():
                    key = f"{model_key}_{stance}"
                    if key not in mlp_affected:
                        mlp_affected[key] = []
                    mlp_affected[key].append(change)

    report.append(f"**Attention layers affected**: {len(attention_affected)} model×stance combinations\n\n")
    report.append(f"**MLP layers affected**: {len(mlp_affected)} model×stance combinations\n\n")

    report.append("This suggests stance training primarily modifies ")
    if len(attention_affected) > len(mlp_affected):
        report.append("**attention mechanisms**.\n\n")
    elif len(mlp_affected) > len(attention_affected):
        report.append("**MLP/feedforward layers**.\n\n")
    else:
        report.append("both **attention and MLP layers**.\n\n")

    with open("weight_analysis_results/CROSS_MODEL_REPORT.md", "w") as f:
        f.writelines(report)

    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80 + "\n")
    print("✓ Full data: weight_analysis_results/cross_model_analysis.json")
    print("✓ Report: weight_analysis_results/CROSS_MODEL_REPORT.md\n")


if __name__ == "__main__":
    main()
