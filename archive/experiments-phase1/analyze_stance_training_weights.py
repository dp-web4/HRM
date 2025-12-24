"""
WeightWatcher Analysis of Epistemic Stance Training

Compare baseline vs trained models to understand:
1. What actually changes in the weight matrices?
2. Is 1.6 seconds of training enough?
3. Are the models truly different or just superficially?

Using WeightWatcher to analyze:
- Power law exponents (alpha)
- Spectral norms
- Layer-specific changes
- Over/under-training indicators
"""

import torch
from transformers import AutoModelForCausalLM
import weightwatcher as ww
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_model(model_path, model_name):
    """Analyze a single model with WeightWatcher"""

    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"  # WW works on CPU
    )
    print(f"✓ Loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)\n")

    # Analyze with WeightWatcher
    print("Running WeightWatcher analysis...")
    watcher = ww.WeightWatcher(model=model)

    # Get detailed layer analysis
    details = watcher.analyze(
        plot=False,  # Don't plot yet
        randomize=False,  # We want actual weights, not randomized
        mp_fit=True,  # Fit to Marchenko-Pastur distribution
        min_evals=10  # Lower threshold for small models
    )

    # Get summary metrics
    summary = watcher.get_summary(details)

    print(f"\n✓ Analysis complete\n")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return details, summary


def compare_models(baseline_details, baseline_summary,
                   trained_details, trained_summary,
                   stance_name):
    """Compare baseline vs trained model"""

    print(f"\n{'='*80}")
    print(f"COMPARISON: Baseline vs {stance_name}")
    print(f"{'='*80}\n")

    # Summary metrics comparison
    print("Summary Metrics:")
    print(f"{'Metric':<25} | {'Baseline':<12} | {'Trained':<12} | {'Δ':<12}")
    print("-" * 70)

    for key in ['alpha', 'log_norm', 'log_spectral_norm', 'stable_rank']:
        if key in baseline_summary and key in trained_summary:
            baseline_val = baseline_summary[key]
            trained_val = trained_summary[key]
            delta = trained_val - baseline_val
            delta_pct = (delta / baseline_val * 100) if baseline_val != 0 else 0

            print(f"{key:<25} | {baseline_val:>11.3f} | {trained_val:>11.3f} | {delta:>+6.3f} ({delta_pct:+.1f}%)")

    print()

    # Layer-by-layer comparison
    print("\nLayer-by-Layer Changes:")
    print(f"{'Layer':<50} | {'Baseline α':<12} | {'Trained α':<12} | {'Δα':<12}")
    print("-" * 95)

    # Match layers by layer_id (more reliable than name)
    baseline_layers = {row['layer_id']: row for _, row in baseline_details.iterrows()}
    trained_layers = {row['layer_id']: row for _, row in trained_details.iterrows()}

    significant_changes = []

    for layer_id in baseline_layers.keys():
        if layer_id in trained_layers:
            baseline_row = baseline_layers[layer_id]
            trained_row = trained_layers[layer_id]

            baseline_alpha = baseline_row.get('alpha', None)
            trained_alpha = trained_row.get('alpha', None)

            # Get layer name for display
            layer_display = f"{baseline_row.get('longname', baseline_row.get('name', f'layer_{layer_id}'))}"

            if baseline_alpha is not None and trained_alpha is not None and not pd.isna(baseline_alpha) and not pd.isna(trained_alpha):
                delta = trained_alpha - baseline_alpha
                delta_pct = (delta / baseline_alpha * 100) if baseline_alpha != 0 else 0

                # Show only significant changes (>5%)
                if abs(delta_pct) > 5:
                    print(f"{layer_display[:49]:<50} | {baseline_alpha:>11.3f} | {trained_alpha:>11.3f} | {delta:>+6.3f} ({delta_pct:+.1f}%)")
                    significant_changes.append({
                        'layer_id': layer_id,
                        'layer': layer_display,
                        'baseline_alpha': baseline_alpha,
                        'trained_alpha': trained_alpha,
                        'delta': delta,
                        'delta_pct': delta_pct
                    })

    if not significant_changes:
        print("No significant layer changes (>5%)")

    print(f"\nTotal layers with significant changes: {len(significant_changes)}")

    return {
        'summary_comparison': {
            key: {
                'baseline': baseline_summary.get(key),
                'trained': trained_summary.get(key),
                'delta': trained_summary.get(key, 0) - baseline_summary.get(key, 0)
            }
            for key in ['alpha', 'log_norm', 'log_spectral_norm', 'stable_rank']
            if key in baseline_summary and key in trained_summary
        },
        'significant_layer_changes': significant_changes,
        'num_significant_changes': len(significant_changes)
    }


def main():
    print("🔬 WeightWatcher Analysis: Epistemic Stance Training")
    print("="*80)
    print("\nComparing baseline vs stance-trained models")
    print("Understanding what 1.6 seconds of training actually changes\n")

    # Model to analyze - using qwen2-0.5b as it's the one we have all three stances for
    model_family = "qwen2-0.5b"
    base_model = "Qwen/Qwen2-0.5B"

    model_zoo_path = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances"

    analyses = {}

    # 1. Analyze baseline model
    baseline_details, baseline_summary = analyze_model(base_model, "Baseline (untrained)")
    analyses['baseline'] = {
        'details': baseline_details,
        'summary': baseline_summary
    }

    # 2. Analyze each trained stance
    stances = ['curious-uncertainty', 'confident-expertise']

    for stance in stances:
        trained_path = f"{model_zoo_path}/{model_family}/{stance}"
        trained_details, trained_summary = analyze_model(trained_path, f"{stance} trained")

        analyses[stance] = {
            'details': trained_details,
            'summary': trained_summary
        }

        # Compare to baseline
        comparison = compare_models(
            baseline_details, baseline_summary,
            trained_details, trained_summary,
            stance
        )

        analyses[f'{stance}_comparison'] = comparison

    # Save results
    results_dir = Path("weight_analysis_results")
    results_dir.mkdir(exist_ok=True)

    # Save detailed DataFrames
    for name, data in analyses.items():
        if 'details' in data:
            data['details'].to_csv(f"weight_analysis_results/{model_family}_{name}_details.csv")

    # Save summaries and comparisons as JSON
    json_safe_analyses = {}
    for name, data in analyses.items():
        if 'summary' in data:
            # Convert to serializable format
            json_safe_analyses[name] = {
                'summary': {k: float(v) if not pd.isna(v) else None for k, v in data['summary'].items()}
            }
        elif 'summary_comparison' in data:
            json_safe_analyses[name] = data

    with open("weight_analysis_results/analysis_summary.json", "w") as f:
        json.dump(json_safe_analyses, f, indent=2)

    # Generate report
    report = ["# WeightWatcher Analysis: Epistemic Stance Training\n\n"]
    report.append(f"**Model**: {model_family}\n")
    report.append(f"**Training**: 5 examples, 2 epochs, ~1.6 seconds\n\n")
    report.append("---\n\n")

    report.append("## Summary Metrics\n\n")
    report.append("### Baseline Model\n\n")

    for key, value in baseline_summary.items():
        if not pd.isna(value):
            report.append(f"- **{key}**: {value:.3f}\n")

    report.append("\n### After Training\n\n")

    for stance in stances:
        report.append(f"#### {stance}\n\n")
        if stance in analyses:
            for key, value in analyses[stance]['summary'].items():
                if not pd.isna(value):
                    baseline_val = baseline_summary.get(key)
                    if baseline_val is not None and not pd.isna(baseline_val):
                        delta = value - baseline_val
                        delta_pct = (delta / baseline_val * 100) if baseline_val != 0 else 0
                        report.append(f"- **{key}**: {value:.3f} (Δ {delta:+.3f}, {delta_pct:+.1f}%)\n")
                    else:
                        report.append(f"- **{key}**: {value:.3f}\n")
            report.append("\n")

    report.append("---\n\n")
    report.append("## Significant Layer Changes\n\n")

    for stance in stances:
        comparison_key = f'{stance}_comparison'
        if comparison_key in analyses:
            comp = analyses[comparison_key]
            report.append(f"### {stance}\n\n")
            report.append(f"Layers with >5% change: {comp['num_significant_changes']}\n\n")

            if comp['significant_layer_changes']:
                report.append("| Layer | Baseline α | Trained α | Δα | Δ% |\n")
                report.append("|-------|------------|-----------|-------|-------|\n")

                for change in sorted(comp['significant_layer_changes'],
                                   key=lambda x: abs(x['delta_pct']),
                                   reverse=True)[:10]:  # Top 10
                    report.append(
                        f"| {change['layer'][:30]} | {change['baseline_alpha']:.3f} | "
                        f"{change['trained_alpha']:.3f} | {change['delta']:+.3f} | {change['delta_pct']:+.1f}% |\n"
                    )
                report.append("\n")

    with open("weight_analysis_results/WEIGHT_ANALYSIS_REPORT.md", "w") as f:
        f.writelines(report)

    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80 + "\n")

    print("✓ CSV files: weight_analysis_results/<model>_<stance>_details.csv")
    print("✓ Summary JSON: weight_analysis_results/analysis_summary.json")
    print("✓ Report: weight_analysis_results/WEIGHT_ANALYSIS_REPORT.md\n")


if __name__ == "__main__":
    main()
