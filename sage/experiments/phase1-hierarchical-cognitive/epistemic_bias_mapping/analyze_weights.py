#!/usr/bin/env python3
"""
WeightWatcher Analysis of Epistemic Stance Training

Analyzes weight changes across training to identify which layers
were modified during epistemic stance fine-tuning.

Focus: Layers 13 and 15 (Nova's hypothesis)
"""

import json
import torch
import weightwatcher as ww
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM
import pandas as pd


def analyze_model(model_path: str, model_name: str) -> dict:
    """Analyze a model using WeightWatcher"""
    print(f"\nAnalyzing {model_name}...")
    print(f"Loading from: {model_path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"  # CPU for weight analysis
    )

    # Run WeightWatcher analysis
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze()

    # Extract key metrics per layer
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "summary": watcher.get_summary(),
        "details": details.to_dict('records') if hasattr(details, 'to_dict') else details
    }

    # Clear memory
    del model
    del watcher
    torch.cuda.empty_cache()

    print(f"✓ Analysis complete for {model_name}")
    return results


def compare_layers(base_results: dict, checkpoint_results: dict, layer_focus: list = [13, 15]) -> dict:
    """Compare specific layers between base and checkpoint"""
    base_df = pd.DataFrame(base_results["details"])
    ckpt_df = pd.DataFrame(checkpoint_results["details"])

    comparison = {
        "checkpoint": checkpoint_results["model_name"],
        "layer_changes": []
    }

    # Look for attention and MLP layers in the specified layers
    for layer_num in layer_focus:
        # Find rows matching this layer
        base_layer = base_df[base_df['layer_name'].str.contains(f'layers.{layer_num}.', na=False)]
        ckpt_layer = ckpt_df[ckpt_df['layer_name'].str.contains(f'layers.{layer_num}.', na=False)]

        if len(base_layer) > 0 and len(ckpt_layer) > 0:
            for _, base_row in base_layer.iterrows():
                layer_name = base_row['layer_name']
                ckpt_row = ckpt_layer[ckpt_layer['layer_name'] == layer_name]

                if len(ckpt_row) > 0:
                    ckpt_row = ckpt_row.iloc[0]

                    # Calculate changes in key metrics
                    alpha_change = ckpt_row.get('alpha', 0) - base_row.get('alpha', 0)
                    spectral_norm_change = ckpt_row.get('spectral_norm', 0) - base_row.get('spectral_norm', 0)

                    comparison["layer_changes"].append({
                        "layer_name": layer_name,
                        "layer_num": layer_num,
                        "base_alpha": float(base_row.get('alpha', 0)),
                        "ckpt_alpha": float(ckpt_row.get('alpha', 0)),
                        "alpha_change": float(alpha_change),
                        "base_spectral_norm": float(base_row.get('spectral_norm', 0)),
                        "ckpt_spectral_norm": float(ckpt_row.get('spectral_norm', 0)),
                        "spectral_norm_change": float(spectral_norm_change)
                    })

    return comparison


def main():
    output_dir = Path("weightwatcher_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models to analyze
    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", "base_model"),
        ("fine_tuned_model/checkpoints/checkpoint-010", "checkpoint_010"),
        ("fine_tuned_model/checkpoints/checkpoint-050", "checkpoint_050"),
        ("fine_tuned_model/checkpoints/checkpoint-100", "checkpoint_100"),
        ("fine_tuned_model/checkpoints/checkpoint-200", "checkpoint_200"),
    ]

    print("=" * 80)
    print("WeightWatcher Analysis: Epistemic Stance Training")
    print("=" * 80)
    print(f"\nAnalyzing {len(models)} models...")
    print("Focus: Layers 13 and 15 (Nova's hypothesis)\n")

    all_results = {}

    # Analyze each model
    for model_path, model_name in models:
        try:
            results = analyze_model(model_path, model_name)
            all_results[model_name] = results

            # Save individual results
            result_file = output_dir / f"{model_name}_ww.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved to: {result_file}")

        except Exception as e:
            print(f"  Error analyzing {model_name}: {e}")
            continue

    # Compare checkpoints to base
    if "base_model" in all_results:
        print("\n" + "=" * 80)
        print("Layer-by-Layer Comparison (Base vs Checkpoints)")
        print("=" * 80)

        comparisons = {}
        for ckpt_name in ["checkpoint_010", "checkpoint_050", "checkpoint_100", "checkpoint_200"]:
            if ckpt_name in all_results:
                comparison = compare_layers(
                    all_results["base_model"],
                    all_results[ckpt_name],
                    layer_focus=[13, 15]
                )
                comparisons[ckpt_name] = comparison

                print(f"\n{ckpt_name}:")
                for change in comparison["layer_changes"]:
                    print(f"  {change['layer_name']}:")
                    print(f"    Alpha: {change['base_alpha']:.4f} → {change['ckpt_alpha']:.4f} (Δ{change['alpha_change']:+.4f})")
                    print(f"    Spectral Norm: {change['base_spectral_norm']:.4f} → {change['ckpt_spectral_norm']:.4f} (Δ{change['spectral_norm_change']:+.4f})")

        # Save comparisons
        comparison_file = output_dir / "layer_comparisons.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparisons, f, indent=2)
        print(f"\n✓ Comparisons saved to: {comparison_file}")

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_dir / f"all_results_{timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✅ Analysis complete!")
    print(f"Combined results: {combined_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
