#!/usr/bin/env python3
"""
WeightWatcher Comparison: Original Qwen vs Phase 1 vs Phase 2.1

Analyzes weight matrices to understand training effects:
- Alpha (power law exponent) - lower = more trained/refined
- Spectral norm - gradient flow and stability
- Log norm - weight magnitude distribution
"""

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import weightwatcher as ww
import json
from pathlib import Path
import pandas as pd


def analyze_model(model_name, model_path=None, is_peft=False, base_model="Qwen/Qwen2.5-0.5B-Instruct"):
    """Run weightwatcher analysis on a model."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}")

    if is_peft:
        # Load PEFT model with adapter
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="cpu"  # CPU for weightwatcher analysis
        )
        print(f"Loading adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        # Merge adapter weights for full analysis
        model = model.merge_and_unload()
    else:
        # Load full model
        print(f"Loading model: {model_path or model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path or model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )

    print("Running weightwatcher analysis...")
    watcher = ww.WeightWatcher(model=model)

    # Analyze with multiple metrics
    details = watcher.analyze(
        layers=[],  # All layers
        min_evals=0,
        max_evals=None,
        randomize=False,
        mp_fit=True,  # Marchenko-Pastur fit
        svd_method='accurate'  # Most accurate but slower
    )

    # Get summary statistics
    summary = watcher.get_summary()

    print(f"\n{'='*80}")
    print(f"Summary for {model_name}")
    print(f"{'='*80}")

    # Key metrics
    print(f"\nGlobal Metrics:")
    print(f"  Mean Alpha: {summary.get('alpha', 'N/A'):.4f} (power law exponent)")
    print(f"  Mean Log Norm: {summary.get('log_norm', 'N/A'):.4f} (weight magnitude)")
    print(f"  Mean Spectral Norm: {summary.get('log_spectral_norm', 'N/A'):.4f} (gradient stability)")

    if 'alpha_weighted' in summary:
        print(f"  Weighted Alpha: {summary['alpha_weighted']:.4f} (size-weighted)")

    # Layer-wise statistics
    if details is not None and len(details) > 0:
        df = pd.DataFrame(details)

        # Focus on attention and MLP layers
        if 'layer_id' in df.columns:
            print(f"\n  Total layers analyzed: {len(df)}")

            # Attention layers
            attn_layers = df[df['layer_id'].str.contains('attn|attention', case=False, na=False)]
            if not attn_layers.empty:
                print(f"\n  Attention Layers:")
                print(f"    Mean Alpha: {attn_layers['alpha'].mean():.4f}")
                print(f"    Mean Log Norm: {attn_layers['log_norm'].mean():.4f}")

            # MLP layers
            mlp_layers = df[df['layer_id'].str.contains('mlp|fc', case=False, na=False)]
            if not mlp_layers.empty:
                print(f"\n  MLP Layers:")
                print(f"    Mean Alpha: {mlp_layers['alpha'].mean():.4f}")
                print(f"    Mean Log Norm: {mlp_layers['log_norm'].mean():.4f}")

    return summary, details


def compare_models():
    """Compare original Qwen, Phase 1, and Phase 2.1 models."""
    print("="*80)
    print("WeightWatcher Model Comparison")
    print("="*80)
    print("\nThis will analyze three models:")
    print("  1. Original Qwen/Qwen2.5-0.5B-Instruct (baseline)")
    print("  2. Phase 1: Trained on 40 epistemic stance examples")
    print("  3. Phase 2.1: Trained on 115 genuine introspection examples")
    print()
    print("Analyzing weight matrices for:")
    print("  - Alpha (power law exponent) - training refinement")
    print("  - Spectral norm - gradient flow stability")
    print("  - Log norm - weight magnitude distribution")
    print()

    results = {}

    # 1. Original Qwen
    try:
        summary_orig, details_orig = analyze_model(
            "Original Qwen",
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            is_peft=False
        )
        results['original'] = summary_orig
    except Exception as e:
        print(f"Error analyzing original Qwen: {e}")
        results['original'] = None

    # 2. Phase 1
    try:
        summary_p1, details_p1 = analyze_model(
            "Phase 1 (epistemic-pragmatism)",
            model_path="./fine_tuned_model/final_model",
            is_peft=True
        )
        results['phase1'] = summary_p1
    except Exception as e:
        print(f"Error analyzing Phase 1: {e}")
        results['phase1'] = None

    # 3. Phase 2.1
    try:
        summary_p2, details_p2 = analyze_model(
            "Phase 2.1 (genuine introspection)",
            model_path="./phase2.1_sft_model/final_model",
            is_peft=True
        )
        results['phase2.1'] = summary_p2
    except Exception as e:
        print(f"Error analyzing Phase 2.1: {e}")
        results['phase2.1'] = None

    # Comparative analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")

    if all(results.values()):
        print("\nAlpha (Power Law Exponent) - Lower = More Refined")
        print(f"  Original:  {results['original'].get('alpha', 0):.4f}")
        print(f"  Phase 1:   {results['phase1'].get('alpha', 0):.4f}")
        print(f"  Phase 2.1: {results['phase2.1'].get('alpha', 0):.4f}")

        # Calculate deltas
        delta_p1 = results['phase1'].get('alpha', 0) - results['original'].get('alpha', 0)
        delta_p2 = results['phase2.1'].get('alpha', 0) - results['phase1'].get('alpha', 0)

        print(f"\n  Δ Original → Phase 1:   {delta_p1:+.4f}")
        print(f"  Δ Phase 1 → Phase 2.1:  {delta_p2:+.4f}")

        print("\nLog Norm (Weight Magnitude)")
        print(f"  Original:  {results['original'].get('log_norm', 0):.4f}")
        print(f"  Phase 1:   {results['phase1'].get('log_norm', 0):.4f}")
        print(f"  Phase 2.1: {results['phase2.1'].get('log_norm', 0):.4f}")

        print("\nSpectral Norm (Gradient Stability)")
        print(f"  Original:  {results['original'].get('log_spectral_norm', 0):.4f}")
        print(f"  Phase 1:   {results['phase1'].get('log_spectral_norm', 0):.4f}")
        print(f"  Phase 2.1: {results['phase2.1'].get('log_spectral_norm', 0):.4f}")

    # Save results
    output_path = Path("weightwatcher_comparison.json")
    with open(output_path, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for k, v in results.items():
            if v is not None:
                serializable_results[k] = {key: float(val) if isinstance(val, (int, float)) else str(val)
                                          for key, val in v.items()}
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    # Interpretation guide
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("\nAlpha (Power Law Exponent):")
    print("  - Lower alpha = more trained/refined weights")
    print("  - Heavy-tailed distributions indicate generalization")
    print("  - Too low (<2.0) may indicate overtraining")
    print("\nLog Norm:")
    print("  - Measures overall weight magnitude")
    print("  - Changes indicate how much weights shifted during training")
    print("\nSpectral Norm:")
    print("  - Indicates gradient flow and stability")
    print("  - Higher = more stable but potentially less flexible")
    print("  - Lower = more flexible but potentially unstable")
    print("\nWhat to look for:")
    print("  ✓ Phase 1 should show refinement vs Original (lower alpha)")
    print("  ✓ Phase 2.1 should show further refinement vs Phase 1")
    print("  ✓ But not too much change (avoid overfitting)")
    print("  ✓ Stable spectral norms (no gradient explosion)")
    print("="*80)

    return results


if __name__ == "__main__":
    import sys

    print("\n" + "="*80)
    print("WeightWatcher Model Comparison")
    print("="*80)
    print("\nWARNING: This analysis requires significant memory and time.")
    print("Expected duration: 10-30 minutes per model")
    print("Memory requirement: ~8GB RAM")
    print()

    response = input("Continue with analysis? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Analysis cancelled.")
        sys.exit(0)

    results = compare_models()

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review weightwatcher_comparison.json for detailed metrics")
    print("2. Compare alpha values to assess training refinement")
    print("3. Check spectral norms for training stability")
    print("4. If Phase 2.1 shows good refinement without overtraining:")
    print("   → Proceed with SAGE-IRP deployment")
    print("   → Test on Jetson")
    print("="*80)
