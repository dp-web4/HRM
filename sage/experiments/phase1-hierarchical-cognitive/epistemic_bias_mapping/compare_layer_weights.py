#!/usr/bin/env python3
"""
Compare Layer Weights: Focus on Layers 13 and 15

Analyzes the WeightWatcher results to see which layers changed
during epistemic stance training.
"""

import json
import pandas as pd
from pathlib import Path


def load_ww_results(filepath):
    """Load WeightWatcher results"""
    with open(filepath) as f:
        data = json.load(f)
    return pd.DataFrame(data['details'])


def extract_layer_metrics(df, layer_nums=[13, 15]):
    """Extract metrics for specific layers"""
    results = {}
    for layer_num in layer_nums:
        layer_data = df[df['longname'].str.contains(f'layers.{layer_num}.', na=False)]
        if len(layer_data) > 0:
            results[f'layer_{layer_num}'] = layer_data
    return results


def compare_checkpoints():
    """Compare all checkpoints to base model"""
    ww_dir = Path("weightwatcher_analysis")

    # Load all results
    print("=" * 80)
    print("WeightWatcher Layer Comparison: Layers 13 and 15")
    print("=" * 80)

    base_df = load_ww_results(ww_dir / "base_model_ww.json")
    checkpoints = {
        "checkpoint_010": load_ww_results(ww_dir / "checkpoint_010_ww.json"),
        "checkpoint_050": load_ww_results(ww_dir / "checkpoint_050_ww.json"),
        "checkpoint_100": load_ww_results(ww_dir / "checkpoint_100_ww.json"),
        "checkpoint_200": load_ww_results(ww_dir / "checkpoint_200_ww.json"),
    }

    # Extract layer 13 and 15 data
    base_layers = extract_layer_metrics(base_df)

    print("\n" + "=" * 80)
    print("LAYER 13 ANALYSIS")
    print("=" * 80)

    if 'layer_13' in base_layers:
        layer_13_base = base_layers['layer_13']
        print(f"\nBase Model - Layer 13 ({len(layer_13_base)} components):")
        print(layer_13_base[['longname', 'alpha', 'spectral_norm', 'log_norm', 'warning']].to_string(index=False))

        for ckpt_name, ckpt_df in checkpoints.items():
            ckpt_layers = extract_layer_metrics(ckpt_df)
            if 'layer_13' in ckpt_layers:
                layer_13_ckpt = ckpt_layers['layer_13']
                print(f"\n{ckpt_name} - Layer 13:")

                # Compare each component
                for _, base_row in layer_13_base.iterrows():
                    longname = base_row['longname']
                    ckpt_row = layer_13_ckpt[layer_13_ckpt['longname'] == longname]

                    if len(ckpt_row) > 0:
                        ckpt_row = ckpt_row.iloc[0]
                        alpha_change = ckpt_row['alpha'] - base_row['alpha']
                        spectral_change = ckpt_row['spectral_norm'] - base_row['spectral_norm']
                        norm_change = ckpt_row['log_norm'] - base_row['log_norm']

                        print(f"\n  {longname}:")
                        print(f"    Alpha:         {base_row['alpha']:.4f} → {ckpt_row['alpha']:.4f} (Δ{alpha_change:+.4f})")
                        print(f"    Spectral Norm: {base_row['spectral_norm']:.2f} → {ckpt_row['spectral_norm']:.2f} (Δ{spectral_change:+.2f})")
                        print(f"    Log Norm:      {base_row['log_norm']:.4f} → {ckpt_row['log_norm']:.4f} (Δ{norm_change:+.4f})")
                        if base_row['warning'] != ckpt_row['warning']:
                            print(f"    Warning:       {base_row['warning']} → {ckpt_row['warning']}")

    print("\n" + "=" * 80)
    print("LAYER 15 ANALYSIS")
    print("=" * 80)

    if 'layer_15' in base_layers:
        layer_15_base = base_layers['layer_15']
        print(f"\nBase Model - Layer 15 ({len(layer_15_base)} components):")
        print(layer_15_base[['longname', 'alpha', 'spectral_norm', 'log_norm', 'warning']].to_string(index=False))

        for ckpt_name, ckpt_df in checkpoints.items():
            ckpt_layers = extract_layer_metrics(ckpt_df)
            if 'layer_15' in ckpt_layers:
                layer_15_ckpt = ckpt_layers['layer_15']
                print(f"\n{ckpt_name} - Layer 15:")

                # Compare each component
                for _, base_row in layer_15_base.iterrows():
                    longname = base_row['longname']
                    ckpt_row = layer_15_ckpt[layer_15_ckpt['longname'] == longname]

                    if len(ckpt_row) > 0:
                        ckpt_row = ckpt_row.iloc[0]
                        alpha_change = ckpt_row['alpha'] - base_row['alpha']
                        spectral_change = ckpt_row['spectral_norm'] - base_row['spectral_norm']
                        norm_change = ckpt_row['log_norm'] - base_row['log_norm']

                        print(f"\n  {longname}:")
                        print(f"    Alpha:         {base_row['alpha']:.4f} → {ckpt_row['alpha']:.4f} (Δ{alpha_change:+.4f})")
                        print(f"    Spectral Norm: {base_row['spectral_norm']:.2f} → {ckpt_row['spectral_norm']:.2f} (Δ{spectral_change:+.2f})")
                        print(f"    Log Norm:      {base_row['log_norm']:.4f} → {ckpt_row['log_norm']:.4f} (Δ{norm_change:+.4f})")
                        if base_row['warning'] != ckpt_row['warning']:
                            print(f"    Warning:       {base_row['warning']} → {ckpt_row['warning']}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY: Largest Changes")
    print("=" * 80)

    all_changes = []
    for layer_num in [13, 15]:
        if f'layer_{layer_num}' in base_layers:
            layer_base = base_layers[f'layer_{layer_num}']
            for ckpt_name, ckpt_df in checkpoints.items():
                ckpt_layers = extract_layer_metrics(ckpt_df)
                if f'layer_{layer_num}' in ckpt_layers:
                    layer_ckpt = ckpt_layers[f'layer_{layer_num}']
                    for _, base_row in layer_base.iterrows():
                        longname = base_row['longname']
                        ckpt_row = layer_ckpt[layer_ckpt['longname'] == longname]
                        if len(ckpt_row) > 0:
                            ckpt_row = ckpt_row.iloc[0]
                            all_changes.append({
                                'checkpoint': ckpt_name,
                                'layer': longname,
                                'alpha_change': abs(ckpt_row['alpha'] - base_row['alpha']),
                                'spectral_change': abs(ckpt_row['spectral_norm'] - base_row['spectral_norm']),
                                'norm_change': abs(ckpt_row['log_norm'] - base_row['log_norm'])
                            })

    if all_changes:
        changes_df = pd.DataFrame(all_changes)

        print("\nTop 10 Alpha Changes:")
        print(changes_df.nlargest(10, 'alpha_change')[['checkpoint', 'layer', 'alpha_change']].to_string(index=False))

        print("\nTop 10 Spectral Norm Changes:")
        print(changes_df.nlargest(10, 'spectral_change')[['checkpoint', 'layer', 'spectral_change']].to_string(index=False))

        # Save detailed comparison
        output_file = Path("weightwatcher_analysis/layer_13_15_comparison.json")
        with open(output_file, 'w') as f:
            json.dump(all_changes, f, indent=2)
        print(f"\n✓ Detailed comparison saved to: {output_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_checkpoints()
