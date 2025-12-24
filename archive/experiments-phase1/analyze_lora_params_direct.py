"""
Direct LoRA Parameter Analysis

Skip WeightWatcher - directly analyze LoRA adapter parameters to find:
- Which layers have strongest LoRA updates
- Parameter norms by layer
- Localization patterns across stances
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
import json
from pathlib import Path
import pandas as pd


def analyze_lora_params(adapter_path, stance_name):
    """Analyze LoRA parameters directly"""

    print(f"\n{'='*80}")
    print(f"Analyzing: {stance_name}")
    print(f"{'='*80}\n")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Load with adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16
    )

    print("LoRA Parameters by Layer:")
    print("-" * 80)

    layer_stats = []

    for name, param in model.named_parameters():
        if 'lora' in name.lower():  # Don't check requires_grad - adapters have it False when loaded
            # Extract layer number and type
            parts = name.split('.')
            layer_num = None
            proj_type = None

            # Find layer number (should be after "layers")
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i+1].isdigit():
                    layer_num = int(parts[i+1])
                    break

            # Extract projection type
            if 'q_proj' in name:
                proj_type = 'q_proj'
            elif 'v_proj' in name:
                proj_type = 'v_proj'

            lora_type = 'A' if 'lora_A' in name else 'B'

            norm = param.norm().item()
            mean = param.mean().item()
            std = param.std().item()

            layer_stats.append({
                'layer': layer_num,
                'projection': proj_type,
                'lora_matrix': lora_type,
                'norm': norm,
                'mean': mean,
                'std': std,
                'numel': param.numel(),
                'param_name': name
            })

    # Convert to DataFrame and aggregate by layer
    df = pd.DataFrame(layer_stats)

    # Debug: print first few rows
    print(f"Found {len(df)} LoRA parameters")
    if len(df) > 0:
        print("\nFirst few entries:")
        print(df.head()[['param_name', 'layer', 'projection', 'norm']].to_string())

        # Check if layer column has valid values
        print(f"\nUnique layers found: {sorted(df['layer'].dropna().unique())}")

    # Aggregate by layer (skip None values)
    df_valid = df[df['layer'].notna()]
    if len(df_valid) == 0:
        print("\n⚠ No valid layer numbers found in LoRA parameters!")
        return None

    layer_agg = df_valid.groupby('layer').agg({
        'norm': 'sum',
        'numel': 'sum'
    }).reset_index()

    layer_agg['norm_per_param'] = layer_agg['norm'] / layer_agg['numel']
    layer_agg = layer_agg.sort_values('norm', ascending=False)

    print("\nTop 10 layers by total LoRA norm:")
    print(layer_agg.head(10)[['layer', 'norm', 'norm_per_param']].to_string())

    # Aggregate by projection type
    proj_agg = df.groupby('projection').agg({
        'norm': 'sum',
        'numel': 'sum'
    }).reset_index()

    print("\n\nLoRA updates by projection type:")
    print(proj_agg[['projection', 'norm']].to_string())

    # Cleanup
    del model
    del base_model
    torch.cuda.empty_cache()

    return {
        'stance': stance_name,
        'layer_stats': layer_stats,
        'top_layers': layer_agg.head(10).to_dict('records'),
        'projection_stats': proj_agg.to_dict('records')
    }


def main():
    print("\n" + "="*80)
    print("PHI-2 LoRA PARAMETER ANALYSIS")
    print("="*80)
    print("\nDirect analysis of LoRA adapter weights\n")

    adapter_base = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/phi-2-lora"

    stances = [
        "curious-uncertainty",
        "confident-expertise",
        "engaged-difficulty"
    ]

    all_results = {}

    for stance in stances:
        adapter_path = f"{adapter_base}/{stance}"

        if not Path(adapter_path).exists():
            print(f"⚠ Adapter not found: {adapter_path}")
            continue

        results = analyze_lora_params(adapter_path, stance)
        all_results[stance] = results

        print("\n")

    # Cross-stance comparison
    print("="*80)
    print("CROSS-STANCE COMPARISON")
    print("="*80 + "\n")

    # Compare which layers are most important across stances
    for stance, results in all_results.items():
        print(f"\n{stance} - Top 3 layers:")
        for i, layer_info in enumerate(results['top_layers'][:3], 1):
            print(f"  {i}. Layer {layer_info['layer']}: norm={layer_info['norm']:.4f}")

    # Save results
    output_dir = Path("weight_analysis_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "phi2_lora_param_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/phi2_lora_param_analysis.json\n")


if __name__ == "__main__":
    main()
