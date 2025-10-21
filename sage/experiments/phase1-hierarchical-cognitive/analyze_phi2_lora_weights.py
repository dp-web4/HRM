"""
Analyze Phi-2 LoRA Adapter Weights

Question: Where does epistemic stance live in low-rank adaptations?
- Which layers have the most significant LoRA changes?
- What's the intrinsic dimensionality of stance?
- How does LoRA localization compare with full fine-tuning?
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import weightwatcher as ww
import json
from pathlib import Path
import pandas as pd


def analyze_lora_adapter(adapter_path, stance_name):
    """Analyze a single LoRA adapter"""

    print(f"\n{'='*80}")
    print(f"Analyzing LoRA adapter: {stance_name}")
    print(f"{'='*80}\n")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="cpu",  # Keep on CPU for analysis
        trust_remote_code=True
    )

    # Load with adapter
    model_with_adapter = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16
    )

    print(f"✓ Loaded adapter from {adapter_path}\n")

    # Analyze adapter parameters directly
    print("LoRA Adapter Parameters:")
    print("-" * 80)

    adapter_params = {}
    for name, param in model_with_adapter.named_parameters():
        if 'lora' in name.lower():
            print(f"{name}: {param.shape} ({param.numel():,} params)")
            adapter_params[name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'norm': param.norm().item() if param.numel() > 0 else 0,
                'mean': param.mean().item() if param.numel() > 0 else 0,
                'std': param.std().item() if param.numel() > 0 else 0
            }

    print(f"\nTotal LoRA parameters: {sum(p['numel'] for p in adapter_params.values()):,}")

    # Analyze with WeightWatcher
    print("\n" + "="*80)
    print("WeightWatcher Analysis")
    print("="*80 + "\n")

    watcher = ww.WeightWatcher(model=model_with_adapter)
    details = watcher.analyze(randomize=False, mp_fit=True)
    summary = watcher.get_summary()

    print("Summary metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Find layers with LoRA modifications
    lora_layer_changes = []

    for _, row in details.iterrows():
        layer_name = str(row.get('longname', row.get('name', '')))
        if 'lora' in layer_name.lower():
            lora_layer_changes.append({
                'layer_name': layer_name,
                'alpha': row.get('alpha', 0),
                'log_norm': row.get('log_norm', 0),
                'log_spectral_norm': row.get('log_spectral_norm', 0)
            })

    print(f"\n\nFound {len(lora_layer_changes)} LoRA-modified layers")
    print("\nTop LoRA layers by alpha:")
    lora_df = pd.DataFrame(lora_layer_changes)
    if not lora_df.empty and 'alpha' in lora_df.columns:
        top_lora = lora_df.nlargest(10, 'alpha')
        print(top_lora[['layer_name', 'alpha', 'log_norm']].to_string())

    # Cleanup
    del model_with_adapter
    del base_model
    torch.cuda.empty_cache()

    return {
        'stance': stance_name,
        'adapter_params': adapter_params,
        'lora_layer_changes': lora_layer_changes,
        'summary': summary
    }


def main():
    print("\n" + "="*80)
    print("PHI-2 LoRA ADAPTER WEIGHT ANALYSIS")
    print("="*80)
    print("\nAnalyzing where epistemic stance lives in low-rank adaptations\n")

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

        results = analyze_lora_adapter(adapter_path, stance)
        all_results[stance] = results

        print("\n" + "="*80 + "\n")

    # Save results
    output_dir = Path("weight_analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {}
    for stance, results in all_results.items():
        json_results[stance] = {
            'stance': results['stance'],
            'adapter_params': results['adapter_params'],
            'summary': {k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in results['summary'].items()},
            'num_lora_layers': len(results['lora_layer_changes'])
        }

    with open(output_dir / "phi2_lora_analysis.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✓ Results saved to {output_dir}/phi2_lora_analysis.json\n")

    # Summary comparison
    print("LoRA Adapter Comparison:")
    print("-" * 80)
    for stance, results in all_results.items():
        total_params = sum(p['numel'] for p in results['adapter_params'].values())
        total_norm = sum(p['norm'] for p in results['adapter_params'].values())
        print(f"\n{stance}:")
        print(f"  Total LoRA params: {total_params:,}")
        print(f"  Total parameter norm: {total_norm:.4f}")
        print(f"  Average alpha: {results['summary'].get('alpha', 0):.4f}")


if __name__ == "__main__":
    main()
