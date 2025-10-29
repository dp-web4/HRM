#!/usr/bin/env python3
"""
WeightWatcher Analysis: Compare Original Qwen → Phase 1 → Phase 2.1

Analyzes weight distributions to understand what training actually changed.

Models:
1. Original: Qwen/Qwen2.5-0.5B-Instruct (base model)
2. Phase 1: epistemic-pragmatism (25 examples)
3. Phase 2.1: Introspective-Qwen (115 examples)

Key Metrics:
- Alpha: Power law exponent (heavy tails = good generalization)
- Log Norm: Frobenius norm in log space
- Spectral Norm: Largest singular value
- Stable Rank: Effective dimensionality
"""

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import weightwatcher as ww
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class WeightDistributionAnalyzer:
    """Analyze and compare weight distributions across models"""

    def __init__(self, base_model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.base_model_name = base_model_name
        self.results = {}

        print("=" * 80)
        print("WeightWatcher Comparison: Original → Phase 1 → Phase 2.1")
        print("=" * 80)
        print()

    def analyze_model(self, model, name):
        """Run weightwatcher analysis on a model"""
        print(f"\n{'='*80}")
        print(f"Analyzing: {name}")
        print(f"{'='*80}\n")

        # Initialize weightwatcher
        watcher = ww.WeightWatcher(model=model)

        # Run analysis
        print("Running WeightWatcher analysis...")
        details = watcher.analyze(randomize=False)
        summary = watcher.get_summary()

        # Extract key metrics
        results = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'alpha': summary.get('alpha', None),
                'alpha_weighted': summary.get('alpha_weighted', None),
                'log_norm': summary.get('log_norm', None),
                'log_spectral_norm': summary.get('log_spectral_norm', None),
                'num_layers': summary.get('num_layers', None),
            },
            'layer_details': details.to_dict('records') if details is not None else []
        }

        # Print summary
        print(f"\n{'-'*80}")
        print(f"Summary for {name}:")
        print(f"{'-'*80}")
        for key, value in results['summary'].items():
            if value is not None:
                print(f"  {key:20s}: {value:.4f}" if isinstance(value, float) else f"  {key:20s}: {value}")

        return results

    def load_and_analyze_original(self):
        """Load and analyze original base model"""
        print(f"\nLoading original base model: {self.base_model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        results = self.analyze_model(model, "Original Qwen2.5-0.5B-Instruct")
        self.results['original'] = results

        # Clean up
        del model
        torch.cuda.empty_cache()

        return results

    def load_and_analyze_phase1(self):
        """Load and analyze Phase 1 model (epistemic-pragmatism)"""
        phase1_path = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

        print(f"\nLoading Phase 1 model from: {phase1_path}")

        # Load base + adapter
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        model = PeftModel.from_pretrained(base, phase1_path)

        # Merge adapter for analysis
        print("Merging LoRA adapter...")
        model = model.merge_and_unload()

        results = self.analyze_model(model, "Phase 1 (epistemic-pragmatism, 25 examples)")
        self.results['phase1'] = results

        # Clean up
        del model
        torch.cuda.empty_cache()

        return results

    def load_and_analyze_phase2(self):
        """Load and analyze Phase 2.1 model (Introspective-Qwen)"""
        phase2_path = "./Introspective-Qwen-0.5B-v2.1/model"

        print(f"\nLoading Phase 2.1 model from: {phase2_path}")

        # Load base + adapter
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        model = PeftModel.from_pretrained(base, phase2_path)

        # Merge adapter for analysis
        print("Merging LoRA adapter...")
        model = model.merge_and_unload()

        results = self.analyze_model(model, "Phase 2.1 (Introspective-Qwen, 115 examples)")
        self.results['phase2.1'] = results

        # Clean up
        del model
        torch.cuda.empty_cache()

        return results

    def compare_results(self):
        """Generate comparison analysis"""
        print(f"\n{'='*80}")
        print("COMPARISON ANALYSIS")
        print(f"{'='*80}\n")

        # Extract summaries
        orig = self.results['original']['summary']
        p1 = self.results['phase1']['summary']
        p2 = self.results['phase2.1']['summary']

        # Create comparison table
        comparison = {
            'Metric': [],
            'Original': [],
            'Phase 1': [],
            'Phase 2.1': [],
            'Δ (P1-Orig)': [],
            'Δ (P2.1-P1)': []
        }

        for key in ['alpha', 'alpha_weighted', 'log_norm', 'log_spectral_norm']:
            if all(results['summary'][key] is not None for results in [orig, p1, p2]):
                comparison['Metric'].append(key)
                comparison['Original'].append(f"{orig[key]:.4f}")
                comparison['Phase 1'].append(f"{p1[key]:.4f}")
                comparison['Phase 2.1'].append(f"{p2[key]:.4f}")

                delta1 = p1[key] - orig[key]
                delta2 = p2[key] - p1[key]
                comparison['Δ (P1-Orig)'].append(f"{delta1:+.4f}")
                comparison['Δ (P2.1-P1)'].append(f"{delta2:+.4f}")

        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))

        # Interpretation
        print(f"\n{'-'*80}")
        print("INTERPRETATION:")
        print(f"{'-'*80}")

        print("\n📊 Alpha (Power Law Exponent):")
        print("   Higher alpha → More heavy-tailed → Better generalization")
        print("   Ideal range: 2.0 - 4.0")

        if orig['alpha'] and p1['alpha'] and p2['alpha']:
            if p1['alpha'] > orig['alpha']:
                print(f"   ✓ Phase 1 increased alpha by {p1['alpha'] - orig['alpha']:.4f}")
            if p2['alpha'] > p1['alpha']:
                print(f"   ✓ Phase 2.1 increased alpha by {p2['alpha'] - p1['alpha']:.4f}")

        print("\n📐 Log Norm (Model Complexity):")
        print("   Lower log norm → Less complex → Better regularization")

        if orig['log_norm'] and p1['log_norm'] and p2['log_norm']:
            if p1['log_norm'] < orig['log_norm']:
                print(f"   ✓ Phase 1 reduced complexity")
            if p2['log_norm'] < p1['log_norm']:
                print(f"   ✓ Phase 2.1 further reduced complexity")

        print("\n🎯 Spectral Norm (Stability):")
        print("   Controlled spectral norm → Better training stability")

        # Save comparison
        comparison_path = Path("./exploration/weightwatcher_comparison.json")
        comparison_path.parent.mkdir(exist_ok=True)

        with open(comparison_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Full results saved to: {comparison_path}")

        return comparison

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        try:
            # Analyze all three models
            self.load_and_analyze_original()
            self.load_and_analyze_phase1()
            self.load_and_analyze_phase2()

            # Generate comparison
            self.compare_results()

            print(f"\n{'='*80}")
            print("✓ Analysis complete!")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run weightwatcher comparison"""

    # Check for exploration directory
    Path("./exploration").mkdir(exist_ok=True)

    # Run analysis
    analyzer = WeightDistributionAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
