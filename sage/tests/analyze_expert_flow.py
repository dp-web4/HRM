#!/usr/bin/env python3
"""
Analyze expert selection patterns across layers.

Purpose: Determine bundling opportunities by identifying:
1. Which experts are selected together across layers
2. Cross-layer correlations (does L1.45 → L2.100 happen frequently?)
3. Per-token expert flow patterns
4. Opportunities for bundling vs per-layer strategies
"""

import torch
import sys
import os
from collections import defaultdict
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from transformers import AutoTokenizer
from sage.compression.selective_language_model import SelectiveLanguageModel

def analyze_expert_flow():
    """Analyze which experts get selected across layers."""

    print("=" * 80)
    print("EXPERT FLOW ANALYSIS - Cross-Layer Correlation Study")
    print("=" * 80)
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        component="thinker",
        num_experts_per_tok=4,
        device="cpu"
    )

    print(f"✅ Model loaded")
    print()

    # Test prompts covering different domains
    prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, humanity will",
        "The algorithm works by",
        "Once upon a time in a distant land",
        "The derivative of f(x) = x² is",
    ]

    # Data structures for analysis
    layer_selections = defaultdict(lambda: defaultdict(int))  # {layer: {expert: count}}
    cross_layer_pairs = defaultdict(int)  # {(L1.expert, L2.expert): count}
    per_token_flows = []  # [{layer: [experts]}, ...]

    # Patch expert loader to capture selections
    original_select = model.expert_loader.select_experts

    def tracking_select(layer_id, hidden_states, num_experts, snarc_salience=None):
        # Call original
        expert_ids, logits = original_select(layer_id, hidden_states, num_experts, snarc_salience)

        # Track selections
        for expert_id in expert_ids:
            layer_selections[layer_id][expert_id] += 1

        # Store for cross-layer analysis
        if not hasattr(tracking_select, 'current_token_flow'):
            tracking_select.current_token_flow = {}
        tracking_select.current_token_flow[layer_id] = expert_ids

        return expert_ids, logits

    model.expert_loader.select_experts = tracking_select

    # Run generation on each prompt
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Processing: \"{prompt}\"")

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            # Generate tokens
            for _ in range(10):  # Generate 10 tokens
                tracking_select.current_token_flow = {}

                output = model(input_ids)
                next_token = output.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Capture flow for this token
                if hasattr(tracking_select, 'current_token_flow'):
                    per_token_flows.append(dict(tracking_select.current_token_flow))

                    # Build cross-layer pairs
                    flow = tracking_select.current_token_flow
                    for layer_id in sorted(flow.keys())[:-1]:  # All but last layer
                        next_layer = layer_id + 1
                        if next_layer in flow:
                            for expert_a in flow[layer_id]:
                                for expert_b in flow[next_layer]:
                                    key = (f"L{layer_id}.E{expert_a}", f"L{next_layer}.E{expert_b}")
                                    cross_layer_pairs[key] += 1

        print(f"    Processed {len(per_token_flows)} token transitions")

    # Restore original
    model.expert_loader.select_experts = original_select

    print()
    print("=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print()

    # 1. Per-layer expert usage
    print("📊 Expert Usage by Layer (Top 10 per layer)")
    print("-" * 80)
    for layer in sorted(layer_selections.keys())[:5]:  # Show first 5 layers
        experts = layer_selections[layer]
        top_experts = sorted(experts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nLayer {layer}:")
        for expert_id, count in top_experts:
            print(f"  Expert {expert_id:3d}: {count:3d} activations")

    print()
    print("=" * 80)
    print()

    # 2. Cross-layer correlations (bundling opportunities)
    print("🔗 Cross-Layer Correlations (Top 20 pairs)")
    print("-" * 80)
    print("Pattern: L1.E45 → L2.E100 means layer 1 expert 45 often flows to layer 2 expert 100")
    print()

    top_pairs = sorted(cross_layer_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
    for (pair, count) in top_pairs:
        src, dst = pair
        print(f"  {src} → {dst}: {count:3d} times")

    print()
    print("=" * 80)
    print()

    # 3. Bundling analysis
    print("📦 Bundling Opportunity Analysis")
    print("-" * 80)

    # Find experts that appear together frequently across layers
    expert_cooccurrence = defaultdict(lambda: defaultdict(int))
    for flow in per_token_flows:
        experts_this_token = set()
        for layer_id, expert_ids in flow.items():
            for expert_id in expert_ids:
                experts_this_token.add(expert_id)

        # Count co-occurrences (same expert across different layers in same token)
        for expert_id in experts_this_token:
            layers_with_expert = [l for l, exps in flow.items() if expert_id in exps]
            if len(layers_with_expert) > 1:
                expert_cooccurrence[expert_id]['multi_layer_activations'] += 1
                expert_cooccurrence[expert_id]['total_layers'] += len(layers_with_expert)

    print("\nExperts that activate across multiple layers (bundling candidates):")
    bundling_candidates = sorted(
        [(exp, data['multi_layer_activations'], data['total_layers'] / data['multi_layer_activations'])
         for exp, data in expert_cooccurrence.items()],
        key=lambda x: x[1],
        reverse=True
    )[:15]

    for expert_id, multi_activations, avg_layers in bundling_candidates:
        print(f"  Expert {expert_id:3d}: {multi_activations:3d} multi-layer tokens, avg {avg_layers:.1f} layers/token")

    print()
    print("=" * 80)
    print()

    # 4. Per-layer strategy analysis
    print("🎯 Per-Layer Strategy Insights")
    print("-" * 80)

    # Calculate expert diversity per layer
    for layer in sorted(layer_selections.keys())[:5]:
        experts = layer_selections[layer]
        total = sum(experts.values())
        unique = len(experts)
        entropy = -sum((count/total) * torch.log2(torch.tensor(count/total)).item()
                      for count in experts.values() if count > 0)

        print(f"\nLayer {layer}:")
        print(f"  Unique experts used: {unique}/128 ({unique/128*100:.1f}%)")
        print(f"  Entropy (diversity): {entropy:.2f} bits")
        print(f"  Interpretation: {'High diversity - per-layer loading optimal' if entropy > 4 else 'Low diversity - bundling may help'}")

    print()
    print("=" * 80)
    print()

    # Save detailed results
    results = {
        'layer_selections': {k: dict(v) for k, v in layer_selections.items()},
        'cross_layer_pairs': {f"{k[0]}→{k[1]}": v for k, v in cross_layer_pairs.items()},
        'bundling_candidates': [
            {'expert_id': exp, 'multi_layer_activations': act, 'avg_layers': avg}
            for exp, act, avg in bundling_candidates
        ],
        'per_token_flows': per_token_flows[:10],  # Sample for inspection
    }

    output_file = '/tmp/expert_flow_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📄 Detailed results saved to: {output_file}")
    print()
    print("=" * 80)
    print()

    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 80)
    print()
    print("Based on the analysis:")
    print()
    print("1. If expert diversity per layer is HIGH (>50% unique experts):")
    print("   → Per-layer loading is optimal (current approach)")
    print("   → Load only needed experts per layer (9MB each)")
    print()
    print("2. If certain experts appear across MANY layers:")
    print("   → Consider bundling those specific experts")
    print("   → Hybrid: bundle common experts, per-layer for rare ones")
    print()
    print("3. If strong L1→L2 correlations exist:")
    print("   → Predictive loading: when L1.45 selected, pre-load L2.100")
    print("   → Reduces latency without increasing memory")
    print()
    print("=" * 80)

if __name__ == "__main__":
    analyze_expert_flow()
