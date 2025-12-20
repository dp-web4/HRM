#!/usr/bin/env python3
"""
Verify our implementation matches direct weight calculations

This doesn't require loading the full model - just verifies we're using
the extracted weights correctly at each stage.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import safetensors
import sys
import os

def main():
    print("="*80)
    print("🔍 Verifying Weight Calculations")
    print("="*80)

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    model_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Test prompt
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"\n💬 Prompt: '{prompt}'")
    print(f"   Token IDs: {input_ids[0].tolist()}")

    # Step 1: Load embeddings and verify shape
    print("\n📝 Step 1: Embeddings")
    embed_path = os.path.join(extraction_dir, "embeddings", "thinker_embeddings.safetensors")
    with safetensors.safe_open(embed_path, framework="pt") as f:
        embed_weight = f.get_tensor('embed_tokens.weight').float()  # Convert to float32

    print(f"   Embedding table shape: {embed_weight.shape}")
    print(f"   Embedding dtype: {embed_weight.dtype}")
    embedded = embed_weight[input_ids[0]]  # [seq_len, hidden_size]
    print(f"   Embedded sequence shape: {embedded.shape}")
    print(f"   Embedded mean: {embedded.mean():.4f}, std: {embedded.std():.4f}")

    # Step 2: Load router for layer 0 and verify routing
    print("\n🧭 Step 2: Router (Layer 0)")
    router_path = os.path.join(extraction_dir, "routers", "thinker_router_layer_00.safetensors")
    with safetensors.safe_open(router_path, framework="pt") as f:
        router_weight = f.get_tensor('thinker.model.layers.0.mlp.gate.weight').float()  # Convert to float32

    print(f"   Router weight shape: {router_weight.shape}")  # [num_experts, hidden_size]
    print(f"   Router dtype: {router_weight.dtype}")

    # Compute router logits for last token
    last_token_hidden = embedded[-1:,  :]  # [1, hidden_size]
    router_logits = F.linear(last_token_hidden, router_weight)  # [1, num_experts]
    print(f"   Router logits shape: {router_logits.shape}")

    # Apply softmax
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

    # Select top-8
    top_k_values, top_k_indices = torch.topk(routing_weights, k=8, dim=-1)

    # Normalize (norm_topk_prob=True)
    top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)

    print(f"   Top-8 experts: {top_k_indices[0].tolist()}")
    print(f"   Top-8 weights: {top_k_values[0].tolist()}")
    print(f"   Weights sum to: {top_k_values.sum():.4f}")

    # Step 3: Verify expert computation for first selected expert
    print("\n⚙️  Step 3: Expert Computation (Expert #{})".format(top_k_indices[0, 0].item()))
    expert_id = top_k_indices[0, 0].item()

    # Load expert weights
    expert_dir = os.path.join(extraction_dir, "experts")
    expert_file = f"expert_{expert_id:03d}_layer_00.safetensors"
    expert_path = os.path.join(expert_dir, expert_file)

    if os.path.exists(expert_path):
        with safetensors.safe_open(expert_path, framework="pt") as f:
            gate_proj = f.get_tensor(f"thinker.model.layers.0.mlp.experts.{expert_id}.gate_proj.weight")
            up_proj = f.get_tensor(f"thinker.model.layers.0.mlp.experts.{expert_id}.up_proj.weight")
            down_proj = f.get_tensor(f"thinker.model.layers.0.mlp.experts.{expert_id}.down_proj.weight")

        print(f"   Gate proj: {gate_proj.shape}")
        print(f"   Up proj: {up_proj.shape}")
        print(f"   Down proj: {down_proj.shape}")

        # Compute expert output: down(silu(gate(x)) * up(x))
        gate_out = F.linear(last_token_hidden, gate_proj)
        up_out = F.linear(last_token_hidden, up_proj)
        intermediate = F.silu(gate_out) * up_out
        expert_out = F.linear(intermediate, down_proj)

        print(f"   Expert output shape: {expert_out.shape}")
        print(f"   Expert output mean: {expert_out.mean():.4f}, std: {expert_out.std():.4f}")
    else:
        print(f"   ⚠️  Expert file not found: {expert_file}")

    # Step 4: Now compare with our implementation
    print("\n🔄 Step 4: Compare with our implementation")
    sys.path.insert(0, 'sage/compression')
    from selective_expert_loader import SelectiveExpertLoader

    loader = SelectiveExpertLoader(
        extraction_dir=extraction_dir,
        component="thinker",
        device="cpu",
        max_loaded_experts=16
    )

    # Get our router selection
    our_expert_ids, our_weights = loader.select_experts_snarc(
        embedded.unsqueeze(0),  # Add batch dimension
        layer_id=0,
        num_experts=8,
        snarc_salience=None,  # No SNARC, just standard routing
        metabolic_state="FOCUS"
    )

    print(f"   Our top-8 experts: {our_expert_ids[0, -1].tolist()}")
    print(f"   Our top-8 weights: {our_weights[0, -1].tolist()}")

    # Compare
    our_ids = set(our_expert_ids[0, -1].tolist())
    ref_ids = set(top_k_indices[0].tolist())

    if our_ids == ref_ids:
        print(f"   ✅ Expert selection MATCHES!")
    else:
        print(f"   ❌ Expert selection DIFFERS!")
        print(f"      Missing: {ref_ids - our_ids}")
        print(f"      Extra: {our_ids - ref_ids}")

    # Compare weights (allowing for small numerical differences)
    weight_diff = torch.abs(our_weights[0, -1] - top_k_values[0]).max().item()
    print(f"   Weight difference: {weight_diff:.6f}")
    if weight_diff < 1e-4:
        print(f"   ✅ Weights MATCH (within tolerance)!")
    else:
        print(f"   ⚠️  Weights differ more than expected")

    print("\n" + "="*80)
    print("✅ Verification Complete")
    print("="*80)

if __name__ == "__main__":
    main()
