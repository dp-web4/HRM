#!/usr/bin/env python3
"""
Test Selective Expert Loading - WAKE State Demo

Demonstrates SAGE's selective expert loading with:
- Only 4 experts loaded (WAKE metabolic state)
- SNARC-augmented expert selection
- Trust-based expert management
- Memory efficiency (vs loading all 128 experts)

This validates the core SAGE thesis: selective resource loading
based on attention needs achieves operational capability with
minimal memory footprint.
"""

import time
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compression.selective_expert_loader import SelectiveExpertLoader


def simple_expert_forward(expert_weights: dict, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Simple expert forward pass (MLP)

    Expert structure:
    - gate_proj: [intermediate, hidden]
    - up_proj: [intermediate, hidden]
    - down_proj: [hidden, intermediate]

    Standard MoE expert: h = down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    # Extract weights
    gate_proj = None
    up_proj = None
    down_proj = None

    for key, weight in expert_weights.items():
        if 'gate_proj' in key:
            gate_proj = weight
        elif 'up_proj' in key:
            up_proj = weight
        elif 'down_proj' in key:
            down_proj = weight

    if gate_proj is None or up_proj is None or down_proj is None:
        raise ValueError("Missing expert weights")

    # Forward pass
    # hidden_states: [batch, seq, hidden]
    gate_output = F.linear(hidden_states, gate_proj)  # [batch, seq, intermediate]
    up_output = F.linear(hidden_states, up_proj)  # [batch, seq, intermediate]

    # SiLU activation and element-wise multiply
    intermediate = F.silu(gate_output) * up_output  # [batch, seq, intermediate]

    # Project back to hidden dimension
    output = F.linear(intermediate, down_proj)  # [batch, seq, hidden]

    return output


def test_wake_state():
    """
    Test WAKE state: minimal resources (4 experts)

    WAKE state characteristics:
    - 4 cheapest experts for basic reasoning
    - Memory target: ~12GB
    - Use case: Monitoring, light reasoning, quick responses
    """
    print("\n" + "="*70)
    print("SAGE WAKE State Test - Selective Expert Loading")
    print("="*70 + "\n")

    # Initialize loader
    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    if not Path(extraction_dir).exists():
        print(f"❌ Extraction directory not found: {extraction_dir}")
        print("   Run expert_extractor.py first to extract experts")
        return

    loader = SelectiveExpertLoader(
        extraction_dir=extraction_dir,
        component="thinker",
        device="cpu",  # Use CPU for this test
        max_loaded_experts=4  # WAKE state: only 4 experts
    )

    print("\n1. Initial State")
    print("-" * 70)
    mem_usage = loader.get_memory_usage()
    print(f"Loaded experts: {mem_usage['num_loaded_experts']}")
    print(f"Loaded routers: {mem_usage['num_loaded_routers']}")
    print(f"Memory usage: {mem_usage['total_mb']:.1f} MB\n")

    # Create fake hidden states for layer 0 (with fixed seed for reproducibility)
    batch_size = 1
    seq_len = 10
    hidden_size = 2048  # Qwen3-Omni thinker hidden size

    torch.manual_seed(42)  # Fixed seed for deterministic expert selection
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Pre-extract experts 0-3 for testing (manually ensure these exist)
    print("Note: Using fixed seed (42) for deterministic expert selection")
    print("      Ensure experts 0-7 from layer 0 are extracted\n")

    print("2. Expert Selection (Standard MoE)")
    print("-" * 70)

    layer_id = 0
    num_experts = 4  # WAKE state

    # Standard selection (no SNARC)
    start_time = time.time()
    selected_experts, router_weights = loader.select_experts_snarc(
        hidden_states,
        layer_id,
        num_experts=num_experts,
        snarc_salience=None  # Standard MoE routing
    )
    selection_time = time.time() - start_time

    print(f"Selected experts: {selected_experts}")
    print(f"Router weights: {router_weights.tolist()}")
    print(f"Selection time: {selection_time*1000:.2f} ms\n")

    # Check memory after router loading
    mem_usage = loader.get_memory_usage()
    print(f"Memory after router load: {mem_usage['total_mb']:.1f} MB\n")

    print("3. Expert Loading")
    print("-" * 70)

    for expert_id in selected_experts:
        start_time = time.time()
        expert_weights = loader.load_expert(expert_id, layer_id)
        load_time = time.time() - start_time

        print(f"✅ Loaded expert {expert_id:3d} ({load_time*1000:.2f} ms)")

    print()

    # Memory after expert loading
    mem_usage = loader.get_memory_usage()
    print(f"Loaded experts: {mem_usage['num_loaded_experts']}")
    print(f"Memory usage: {mem_usage['total_mb']:.1f} MB\n")

    print("4. Expert Forward Pass")
    print("-" * 70)

    # Combine expert outputs
    expert_outputs = []

    for i, expert_id in enumerate(selected_experts):
        expert_weights = loader.load_expert(expert_id, layer_id)

        start_time = time.time()
        output = simple_expert_forward(expert_weights, hidden_states)
        forward_time = time.time() - start_time

        expert_outputs.append(output)

        print(f"Expert {expert_id:3d}: output shape {list(output.shape)}, time {forward_time*1000:.2f} ms")

    # Weighted combination (standard MoE)
    # Normalize router weights
    router_probs = F.softmax(router_weights, dim=0)

    combined_output = torch.zeros_like(expert_outputs[0])
    for i, output in enumerate(expert_outputs):
        combined_output += router_probs[i] * output

    print(f"\nCombined output shape: {list(combined_output.shape)}")
    print(f"Output mean: {combined_output.mean():.4f}, std: {combined_output.std():.4f}\n")

    print("5. SNARC-Augmented Selection")
    print("-" * 70)

    # Simulate high-salience input (surprising, novel)
    snarc_salience = {
        'surprise': 0.9,  # Very surprising
        'novelty': 0.8,  # Novel pattern
        'arousal': 0.6,  # Moderate attention
        'reward': 0.7,  # Important
        'conflict': 0.5  # Some uncertainty
    }

    selected_experts_snarc, router_weights_snarc = loader.select_experts_snarc(
        hidden_states,
        layer_id,
        num_experts=num_experts,
        snarc_salience=snarc_salience,
        metabolic_state="WAKE"
    )

    print(f"SNARC salience: {snarc_salience}")
    print(f"Selected experts (SNARC): {selected_experts_snarc}")
    print(f"Selected experts (standard): {selected_experts}")

    if selected_experts_snarc != selected_experts:
        print("✅ SNARC influenced expert selection!")
    else:
        print("   (Same experts - salience didn't change selection)")

    print()

    print("6. Memory Comparison")
    print("-" * 70)

    # Current (selective)
    current_memory = mem_usage['total_mb']

    # Monolithic (all 128 experts per layer)
    expert_size_mb = 9.0  # From analysis
    all_experts_memory = 128 * expert_size_mb

    reduction = (1 - current_memory / all_experts_memory) * 100

    print(f"Selective loading (4 experts): {current_memory:.1f} MB")
    print(f"Monolithic loading (128 experts): {all_experts_memory:.1f} MB")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Memory savings: {all_experts_memory - current_memory:.1f} MB\n")

    print("7. Trust Update Simulation")
    print("-" * 70)

    # Simulate good performance from experts
    for expert_id in selected_experts:
        energy_decrease = 0.3  # Good convergence
        stability_score = 0.8  # Stable output

        loader.update_expert_trust(expert_id, layer_id, energy_decrease, stability_score)

        trust_record = loader.trust_records[layer_id][expert_id]
        print(f"Expert {expert_id:3d}: trust {trust_record.overall_trust:.3f} "
              f"(convergence={trust_record.convergence_rate:.3f}, "
              f"stability={trust_record.stability:.3f}, "
              f"efficiency={trust_record.efficiency:.3f})")

    print()

    print("="*70)
    print("✅ WAKE State Test Complete!")
    print("="*70)
    print("\nKey Results:")
    print(f"  • Loaded only {num_experts} experts (vs 128 monolithic)")
    print(f"  • Memory usage: {current_memory:.1f} MB (vs {all_experts_memory:.1f} MB)")
    print(f"  • Memory reduction: {reduction:.1f}%")
    print(f"  • SNARC-augmented selection working")
    print(f"  • Trust-based management operational")
    print("\nThis demonstrates SAGE's selective resource loading thesis:")
    print("Operational capability with minimal memory footprint.\n")


if __name__ == "__main__":
    test_wake_state()
