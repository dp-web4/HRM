#!/usr/bin/env python3
"""Debug mRoPE shape mismatches"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from selective_transformer_layer import MultimodalRotaryEmbedding, apply_multimodal_rotary_pos_emb

# Test shapes
batch_size = 1
num_heads = 32
seq_len = 5
head_dim = 128

# Create mRoPE
mrope = MultimodalRotaryEmbedding(
    dim=head_dim,
    mrope_section=[24, 20, 20],
)

# Get cos/sin
x = torch.randn(batch_size, seq_len, head_dim)
cos, sin = mrope(x, seq_len=seq_len)

print(f"After mRoPE forward:")
print(f"  cos shape: {cos.shape}")
print(f"  sin shape: {sin.shape}")

# Simulate query/key states after projection and reshape
q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)

print(f"\nQuery/Key shapes:")
print(f"  q shape: {q.shape}")
print(f"  k shape: {k.shape}")

# Try to apply
try:
    q_embed, k_embed = apply_multimodal_rotary_pos_emb(
        q, k, cos, sin,
        mrope_section=[24, 20, 20],
    )
    print(f"\n✅ Success!")
    print(f"  q_embed shape: {q_embed.shape}")
    print(f"  k_embed shape: {k_embed.shape}")
except Exception as e:
    print(f"\n❌ Error: {e}")

    # Debug what's happening inside
    print(f"\nDEBUG: Inside apply_multimodal_rotary_pos_emb:")
    mrope_section_doubled = [s * 2 for s in [24, 20, 20]]
    print(f"  mrope_section_doubled: {mrope_section_doubled}")

    cos_chunks = cos.split(mrope_section_doubled, dim=-1)
    print(f"  len(cos_chunks): {len(cos_chunks)}")
    for i, chunk in enumerate(cos_chunks):
        print(f"  cos_chunks[{i}] shape: {chunk.shape}")

    # Interleave
    cos_interleaved = torch.cat([m[i % 3] for i, m in enumerate(cos_chunks)], dim=-1)
    print(f"  After interleave, cos shape: {cos_interleaved.shape}")

    cos_final = cos_interleaved.unsqueeze(2)
    print(f"  After unsqueeze(2), cos shape: {cos_final.shape}")

    print(f"\nExpected for broadcasting with q {q.shape}:")
    print(f"  Need: [batch=1, 1, seq_len=5, head_dim=128]")
