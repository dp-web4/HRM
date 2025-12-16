#!/usr/bin/env python3
"""Debug mRoPE dimensions"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from selective_transformer_layer import MultimodalRotaryEmbedding

# Test mRoPE with Q3-Omni config
mrope = MultimodalRotaryEmbedding(
    dim=128,  # head_dim
    mrope_section=[24, 20, 20],  # Q3-Omni config
)

print(f"mRoPE sections: {mrope.mrope_section}")
print(f"Number of sections: {len(mrope.inv_freqs)}")

for i, inv_freq in enumerate(mrope.inv_freqs):
    print(f"\nSection {i}:")
    print(f"  inv_freq length: {len(inv_freq)}")

# Test forward
x = torch.randn(1, 5, 128)  # [batch, seq, dim]
cos, sin = mrope(x, seq_len=5)

print(f"\nOutput dimensions:")
print(f"  cos: {cos.shape}")
print(f"  sin: {sin.shape}")
print(f"\nExpected: [seq_len=5, dim=128]")
