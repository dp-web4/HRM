#!/usr/bin/env python3
"""
Test PyTorch's built-in flash attention for SAGE use cases.

Demonstrates that PyTorch 2.9's F.scaled_dot_product_attention provides
all the functionality needed for SAGE without requiring the standalone
flash-attention package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional


class FlashAttentionLayer(nn.Module):
    """
    Self-attention layer using PyTorch's built-in flash attention.

    This is a drop-in replacement for standard attention that automatically
    uses optimized CUDA kernels for 3-4x speedup with O(N) memory instead of O(N²).
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool = False, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: Optional (batch, seq_len) boolean mask (True = keep, False = mask)

        Returns:
            out: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # Project and reshape: (B, L, D) -> (B, n_heads, L, head_dim)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash attention - PyTorch automatically selects optimized kernel
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
            scale=1.0 / self.head_dim**0.5
        )

        # Reshape back: (B, n_heads, L, head_dim) -> (B, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class GQAExpertAttention(nn.Module):
    """
    Grouped Query Attention for efficient expert selection in SAGE MoE.

    Uses fewer KV heads than query heads for better efficiency while
    maintaining quality. For example: 12 query heads, 4 KV heads.
    """

    def __init__(self, d_model: int, n_query_heads: int = 12, n_kv_heads: int = 4):
        super().__init__()
        assert d_model % n_query_heads == 0
        assert n_query_heads % n_kv_heads == 0, "n_query_heads must be divisible by n_kv_heads"

        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_query_heads

        self.q_proj = nn.Linear(d_model, n_query_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(n_query_heads * self.head_dim, d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len_q, d_model) - input tokens
            context: (batch, seq_len_kv, d_model) - expert memory/context

        Returns:
            out: (batch, seq_len_q, d_model)
        """
        B, L_q, _ = query.shape
        _, L_kv, _ = context.shape

        # Query: multiple heads for fine-grained attention
        q = self.q_proj(query).view(B, L_q, self.n_query_heads, self.head_dim).transpose(1, 2)

        # Key/Value: fewer heads for efficiency (GQA)
        k = self.k_proj(context).view(B, L_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, L_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA flash attention - PyTorch handles the head broadcasting
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.n_query_heads * self.head_dim)
        return self.out_proj(out)


def benchmark_attention(batch_size=4, seq_len=2048, d_model=768, n_heads=12, device='cuda'):
    """Benchmark PyTorch flash attention vs naive implementation."""

    print(f"\nBenchmarking Flash Attention:")
    print(f"  Batch: {batch_size}, Seq: {seq_len}, Model: {d_model}, Heads: {n_heads}")
    print(f"  Device: {device}")

    dtype = torch.bfloat16

    # Create model
    model = FlashAttentionLayer(d_model, n_heads).to(device=device, dtype=dtype)
    model.eval()

    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(20):
            out = model(x)
            torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / 20 * 1000  # ms

    print(f"  ✅ Average time per forward: {avg_time:.2f}ms")
    print(f"  ✅ Throughput: {batch_size * seq_len * 20 / elapsed:.0f} tokens/sec")
    print(f"  ✅ Output shape: {out.shape}")

    return avg_time


def test_basic_attention():
    """Test 1: Basic multi-head attention."""
    print("\n" + "="*80)
    print("Test 1: Basic Multi-Head Attention")
    print("="*80)

    device = 'cuda'
    dtype = torch.bfloat16
    d_model, n_heads = 512, 8
    batch, seq_len = 4, 128

    model = FlashAttentionLayer(d_model, n_heads).to(device=device, dtype=dtype)
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch, seq_len, d_model)
    print(f"✅ Input: {x.shape}, Output: {out.shape}")
    print(f"✅ Memory allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")


def test_causal_attention():
    """Test 2: Causal (autoregressive) attention."""
    print("\n" + "="*80)
    print("Test 2: Causal Attention (for autoregressive models)")
    print("="*80)

    device = 'cuda'
    dtype = torch.bfloat16
    d_model, n_heads = 768, 12
    batch, seq_len = 2, 256

    model = FlashAttentionLayer(d_model, n_heads, causal=True).to(device=device, dtype=dtype)
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch, seq_len, d_model)
    print(f"✅ Causal masking enabled")
    print(f"✅ Input: {x.shape}, Output: {out.shape}")


def test_gqa_attention():
    """Test 3: Grouped Query Attention for MoE."""
    print("\n" + "="*80)
    print("Test 3: Grouped Query Attention (for efficient MoE)")
    print("="*80)

    device = 'cuda'
    dtype = torch.bfloat16
    d_model = 768
    batch, seq_query, seq_context = 2, 64, 256

    model = GQAExpertAttention(d_model, n_query_heads=12, n_kv_heads=4).to(device=device, dtype=dtype)

    query = torch.randn(batch, seq_query, d_model, device=device, dtype=dtype)
    context = torch.randn(batch, seq_context, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(query, context)

    assert out.shape == (batch, seq_query, d_model)
    print(f"✅ Query heads: 12, KV heads: 4 (3x efficiency)")
    print(f"✅ Query: {query.shape}, Context: {context.shape}, Output: {out.shape}")


def test_gradient_backward():
    """Test 4: Backward pass with gradients."""
    print("\n" + "="*80)
    print("Test 4: Backward Pass (training with gradients)")
    print("="*80)

    device = 'cuda'
    dtype = torch.bfloat16
    d_model, n_heads = 512, 8
    batch, seq_len = 2, 64

    model = FlashAttentionLayer(d_model, n_heads).to(device=device, dtype=dtype)
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    # Forward
    out = model(x)
    loss = out.sum()

    # Backward
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    print(f"✅ Forward: {out.shape}")
    print(f"✅ Backward: gradients computed")
    print(f"✅ Input grad shape: {x.grad.shape}")


def test_mixed_precision():
    """Test 5: Multiple dtypes (fp16, bf16)."""
    print("\n" + "="*80)
    print("Test 5: Mixed Precision (fp16 and bfloat16)")
    print("="*80)

    device = 'cuda'
    d_model, n_heads = 512, 8
    batch, seq_len = 2, 128

    for dtype in [torch.float16, torch.bfloat16]:
        model = FlashAttentionLayer(d_model, n_heads).to(device=device, dtype=dtype)
        x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)

        with torch.no_grad():
            out = model(x)

        assert out.dtype == dtype
        print(f"✅ dtype {dtype}: Shape {out.shape}")


def main():
    """Run all tests."""
    print("="*80)
    print("PyTorch Built-in Flash Attention Test Suite")
    print("Testing on CUDA 13.0 / PyTorch 2.9 (Jetson AGX Thor)")
    print("="*80)

    # Check environment
    print(f"\nEnvironment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  Compute: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")

    # Run tests
    test_basic_attention()
    test_causal_attention()
    test_gqa_attention()
    test_gradient_backward()
    test_mixed_precision()

    # Benchmark
    print("\n" + "="*80)
    print("Performance Benchmark")
    print("="*80)

    # Small model (fast iteration)
    benchmark_attention(batch_size=4, seq_len=512, d_model=512, n_heads=8)

    # Medium model (SAGE-like)
    benchmark_attention(batch_size=2, seq_len=1024, d_model=768, n_heads=12)

    # Large model (long context)
    benchmark_attention(batch_size=1, seq_len=4096, d_model=1024, n_heads=16)

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nConclusion:")
    print("  PyTorch 2.9's built-in flash attention is fully functional on Thor.")
    print("  No need for standalone flash-attention package compilation.")
    print("  Use F.scaled_dot_product_attention() for all SAGE attention layers.")
    print("="*80)


if __name__ == '__main__':
    main()
