"""
Modified layers for Jetson - uses standard attention instead of flash attention
"""
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalization"""
    return x * torch.rsqrt((x**2).mean(-1, keepdim=True) + eps) * weight


class Attention(nn.Module):
    """Multi-head attention with Jetson-compatible implementation"""
    
    def __init__(
        self,
        dim: int,
        head_dim: int,
        use_lrpe: bool = True,
        lrpe_hf_compat: bool = True,
        lrpe_base: int = 512,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.use_lrpe = use_lrpe
        self.lrpe_hf_compat = lrpe_hf_compat
        self.lrpe_base = lrpe_base
        
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        if use_lrpe:
            self.rotary_emb = RotaryEmbedding(
                head_dim, 
                hf_compat=lrpe_hf_compat, 
                base=lrpe_base
            )
    
    def forward(
        self,
        x: torch.Tensor,
        cos_sin: CosSin | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CosSin]:
        B, L, D = x.shape
        
        # QKV projection
        qkv = self.wqkv(x)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Apply rotary embeddings if enabled
        if self.use_lrpe:
            if cos_sin is None:
                cos_sin = self.rotary_emb(q)
            q, k = apply_rotary_pos_emb(q, k, cos_sin)
        
        # Handle cache for generation
        if cache is not None:
            # Append current k,v to cache
            k = torch.cat([cache[:, :, 0], k], dim=1)
            v = torch.cat([cache[:, :, 1], v], dim=1)
            # Update cache
            cache = torch.stack([k[:, 1:], v[:, 1:]], dim=2)
        
        # Standard scaled dot-product attention (no flash attention)
        # Reshape for attention: (B, n_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask if needed
        if cu_seqlens is None:  # Standard causal mask
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            scores = scores + mask
        
        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        
        # Output projection
        out = self.wo(out)
        
        return out, cos_sin


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings"""
    
    def __init__(self, dim: int, base: int = 10000, hf_compat: bool = True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.hf_compat = hf_compat
        self.register_buffer('inv_freq', self._compute_inv_freq())
    
    def _compute_inv_freq(self):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
    
    def forward(self, x: torch.Tensor) -> CosSin:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos_sin: CosSin) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys"""
    cos, sin = cos_sin
    
    # Reshape cos and sin to match q, k dimensions
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class CastedEmbedding(nn.Embedding):
    """Embedding layer with optional dtype casting"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.embedding_dim**-0.5)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


class CastedLinear(nn.Linear):
    """Linear layer with optional dtype casting"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = self.in_features**-0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)