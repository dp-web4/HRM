#!/usr/bin/env python3
"""
Selective Transformer Layer for SAGE + Qwen3-Omni

Implements full transformer layer with:
- RMSNorm
- Multi-head attention (GQA)
- RoPE positional encoding
- Selective MoE (SAGE's contribution!)
- Residual connections

This demonstrates SAGE's selective resource loading integrated
into a complete transformer architecture.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .selective_expert_loader import SelectiveExpertLoader
except ImportError:
    from selective_expert_loader import SelectiveExpertLoader


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden]
        Returns:
            normalized: [batch, seq, hidden]
        """
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int, max_position_embeddings: int = 65536, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (not actually used, just for device/dtype)
            seq_len: Sequence length
        Returns:
            (cos, sin) embeddings for RoPE
        """
        # Generate position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)

        # Different from paper, but mimics Qwen implementation
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA)

    GQA: Fewer KV heads than Q heads for efficiency
    Qwen3-Omni: 32 Q heads, 4 KV heads
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 65536,
        rope_theta: float = 1000000.0,
        extraction_dir: str = None,
        layer_id: int = None,
        component: str = "thinker",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        # GQA: num_q_heads > num_kv_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Q/K normalization (Q3-Omni uses this!)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        # Load REAL attention weights if extraction_dir provided
        if extraction_dir is not None and layer_id is not None:
            import safetensors
            import os
            attn_file = os.path.join(
                extraction_dir,
                "attention",
                f"{component}_attention_layer_{layer_id:02d}.safetensors"
            )
            if os.path.exists(attn_file):
                with safetensors.safe_open(attn_file, framework="pt") as f:
                    prefix = f"{component}.model.layers.{layer_id}.self_attn"

                    # Load Q, K, V, O projections (convert to float32 for CPU compatibility)
                    if f"{prefix}.q_proj.weight" in f.keys():
                        self.q_proj.weight = nn.Parameter(f.get_tensor(f"{prefix}.q_proj.weight").float())
                    if f"{prefix}.k_proj.weight" in f.keys():
                        self.k_proj.weight = nn.Parameter(f.get_tensor(f"{prefix}.k_proj.weight").float())
                    if f"{prefix}.v_proj.weight" in f.keys():
                        self.v_proj.weight = nn.Parameter(f.get_tensor(f"{prefix}.v_proj.weight").float())
                    if f"{prefix}.o_proj.weight" in f.keys():
                        self.o_proj.weight = nn.Parameter(f.get_tensor(f"{prefix}.o_proj.weight").float())

                    # Load Q/K norms (critical for Q3-Omni!)
                    if f"{prefix}.q_norm.weight" in f.keys():
                        self.q_norm.weight = nn.Parameter(f.get_tensor(f"{prefix}.q_norm.weight").float())
                    if f"{prefix}.k_norm.weight" in f.keys():
                        self.k_norm.weight = nn.Parameter(f.get_tensor(f"{prefix}.k_norm.weight").float())

                    print(f"✅ Loaded real attention weights for layer {layer_id}")
            else:
                print(f"⚠️  Attention file not found: {attn_file}, using random weights")

        # RoPE
        self.rotary_emb = RotaryEmbedding(head_dim, max_position_embeddings, rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: Optional [batch, 1, seq, seq] causal mask
        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply Q/K normalization (Q3-Omni critical feature!)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.num_attention_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match Q heads (for GQA)"""
        if n_rep == 1:
            return hidden_states

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SelectiveMoELayer(nn.Module):
    """
    Selective Mixture of Experts with SAGE integration

    Key difference from standard MoE:
    - Loads experts on-demand based on router + SNARC salience
    - Trust-based eviction policy
    - Metabolic state determines expert budget
    """

    def __init__(
        self,
        hidden_size: int,
        expert_loader: SelectiveExpertLoader,
        layer_id: int,
        num_experts_per_tok: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_loader = expert_loader
        self.layer_id = layer_id
        self.num_experts_per_tok = num_experts_per_tok

    def forward(
        self,
        hidden_states: torch.Tensor,
        snarc_salience: Optional[Dict[str, float]] = None,
        metabolic_state: str = "FOCUS",
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden]
            snarc_salience: Optional SNARC scores for expert selection
            metabolic_state: WAKE/FOCUS/CRISIS (determines expert count)
        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_length, hidden_dim = hidden_states.shape

        # Select experts (SNARC-augmented or standard)
        selected_expert_ids, router_weights = self.expert_loader.select_experts_snarc(
            hidden_states,
            self.layer_id,
            num_experts=self.num_experts_per_tok,
            snarc_salience=snarc_salience,
            metabolic_state=metabolic_state
        )

        # Process each expert
        expert_outputs = []
        valid_expert_indices = []

        for i, expert_id in enumerate(selected_expert_ids):
            # Load expert (from memory or disk)
            expert_weights = self.expert_loader.load_expert(expert_id, self.layer_id)

            # Skip if expert doesn't exist (sparse layers)
            if expert_weights is None:
                continue

            # Expert forward pass
            output = self._expert_forward(hidden_states, expert_weights)
            expert_outputs.append(output)
            valid_expert_indices.append(i)

        # If no valid experts, return input unchanged (residual path)
        if len(expert_outputs) == 0:
            print(f"⚠️  No valid experts in layer {self.layer_id}, using identity")
            return hidden_states

        # Weighted combination (only use weights for valid experts)
        valid_router_weights = router_weights[valid_expert_indices]
        router_probs = F.softmax(valid_router_weights, dim=0)

        combined_output = torch.zeros_like(hidden_states)
        for i, output in enumerate(expert_outputs):
            combined_output += router_probs[i] * output

        return combined_output

    def _expert_forward(self, x: torch.Tensor, expert_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Standard expert MLP: h = down(silu(gate(x)) * up(x))"""
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

        # Validate all weights are present
        if gate_proj is None or up_proj is None or down_proj is None:
            print(f"⚠️  Expert missing weights! gate:{gate_proj is not None}, up:{up_proj is not None}, down:{down_proj is not None}")
            print(f"   Available keys: {list(expert_weights.keys())}")
            # Return input unchanged if weights are missing
            return x

        gate_output = F.linear(x, gate_proj)
        up_output = F.linear(x, up_proj)
        intermediate = F.silu(gate_output) * up_output
        output = F.linear(intermediate, down_proj)

        return output


class SelectiveTransformerLayer(nn.Module):
    """
    Complete transformer layer with selective expert loading

    Architecture:
    x = x + attention(norm(x))
    x = x + moe(norm(x))

    Where MoE uses SAGE's selective expert loading
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        expert_loader: Optional[SelectiveExpertLoader] = None,
        layer_id: int = 0,
        num_experts_per_tok: int = 8,
        rms_norm_eps: float = 1e-6,
        extraction_dir: str = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        # Pre-normalization (load real weights if available!)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Load REAL norm weights if extraction_dir provided
        if extraction_dir is not None and layer_id is not None:
            import safetensors
            import os
            norms_file = os.path.join(
                extraction_dir,
                "norms",
                f"thinker_norms_layer_{layer_id:02d}.safetensors"
            )
            if os.path.exists(norms_file):
                with safetensors.safe_open(norms_file, framework="pt") as f:
                    prefix = f"thinker.model.layers.{layer_id}"

                    # Load input layernorm
                    if f"{prefix}.input_layernorm.weight" in f.keys():
                        self.input_layernorm.weight = nn.Parameter(
                            f.get_tensor(f"{prefix}.input_layernorm.weight")
                        )

                    # Load post-attention layernorm
                    if f"{prefix}.post_attention_layernorm.weight" in f.keys():
                        self.post_attention_layernorm.weight = nn.Parameter(
                            f.get_tensor(f"{prefix}.post_attention_layernorm.weight")
                        )

                    print(f"✅ Loaded real layer norms for layer {layer_id}")
            else:
                print(f"⚠️  Norms file not found for layer {layer_id}, using default initialization")

        # Self-attention (with REAL weights if extraction_dir provided!)
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            extraction_dir=extraction_dir,
            layer_id=layer_id,
            component="thinker",
        )

        # Selective MoE
        if expert_loader is None:
            raise ValueError("expert_loader required for SelectiveTransformerLayer")

        self.moe = SelectiveMoELayer(
            hidden_size=hidden_size,
            expert_loader=expert_loader,
            layer_id=layer_id,
            num_experts_per_tok=num_experts_per_tok,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        snarc_salience: Optional[Dict[str, float]] = None,
        metabolic_state: str = "FOCUS",
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: Optional causal mask
            snarc_salience: Optional SNARC scores
            metabolic_state: Resource budget
        Returns:
            output: [batch, seq, hidden]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MoE with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(
            hidden_states,
            snarc_salience=snarc_salience,
            metabolic_state=metabolic_state
        )
        hidden_states = residual + hidden_states

        return hidden_states


def create_causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask (lower triangular)"""
    mask = torch.full((seq_length, seq_length), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
